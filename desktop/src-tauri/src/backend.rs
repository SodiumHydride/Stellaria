//! Supervises the Python backend (`python -m stellaria_backend`): start, readiness
//! (`STELLARIA_READY {json}` + `GET /api/health`), log capture, crash restarts and a
//! graceful shutdown (`POST /api/shutdown`) that lets training save a checkpoint.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, watch};

use crate::error::{ShellError, ShellResult};
use crate::fsutil::{self, RotatingFile};
use crate::paths::AppPaths;
use crate::procutil::{hidden_command, spawn_line_reader, ProcessTree, Stream};
use crate::pycheck::apply_custom_env;
use crate::runtime::{apply_python_env, Interpreter};
use crate::Shell;

pub const STATUS_EVENT: &str = "backend://status";
pub const LOG_EVENT: &str = "backend://log";
/// The shell's capture of the backend's stdout/stderr. (`backend.log` itself is written by
/// the backend's own logger, see docs/api.md §1.)
pub const CONSOLE_LOG: &str = "backend-console.log";

const READY_TIMEOUT: Duration = Duration::from_secs(90);
const SHUTDOWN_GRACE: Duration = Duration::from_secs(25);
const MAX_RESTARTS: u32 = 3;
const RESTART_BACKOFF: [u64; 3] = [2, 5, 10];
/// A run this long counts as healthy: the crash counter starts over.
const STABLE_RUN: Duration = Duration::from_secs(300);
const RING_LINES: usize = 5000;
const STDERR_TAIL: usize = 30;
const LOG_EVENTS_PER_SECOND: u32 = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendState {
    Stopped,
    Starting,
    Ready,
    Crashed,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BackendStatus {
    pub status: BackendState,
    /// Origin of the backend, e.g. `http://127.0.0.1:51234` (the API lives under `/api`).
    pub url: Option<String>,
    pub token: Option<String>,
    pub error: Option<String>,
    pub restarts: u32,
}

impl Default for BackendStatus {
    fn default() -> Self {
        Self {
            status: BackendState::Stopped,
            url: None,
            token: None,
            error: None,
            restarts: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ReadyLine {
    port: u16,
    token: String,
    #[serde(default)]
    version: Option<String>,
}

#[derive(Serialize, Clone)]
struct LogEvent<'a> {
    stream: &'a str,
    line: &'a str,
}

struct Running {
    /// The interpreter it runs with.
    python: PathBuf,
    tree: Arc<ProcessTree>,
    exited: ExitWatch,
    port: Option<u16>,
    token: String,
    /// Set when we stop it on purpose: its exit is not a crash.
    stopping: bool,
}

struct Inner {
    status: BackendStatus,
    generation: u64,
    process: Option<Running>,
    ready_since: Option<Instant>,
}

pub struct BackendManager {
    inner: Mutex<Inner>,
    /// Serializes start / stop / restart.
    ops: tokio::sync::Mutex<()>,
    logs: Arc<LogHub>,
    http: reqwest::Client,
}

impl BackendManager {
    pub fn new(paths: &AppPaths, http: reqwest::Client) -> Self {
        Self {
            inner: Mutex::new(Inner {
                status: BackendStatus::default(),
                generation: 0,
                process: None,
                ready_since: None,
            }),
            ops: tokio::sync::Mutex::new(()),
            logs: Arc::new(LogHub::new(paths.logs_dir.join(CONSOLE_LOG))),
            http,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn status(&self) -> BackendStatus {
        self.lock().status.clone()
    }

    pub fn is_running(&self) -> bool {
        self.lock().process.is_some()
    }

    /// The interpreter of the running backend process, if any.
    pub fn current_python(&self) -> Option<PathBuf> {
        self.lock().process.as_ref().map(|r| r.python.clone())
    }

    /// `(origin, token)` while the backend is ready.
    pub fn endpoint(&self) -> Option<(String, String)> {
        let inner = self.lock();
        match (&inner.status.status, &inner.status.url, &inner.status.token) {
            (BackendState::Ready, Some(url), Some(token)) => Some((url.clone(), token.clone())),
            _ => None,
        }
    }

    pub fn logs(&self, tail: usize) -> Vec<String> {
        self.logs.tail(tail)
    }

    fn update_status(
        &self,
        app: &AppHandle,
        change: impl FnOnce(&mut BackendStatus),
    ) -> BackendStatus {
        let status = {
            let mut inner = self.lock();
            change(&mut inner.status);
            inner.status.clone()
        };
        emit_status(app, &status);
        status
    }

    // ------------------------------------------------------------------ start

    pub async fn start(&self, app: &AppHandle, reset_restarts: bool) -> ShellResult<BackendStatus> {
        let _op = self.ops.lock().await;
        self.start_locked(app, reset_restarts).await
    }

    pub async fn restart(&self, app: &AppHandle) -> ShellResult<BackendStatus> {
        let _op = self.ops.lock().await;
        self.stop_locked(app, true).await;
        self.start_locked(app, true).await
    }

    async fn start_locked(
        &self,
        app: &AppHandle,
        reset_restarts: bool,
    ) -> ShellResult<BackendStatus> {
        {
            let mut inner = self.lock();
            if inner.process.is_some()
                && matches!(
                    inner.status.status,
                    BackendState::Starting | BackendState::Ready
                )
            {
                return Ok(inner.status.clone());
            }
            if reset_restarts {
                inner.status.restarts = 0;
            }
        }
        let shell = crate::shell(app);
        let token = random_token()?;
        let (mut cmd, python) = match launch_command(&shell, &token) {
            Ok(launch) => launch,
            Err(err) => {
                self.update_status(app, |s| {
                    s.status = BackendState::Stopped;
                    s.error = Some(err.message.clone());
                });
                return Err(err);
            }
        };
        self.update_status(app, |s| {
            s.status = BackendState::Starting;
            s.url = None;
            s.token = None;
            s.error = None;
        });
        self.logs
            .push(app, Stream::Stderr, "[stellaria] starting the backend…");

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(err) => {
                let message = format!("Cannot start the backend: {err}");
                self.update_status(app, |s| {
                    s.status = BackendState::Crashed;
                    s.error = Some(message.clone());
                });
                return Err(ShellError::new("backend.spawn_failed", message));
            }
        };
        let tree = Arc::new(ProcessTree::attach(&child));
        let (exit_tx, exit_rx) = watch::channel(None);
        let generation = {
            let mut inner = self.lock();
            inner.generation += 1;
            inner.ready_since = None;
            inner.process = Some(Running {
                python,
                tree: tree.clone(),
                exited: exit_rx.clone(),
                port: None,
                token: token.clone(),
                stopping: false,
            });
            inner.generation
        };

        // Output: tee to the log hub, watch stdout for the READY line.
        let (line_tx, mut line_rx) = mpsc::unbounded_channel();
        if let Some(stdout) = child.stdout.take() {
            spawn_line_reader(stdout, Stream::Stdout, line_tx.clone());
        }
        if let Some(stderr) = child.stderr.take() {
            spawn_line_reader(stderr, Stream::Stderr, line_tx.clone());
        }
        drop(line_tx);
        let (ready_tx, ready_rx) = oneshot::channel::<ReadyLine>();
        let logs = self.logs.clone();
        let log_app = app.clone();
        tokio::spawn(async move {
            let mut ready_tx = Some(ready_tx);
            while let Some((stream, line)) = line_rx.recv().await {
                if stream == Stream::Stdout {
                    if let Some(json) = line.strip_prefix("STELLARIA_READY ") {
                        if let Ok(ready) = serde_json::from_str::<ReadyLine>(json) {
                            if let Some(tx) = ready_tx.take() {
                                let _ = tx.send(ready);
                            }
                            // The token is a secret: keep it out of logs and events.
                            logs.push(&log_app, stream, "STELLARIA_READY {…}");
                            continue;
                        }
                    }
                }
                logs.push(&log_app, stream, &line);
            }
        });

        // Exit monitor.
        let monitor_app = app.clone();
        let monitor_tree = tree.clone();
        tokio::spawn(async move {
            let status = child.wait().await;
            let code = describe_exit(status.as_ref().ok());
            // Nothing the backend started may outlive it (orphaned training etc.).
            monitor_tree.kill();
            let _ = exit_tx.send(Some(code.clone()));
            crate::shell(&monitor_app)
                .backend
                .on_exit(&monitor_app, generation, code);
        });

        // Wait for READY, an early exit or the timeout.
        let mut exited = exit_rx;
        let mut exited_after_eof = exited.clone();
        let outcome = tokio::select! {
            ready = ready_rx => match ready {
                Ok(line) => StartOutcome::Ready(line),
                Err(_) => {
                    // stdout closed without READY: the process is going away.
                    let _ = tokio::time::timeout(Duration::from_secs(5), wait_exited(&mut exited_after_eof)).await;
                    StartOutcome::Exited
                }
            },
            () = wait_exited(&mut exited) => StartOutcome::Exited,
            _ = tokio::time::sleep(READY_TIMEOUT) => StartOutcome::TimedOut,
        };

        let failure = match outcome {
            StartOutcome::Ready(line) => {
                let url = format!("http://127.0.0.1:{}", line.port);
                if self.health_check(&url).await {
                    let status = {
                        let mut inner = self.lock();
                        if inner.generation == generation {
                            if let Some(running) = inner.process.as_mut() {
                                running.port = Some(line.port);
                                running.token = line.token.clone();
                            }
                            inner.ready_since = Some(Instant::now());
                            inner.status.status = BackendState::Ready;
                            inner.status.url = Some(url);
                            inner.status.token = Some(line.token.clone());
                            inner.status.error = None;
                        }
                        inner.status.clone()
                    };
                    emit_status(app, &status);
                    log::info!(
                        "backend ready on port {} (version {})",
                        line.port,
                        line.version.as_deref().unwrap_or("?")
                    );
                    return Ok(status);
                }
                format!("The backend did not answer its health check on {url}.")
            }
            StartOutcome::Exited => {
                let how = exited
                    .borrow()
                    .clone()
                    .unwrap_or_else(|| "still running".into());
                format!("The backend stopped during startup ({how}).")
            }
            StartOutcome::TimedOut => format!(
                "The backend did not become ready within {} s.",
                READY_TIMEOUT.as_secs()
            ),
        };

        tree.kill();
        let message = format!("{failure}\n{}", self.logs.stderr_tail());
        let status = {
            let mut inner = self.lock();
            if inner.generation == generation {
                inner.process = None;
                inner.status.status = BackendState::Crashed;
                inner.status.url = None;
                inner.status.token = None;
                inner.status.error = Some(message.clone());
            }
            inner.status.clone()
        };
        emit_status(app, &status);
        Err(ShellError::new("backend.start_failed", failure).with_detail(message))
    }

    async fn health_check(&self, url: &str) -> bool {
        for _ in 0..20 {
            let ok = self
                .http
                .get(format!("{url}/api/health"))
                .timeout(Duration::from_secs(3))
                .send()
                .await
                .is_ok_and(|r| r.status().is_success());
            if ok {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        false
    }

    /// Called by the exit monitor of process `generation`.
    fn on_exit(&self, app: &AppHandle, generation: u64, code: String) {
        let restart = {
            let mut inner = self.lock();
            if inner.generation != generation {
                return;
            }
            let stopping = inner.process.take().is_some_and(|r| r.stopping);
            if stopping {
                inner.status = BackendStatus {
                    restarts: inner.status.restarts,
                    ..BackendStatus::default()
                };
                emit_status(app, &inner.status);
                return;
            }
            if inner.status.status != BackendState::Ready {
                // A failed start is reported by `start_locked`.
                return;
            }
            if inner.ready_since.is_some_and(|t| t.elapsed() >= STABLE_RUN) {
                inner.status.restarts = 0;
            }
            let quitting = crate::shell(app).is_quitting();
            let will_restart = inner.status.restarts < MAX_RESTARTS && !quitting;
            inner.status.status = BackendState::Crashed;
            inner.status.url = None;
            inner.status.token = None;
            inner.status.error = Some(format!(
                "The backend stopped unexpectedly ({code}).\n{}",
                self.logs.stderr_tail()
            ));
            if will_restart {
                inner.status.restarts += 1;
            }
            emit_status(app, &inner.status);
            will_restart
        };
        log::warn!("backend exited unexpectedly; restart: {restart}");
        if restart {
            let app = app.clone();
            tauri::async_runtime::spawn(async move { restart_loop(app).await });
        }
    }

    // ------------------------------------------------------------------ stop

    pub async fn stop(&self, app: &AppHandle, graceful: bool) -> BackendStatus {
        let _op = self.ops.lock().await;
        self.stop_locked(app, graceful).await
    }

    /// Graceful stop without waiting for other operations (used when quitting).
    pub async fn shutdown_for_quit(&self, app: &AppHandle, graceful: bool) {
        self.stop_locked(app, graceful).await;
    }

    async fn stop_locked(&self, app: &AppHandle, graceful: bool) -> BackendStatus {
        let target = {
            let mut inner = self.lock();
            inner.process.as_mut().map(|r| {
                r.stopping = true;
                (r.tree.clone(), r.exited.clone(), r.port, r.token.clone())
            })
        };
        if let Some((tree, mut exited, port, token)) = target {
            let wait = |limit: Duration| {
                let mut rx = exited.clone();
                async move {
                    tokio::time::timeout(limit, wait_exited(&mut rx))
                        .await
                        .is_ok()
                }
            };
            let mut done = false;
            if graceful {
                if let Some(port) = port {
                    self.logs.push(
                        app,
                        Stream::Stderr,
                        "[stellaria] asking the backend to shut down…",
                    );
                    let accepted = self
                        .http
                        .post(format!("http://127.0.0.1:{port}/api/shutdown"))
                        .bearer_auth(&token)
                        .timeout(Duration::from_secs(5))
                        .send()
                        .await
                        .is_ok_and(|r| r.status().is_success());
                    done = wait(if accepted {
                        SHUTDOWN_GRACE
                    } else {
                        Duration::from_secs(3)
                    })
                    .await;
                }
            }
            if !done {
                tree.terminate();
                done = wait(Duration::from_secs(3)).await;
            }
            if !done {
                tree.kill();
                let _ =
                    tokio::time::timeout(Duration::from_secs(5), wait_exited(&mut exited)).await;
            }
        }
        let status = {
            let mut inner = self.lock();
            if inner.process.as_ref().is_some_and(|r| r.stopping) {
                inner.process = None;
            }
            let changed = inner.status.status != BackendState::Stopped;
            inner.status = BackendStatus {
                restarts: inner.status.restarts,
                ..BackendStatus::default()
            };
            (inner.status.clone(), changed)
        };
        let (status, changed) = status;
        // The exit monitor usually reported "stopped" already.
        if changed {
            emit_status(app, &status);
        }
        status
    }

    /// Last resort on exit: kill the tree synchronously.
    pub fn kill_now(&self) {
        let mut inner = self.lock();
        if let Some(running) = inner.process.as_mut() {
            running.stopping = true;
            running.tree.kill();
        }
    }

    // ------------------------------------------------------------------ API helpers

    /// Whether the backend reports running/queued jobs (`GET /api/jobs?active=1`).
    pub async fn active_jobs(&self) -> bool {
        self.jobs_active()
            .await
            .is_some_and(|jobs| !jobs.is_empty())
    }

    async fn jobs_active(&self) -> Option<Vec<serde_json::Value>> {
        let (url, token) = self.endpoint()?;
        let response = self
            .http
            .get(format!("{url}/api/jobs?active=1"))
            .bearer_auth(token)
            .timeout(Duration::from_millis(1500))
            .send()
            .await
            .ok()?;
        response.json::<Vec<serde_json::Value>>().await.ok()
    }

    /// Pauses every running training (used by the tray when no frontend is listening).
    /// Asks the backend to free what the singing models hold (GPU memory).
    pub async fn release_render_memory(&self) {
        let Some((url, token)) = self.endpoint() else {
            return;
        };
        let result = self
            .http
            .post(format!("{url}/api/system/render/release"))
            .bearer_auth(&token)
            .timeout(Duration::from_secs(10))
            .send()
            .await;
        if let Err(err) = result {
            log::warn!("release request failed: {err}");
        }
    }

    pub async fn pause_training(&self) {
        let Some((url, token)) = self.endpoint() else {
            return;
        };
        let Some(jobs) = self.jobs_active().await else {
            return;
        };
        for job in jobs {
            let is_train = job.get("type").and_then(|t| t.as_str()) == Some("train");
            let model = job
                .get("target")
                .and_then(|t| t.get("modelId"))
                .and_then(|m| m.as_str());
            if let (true, Some(model)) = (is_train, model) {
                let result = self
                    .http
                    .post(format!("{url}/api/models/{model}/pause"))
                    .bearer_auth(&token)
                    .timeout(Duration::from_secs(10))
                    .send()
                    .await;
                if let Err(err) = result {
                    log::warn!("pause request failed: {err}");
                }
            }
        }
    }
}

/// `None` while running, then how the process ended ("exit code 1", "signal 9").
type ExitWatch = watch::Receiver<Option<String>>;

fn describe_exit(status: Option<&std::process::ExitStatus>) -> String {
    let Some(status) = status else {
        return "unknown exit".into();
    };
    if let Some(code) = status.code() {
        return format!("exit code {code}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("signal {signal}");
        }
    }
    "unknown exit".into()
}

/// Resolves once the process has exited (the watch guard is not held across awaits).
async fn wait_exited(rx: &mut ExitWatch) {
    let _ = rx.wait_for(Option::is_some).await;
}

enum StartOutcome {
    Ready(ReadyLine),
    Exited,
    TimedOut,
}

/// Restarts a crashed backend with backoff, up to [`MAX_RESTARTS`] attempts in total.
async fn restart_loop(app: AppHandle) {
    let shell = crate::shell(&app);
    loop {
        let (restarts, generation) = {
            let inner = shell.backend.lock();
            (inner.status.restarts, inner.generation)
        };
        let delay = RESTART_BACKOFF[(restarts.max(1) as usize - 1).min(RESTART_BACKOFF.len() - 1)];
        tokio::time::sleep(Duration::from_secs(delay)).await;
        {
            let inner = shell.backend.lock();
            let still_crashed =
                inner.status.status == BackendState::Crashed && inner.generation == generation;
            if !still_crashed || shell.is_quitting() {
                return;
            }
        }
        match shell.backend.start(&app, false).await {
            Ok(status) if status.status == BackendState::Ready => return,
            _ => {
                let mut inner = shell.backend.lock();
                if inner.status.status != BackendState::Crashed
                    || inner.status.restarts >= MAX_RESTARTS
                {
                    return;
                }
                inner.status.restarts += 1;
                emit_status(&app, &inner.status);
            }
        }
    }
}

fn emit_status(app: &AppHandle, status: &BackendStatus) {
    if let Err(err) = app.emit(STATUS_EVENT, status) {
        log::warn!("cannot emit backend status: {err}");
    }
}

fn random_token() -> ShellResult<String> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).map_err(|e| ShellError::internal(format!("random token: {e}")))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// `python -m stellaria_backend …` with the environment from docs/api.md §1.
fn launch_command(shell: &Shell, token: &str) -> ShellResult<(Command, PathBuf)> {
    let paths = &shell.paths;
    let resources = &shell.resources;
    let interpreter = shell.backend_interpreter()?;
    let python = interpreter.path().to_path_buf();
    let library = shell.library_path();
    std::fs::create_dir_all(&library).map_err(|e| {
        ShellError::io(
            format!("Cannot create the library at {}", library.display()),
            &e,
        )
    })?;
    std::fs::create_dir_all(&paths.logs_dir)?;

    let mut cmd = hidden_command(&python);
    cmd.args(["-m", "stellaria_backend", "--library"])
        .arg(&library)
        .arg("--data")
        .arg(&paths.data_dir)
        .arg("--engine")
        .arg(&resources.engine)
        .arg("--resources")
        .arg(&resources.manifest);
    if resources.uv.is_file() {
        cmd.arg("--uv").arg(&resources.uv);
    }
    cmd.args(["--port", "0", "--token", token]);
    if resources.dev {
        // Debug builds load the frontend from the Vite dev server (http://127.0.0.1:1420),
        // which the backend only accepts with `--dev` (CORS *). With an explicit port and
        // token, `--dev` changes nothing else (docs/api.md §1).
        cmd.arg("--dev");
    }
    cmd.current_dir(&paths.data_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    match &interpreter {
        Interpreter::Managed(_) => {
            apply_python_env(&mut cmd, paths);
            cmd.env("VIRTUAL_ENV", &paths.venv_dir);
        }
        Interpreter::Dev(_) => apply_python_env(&mut cmd, paths),
        Interpreter::Custom { base, .. } => {
            apply_custom_env(&mut cmd, base, Some(paths));
            // the backend never installs into the user's own environment (only into a layer)
            cmd.env("STELLARIA_CUSTOM_RUNTIME", "1");
        }
    }
    cmd.env("PYTHONPATH", &resources.backend)
        .env("STELLARIA_SHELL_PID", std::process::id().to_string())
        // The backend's own uv calls (add-ons) share the runtime layout.
        .env("UV_CACHE_DIR", &paths.uv_cache_dir)
        .env("UV_PYTHON_INSTALL_DIR", &paths.python_dir)
        .env("UV_PYTHON_PREFERENCE", "only-managed")
        .env("UV_NO_CONFIG", "1");
    if cfg!(windows) {
        cmd.env("UV_LINK_MODE", "copy");
    }
    Ok((cmd, python))
}

/// Ring buffer + `backend-console.log` + throttled `backend://log` events.
struct LogHub {
    lines: Mutex<VecDeque<String>>,
    stderr: Mutex<VecDeque<String>>,
    file: Mutex<RotatingFile>,
    throttle: Mutex<Throttle>,
}

struct Throttle {
    window: Instant,
    sent: u32,
    dropped: u32,
}

impl LogHub {
    fn new(path: std::path::PathBuf) -> Self {
        Self {
            lines: Mutex::new(VecDeque::with_capacity(RING_LINES)),
            stderr: Mutex::new(VecDeque::with_capacity(STDERR_TAIL)),
            file: Mutex::new(RotatingFile::new(path, 4 * 1024 * 1024, 2)),
            throttle: Mutex::new(Throttle {
                window: Instant::now(),
                sent: 0,
                dropped: 0,
            }),
        }
    }

    fn push(&self, app: &AppHandle, stream: Stream, line: &str) {
        let stamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ");
        lock(&self.file).write_line(&format!("{stamp} [{}] {line}", stream.as_str()));
        push_bounded(&mut lock(&self.lines), line.to_string(), RING_LINES);
        if stream == Stream::Stderr {
            push_bounded(&mut lock(&self.stderr), line.to_string(), STDERR_TAIL);
        }

        let mut throttle = lock(&self.throttle);
        if throttle.window.elapsed() >= Duration::from_secs(1) {
            if throttle.dropped > 0 {
                let note = format!(
                    "[stellaria] {} log lines not shown here (see {CONSOLE_LOG})",
                    throttle.dropped
                );
                let _ = app.emit(
                    LOG_EVENT,
                    LogEvent {
                        stream: "stderr",
                        line: &note,
                    },
                );
            }
            throttle.window = Instant::now();
            throttle.sent = 0;
            throttle.dropped = 0;
        }
        if throttle.sent < LOG_EVENTS_PER_SECOND {
            throttle.sent += 1;
            let _ = app.emit(
                LOG_EVENT,
                LogEvent {
                    stream: stream.as_str(),
                    line,
                },
            );
        } else {
            throttle.dropped += 1;
        }
    }

    fn tail(&self, n: usize) -> Vec<String> {
        let lines = lock(&self.lines);
        if lines.is_empty() {
            drop(lines);
            let path = lock(&self.file).path().to_path_buf();
            return fsutil::tail_lines(&path, n);
        }
        let skip = lines.len().saturating_sub(n);
        lines.iter().skip(skip).cloned().collect()
    }

    fn stderr_tail(&self) -> String {
        lock(&self.stderr)
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn push_bounded(queue: &mut VecDeque<String>, line: String, max: usize) {
    if queue.len() >= max {
        queue.pop_front();
    }
    queue.push_back(line);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ready_line() {
        let line: ReadyLine =
            serde_json::from_str(r#"{"port": 51234, "token": "abc", "version": "0.1.0"}"#).unwrap();
        assert_eq!(line.port, 51234);
        assert_eq!(line.token, "abc");
    }

    #[test]
    fn token_is_32_hex() {
        let token = random_token().unwrap();
        assert_eq!(token.len(), 32);
        assert!(token.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn status_shape() {
        let v = serde_json::to_value(BackendStatus::default()).unwrap();
        assert_eq!(v["status"], "stopped");
        assert!(v["url"].is_null());
        assert_eq!(v["restarts"], 0);
    }
}

//! First-run bootstrap of the Python runtime with the bundled `uv`:
//! prepare → python → venv → torch → packages → verify → `runtime/state.json`.
//!
//! Every stage that downloads walks an ordered list of mirrors (see [`sources`]) and falls
//! over to the next one on failure. Progress is streamed as `bootstrap://progress`.
//!
//! The same machinery (one job at a time, cancel, progress) installs the app's packages into
//! a user's own interpreter ([`requirements`]).

mod pipeline;
mod progress;
mod requirements;
mod sources;
mod tool;
mod uvlog;
mod wheel;

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use tauri::AppHandle;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

pub(crate) use pipeline::uv_command;
use pipeline::Pipeline;
use progress::Reporter;

use crate::error::{ShellError, ShellResult};
use crate::manifest::Manifest;
use crate::types::{Region, RuntimeMode, TorchVariant};

pub(crate) enum StageError {
    Cancelled,
    Failed(ShellError),
}

impl From<ShellError> for StageError {
    fn from(err: ShellError) -> Self {
        Self::Failed(err)
    }
}

pub(crate) type StageResult<T> = Result<T, StageError>;

/// How a job ended; turned into the final `bootstrap://progress` event by the manager.
pub(crate) enum Outcome {
    Done(String),
    Failed(ShellError),
    Cancelled,
}

impl From<StageError> for Outcome {
    fn from(err: StageError) -> Self {
        match err {
            StageError::Cancelled => Self::Cancelled,
            StageError::Failed(err) => Self::Failed(err),
        }
    }
}

impl Outcome {
    fn report(self, rep: &mut Reporter) {
        match self {
            Self::Done(message) => rep.finish(message),
            Self::Failed(err) => rep.fail(err),
            Self::Cancelled => rep.cancelled(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// `bootstrap_start`: install (or switch to) a variant.
    Install,
    /// `bootstrap_repair`: same pipeline, plus reinstalling torch if it no longer imports.
    Repair,
}

#[derive(Debug, Clone, Copy)]
pub struct Request {
    pub variant: TorchVariant,
    pub region: Region,
    pub mode: Mode,
}

/// What the single bootstrap slot runs.
#[derive(Debug, Clone)]
pub enum Job {
    /// The managed runtime (`bootstrap_start` / `bootstrap_repair`).
    Runtime(Request),
    /// `install_runtime_requirements`: the app's packages (not torch) for the user's Python —
    /// into Stellaria's layer over it (their environment is never written to).
    Requirements { python: PathBuf, region: Region },
}

struct Running {
    cancel: CancellationToken,
    done: watch::Receiver<bool>,
}

#[derive(Default)]
pub struct BootstrapManager {
    running: Mutex<Option<Running>>,
}

impl BootstrapManager {
    pub fn is_running(&self) -> bool {
        self.lock().is_some()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Running>> {
        self.running.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Starts the pipeline in the background; progress arrives as events.
    pub fn start(&self, app: &AppHandle, job: Job) -> ShellResult<()> {
        let cancel = CancellationToken::new();
        let (done_tx, done_rx) = watch::channel(false);
        {
            let mut guard = self.lock();
            if guard.is_some() {
                return Err(ShellError::new(
                    "bootstrap.running",
                    "The runtime installation is already running",
                ));
            }
            *guard = Some(Running {
                cancel: cancel.clone(),
                done: done_rx,
            });
        }
        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            let shell = crate::shell(&app);
            let (mut rep, outcome, start_backend) = match job {
                Job::Runtime(request) => {
                    let (rep, outcome) = run(&app, request, cancel).await;
                    let start =
                        matches!(outcome, Outcome::Done(_)) && shell.settings.get().first_run_done;
                    (rep, outcome, start)
                }
                // Restart a backend we stopped for the install, whatever the outcome.
                Job::Requirements { python, region } => {
                    requirements::run(&app, python, region, cancel).await
                }
            };
            // Free the slot first: whoever reacts to the final event must already see
            // `installing` gone and be able to start the next job.
            shell.bootstrap.lock().take();
            outcome.report(&mut rep);
            let _ = done_tx.send(true);
            if start_backend && !shell.is_quitting() {
                if let Err(err) = shell.backend.start(&app, true).await {
                    log::warn!("backend did not start after the runtime job: {err}");
                }
            }
        });
        Ok(())
    }

    /// Cancels a running bootstrap (kills uv's process tree) and waits for it to wind down.
    pub async fn cancel(&self) {
        let job = self
            .lock()
            .as_ref()
            .map(|r| (r.cancel.clone(), r.done.clone()));
        if let Some((cancel, mut done)) = job {
            cancel.cancel();
            let _ = tokio::time::timeout(Duration::from_secs(15), done.wait_for(|d| *d)).await;
        }
    }
}

async fn run(app: &AppHandle, request: Request, cancel: CancellationToken) -> (Reporter, Outcome) {
    let shell = crate::shell(app);
    let mut rep = Reporter::new(
        progress::event_sink(app.clone()),
        progress::weights(request.variant),
        shell.paths.logs_dir.join("bootstrap.log"),
    );
    rep.log(&format!(
        "=== bootstrap {:?}: variant {}, region {:?}, app {}",
        request.mode,
        request.variant.as_str(),
        request.region,
        app.package_info().version
    ));
    rep.emit(true);

    if let Some(python) = &shell.resources.dev_python {
        return (
            rep,
            Outcome::Done(format!("Dev mode: using {}", python.display())),
        );
    }

    // The user's own Python: nothing to install when it passes the check.
    let custom = shell
        .settings
        .get()
        .runtime
        .custom_python()
        .map(Path::to_path_buf);
    if let Some(python) = custom {
        rep.set_detail("Checking your Python…");
        rep.emit(true);
        let check = tokio::select! {
            _ = cancel.cancelled() => {
                return (rep, Outcome::Cancelled);
            }
            check = shell.check_python(&python) => check,
        };
        if check.ok {
            return (
                rep,
                Outcome::Done(format!(
                    "Using your own Python {} ({})",
                    check.version.unwrap_or_default(),
                    python.display()
                )),
            );
        }
        rep.log(&format!(
            "custom Python {} failed its check: {:?}",
            python.display(),
            check.error
        ));
        if request.mode == Mode::Repair {
            // Repairing must not silently replace the user's environment with ours.
            return (rep, Outcome::Failed(crate::pycheck::invalid_error(&check)));
        }
        // An explicit install request: the managed runtime it is.
        if let Err(err) = shell
            .settings
            .update(|s| s.runtime.mode = RuntimeMode::Managed)
        {
            return (rep, Outcome::Failed(err));
        }
    }

    // The backend runs from the venv we are about to change.
    if shell.backend.is_running() {
        rep.set_detail("Stopping the backend…");
        rep.emit(true);
        shell.backend.stop(app, true).await;
    }

    let manifest = match Manifest::load(&shell.resources.manifest) {
        Ok(manifest) => manifest,
        Err(err) => {
            return (rep, Outcome::Failed(err));
        }
    };
    let mut pipeline = Pipeline::new(
        shell.paths.clone(),
        shell.resources.clone(),
        manifest,
        request,
        cancel,
        shell.net_http.clone(),
    );
    let outcome = match pipeline.run(&mut rep).await {
        Ok(state) => Outcome::Done(format!(
            "Runtime ready: Python {}, PyTorch {}",
            state.python, state.torch
        )),
        Err(err) => err.into(),
    };
    (rep, outcome)
}

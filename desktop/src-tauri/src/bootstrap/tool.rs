//! Running setup tools (uv, the venv's Python): output streaming into progress observers,
//! cancellation and timeouts, and turning failures into actionable errors.

use std::collections::VecDeque;
use std::process::Stdio;
use std::time::{Duration, Instant};

use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::progress::{creep, Reporter};
use super::uvlog::{PipTracker, PythonTracker};
use super::{StageError, StageResult};
use crate::error::ShellError;
use crate::procutil::{spawn_line_reader, ProcessTree, Stream};

const TAIL_LINES: usize = 60;

#[derive(Default)]
pub(super) struct ToolOutput {
    pub success: bool,
    pub code: Option<i32>,
    pub stdout: Vec<String>,
    tail: VecDeque<String>,
}

impl ToolOutput {
    fn push(
        &mut self,
        rep: &mut Reporter,
        observer: &mut dyn Observer,
        stream: Stream,
        line: String,
    ) {
        rep.log(&line);
        observer.line(rep, &line);
        if stream == Stream::Stdout && self.stdout.len() < 10_000 {
            self.stdout.push(line.clone());
        }
        self.tail.push_back(line);
        if self.tail.len() > TAIL_LINES {
            self.tail.pop_front();
        }
    }

    pub fn tail_text(&self) -> String {
        self.tail.iter().cloned().collect::<Vec<_>>().join("\n")
    }
}

/// Runs `cmd` to completion, streaming its output into `observer` (and the log).
/// Cancellation and the optional `timeout` kill the whole process tree.
pub(super) async fn run_tool(
    cancel: &CancellationToken,
    mut cmd: Command,
    rep: &mut Reporter,
    observer: &mut dyn Observer,
    timeout: Option<Duration>,
) -> StageResult<ToolOutput> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    rep.log(&format!("$ {}", describe(&cmd)));
    let mut child = cmd.spawn().map_err(|e| {
        ShellError::new("bootstrap.spawn_failed", "Cannot start a setup tool")
            .with_detail(format!("{}: {e}", describe(&cmd)))
    })?;
    let tree = ProcessTree::attach(&child);
    let (tx, mut rx) = mpsc::unbounded_channel();
    if let Some(stdout) = child.stdout.take() {
        spawn_line_reader(stdout, Stream::Stdout, tx.clone());
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_line_reader(stderr, Stream::Stderr, tx.clone());
    }
    drop(tx);

    let mut out = ToolOutput::default();
    let mut tick = tokio::time::interval(Duration::from_millis(500));
    let deadline = tokio::time::sleep(timeout.unwrap_or(Duration::from_secs(60 * 60 * 24 * 7)));
    tokio::pin!(deadline);
    let mut lines_open = true;
    let started = Instant::now();
    let status = loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                tree.kill();
                let _ = child.wait().await;
                return Err(StageError::Cancelled);
            }
            _ = &mut deadline => {
                tree.kill();
                let _ = child.wait().await;
                return Err(ShellError::new("bootstrap.timeout", "A setup step took too long")
                    .with_detail(format!("{} (after {} s)", describe(&cmd), started.elapsed().as_secs()))
                    .into());
            }
            message = rx.recv(), if lines_open => match message {
                Some((stream, line)) => out.push(rep, observer, stream, line),
                None => lines_open = false,
            },
            _ = tick.tick() => observer.tick(rep),
            status = child.wait() => break status,
        }
    };
    // Stragglers holding the pipes open must not keep us waiting.
    tree.kill();
    let _ = tokio::time::timeout(Duration::from_secs(2), async {
        while let Some((stream, line)) = rx.recv().await {
            out.push(rep, observer, stream, line);
        }
    })
    .await;
    let status = status.map_err(|e| ShellError::io("A setup tool failed", &e))?;
    out.success = status.success();
    out.code = status.code();
    rep.log(&format!(
        "exit: {:?} after {} s",
        out.code,
        started.elapsed().as_secs()
    ));
    if out.success {
        observer.tick(rep);
    }
    Ok(out)
}

fn describe(cmd: &Command) -> String {
    let std = cmd.as_std();
    let mut parts = vec![std.get_program().to_string_lossy().into_owned()];
    parts.extend(std.get_args().map(|a| {
        let a = a.to_string_lossy();
        if a.contains('\n') {
            "<script>".to_string()
        } else {
            a.into_owned()
        }
    }));
    parts.join(" ")
}

pub(super) const DISK_FULL: &str = "bootstrap.disk_full";

/// Maps tool failures to actionable codes (disk full / network) when the output says so.
pub(super) fn tool_error(code: &str, message: &str, out: &ToolOutput) -> ShellError {
    let detail = out.tail_text();
    let lower = detail.to_lowercase();
    const DISK: &[&str] = &[
        "no space left on device",
        "os error 112",
        "os error 28",
        "not enough space on the disk",
    ];
    const NETWORK: &[&str] = &[
        "error sending request",
        "dns error",
        "tcp connect error",
        "connection reset",
        "connection refused",
        "connection closed",
        "timed out",
        "failed to fetch",
        "failed to download",
        "network is unreachable",
        "certificate",
        "403 forbidden",
        "404 not found",
    ];
    let (code, message) = if DISK.iter().any(|m| lower.contains(m)) {
        (DISK_FULL, "Not enough free disk space")
    } else if NETWORK.iter().any(|m| lower.contains(m)) {
        ("bootstrap.network", "Could not download from the mirror")
    } else {
        (code, message)
    };
    ShellError::new(code, message).with_detail(detail)
}

/// Failures of one stage across mirrors. The first one (the preferred / fastest mirror)
/// is reported; the others are summarized in its detail.
#[derive(Default)]
pub(super) struct Failures {
    list: Vec<(String, ShellError)>,
}

impl Failures {
    /// Records a failure; `Err` when retrying elsewhere is pointless (disk full).
    pub fn push(&mut self, source: &str, err: ShellError) -> StageResult<()> {
        if err.code == DISK_FULL {
            return Err(err.into());
        }
        self.list.push((source.to_string(), err));
        Ok(())
    }

    pub fn into_error(self, code: &str, message: &str) -> StageError {
        let mut list = self.list.into_iter();
        let Some((source, mut first)) = list.next() else {
            return ShellError::new(code, message).into();
        };
        let others: Vec<String> = list
            .map(|(id, err)| format!("- {id}: {} ({})", err.message, err.code))
            .collect();
        first.detail = format!("[{source}]\n{}", first.detail);
        if !others.is_empty() {
            first
                .detail
                .push_str(&format!("\n\nOther mirrors tried:\n{}", others.join("\n")));
        }
        first.into()
    }
}

pub(super) trait Observer: Send {
    fn line(&mut self, rep: &mut Reporter, line: &str);
    fn tick(&mut self, rep: &mut Reporter);
}

/// `uv pip install`: phases, announced download sizes, estimated bytes and speed.
pub(super) struct PipObserver {
    pub tracker: PipTracker,
    /// Part of the stage's progress this invocation covers.
    pub range: (f64, f64),
}

impl Observer for PipObserver {
    fn line(&mut self, rep: &mut Reporter, line: &str) {
        match self.tracker.line(line) {
            Some(detail) => rep.set_detail(detail),
            None => rep.set_detail(line.trim()),
        }
        self.tick(rep);
    }

    fn tick(&mut self, rep: &mut Reporter) {
        let snap = self.tracker.snapshot();
        rep.set_progress(self.range.0 + (self.range.1 - self.range.0) * snap.fraction);
        if snap.bytes_total.is_some() {
            rep.set_bytes(snap.bytes_done, snap.bytes_total, snap.speed);
        }
        rep.emit(false);
    }
}

/// `uv python install`.
pub(super) struct PythonObserver {
    pub tracker: PythonTracker,
}

impl Observer for PythonObserver {
    fn line(&mut self, rep: &mut Reporter, line: &str) {
        if let Some(detail) = self.tracker.line(line) {
            rep.set_detail(detail);
        }
        self.tick(rep);
    }

    fn tick(&mut self, rep: &mut Reporter) {
        let (fraction, done, total, speed) = self.tracker.snapshot();
        rep.set_progress(fraction);
        if total.is_some() {
            rep.set_bytes(done, total, speed);
        }
        rep.emit(false);
    }
}

/// For silent steps: progress approaches the end of `range` with time constant `tau`.
pub(super) struct CreepObserver {
    started: Instant,
    tau: f64,
    range: (f64, f64),
}

impl CreepObserver {
    pub fn new(tau: f64, range: (f64, f64)) -> Self {
        Self {
            started: Instant::now(),
            tau,
            range,
        }
    }
}

impl Observer for CreepObserver {
    fn line(&mut self, rep: &mut Reporter, line: &str) {
        if !line.starts_with("STELLARIA_") {
            rep.set_detail(line.trim());
        }
        self.tick(rep);
    }

    fn tick(&mut self, rep: &mut Reporter) {
        rep.set_progress(creep(
            self.range.0,
            self.range.1,
            self.started.elapsed(),
            self.tau,
        ));
        rep.emit(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output(lines: &[&str]) -> ToolOutput {
        let mut out = ToolOutput::default();
        out.tail.extend(lines.iter().map(|l| l.to_string()));
        out
    }

    #[test]
    fn classifies_tool_errors() {
        let net = output(&[
            "error: Failed to fetch: `https://x`",
            "  Caused by: error sending request for url",
        ]);
        assert_eq!(
            tool_error("bootstrap.torch_failed", "x", &net).code,
            "bootstrap.network"
        );
        let disk = output(&["No space left on device (os error 28)"]);
        assert_eq!(
            tool_error("bootstrap.torch_failed", "x", &disk).code,
            DISK_FULL
        );
        let other = output(&["× No solution found when resolving dependencies"]);
        assert_eq!(
            tool_error("bootstrap.torch_failed", "x", &other).code,
            "bootstrap.torch_failed"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancel_kills_the_tool_tree() {
        use crate::procutil::hidden_command;
        let dir = std::env::temp_dir().join(format!("stellaria-cancel-{}", std::process::id()));
        let mut rep = Reporter::new(
            Box::new(|_| {}),
            super::super::progress::weights(crate::types::TorchVariant::Cpu),
            dir.join("bootstrap.log"),
        );
        let mut cmd = hidden_command("sh");
        // A child that would outlive its parent if only the parent were killed.
        cmd.args(["-c", "sleep 30 & echo started; wait"]);
        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            trigger.cancel();
        });
        let mut observer = CreepObserver::new(5.0, (0.0, 1.0));
        let started = Instant::now();
        let result = run_tool(&cancel, cmd, &mut rep, &mut observer, None).await;
        assert!(matches!(result, Err(StageError::Cancelled)));
        assert!(started.elapsed() < Duration::from_secs(5));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_is_an_error() {
        use crate::procutil::hidden_command;
        let dir = std::env::temp_dir().join(format!("stellaria-timeout-{}", std::process::id()));
        let mut rep = Reporter::new(
            Box::new(|_| {}),
            super::super::progress::weights(crate::types::TorchVariant::Cpu),
            dir.join("bootstrap.log"),
        );
        let mut cmd = hidden_command("sleep");
        cmd.arg("30");
        let mut observer = CreepObserver::new(5.0, (0.0, 1.0));
        let result = run_tool(
            &CancellationToken::new(),
            cmd,
            &mut rep,
            &mut observer,
            Some(Duration::from_millis(300)),
        )
        .await;
        match result {
            Err(StageError::Failed(err)) => assert_eq!(err.code, "bootstrap.timeout"),
            _ => panic!("expected a timeout"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn failures_report_the_first_mirror() {
        let mut failures = Failures::default();
        assert!(failures
            .push(
                "pypi",
                ShellError::new("bootstrap.torch_failed", "first").with_detail("why")
            )
            .is_ok());
        assert!(failures
            .push("tuna", ShellError::new("bootstrap.network", "second"))
            .is_ok());
        match failures.into_error("bootstrap.torch_failed", "fallback") {
            StageError::Failed(err) => {
                assert_eq!(err.message, "first");
                assert!(err.detail.starts_with("[pypi]\nwhy"));
                assert!(err.detail.contains("- tuna: second (bootstrap.network)"));
            }
            StageError::Cancelled => panic!("unexpected"),
        }
        let mut failures = Failures::default();
        assert!(failures
            .push("pypi", ShellError::new(DISK_FULL, "full"))
            .is_err());
    }
}

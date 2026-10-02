//! `bootstrap://progress` events: stage bookkeeping, weighted overall progress, throttling
//! and a plain-text log (`logs/bootstrap.log`).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

/// Where progress goes: `bootstrap://progress` in the app, a collector in tests.
pub type Sink = Box<dyn Fn(&BootstrapProgress) + Send + Sync>;

/// Sink emitting `bootstrap://progress` to the frontend.
pub fn event_sink(app: AppHandle) -> Sink {
    Box::new(move |progress| {
        if let Err(err) = app.emit(EVENT, progress) {
            log::warn!("cannot emit bootstrap progress: {err}");
        }
    })
}

use crate::error::ShellError;
use crate::fsutil::RotatingFile;
use crate::types::TorchVariant;

pub const EVENT: &str = "bootstrap://progress";
const EMIT_INTERVAL: Duration = Duration::from_millis(150);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    Prepare,
    Python,
    Venv,
    Torch,
    Packages,
    Verify,
    Done,
    Error,
    Cancelled,
}

/// The six working stages, in order.
pub const STAGES: [Stage; 6] = [
    Stage::Prepare,
    Stage::Python,
    Stage::Venv,
    Stage::Torch,
    Stage::Packages,
    Stage::Verify,
];

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BootstrapProgress {
    pub stage: Stage,
    pub stage_index: usize,
    pub stage_count: usize,
    pub progress: Option<f64>,
    pub overall: Option<f64>,
    pub message: String,
    pub detail: Option<String>,
    pub bytes_done: Option<u64>,
    pub bytes_total: Option<u64>,
    pub speed: Option<f64>,
    pub source: Option<String>,
    pub error: Option<ShellError>,
}

/// Relative duration of each stage; the torch download dominates the CUDA variants.
pub fn weights(variant: TorchVariant) -> [f64; 6] {
    if variant.is_cuda() {
        [0.02, 0.05, 0.02, 0.70, 0.16, 0.05]
    } else {
        [0.03, 0.10, 0.03, 0.35, 0.42, 0.07]
    }
}

/// For jobs that consist of one stage (installing requirements into the user's Python):
/// `overall` equals that stage's progress.
pub fn single_stage(stage: Stage) -> [f64; 6] {
    let mut weights = [0.0; 6];
    if let Some(index) = STAGES.iter().position(|s| *s == stage) {
        weights[index] = 1.0;
    }
    weights
}

pub struct Reporter {
    sink: Sink,
    weights: [f64; 6],
    current: BootstrapProgress,
    last_emit: Option<Instant>,
    log: RotatingFile,
}

impl Reporter {
    pub fn new(sink: Sink, weights: [f64; 6], log_path: PathBuf) -> Self {
        Self {
            sink,
            weights,
            current: BootstrapProgress {
                stage: Stage::Prepare,
                stage_index: 0,
                stage_count: STAGES.len(),
                progress: Some(0.0),
                overall: Some(0.0),
                message: "Preparing".into(),
                detail: None,
                bytes_done: None,
                bytes_total: None,
                speed: None,
                source: None,
                error: None,
            },
            last_emit: None,
            log: RotatingFile::new(log_path, 2 * 1024 * 1024, 2),
        }
    }

    pub fn begin_stage(&mut self, stage: Stage, message: impl Into<String>) {
        let index = STAGES.iter().position(|s| *s == stage).unwrap_or(0);
        let message = message.into();
        self.log(&format!("--- {message}"));
        self.current.stage = stage;
        self.current.stage_index = index;
        self.current.message = message;
        self.current.detail = None;
        self.current.source = None;
        self.clear_bytes();
        self.current.progress = Some(0.0);
        self.update_overall();
        self.emit(true);
    }

    /// Progress within the stage (0..1). Never moves backwards unless [`Self::reset_progress`].
    pub fn set_progress(&mut self, progress: f64) {
        let p = progress.clamp(0.0, 1.0);
        let current = self.current.progress.unwrap_or(0.0);
        if p > current {
            self.current.progress = Some(p);
            self.update_overall();
        }
    }

    /// Used when a stage starts over with another mirror.
    pub fn reset_progress(&mut self) {
        self.current.progress = Some(0.0);
        self.clear_bytes();
        self.update_overall();
    }

    pub fn set_detail(&mut self, detail: impl Into<String>) {
        self.current.detail = Some(truncate(&detail.into(), 240));
    }

    pub fn set_source(&mut self, source: Option<String>) {
        if let Some(s) = &source {
            self.log(&format!("source: {s}"));
        }
        self.current.source = source;
    }

    pub fn set_bytes(&mut self, done: Option<u64>, total: Option<u64>, speed: Option<f64>) {
        self.current.bytes_done = done;
        self.current.bytes_total = total;
        self.current.speed = speed;
    }

    pub fn clear_bytes(&mut self) {
        self.set_bytes(None, None, None);
    }

    pub fn log(&mut self, line: &str) {
        let stamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
        self.log.write_line(&format!("{stamp} {line}"));
    }

    pub fn emit(&mut self, force: bool) {
        let now = Instant::now();
        if !force
            && self
                .last_emit
                .is_some_and(|t| now.duration_since(t) < EMIT_INTERVAL)
        {
            return;
        }
        self.last_emit = Some(now);
        (self.sink)(&self.current);
    }

    pub fn finish(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.log(&format!("=== done: {message}"));
        self.current.stage = Stage::Done;
        self.current.stage_index = STAGES.len();
        self.current.progress = Some(1.0);
        self.current.overall = Some(1.0);
        self.current.message = message;
        self.current.detail = None;
        self.clear_bytes();
        self.emit(true);
    }

    pub fn fail(&mut self, error: ShellError) {
        self.log(&format!(
            "=== error {}: {}\n{}",
            error.code, error.message, error.detail
        ));
        self.current.stage = Stage::Error;
        self.current.message = error.message.clone();
        self.current.error = Some(error);
        self.current.speed = None;
        self.emit(true);
    }

    pub fn cancelled(&mut self) {
        self.log("=== cancelled");
        self.current.stage = Stage::Cancelled;
        self.current.message = "Cancelled".into();
        self.current.speed = None;
        self.emit(true);
    }

    fn update_overall(&mut self) {
        let index = self.current.stage_index.min(self.weights.len() - 1);
        let done: f64 = self.weights[..index].iter().sum();
        let within = self.weights[index] * self.current.progress.unwrap_or(0.0);
        self.current.overall = Some((done + within).clamp(0.0, 1.0));
    }
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars).collect();
    out.push('…');
    out
}

/// Exponential approach from `from` to `to`: keeps progress moving while a tool is silent.
pub fn creep(from: f64, to: f64, elapsed: Duration, tau_secs: f64) -> f64 {
    from + (to - from) * (1.0 - (-elapsed.as_secs_f64() / tau_secs).exp())
}

/// Bytes/second over a sliding window of a few seconds.
#[derive(Default)]
pub struct SpeedMeter {
    samples: std::collections::VecDeque<(Instant, u64)>,
}

impl SpeedMeter {
    pub fn sample(&mut self, bytes: u64) -> Option<f64> {
        let now = Instant::now();
        self.samples.push_back((now, bytes));
        while self
            .samples
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t) > Duration::from_secs(5))
        {
            self.samples.pop_front();
        }
        let (t0, b0) = *self.samples.front()?;
        let dt = now.duration_since(t0).as_secs_f64();
        (dt >= 0.5).then(|| bytes.saturating_sub(b0) as f64 / dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creep_is_bounded() {
        assert!((creep(0.1, 0.9, Duration::ZERO, 10.0) - 0.1).abs() < 1e-9);
        let late = creep(0.1, 0.9, Duration::from_secs(1000), 10.0);
        assert!(late <= 0.9 && late > 0.89);
    }

    #[test]
    fn weights_sum_to_one() {
        for v in [TorchVariant::Cu128, TorchVariant::Mac] {
            let sum: f64 = weights(v).iter().sum();
            assert!((sum - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn progress_serializes_camel_case() {
        let p = BootstrapProgress {
            stage: Stage::Torch,
            stage_index: 3,
            stage_count: 6,
            progress: Some(0.5),
            overall: Some(0.4),
            message: "x".into(),
            detail: None,
            bytes_done: Some(1),
            bytes_total: Some(2),
            speed: None,
            source: Some("pytorch".into()),
            error: None,
        };
        let v = serde_json::to_value(p).unwrap();
        assert_eq!(v["stage"], "torch");
        assert_eq!(v["stageIndex"], 3);
        assert_eq!(v["bytesTotal"], 2);
        assert!(v["error"].is_null());
    }
}

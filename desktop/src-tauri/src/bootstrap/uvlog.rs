//! Turns uv's non-interactive output into progress.
//!
//! `uv pip install` without a TTY prints (0.8.x):
//! ```text
//! Resolved 24 packages in 1.23s
//! Downloading torch (846.8MiB)        <- start of a large download, with its size
//!  Downloading torch                  <- (leading space) that download finished
//! Prepared 24 packages in 45.6s
//! Installed 24 packages in 300ms
//!  + torch==2.9.1
//! ```
//! Bytes of in-flight downloads are estimated from the growth of the uv cache (uv unpacks
//! wheels into it while streaming), which also yields a live speed.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use regex::Regex;

use super::progress::{creep, SpeedMeter};
use crate::fsutil::dir_size;

/// Unpacked wheels take roughly twice their download size in the cache.
const UNPACK_RATIO: f64 = 2.0;

/// Measures a directory's size every 1.5 s on a blocking thread.
pub struct DirSampler {
    size: Arc<AtomicU64>,
    task: tokio::task::JoinHandle<()>,
}

impl DirSampler {
    pub fn start(dir: PathBuf) -> Self {
        let size = Arc::new(AtomicU64::new(dir_size(&dir)));
        let shared = size.clone();
        let task = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(1500)).await;
                let dir = dir.clone();
                if let Ok(bytes) = tokio::task::spawn_blocking(move || dir_size(&dir)).await {
                    shared.store(bytes, Ordering::Relaxed);
                }
            }
        });
        Self { size, task }
    }

    pub fn bytes(&self) -> u64 {
        self.size.load(Ordering::Relaxed)
    }
}

impl Drop for DirSampler {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Resolving,
    Downloading,
    Installing,
    Done,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    /// 0..1 for this uv invocation.
    pub fraction: f64,
    pub bytes_done: Option<u64>,
    pub bytes_total: Option<u64>,
    pub speed: Option<f64>,
}

pub struct PipTracker {
    phase: Phase,
    phase_started: Instant,
    announced: HashMap<String, u64>,
    total: u64,
    finished: u64,
    finished_count: usize,
    all_downloaded: Option<Instant>,
    cache: Option<DirSampler>,
    cache_base: u64,
    meter: SpeedMeter,
    pub installed: Vec<String>,
}

impl PipTracker {
    pub fn new(cache: Option<DirSampler>) -> Self {
        let cache_base = cache.as_ref().map(DirSampler::bytes).unwrap_or(0);
        Self {
            phase: Phase::Resolving,
            phase_started: Instant::now(),
            announced: HashMap::new(),
            total: 0,
            finished: 0,
            finished_count: 0,
            all_downloaded: None,
            cache,
            cache_base,
            meter: SpeedMeter::default(),
            installed: Vec::new(),
        }
    }

    fn enter(&mut self, phase: Phase) {
        if self.phase != phase {
            self.phase = phase;
            self.phase_started = Instant::now();
        }
    }

    /// Feeds one output line; returns a friendlier detail text when there is one.
    pub fn line(&mut self, line: &str) -> Option<String> {
        let re = patterns();
        if re.resolved.is_match(line) {
            self.enter(Phase::Downloading);
            return Some(line.trim().to_string());
        }
        if let Some(c) = re.download_start.captures(line) {
            let name = c[1].to_string();
            let bytes = parse_size(&c[2], &c[3]);
            if self.announced.insert(name, bytes).is_none() {
                self.total += bytes;
            }
            self.enter(Phase::Downloading);
            return Some(format!(
                "Downloading {} packages ({})",
                self.announced.len(),
                human_size(self.total)
            ));
        }
        if let Some(c) = re.download_done.captures(line) {
            if let Some(bytes) = self.announced.get(&c[1]) {
                self.finished += bytes;
                self.finished_count += 1;
            }
            if self.finished_count == self.announced.len() {
                // uv still unpacks (and maybe builds) before it reports "Prepared".
                self.all_downloaded = Some(Instant::now());
                return Some("Downloads finished, unpacking packages…".into());
            }
            return Some(format!(
                "Downloaded {} ({}/{})",
                &c[1],
                self.finished_count,
                self.announced.len()
            ));
        }
        if re.prepared.is_match(line) {
            self.enter(Phase::Installing);
            return Some("Installing packages…".into());
        }
        if re.finished.is_match(line) {
            self.enter(Phase::Done);
            return Some(line.trim().to_string());
        }
        if let Some(c) = re.added.captures(line) {
            self.installed.push(c[1].to_string());
            return Some(format!("Installed {}", &c[1]));
        }
        None
    }

    pub fn snapshot(&mut self) -> Snapshot {
        let elapsed = self.phase_started.elapsed();
        let mut bytes_done = None;
        let fraction = match self.phase {
            Phase::Resolving => creep(0.0, 0.10, elapsed, 15.0),
            Phase::Downloading if self.all_downloaded.is_some() => {
                let since = self.all_downloaded.map(|t| t.elapsed()).unwrap_or_default();
                bytes_done = Some(self.total);
                creep(0.88, 0.97, since, 30.0)
            }
            Phase::Downloading if self.total > 0 => {
                let grown = self
                    .cache
                    .as_ref()
                    .map(|c| c.bytes().saturating_sub(self.cache_base))
                    .unwrap_or(0);
                let estimate = (grown as f64 / UNPACK_RATIO) as u64;
                let cap = if self.finished >= self.total {
                    self.total
                } else {
                    (self.total as f64 * 0.99) as u64
                };
                let done = estimate.max(self.finished).min(cap);
                bytes_done = Some(done);
                0.10 + 0.78 * (done as f64 / self.total as f64)
            }
            Phase::Downloading => creep(0.10, 0.85, elapsed, 45.0),
            Phase::Installing => creep(0.88, 0.99, elapsed, 20.0),
            Phase::Done => 1.0,
        };
        let speed = match self.all_downloaded {
            Some(_) => None,
            None => bytes_done.and_then(|b| self.meter.sample(b)),
        };
        Snapshot {
            fraction,
            bytes_done,
            bytes_total: (self.total > 0 && bytes_done.is_some()).then_some(self.total),
            speed,
        }
    }
}

/// `uv python install` prints `Downloading cpython-… (download) (27.7MiB)` once and a
/// line with a leading space when that finished; bytes in between are estimated from the
/// growth of the install dir (the archive is unpacked while streaming, ~3.5× its size).
pub struct PythonTracker {
    started: Instant,
    dir: DirSampler,
    base: u64,
    total: Option<u64>,
    downloaded: bool,
    done: bool,
    meter: SpeedMeter,
}

const PYTHON_UNPACK_RATIO: f64 = 3.5;

impl PythonTracker {
    pub fn new(dir: DirSampler) -> Self {
        let base = dir.bytes();
        Self {
            started: Instant::now(),
            dir,
            base,
            total: None,
            downloaded: false,
            done: false,
            meter: SpeedMeter::default(),
        }
    }

    pub fn line(&mut self, line: &str) -> Option<String> {
        static START: OnceLock<Regex> = OnceLock::new();
        let start = START.get_or_init(|| {
            Regex::new(r"^Downloading (.+) \(([\d.]+)\s*([KMGT]i?B|B)\)\s*$").expect("valid regex")
        });
        let trimmed = line.trim();
        if let Some(c) = start.captures(line) {
            self.total = Some(parse_size(&c[2], &c[3]));
            return Some(format!("Downloading Python ({}{})", &c[2], &c[3]));
        }
        if line.starts_with(" Download") {
            self.downloaded = true;
            return Some("Unpacking Python…".into());
        }
        if trimmed.starts_with("Installed Python") || trimmed.contains("already installed") {
            self.done = true;
        }
        Some(trimmed.to_string())
    }

    /// `(fraction, bytes_done, bytes_total, speed)`.
    pub fn snapshot(&mut self) -> (f64, Option<u64>, Option<u64>, Option<f64>) {
        if self.done {
            return (1.0, self.total, self.total, None);
        }
        let time = creep(0.0, 0.9, self.started.elapsed(), 40.0);
        let Some(total) = self.total.filter(|t| *t > 0) else {
            return (time, None, None, None);
        };
        let grown = self.dir.bytes().saturating_sub(self.base) as f64;
        let done = if self.downloaded {
            total
        } else {
            ((grown / PYTHON_UNPACK_RATIO) as u64).min(total * 99 / 100)
        };
        let fraction = (0.9 * done as f64 / total as f64).max(time.min(0.85));
        (fraction, Some(done), Some(total), self.meter.sample(done))
    }
}

struct Patterns {
    resolved: Regex,
    download_start: Regex,
    download_done: Regex,
    prepared: Regex,
    finished: Regex,
    added: Regex,
}

fn patterns() -> &'static Patterns {
    static P: OnceLock<Patterns> = OnceLock::new();
    P.get_or_init(|| {
        let re = |s: &str| Regex::new(s).expect("valid regex");
        Patterns {
            resolved: re(r"^Resolved \d+ packages?"),
            download_start: re(r"^Downloading (\S+) \(([\d.]+)\s*([KMGT]i?B|B)\)\s*$"),
            download_done: re(r"^ (?:Downloading|Downloaded) (\S+)\s*$"),
            prepared: re(r"^Prepared \d+ packages?"),
            finished: re(r"^(?:Installed|Audited) \d+ packages?"),
            added: re(r"^ \+ (\S+)"),
        }
    })
}

fn human_size(bytes: u64) -> String {
    let mib = bytes as f64 / (1024.0 * 1024.0);
    if mib >= 1024.0 {
        format!("{:.1} GiB", mib / 1024.0)
    } else {
        format!("{mib:.0} MiB")
    }
}

fn parse_size(number: &str, unit: &str) -> u64 {
    let value: f64 = number.parse().unwrap_or(0.0);
    let factor = match unit {
        "KiB" => 1024.0,
        "MiB" => 1024.0 * 1024.0,
        "GiB" => 1024.0 * 1024.0 * 1024.0,
        "TiB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        "kB" | "KB" => 1e3,
        "MB" => 1e6,
        "GB" => 1e9,
        "TB" => 1e12,
        _ => 1.0,
    };
    (value * factor) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_uv_output() {
        let mut t = PipTracker::new(None);
        assert!(t.snapshot().fraction < 0.11);
        t.line("Resolved 2 packages in 124ms");
        assert_eq!(
            t.line("Downloading scipy (35.9MiB)").as_deref(),
            Some("Downloading 1 packages (36 MiB)")
        );
        t.line("Downloading numpy (16.0MiB)");
        assert_eq!(
            t.total,
            parse_size("35.9", "MiB") + parse_size("16.0", "MiB")
        );
        t.line(" Downloading numpy");
        assert_eq!(t.finished, parse_size("16.0", "MiB"));
        let s = t.snapshot();
        assert!(s.fraction > 0.1 && s.fraction < 0.88);
        assert_eq!(s.bytes_total, Some(t.total));
        t.line(" Downloaded scipy");
        t.line("Prepared 2 packages in 1.20s");
        assert!(t.snapshot().fraction >= 0.88);
        t.line("Installed 2 packages in 24ms");
        t.line(" + numpy==2.2.6");
        assert_eq!(t.snapshot().fraction, 1.0);
        assert_eq!(t.installed, vec!["numpy==2.2.6".to_string()]);
    }

    #[test]
    fn python_download_line() {
        let re = Regex::new(r"^Downloading (.+) \(([\d.]+)\s*([KMGT]i?B|B)\)\s*$").unwrap();
        let c = re
            .captures("Downloading cpython-3.11.13-linux-x86_64-gnu (download) (27.7MiB)")
            .unwrap();
        assert_eq!(&c[2], "27.7");
        assert_eq!(&c[3], "MiB");
    }

    #[test]
    fn audited_means_done() {
        let mut t = PipTracker::new(None);
        t.line("Audited 3 packages in 5ms");
        assert_eq!(t.snapshot().fraction, 1.0);
    }
}

//! The torch wheel is by far the largest download (≈2.5–3 GB for CUDA on Windows), so the
//! shell fetches it itself — resumable (HTTP Range), verified (sha256 from the index) and
//! with exact byte progress — and hands the file to uv.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use percent_encoding::percent_decode_str;
use regex::Regex;
use reqwest::header::{CONTENT_RANGE, RANGE};
use reqwest::{StatusCode, Url};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

use super::progress::SpeedMeter;
use super::{StageError, StageResult};
use crate::error::ShellError;
use crate::manifest::SourceKind;

const MAX_ATTEMPTS: u32 = 5;
const CHUNK_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WheelRef {
    pub url: Url,
    pub filename: String,
    pub sha256: Option<String>,
}

/// Page listing the wheels of `dist` for a torch source.
pub fn listing_url(kind: SourceKind, source_url: &str, dist: &str) -> String {
    let base = source_url.trim_end_matches('/');
    match kind {
        SourceKind::FindLinks => format!("{base}/"),
        SourceKind::Index | SourceKind::Pypi => format!("{base}/{dist}/"),
    }
}

/// `cp311` for `3.11`.
pub fn python_tag(python: &str) -> String {
    let mut parts = python.split('.');
    let major = parts.next().unwrap_or("3");
    let minor = parts.next().unwrap_or("11");
    format!("cp{major}{minor}")
}

pub async fn find_wheel(
    client: &reqwest::Client,
    page_url: &str,
    dist: &str,
    version: &str,
    py_tag: &str,
) -> Result<WheelRef, ShellError> {
    let not_found = |detail: String| {
        ShellError::new(
            "bootstrap.wheel_not_found",
            format!("{dist} {version} is not available from this mirror"),
        )
        .with_detail(detail)
    };
    let response = client
        .get(page_url)
        .timeout(Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| network_error(page_url, &e))?;
    if !response.status().is_success() {
        return Err(not_found(format!("{page_url}: HTTP {}", response.status())));
    }
    let base = response.url().clone();
    let html = response
        .text()
        .await
        .map_err(|e| network_error(page_url, &e))?;
    parse_listing(&html, &base)
        .into_iter()
        .find(|w| {
            wheel_matches(
                &w.filename,
                dist,
                version,
                py_tag,
                std::env::consts::OS,
                std::env::consts::ARCH,
            )
        })
        .ok_or_else(|| {
            not_found(format!(
                "{page_url}: no {dist}-{version}-{py_tag} wheel for this platform"
            ))
        })
}

/// All `<a href>` targets of a PEP 503 page or a flat directory listing.
pub fn parse_listing(html: &str, base: &Url) -> Vec<WheelRef> {
    static HREF: OnceLock<Regex> = OnceLock::new();
    let re = HREF.get_or_init(|| {
        Regex::new(r#"(?i)<a\s[^>]*?href\s*=\s*["']([^"']+)["']"#).expect("valid regex")
    });
    re.captures_iter(html)
        .filter_map(|caps| {
            let href = caps.get(1)?.as_str().replace("&amp;", "&");
            let (path, fragment) = match href.split_once('#') {
                Some((p, f)) => (p.to_string(), Some(f.to_string())),
                None => (href.clone(), None),
            };
            let url = base.join(&path).ok()?;
            let last = url.path_segments()?.next_back()?.to_string();
            let filename = percent_decode_str(&last).decode_utf8().ok()?.into_owned();
            if !filename.ends_with(".whl") {
                return None;
            }
            let sha256 = fragment
                .as_deref()
                .and_then(|f| f.strip_prefix("sha256="))
                .map(str::to_ascii_lowercase);
            Some(WheelRef {
                url,
                filename,
                sha256,
            })
        })
        .collect()
}

/// `torch-2.9.1+cu128-cp311-cp311-win_amd64.whl` for dist `torch`, version `2.9.1+cu128`.
pub fn wheel_matches(
    filename: &str,
    dist: &str,
    version: &str,
    py_tag: &str,
    os: &str,
    arch: &str,
) -> bool {
    let Some(stem) = filename.strip_suffix(".whl") else {
        return false;
    };
    let parts: Vec<&str> = stem.split('-').collect();
    if parts.len() != 5 && parts.len() != 6 {
        return false;
    }
    let n = parts.len();
    let (name, ver, py, abi, plat) = (parts[0], parts[1], parts[n - 3], parts[n - 2], parts[n - 1]);
    if !name.eq_ignore_ascii_case(&dist.replace('-', "_")) || ver != version {
        return false;
    }
    if !py.split('.').any(|t| t == py_tag) {
        return false;
    }
    if !abi
        .split('.')
        .any(|t| t == py_tag || t == "none" || t == "abi3")
    {
        return false;
    }
    plat.split('.').any(|p| platform_matches(p, os, arch))
}

fn platform_matches(tag: &str, os: &str, arch: &str) -> bool {
    match (os, arch) {
        ("windows", "x86_64") => tag == "win_amd64",
        ("windows", "aarch64") => tag == "win_arm64",
        ("macos", "aarch64") => {
            tag.starts_with("macosx_") && (tag.ends_with("_arm64") || tag.ends_with("_universal2"))
        }
        ("macos", "x86_64") => {
            tag.starts_with("macosx_") && (tag.ends_with("_x86_64") || tag.ends_with("_universal2"))
        }
        ("linux", arch) => {
            (tag.starts_with("manylinux") || tag.starts_with("linux_"))
                && tag.ends_with(&format!("_{arch}"))
        }
        _ => false,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DownloadProgress {
    pub done: u64,
    pub total: Option<u64>,
    pub speed: Option<f64>,
    pub verifying: bool,
}

/// Downloads `wheel` into `dir` (resuming a previous `.part`), verifies it and returns the
/// final path. `on_progress` is called at most every ~200 ms.
pub async fn download(
    client: &reqwest::Client,
    wheel: &WheelRef,
    dir: &Path,
    cancel: &CancellationToken,
    mut on_progress: impl FnMut(DownloadProgress),
) -> StageResult<PathBuf> {
    tokio::fs::create_dir_all(dir)
        .await
        .map_err(|e| ShellError::io("Cannot create the download folder", &e))?;
    let final_path = dir.join(&wheel.filename);
    if final_path.is_file() {
        return Ok(final_path);
    }
    let part = dir.join(format!("{}.part", wheel.filename));
    let mut meter = SpeedMeter::default();
    let mut attempt = 0;
    let mut checksum_failures = 0;
    loop {
        match download_once(client, wheel, &part, cancel, &mut meter, &mut on_progress).await {
            Ok(total) => {
                on_progress(DownloadProgress {
                    done: total,
                    total: Some(total),
                    speed: None,
                    verifying: true,
                });
                match verify(&part, wheel.sha256.as_deref(), cancel).await? {
                    true => break,
                    false => {
                        let _ = tokio::fs::remove_file(&part).await;
                        checksum_failures += 1;
                        if checksum_failures >= 2 {
                            return Err(StageError::Failed(
                                ShellError::new(
                                    "bootstrap.checksum",
                                    "The downloaded PyTorch package is damaged",
                                )
                                .with_detail(format!("sha256 mismatch for {}", wheel.url)),
                            ));
                        }
                    }
                }
            }
            Err(Attempt::Cancelled) => return Err(StageError::Cancelled),
            Err(Attempt::Fatal(err)) => return Err(StageError::Failed(err)),
            Err(Attempt::Retry(err)) => {
                attempt += 1;
                if attempt >= MAX_ATTEMPTS {
                    return Err(StageError::Failed(err));
                }
                log::warn!("download attempt {attempt} failed: {}", err.detail);
                let backoff = Duration::from_secs(2u64.pow(attempt.min(4)));
                tokio::select! {
                    _ = cancel.cancelled() => return Err(StageError::Cancelled),
                    _ = tokio::time::sleep(backoff) => {}
                }
            }
        }
    }
    tokio::fs::rename(&part, &final_path)
        .await
        .map_err(|e| ShellError::io("Cannot store the downloaded package", &e))?;
    Ok(final_path)
}

enum Attempt {
    Cancelled,
    Retry(ShellError),
    Fatal(ShellError),
}

async fn download_once(
    client: &reqwest::Client,
    wheel: &WheelRef,
    part: &Path,
    cancel: &CancellationToken,
    meter: &mut SpeedMeter,
    on_progress: &mut impl FnMut(DownloadProgress),
) -> Result<u64, Attempt> {
    let have = tokio::fs::metadata(part)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    let mut request = client.get(wheel.url.clone());
    if have > 0 {
        request = request.header(RANGE, format!("bytes={have}-"));
    }
    let mut response = tokio::select! {
        _ = cancel.cancelled() => return Err(Attempt::Cancelled),
        r = request.send() => r.map_err(|e| Attempt::Retry(network_error(wheel.url.as_str(), &e)))?,
    };
    let status = response.status();
    let (append, total) = match status {
        StatusCode::PARTIAL_CONTENT => {
            let total = response
                .headers()
                .get(CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.rsplit('/').next())
                .and_then(|v| v.parse::<u64>().ok())
                .or_else(|| response.content_length().map(|l| l + have));
            (true, total)
        }
        s if s.is_success() => (false, response.content_length()),
        StatusCode::RANGE_NOT_SATISFIABLE => {
            let _ = tokio::fs::remove_file(part).await;
            return Err(Attempt::Retry(http_error(wheel.url.as_str(), status)));
        }
        s if s.is_server_error() || s == StatusCode::TOO_MANY_REQUESTS => {
            return Err(Attempt::Retry(http_error(wheel.url.as_str(), s)));
        }
        s => return Err(Attempt::Fatal(http_error(wheel.url.as_str(), s))),
    };

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(part)
        .await
        .map_err(|e| Attempt::Fatal(ShellError::io("Cannot write the download", &e)))?;
    let mut done = if append { have } else { 0 };
    let mut last_report = Instant::now() - Duration::from_secs(1);
    loop {
        let chunk = tokio::select! {
            _ = cancel.cancelled() => {
                let _ = file.flush().await;
                return Err(Attempt::Cancelled);
            }
            c = tokio::time::timeout(CHUNK_TIMEOUT, response.chunk()) => c,
        };
        let bytes = match chunk {
            Ok(Ok(Some(bytes))) => bytes,
            Ok(Ok(None)) => break,
            Ok(Err(e)) => {
                let _ = file.flush().await;
                return Err(Attempt::Retry(network_error(wheel.url.as_str(), &e)));
            }
            Err(_) => {
                let _ = file.flush().await;
                return Err(Attempt::Retry(
                    ShellError::new("bootstrap.network", "The download stalled").with_detail(
                        format!(
                            "no data for {} s from {}",
                            CHUNK_TIMEOUT.as_secs(),
                            wheel.url
                        ),
                    ),
                ));
            }
        };
        file.write_all(&bytes)
            .await
            .map_err(|e| Attempt::Fatal(ShellError::io("Cannot write the download", &e)))?;
        done += bytes.len() as u64;
        if last_report.elapsed() >= Duration::from_millis(200) {
            last_report = Instant::now();
            on_progress(DownloadProgress {
                done,
                total,
                speed: meter.sample(done),
                verifying: false,
            });
        }
    }
    file.flush()
        .await
        .map_err(|e| Attempt::Fatal(ShellError::io("Cannot write the download", &e)))?;
    drop(file);
    match total {
        Some(total) if done < total => Err(Attempt::Retry(
            ShellError::new("bootstrap.network", "The download was interrupted")
                .with_detail(format!("{done} of {total} bytes from {}", wheel.url)),
        )),
        _ => Ok(done),
    }
}

/// `Ok(true)` when the file matches `sha256` (or no hash is known).
async fn verify(
    path: &Path,
    sha256: Option<&str>,
    cancel: &CancellationToken,
) -> StageResult<bool> {
    let Some(expected) = sha256.map(str::to_owned) else {
        return Ok(true);
    };
    let path = path.to_path_buf();
    let token = cancel.clone();
    let digest = tokio::task::spawn_blocking(move || -> std::io::Result<Option<String>> {
        use std::io::Read;
        let mut file = std::fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            if token.is_cancelled() {
                return Ok(None);
            }
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(Some(hex(&hasher.finalize())))
    })
    .await
    .map_err(ShellError::internal)?
    .map_err(|e| ShellError::io("Cannot read the download", &e))?;
    match digest {
        None => Err(StageError::Cancelled),
        Some(actual) => Ok(actual == expected),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn network_error(url: &str, err: &reqwest::Error) -> ShellError {
    let mut source: Option<&dyn std::error::Error> = Some(err);
    let mut chain = Vec::new();
    while let Some(e) = source {
        chain.push(e.to_string());
        source = e.source();
    }
    ShellError::new("bootstrap.network", "Network error while downloading")
        .with_detail(format!("{url}: {}", chain.join(": ")))
}

fn http_error(url: &str, status: StatusCode) -> ShellError {
    ShellError::new("bootstrap.network", format!("The server answered {status}"))
        .with_detail(url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_platform_wheels() {
        let ok = |f: &str, os: &str, arch: &str| {
            wheel_matches(f, "torch", "2.9.1+cu128", "cp311", os, arch)
        };
        assert!(ok(
            "torch-2.9.1+cu128-cp311-cp311-win_amd64.whl",
            "windows",
            "x86_64"
        ));
        assert!(!ok(
            "torch-2.9.1+cu128-cp312-cp312-win_amd64.whl",
            "windows",
            "x86_64"
        ));
        assert!(!ok(
            "torch-2.9.1+cu126-cp311-cp311-win_amd64.whl",
            "windows",
            "x86_64"
        ));
        assert!(ok(
            "torch-2.9.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl",
            "linux",
            "x86_64"
        ));
        assert!(!ok(
            "torch-2.9.1+cu128-cp311-cp311-manylinux_2_28_aarch64.whl",
            "linux",
            "x86_64"
        ));
        let mac = |f: &str| wheel_matches(f, "torch", "2.9.1", "cp311", "macos", "aarch64");
        assert!(mac("torch-2.9.1-cp311-none-macosx_11_0_arm64.whl"));
        assert!(!mac("torch-2.9.1-cp311-none-macosx_11_0_x86_64.whl"));
    }

    #[test]
    fn parses_index_and_flat_listings() {
        let base = Url::parse("https://download.pytorch.org/whl/cu128/torch/").unwrap();
        let html = r#"<html><body>
            <a href="/whl/cu128/torch-2.9.1%2Bcu128-cp311-cp311-win_amd64.whl#sha256=ABCDEF" data-requires-python="&gt;=3.10">torch-2.9.1+cu128-cp311-cp311-win_amd64.whl</a><br/>
            <a href="../">..</a>
        </body></html>"#;
        let wheels = parse_listing(html, &base);
        assert_eq!(wheels.len(), 1);
        assert_eq!(
            wheels[0].filename,
            "torch-2.9.1+cu128-cp311-cp311-win_amd64.whl"
        );
        assert_eq!(wheels[0].sha256.as_deref(), Some("abcdef"));
        assert_eq!(
            wheels[0].url.as_str(),
            "https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp311-cp311-win_amd64.whl"
        );

        let flat = Url::parse("https://mirrors.aliyun.com/pytorch-wheels/cu128/").unwrap();
        let wheels = parse_listing(
            r#"<a href="torch-2.9.1+cu128-cp311-cp311-win_amd64.whl">torch</a>"#,
            &flat,
        );
        assert_eq!(
            wheels[0].url.as_str(),
            "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.9.1+cu128-cp311-cp311-win_amd64.whl"
        );
    }

    #[test]
    fn listing_urls() {
        assert_eq!(
            listing_url(
                SourceKind::Index,
                "https://download.pytorch.org/whl/cu128",
                "torch"
            ),
            "https://download.pytorch.org/whl/cu128/torch/"
        );
        assert_eq!(
            listing_url(
                SourceKind::FindLinks,
                "https://mirrors.aliyun.com/pytorch-wheels/cu128/",
                "torch"
            ),
            "https://mirrors.aliyun.com/pytorch-wheels/cu128/"
        );
        assert_eq!(python_tag("3.11"), "cp311");
    }
}

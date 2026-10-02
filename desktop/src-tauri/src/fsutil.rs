//! File-system helpers: atomic JSON writes, directory sizes, free disk space, rotating logs.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;

/// Writes `value` as pretty JSON to `path` atomically (temp file + rename).
pub fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(io::Error::other)?;
    write_atomic(path, &bytes)
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".into());
    let tmp = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    {
        let mut file = File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    // std::fs::rename replaces the destination on every platform (MoveFileExW on Windows).
    fs::rename(&tmp, path).inspect_err(|_| {
        let _ = fs::remove_file(&tmp);
    })
}

/// Total size of all regular files below `dir` (errors are ignored: this is for progress only).
pub fn dir_size(dir: &Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(entry.path());
            } else if file_type.is_file() {
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}

/// Removes a directory tree, retrying a few times (Windows virus scanners and the search
/// indexer briefly hold files open right after they are written).
pub fn remove_dir_all_retry(dir: &Path) -> io::Result<()> {
    let mut last_err = None;
    for attempt in 0..5 {
        match fs::remove_dir_all(dir) {
            Ok(()) => return Ok(()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(err) => {
                last_err = Some(err);
                std::thread::sleep(std::time::Duration::from_millis(300 * (attempt + 1)));
            }
        }
    }
    Err(last_err.unwrap_or_else(|| io::Error::other("remove failed")))
}

/// Closest existing ancestor of `path` (the path itself if it exists).
pub fn existing_ancestor(path: &Path) -> Option<PathBuf> {
    let mut current = Some(path);
    while let Some(p) = current {
        if p.exists() {
            return Some(p.to_path_buf());
        }
        current = p.parent();
    }
    None
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct DiskSpace {
    pub free: u64,
    pub total: u64,
}

/// Free (available to the current user) and total bytes of the volume holding `path`.
pub fn disk_space(path: &Path) -> io::Result<DiskSpace> {
    let existing = existing_ancestor(path)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no existing ancestor"))?;
    platform_disk_space(&existing)
}

#[cfg(unix)]
fn platform_disk_space(path: &Path) -> io::Result<DiskSpace> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes()).map_err(io::Error::other)?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: c_path is a valid NUL-terminated string and stat points to writable memory.
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    #[allow(clippy::unnecessary_cast)] // field widths differ between Linux and macOS
    let block = stat.f_frsize as u64;
    #[allow(clippy::unnecessary_cast)]
    Ok(DiskSpace {
        free: stat.f_bavail as u64 * block,
        total: stat.f_blocks as u64 * block,
    })
}

#[cfg(windows)]
fn platform_disk_space(path: &Path) -> io::Result<DiskSpace> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut available = 0u64;
    let mut total = 0u64;
    let mut free = 0u64;
    // SAFETY: wide is NUL-terminated; the out pointers are valid for writes.
    let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut available, &mut total, &mut free) };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(DiskSpace {
        free: available,
        total,
    })
}

/// Append-only log file that rotates to `<name>.1`, `<name>.2` once it exceeds `max_bytes`.
pub struct RotatingFile {
    path: PathBuf,
    max_bytes: u64,
    keep: u32,
    file: Option<File>,
    written: u64,
}

impl RotatingFile {
    pub fn new(path: PathBuf, max_bytes: u64, keep: u32) -> Self {
        Self {
            path,
            max_bytes,
            keep,
            file: None,
            written: 0,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn write_line(&mut self, line: &str) {
        if self.file.is_none() {
            self.open();
        }
        if self.written >= self.max_bytes {
            self.rotate();
        }
        if let Some(file) = self.file.as_mut() {
            if writeln!(file, "{line}").is_ok() {
                self.written += line.len() as u64 + 1;
            }
        }
    }

    fn open(&mut self) {
        if let Some(parent) = self.path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            Ok(file) => {
                self.written = file.metadata().map(|m| m.len()).unwrap_or(0);
                self.file = Some(file);
            }
            Err(err) => log::warn!("cannot open log {}: {err}", self.path.display()),
        }
    }

    fn rotate(&mut self) {
        self.file = None;
        for index in (1..self.keep).rev() {
            let from = numbered(&self.path, index);
            let to = numbered(&self.path, index + 1);
            let _ = fs::rename(from, to);
        }
        if self.keep > 0 {
            let _ = fs::rename(&self.path, numbered(&self.path, 1));
        } else {
            let _ = fs::remove_file(&self.path);
        }
        self.open();
    }
}

fn numbered(path: &Path, index: u32) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(format!(".{index}"));
    PathBuf::from(name)
}

/// Last `n` lines of a text file (reads at most the final 1 MiB).
pub fn tail_lines(path: &Path, n: usize) -> Vec<String> {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut file) = File::open(path) else {
        return Vec::new();
    };
    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let start = len.saturating_sub(1024 * 1024);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return Vec::new();
    }
    let mut buf = Vec::new();
    if file.read_to_end(&mut buf).is_err() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&buf);
    let lines: Vec<&str> = text.lines().collect();
    let skip = lines.len().saturating_sub(n);
    lines[skip..].iter().map(|l| (*l).to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotating_file_rotates() {
        let dir = std::env::temp_dir().join(format!("stellaria-rot-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut log = RotatingFile::new(dir.join("x.log"), 64, 2);
        for i in 0..40 {
            log.write_line(&format!("line {i:03}"));
        }
        assert!(dir.join("x.log").exists());
        assert!(dir.join("x.log.1").exists());
        assert!(dir.join("x.log.2").exists());
        assert!(!dir.join("x.log.3").exists());
        let tail = tail_lines(&dir.join("x.log"), 1);
        assert_eq!(tail, vec!["line 039".to_string()]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn atomic_write_replaces() {
        let dir = std::env::temp_dir().join(format!("stellaria-aw-{}", std::process::id()));
        let path = dir.join("a.json");
        write_json_atomic(&path, &serde_json::json!({"a": 1})).unwrap();
        write_json_atomic(&path, &serde_json::json!({"a": 2})).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains('2'));
        let _ = fs::remove_dir_all(&dir);
    }
}

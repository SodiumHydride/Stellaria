//! The one error shape that reaches the frontend: `{code, message, detail}`.
//!
//! `code` is stable and localized by the frontend, `message` is an English fallback and
//! `detail` carries technical context (paths, process output) for the "details" disclosure.

use serde::Serialize;

#[derive(Debug, Clone, Serialize, thiserror::Error)]
#[error("{code}: {message}")]
pub struct ShellError {
    pub code: String,
    pub message: String,
    pub detail: String,
}

pub type ShellResult<T> = Result<T, ShellError>;

impl ShellError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            detail: String::new(),
        }
    }

    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }

    pub fn internal(err: impl std::fmt::Display) -> Self {
        Self::new("internal", "Unexpected error in the desktop shell").with_detail(err.to_string())
    }

    pub fn io(context: impl std::fmt::Display, err: &std::io::Error) -> Self {
        let code = if is_disk_full(err) { "disk.full" } else { "io" };
        Self::new(code, format!("{context}")).with_detail(err.to_string())
    }
}

impl From<std::io::Error> for ShellError {
    fn from(err: std::io::Error) -> Self {
        Self::io("File system error", &err)
    }
}

impl From<tauri::Error> for ShellError {
    fn from(err: tauri::Error) -> Self {
        Self::new("shell.tauri", "Window system error").with_detail(err.to_string())
    }
}

impl From<anyhow::Error> for ShellError {
    fn from(err: anyhow::Error) -> Self {
        Self::internal(format!("{err:#}"))
    }
}

/// ENOSPC on Unix, ERROR_DISK_FULL / ERROR_HANDLE_DISK_FULL on Windows.
pub fn is_disk_full(err: &std::io::Error) -> bool {
    if err.kind() == std::io::ErrorKind::StorageFull {
        return true;
    }
    match err.raw_os_error() {
        #[cfg(unix)]
        Some(code) => code == libc::ENOSPC,
        #[cfg(windows)]
        Some(code) => code == 112 || code == 39,
        _ => false,
    }
}

//! Child processes without console windows, killable as a whole tree.
//!
//! * Windows: every child gets `CREATE_NO_WINDOW`; long-lived children (uv, the backend) are
//!   put into their own Job Object with `KILL_ON_JOB_CLOSE`, so killing the job kills the
//!   whole tree and nothing outlives a crash of the shell (the OS closes our handles).
//! * Unix: every child starts its own process group; killing the group kills the tree.

use std::ffi::OsStr;
use std::process::Stdio;
use std::sync::OnceLock;
use std::time::Duration;

use regex::Regex;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

#[cfg(windows)]
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// A command that never shows a console window and has stdin closed.
pub fn hidden_command(program: impl AsRef<OsStr>) -> Command {
    let mut cmd = Command::new(program);
    cmd.stdin(Stdio::null());
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    #[cfg(unix)]
    cmd.process_group(0);
    cmd
}

#[derive(Debug)]
pub struct CommandOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

/// Runs a short command to completion, capturing output; kills it after `timeout`.
pub async fn run_capture(mut cmd: Command, timeout: Duration) -> std::io::Result<CommandOutput> {
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let child = cmd.spawn()?;
    let tree = ProcessTree::attach(&child);
    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(output) => {
            let output = output?;
            Ok(CommandOutput {
                success: output.status.success(),
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            })
        }
        Err(_) => {
            tree.kill();
            Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!("timed out after {} s", timeout.as_secs()),
            ))
        }
    }
}

/// Handle to a child's whole process tree.
pub struct ProcessTree {
    #[cfg(windows)]
    job: Option<win::Job>,
    #[cfg(unix)]
    pgid: Option<i32>,
}

impl ProcessTree {
    /// Must be called right after `spawn()`.
    pub fn attach(child: &Child) -> Self {
        #[cfg(windows)]
        {
            let job = child.raw_handle().and_then(|handle| {
                win::Job::for_process(handle)
                    .inspect_err(|err| log::warn!("job object unavailable: {err}"))
                    .ok()
            });
            Self { job }
        }
        #[cfg(unix)]
        {
            Self {
                pgid: child.id().and_then(|id| i32::try_from(id).ok()),
            }
        }
    }

    /// Asks the tree to stop (SIGTERM on Unix; Windows has no equivalent, so this kills).
    pub fn terminate(&self) {
        #[cfg(unix)]
        self.signal(libc::SIGTERM);
        #[cfg(windows)]
        self.kill();
    }

    /// Kills every process of the tree immediately.
    pub fn kill(&self) {
        #[cfg(unix)]
        self.signal(libc::SIGKILL);
        #[cfg(windows)]
        if let Some(job) = &self.job {
            job.terminate();
        }
    }

    #[cfg(unix)]
    fn signal(&self, signal: i32) {
        if let Some(pgid) = self.pgid.filter(|p| *p > 1) {
            // SAFETY: plain syscall; a negative pid addresses the process group.
            unsafe {
                libc::kill(-pgid, signal);
            }
        }
    }
}

#[cfg(windows)]
mod win {
    use std::io;
    use std::os::windows::io::RawHandle;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    /// Owned Job Object handle. Closing it (drop, or the OS on shell exit) kills the members.
    pub struct Job(HANDLE);

    // SAFETY: a job handle is a kernel object handle, usable from any thread.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        pub fn for_process(process: RawHandle) -> io::Result<Self> {
            // SAFETY: FFI calls with valid arguments; every failure is checked.
            unsafe {
                let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if handle.is_null() {
                    return Err(io::Error::last_os_error());
                }
                let job = Job(handle);
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(
                    job.0,
                    JobObjectExtendedLimitInformation,
                    std::ptr::addr_of!(info).cast(),
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                if ok == 0 {
                    return Err(io::Error::last_os_error());
                }
                if AssignProcessToJobObject(job.0, process as HANDLE) == 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(job)
            }
        }

        pub fn terminate(&self) {
            // SAFETY: self.0 is a valid job handle for the lifetime of self.
            unsafe {
                TerminateJobObject(self.0, 1);
            }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            // SAFETY: we own the handle; KILL_ON_JOB_CLOSE ends any remaining member.
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
}

impl Stream {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stdout => "stdout",
            Self::Stderr => "stderr",
        }
    }
}

/// Reads `reader` line by line (lossy UTF-8, `\r` treated as a line break so progress
/// redraws become separate lines, ANSI escapes removed) and sends lines to `tx`.
pub fn spawn_line_reader<R>(
    reader: R,
    stream: Stream,
    tx: mpsc::UnboundedSender<(Stream, String)>,
) -> tokio::task::JoinHandle<()>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut reader = BufReader::with_capacity(64 * 1024, reader);
        let mut buf = Vec::with_capacity(1024);
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let text = String::from_utf8_lossy(&buf);
                    for part in text.split(['\r', '\n']) {
                        let clean = strip_ansi(part);
                        let clean = clean.trim_end();
                        if !clean.is_empty() && tx.send((stream, clean.to_string())).is_err() {
                            return;
                        }
                    }
                }
            }
        }
    })
}

pub fn strip_ansi(text: &str) -> std::borrow::Cow<'_, str> {
    static ANSI: OnceLock<Regex> = OnceLock::new();
    let re = ANSI.get_or_init(|| {
        Regex::new(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07").expect("valid regex")
    });
    re.replace_all(text, "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_ansi() {
        assert_eq!(strip_ansi("\x1b[1;32mResolved\x1b[0m 3"), "Resolved 3");
    }

    #[tokio::test]
    async fn capture_and_tree_kill() {
        let mut cmd = hidden_command(if cfg!(windows) { "cmd" } else { "sh" });
        if cfg!(windows) {
            cmd.args(["/C", "echo hello"]);
        } else {
            cmd.args(["-c", "echo hello"]);
        }
        let out = run_capture(cmd, Duration::from_secs(10)).await.unwrap();
        assert!(out.success);
        assert!(out.stdout.contains("hello"));

        #[cfg(unix)]
        {
            let mut cmd = hidden_command("sh");
            cmd.args(["-c", "sleep 30 & sleep 30"]);
            let mut child = cmd.spawn().unwrap();
            let tree = ProcessTree::attach(&child);
            tokio::time::sleep(Duration::from_millis(100)).await;
            tree.kill();
            let status = tokio::time::timeout(Duration::from_secs(5), child.wait())
                .await
                .expect("killed in time")
                .unwrap();
            assert!(!status.success());
        }
    }
}

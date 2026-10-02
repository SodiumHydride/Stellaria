//! The user's own Python (docs/creation-v2.md §4): `check_python` and the environment such an
//! interpreter runs with.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::process::Command;

use crate::paths::AppPaths;
use crate::procutil::{hidden_command, run_capture, CommandOutput};
use crate::types::Msg;

/// Supported interpreter versions (inclusive), as `(major, minor)`.
pub const MIN_VERSION: (u32, u32) = (3, 10);
pub const MAX_VERSION: (u32, u32) = (3, 12);
/// Importing torch (and initialising CUDA) can take a minute on a cold Windows machine.
const CHECK_TIMEOUT: Duration = Duration::from_secs(180);
const PREFIX: &str = "STELLARIA_PYCHECK ";

/// `PythonCheck` on the wire (same shape as the backend's `/python/check`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PythonCheck {
    pub ok: bool,
    pub executable: String,
    pub version: Option<String>,
    pub missing: Vec<String>,
    pub error: Option<Msg>,
    pub torch: Option<TorchInfo>,
    /// Checked through Stellaria's layer over the interpreter (`crate::layer`): packages it
    /// lacked are installed there, the interpreter's own environment is untouched.
    #[serde(default)]
    pub layered: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TorchInfo {
    pub version: String,
    pub cuda: Option<String>,
    pub cuda_available: bool,
}

/// What the embedded script prints after `STELLARIA_PYCHECK `.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScriptReport {
    executable: String,
    version: String,
    #[serde(default)]
    missing: Vec<String>,
    torch: Option<TorchInfo>,
    torch_error: Option<String>,
    requirements_error: Option<String>,
}

/// Runs the check script with `python`; `requirements` is `requirements-runtime.txt`. `env_of`
/// is the user's interpreter whose environment variables apply (`python` itself, or the one
/// Stellaria's layer sits on).
pub async fn check_python(
    python: &Path,
    env_of: &Path,
    requirements: Option<&Path>,
) -> PythonCheck {
    if !python.is_file() {
        return failure(
            python,
            Msg::new("python.not_found").with("path", python.display().to_string()),
        );
    }
    let mut cmd = hidden_command(python);
    cmd.arg("-c").arg(CHECK_SCRIPT);
    cmd.arg(
        requirements
            .map(|p| p.as_os_str().to_owned())
            .unwrap_or_default(),
    );
    // `-c` puts the working directory on sys.path: run somewhere nothing can shadow imports.
    cmd.current_dir(std::env::temp_dir());
    apply_custom_env(&mut cmd, env_of, None);
    let output = run_capture(cmd, CHECK_TIMEOUT).await;
    interpret(python, output)
}

/// Turns the script's output (or the failure to run it) into a `PythonCheck`.
fn interpret(python: &Path, output: std::io::Result<CommandOutput>) -> PythonCheck {
    let output = match output {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::TimedOut => {
            return failure(
                python,
                Msg::new("python.timeout").with("seconds", CHECK_TIMEOUT.as_secs()),
            )
        }
        Err(err) => {
            return failure(
                python,
                Msg::new("python.failed").with("reason", err.to_string()),
            )
        }
    };
    let report = output
        .stdout
        .lines()
        .find_map(|l| l.trim_end().strip_prefix(PREFIX))
        .and_then(|json| serde_json::from_str::<ScriptReport>(json).ok());
    let Some(report) = report else {
        let reason = output
            .stderr
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("the interpreter produced no report")
            .trim()
            .to_string();
        return failure(python, Msg::new("python.failed").with("reason", reason));
    };

    let error = if !version_supported(&report.version) {
        Some(
            Msg::new("python.version_unsupported")
                .with("version", report.version.clone())
                .with("min", format!("{}.{}", MIN_VERSION.0, MIN_VERSION.1))
                .with("max", format!("{}.{}", MAX_VERSION.0, MAX_VERSION.1)),
        )
    } else if report.torch.is_none() {
        Some(
            Msg::new("python.torch_missing").with(
                "reason",
                report
                    .torch_error
                    .clone()
                    .unwrap_or_else(|| "torch is not installed".into()),
            ),
        )
    } else if !report.missing.is_empty() {
        Some(
            Msg::new("python.packages_missing")
                .with("packages", report.missing.join(", "))
                .with("count", report.missing.len()),
        )
    } else {
        report
            .requirements_error
            .as_ref()
            .map(|reason| Msg::new("python.check_failed").with("reason", reason.clone()))
    };
    PythonCheck {
        ok: error.is_none(),
        executable: report.executable,
        version: Some(report.version),
        missing: report.missing,
        error,
        torch: report.torch,
        layered: false,
    }
}

fn failure(python: &Path, error: Msg) -> PythonCheck {
    PythonCheck {
        ok: false,
        executable: python.display().to_string(),
        version: None,
        missing: Vec::new(),
        error: Some(error),
        torch: None,
        layered: false,
    }
}

/// English summary of a failed check (the frontend localizes `error.code` itself).
pub fn describe(check: &PythonCheck) -> String {
    let param = |key: &str| {
        check
            .error
            .as_ref()
            .and_then(|e| e.params.as_ref())
            .and_then(|p| p.get(key))
            .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string))
            .unwrap_or_default()
    };
    match check.error.as_ref().map(|e| e.code.as_str()) {
        None => "The Python environment is usable".into(),
        Some("python.not_found") => "The Python executable was not found".into(),
        Some("python.timeout") => "Checking the Python environment timed out".into(),
        Some("python.version_unsupported") => format!(
            "Python {} is not supported (use {}–{})",
            param("version"),
            param("min"),
            param("max")
        ),
        Some("python.torch_missing") => "PyTorch cannot be imported in this Python".into(),
        Some("python.packages_missing") => {
            format!("Packages are missing: {}", param("packages"))
        }
        Some(_) => "The Python executable could not be run".into(),
    }
}

/// `runtime.custom_invalid`: the interpreter cannot run the app (detail: the check's error).
pub fn invalid_error(check: &PythonCheck) -> crate::error::ShellError {
    let detail = check
        .error
        .as_ref()
        .and_then(|e| serde_json::to_string(e).ok())
        .unwrap_or_default();
    crate::error::ShellError::new("runtime.custom_invalid", describe(check)).with_detail(detail)
}

pub fn version_supported(version: &str) -> bool {
    let mut parts = version.split('.').map(|p| p.parse::<u32>().ok());
    match (parts.next().flatten(), parts.next().flatten()) {
        (Some(major), Some(minor)) => (MIN_VERSION..=MAX_VERSION).contains(&(major, minor)),
        _ => false,
    }
}

/// Environment for the user's interpreter. Unlike the managed runtime, user site-packages
/// stay enabled (it is their environment); inherited `PYTHONHOME`/`PYTHONPATH` are dropped and
/// conda environments get the `PATH` entries their activation would add (their DLLs live in
/// `Library\bin` on Windows). Without `paths` (checks), no bytecode is written.
pub fn apply_custom_env(cmd: &mut Command, python: &Path, paths: Option<&AppPaths>) {
    for var in [
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONEXECUTABLE",
        "PYTHONINSPECT",
        "PYTHONNOUSERSITE",
    ] {
        cmd.env_remove(var);
    }
    cmd.env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1");
    match paths {
        Some(paths) => {
            cmd.env("PYTHONPYCACHEPREFIX", &paths.pycache_dir);
        }
        // checks only read the environment: not even bytecode caches are written into it
        None => {
            cmd.env("PYTHONDONTWRITEBYTECODE", "1");
        }
    }
    if let Some(root) = conda_root(python) {
        let mut entries = conda_path_entries(&root);
        if let Some(path) = std::env::var_os("PATH") {
            entries.extend(std::env::split_paths(&path));
        }
        if let Ok(joined) = std::env::join_paths(entries) {
            cmd.env("PATH", joined);
        }
        cmd.env("CONDA_PREFIX", &root);
    }
}

/// The environment root if `python` belongs to a conda environment.
fn conda_root(python: &Path) -> Option<PathBuf> {
    let dir = python.parent()?;
    [Some(dir), dir.parent()]
        .into_iter()
        .flatten()
        .find(|d| d.join("conda-meta").is_dir())
        .map(Path::to_path_buf)
}

fn conda_path_entries(root: &Path) -> Vec<PathBuf> {
    if cfg!(windows) {
        let library = root.join("Library");
        vec![
            root.to_path_buf(),
            library.join("mingw-w64").join("bin"),
            library.join("usr").join("bin"),
            library.join("bin"),
            root.join("Scripts"),
            root.join("bin"),
        ]
    } else {
        vec![root.join("bin")]
    }
}

/// Lists missing requirements by distribution name (case and `-`/`_`/`.` normalised, markers
/// evaluated), reports the version and what torch says about CUDA. Runs on Python ≥ 3.6 so
/// that old interpreters still get a clear "version unsupported".
const CHECK_SCRIPT: &str = r##"
import json, os, re, sys
out = {"executable": sys.executable, "version": "%d.%d.%d" % tuple(sys.version_info[:3]),
       "missing": [], "torch": None, "torchError": None, "requirementsError": None}

def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()

installed = set()
try:
    from importlib import metadata as md
    for dist in md.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:
            name = None
        if name:
            installed.add(norm(name))
except Exception as exc:
    out["requirementsError"] = "importlib.metadata: %s" % exc

def marker_applies(marker):
    marker = marker.strip()
    if not marker:
        return True
    for module in ("packaging.markers", "pip._vendor.packaging.markers"):
        try:
            markers = __import__(module, fromlist=["Marker"])
        except Exception:
            continue
        try:
            return bool(markers.Marker(marker).evaluate())
        except Exception:
            return True
    return True

def requirements(path, seen):
    path = os.path.abspath(path)
    if path in seen:
        return []
    seen.add(path)
    names = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split(" #", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r ") or line.startswith("--requirement "):
                nested = line.split(None, 1)[1].strip()
                names += requirements(os.path.join(os.path.dirname(path), nested), seen)
                continue
            if line.startswith("-"):
                continue
            spec, _, marker = line.partition(";")
            match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
            if match and marker_applies(marker):
                names.append(match.group(1))
    return names

if len(sys.argv) > 1 and sys.argv[1] and installed:
    try:
        for name in requirements(sys.argv[1], set()):
            if norm(name) not in installed and name not in out["missing"]:
                out["missing"].append(name)
    except Exception as exc:
        out["requirementsError"] = "%s: %s" % (type(exc).__name__, exc)

try:
    import torch
    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False
    out["torch"] = {"version": str(torch.__version__), "cuda": getattr(torch.version, "cuda", None),
                    "cudaAvailable": available}
except BaseException as exc:
    out["torchError"] = "%s: %s" % (type(exc).__name__, exc)

sys.stdout.write("STELLARIA_PYCHECK " + json.dumps(out) + "\n")
sys.stdout.flush()
"##;

#[cfg(test)]
mod tests {
    use super::*;

    fn output(stdout: &str, stderr: &str, success: bool) -> std::io::Result<CommandOutput> {
        Ok(CommandOutput {
            success,
            stdout: stdout.into(),
            stderr: stderr.into(),
        })
    }

    fn code(check: &PythonCheck) -> Option<&str> {
        check.error.as_ref().map(|e| e.code.as_str())
    }

    const PY: &str = "/opt/py/bin/python";

    #[test]
    fn parses_a_good_report() {
        let stdout = "noise printed by sitecustomize\n\
            STELLARIA_PYCHECK {\"executable\": \"/opt/py/bin/python3.11\", \"version\": \"3.11.9\", \"missing\": [], \
            \"torch\": {\"version\": \"2.9.1+cu128\", \"cuda\": \"12.8\", \"cudaAvailable\": true}, \
            \"torchError\": null, \"requirementsError\": null}\n";
        let check = interpret(Path::new(PY), output(stdout, "", true));
        assert!(check.ok, "{check:?}");
        assert_eq!(check.executable, "/opt/py/bin/python3.11");
        assert_eq!(check.version.as_deref(), Some("3.11.9"));
        let torch = check.torch.unwrap();
        assert_eq!(torch.cuda.as_deref(), Some("12.8"));
        assert!(torch.cuda_available);
    }

    #[test]
    fn reports_problems_in_priority_order() {
        let report = |version: &str, missing: &str, torch: &str| {
            format!(
                "STELLARIA_PYCHECK {{\"executable\": \"x\", \"version\": \"{version}\", \"missing\": [{missing}], \
                 \"torch\": {torch}, \"torchError\": \"ModuleNotFoundError: No module named 'torch'\", \"requirementsError\": null}}"
            )
        };
        let torch = "{\"version\": \"2.9.1\", \"cuda\": null, \"cudaAvailable\": false}";

        let old = interpret(
            Path::new(PY),
            output(&report("3.9.18", "\"librosa\"", "null"), "", true),
        );
        assert_eq!(code(&old), Some("python.version_unsupported"));
        assert!(!old.ok);
        let new = interpret(
            Path::new(PY),
            output(&report("3.13.0", "", torch), "", true),
        );
        assert_eq!(code(&new), Some("python.version_unsupported"));

        let no_torch = interpret(
            Path::new(PY),
            output(&report("3.12.1", "\"librosa\"", "null"), "", true),
        );
        assert_eq!(code(&no_torch), Some("python.torch_missing"));
        assert_eq!(no_torch.missing, vec!["librosa".to_string()]);

        let missing = interpret(
            Path::new(PY),
            output(
                &report("3.10.0", "\"librosa\", \"fastapi\"", torch),
                "",
                true,
            ),
        );
        assert_eq!(code(&missing), Some("python.packages_missing"));
        let params = missing.error.unwrap().params.unwrap();
        assert_eq!(params["packages"], "librosa, fastapi");
        assert_eq!(params["count"], 2);
    }

    #[test]
    fn interpreter_failures() {
        let crashed = interpret(
            Path::new(PY),
            output(
                "",
                "  File \"<string>\", line 1\nSyntaxError: invalid syntax\n",
                false,
            ),
        );
        assert_eq!(code(&crashed), Some("python.failed"));
        assert_eq!(
            crashed.error.unwrap().params.unwrap()["reason"],
            "SyntaxError: invalid syntax"
        );
        assert_eq!(crashed.executable, PY);

        let garbage = interpret(
            Path::new(PY),
            output("STELLARIA_PYCHECK {not json", "", true),
        );
        assert_eq!(code(&garbage), Some("python.failed"));

        let timeout = interpret(
            Path::new(PY),
            Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "timed out",
            )),
        );
        assert_eq!(code(&timeout), Some("python.timeout"));
    }

    #[test]
    fn invalid_error_carries_the_check_code() {
        let check = failure(
            Path::new(PY),
            Msg::new("python.version_unsupported")
                .with("version", "3.9.1")
                .with("min", "3.10")
                .with("max", "3.12"),
        );
        let err = invalid_error(&check);
        assert_eq!(err.code, "runtime.custom_invalid");
        assert_eq!(err.message, "Python 3.9.1 is not supported (use 3.10–3.12)");
        assert!(err.detail.contains("python.version_unsupported"));
    }

    #[test]
    fn version_range() {
        assert!(version_supported("3.10.0"));
        assert!(version_supported("3.12.8"));
        assert!(!version_supported("3.9.18"));
        assert!(!version_supported("3.13.1"));
        assert!(!version_supported("2.7.18"));
        assert!(!version_supported("garbage"));
    }

    #[test]
    fn check_shape_on_the_wire() {
        let check = failure(Path::new(PY), Msg::new("python.not_found").with("path", PY));
        let v = serde_json::to_value(check).unwrap();
        assert_eq!(v["ok"], false);
        assert_eq!(v["error"]["code"], "python.not_found");
        assert_eq!(v["error"]["params"]["path"], PY);
        assert!(v["torch"].is_null());
        assert!(v["version"].is_null());
        assert_eq!(v["missing"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn missing_interpreter() {
        let python = Path::new("/definitely/not/python");
        let check = check_python(python, python, None).await;
        assert_eq!(code(&check), Some("python.not_found"));
    }

    /// A real interpreter: the shared e2e venv (torch on CPU + backend requirements) when it
    /// exists, else nothing to do. Never modifies it.
    #[tokio::test]
    async fn real_interpreter_check() {
        let python = Path::new(
            "/tmp/claude-0/-home-user-DDSP-SVC/bcda9d01-7011-5758-9dea-433afeb0195a/scratchpad/e2e/venv/bin/python",
        );
        if !python.exists() {
            eprintln!("skipped: {} not present", python.display());
            return;
        }
        let requirements =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../backend/requirements-runtime.txt");
        let check = check_python(python, python, Some(&requirements)).await;
        println!("{}", serde_json::to_string_pretty(&check).unwrap());
        let version = check.version.as_deref().expect("version");
        assert!(version_supported(version), "{version}");
        let torch = check.torch.as_ref().expect("torch imports");
        assert!(torch.version.starts_with("2."), "{torch:?}");
        assert!(
            check.missing.is_empty() || code(&check) == Some("python.packages_missing"),
            "{check:?}"
        );

        // A requirement that is certainly absent shows up, by distribution name.
        let dir = std::env::temp_dir().join(format!("stellaria-pycheck-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let req = dir.join("req.txt");
        std::fs::write(
            &req,
            "# comment\nNumPy>=1.0\nscikit_learn\nstellaria-not-a-real-package==1.0 ; python_version >= '3'\n\
             only-on-old-pythons ; python_version < '3'\n--index-url https://example.invalid\n",
        )
        .unwrap();
        let check = check_python(python, python, Some(&req)).await;
        assert_eq!(
            check.missing,
            vec!["stellaria-not-a-real-package".to_string()]
        );
        assert_eq!(code(&check), Some("python.packages_missing"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

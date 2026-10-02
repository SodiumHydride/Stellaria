//! The installed Python runtime: `runtime/state.json` and the status shown in `ShellState`.

use std::path::Path;

use serde::{Deserialize, Serialize};

use std::path::PathBuf;

use crate::fsutil;
use crate::paths::AppPaths;
use crate::pycheck::PythonCheck;
use crate::types::{RuntimeMode, TorchVariant};

/// The interpreter that runs the backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Interpreter {
    /// `STELLARIA_PYTHON` in debug builds.
    Dev(PathBuf),
    /// The user's own Python (`runtime.mode = custom`): `python` runs — Stellaria's layer over
    /// `base`, or `base` itself — with `base`'s environment variables.
    Custom { python: PathBuf, base: PathBuf },
    /// The venv the bootstrapper created.
    Managed(PathBuf),
}

impl Interpreter {
    pub fn path(&self) -> &std::path::Path {
        match self {
            Self::Dev(p) | Self::Managed(p) => p,
            Self::Custom { python, .. } => python,
        }
    }
}

/// What `set_runtime_python` does after starting the backend with the user's interpreter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomOutcome {
    /// The backend is ready: keep the custom interpreter.
    Ready,
    /// It failed and the managed runtime is installed: switch back to it.
    FallBackToManaged,
    /// It failed and there is nothing to fall back to: keep the choice so it can be fixed.
    StayCustom,
}

pub fn custom_outcome(backend_ready: bool, managed_ready: bool) -> CustomOutcome {
    match (backend_ready, managed_ready) {
        (true, _) => CustomOutcome::Ready,
        (false, true) => CustomOutcome::FallBackToManaged,
        (false, false) => CustomOutcome::StayCustom,
    }
}

/// Written by the bootstrapper after a successful `verify` stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeState {
    pub runtime_id: String,
    pub variant: TorchVariant,
    pub python: String,
    pub torch: String,
    pub cuda: Option<String>,
    pub created_at: String,
    #[serde(default)]
    pub torchaudio: Option<String>,
    #[serde(default)]
    pub cuda_available: Option<bool>,
}

impl RuntimeState {
    pub fn load(path: &Path) -> Option<Result<Self, String>> {
        match std::fs::read(path) {
            Ok(bytes) => Some(serde_json::from_slice(&bytes).map_err(|e| e.to_string())),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
            Err(err) => Some(Err(err.to_string())),
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        fsutil::write_json_atomic(path, self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeStatus {
    Missing,
    Outdated,
    Ready,
    Installing,
    Broken,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeInfo {
    pub status: RuntimeStatus,
    pub variant: Option<TorchVariant>,
    pub python: Option<String>,
    pub torch: Option<String>,
    pub cuda: Option<String>,
    pub mode: RuntimeMode,
    /// The user's interpreter, if one was configured (also while `mode` is `managed`).
    pub python_path: Option<PathBuf>,
    /// The user's interpreter runs through Stellaria's layer (`crate::layer`).
    pub layered: bool,
}

/// * `installing` while a bootstrap runs,
/// * `missing` without `state.json`,
/// * `broken` if `state.json` is unreadable or the venv interpreter is gone,
/// * `outdated` if `state.json` names another `runtime.id` than the manifest,
/// * `ready` otherwise.
pub fn runtime_info(paths: &AppPaths, manifest_id: Option<&str>, installing: bool) -> RuntimeInfo {
    let state = RuntimeState::load(&paths.state_file);
    let (variant, python, torch, cuda) = match &state {
        Some(Ok(s)) => (
            Some(s.variant),
            Some(s.python.clone()),
            Some(s.torch.clone()),
            s.cuda.clone(),
        ),
        _ => (None, None, None, None),
    };
    let status = if installing {
        RuntimeStatus::Installing
    } else {
        match &state {
            None => RuntimeStatus::Missing,
            Some(Err(_)) => RuntimeStatus::Broken,
            Some(Ok(_)) if !paths.venv_python().is_file() => RuntimeStatus::Broken,
            Some(Ok(s)) if manifest_id.is_some_and(|id| id != s.runtime_id) => {
                RuntimeStatus::Outdated
            }
            Some(Ok(_)) => RuntimeStatus::Ready,
        }
    };
    RuntimeInfo {
        status,
        variant,
        python,
        torch,
        cuda,
        mode: RuntimeMode::Managed,
        python_path: None,
        layered: false,
    }
}

/// The user's own interpreter: `ready` while it exists and its last check (if any) passed;
/// versions come from that check.
pub fn custom_runtime_info(
    python: &std::path::Path,
    check: Option<&PythonCheck>,
    installing: bool,
) -> RuntimeInfo {
    let status = if installing {
        RuntimeStatus::Installing
    } else if !python.is_file() || check.is_some_and(|c| !c.ok) {
        RuntimeStatus::Broken
    } else {
        RuntimeStatus::Ready
    };
    let torch = check.and_then(|c| c.torch.as_ref());
    RuntimeInfo {
        status,
        variant: None,
        python: check.and_then(|c| c.version.clone()),
        torch: torch.map(|t| t.version.clone()),
        cuda: torch.and_then(|t| t.cuda.clone()),
        mode: RuntimeMode::Custom,
        python_path: Some(python.to_path_buf()),
        layered: check.is_some_and(|c| c.layered),
    }
}

/// Environment for every Python process we start from the venv: UTF-8 everywhere, no user
/// site-packages, no inherited `PYTHONHOME`/`PYTHONPATH`, and `.pyc` files outside the
/// (read-only, code-signed) app bundle.
pub fn apply_python_env(cmd: &mut tokio::process::Command, paths: &AppPaths) {
    for var in [
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONEXECUTABLE",
        "PYTHONINSPECT",
    ] {
        cmd.env_remove(var);
    }
    cmd.env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONNOUSERSITE", "1")
        .env("PYTHONPYCACHEPREFIX", &paths.pycache_dir);
}

/// Runtime info for dev mode (`STELLARIA_PYTHON`): always ready, nothing installed by us.
pub fn dev_runtime_info(variant: Option<TorchVariant>) -> RuntimeInfo {
    RuntimeInfo {
        status: RuntimeStatus::Ready,
        variant,
        python: None,
        torch: None,
        cuda: None,
        mode: RuntimeMode::Managed,
        python_path: None,
        layered: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_fallback_rules() {
        assert_eq!(custom_outcome(true, true), CustomOutcome::Ready);
        assert_eq!(custom_outcome(true, false), CustomOutcome::Ready);
        assert_eq!(
            custom_outcome(false, true),
            CustomOutcome::FallBackToManaged
        );
        assert_eq!(custom_outcome(false, false), CustomOutcome::StayCustom);
    }

    #[test]
    fn status_transitions() {
        let root = std::env::temp_dir().join(format!("stellaria-rt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let paths = AppPaths::from_data_dir(root.join("data"), root.join("lib"));

        assert_eq!(
            runtime_info(&paths, Some("a"), false).status,
            RuntimeStatus::Missing
        );
        assert_eq!(
            runtime_info(&paths, Some("a"), true).status,
            RuntimeStatus::Installing
        );

        let state = RuntimeState {
            runtime_id: "a".into(),
            variant: TorchVariant::Cpu,
            python: "3.11.13".into(),
            torch: "2.9.1+cpu".into(),
            cuda: None,
            created_at: "2026-01-01T00:00:00Z".into(),
            torchaudio: None,
            cuda_available: None,
        };
        state.save(&paths.state_file).unwrap();
        assert_eq!(
            runtime_info(&paths, Some("a"), false).status,
            RuntimeStatus::Broken
        );

        let py = paths.venv_python();
        std::fs::create_dir_all(py.parent().unwrap()).unwrap();
        std::fs::write(&py, b"").unwrap();
        let info = runtime_info(&paths, Some("a"), false);
        assert_eq!(info.status, RuntimeStatus::Ready);
        assert_eq!(info.variant, Some(TorchVariant::Cpu));
        assert_eq!(
            runtime_info(&paths, Some("b"), false).status,
            RuntimeStatus::Outdated
        );

        let v = serde_json::to_value(runtime_info(&paths, Some("a"), false)).unwrap();
        assert_eq!(v["mode"], "managed");
        assert!(v["pythonPath"].is_null());

        std::fs::write(&paths.state_file, b"{oops").unwrap();
        assert_eq!(
            runtime_info(&paths, Some("a"), false).status,
            RuntimeStatus::Broken
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}

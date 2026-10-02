//! Where things live: app data (runtime, logs, settings), the default library and the
//! bundled resources (engine, backend, manifest, uv) — or their repository copies in dev.

use std::path::{Path, PathBuf};

use anyhow::Context;
use tauri::{AppHandle, Manager, Runtime};

/// Folder name under the platform's local data dir
/// (`%LOCALAPPDATA%\Stellaria`, `~/Library/Application Support/Stellaria`, `~/.local/share/Stellaria`).
const DATA_DIR_NAME: &str = "Stellaria";

#[derive(Debug, Clone)]
pub struct AppPaths {
    pub data_dir: PathBuf,
    pub runtime_dir: PathBuf,
    /// `UV_PYTHON_INSTALL_DIR`: standalone CPython builds.
    pub python_dir: PathBuf,
    pub venv_dir: PathBuf,
    /// Stellaria's layer over the user's own Python (`layer.rs`).
    pub layer_dir: PathBuf,
    pub uv_cache_dir: PathBuf,
    /// Resumable downloads (the torch wheel) — removed after a successful bootstrap.
    pub downloads_dir: PathBuf,
    /// `PYTHONPYCACHEPREFIX`, keeps `.pyc` files out of the (read-only, signed) app bundle.
    pub pycache_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub state_file: PathBuf,
    pub settings_file: PathBuf,
    pub default_library: PathBuf,
}

impl AppPaths {
    pub fn resolve() -> anyhow::Result<Self> {
        let data_dir = match dev_data_dir_override() {
            Some(dir) => dir,
            None => dirs::data_local_dir()
                .context("cannot determine the local data directory")?
                .join(DATA_DIR_NAME),
        };
        let documents = dirs::document_dir()
            .or_else(dirs::home_dir)
            .context("cannot determine the documents directory")?;
        Ok(Self::from_data_dir(data_dir, documents.join(DATA_DIR_NAME)))
    }

    pub fn from_data_dir(data_dir: PathBuf, default_library: PathBuf) -> Self {
        let runtime_dir = data_dir.join("runtime");
        Self {
            python_dir: runtime_dir.join("python"),
            venv_dir: runtime_dir.join("venv"),
            layer_dir: runtime_dir.join("layer"),
            uv_cache_dir: runtime_dir.join("uv-cache"),
            downloads_dir: runtime_dir.join("downloads"),
            pycache_dir: runtime_dir.join("pycache"),
            state_file: runtime_dir.join("state.json"),
            logs_dir: data_dir.join("logs"),
            settings_file: data_dir.join("shell.json"),
            runtime_dir,
            data_dir,
            default_library,
        }
    }

    pub fn venv_python(&self) -> PathBuf {
        venv_python(&self.venv_dir)
    }

    pub fn layer_python(&self) -> PathBuf {
        venv_python(&self.layer_dir)
    }
}

pub fn venv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

fn dev_data_dir_override() -> Option<PathBuf> {
    if !cfg!(debug_assertions) {
        return None;
    }
    std::env::var_os("STELLARIA_DATA_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Bundled resources (release) or their repository counterparts (debug builds).
#[derive(Debug, Clone)]
pub struct ResourcePaths {
    /// DDSP-SVC sources (contains `train_reflow.py`).
    pub engine: PathBuf,
    /// Directory put on `PYTHONPATH`; contains the `stellaria_backend` package and the
    /// requirements files.
    pub backend: PathBuf,
    pub manifest: PathBuf,
    pub uv: PathBuf,
    /// `STELLARIA_PYTHON` in debug builds: use this interpreter and skip the bootstrap.
    pub dev_python: Option<PathBuf>,
    pub dev: bool,
}

impl ResourcePaths {
    pub fn resolve<R: Runtime>(app: &AppHandle<R>) -> anyhow::Result<Self> {
        if cfg!(debug_assertions) {
            return Ok(Self::dev());
        }
        let root = app
            .path()
            .resource_dir()
            .context("cannot locate the resource directory")?
            .join("resources");
        Ok(Self {
            engine: root.join("engine"),
            backend: root.join("backend"),
            manifest: root.join("manifest.json"),
            uv: root.join("uv").join(exe_name("uv")),
            dev_python: None,
            dev: false,
        })
    }

    /// Repository layout: `src-tauri/../..` is the Stellaria root, with the engine in `engine/`.
    fn dev() -> Self {
        let src_tauri = Path::new(env!("CARGO_MANIFEST_DIR"));
        let stellaria = src_tauri
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .unwrap_or_else(|| src_tauri.to_path_buf());
        let dev_python = std::env::var_os("STELLARIA_PYTHON")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        Self {
            engine: stellaria.join("engine"),
            backend: stellaria.join("backend"),
            manifest: stellaria.join("resources").join("manifest.json"),
            uv: find_in_path("uv").unwrap_or_else(|| PathBuf::from(exe_name("uv"))),
            dev_python,
            dev: true,
        }
    }

    /// `runtime.requirements` in the manifest is relative to the folder that holds `backend/`.
    pub fn requirements(&self, relative: &str) -> PathBuf {
        self.backend
            .parent()
            .map(|p| p.join(relative))
            .unwrap_or_else(|| PathBuf::from(relative))
    }
}

pub fn exe_name(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

/// Minimal `which`: first `name` (with `.exe` on Windows) found on `PATH`.
pub fn find_in_path(name: &str) -> Option<PathBuf> {
    let file = exe_name(name);
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(&file))
        .find(|candidate| candidate.is_file())
}

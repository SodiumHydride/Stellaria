//! The shell bridge of docs/api.md §2. Every error is `{code, message, detail}`.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde::Serialize;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_opener::OpenerExt;

use crate::backend::{BackendState, BackendStatus};
use crate::bootstrap::{Job, Mode, Request};
use crate::error::{ShellError, ShellResult};
use crate::fsutil::{self, DiskSpace};
use crate::hardware::{self, HardwareReport};
use crate::layer;
use crate::pycheck::{self, PythonCheck};
use crate::pyfind::{self, PythonCandidate};
use crate::runtime::{custom_outcome, CustomOutcome, RuntimeInfo, RuntimeState};
use crate::tray::TrayState;
use crate::types::{platform_name, Language, Region, RuntimeMode, TorchVariant};
use crate::{window, Shell};

type ShellRef<'a> = State<'a, Arc<Shell>>;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShellState {
    pub platform: &'static str,
    pub arch: &'static str,
    pub app_version: String,
    pub first_run_done: bool,
    pub library_path: String,
    pub data_path: String,
    pub language: Option<Language>,
    pub terms_accepted: u32,
    pub runtime: RuntimeInfo,
    /// The app's own runtime is installed (the way back when the user's Python breaks).
    pub managed_ready: bool,
    pub backend: BackendStatus,
}

pub fn shell_state(app: &AppHandle, shell: &Shell) -> ShellState {
    let settings = shell.settings.get();
    let runtime = shell.runtime_info();
    ShellState {
        platform: platform_name(),
        arch: std::env::consts::ARCH,
        app_version: app.package_info().version.to_string(),
        first_run_done: settings.first_run_done,
        library_path: shell.library_path().display().to_string(),
        data_path: shell.paths.data_dir.display().to_string(),
        language: settings.language,
        terms_accepted: settings.terms_accepted,
        runtime,
        managed_ready: shell.managed_runtime_ready(),
        backend: shell.backend.status(),
    }
}

#[tauri::command]
pub async fn get_shell_state(app: AppHandle, shell: ShellRef<'_>) -> ShellResult<ShellState> {
    // The frontend is up: from now on it decides what closing the window means.
    shell.frontend.ready.store(true, Ordering::SeqCst);
    Ok(shell_state(&app, &shell))
}

#[tauri::command]
pub async fn detect_hardware() -> ShellResult<HardwareReport> {
    Ok(hardware::detect().await)
}

#[tauri::command]
pub async fn disk_space(path: String) -> ShellResult<DiskSpace> {
    let path = PathBuf::from(path);
    tokio::task::spawn_blocking(move || fsutil::disk_space(&path))
        .await
        .map_err(ShellError::internal)?
        .map_err(|e| ShellError::io("Cannot read the free disk space", &e))
}

#[tauri::command]
pub async fn set_library_path(
    app: AppHandle,
    shell: ShellRef<'_>,
    path: String,
) -> ShellResult<ShellState> {
    let path = PathBuf::from(path.trim());
    if !path.is_absolute() {
        return Err(
            ShellError::new("library.invalid", "Choose a full folder path")
                .with_detail(path.display().to_string()),
        );
    }
    let probe_dir = path.clone();
    tokio::task::spawn_blocking(move || -> ShellResult<()> {
        std::fs::create_dir_all(&probe_dir).map_err(|e| {
            ShellError::new("library.create_failed", "Cannot create the library folder")
                .with_detail(format!("{}: {e}", probe_dir.display()))
        })?;
        let probe = probe_dir.join(".stellaria-write-test");
        std::fs::write(&probe, b"ok").map_err(|e| {
            ShellError::new("library.not_writable", "The library folder is not writable")
                .with_detail(format!("{}: {e}", probe_dir.display()))
        })?;
        let _ = std::fs::remove_file(&probe);
        Ok(())
    })
    .await
    .map_err(ShellError::internal)??;

    shell
        .settings
        .update(|s| s.library_path = Some(path.clone()))?;
    crate::allow_asset_dir(&app, &path);
    if shell.backend.is_running() {
        if let Err(err) = shell.backend.restart(&app).await {
            log::warn!("backend restart after library change failed: {err}");
        }
    }
    Ok(shell_state(&app, &shell))
}

#[tauri::command]
pub async fn set_language(
    app: AppHandle,
    shell: ShellRef<'_>,
    language: Option<Language>,
) -> ShellResult<ShellState> {
    shell.settings.update(|s| s.language = language)?;
    if let Some(tray) = app.try_state::<TrayState>() {
        tray.set_language(shell.language());
    }
    Ok(shell_state(&app, &shell))
}

#[tauri::command]
pub async fn bootstrap_start(
    app: AppHandle,
    shell: ShellRef<'_>,
    variant: TorchVariant,
    region: Region,
) -> ShellResult<()> {
    shell.settings.update(|s| {
        s.variant = Some(variant);
        s.region = region;
    })?;
    shell.bootstrap.start(
        &app,
        Job::Runtime(Request {
            variant,
            region,
            mode: Mode::Install,
        }),
    )
}

#[tauri::command]
pub async fn bootstrap_cancel(shell: ShellRef<'_>) -> ShellResult<()> {
    shell.bootstrap.cancel().await;
    Ok(())
}

#[tauri::command]
pub async fn bootstrap_repair(app: AppHandle, shell: ShellRef<'_>) -> ShellResult<()> {
    let settings = shell.settings.get();
    let installed = RuntimeState::load(&shell.paths.state_file).and_then(Result::ok);
    let variant = match installed.map(|s| s.variant).or(settings.variant) {
        Some(variant) => variant,
        None => hardware::detect().await.recommended_variant,
    };
    shell.bootstrap.start(
        &app,
        Job::Runtime(Request {
            variant,
            region: settings.region,
            mode: Mode::Repair,
        }),
    )
}

// ---------------------------------------------------------------- the user's own Python

#[tauri::command]
pub async fn check_python(shell: ShellRef<'_>, path: String) -> ShellResult<PythonCheck> {
    Ok(shell.check_python(Path::new(path.trim())).await)
}

/// Interpreters someone already has: inside `folder` (a DDSP-SVC checkout or bundle) when
/// given, else conda environments and the usual installs. Nothing is run; check one first.
#[tauri::command]
pub async fn find_pythons(folder: Option<String>) -> ShellResult<Vec<PythonCandidate>> {
    let folder = folder
        .map(|f| PathBuf::from(f.trim()))
        .filter(|p| !p.as_os_str().is_empty());
    Ok(
        tokio::task::spawn_blocking(move || pyfind::find(folder.as_deref()))
            .await
            .unwrap_or_default(),
    )
}

/// Switches the backend to the user's interpreter (`path`) or back to the managed runtime
/// (`null`). A custom interpreter must pass `check_python`; if the backend then does not
/// become ready, the managed runtime takes over again when it is installed.
#[tauri::command]
pub async fn set_runtime_python(
    app: AppHandle,
    shell: ShellRef<'_>,
    path: Option<String>,
) -> ShellResult<ShellState> {
    if shell.bootstrap.is_running() {
        return Err(ShellError::new(
            "bootstrap.running",
            "Wait until the runtime installation has finished",
        ));
    }
    let Some(python) = path
        .map(|p| PathBuf::from(p.trim()))
        .filter(|p| !p.as_os_str().is_empty())
    else {
        shell
            .settings
            .update(|s| s.runtime.mode = RuntimeMode::Managed)?;
        let first_run_done = shell.settings.get().first_run_done;
        if shell.backend.is_running() || (first_run_done && shell.managed_runtime_ready()) {
            if let Err(err) = shell.backend.restart(&app).await {
                log::warn!("backend restart on the managed runtime: {err}");
            }
        }
        return Ok(shell_state(&app, &shell));
    };

    if !python.is_absolute() {
        return Err(ShellError::new(
            "runtime.custom_invalid",
            "Choose the full path of a python executable",
        )
        .with_detail(python.display().to_string()));
    }
    let check = shell.check_python(&python).await;
    if !check.ok {
        return Err(pycheck::invalid_error(&check));
    }
    if !check.layered {
        ensure_layer(&app, &shell, &python).await?;
    }
    shell.settings.update(|s| {
        s.runtime.mode = RuntimeMode::Custom;
        s.runtime.python_path = Some(python.clone());
    })?;
    let started = shell.backend.restart(&app).await;
    let ready = matches!(&started, Ok(status) if status.status == BackendState::Ready);
    match custom_outcome(ready, shell.managed_runtime_ready()) {
        CustomOutcome::Ready => Ok(shell_state(&app, &shell)),
        outcome => {
            let error = custom_failed(&shell, &started);
            if outcome == CustomOutcome::FallBackToManaged {
                shell
                    .settings
                    .update(|s| s.runtime.mode = RuntimeMode::Managed)?;
                if let Err(err) = shell.backend.restart(&app).await {
                    log::warn!("backend on the managed runtime after the fallback: {err}");
                }
            }
            Err(error)
        }
    }
}

/// Puts Stellaria's layer under an interpreter that already has everything, so the add-ons the
/// backend installs later go into the layer instead of the user's environment. Nothing to
/// download: the layer only bridges to their packages. When the interpreter does not work
/// through a layer, the layer is removed and this fails (`runtime.layer_failed`): running their
/// interpreter bare would let installs land in their environment.
pub(crate) async fn ensure_layer(app: &AppHandle, shell: &Shell, python: &Path) -> ShellResult<()> {
    // The layer of another interpreter may be in use (Windows locks its files).
    if shell
        .backend
        .current_python()
        .is_some_and(|current| current.starts_with(&shell.paths.layer_dir))
    {
        shell.backend.stop(app, true).await;
    }
    let uv = shell
        .resources
        .uv
        .is_file()
        .then_some(shell.resources.uv.as_path());
    let built = match layer::query(python).await {
        Ok(info) => layer::create(&shell.paths, uv, python, &info).await,
        Err(err) => Err(err),
    };
    if let Err(err) = built {
        log::warn!("no layer for {}: {err}", python.display());
        layer::remove(&shell.paths).await;
        shell.check_python(python).await;
        return Err(err);
    }
    let layered = shell.check_python(python).await;
    if !layered.ok {
        let reason = pycheck::describe(&layered);
        log::warn!(
            "{} does not work through the layer ({reason})",
            python.display()
        );
        layer::remove(&shell.paths).await;
        shell.check_python(python).await;
        return Err(ShellError::new(
            "runtime.layer_failed",
            "Your Python did not work with the app's layer",
        )
        .with_detail(reason));
    }
    Ok(())
}

/// `runtime.custom_failed`, with whatever the backend said before it gave up.
fn custom_failed(shell: &Shell, started: &ShellResult<BackendStatus>) -> ShellError {
    let detail = match started {
        Err(err) if !err.detail.is_empty() => err.detail.clone(),
        Err(err) => err.message.clone(),
        Ok(_) => shell.backend.status().error.unwrap_or_default(),
    };
    ShellError::new(
        "runtime.custom_failed",
        "The backend did not start with your Python",
    )
    .with_detail(detail)
}

/// Installs `requirements-runtime.txt` (without torch) for `path`: what its environment lacks
/// goes into Stellaria's layer over it; `path`'s own environment is never written to. Progress
/// and cancel as for the bootstrap (`bootstrap://progress`, stage `packages`;
/// `bootstrap_cancel`).
#[tauri::command]
pub async fn install_runtime_requirements(
    app: AppHandle,
    shell: ShellRef<'_>,
    path: String,
) -> ShellResult<()> {
    let python = PathBuf::from(path.trim());
    if !python.is_absolute() || !python.is_file() {
        return Err(ShellError::new(
            "runtime.custom_invalid",
            "The Python executable was not found",
        )
        .with_detail(python.display().to_string()));
    }
    shell.bootstrap.start(
        &app,
        Job::Requirements {
            python,
            region: shell.settings.get().region,
        },
    )
}

#[tauri::command]
pub async fn finish_onboarding(app: AppHandle, shell: ShellRef<'_>) -> ShellResult<ShellState> {
    shell.settings.update(|s| s.first_run_done = true)?;
    crate::autostart_backend(&app);
    Ok(shell_state(&app, &shell))
}

/// Start failures are reported in the returned status (`crashed` + `error`); only unmet
/// preconditions (runtime missing, library unusable) reject.
fn status_or_error(
    shell: &Shell,
    result: ShellResult<BackendStatus>,
) -> ShellResult<BackendStatus> {
    match result {
        Ok(status) => Ok(status),
        Err(err) if err.code.starts_with("backend.") => Ok(shell.backend.status()),
        Err(err) => Err(err),
    }
}

#[tauri::command]
pub async fn backend_start(app: AppHandle, shell: ShellRef<'_>) -> ShellResult<BackendStatus> {
    let result = shell.backend.start(&app, true).await;
    status_or_error(&shell, result)
}

#[tauri::command]
pub async fn backend_restart(app: AppHandle, shell: ShellRef<'_>) -> ShellResult<BackendStatus> {
    let result = shell.backend.restart(&app).await;
    status_or_error(&shell, result)
}

#[tauri::command]
pub async fn backend_stop(app: AppHandle, shell: ShellRef<'_>) -> ShellResult<BackendStatus> {
    Ok(shell.backend.stop(&app, true).await)
}

#[tauri::command]
pub async fn backend_logs(shell: ShellRef<'_>, tail: Option<usize>) -> ShellResult<Vec<String>> {
    Ok(shell.backend.logs(tail.unwrap_or(500).clamp(1, 5000)))
}

/// Records that the user accepted this version of the terms of use (never lowers it).
#[tauri::command]
pub async fn accept_terms(
    app: AppHandle,
    shell: ShellRef<'_>,
    version: u32,
) -> ShellResult<ShellState> {
    shell
        .settings
        .update(|s| s.terms_accepted = s.terms_accepted.max(version))?;
    Ok(shell_state(&app, &shell))
}

#[tauri::command]
pub async fn set_close_to_tray(shell: ShellRef<'_>, enabled: bool) -> ShellResult<()> {
    shell.settings.update(|s| s.close_to_tray = enabled)?;
    Ok(())
}

#[tauri::command]
pub async fn set_tray_status(
    app: AppHandle,
    text: Option<String>,
    progress: Option<f64>,
) -> ShellResult<()> {
    if let Some(tray) = app.try_state::<TrayState>() {
        tray.set_status(&app, text.as_deref(), progress.filter(|p| p.is_finite()));
    }
    Ok(())
}

#[tauri::command]
pub async fn reveal_path(app: AppHandle, path: String) -> ShellResult<()> {
    let path = PathBuf::from(path);
    if !path.exists() {
        return Err(
            ShellError::new("path.not_found", "The file or folder no longer exists")
                .with_detail(path.display().to_string()),
        );
    }
    let result = if path.is_dir() {
        app.opener()
            .open_path(path.display().to_string(), None::<&str>)
    } else {
        app.opener().reveal_item_in_dir(&path)
    };
    result.map_err(|e| {
        ShellError::new("path.open_failed", "Cannot open the file manager")
            .with_detail(e.to_string())
    })
}

#[tauri::command]
pub async fn open_logs_folder(app: AppHandle, shell: ShellRef<'_>) -> ShellResult<()> {
    let logs = shell.paths.logs_dir.clone();
    std::fs::create_dir_all(&logs)?;
    app.opener()
        .open_path(logs.display().to_string(), None::<&str>)
        .map_err(|e| {
            ShellError::new("path.open_failed", "Cannot open the logs folder")
                .with_detail(e.to_string())
        })
}

#[tauri::command]
pub async fn quit_app(app: AppHandle, force: Option<bool>) -> ShellResult<()> {
    window::quit(&app, force.unwrap_or(false)).await;
    Ok(())
}

/// Internal: liveness ping answered by the page (see `window::frontend_alive`).
#[tauri::command]
pub fn frontend_pong(shell: ShellRef<'_>) {
    shell.frontend.pongs.fetch_add(1, Ordering::SeqCst);
}

//! 星籁 Stellaria desktop shell: window, tray, single instance, the first-run Python runtime
//! bootstrap and supervision of the backend. The bridge to the frontend is documented in
//! `stellaria/docs/api.md` §2.

mod backend;
mod bootstrap;
mod commands;
mod error;
mod fsutil;
mod hardware;
mod layer;
mod manifest;
mod paths;
mod procutil;
mod pycheck;
mod pyfind;
mod runtime;
mod settings;
mod tray;
mod types;
mod window;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tauri::{AppHandle, Manager};
use tauri_plugin_log::{RotationStrategy, Target, TargetKind};

use backend::BackendManager;
use bootstrap::BootstrapManager;
use error::{ShellError, ShellResult};
use manifest::Manifest;
use paths::{AppPaths, ResourcePaths};
use pycheck::PythonCheck;
use runtime::{
    custom_runtime_info, dev_runtime_info, runtime_info, Interpreter, RuntimeInfo, RuntimeStatus,
};
use settings::SettingsStore;
use types::Language;

/// Everything the shell knows, shared by commands, events and background tasks.
pub struct Shell {
    pub paths: AppPaths,
    pub resources: ResourcePaths,
    pub settings: SettingsStore,
    pub backend: BackendManager,
    pub bootstrap: BootstrapManager,
    pub frontend: window::FrontendState,
    /// Client for mirrors and downloads (honours system/env proxies).
    pub net_http: reqwest::Client,
    quitting: AtomicBool,
    /// The latest `check_python` result per interpreter.
    python_checks: Mutex<HashMap<PathBuf, PythonCheck>>,
}

impl Shell {
    pub fn is_quitting(&self) -> bool {
        self.quitting.load(Ordering::SeqCst)
    }

    /// Marks the shell as quitting; `false` if it already was.
    pub fn begin_quit(&self) -> bool {
        !self.quitting.swap(true, Ordering::SeqCst)
    }

    pub fn library_path(&self) -> PathBuf {
        self.settings
            .get()
            .library_path
            .unwrap_or_else(|| self.paths.default_library.clone())
    }

    pub fn language(&self) -> Language {
        self.settings
            .get()
            .language
            .unwrap_or_else(Language::system_default)
    }

    fn manifest_runtime_id(&self) -> Option<String> {
        Manifest::load(&self.resources.manifest)
            .ok()
            .map(|m| m.runtime.id)
    }

    /// `requirements-runtime.txt` as named by the manifest.
    pub fn requirements_path(&self) -> Option<PathBuf> {
        Manifest::load(&self.resources.manifest)
            .ok()
            .map(|m| self.resources.requirements(&m.runtime.requirements))
    }

    /// The runtime as `ShellState` reports it: dev interpreter, the user's own, or managed.
    pub fn runtime_info(&self) -> RuntimeInfo {
        let settings = self.settings.get();
        let installing = self.bootstrap.is_running();
        let mut info = if self.resources.dev_python.is_some() {
            dev_runtime_info(settings.variant)
        } else if let Some(python) = settings.runtime.custom_python() {
            let mut info =
                custom_runtime_info(python, self.cached_check(python).as_ref(), installing);
            info.layered = layer::existing(&self.paths, python).is_some();
            info
        } else {
            runtime_info(
                &self.paths,
                self.manifest_runtime_id().as_deref(),
                installing,
            )
        };
        info.python_path = settings.runtime.python_path.clone();
        info
    }

    /// Whether the app-installed runtime is usable (the fallback for a failing custom one).
    pub fn managed_runtime_ready(&self) -> bool {
        runtime_info(&self.paths, self.manifest_runtime_id().as_deref(), false).status
            == RuntimeStatus::Ready
    }

    fn runtime_ready(&self) -> bool {
        self.runtime_info().status == RuntimeStatus::Ready
    }

    /// The interpreter the backend runs with right now.
    pub fn backend_interpreter(&self) -> ShellResult<Interpreter> {
        if let Some(python) = &self.resources.dev_python {
            return Ok(Interpreter::Dev(python.clone()));
        }
        if let Some(python) = self.settings.get().runtime.custom_python() {
            if !python.is_file() {
                return Err(ShellError::new(
                    "runtime.not_ready",
                    "Your Python interpreter was not found",
                )
                .with_detail(python.display().to_string()));
            }
            return Ok(Interpreter::Custom {
                python: layer::existing(&self.paths, python)
                    .unwrap_or_else(|| python.to_path_buf()),
                base: python.to_path_buf(),
            });
        }
        let info = runtime_info(
            &self.paths,
            self.manifest_runtime_id().as_deref(),
            self.bootstrap.is_running(),
        );
        if info.status != RuntimeStatus::Ready {
            return Err(ShellError::new(
                "runtime.not_ready",
                "The Python runtime is not installed or needs repair",
            )
            .with_detail(format!("runtime status: {:?}", info.status)));
        }
        Ok(Interpreter::Managed(self.paths.venv_python()))
    }

    /// Runs `check_python` against the app's requirements — through Stellaria's layer over
    /// `python` when there is one — and remembers the result.
    pub async fn check_python(&self, python: &Path) -> PythonCheck {
        let requirements = self.requirements_path();
        let layer = layer::existing(&self.paths, python);
        let target = layer.as_deref().unwrap_or(python);
        let mut check = pycheck::check_python(target, python, requirements.as_deref()).await;
        check.layered = layer.is_some();
        if layer.is_some() && check.torch.is_none() {
            // A layer that stopped working (the interpreter under it was replaced, say) must not
            // make a working interpreter look broken: judge it as it is. Choosing it again
            // rebuilds the layer.
            let direct = pycheck::check_python(python, python, requirements.as_deref()).await;
            if direct.torch.is_some() {
                check = direct;
            }
        }
        self.python_checks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(python.to_path_buf(), check.clone());
        check
    }

    pub fn cached_check(&self, python: &Path) -> Option<PythonCheck> {
        self.python_checks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(python)
            .cloned()
    }
}

pub(crate) fn shell(app: &AppHandle) -> Arc<Shell> {
    app.state::<Arc<Shell>>().inner().clone()
}

/// Lets `asset:` URLs (convertFileSrc) read files below `dir`.
pub(crate) fn allow_asset_dir(app: &AppHandle, dir: &Path) {
    if let Err(err) = app.asset_protocol_scope().allow_directory(dir, true) {
        log::warn!("asset scope for {}: {err}", dir.display());
    }
}

/// Starts the backend in the background if onboarding is done and the runtime is ready.
pub(crate) fn autostart_backend(app: &AppHandle) {
    let shell = shell(app);
    if !shell.settings.get().first_run_done || !shell.runtime_ready() || shell.is_quitting() {
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        if let Err(err) = shell.backend.start(&app, true).await {
            log::warn!("backend autostart: {err}");
        }
    });
}

fn log_plugin(paths: &AppPaths) -> tauri::plugin::TauriPlugin<tauri::Wry> {
    let mut builder = tauri_plugin_log::Builder::new()
        .clear_targets()
        .target(Target::new(TargetKind::Folder {
            path: paths.logs_dir.clone(),
            file_name: Some("shell".into()),
        }))
        .max_file_size(2 * 1024 * 1024)
        .rotation_strategy(RotationStrategy::KeepSome(3))
        .level(if cfg!(debug_assertions) {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .level_for("tao", log::LevelFilter::Warn)
        .level_for("wry", log::LevelFilter::Warn)
        .level_for("reqwest", log::LevelFilter::Warn)
        .level_for("hyper_util", log::LevelFilter::Warn);
    if cfg!(debug_assertions) {
        builder = builder.target(Target::new(TargetKind::Stdout));
    }
    builder.build()
}

fn setup(app: &mut tauri::App, paths: AppPaths) -> Result<(), Box<dyn std::error::Error>> {
    let handle = app.handle().clone();
    let resources = ResourcePaths::resolve(&handle)?;
    let settings = SettingsStore::load(paths.settings_file.clone());
    let version = handle.package_info().version.to_string();

    // Loopback client: never through a proxy.
    let local_http = reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_secs(3))
        .build()?;
    let net_http = reqwest::Client::builder()
        .user_agent(format!("Stellaria/{version}"))
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(60))
        .build()?;

    log::info!(
        "Stellaria {version} starting: data {}, engine {}, uv {}, dev python {:?}",
        paths.data_dir.display(),
        resources.engine.display(),
        resources.uv.display(),
        resources.dev_python
    );

    let shell = Arc::new(Shell {
        backend: BackendManager::new(&paths, local_http),
        bootstrap: BootstrapManager::default(),
        frontend: window::FrontendState::default(),
        quitting: AtomicBool::new(false),
        python_checks: Mutex::new(HashMap::new()),
        net_http,
        settings,
        resources,
        paths,
    });
    app.manage(shell.clone());

    allow_asset_dir(&handle, &shell.library_path());

    match tray::TrayState::create(&handle, shell.language()) {
        Ok(tray) => {
            app.manage(tray);
        }
        Err(err) => log::warn!("tray icon unavailable: {err}"),
    }

    // The main window is declared with `create: false` so that it only exists once the
    // state above is managed (its first IPC call is `get_shell_state`).
    let main_config = app
        .config()
        .app
        .windows
        .iter()
        .find(|w| w.label == window::MAIN)
        .cloned()
        .ok_or("tauri.conf.json defines no main window")?;
    tauri::WebviewWindowBuilder::from_config(&handle, &main_config)?.build()?;

    // The window starts hidden (no white flash) and is shown when the page has loaded;
    // this is the safety net if that event never arrives.
    let reveal = handle.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(Duration::from_secs(5)).await;
        window::reveal_initial(&reveal);
    });

    #[cfg(unix)]
    quit_on_signals(&handle);
    // A custom interpreter can change behind our back: re-check it once per launch so the
    // runtime status (and its versions) are current. The backend starts meanwhile.
    if let Some(python) = shell.settings.get().runtime.custom_python() {
        let python = python.to_path_buf();
        let shell = shell.clone();
        tauri::async_runtime::spawn(async move {
            let check = shell.check_python(&python).await;
            if !check.ok {
                log::warn!("custom Python check failed: {:?}", check.error);
            }
        });
    }
    autostart_backend(&handle);
    Ok(())
}

/// SIGTERM / SIGINT / SIGHUP (logout, `kill`, a terminal closing in dev) run the normal
/// graceful quit, so the backend and its jobs are not orphaned.
#[cfg(unix)]
fn quit_on_signals(app: &AppHandle) {
    use tokio::signal::unix::{signal, SignalKind};
    for kind in [
        SignalKind::terminate(),
        SignalKind::interrupt(),
        SignalKind::hangup(),
    ] {
        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            let mut stream = match signal(kind) {
                Ok(stream) => stream,
                Err(err) => {
                    log::warn!("cannot watch signal {kind:?}: {err}");
                    return;
                }
            };
            if stream.recv().await.is_some() {
                log::info!("signal {kind:?} received, quitting");
                window::quit(&app, false).await;
            }
        });
    }
}

pub fn run() {
    let paths = AppPaths::resolve().unwrap_or_else(|err| {
        eprintln!("cannot resolve app folders ({err:#}); using a temporary folder");
        let fallback = std::env::temp_dir().join("Stellaria");
        AppPaths::from_data_dir(fallback.join("data"), fallback.join("library"))
    });
    let _ = std::fs::create_dir_all(&paths.logs_dir);

    let builder = tauri::Builder::default()
        // Must come first: a second launch focuses the running instance and exits.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            window::show_main(app);
        }))
        .plugin(log_plugin(&paths))
        .plugin(
            tauri_plugin_window_state::Builder::new()
                .with_state_flags(window::window_state_flags())
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            commands::get_shell_state,
            commands::detect_hardware,
            commands::disk_space,
            commands::set_library_path,
            commands::set_language,
            commands::bootstrap_start,
            commands::bootstrap_cancel,
            commands::bootstrap_repair,
            commands::finish_onboarding,
            commands::backend_start,
            commands::backend_restart,
            commands::backend_stop,
            commands::backend_logs,
            commands::accept_terms,
            commands::set_close_to_tray,
            commands::set_tray_status,
            commands::reveal_path,
            commands::open_logs_folder,
            commands::quit_app,
            commands::frontend_pong,
            commands::check_python,
            commands::find_pythons,
            commands::set_runtime_python,
            commands::install_runtime_requirements,
        ])
        .on_page_load(window::on_page_load)
        .on_window_event(window::on_window_event)
        .setup(move |app| setup(app, paths));

    match builder.build(tauri::generate_context!()) {
        Ok(app) => app.run(window::on_run_event),
        Err(err) => {
            log::error!("failed to start Stellaria: {err}");
            eprintln!("failed to start Stellaria: {err}");
            std::process::exit(1);
        }
    }
}

//! Main-window lifecycle: no white flash on start, close interception, frontend liveness,
//! macOS reopen/quit and the graceful quit sequence.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use serde::Serialize;
use tauri::webview::{PageLoadEvent, PageLoadPayload};
use tauri::{AppHandle, Emitter, Manager, RunEvent, Webview, Window, WindowEvent};
use tauri_plugin_window_state::{AppHandleExt, StateFlags};

use crate::Shell;

pub const MAIN: &str = "main";
pub const CLOSE_EVENT: &str = "app://close-requested";

/// What the window-state plugin restores. Visibility and decorations stay ours.
pub fn window_state_flags() -> StateFlags {
    StateFlags::SIZE | StateFlags::POSITION | StateFlags::MAXIMIZED
}

#[derive(Default)]
pub struct FrontendState {
    /// The page has loaded and called `get_shell_state` (i.e. the app is up).
    pub ready: AtomicBool,
    /// Incremented by the hidden `frontend_pong` command.
    pub pongs: AtomicU64,
    /// The window has been shown once (it starts hidden to avoid a white flash).
    shown: AtomicBool,
    /// A close request is being handled.
    closing: AtomicBool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct CloseRequested {
    active_jobs: bool,
}

fn try_shell(app: &AppHandle) -> Option<std::sync::Arc<Shell>> {
    app.try_state::<std::sync::Arc<Shell>>()
        .map(|s| s.inner().clone())
}

pub fn show_main(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(MAIN) {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
    if let Some(shell) = try_shell(app) {
        shell.frontend.shown.store(true, Ordering::SeqCst);
    }
}

pub fn hide_main(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(MAIN) {
        let _ = window.hide();
    }
}

/// Shows the window the first time only (later reloads must not undo "hidden in tray").
pub fn reveal_initial(app: &AppHandle) {
    let first = try_shell(app).is_none_or(|s| !s.frontend.shown.swap(true, Ordering::SeqCst));
    if first {
        show_main(app);
    }
}

pub fn on_page_load(webview: &Webview, payload: &PageLoadPayload<'_>) {
    if webview.label() != MAIN {
        return;
    }
    let app = webview.app_handle();
    match payload.event() {
        PageLoadEvent::Started => {
            if let Some(shell) = try_shell(app) {
                shell.frontend.ready.store(false, Ordering::SeqCst);
            }
        }
        PageLoadEvent::Finished => reveal_initial(app),
    }
}

pub fn on_window_event(window: &Window, event: &WindowEvent) {
    if window.label() != MAIN {
        return;
    }
    if let WindowEvent::CloseRequested { api, .. } = event {
        // Always ours to decide; while quitting the window simply stays until exit.
        api.prevent_close();
        let app = window.app_handle().clone();
        tauri::async_runtime::spawn(async move { handle_close_request(app).await });
    }
}

pub fn on_run_event(app: &AppHandle, event: RunEvent) {
    match event {
        // `code: None` = user/system initiated (Cmd+Q, Dock → Quit, last window closed).
        // Our own `app.exit(0)` carries `Some(0)` and passes through.
        RunEvent::ExitRequested {
            code: None, api, ..
        } => {
            api.prevent_exit();
            let quitting = try_shell(app).is_some_and(|s| s.is_quitting());
            if !quitting {
                let app = app.clone();
                tauri::async_runtime::spawn(async move { handle_close_request(app).await });
            }
        }
        RunEvent::Exit => {
            if let Some(shell) = try_shell(app) {
                shell.backend.kill_now();
            }
        }
        #[cfg(target_os = "macos")]
        RunEvent::Reopen { .. } => show_main(app),
        _ => {}
    }
}

/// Is the frontend responsive? It must have booted (`get_shell_state`) and answer a ping
/// injected into the page within a second.
pub async fn frontend_alive(app: &AppHandle) -> bool {
    let Some(shell) = try_shell(app) else {
        return false;
    };
    if !shell.frontend.ready.load(Ordering::SeqCst) {
        return false;
    }
    let Some(window) = app.get_webview_window(MAIN) else {
        return false;
    };
    let before = shell.frontend.pongs.load(Ordering::SeqCst);
    let ping = "window.__TAURI_INTERNALS__ && window.__TAURI_INTERNALS__.invoke('frontend_pong')";
    if window.eval(ping).is_err() {
        return false;
    }
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        if shell.frontend.pongs.load(Ordering::SeqCst) > before {
            return true;
        }
    }
    false
}

/// Window close / Cmd+Q: a live frontend decides (`app://close-requested`, then it calls
/// `hide` or `quit_app`); otherwise hide to the tray when work is running and the user
/// wants that, else quit gracefully.
pub async fn handle_close_request(app: AppHandle) {
    let Some(shell) = try_shell(&app) else {
        app.exit(0);
        return;
    };
    if shell.is_quitting() || shell.frontend.closing.swap(true, Ordering::SeqCst) {
        return;
    }
    let (jobs_running, alive) = tokio::join!(shell.backend.active_jobs(), frontend_alive(&app));
    let active = jobs_running || shell.bootstrap.is_running();
    if alive {
        let visible = app
            .get_webview_window(MAIN)
            .and_then(|w| w.is_visible().ok())
            .unwrap_or(false);
        if active && !visible {
            show_main(&app);
        }
        if let Err(err) = app.emit(
            CLOSE_EVENT,
            CloseRequested {
                active_jobs: active,
            },
        ) {
            log::warn!("cannot emit close request: {err}");
        }
    } else if active && shell.settings.get().close_to_tray {
        hide_main(&app);
    } else {
        quit(&app, false).await;
    }
    shell.frontend.closing.store(false, Ordering::SeqCst);
}

/// Stops the bootstrap and the backend (gracefully unless `force`: training saves a
/// checkpoint, up to 25 s), saves the window state and exits.
pub async fn quit(app: &AppHandle, force: bool) {
    let Some(shell) = try_shell(app) else {
        app.exit(0);
        return;
    };
    if !shell.begin_quit() {
        return;
    }
    log::info!("quitting (force: {force})");
    shell.bootstrap.cancel().await;
    shell.backend.shutdown_for_quit(app, !force).await;
    if let Err(err) = app.save_window_state(window_state_flags()) {
        log::warn!("cannot save window state: {err}");
    }
    app.exit(0);
}

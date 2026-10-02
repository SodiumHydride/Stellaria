//! Tray icon with 显示星籁 / 暂停训练 / 退出, its tooltip, and the taskbar/dock progress.

use serde::Serialize;
use tauri::image::Image;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};
use tauri::window::{ProgressBarState, ProgressBarStatus};
use tauri::{AppHandle, Emitter, Manager};

use crate::types::Language;
use crate::window;

pub const ACTION_EVENT: &str = "tray://action";
const TRAY_ID: &str = "main";
const TOOLTIP: &str = "星籁 Stellaria";

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrayAction {
    Show,
    PauseTraining,
    ReleaseMemory,
    Quit,
}

#[derive(Serialize, Clone)]
struct ActionPayload {
    action: TrayAction,
}

pub struct TrayState {
    tray: TrayIcon,
    show: MenuItem<tauri::Wry>,
    pause: MenuItem<tauri::Wry>,
    release: MenuItem<tauri::Wry>,
    quit: MenuItem<tauri::Wry>,
}

struct Labels {
    show: &'static str,
    pause: &'static str,
    release: &'static str,
    quit: &'static str,
}

fn labels(language: Language) -> Labels {
    match language {
        Language::ZhCn => Labels {
            show: "显示星籁",
            pause: "暂停训练",
            release: "释放显存",
            quit: "退出",
        },
        Language::En => Labels {
            show: "Show Stellaria",
            pause: "Pause training",
            release: "Free GPU memory",
            quit: "Quit",
        },
    }
}

fn tray_image(app: &AppHandle) -> Image<'static> {
    if cfg!(target_os = "macos") {
        // Monochrome template image: macOS tints it for light/dark menu bars.
        tauri::include_image!("./icons/tray-template.png")
    } else {
        app.default_window_icon()
            .map(|icon| icon.clone().to_owned())
            .unwrap_or_else(|| tauri::include_image!("./icons/32x32.png"))
    }
}

impl TrayState {
    pub fn create(app: &AppHandle, language: Language) -> tauri::Result<Self> {
        let text = labels(language);
        let show = MenuItem::with_id(app, "show", text.show, true, None::<&str>)?;
        let pause = MenuItem::with_id(app, "pause_training", text.pause, true, None::<&str>)?;
        let release = MenuItem::with_id(app, "release_memory", text.release, true, None::<&str>)?;
        let quit = MenuItem::with_id(app, "quit", text.quit, true, None::<&str>)?;
        let separator = PredefinedMenuItem::separator(app)?;
        let menu = Menu::with_items(app, &[&show, &pause, &release, &separator, &quit])?;

        let tray = TrayIconBuilder::with_id(TRAY_ID)
            .icon(tray_image(app))
            .icon_as_template(cfg!(target_os = "macos"))
            .tooltip(TOOLTIP)
            .menu(&menu)
            // macOS convention: the menu opens on click. Elsewhere a click shows the window.
            .show_menu_on_left_click(cfg!(target_os = "macos"))
            .on_menu_event(|app, event| on_menu(app, event.id().as_ref()))
            .on_tray_icon_event(|tray, event| match event {
                TrayIconEvent::Click {
                    button: MouseButton::Left,
                    button_state: MouseButtonState::Up,
                    ..
                }
                | TrayIconEvent::DoubleClick {
                    button: MouseButton::Left,
                    ..
                } if !cfg!(target_os = "macos") => window::show_main(tray.app_handle()),
                _ => {}
            })
            .build(app)?;

        Ok(Self {
            tray,
            show,
            pause,
            release,
            quit,
        })
    }

    pub fn set_language(&self, language: Language) {
        let text = labels(language);
        for (item, label) in [
            (&self.show, text.show),
            (&self.pause, text.pause),
            (&self.release, text.release),
            (&self.quit, text.quit),
        ] {
            if let Err(err) = item.set_text(label) {
                log::warn!("tray label: {err}");
            }
        }
    }

    /// Tooltip text and taskbar (Windows) / dock (macOS) progress; `None` clears them.
    pub fn set_status(&self, app: &AppHandle, text: Option<&str>, progress: Option<f64>) {
        let tooltip = match text.map(str::trim).filter(|t| !t.is_empty()) {
            Some(text) => format!("{TOOLTIP} — {text}"),
            None => TOOLTIP.to_string(),
        };
        // Unsupported on Linux; harmless elsewhere.
        let _ = self.tray.set_tooltip(Some(tooltip));

        let Some(main) = app.get_webview_window(window::MAIN) else {
            return;
        };
        let percent = progress.map(|p| (p.clamp(0.0, 1.0) * 100.0).round() as u64);
        let state = match percent {
            Some(value) => ProgressBarState {
                status: Some(ProgressBarStatus::Normal),
                progress: Some(value),
            },
            None => ProgressBarState {
                status: Some(ProgressBarStatus::None),
                progress: None,
            },
        };
        if let Err(err) = main.set_progress_bar(state) {
            log::debug!("progress bar: {err}");
        }
        #[cfg(target_os = "macos")]
        if let Err(err) = main.set_badge_label(percent.map(|p| format!("{p}%"))) {
            log::debug!("dock badge: {err}");
        }
    }
}

fn emit_action(app: &AppHandle, action: TrayAction) {
    if let Err(err) = app.emit(ACTION_EVENT, ActionPayload { action }) {
        log::warn!("cannot emit tray action: {err}");
    }
}

fn on_menu(app: &AppHandle, id: &str) {
    match id {
        "show" => {
            window::show_main(app);
            emit_action(app, TrayAction::Show);
        }
        "pause_training" => {
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                // The frontend pauses through the backend API and shows feedback; without
                // a live frontend the shell does it directly.
                if window::frontend_alive(&app).await {
                    emit_action(&app, TrayAction::PauseTraining);
                } else {
                    crate::shell(&app).backend.pause_training().await;
                }
            });
        }
        "release_memory" => {
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                // Same split as pausing: a live frontend frees it and says so.
                if window::frontend_alive(&app).await {
                    emit_action(&app, TrayAction::ReleaseMemory);
                } else {
                    crate::shell(&app).backend.release_render_memory().await;
                }
            });
        }
        "quit" => {
            // Explicit intent: quit gracefully (training saves a checkpoint). A live frontend
            // first sends its unsaved edits and calls `quit_app` itself; if it doesn't within a
            // few seconds the shell quits anyway (`quit` runs once, whoever comes first).
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                if window::frontend_alive(&app).await {
                    emit_action(&app, TrayAction::Quit);
                    tokio::time::sleep(std::time::Duration::from_secs(6)).await;
                }
                window::quit(&app, false).await
            });
        }
        other => log::debug!("unknown tray menu id {other}"),
    }
}

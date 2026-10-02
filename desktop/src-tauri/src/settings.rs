//! `shell.json`: the few settings the shell owns (everything else is backend preferences).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Deserializer, Serialize};

use crate::error::{ShellError, ShellResult};
use crate::fsutil;
use crate::types::{Language, Region, RuntimeMode, TorchVariant};

/// `runtime` in `shell.json` (docs/creation-v2.md §4).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RuntimeSetting {
    pub mode: RuntimeMode,
    /// The user's interpreter; kept when switching back to `managed` so it can be reused.
    pub python_path: Option<PathBuf>,
}

impl RuntimeSetting {
    /// The user's interpreter when it is the one in use.
    pub fn custom_python(&self) -> Option<&Path> {
        match self.mode {
            RuntimeMode::Custom => self.python_path.as_deref(),
            RuntimeMode::Managed => None,
        }
    }
}

/// Older `shell.json` files have no `runtime`; a malformed one must not cost the user the rest
/// of their settings, and `custom` without a path means `managed`.
fn lenient_runtime<'de, D: Deserializer<'de>>(deserializer: D) -> Result<RuntimeSetting, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    let mut setting = serde_json::from_value::<RuntimeSetting>(value).unwrap_or_else(|err| {
        log::warn!("shell.json: invalid runtime setting ({err}); using the managed runtime");
        RuntimeSetting::default()
    });
    if setting.mode == RuntimeMode::Custom && setting.python_path.is_none() {
        setting.mode = RuntimeMode::Managed;
    }
    Ok(setting)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ShellSettings {
    pub first_run_done: bool,
    pub library_path: Option<PathBuf>,
    pub language: Option<Language>,
    pub region: Region,
    pub variant: Option<TorchVariant>,
    pub close_to_tray: bool,
    /// Version of the terms of use the user accepted (0 = none yet).
    pub terms_accepted: u32,
    #[serde(deserialize_with = "lenient_runtime")]
    pub runtime: RuntimeSetting,
}

impl Default for ShellSettings {
    fn default() -> Self {
        Self {
            first_run_done: false,
            library_path: None,
            language: None,
            region: Region::Auto,
            variant: None,
            close_to_tray: true,
            terms_accepted: 0,
            runtime: RuntimeSetting::default(),
        }
    }
}

pub struct SettingsStore {
    path: PathBuf,
    inner: Mutex<ShellSettings>,
}

impl SettingsStore {
    /// Loads `shell.json`; a corrupt file is kept as `shell.json.bad` and defaults are used.
    pub fn load(path: PathBuf) -> Self {
        let settings = match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice::<ShellSettings>(&bytes) {
                Ok(settings) => settings,
                Err(err) => {
                    log::warn!("shell.json is invalid ({err}); starting with defaults");
                    let _ = std::fs::rename(&path, path.with_extension("json.bad"));
                    ShellSettings::default()
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => ShellSettings::default(),
            Err(err) => {
                log::warn!("cannot read shell.json: {err}");
                ShellSettings::default()
            }
        };
        Self {
            path,
            inner: Mutex::new(settings),
        }
    }

    pub fn get(&self) -> ShellSettings {
        self.lock().clone()
    }

    /// Applies `change` and persists the result atomically.
    pub fn update(&self, change: impl FnOnce(&mut ShellSettings)) -> ShellResult<ShellSettings> {
        let mut guard = self.lock();
        let mut next = guard.clone();
        change(&mut next);
        fsutil::write_json_atomic(&self.path, &next)
            .map_err(|err| ShellError::io("Cannot save shell settings", &err))?;
        *guard = next.clone();
        Ok(next)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ShellSettings> {
        // A poisoned lock only means another thread panicked mid-update; the data is
        // still a valid ShellSettings, so keep going.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_defaults() {
        let parsed: ShellSettings =
            serde_json::from_str(r#"{"language":"zh-CN","variant":"cu128"}"#).unwrap();
        assert_eq!(parsed.language, Some(Language::ZhCn));
        assert_eq!(parsed.variant, Some(TorchVariant::Cu128));
        assert_eq!(parsed.region, Region::Auto);
        assert!(parsed.close_to_tray);
        let json = serde_json::to_value(&parsed).unwrap();
        assert_eq!(json["firstRunDone"], false);
        assert_eq!(json["region"], "auto");
        assert_eq!(json["runtime"]["mode"], "managed");
        assert!(json["runtime"]["pythonPath"].is_null());
    }

    #[test]
    fn migrates_files_without_runtime() {
        // shell.json as written by 0.1.0 before custom runtimes existed.
        let old = r#"{"firstRunDone":true,"libraryPath":"/x/lib","language":"en","region":"cn","variant":"cu126","closeToTray":false}"#;
        let parsed: ShellSettings = serde_json::from_str(old).unwrap();
        assert!(parsed.first_run_done);
        assert_eq!(parsed.region, Region::Cn);
        assert_eq!(parsed.runtime, RuntimeSetting::default());
        assert_eq!(parsed.runtime.custom_python(), None);
    }

    #[test]
    fn custom_runtime_roundtrip() {
        let text = r#"{"firstRunDone":true,"runtime":{"mode":"custom","pythonPath":"/opt/py/bin/python"}}"#;
        let parsed: ShellSettings = serde_json::from_str(text).unwrap();
        assert_eq!(parsed.runtime.mode, RuntimeMode::Custom);
        assert_eq!(
            parsed.runtime.custom_python(),
            Some(Path::new("/opt/py/bin/python"))
        );
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(back["runtime"]["mode"], "custom");
        assert_eq!(back["runtime"]["pythonPath"], "/opt/py/bin/python");
        // Switching back to managed keeps the path for later reuse.
        let mut managed = parsed.clone();
        managed.runtime.mode = RuntimeMode::Managed;
        assert_eq!(managed.runtime.custom_python(), None);
        assert!(managed.runtime.python_path.is_some());
    }

    #[test]
    fn malformed_runtime_keeps_other_settings() {
        for runtime in [
            r#"{"mode":"conda","pythonPath":"/x"}"#,
            r#""custom""#,
            r#"{"mode":"custom","pythonPath":null}"#,
            r#"{"mode":"custom"}"#,
        ] {
            let text = format!(r#"{{"firstRunDone":true,"language":"zh-CN","runtime":{runtime}}}"#);
            let parsed: ShellSettings = serde_json::from_str(&text).unwrap();
            assert!(parsed.first_run_done, "{runtime}");
            assert_eq!(parsed.language, Some(Language::ZhCn));
            assert_eq!(parsed.runtime.mode, RuntimeMode::Managed, "{runtime}");
        }
    }
}

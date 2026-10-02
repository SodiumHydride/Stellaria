//! Small enums shared by several modules and serialized exactly as the contract spells them.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TorchVariant {
    Cu128,
    Cu126,
    Cpu,
    Mac,
}

impl TorchVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cu128 => "cu128",
            Self::Cu126 => "cu126",
            Self::Cpu => "cpu",
            Self::Mac => "mac",
        }
    }

    pub fn is_cuda(self) -> bool {
        matches!(self, Self::Cu128 | Self::Cu126)
    }

    /// Local version label of the wheels on the PyTorch indexes (`2.9.1+cu128`).
    /// The macOS build comes from PyPI and has none.
    pub fn local_label(self) -> Option<&'static str> {
        match self {
            Self::Mac => None,
            other => Some(other.as_str()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language {
    #[serde(rename = "zh-CN")]
    ZhCn,
    #[serde(rename = "en")]
    En,
}

impl Language {
    /// Language to use when the user has not picked one yet.
    pub fn system_default() -> Self {
        match sys_locale::get_locale() {
            Some(locale) if locale.to_ascii_lowercase().starts_with("zh") => Self::ZhCn,
            _ => Self::En,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Region {
    #[default]
    Auto,
    Cn,
    Global,
}

/// Which Python runs the backend: the runtime the app installed, or the user's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeMode {
    #[default]
    Managed,
    Custom,
}

/// `'windows' | 'macos' | 'linux'` as used in `ShellState.platform` and `HardwareReport.os`.
pub fn platform_name() -> &'static str {
    match std::env::consts::OS {
        "windows" => "windows",
        "macos" => "macos",
        _ => "linux",
    }
}

/// A localizable message: `{code, params?}` (docs/api.md `Msg`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Msg {
    pub code: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<BTreeMap<String, Value>>,
}

impl Msg {
    pub fn new(code: &str) -> Self {
        Self {
            code: code.into(),
            params: None,
        }
    }

    #[must_use]
    pub fn with(mut self, key: &str, value: impl Into<Value>) -> Self {
        self.params
            .get_or_insert_with(BTreeMap::new)
            .insert(key.into(), value.into());
        self
    }
}

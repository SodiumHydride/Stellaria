//! The part of `resources/manifest.json` the shell needs: the Python runtime definition.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::{ShellError, ShellResult};
use crate::types::TorchVariant;

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub runtime: RuntimeManifest,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeManifest {
    pub id: String,
    pub python: String,
    pub torch: String,
    pub torchaudio: String,
    pub requirements: String,
    #[serde(default)]
    pub python_install_mirrors: Vec<Source>,
    #[serde(default)]
    pub pypi: Vec<Source>,
    #[serde(default)]
    pub variants: HashMap<String, VariantManifest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Source {
    pub id: String,
    /// `global`, `cn` or `any`.
    #[serde(default = "any_region")]
    pub region: String,
    pub url: Option<String>,
    #[serde(rename = "type", default)]
    pub kind: Option<SourceKind>,
}

fn any_region() -> String {
    "any".into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceKind {
    /// PEP 503 simple index (`--extra-index-url`).
    Index,
    /// Flat HTML listing of wheels (`--find-links`).
    FindLinks,
    /// Plain PyPI (the chosen PyPI mirror).
    Pypi,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VariantManifest {
    #[serde(default)]
    pub torch_sources: Vec<Source>,
    #[serde(default)]
    pub extra_packages: HashMap<String, Vec<String>>,
}

impl Manifest {
    pub fn load(path: &Path) -> ShellResult<Self> {
        let bytes = std::fs::read(path).map_err(|err| {
            ShellError::new("manifest.missing", "The resource manifest is missing")
                .with_detail(format!("{}: {err}", path.display()))
        })?;
        serde_json::from_slice(&bytes).map_err(|err| {
            ShellError::new("manifest.invalid", "The resource manifest is invalid")
                .with_detail(format!("{}: {err}", path.display()))
        })
    }

    pub fn variant(&self, variant: TorchVariant) -> ShellResult<&VariantManifest> {
        self.runtime.variants.get(variant.as_str()).ok_or_else(|| {
            ShellError::new(
                "manifest.variant_missing",
                format!("The manifest has no runtime variant '{}'", variant.as_str()),
            )
        })
    }
}

impl VariantManifest {
    /// Extra packages for the current OS (`windows` / `macos` / `linux`).
    pub fn extra_packages_for_host(&self) -> Vec<String> {
        self.extra_packages
            .get(crate::types::platform_name())
            .cloned()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_repository_manifest() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resources/manifest.json");
        let manifest = Manifest::load(&path).expect("manifest parses");
        assert!(!manifest.runtime.id.is_empty());
        let cu128 = manifest.variant(TorchVariant::Cu128).unwrap();
        assert!(cu128
            .torch_sources
            .iter()
            .any(|s| s.kind == Some(SourceKind::FindLinks)));
        let mac = manifest.variant(TorchVariant::Mac).unwrap();
        assert_eq!(mac.torch_sources[0].kind, Some(SourceKind::Pypi));
        assert!(manifest.runtime.pypi.iter().any(|s| s.region == "cn"));
    }
}

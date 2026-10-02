//! Stellaria's layer over the user's own Python (their conda environment, a DDSP-SVC venv, a
//! bundle's embeddable Python, …): a small interpreter setup in the app's data folder that sees
//! their packages — a `.pth` file adds their site-packages after its own — and holds only what
//! the app needs and they lack. Their environment is never written to; the add-ons the backend
//! installs land in the layer too.
//!
//! Two kinds:
//! * `venv` — a venv over their interpreter (the usual case).
//! * `shadow` — for embeddable Pythons (a `python3XX._pth` next to the executable), which ignore
//!   venvs: a copy of their executable (and its DLLs) with our own `._pth` listing their stdlib
//!   and folders plus the layer's site-packages. A `._pth` also makes Python skip `PYTHONPATH`
//!   and the script's folder on `sys.path`, which the app's processes rely on; the shim `.pth`
//!   ([`SHIM`]) puts both back. (`PYTHONUTF8`, `PYTHONIOENCODING`, `PYTHONUNBUFFERED`, … are read
//!   before the `._pth` and still apply.)

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::{ShellError, ShellResult};
use crate::fsutil;
use crate::paths::AppPaths;
use crate::procutil::{hidden_command, run_capture};

/// In the layer folder: the interpreter it sits on (the backend reads `sites` from it too).
pub const MARKER: &str = "stellaria-layer.json";
/// In the layer's site-packages: adds the user's site-packages after the layer's own.
pub const BRIDGE: &str = "_stellaria_base.pth";
/// In a shadow layer's site-packages: `PYTHONPATH` and the script's folder (or the working
/// directory for `-m`/`-c`) at the front of `sys.path`, as a Python without a `._pth` does.
pub const SHIM: &str = "_stellaria_paths.pth";
const SHIM_PTH: &str = "import os, sys; exec(\"a = sys.argv[0] if sys.argv else ''\\nhead = [os.getcwd() if a in ('', '-c', '-m') else os.path.dirname(os.path.abspath(a))]\\nhead += [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]\\nsys.path[0:0] = [p for p in head if p not in sys.path]\\ndel a, head\")\n";
const QUERY_TIMEOUT: Duration = Duration::from_secs(90);
const CREATE_TIMEOUT: Duration = Duration::from_secs(120);
const PREFIX: &str = "STELLARIA_BASE ";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Marker {
    /// The user's interpreter.
    pub base: PathBuf,
    /// `major.minor` of it; a layer for another version is rebuilt.
    pub version: String,
    /// Their site-packages folders, in the order the bridge adds them.
    pub sites: Vec<String>,
    /// `venv` or `shadow` (see the module docs).
    #[serde(default = "venv_kind")]
    pub kind: String,
    pub created_at: String,
}

fn venv_kind() -> String {
    "venv".into()
}

/// What the user's interpreter reports about itself.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BaseInfo {
    /// `major.minor`
    pub version: String,
    pub sites: Vec<String>,
    /// The bridge `.pth` (ASCII: Python writes the paths as escaped literals, since `.pth` files
    /// are read with the locale encoding).
    pub bridge: String,
    /// Distribution names installed there.
    pub installed: Vec<String>,
}

/// Whether `base` is an embeddable Python (a `python*._pth` next to it): it gets a shadow layer.
pub fn is_embeddable(base: &Path) -> bool {
    base.parent().is_some_and(|dir| !pth_files(dir).is_empty())
}

/// `python*._pth` files in `dir`.
fn pth_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            name.starts_with("python") && name.ends_with("._pth")
        })
        .collect();
    found.sort();
    found
}

/// The layer's interpreter for `base`: the venv's, or the shadow's copy of their executable.
pub fn interpreter(paths: &AppPaths, base: &Path) -> PathBuf {
    match base.file_name() {
        Some(name) if is_embeddable(base) => paths.layer_dir.join(name),
        _ => paths.layer_python(),
    }
}

pub fn read_marker(paths: &AppPaths) -> Option<Marker> {
    let bytes = std::fs::read(paths.layer_dir.join(MARKER)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// The layer's interpreter when a layer over `base` exists.
pub fn existing(paths: &AppPaths, base: &Path) -> Option<PathBuf> {
    let python = interpreter(paths, base);
    read_marker(paths)
        .filter(|m| m.base == base && python.is_file())
        .map(|_| python)
}

/// Removes the layer (a broken one, before falling back to the user's interpreter itself).
pub async fn remove(paths: &AppPaths) {
    let dir = paths.layer_dir.clone();
    if dir.exists() {
        let _ = tokio::task::spawn_blocking(move || fsutil::remove_dir_all_retry(&dir)).await;
    }
}

/// Asks `base` for its site-packages, the bridge that adds them and what is installed there.
/// Only reads: nothing is written to the user's environment.
pub async fn query(base: &Path) -> ShellResult<BaseInfo> {
    let mut cmd = hidden_command(base);
    cmd.arg("-c")
        .arg(QUERY_SCRIPT)
        .current_dir(std::env::temp_dir());
    crate::pycheck::apply_custom_env(&mut cmd, base, None);
    let failed = |detail: String| {
        ShellError::new(
            "runtime.layer_failed",
            "Could not read your Python environment",
        )
        .with_detail(detail)
    };
    let out = run_capture(cmd, QUERY_TIMEOUT)
        .await
        .map_err(|e| failed(format!("{}: {e}", base.display())))?;
    out.stdout
        .lines()
        .find_map(|l| l.trim_end().strip_prefix(PREFIX))
        .and_then(|json| serde_json::from_str::<BaseInfo>(json).ok())
        .ok_or_else(|| failed(tail(&out.stderr)))
}

/// Creates the layer over `base` — or keeps the one already there for the same interpreter and
/// version — and (re)writes its bridge and marker. Returns the layer's interpreter.
///
/// `uv` creates a venv when given (the user's Python needs no `venv` module then), else
/// `base -m venv --without-pip`; embeddable Pythons get a shadow instead (no tool needed).
pub async fn create(
    paths: &AppPaths,
    uv: Option<&Path>,
    base: &Path,
    info: &BaseInfo,
) -> ShellResult<PathBuf> {
    let dir = &paths.layer_dir;
    let shadow = is_embeddable(base);
    let kind = if shadow { "shadow" } else { "venv" };
    let python = interpreter(paths, base);
    let reusable = read_marker(paths)
        .is_some_and(|m| m.base == base && m.version == info.version && m.kind == kind)
        && python.is_file();
    if !reusable {
        remove(paths).await;
        if let Some(parent) = dir.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ShellError::io(format!("Cannot create {}", parent.display()), &e))?;
        }
        if shadow {
            create_shadow(dir, base)?;
        } else {
            create_venv(paths, uv, base).await?;
        }
        if !python.is_file() {
            return Err(ShellError::new(
                "runtime.layer_failed",
                "Could not create the app's layer",
            )
            .with_detail(python.display().to_string()));
        }
    }
    let site = if shadow {
        shadow_site(&python, base).await?
    } else {
        site_packages(dir, &info.version)
    };
    std::fs::create_dir_all(&site)
        .map_err(|e| ShellError::io(format!("Cannot create {}", site.display()), &e))?;
    fsutil::write_atomic(&site.join(BRIDGE), info.bridge.as_bytes())
        .map_err(|e| ShellError::io("Cannot write the layer's bridge", &e))?;
    if shadow {
        fsutil::write_atomic(&site.join(SHIM), SHIM_PTH.as_bytes())
            .map_err(|e| ShellError::io("Cannot write the layer's bridge", &e))?;
    }
    let marker = Marker {
        base: base.to_path_buf(),
        version: info.version.clone(),
        sites: info.sites.clone(),
        kind: kind.into(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    fsutil::write_json_atomic(&dir.join(MARKER), &marker)
        .map_err(|e| ShellError::io("Cannot write the layer's marker", &e))?;
    Ok(python)
}

async fn create_venv(paths: &AppPaths, uv: Option<&Path>, base: &Path) -> ShellResult<()> {
    let dir = &paths.layer_dir;
    let mut cmd = match uv {
        Some(uv) => {
            let mut cmd = crate::bootstrap::uv_command(paths, uv);
            cmd.args(["venv", "--quiet", "--python"])
                .arg(base)
                .arg(dir)
                .env("UV_PYTHON_DOWNLOADS", "never");
            cmd
        }
        None => {
            let mut cmd = hidden_command(base);
            cmd.args(["-m", "venv", "--without-pip"]).arg(dir);
            crate::pycheck::apply_custom_env(&mut cmd, base, None);
            cmd
        }
    };
    cmd.current_dir(&paths.runtime_dir);
    std::fs::create_dir_all(&paths.runtime_dir).ok();
    let out = run_capture(cmd, CREATE_TIMEOUT).await.map_err(|e| {
        ShellError::new("runtime.layer_failed", "Could not create the app's layer")
            .with_detail(e.to_string())
    })?;
    if !out.success {
        return Err(
            ShellError::new("runtime.layer_failed", "Could not create the app's layer")
                .with_detail(tail(&format!("{}\n{}", out.stdout, out.stderr))),
        );
    }
    Ok(())
}

/// The shadow of an embeddable `base` in `dir`: its executable, its Python and runtime DLLs, and
/// `._pth` files (named after the executable and each Python DLL, wherever Python looks first)
/// with the base's own entries made absolute, then `import site` (which adds the layer's
/// site-packages and runs the bridge and the shim).
fn create_shadow(dir: &Path, base: &Path) -> ShellResult<()> {
    let failed = |what: &str, e: &std::io::Error| {
        ShellError::new("runtime.layer_failed", "Could not create the app's layer")
            .with_detail(format!("{what}: {e}"))
    };
    let base_dir = base.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir).map_err(|e| failed(&dir.display().to_string(), &e))?;
    let pth_text = pth_files(base_dir)
        .into_iter()
        .find_map(|p| std::fs::read_to_string(p).ok())
        .unwrap_or_default();
    let pth = shadow_pth(base_dir, &pth_text);

    let mut names: Vec<std::ffi::OsString> = base.file_name().into_iter().map(Into::into).collect();
    let mut stems: Vec<String> = base
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .into_iter()
        .collect();
    if let Ok(entries) = std::fs::read_dir(base_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let lower = name.to_string_lossy().to_lowercase();
            let runtime_dll = lower.ends_with(".dll")
                && ["python", "vcruntime", "msvcp"]
                    .iter()
                    .any(|p| lower.starts_with(p));
            if runtime_dll && entry.path().is_file() {
                if lower.starts_with("python") {
                    stems.push(lower.trim_end_matches(".dll").to_string());
                }
                names.push(name);
            }
        }
    }
    for name in &names {
        let from = base_dir.join(name);
        let to = dir.join(name);
        std::fs::copy(&from, &to).map_err(|e| failed(&from.display().to_string(), &e))?;
    }
    for stem in stems {
        let file = dir.join(format!("{stem}._pth"));
        fsutil::write_atomic(&file, pth.as_bytes())
            .map_err(|e| failed(&file.display().to_string(), &e))?;
    }
    Ok(())
}

/// The shadow's `._pth`: the base's entries (relative ones resolved against its folder; its
/// `import site` and comments dropped), then `import site`. UTF-8, as Python reads it.
pub fn shadow_pth(base_dir: &Path, base_pth: &str) -> String {
    let mut out = String::new();
    for raw in base_pth.lines() {
        let line = raw.trim_start_matches('\u{feff}');
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with("import ") {
            continue;
        }
        let path = Path::new(line);
        let abs = if path.is_absolute() {
            path.to_path_buf()
        } else if line == "." {
            base_dir.to_path_buf()
        } else {
            base_dir.join(path)
        };
        out.push_str(&abs.to_string_lossy());
        out.push('\n');
    }
    out.push_str("import site\n");
    out
}

/// Where the shadow installs packages (and where `site` reads the bridge and shim from).
async fn shadow_site(python: &Path, base: &Path) -> ShellResult<PathBuf> {
    let mut cmd = hidden_command(python);
    cmd.arg("-c")
        .arg("import json, sysconfig; print('STELLARIA_SITE ' + json.dumps(sysconfig.get_paths()['purelib']))")
        .current_dir(std::env::temp_dir());
    crate::pycheck::apply_custom_env(&mut cmd, base, None);
    let failed = |detail: String| {
        ShellError::new("runtime.layer_failed", "Could not create the app's layer")
            .with_detail(detail)
    };
    let out = run_capture(cmd, QUERY_TIMEOUT)
        .await
        .map_err(|e| failed(format!("{}: {e}", python.display())))?;
    out.stdout
        .lines()
        .find_map(|l| l.trim_end().strip_prefix("STELLARIA_SITE "))
        .and_then(|json| serde_json::from_str::<String>(json).ok())
        .map(PathBuf::from)
        .ok_or_else(|| failed(tail(&format!("{}\n{}", out.stdout, out.stderr))))
}

fn site_packages(venv: &Path, version: &str) -> PathBuf {
    if cfg!(windows) {
        venv.join("Lib").join("site-packages")
    } else {
        venv.join("lib")
            .join(format!("python{version}"))
            .join("site-packages")
    }
}

fn normalize(name: &str) -> String {
    static SEP: OnceLock<Regex> = OnceLock::new();
    SEP.get_or_init(|| Regex::new(r"[-_.]+").unwrap())
        .replace_all(name, "-")
        .to_lowercase()
}

/// The app's requirements with every package the user already has reduced to its bare name:
/// resolving that against their environment keeps their versions and only fills the gaps.
/// Options (`-r`, `--index-url`, …) are dropped; the caller passes its own index.
pub fn relaxed_requirements(text: &str, installed: &[String]) -> String {
    static NAME: OnceLock<Regex> = OnceLock::new();
    let name_re = NAME
        .get_or_init(|| Regex::new(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?").unwrap());
    let have: HashSet<String> = installed.iter().map(|n| normalize(n)).collect();
    let mut out = String::new();
    for raw in text.lines() {
        let line = raw.split(" #").next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('-') {
            continue;
        }
        let (spec, marker) = match line.split_once(';') {
            Some((spec, marker)) => (spec, Some(marker.trim())),
            None => (line, None),
        };
        let Some(caps) = name_re.captures(spec) else {
            continue;
        };
        if !have.contains(&normalize(&caps[1])) {
            out.push_str(line);
        } else {
            out.push_str(&caps[1]);
            if let Some(extras) = caps.get(2) {
                out.push_str(extras.as_str());
            }
            if let Some(marker) = marker.filter(|m| !m.is_empty()) {
                out.push_str(" ; ");
                out.push_str(marker);
            }
        }
        out.push('\n');
    }
    out
}

/// A `+ name==version` line of a `uv pip install --dry-run` plan: something the layer must hold.
pub fn planned(line: &str) -> Option<String> {
    static PLAN: OnceLock<Regex> = OnceLock::new();
    let re = PLAN.get_or_init(|| {
        Regex::new(r"^\s*\+\s+([A-Za-z0-9][A-Za-z0-9._-]*(?:\[[^\]]*\])?==\S+)\s*$").unwrap()
    });
    re.captures(line).map(|c| c[1].to_string())
}

fn tail(text: &str) -> String {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(20)..].join("\n")
}

/// Site-packages (and the user site when it is on), the bridge that adds them — conda
/// environments on Windows also get their DLL folders, as activation would — and the
/// installed distributions.
const QUERY_SCRIPT: &str = r##"
import json, os, site, sys, sysconfig
dirs = []
def add(p):
    if p and os.path.isdir(p):
        p = os.path.abspath(p)
        if p not in dirs:
            dirs.append(p)
try:
    for p in site.getsitepackages():
        add(p)
except Exception:
    pass
paths = sysconfig.get_paths()
add(paths.get("purelib"))
add(paths.get("platlib"))
# site folders listed elsewhere (an embeddable Python's ._pth, say): their .pth files must run too
for p in sys.path:
    if os.path.basename(os.path.normpath(p or ".")).lower() in ("site-packages", "dist-packages"):
        add(p)
try:
    if site.ENABLE_USER_SITE:
        add(site.getusersitepackages())
except Exception:
    pass
lines = ["import site; site.addsitedir(%s)\n" % ascii(d) for d in dirs]
if os.name == "nt":
    dll = []
    for root in dict.fromkeys([sys.prefix, sys.base_prefix]):
        if os.path.isdir(os.path.join(root, "conda-meta")):
            lib = os.path.join(root, "Library")
            for d in (root, os.path.join(lib, "mingw-w64", "bin"), os.path.join(lib, "usr", "bin"),
                      os.path.join(lib, "bin"), os.path.join(root, "Scripts"), os.path.join(root, "bin")):
                if os.path.isdir(d) and d not in dll:
                    dll.append(d)
    if dll:
        lines.insert(0, "import os, sys; sys._stellaria_dll_dirs = [os.add_dll_directory(d) for d in %s if os.path.isdir(d)]\n" % ascii(dll))
installed = set()
try:
    from importlib import metadata
    for dist in metadata.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:
            name = None
        if name:
            installed.add(name)
except Exception:
    pass
out = {"version": "%d.%d" % sys.version_info[:2], "sites": dirs, "bridge": "".join(lines),
       "installed": sorted(installed)}
sys.stdout.write("STELLARIA_BASE " + json.dumps(out) + "\n")
sys.stdout.flush()
"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relaxes_what_the_user_has() {
        let text = "# comment\nlibrosa==0.11.0\nnumpy==1.26.4  # pinned\n--index-url https://x\n\
                    nvidia-ml-py==13.615.71; sys_platform != \"darwin\"\nuvicorn[standard]==0.54.0\n\
                    Python_Multipart==0.0.32\n";
        let installed = vec![
            "NumPy".to_string(),
            "nvidia_ml_py".to_string(),
            "uvicorn".to_string(),
        ];
        assert_eq!(
            relaxed_requirements(text, &installed),
            "librosa==0.11.0\nnumpy\nnvidia-ml-py ; sys_platform != \"darwin\"\n\
             uvicorn[standard]\nPython_Multipart==0.0.32\n"
        );
    }

    #[test]
    fn reads_the_dry_run_plan() {
        let plan = "Using Python 3.11.13 environment at: /x\nResolved 7 packages in 78ms\n\
                    Would uninstall 1 package\nWould install 3 packages\n + certifi==2026.7.22\n \
                    - numpy==1.26.4\n + numpy==2.0.0\n + uvicorn[standard]==0.54.0\n";
        let got: Vec<String> = plan.lines().filter_map(planned).collect();
        assert_eq!(
            got,
            [
                "certifi==2026.7.22",
                "numpy==2.0.0",
                "uvicorn[standard]==0.54.0"
            ]
        );
        assert_eq!(planned(" + pkg @ file:///x"), None);
    }

    #[test]
    fn embeddable_pythons_get_a_shadow() {
        let dir = std::env::temp_dir().join(format!("stellaria-pth-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let paths = AppPaths::from_data_dir(dir.join("data"), dir.join("lib"));
        let python = dir.join("runtime").join("python.exe");
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, b"").unwrap();
        assert!(!is_embeddable(&python));
        assert_eq!(interpreter(&paths, &python), paths.layer_python());
        std::fs::write(
            python.with_file_name("python311._pth"),
            b"python311.zip\n.\n",
        )
        .unwrap();
        assert!(is_embeddable(&python));
        assert_eq!(
            interpreter(&paths, &python),
            paths.layer_dir.join("python.exe")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_shadow_pth_points_at_the_base() {
        let base = Path::new("/bundle/runtime");
        let text = "\u{feff}python311.zip\n.\n# a comment\nLib/site-packages  # theirs\n\
                    /abs/extra\nimport site\n\n";
        let expected = [
            base.join("python311.zip"),
            base.to_path_buf(),
            base.join("Lib/site-packages"),
            PathBuf::from("/abs/extra"),
        ]
        .iter()
        .map(|p| format!("{}\n", p.display()))
        .collect::<String>()
            + "import site\n";
        assert_eq!(shadow_pth(base, text), expected);
        assert_eq!(shadow_pth(base, ""), "import site\n");
    }

    #[test]
    fn a_layer_belongs_to_one_interpreter() {
        let root = std::env::temp_dir().join(format!("stellaria-layer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let paths = AppPaths::from_data_dir(root.join("data"), root.join("lib"));
        let base = Path::new("/envs/svc/bin/python");
        assert_eq!(existing(&paths, base), None);
        let marker = Marker {
            base: base.to_path_buf(),
            version: "3.11".into(),
            sites: vec!["/envs/svc/lib/python3.11/site-packages".into()],
            kind: "venv".into(),
            created_at: "2026-10-02T00:00:00Z".into(),
        };
        std::fs::create_dir_all(&paths.layer_dir).unwrap();
        fsutil::write_json_atomic(&paths.layer_dir.join(MARKER), &marker).unwrap();
        assert_eq!(existing(&paths, base), None, "no interpreter yet");
        let python = paths.layer_python();
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, b"").unwrap();
        assert_eq!(existing(&paths, base), Some(python));
        assert_eq!(existing(&paths, Path::new("/envs/other/bin/python")), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A real layer over a real interpreter (needs `uv` on PATH; reads the interpreter only):
    /// `STELLARIA_LAYER_BASE=/path/to/python cargo test -- --ignored layer_over_a_real_python --nocapture`
    #[tokio::test]
    #[ignore = "needs a Python with packages to layer over"]
    async fn layer_over_a_real_python() {
        let base = PathBuf::from(std::env::var_os("STELLARIA_LAYER_BASE").expect("base"));
        let root = std::env::temp_dir().join(format!("stellaria-layer-e2e-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let paths = AppPaths::from_data_dir(root.join("data"), root.join("lib"));
        let uv = crate::paths::find_in_path("uv").expect("uv on PATH");
        let info = query(&base).await.unwrap();
        println!("{info:#?}");
        assert!(!info.sites.is_empty());
        let python = create(&paths, Some(&uv), &base, &info).await.unwrap();
        // The layer sees the base's packages, after its own site-packages.
        let out = std::process::Command::new(&python)
            .args(["-c", "import sys, json; print(json.dumps(sys.path))"])
            .output()
            .unwrap();
        let sys_path: Vec<String> = serde_json::from_slice(&out.stdout).unwrap();
        println!("{sys_path:#?}");
        let own = sys_path
            .iter()
            .position(|p| p.starts_with(&*paths.layer_dir.to_string_lossy()))
            .expect("the layer's site-packages");
        for site in &info.sites {
            let at = sys_path.iter().position(|p| p == site).expect(site);
            assert!(at > own, "{site} must come after the layer's own packages");
        }
        // Kept for the same interpreter, rebuilt for another one.
        let again = create(&paths, Some(&uv), &base, &info).await.unwrap();
        assert_eq!(again, python);
        assert_eq!(existing(&paths, &base), Some(python));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A shadow over a simulated embeddable Python (a copy of a self-contained interpreter with
    /// a `._pth`, as Windows bundles ship): it sees the base's packages and `.pth` files, honours
    /// `PYTHONPATH` and the script's folder, and the base is left alone. Unix only (the `._pth`
    /// rules are the same):
    /// `STELLARIA_SHADOW_PYTHON=/usr/bin/python3.11 cargo test -- --ignored shadow_over --nocapture`
    #[cfg(unix)]
    #[tokio::test]
    #[ignore = "needs a self-contained python binary"]
    async fn shadow_over_an_embeddable_python() {
        let real = PathBuf::from(std::env::var_os("STELLARIA_SHADOW_PYTHON").expect("python"));
        let root = std::env::temp_dir().join(format!("stellaria-shadow-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let stdlib = std::process::Command::new(&real)
            .args([
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['stdlib'])",
            ])
            .output()
            .unwrap();
        let stdlib = String::from_utf8_lossy(&stdlib.stdout).trim().to_string();
        // the "bundle": its python, a ._pth, its own site-packages with a package and a .pth
        let base_dir = root.join("整合包").join("runtime");
        let site = base_dir.join("site-packages");
        std::fs::create_dir_all(site.join("basepkg")).unwrap();
        std::fs::write(site.join("basepkg/__init__.py"), "X = 'base'\n").unwrap();
        std::fs::write(
            site.join("zz_base.pth"),
            "import sys; sys._base_pth = True\n",
        )
        .unwrap();
        let base = base_dir.join("python3");
        std::fs::copy(&real, &base).unwrap();
        std::fs::write(
            base_dir.join("python3._pth"),
            format!("{stdlib}\n{stdlib}/lib-dynload\nsite-packages\nimport site\n"),
        )
        .unwrap();
        let before = std::fs::read_dir(&base_dir).unwrap().count();

        let paths = AppPaths::from_data_dir(root.join("data"), root.join("lib"));
        let info = query(&base).await.unwrap();
        let python = create(&paths, None, &base, &info).await.unwrap();
        assert_eq!(python, paths.layer_dir.join("python3"));
        assert_eq!(read_marker(&paths).unwrap().kind, "shadow");
        assert_eq!(existing(&paths, &base), Some(python.clone()));

        // PYTHONPATH and the script's folder, like any Python; the base's packages and .pth
        let extra = root.join("extra");
        std::fs::create_dir_all(extra.join("ppmod")).unwrap();
        std::fs::write(extra.join("ppmod/__init__.py"), "Y = 'pp'\n").unwrap();
        let work = root.join("work");
        std::fs::create_dir_all(work.join("localmod")).unwrap();
        std::fs::write(work.join("localmod/__init__.py"), "Z = 'local'\n").unwrap();
        std::fs::write(
            work.join("run.py"),
            "import sys, basepkg, ppmod, localmod\n\
             print(basepkg.X, ppmod.Y, localmod.Z, getattr(sys, '_base_pth', False), \
             sys.flags.utf8_mode)\n",
        )
        .unwrap();
        let out = std::process::Command::new(&python)
            .arg(work.join("run.py"))
            .current_dir(&root)
            .env("PYTHONPATH", &extra)
            .env("PYTHONUTF8", "1")
            .output()
            .unwrap();
        println!("{}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            "base pp local True 1"
        );
        let out = std::process::Command::new(&python)
            .args(["-c", "import localmod, ppmod; print(localmod.Z)"])
            .current_dir(&work)
            .env("PYTHONPATH", &extra)
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "local");

        // installs land in the layer (the folder `site` reads the bridge from)
        let site_dir = shadow_site(&python, &base).await.unwrap();
        assert!(site_dir.starts_with(&paths.layer_dir));
        assert!(site_dir.join(BRIDGE).is_file() && site_dir.join(SHIM).is_file());
        // kept for the same base; the base itself is untouched
        assert_eq!(create(&paths, None, &base, &info).await.unwrap(), python);
        assert_eq!(std::fs::read_dir(&base_dir).unwrap().count(), before);
        let _ = std::fs::remove_dir_all(&root);
    }
}

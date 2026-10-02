//! Python interpreters someone already has: inside a folder they point at (a DDSP-SVC checkout
//! or bundle keeps its own in `venv`, `runtime`, `workenv`, …), conda environments, and the
//! usual install locations. Nothing is run here; `check_python` decides whether one is usable.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PythonCandidate {
    pub path: PathBuf,
    /// `folder` (inside the chosen folder), `conda` or `system`
    pub source: &'static str,
    /// a short name: the environment, or where in the folder it sits
    pub label: String,
}

const MAX_CANDIDATES: usize = 24;
/// Where checkouts and bundles keep their interpreter, relative to their folder.
const FOLDER_ENVS: &[&str] = &[
    "",
    "venv",
    ".venv",
    "env",
    "runtime",
    "workenv",
    "python",
    "py310",
    "py311",
    "python310",
    "python311",
    "python312",
    "installer_files/env",
];

fn exe_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["python.exe"]
    } else {
        &["python3", "python"]
    }
}

/// The interpreter of an environment (or install) folder, if it has one.
fn env_python(dir: &Path) -> Option<PathBuf> {
    let bins: &[&str] = if cfg!(windows) {
        &["", "Scripts"]
    } else {
        &["bin"]
    };
    for bin in bins {
        for exe in exe_names() {
            let p = if bin.is_empty() {
                dir.join(exe)
            } else {
                dir.join(bin).join(exe)
            };
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// Interpreters inside `root` and its direct subfolders (an unpacked bundle often wraps the
/// checkout in one more folder).
pub fn in_folder(root: &Path) -> Vec<PythonCandidate> {
    let mut out = Vec::new();
    let mut scan = |base: &Path, prefix: &str| {
        for env in FOLDER_ENVS {
            let dir = if env.is_empty() {
                base.to_path_buf()
            } else {
                base.join(env)
            };
            if let Some(path) = env_python(&dir) {
                let label = [prefix, env]
                    .iter()
                    .filter(|s| !s.is_empty())
                    .copied()
                    .collect::<Vec<_>>()
                    .join("/");
                out.push(PythonCandidate {
                    path,
                    source: "folder",
                    label: if label.is_empty() { ".".into() } else { label },
                });
            }
        }
    };
    scan(root, "");
    if let Ok(entries) = std::fs::read_dir(root) {
        let mut dirs: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                !name.starts_with('.') && !FOLDER_ENVS.contains(&name)
            })
            .collect();
        dirs.sort();
        for dir in dirs.into_iter().take(64) {
            let name = dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            scan(&dir, &name);
        }
    }
    dedupe(out)
}

/// Conda environments: `~/.conda/environments.txt` lists every one conda created, `envs_dirs`
/// in `~/.condarc` says where they go when moved off the system drive, and the usual install
/// folders (in the home folder, plus `extra_roots`) are looked at too.
fn conda(home: &Path, extra_roots: &[PathBuf]) -> Vec<PythonCandidate> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(text) = std::fs::read_to_string(home.join(".conda").join("environments.txt")) {
        dirs.extend(
            text.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(PathBuf::from),
        );
    }
    let mut roots: Vec<PathBuf> = [
        "miniconda3",
        "anaconda3",
        "miniforge3",
        "miniforge",
        "mambaforge",
        "micromamba",
        ".local/share/mamba",
    ]
    .iter()
    .map(|n| home.join(n))
    .collect();
    roots.extend(extra_roots.iter().cloned());
    for root in roots.into_iter().filter(|r| r.is_dir()) {
        dirs.push(root.clone());
        dirs.extend(subdirs(&root.join("envs")));
    }
    for envs in condarc_envs_dirs(home) {
        dirs.extend(subdirs(&envs));
    }
    dirs.iter()
        .filter_map(|dir| {
            env_python(dir).map(|path| PythonCandidate {
                path,
                source: "conda",
                label: dir
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            })
        })
        .collect()
}

fn subdirs(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| entries.filter_map(|e| e.ok()).map(|e| e.path()).collect())
        .unwrap_or_default();
    out.sort();
    out
}

/// The `envs_dirs` list of `~/.condarc` (block or `[a, b]` form; `~` is the home folder).
fn condarc_envs_dirs(home: &Path) -> Vec<PathBuf> {
    let Ok(text) = std::fs::read_to_string(home.join(".condarc")) else {
        return Vec::new();
    };
    let clean = |item: &str| -> Option<PathBuf> {
        let item = item.trim().trim_matches(|c| c == '"' || c == '\'');
        if item.is_empty() {
            return None;
        }
        Some(match item.strip_prefix('~') {
            Some(rest) => home.join(rest.trim_start_matches(&['/', '\\'][..])),
            None => PathBuf::from(item),
        })
    };
    let mut out = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        let content = line.split(" #").next().unwrap_or("").trim_end();
        if content.trim().is_empty() || content.trim_start().starts_with('#') {
            continue;
        }
        if !line.starts_with(&[' ', '\t', '-'][..]) {
            inside = false;
            if let Some(rest) = content.strip_prefix("envs_dirs:") {
                let rest = rest.trim();
                if let Some(list) = rest.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
                    out.extend(list.split(',').filter_map(clean));
                } else {
                    inside = rest.is_empty();
                }
            }
            continue;
        }
        if inside {
            if let Some(item) = content.trim_start().strip_prefix('-') {
                out.extend(clean(item));
            }
        }
    }
    out
}

/// Conda installs outside the home folder: for all users, at a drive's root (common when the
/// system drive is small), and the one a launching shell had active.
fn machine_conda_roots() -> Vec<PathBuf> {
    let names = ["miniconda3", "anaconda3", "miniforge3"];
    let mut roots: Vec<PathBuf> = Vec::new();
    for var in ["CONDA_PREFIX", "MAMBA_ROOT_PREFIX", "CONDA_ROOT"] {
        if let Some(dir) = std::env::var_os(var) {
            roots.push(PathBuf::from(dir));
        }
    }
    // <root>/Scripts/conda.exe or <root>/bin/conda
    if let Some(exe) = std::env::var_os("CONDA_EXE") {
        if let Some(root) = Path::new(&exe).parent().and_then(Path::parent) {
            roots.push(root.to_path_buf());
        }
    }
    if cfg!(windows) {
        for var in ["ProgramData", "LOCALAPPDATA"] {
            if let Some(base) = std::env::var_os(var) {
                roots.extend(names.map(|n| PathBuf::from(&base).join(n)));
            }
        }
        for drive in 'C'..='H' {
            let root = PathBuf::from(format!("{drive}:\\"));
            if root.is_dir() {
                roots.extend(names.map(|n| root.join(n)));
                roots.extend(names.map(|n| root.join("ProgramData").join(n)));
            }
        }
    } else {
        roots.extend(
            [
                "/opt/miniconda3",
                "/opt/anaconda3",
                "/opt/miniforge3",
                "/opt/conda",
                "/opt/homebrew/Caskroom/miniconda/base",
                "/opt/homebrew/Caskroom/miniforge/base",
            ]
            .map(PathBuf::from),
        );
    }
    roots
}

/// python.org / Homebrew / distribution installs and whatever is on PATH (3.10–3.12).
fn system() -> Vec<PythonCandidate> {
    let mut paths: Vec<PathBuf> = Vec::new();
    let versions = ["3.10", "3.11", "3.12"];
    if cfg!(windows) {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            for v in ["310", "311", "312"] {
                paths.push(
                    PathBuf::from(&local)
                        .join("Programs")
                        .join("Python")
                        .join(format!("Python{v}"))
                        .join("python.exe"),
                );
                paths.push(PathBuf::from(format!("C:\\Python{v}\\python.exe")));
            }
        }
    } else {
        for v in versions {
            for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"] {
                paths.push(PathBuf::from(dir).join(format!("python{v}")));
            }
            paths.push(PathBuf::from(format!(
                "/Library/Frameworks/Python.framework/Versions/{v}/bin/python3"
            )));
        }
    }
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            // the Microsoft Store stub opens the Store instead of running Python
            if dir.to_string_lossy().contains("WindowsApps") {
                continue;
            }
            let names: Vec<String> = if cfg!(windows) {
                vec!["python.exe".into()]
            } else {
                versions
                    .iter()
                    .map(|v| format!("python{v}"))
                    .chain(["python3".to_string()])
                    .collect()
            };
            for name in names {
                paths.push(dir.join(name));
            }
        }
    }
    paths
        .into_iter()
        .filter(|p| p.is_file())
        .map(|path| {
            let label = if cfg!(windows) {
                path.parent()
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            } else {
                path.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            };
            PythonCandidate {
                path,
                source: "system",
                label,
            }
        })
        .collect()
}

/// Interpreters registered under `Software\Python` (PEP 514): the Anaconda / Miniconda
/// installers and python.org put theirs there, wherever they were installed.
#[cfg(windows)]
fn registered() -> Vec<PythonCandidate> {
    use std::os::windows::process::CommandExt;

    let mut text = String::new();
    for key in ["HKCU\\Software\\Python", "HKLM\\Software\\Python"] {
        let out = std::process::Command::new("reg")
            .args(["query", key, "/s", "/v", "ExecutablePath"])
            .stdin(std::process::Stdio::null())
            .creation_flags(crate::procutil::CREATE_NO_WINDOW)
            .output();
        if let Ok(out) = out {
            text.push_str(&String::from_utf8_lossy(&out.stdout));
        }
    }
    registry_executables(&text)
        .into_iter()
        .filter(|p| p.is_file())
        .map(registered_candidate)
        .collect()
}

#[cfg(not(windows))]
fn registered() -> Vec<PythonCandidate> {
    Vec::new()
}

/// The `ExecutablePath` values in `reg query … /s /v ExecutablePath` output.
#[cfg_attr(not(windows), allow(dead_code))]
fn registry_executables(text: &str) -> Vec<PathBuf> {
    text.lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix("ExecutablePath")?;
            let (_, value) = rest
                .split_once("REG_SZ")
                .or_else(|| rest.split_once("REG_EXPAND_SZ"))?;
            let value = value.trim();
            (!value.is_empty()).then(|| PathBuf::from(value))
        })
        .collect()
}

/// A registered interpreter: a conda installation's base environment, or a plain install.
#[cfg_attr(not(windows), allow(dead_code))]
fn registered_candidate(path: PathBuf) -> PythonCandidate {
    let dir = path.parent().unwrap_or(Path::new(""));
    let label = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let source = if dir.join("conda-meta").is_dir() {
        "conda"
    } else {
        "system"
    };
    PythonCandidate {
        path,
        source,
        label,
    }
}

fn dedupe(list: Vec<PythonCandidate>) -> Vec<PythonCandidate> {
    // by the path as found: a venv's interpreter links to its base one, but they are different
    // environments, so resolving links would merge them
    let mut seen = HashSet::new();
    list.into_iter()
        .filter(|c| seen.insert(c.path.clone()))
        .take(MAX_CANDIDATES)
        .collect()
}

/// Candidates in `folder` when one is given, else conda environments, registered interpreters,
/// then system installs.
pub fn find(folder: Option<&Path>) -> Vec<PythonCandidate> {
    match folder {
        Some(root) => in_folder(root),
        None => {
            let mut out = dirs::home_dir()
                .map(|h| conda(&h, &machine_conda_roots()))
                .unwrap_or_default();
            out.extend(registered());
            out.extend(system());
            dedupe(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"").unwrap();
    }

    fn exe(dir: &Path) -> PathBuf {
        if cfg!(windows) {
            dir.join("python.exe")
        } else {
            dir.join("bin").join("python3")
        }
    }

    #[test]
    fn finds_bundle_and_checkout_interpreters() {
        let root = std::env::temp_dir().join(format!("stellaria-pyfind-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let bundle = root.join("DDSP-SVC 整合包");
        touch(&exe(&bundle.join("workenv")));
        let venv = if cfg!(windows) {
            bundle
                .join("DDSP-SVC")
                .join("venv")
                .join("Scripts")
                .join("python.exe")
        } else {
            bundle
                .join("DDSP-SVC")
                .join("venv")
                .join("bin")
                .join("python3")
        };
        touch(&venv);
        touch(&bundle.join("pretrain").join("rmvpe").join("model.pt"));

        let found = in_folder(&bundle);
        let labels: Vec<&str> = found.iter().map(|c| c.label.as_str()).collect();
        assert_eq!(labels, vec!["workenv", "DDSP-SVC/venv"], "{found:?}");
        assert!(found
            .iter()
            .all(|c| c.source == "folder" && c.path.is_file()));
        assert_eq!(find(Some(&bundle)), found);
        assert!(in_folder(&root.join("missing")).is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn registry_values() {
        let text = "\r\nHKEY_CURRENT_USER\\Software\\Python\\ContinuumAnalytics\\Anaconda39-64\\InstallPath\r\n    \
                    ExecutablePath    REG_SZ    D:\\Anaconda3\\python.exe\r\n\r\n\
                    HKEY_CURRENT_USER\\Software\\Python\\PythonCore\\3.11\\InstallPath\r\n    \
                    ExecutablePath    REG_SZ    C:\\Program Files\\Python311\\python.exe\r\n    \
                    ExecutablePath    REG_SZ    \r\n\r\n搜索结束: 找到 2 匹配。\r\n";
        assert_eq!(
            registry_executables(text),
            vec![
                PathBuf::from("D:\\Anaconda3\\python.exe"),
                PathBuf::from("C:\\Program Files\\Python311\\python.exe")
            ]
        );
        let base = std::env::temp_dir().join(format!("stellaria-reg-{}", std::process::id()));
        std::fs::create_dir_all(base.join("Anaconda3").join("conda-meta")).unwrap();
        let conda = registered_candidate(base.join("Anaconda3").join("python.exe"));
        assert_eq!((conda.source, conda.label.as_str()), ("conda", "Anaconda3"));
        let plain = registered_candidate(base.join("Python311").join("python.exe"));
        assert_eq!(
            (plain.source, plain.label.as_str()),
            ("system", "Python311")
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn conda_environments_from_the_list_and_install_folders() {
        let home = std::env::temp_dir().join(format!("stellaria-conda-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let base = home.join("miniconda3");
        touch(&exe(&base));
        touch(&exe(&base.join("envs").join("ddsp")));
        let elsewhere = home.join("work").join("svc-env");
        touch(&exe(&elsewhere));
        std::fs::create_dir_all(home.join(".conda")).unwrap();
        std::fs::write(
            home.join(".conda").join("environments.txt"),
            format!("{}\n\n", elsewhere.display()),
        )
        .unwrap();

        let labels: Vec<String> = dedupe(conda(&home, &[]))
            .into_iter()
            .map(|c| c.label)
            .collect();
        assert_eq!(labels, vec!["svc-env", "miniconda3", "ddsp"]);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn conda_envs_moved_by_condarc_and_installs_elsewhere() {
        let home = std::env::temp_dir().join(format!("stellaria-condarc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let moved = home.join("data").join("conda envs");
        touch(&exe(&moved.join("svc")));
        let tilde = home.join("envs2");
        touch(&exe(&tilde.join("old")));
        std::fs::write(
            home.join(".condarc"),
            format!(
                "channels:\n  - defaults\nenvs_dirs:\n  - \"{}\"  # moved off C:\n  - ~/envs2\nshow_channel_urls: true\n",
                moved.display()
            ),
        )
        .unwrap();
        let elsewhere = home.join("drive").join("anaconda3");
        touch(&exe(&elsewhere));
        touch(&exe(&elsewhere.join("envs").join("ddsp6")));

        let labels: Vec<String> = dedupe(conda(&home, std::slice::from_ref(&elsewhere)))
            .into_iter()
            .map(|c| c.label)
            .collect();
        assert_eq!(labels, vec!["anaconda3", "ddsp6", "svc", "old"]);

        std::fs::write(home.join(".condarc"), "envs_dirs: [~/envs2]\n").unwrap();
        assert_eq!(condarc_envs_dirs(&home), vec![home.join("envs2")]);
        let _ = std::fs::remove_dir_all(&home);
    }
}

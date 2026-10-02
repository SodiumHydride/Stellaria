//! `install_runtime_requirements`: the app's packages, **except torch**, for the user's own
//! interpreter with the bundled uv — same mirrors and fallbacks as the bootstrap, progress as
//! a single `packages` stage on `bootstrap://progress`.
//!
//! They go into Stellaria's layer over that interpreter (`crate::layer`): a dry run against
//! the user's environment says exactly what it lacks (their versions are kept), and only that
//! is installed into the layer, without dependencies. Their environment is only ever read;
//! when a layer cannot work, the answer is the app's own runtime, not changing theirs.

use std::path::{Path, PathBuf};

use tauri::AppHandle;
use tokio_util::sync::CancellationToken;

use super::pipeline::{pypi_sources, uv_command, DEFAULT_PYPI};
use super::progress::{self, Reporter, Stage};
use super::sources::{self, SourceList};
use super::tool::{run_tool, tool_error, Failures, Observer, PipObserver};
use super::uvlog::{DirSampler, PipTracker};
use super::{Outcome, StageError, StageResult};
use crate::error::ShellError;
use crate::layer;
use crate::manifest::{Manifest, Source};
use crate::types::Region;
use crate::Shell;

/// Overrides with a never-true marker take torch out of the resolution entirely, so packages
/// that depend on it (torchcrepe, torchfcpe) neither pull a PyTorch build nor touch the one
/// the user has.
const NO_TORCH_OVERRIDES: &str = "torch ; sys_platform == 'never'\n\
                                  torchaudio ; sys_platform == 'never'\n";

/// Runs the job. Returns the reporter, the outcome and whether the backend was stopped for it.
pub(super) async fn run(
    app: &AppHandle,
    python: PathBuf,
    region: Region,
    cancel: CancellationToken,
) -> (Reporter, Outcome, bool) {
    let shell = crate::shell(app);
    let mut rep = Reporter::new(
        progress::event_sink(app.clone()),
        progress::single_stage(Stage::Packages),
        shell.paths.logs_dir.join("bootstrap.log"),
    );
    rep.log(&format!(
        "=== install runtime requirements for {} (layer{}, region {region:?})",
        python.display(),
        if layer::is_embeddable(&python) {
            ", shadow"
        } else {
            ""
        }
    ));
    rep.begin_stage(
        Stage::Packages,
        "Adding the missing packages next to your Python",
    );

    // Never change an environment under a running backend (Windows locks its files).
    let target = layer::interpreter(&shell.paths, &python);
    let stopped = shell
        .backend
        .current_python()
        .is_some_and(|current| same_interpreter(&current, &target));
    if stopped {
        rep.set_detail("Stopping the backend…");
        rep.emit(true);
        shell.backend.stop(app, true).await;
    }

    // what the interpreter offered before (the check the UI showed): no torch there means a
    // check without torch afterwards is not the layer's doing
    let had_torch = shell
        .cached_check(&python)
        .is_none_or(|c| c.torch.is_some());
    let result = install_layered(&mut rep, &shell, &python, region, &cancel).await;
    let outcome = match result {
        Ok(()) => {
            // Refresh the cached check (runtime status and versions in ShellState).
            rep.set_detail("Checking your Python…");
            rep.set_progress(1.0);
            rep.emit(true);
            let check = shell.check_python(&python).await;
            if had_torch && !check.ok && check.torch.is_none() {
                // The layer itself broke the environment (a DLL the base needs, say): drop it so
                // the user's interpreter is checked as it is, and say so.
                layer::remove(&shell.paths).await;
                shell.check_python(&python).await;
                let reason = check
                    .error
                    .as_ref()
                    .and_then(|e| serde_json::to_string(e).ok())
                    .unwrap_or_default();
                Outcome::Failed(
                    ShellError::new(
                        "runtime.layer_failed",
                        "Your Python did not work with the app's layer",
                    )
                    .with_detail(reason),
                )
            } else {
                Outcome::Done("Packages installed".into())
            }
        }
        Err(err) => err.into(),
    };
    (rep, outcome, stopped)
}

struct Prepared {
    requirements: PathBuf,
    overrides: PathBuf,
    mirrors: Vec<Source>,
}

/// The requirements file, the no-torch overrides and the mirrors to try, fastest first.
async fn prepare(
    rep: &mut Reporter,
    shell: &Shell,
    python: &Path,
    region: Region,
    cancel: &CancellationToken,
) -> StageResult<Prepared> {
    if !python.is_file() {
        return Err(ShellError::new(
            "runtime.custom_invalid",
            "The Python executable was not found",
        )
        .with_detail(python.display().to_string())
        .into());
    }
    let manifest = Manifest::load(&shell.resources.manifest)?;
    let requirements = shell.resources.requirements(&manifest.runtime.requirements);
    if !requirements.is_file() {
        return Err(ShellError::new(
            "bootstrap.requirements_missing",
            "The package list of the app is missing",
        )
        .with_detail(requirements.display().to_string())
        .into());
    }
    let paths = &shell.paths;
    for dir in [&paths.runtime_dir, &paths.uv_cache_dir] {
        std::fs::create_dir_all(dir)
            .map_err(|e| ShellError::io(format!("Cannot create {}", dir.display()), &e))?;
    }
    let overrides = paths.runtime_dir.join("overrides-no-torch.txt");
    crate::fsutil::write_atomic(&overrides, NO_TORCH_OVERRIDES.as_bytes())
        .map_err(|e| ShellError::io("Cannot write the overrides file", &e))?;

    let candidates = pypi_sources(&manifest);
    let latencies = if region == Region::Auto {
        rep.set_detail("Measuring mirror speed…");
        rep.emit(true);
        let lists = [(SourceList::Pypi, candidates.as_slice())];
        tokio::select! {
            _ = cancel.cancelled() => return Err(StageError::Cancelled),
            l = sources::probe_all(&shell.net_http, &lists) => l,
        }
    } else {
        Vec::new()
    };
    Ok(Prepared {
        requirements,
        overrides,
        mirrors: sources::order(&candidates, region, SourceList::Pypi, &latencies),
    })
}

fn pip_observer(shell: &Shell) -> PipObserver {
    PipObserver {
        tracker: PipTracker::new(Some(DirSampler::start(shell.paths.uv_cache_dir.clone()))),
        range: (0.0, 1.0),
    }
}

/// Collects the `+ name==version` lines of a dry-run plan.
#[derive(Default)]
struct PlanObserver {
    plan: Vec<String>,
}

impl Observer for PlanObserver {
    fn line(&mut self, rep: &mut Reporter, line: &str) {
        if let Some(item) = layer::planned(line) {
            self.plan.push(item);
        }
        rep.set_detail(line.trim());
        rep.emit(false);
    }

    fn tick(&mut self, rep: &mut Reporter) {
        rep.emit(false);
    }
}

/// Into Stellaria's layer over the interpreter: only what its environment lacks.
async fn install_layered(
    rep: &mut Reporter,
    shell: &Shell,
    base: &Path,
    region: Region,
    cancel: &CancellationToken,
) -> StageResult<()> {
    let prepared = prepare(rep, shell, base, region, cancel).await?;
    let paths = &shell.paths;

    rep.set_detail("Reading your Python environment…");
    rep.emit(true);
    let info = tokio::select! {
        _ = cancel.cancelled() => return Err(StageError::Cancelled),
        info = layer::query(base) => info?,
    };
    let text = std::fs::read_to_string(&prepared.requirements)
        .map_err(|e| ShellError::io("Cannot read the package list of the app", &e))?;
    let relaxed = paths.runtime_dir.join("requirements-layer.txt");
    crate::fsutil::write_atomic(
        &relaxed,
        layer::relaxed_requirements(&text, &info.installed).as_bytes(),
    )
    .map_err(|e| ShellError::io("Cannot write the package list", &e))?;

    rep.set_detail("Preparing the app's own folder…");
    rep.emit(true);
    let layer_python = layer::create(paths, Some(&shell.resources.uv), base, &info).await?;
    let plan_file = paths.runtime_dir.join("layer-install.txt");

    let mut failures = Failures::default();
    for (attempt, mirror) in prepared.mirrors.iter().enumerate() {
        let url = mirror.url.clone().unwrap_or_else(|| DEFAULT_PYPI.into());
        rep.reset_progress();
        rep.set_source(Some(mirror.id.clone()));
        rep.set_detail(if attempt > 0 {
            format!("Trying another mirror: {}", mirror.id)
        } else {
            "Working out what is missing…".to_string()
        });
        rep.emit(true);

        // What installing into the user's environment would do — without doing it.
        let mut cmd = uv_command(paths, &shell.resources.uv);
        cmd.args([
            "pip",
            "install",
            "--dry-run",
            "--break-system-packages",
            "--python",
        ])
        .arg(base)
        .args(["--index-url", &url])
        .arg("--overrides")
        .arg(&prepared.overrides)
        .arg("--requirements")
        .arg(&relaxed)
        .env("UV_PYTHON_DOWNLOADS", "never");
        let mut observer = PlanObserver::default();
        let out = run_tool(cancel, cmd, rep, &mut observer, None).await?;
        if !out.success {
            failures.push(
                &mirror.id,
                tool_error(
                    "runtime.install_failed",
                    "Could not work out the missing packages",
                    &out,
                ),
            )?;
            continue;
        }
        if observer.plan.is_empty() {
            rep.log("nothing missing: the layer only bridges to the environment");
            return Ok(());
        }
        rep.log(&format!("into the layer: {}", observer.plan.join(" ")));
        crate::fsutil::write_atomic(&plan_file, (observer.plan.join("\n") + "\n").as_bytes())
            .map_err(|e| ShellError::io("Cannot write the package list", &e))?;

        let mut cmd = uv_command(paths, &shell.resources.uv);
        cmd.args(["pip", "install", "--no-deps", "--python"])
            .arg(&layer_python)
            .args(["--index-url", &url])
            .arg("--requirements")
            .arg(&plan_file)
            .env("UV_PYTHON_DOWNLOADS", "never");
        let out = run_tool(cancel, cmd, rep, &mut pip_observer(shell), None).await?;
        if out.success {
            return Ok(());
        }
        failures.push(
            &mirror.id,
            tool_error(
                "runtime.install_failed",
                "Could not install the missing packages",
                &out,
            ),
        )?;
    }
    Err(failures.into_error(
        "runtime.install_failed",
        "Could not install the missing packages",
    ))
}

/// Same environment? Directories are resolved, the executable itself is not: a venv's
/// `bin/python` is usually a symlink to the base interpreter it shares with other venvs.
fn same_interpreter(a: &Path, b: &Path) -> bool {
    let key = |p: &Path| {
        let dir = p
            .parent()
            .map(|d| d.canonicalize().unwrap_or_else(|_| d.to_path_buf()));
        (dir, p.file_name().map(|n| n.to_os_string()))
    };
    key(a) == key(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_exclude_torch() {
        let lines: Vec<&str> = NO_TORCH_OVERRIDES.lines().map(str::trim).collect();
        assert_eq!(
            lines,
            [
                "torch ; sys_platform == 'never'",
                "torchaudio ; sys_platform == 'never'"
            ]
        );
    }

    /// A shell for the install tests below: data under `root`, `uv` from PATH.
    fn test_shell(root: &Path, manifest: PathBuf, backend: PathBuf) -> Shell {
        use std::sync::atomic::AtomicBool;
        use std::sync::Mutex;

        use crate::paths::{find_in_path, AppPaths, ResourcePaths};

        let stellaria = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let paths = AppPaths::from_data_dir(root.join("data"), root.join("library"));
        Shell {
            backend: crate::backend::BackendManager::new(&paths, reqwest::Client::new()),
            bootstrap: super::super::BootstrapManager::default(),
            frontend: crate::window::FrontendState::default(),
            quitting: AtomicBool::new(false),
            python_checks: Mutex::new(std::collections::HashMap::new()),
            net_http: reqwest::Client::new(),
            settings: crate::settings::SettingsStore::load(paths.settings_file.clone()),
            resources: ResourcePaths {
                engine: stellaria.join(".."),
                backend,
                manifest,
                uv: find_in_path("uv").expect("uv on PATH"),
                dev_python: None,
                dev: true,
            },
            paths,
        }
    }

    fn printing_reporter(shell: &Shell) -> Reporter {
        let mut rep = Reporter::new(
            Box::new(|p| {
                println!(
                    "[{:?}] {:>5.1}% | {} | {:?}/{:?} | {:?}",
                    p.stage,
                    p.overall.unwrap_or(0.0) * 100.0,
                    p.detail.as_deref().unwrap_or(""),
                    p.bytes_done,
                    p.bytes_total,
                    p.error
                )
            }),
            progress::single_stage(Stage::Packages),
            shell.paths.logs_dir.join("bootstrap.log"),
        );
        rep.begin_stage(Stage::Packages, "Installing packages");
        rep
    }

    /// Real install for a fresh venv — into Stellaria's layer over it (needs network and `uv` on
    /// PATH):
    /// `STELLARIA_E2E_DIR=<dir> cargo test -- --ignored requirements_into_fresh_venv --nocapture`
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads the runtime requirements"]
    async fn requirements_into_fresh_venv() {
        let root = std::env::var_os("STELLARIA_E2E_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("stellaria-req-e2e"));
        let stellaria = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let shell = test_shell(
            &root,
            stellaria.join("resources/manifest.json"),
            stellaria.join("backend"),
        );
        let venv = root.join("user-venv");
        let status = std::process::Command::new(&shell.resources.uv)
            .args(["venv", "--quiet", "--python", "3.11"])
            .arg(&venv)
            .status()
            .unwrap();
        assert!(status.success());
        let python = crate::paths::venv_python(&venv);
        let mut rep = printing_reporter(&shell);
        let result = install_layered(
            &mut rep,
            &shell,
            &python,
            Region::Global,
            &CancellationToken::new(),
        )
        .await;
        if let Err(StageError::Failed(err)) = &result {
            panic!("install failed: {err:?}");
        }
        assert!(result.is_ok());
        let check = shell.check_python(&python).await;
        println!("{}", serde_json::to_string_pretty(&check).unwrap());
        assert!(check.missing.is_empty(), "{check:?}");
        assert!(
            check.torch.is_none(),
            "torch must not be installed: {check:?}"
        );
        assert_eq!(
            check.error.as_ref().map(|e| e.code.as_str()),
            Some("python.torch_missing")
        );
        assert_eq!(
            shell.cached_check(&python).map(|c| c.ok),
            Some(false),
            "the check is cached"
        );
    }

    /// Every file under `dir` with its size and modification time.
    fn snapshot(dir: &Path) -> Vec<(PathBuf, u64, std::time::SystemTime)> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                let meta = e.metadata().unwrap();
                if meta.is_dir() {
                    stack.push(e.path());
                } else {
                    out.push((e.path(), meta.len(), meta.modified().unwrap()));
                }
            }
        }
        out.sort();
        out
    }

    /// The layer gets only what the user's Python lacks, keeps their versions, and their
    /// environment is not written to. Needs network, `uv` on PATH and a Python with numpy and
    /// six installed (not a venv):
    /// `STELLARIA_LAYER_BASE=/path/to/python cargo test -- --ignored layered_install --nocapture`
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads packages"]
    async fn layered_install_leaves_the_base_alone() {
        let base = PathBuf::from(std::env::var_os("STELLARIA_LAYER_BASE").expect("base"));
        let root = std::env::temp_dir().join(format!("stellaria-layered-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let stage = root.join("stage");
        std::fs::create_dir_all(stage.join("backend")).unwrap();
        std::fs::write(
            stage.join("req.txt"),
            "numpy==2.0.0\nsix==1.17.0\ntomli-w==1.0.0\nwcwidth==0.2.13\n",
        )
        .unwrap();
        let stellaria = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(stellaria.join("resources/manifest.json")).unwrap(),
        )
        .unwrap();
        manifest["runtime"]["requirements"] = "req.txt".into();
        std::fs::write(
            stage.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let shell = test_shell(&root, stage.join("manifest.json"), stage.join("backend"));

        let base_env = base.parent().unwrap().parent().unwrap().to_path_buf();
        let before = snapshot(&base_env);
        let mut rep = printing_reporter(&shell);
        let result = install_layered(
            &mut rep,
            &shell,
            &base,
            Region::Global,
            &CancellationToken::new(),
        )
        .await;
        if let Err(StageError::Failed(err)) = &result {
            panic!("install failed: {err:?}");
        }
        assert!(result.is_ok());
        assert_eq!(
            snapshot(&base_env),
            before,
            "the user's environment changed"
        );

        let plan =
            std::fs::read_to_string(shell.paths.runtime_dir.join("layer-install.txt")).unwrap();
        assert_eq!(plan, "tomli-w==1.0.0\nwcwidth==0.2.13\n");
        let layer_python = layer::existing(&shell.paths, &base).expect("a layer over the base");
        let out = std::process::Command::new(&layer_python)
            .args([
                "-c",
                "import numpy, six, tomli_w, wcwidth; \
                 print(numpy.__version__, six.__version__, numpy.__file__, tomli_w.__file__)",
            ])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        println!("{text}{}", String::from_utf8_lossy(&out.stderr));
        let parts: Vec<&str> = text.split_whitespace().collect();
        assert_eq!(parts[0], "1.26.4", "the user's numpy is kept");
        assert_eq!(parts[1], "1.16.0", "the user's six is kept");
        assert!(parts[2].starts_with(&*base_env.to_string_lossy()));
        assert!(parts[3].starts_with(&*shell.paths.layer_dir.to_string_lossy()));

        let check = shell.check_python(&base).await;
        assert!(check.layered, "{check:?}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn venvs_sharing_a_base_interpreter_differ() {
        let dir = std::env::temp_dir().join(format!("stellaria-same-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let base = dir.join("base").join("python3");
        std::fs::create_dir_all(base.parent().unwrap()).unwrap();
        std::fs::write(&base, b"").unwrap();
        for venv in ["a", "b"] {
            let bin = dir.join(venv).join("bin");
            std::fs::create_dir_all(&bin).unwrap();
            std::os::unix::fs::symlink(&base, bin.join("python")).unwrap();
        }
        let a = dir.join("a/bin/python");
        assert!(!same_interpreter(&a, &dir.join("b/bin/python")));
        assert!(same_interpreter(&a, &dir.join("a/./bin/python")));
        // A symlinked environment directory is the same environment.
        std::os::unix::fs::symlink(dir.join("a"), dir.join("link")).unwrap();
        assert!(same_interpreter(&a, &dir.join("link/bin/python")));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

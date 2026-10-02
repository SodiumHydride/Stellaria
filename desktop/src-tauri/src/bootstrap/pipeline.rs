//! The stages: prepare → python → venv → torch → packages → verify.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use super::progress::{Reporter, Stage};
use super::sources::{self, SourceList};
use super::tool::{
    run_tool, tool_error, CreepObserver, Failures, PipObserver, PythonObserver, ToolOutput,
};
use super::uvlog::{DirSampler, PipTracker, PythonTracker};
use super::{wheel, Mode, Request, StageError, StageResult};
use crate::error::{ShellError, ShellResult};
use crate::fsutil;
use crate::manifest::{Manifest, Source, SourceKind, VariantManifest};
use crate::paths::{AppPaths, ResourcePaths};
use crate::procutil::{hidden_command, run_capture};
use crate::runtime::{apply_python_env, RuntimeState};
use crate::types::{Region, TorchVariant};

pub(super) const DEFAULT_PYPI: &str = "https://pypi.org/simple";
const GIB: u64 = 1024 * 1024 * 1024;

pub(super) struct Pipeline {
    paths: AppPaths,
    resources: ResourcePaths,
    manifest: Manifest,
    request: Request,
    cancel: CancellationToken,
    net: reqwest::Client,
    python_mirrors: Vec<Source>,
    pypi: Vec<Source>,
    torch_sources: Vec<Source>,
    /// The torch source that worked (reused for the packages stage).
    torch_source: Option<Source>,
    /// Index into `pypi` of the mirror that worked last.
    pypi_index: usize,
    state_invalidated: bool,
}

impl Pipeline {
    pub fn new(
        paths: AppPaths,
        resources: ResourcePaths,
        manifest: Manifest,
        request: Request,
        cancel: CancellationToken,
        net: reqwest::Client,
    ) -> Self {
        Self {
            paths,
            resources,
            manifest,
            request,
            cancel,
            net,
            python_mirrors: Vec::new(),
            pypi: Vec::new(),
            torch_sources: Vec::new(),
            torch_source: None,
            pypi_index: 0,
            state_invalidated: false,
        }
    }

    pub async fn run(&mut self, rep: &mut Reporter) -> StageResult<RuntimeState> {
        self.stage_prepare(rep).await?;
        self.stage_python(rep).await?;
        self.stage_venv(rep).await?;
        self.stage_torch(rep).await?;
        self.stage_packages(rep).await?;
        self.stage_verify(rep).await
    }

    fn variant(&self) -> TorchVariant {
        self.request.variant
    }

    fn variant_manifest(&self) -> StageResult<&VariantManifest> {
        Ok(self.manifest.variant(self.variant())?)
    }

    fn python_version(&self) -> &str {
        &self.manifest.runtime.python
    }

    /// `2.9.1` → `2.9.1+cu128` for PyTorch's own indexes and their mirrors, which label
    /// their builds; PyPI wheels never carry a local label.
    fn version_for(&self, version: &str, source: &Source) -> String {
        match (source.kind, self.variant().local_label()) {
            (Some(SourceKind::Index | SourceKind::FindLinks), Some(label))
                if !version.contains('+') =>
            {
                format!("{version}+{label}")
            }
            _ => version.to_string(),
        }
    }

    fn pypi_url(&self, index: usize) -> String {
        self.pypi
            .get(index % self.pypi.len().max(1))
            .and_then(|s| s.url.clone())
            .unwrap_or_else(|| DEFAULT_PYPI.to_string())
    }

    fn pypi_id(&self, index: usize) -> String {
        self.pypi
            .get(index % self.pypi.len().max(1))
            .map(|s| s.id.clone())
            .unwrap_or_else(|| "pypi".into())
    }

    fn requirements_path(&self) -> PathBuf {
        self.resources
            .requirements(&self.manifest.runtime.requirements)
    }

    /// Removes `state.json` right before the first change to the venv, so an interrupted
    /// install never looks "ready".
    fn invalidate_state(&mut self) {
        if !self.state_invalidated {
            self.state_invalidated = true;
            let _ = std::fs::remove_file(&self.paths.state_file);
        }
    }

    fn uv(&self) -> Command {
        uv_command(&self.paths, &self.resources.uv)
    }

    fn venv_python_cmd(&self, code: &str) -> Command {
        let mut cmd = hidden_command(self.paths.venv_python());
        cmd.args(["-c", code]).current_dir(&self.paths.runtime_dir);
        apply_python_env(&mut cmd, &self.paths);
        cmd
    }

    async fn tool(
        &self,
        cmd: Command,
        rep: &mut Reporter,
        observer: &mut dyn super::tool::Observer,
        timeout: Option<Duration>,
    ) -> StageResult<ToolOutput> {
        run_tool(&self.cancel, cmd, rep, observer, timeout).await
    }

    // ------------------------------------------------------------------ prepare

    async fn stage_prepare(&mut self, rep: &mut Reporter) -> StageResult<()> {
        rep.begin_stage(Stage::Prepare, "Preparing");
        check_platform(self.variant())?;
        let variant_manifest = self.variant_manifest()?.clone();

        for dir in [
            &self.paths.runtime_dir,
            &self.paths.python_dir,
            &self.paths.uv_cache_dir,
            &self.paths.downloads_dir,
            &self.paths.logs_dir,
        ] {
            std::fs::create_dir_all(dir)
                .map_err(|e| ShellError::io(format!("Cannot create {}", dir.display()), &e))?;
        }

        rep.set_detail("Checking the installer tool…");
        rep.set_progress(0.05);
        rep.emit(true);
        let mut cmd = self.uv();
        cmd.arg("--version");
        let uv_missing = |detail: String| {
            ShellError::new(
                "bootstrap.uv_missing",
                "The bundled installer (uv) is missing or broken",
            )
            .with_detail(format!("{}: {detail}", self.resources.uv.display()))
        };
        let version = match run_capture(cmd, Duration::from_secs(30)).await {
            Ok(out) if out.success => out.stdout,
            Ok(out) => return Err(uv_missing(out.stderr).into()),
            Err(err) => return Err(uv_missing(err.to_string()).into()),
        };
        rep.log(version.trim());

        let requirements = self.requirements_path();
        if !requirements.is_file() {
            return Err(ShellError::new(
                "bootstrap.requirements_missing",
                "The package list of the app is missing",
            )
            .with_detail(requirements.display().to_string())
            .into());
        }

        let needed = if self.paths.venv_python().is_file() {
            3 * GIB
        } else if self.variant().is_cuda() {
            12 * GIB
        } else {
            4 * GIB
        };
        if let Ok(space) = fsutil::disk_space(&self.paths.runtime_dir) {
            if space.free < needed {
                return Err(ShellError::new(
                    super::tool::DISK_FULL,
                    format!(
                        "Not enough free disk space: {:.1} GB needed, {:.1} GB available",
                        needed as f64 / GIB as f64,
                        space.free as f64 / GIB as f64
                    ),
                )
                .with_detail(self.paths.runtime_dir.display().to_string())
                .into());
            }
        }
        rep.set_progress(0.2);

        let python_mirrors = non_empty_or(
            &self.manifest.runtime.python_install_mirrors,
            Source {
                id: "github".into(),
                region: "global".into(),
                url: None,
                kind: None,
            },
        );
        let pypi = pypi_sources(&self.manifest);
        let torch_sources = variant_manifest.torch_sources.clone();
        if torch_sources.is_empty() {
            return Err(ShellError::new(
                "manifest.invalid",
                "The manifest lists no PyTorch source for this variant",
            )
            .into());
        }

        let latencies = if self.request.region == Region::Auto {
            rep.set_detail("Measuring mirror speed…");
            rep.emit(true);
            let lists = [
                (SourceList::PythonInstall, python_mirrors.as_slice()),
                (SourceList::Pypi, pypi.as_slice()),
                (SourceList::Torch, torch_sources.as_slice()),
            ];
            tokio::select! {
                _ = self.cancel.cancelled() => return Err(StageError::Cancelled),
                l = sources::probe_all(&self.net, &lists) => l,
            }
        } else {
            Vec::new()
        };
        let region = self.request.region;
        self.python_mirrors = sources::order(
            &python_mirrors,
            region,
            SourceList::PythonInstall,
            &latencies,
        );
        self.pypi = sources::order(&pypi, region, SourceList::Pypi, &latencies);
        self.torch_sources = sources::order(&torch_sources, region, SourceList::Torch, &latencies);
        if !latencies.is_empty() {
            for list in [
                SourceList::PythonInstall,
                SourceList::Pypi,
                SourceList::Torch,
            ] {
                rep.log(&format!(
                    "latency {list:?}: {}",
                    sources::summary(&latencies, list)
                ));
            }
            rep.set_detail(sources::summary(&latencies, SourceList::Pypi));
        }
        rep.log(&format!(
            "order: python {:?}, pypi {:?}, torch {:?}",
            ids(&self.python_mirrors),
            ids(&self.pypi),
            ids(&self.torch_sources)
        ));
        rep.set_progress(1.0);
        rep.emit(true);
        Ok(())
    }

    // ------------------------------------------------------------------ python

    async fn stage_python(&mut self, rep: &mut Reporter) -> StageResult<()> {
        let version = self.python_version().to_string();
        rep.begin_stage(Stage::Python, format!("Installing Python {version}"));
        let mirrors = self.python_mirrors.clone();
        let mut failures = Failures::default();
        for (attempt, mirror) in mirrors.iter().enumerate() {
            rep.reset_progress();
            rep.set_source(Some(mirror.id.clone()));
            if attempt > 0 {
                rep.set_detail(format!("Trying another mirror: {}", mirror.id));
            }
            rep.emit(true);
            let mut cmd = self.uv();
            cmd.args(["python", "install", &version, "--no-bin", "--no-registry"]);
            if let Some(url) = &mirror.url {
                cmd.env("UV_PYTHON_INSTALL_MIRROR", url);
            }
            let mut observer = PythonObserver {
                tracker: PythonTracker::new(DirSampler::start(self.paths.python_dir.clone())),
            };
            let out = self.tool(cmd, rep, &mut observer, None).await?;
            if out.success {
                return Ok(());
            }
            failures.push(
                &mirror.id,
                tool_error("bootstrap.python_failed", "Could not install Python", &out),
            )?;
        }
        Err(failures.into_error("bootstrap.python_failed", "Could not install Python"))
    }

    // ------------------------------------------------------------------ venv

    async fn stage_venv(&mut self, rep: &mut Reporter) -> StageResult<()> {
        rep.begin_stage(Stage::Venv, "Creating the Python environment");
        let python = self.paths.venv_python();
        if python.is_file() {
            let cmd = self.venv_python_cmd("import sys; print('%d.%d' % sys.version_info[:2])");
            let healthy = run_capture(cmd, Duration::from_secs(60))
                .await
                .ok()
                .filter(|o| o.success)
                .is_some_and(|o| o.stdout.trim() == self.python_version());
            if healthy {
                rep.set_detail("Reusing the existing environment");
                rep.set_progress(1.0);
                rep.emit(true);
                return Ok(());
            }
        }

        self.invalidate_state();
        if self.paths.venv_dir.exists() {
            rep.set_detail("Removing the old environment…");
            rep.emit(true);
            let dir = self.paths.venv_dir.clone();
            tokio::task::spawn_blocking(move || fsutil::remove_dir_all_retry(&dir))
                .await
                .map_err(ShellError::internal)?
                .map_err(|e| {
                    ShellError::new(
                        "bootstrap.venv_locked",
                        "The old Python environment is in use and cannot be replaced",
                    )
                    .with_detail(e.to_string())
                })?;
        }
        let mut cmd = self.uv();
        cmd.arg("venv")
            .arg(&self.paths.venv_dir)
            .args(["--python", self.python_version(), "--no-project"])
            .env("UV_PYTHON_DOWNLOADS", "never");
        let mut observer = CreepObserver::new(5.0, (0.0, 1.0));
        let out = self
            .tool(cmd, rep, &mut observer, Some(Duration::from_secs(300)))
            .await?;
        if !out.success {
            return Err(tool_error(
                "bootstrap.venv_failed",
                "Could not create the Python environment",
                &out,
            )
            .into());
        }
        Ok(())
    }

    // ------------------------------------------------------------------ torch

    async fn installed_versions(&self) -> HashMap<String, String> {
        const CODE: &str = "import importlib.metadata as m, json\n\
            out = {}\n\
            for n in ('torch', 'torchaudio'):\n    \
                try: out[n] = m.version(n)\n    \
                except Exception: pass\n\
            print(json.dumps(out))";
        let cmd = self.venv_python_cmd(CODE);
        run_capture(cmd, Duration::from_secs(60))
            .await
            .ok()
            .filter(|o| o.success)
            .and_then(|o| serde_json::from_str(o.stdout.trim()).ok())
            .unwrap_or_default()
    }

    async fn torch_imports(&self) -> bool {
        let cmd = self.venv_python_cmd("import torch, torchaudio");
        run_capture(cmd, Duration::from_secs(300))
            .await
            .is_ok_and(|o| o.success)
    }

    async fn stage_torch(&mut self, rep: &mut Reporter) -> StageResult<()> {
        rep.begin_stage(Stage::Torch, "Installing PyTorch");
        let (torch, torchaudio) = (
            self.manifest.runtime.torch.clone(),
            self.manifest.runtime.torchaudio.clone(),
        );

        rep.set_detail("Checking the installed PyTorch…");
        rep.emit(true);
        let installed = self.installed_versions().await;
        let installed_from = self.torch_sources.iter().find(|source| {
            installed.get("torch") == Some(&self.version_for(&torch, source))
                && installed.get("torchaudio") == Some(&self.version_for(&torchaudio, source))
        });
        let mut reinstall = false;
        if let Some(source) = installed_from.cloned() {
            if self.request.mode == Mode::Install || self.torch_imports().await {
                rep.set_detail(format!(
                    "PyTorch {} is already installed",
                    self.version_for(&torch, &source)
                ));
                rep.set_progress(1.0);
                rep.emit(true);
                self.torch_source = Some(source);
                return Ok(());
            }
            reinstall = true;
        }

        let sources = self.torch_sources.clone();
        let attempts = sources.len().max(self.pypi.len()).max(1);
        let py_tag = wheel::python_tag(self.python_version());
        let mut failures = Failures::default();
        for attempt in 0..attempts {
            let source = sources[attempt % sources.len()].clone();
            let pypi_url = self.pypi_url(attempt);
            let label = match source.kind {
                Some(SourceKind::Pypi) | None => self.pypi_id(attempt),
                _ => source.id.clone(),
            };
            let versions = (
                self.version_for(&torch, &source),
                self.version_for(&torchaudio, &source),
            );
            rep.reset_progress();
            rep.set_source(Some(label.clone()));
            if attempt > 0 {
                rep.set_detail(format!("Trying another mirror: {label}"));
            }
            rep.emit(true);

            // 1. Our own resumable download of the big wheel.
            let kind = source.kind.unwrap_or(SourceKind::Pypi);
            let listing_base = match kind {
                SourceKind::Pypi => Some(pypi_url.clone()),
                _ => source.url.clone(),
            };
            let local_wheel = match listing_base {
                Some(base) => {
                    let page = wheel::listing_url(kind, &base, "torch");
                    match self.fetch_wheel(rep, &page, &versions.0, &py_tag).await {
                        Ok(path) => Some(path),
                        Err(StageError::Cancelled) => return Err(StageError::Cancelled),
                        Err(StageError::Failed(err)) if err.code == super::tool::DISK_FULL => {
                            return Err(err.into())
                        }
                        Err(StageError::Failed(err)) => {
                            // uv still gets its chance: it may reach hosts we cannot
                            // (e.g. proxy settings only it understands).
                            rep.log(&format!(
                                "direct download unavailable ({}): {}",
                                err.message, err.detail
                            ));
                            None
                        }
                    }
                }
                None => None,
            };

            // 2. uv installs it plus torchaudio and the dependencies.
            let range = if local_wheel.is_some() {
                (0.8, 1.0)
            } else {
                (0.0, 1.0)
            };
            let install = TorchInstall {
                source: &source,
                pypi_url: &pypi_url,
                versions: &versions,
                reinstall,
            };
            let mut out = self
                .uv_install_torch(rep, &install, local_wheel.as_ref(), range)
                .await?;
            if !out.success
                && local_wheel.is_some()
                && out.tail_text().contains("not supported on this platform")
            {
                rep.log("local wheel rejected by uv, letting uv pick the wheel");
                out = self
                    .uv_install_torch(rep, &install, None, (0.0, 1.0))
                    .await?;
            }
            if out.success {
                self.torch_source = Some(source);
                self.pypi_index = attempt % self.pypi.len().max(1);
                return Ok(());
            }
            failures.push(
                &label,
                tool_error("bootstrap.torch_failed", "Could not install PyTorch", &out),
            )?;
        }
        Err(failures.into_error("bootstrap.torch_failed", "Could not install PyTorch"))
    }

    async fn fetch_wheel(
        &self,
        rep: &mut Reporter,
        page: &str,
        version: &str,
        py_tag: &str,
    ) -> StageResult<PathBuf> {
        rep.set_detail("Looking up the PyTorch package…");
        rep.emit(true);
        let found = tokio::select! {
            _ = self.cancel.cancelled() => return Err(StageError::Cancelled),
            f = wheel::find_wheel(&self.net, page, "torch", version, py_tag) => f?,
        };
        rep.log(&format!("torch wheel: {}", found.url));
        let label = found.filename.clone();
        rep.set_detail(format!("Downloading {label}"));
        rep.emit(true);
        wheel::download(
            &self.net,
            &found,
            &self.paths.downloads_dir,
            &self.cancel,
            |p| {
                if p.verifying {
                    rep.set_detail("Verifying the download…");
                    rep.set_bytes(Some(p.done), p.total, None);
                    rep.emit(true);
                    return;
                }
                if let Some(total) = p.total.filter(|t| *t > 0) {
                    rep.set_progress(0.8 * p.done as f64 / total as f64);
                }
                rep.set_detail(format!("Downloading {label}"));
                rep.set_bytes(Some(p.done), p.total, p.speed);
                rep.emit(false);
            },
        )
        .await
    }

    async fn uv_install_torch(
        &mut self,
        rep: &mut Reporter,
        install: &TorchInstall<'_>,
        local_wheel: Option<&PathBuf>,
        range: (f64, f64),
    ) -> StageResult<ToolOutput> {
        self.invalidate_state();
        rep.set_detail("Installing PyTorch…");
        rep.clear_bytes();
        rep.emit(true);
        let mut cmd = self.uv();
        cmd.args(["pip", "install", "--python"])
            .arg(self.paths.venv_python())
            .args([
                "--index-strategy",
                "unsafe-best-match",
                "--index-url",
                install.pypi_url,
            ])
            .args(torch_index_args(install.source))
            .env("UV_PYTHON_DOWNLOADS", "never");
        if install.reinstall {
            cmd.args([
                "--reinstall-package",
                "torch",
                "--reinstall-package",
                "torchaudio",
            ]);
        }
        match local_wheel {
            Some(path) => cmd.arg(path),
            None => cmd.arg(format!("torch=={}", install.versions.0)),
        };
        cmd.arg(format!("torchaudio=={}", install.versions.1));
        let mut observer = PipObserver {
            tracker: PipTracker::new(Some(DirSampler::start(self.paths.uv_cache_dir.clone()))),
            range,
        };
        self.tool(cmd, rep, &mut observer, None).await
    }

    // ------------------------------------------------------------------ packages

    async fn stage_packages(&mut self, rep: &mut Reporter) -> StageResult<()> {
        rep.begin_stage(Stage::Packages, "Installing audio and AI packages");
        let installed = self.installed_versions().await;
        let source = self
            .torch_source
            .clone()
            .or_else(|| self.torch_sources.first().cloned());
        let expected = |version: &str| match &source {
            Some(source) => self.version_for(version, source),
            None => version.to_string(),
        };
        let torch = installed
            .get("torch")
            .cloned()
            .unwrap_or_else(|| expected(&self.manifest.runtime.torch));
        let audio = installed
            .get("torchaudio")
            .cloned()
            .unwrap_or_else(|| expected(&self.manifest.runtime.torchaudio));
        // Keep the CUDA build of torch no matter what the requirements pull in.
        let constraints = self.paths.runtime_dir.join("constraints.txt");
        fsutil::write_atomic(
            &constraints,
            format!("torch=={torch}\ntorchaudio=={audio}\n").as_bytes(),
        )
        .map_err(|e| ShellError::io("Cannot write the constraints file", &e))?;

        let requirements = self.requirements_path();
        let extras = self.variant_manifest()?.extra_packages_for_host();
        let mirrors = self.pypi.len().max(1);
        let mut failures = Failures::default();
        for offset in 0..mirrors {
            let index = (self.pypi_index + offset) % mirrors;
            let pypi_url = self.pypi_url(index);
            let mirror_id = self.pypi_id(index);
            rep.reset_progress();
            rep.set_source(Some(mirror_id.clone()));
            if offset > 0 {
                rep.set_detail(format!("Trying another mirror: {mirror_id}"));
            }
            rep.emit(true);
            self.invalidate_state();
            let mut cmd = self.uv();
            cmd.args(["pip", "install", "--python"])
                .arg(self.paths.venv_python())
                .args([
                    "--index-strategy",
                    "unsafe-best-match",
                    "--index-url",
                    &pypi_url,
                ])
                .args(source.as_ref().map(torch_index_args).unwrap_or_default())
                .arg("--constraints")
                .arg(&constraints)
                .arg("--requirements")
                .arg(&requirements)
                .args(&extras)
                .env("UV_PYTHON_DOWNLOADS", "never");
            let mut observer = PipObserver {
                tracker: PipTracker::new(Some(DirSampler::start(self.paths.uv_cache_dir.clone()))),
                range: (0.0, 1.0),
            };
            let out = self.tool(cmd, rep, &mut observer, None).await?;
            if out.success {
                self.pypi_index = index;
                return Ok(());
            }
            failures.push(
                &mirror_id,
                tool_error(
                    "bootstrap.packages_failed",
                    "Could not install the required packages",
                    &out,
                ),
            )?;
        }
        Err(failures.into_error(
            "bootstrap.packages_failed",
            "Could not install the required packages",
        ))
    }

    // ------------------------------------------------------------------ verify

    async fn stage_verify(&mut self, rep: &mut Reporter) -> StageResult<RuntimeState> {
        rep.begin_stage(Stage::Verify, "Checking the installation");
        rep.set_detail("Loading PyTorch…");
        rep.emit(true);
        let cmd = self.venv_python_cmd(VERIFY_SCRIPT);
        let mut observer = CreepObserver::new(20.0, (0.0, 0.85));
        let out = self
            .tool(cmd, rep, &mut observer, Some(Duration::from_secs(600)))
            .await?;
        let report = out
            .stdout
            .iter()
            .find_map(|l| l.strip_prefix("STELLARIA_VERIFY "))
            .and_then(|json| serde_json::from_str::<VerifyReport>(json).ok());
        let report = match report {
            Some(r) if out.success && r.errors.is_empty() => r,
            other => return Err(verify_error(&out, other.as_ref()).into()),
        };
        rep.set_detail(format!(
            "Python {} · PyTorch {}{}",
            report.python,
            report.torch.clone().unwrap_or_default(),
            match (&report.cuda, report.cuda_available) {
                (Some(cuda), Some(true)) => format!(" · CUDA {cuda}"),
                (Some(cuda), _) => format!(" · CUDA {cuda} (no usable GPU right now)"),
                _ => String::new(),
            }
        ));
        rep.set_progress(0.9);
        rep.emit(true);

        let state = RuntimeState {
            runtime_id: self.manifest.runtime.id.clone(),
            variant: self.variant(),
            python: report.python.clone(),
            torch: report.torch.clone().unwrap_or_default(),
            cuda: report.cuda.clone(),
            created_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            torchaudio: report.torchaudio.clone(),
            cuda_available: report.cuda_available,
        };
        state
            .save(&self.paths.state_file)
            .map_err(|e| ShellError::io("Cannot save the runtime state", &e))?;

        // Free the space taken by the wheel and the unpacked cache (several GB for CUDA).
        rep.set_detail("Cleaning up…");
        rep.emit(true);
        let downloads = self.paths.downloads_dir.clone();
        let _ = tokio::task::spawn_blocking(move || fsutil::remove_dir_all_retry(&downloads)).await;
        let mut cmd = self.uv();
        cmd.args(["cache", "clean"]);
        let mut observer = CreepObserver::new(10.0, (0.9, 1.0));
        match self
            .tool(cmd, rep, &mut observer, Some(Duration::from_secs(300)))
            .await
        {
            Ok(out) if !out.success => rep.log("uv cache clean failed (ignored)"),
            Err(StageError::Failed(err)) => rep.log(&format!("uv cache clean: {}", err.detail)),
            Err(StageError::Cancelled) => return Err(StageError::Cancelled),
            Ok(_) => {}
        }
        rep.set_progress(1.0);
        Ok(state)
    }
}

/// `uv` with our cache and Python dirs, plain output, and nothing from the user's own
/// uv/pip/conda setup that could redirect the install.
pub(crate) fn uv_command(paths: &AppPaths, uv: &std::path::Path) -> Command {
    let mut cmd = hidden_command(uv);
    cmd.current_dir(&paths.runtime_dir)
        .env("UV_CACHE_DIR", &paths.uv_cache_dir)
        .env("UV_PYTHON_INSTALL_DIR", &paths.python_dir)
        .env("UV_PYTHON_PREFERENCE", "only-managed")
        .env("UV_NO_CONFIG", "1")
        .env("UV_NO_PROGRESS", "1")
        .env("UV_HTTP_TIMEOUT", "300")
        .env("NO_COLOR", "1");
    for var in [
        "UV_INDEX",
        "UV_INDEX_URL",
        "UV_DEFAULT_INDEX",
        "UV_EXTRA_INDEX_URL",
        "UV_FIND_LINKS",
        "UV_INDEX_STRATEGY",
        "UV_PYTHON",
        "UV_SYSTEM_PYTHON",
        "UV_MANAGED_PYTHON",
        "UV_NO_MANAGED_PYTHON",
        "UV_NO_CACHE",
        "UV_OFFLINE",
        "UV_CONSTRAINT",
        "UV_OVERRIDE",
        "UV_TORCH_BACKEND",
        "UV_PYTHON_INSTALL_MIRROR",
        "UV_LINK_MODE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PYTHONHOME",
        "PYTHONPATH",
    ] {
        cmd.env_remove(var);
    }
    if cfg!(windows) {
        cmd.env("UV_LINK_MODE", "copy");
    }
    cmd
}

/// The manifest's PyPI mirrors (pypi.org if it lists none).
pub(super) fn pypi_sources(manifest: &Manifest) -> Vec<Source> {
    non_empty_or(
        &manifest.runtime.pypi,
        Source {
            id: "pypi".into(),
            region: "global".into(),
            url: Some(DEFAULT_PYPI.into()),
            kind: None,
        },
    )
}

/// Arguments shared by the attempts of one torch install.
struct TorchInstall<'a> {
    source: &'a Source,
    pypi_url: &'a str,
    /// `(torch, torchaudio)` versions as the source labels them.
    versions: &'a (String, String),
    reinstall: bool,
}

fn non_empty_or(sources: &[Source], fallback: Source) -> Vec<Source> {
    if sources.is_empty() {
        vec![fallback]
    } else {
        sources.to_vec()
    }
}

fn check_platform(variant: TorchVariant) -> ShellResult<()> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    if os == "macos" && arch != "aarch64" {
        return Err(ShellError::new(
            "bootstrap.unsupported_platform",
            "PyTorch no longer supports Intel Macs, so the runtime cannot be installed",
        ));
    }
    let fits = match variant {
        TorchVariant::Mac => os == "macos",
        TorchVariant::Cu128 | TorchVariant::Cu126 | TorchVariant::Cpu => {
            os == "windows" || os == "linux"
        }
    };
    if !fits {
        return Err(ShellError::new(
            "bootstrap.variant_unsupported",
            format!(
                "The '{}' runtime is not available on this system",
                variant.as_str()
            ),
        ));
    }
    Ok(())
}

fn torch_index_args(source: &Source) -> Vec<String> {
    match (source.kind, &source.url) {
        (Some(SourceKind::Index), Some(url)) => vec!["--extra-index-url".into(), url.clone()],
        (Some(SourceKind::FindLinks), Some(url)) => vec!["--find-links".into(), url.clone()],
        _ => Vec::new(),
    }
}

fn ids(sources: &[Source]) -> Vec<&str> {
    sources.iter().map(|s| s.id.as_str()).collect()
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct VerifyReport {
    python: String,
    torch: Option<String>,
    torchaudio: Option<String>,
    cuda: Option<String>,
    cuda_available: Option<bool>,
    #[serde(default)]
    errors: HashMap<String, String>,
}

fn verify_error(out: &ToolOutput, report: Option<&VerifyReport>) -> ShellError {
    let mut detail = out.tail_text();
    if let Some(report) = report {
        for (module, err) in &report.errors {
            detail.push_str(&format!("\n{module}: {err}"));
        }
    }
    // torch's DLLs need the MSVC runtime, which some Windows installs lack.
    let missing_vc = cfg!(windows)
        && ["WinError 126", "c10.dll", "fbgemm.dll", "DLL load failed"]
            .iter()
            .any(|m| detail.contains(m));
    let err = if missing_vc {
        ShellError::new(
            "bootstrap.vcredist_missing",
            "PyTorch needs the Microsoft Visual C++ Redistributable (x64)",
        )
    } else {
        ShellError::new(
            "bootstrap.verify_failed",
            "The installed runtime does not work",
        )
    };
    err.with_detail(detail)
}

/// Imports what the backend and engine need and reports versions as one JSON line.
const VERIFY_SCRIPT: &str = r#"
import importlib, json, sys
info = {"python": sys.version.split()[0], "errors": {}}
for name in ("numpy", "torch", "torchaudio", "fastapi", "librosa"):
    try:
        module = importlib.import_module(name)
        info[name] = getattr(module, "__version__", "")
    except Exception as exc:
        info["errors"][name] = "%s: %s" % (type(exc).__name__, exc)
try:
    import torch
    info["cuda"] = torch.version.cuda
    info["cudaAvailable"] = bool(torch.cuda.is_available())
    mps = getattr(torch.backends, "mps", None)
    info["mps"] = bool(mps is not None and mps.is_available())
except Exception:
    pass
print("STELLARIA_VERIFY " + json.dumps(info), flush=True)
"#;

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::super::progress::{self, BootstrapProgress};
    use super::*;

    #[test]
    fn torch_args() {
        let src = Source {
            id: "aliyun".into(),
            region: "cn".into(),
            url: Some("https://mirrors.aliyun.com/pytorch-wheels/cu128/".into()),
            kind: Some(SourceKind::FindLinks),
        };
        assert_eq!(
            torch_index_args(&src),
            [
                "--find-links",
                "https://mirrors.aliyun.com/pytorch-wheels/cu128/"
            ]
        );
    }

    /// Minimal stand-ins for torch/torchaudio, for sandboxes that cannot download PyTorch.
    fn install_fake_torch(paths: &AppPaths, version: &str) {
        let out = std::process::Command::new(paths.venv_python())
            .args([
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ])
            .output()
            .unwrap();
        let site = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
        for name in ["torch", "torchaudio"] {
            let pkg = site.join(name);
            std::fs::create_dir_all(&pkg).unwrap();
            std::fs::write(
                pkg.join("__init__.py"),
                format!(
                    "__version__ = '{version}'\n\
                     class version:\n    cuda = None\n\
                     class cuda:\n    @staticmethod\n    def is_available():\n        return False\n\
                     class backends:\n    mps = None\n"
                ),
            )
            .unwrap();
            let info = site.join(format!("{name}-{version}.dist-info"));
            std::fs::create_dir_all(&info).unwrap();
            std::fs::write(
                info.join("METADATA"),
                format!("Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"),
            )
            .unwrap();
            std::fs::write(info.join("INSTALLER"), "stellaria-test\n").unwrap();
            std::fs::write(info.join("RECORD"), "").unwrap();
        }
    }

    /// Full pipeline against the real network with the `uv` on PATH (CPU variant, Linux).
    /// `cargo test -- --ignored bootstrap_end_to_end --nocapture`
    ///
    /// Env: `STELLARIA_E2E_DIR` (work dir), `STELLARIA_E2E_REGION` (auto|cn|global),
    /// `STELLARIA_E2E_TORCH`: unset = the manifest's torch sources, `pypi` = torch from
    /// PyPI, `fake` = stand-in torch (tests every stage but the torch download).
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads Python and PyTorch"]
    async fn bootstrap_end_to_end() {
        let root = std::env::var_os("STELLARIA_E2E_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("stellaria-shell-e2e"));
        let region = match std::env::var("STELLARIA_E2E_REGION").as_deref() {
            Ok("cn") => Region::Cn,
            Ok("global") => Region::Global,
            _ => Region::Auto,
        };
        let torch_mode = std::env::var("STELLARIA_E2E_TORCH").unwrap_or_default();
        // `STELLARIA_E2E_BACKEND`: a backend dir whose requirements-runtime.txt to install.
        let backend = match std::env::var_os("STELLARIA_E2E_BACKEND") {
            Some(dir) => PathBuf::from(dir),
            None => {
                let backend = root.join("backend");
                std::fs::create_dir_all(&backend).unwrap();
                std::fs::write(
                    backend.join("requirements-runtime.txt"),
                    "numpy\nfastapi\nlibrosa\n",
                )
                .unwrap();
                backend
            }
        };
        let repo_manifest =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resources/manifest.json");
        let paths = AppPaths::from_data_dir(root.join("data"), root.join("library"));
        let resources = ResourcePaths {
            engine: root.clone(),
            backend,
            manifest: repo_manifest.clone(),
            uv: crate::paths::find_in_path("uv").expect("uv on PATH"),
            dev_python: None,
            dev: true,
        };
        let events: Arc<Mutex<Vec<BootstrapProgress>>> = Arc::default();
        let collected = events.clone();
        let sink: progress::Sink = Box::new(move |p| {
            println!(
                "[{:?} {}/{}] {:>5.1}% overall {:>5.1}% | {} | {} | {:?}/{:?} @ {:?} | {:?}",
                p.stage,
                p.stage_index,
                p.stage_count,
                p.progress.unwrap_or(0.0) * 100.0,
                p.overall.unwrap_or(0.0) * 100.0,
                p.message,
                p.detail.as_deref().unwrap_or(""),
                p.bytes_done,
                p.bytes_total,
                p.speed.map(|s| (s / 1e6 * 10.0).round() / 10.0),
                p.source
            );
            if let Some(err) = &p.error {
                println!("ERROR {err:?}");
            }
            collected.lock().unwrap().push(p.clone());
        });
        let mut rep = Reporter::new(
            sink,
            progress::weights(TorchVariant::Cpu),
            paths.logs_dir.join("bootstrap.log"),
        );
        let net = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .read_timeout(Duration::from_secs(60))
            .build()
            .unwrap();
        let mut manifest = Manifest::load(&repo_manifest).unwrap();
        if torch_mode == "pypi" || torch_mode == "fake" {
            // Sandboxes that only reach PyPI: take torch from there (no local label).
            let cpu = manifest.runtime.variants.get_mut("cpu").unwrap();
            cpu.torch_sources = vec![Source {
                id: "pypi".into(),
                region: "any".into(),
                url: None,
                kind: Some(SourceKind::Pypi),
            }];
        }
        let torch_version = manifest.runtime.torch.clone();
        let mut pipeline = Pipeline::new(
            paths.clone(),
            resources,
            manifest,
            Request {
                variant: TorchVariant::Cpu,
                region,
                mode: Mode::Install,
            },
            CancellationToken::new(),
            net,
        );
        let result = async {
            pipeline.stage_prepare(&mut rep).await?;
            pipeline.stage_python(&mut rep).await?;
            pipeline.stage_venv(&mut rep).await?;
            if torch_mode == "fake" {
                install_fake_torch(&paths, &torch_version);
            }
            pipeline.stage_torch(&mut rep).await?;
            pipeline.stage_packages(&mut rep).await?;
            pipeline.stage_verify(&mut rep).await
        }
        .await;
        let state = match result {
            Ok(state) => state,
            Err(StageError::Failed(err)) => panic!("bootstrap failed: {err:?}"),
            Err(StageError::Cancelled) => panic!("cancelled"),
        };
        rep.finish("done");
        assert!(state.torch.starts_with("2.9.1"), "{state:?}");
        let info = crate::runtime::runtime_info(&paths, Some(&state.runtime_id), false);
        assert_eq!(info.status, crate::runtime::RuntimeStatus::Ready);
        let events = events.lock().unwrap();
        let mut last = 0.0;
        for e in events.iter().filter(|e| e.stage != Stage::Done) {
            let overall = e.overall.unwrap_or(0.0);
            // Only a mirror retry may move the bar back.
            assert!(
                overall + 0.05 >= last,
                "overall went back: {last} -> {overall}"
            );
            last = overall;
        }
    }
}

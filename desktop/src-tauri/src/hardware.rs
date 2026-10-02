//! `detect_hardware`: GPUs, CPU, RAM and the recommended torch variant.

use std::path::PathBuf;
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;

use crate::paths::find_in_path;
use crate::procutil::{hidden_command, run_capture};
use crate::types::{platform_name, Msg, TorchVariant};

/// cu12x wheels need at least these drivers (CUDA minor-version compatibility).
const MIN_DRIVER_WINDOWS: &str = "528.33";
const MIN_DRIVER_LINUX: &str = "525.60.13";
const LOW_VRAM_BYTES: u64 = 6 * 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareReport {
    pub os: String,
    pub os_version: String,
    pub cpu: String,
    pub ram_total: u64,
    pub gpus: Vec<GpuReport>,
    pub apple_silicon: bool,
    pub recommended_variant: TorchVariant,
    pub can_train: bool,
    pub notes: Vec<Msg>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuReport {
    pub vendor: Vendor,
    pub name: String,
    /// Bytes (dedicated VRAM); `null` when unknown or unified memory.
    pub vram: Option<u64>,
    pub driver: Option<String>,
    pub compute_capability: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other,
}

pub async fn detect() -> HardwareReport {
    let basics = tokio::task::spawn_blocking(system_basics)
        .await
        .unwrap_or_default();
    let os = platform_name();

    let mut gpus = Vec::new();
    let mut apple_silicon = false;
    let mut cpu = basics.cpu;
    let mut ram_total = basics.ram_total;

    match os {
        "macos" => {
            let mac = mac_sysctl().await;
            apple_silicon = mac.arm64 || std::env::consts::ARCH == "aarch64";
            if let Some(brand) = mac.cpu_brand {
                cpu = brand;
            }
            if let Some(mem) = mac.memsize {
                ram_total = mem;
            }
            if apple_silicon {
                gpus.push(GpuReport {
                    vendor: Vendor::Apple,
                    name: cpu.clone(),
                    vram: None,
                    driver: None,
                    compute_capability: None,
                });
            }
        }
        _ => {
            let (nvidia, others) = tokio::join!(nvidia_smi(), other_gpus());
            let have_nvidia = !nvidia.is_empty();
            gpus.extend(nvidia);
            gpus.extend(
                others
                    .into_iter()
                    .filter(|g| !(have_nvidia && g.vendor == Vendor::Nvidia)),
            );
        }
    }

    let assessment = assess(os, apple_silicon, &gpus);
    HardwareReport {
        os: os.into(),
        os_version: basics.os_version,
        cpu,
        ram_total,
        gpus,
        apple_silicon,
        recommended_variant: assessment.variant,
        can_train: assessment.can_train,
        notes: assessment.notes,
    }
}

#[derive(Debug, Default)]
struct Basics {
    os_version: String,
    cpu: String,
    ram_total: u64,
}

fn system_basics() -> Basics {
    use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
    let sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::nothing())
            .with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    Basics {
        os_version: System::long_os_version()
            .or_else(System::os_version)
            .unwrap_or_default(),
        cpu: sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .unwrap_or_default(),
        ram_total: sys.total_memory(),
    }
}

struct Assessment {
    variant: TorchVariant,
    can_train: bool,
    notes: Vec<Msg>,
}

/// The variant rules from docs/architecture.md, kept pure for testing.
fn assess(os: &str, apple_silicon: bool, gpus: &[GpuReport]) -> Assessment {
    let mut notes = Vec::new();
    if os == "macos" {
        if apple_silicon {
            notes.push(Msg::new("mac.train_unsupported"));
            return Assessment {
                variant: TorchVariant::Mac,
                can_train: false,
                notes,
            };
        }
        // No torch builds for Intel Macs since 2.2: the runtime cannot be installed.
        notes.push(Msg::new("mac.intel_unsupported"));
        return Assessment {
            variant: TorchVariant::Cpu,
            can_train: false,
            notes,
        };
    }

    // Best NVIDIA GPU: highest compute capability, then most VRAM.
    let best = gpus
        .iter()
        .filter(|g| g.vendor == Vendor::Nvidia)
        .max_by_key(|g| (effective_cc(g).unwrap_or((0, 0)), g.vram.unwrap_or(0)));

    let Some(gpu) = best else {
        notes.push(Msg::new("gpu.no_nvidia"));
        return Assessment {
            variant: TorchVariant::Cpu,
            can_train: false,
            notes,
        };
    };

    let cc = effective_cc(gpu);
    if matches!(cc, Some((major, _)) if major < 5) {
        // Kepler and older: no current CUDA wheel supports them.
        notes.push(
            Msg::new("gpu.too_old")
                .with("name", gpu.name.clone())
                .with("computeCapability", format_cc(cc)),
        );
        return Assessment {
            variant: TorchVariant::Cpu,
            can_train: false,
            notes,
        };
    }
    let variant = match cc {
        Some(cc) if cc >= (7, 5) => TorchVariant::Cu128,
        _ => TorchVariant::Cu126,
    };

    if let Some(vram) = gpu.vram.filter(|v| *v < LOW_VRAM_BYTES) {
        let gb = (vram as f64 / 1024f64.powi(3) * 10.0).round() / 10.0;
        notes.push(
            Msg::new("gpu.low_vram")
                .with("name", gpu.name.clone())
                .with("vramGb", gb),
        );
    }

    let required = if os == "windows" {
        MIN_DRIVER_WINDOWS
    } else {
        MIN_DRIVER_LINUX
    };
    let driver_ok = match gpu.driver.as_deref() {
        Some(found) => version_at_least(found, required),
        None => false,
    };
    if !driver_ok {
        notes.push(
            Msg::new("driver.too_old")
                .with(
                    "found",
                    gpu.driver.clone().unwrap_or_else(|| "unknown".into()),
                )
                .with("required", required),
        );
    }

    Assessment {
        variant,
        can_train: driver_ok,
        notes,
    }
}

fn effective_cc(gpu: &GpuReport) -> Option<(u32, u32)> {
    gpu.compute_capability
        .as_deref()
        .and_then(parse_cc)
        .or_else(|| infer_cc_from_name(&gpu.name))
}

fn parse_cc(text: &str) -> Option<(u32, u32)> {
    let (major, minor) = text.trim().split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

fn format_cc(cc: Option<(u32, u32)>) -> String {
    cc.map(|(a, b)| format!("{a}.{b}")).unwrap_or_default()
}

/// Fallback for drivers too old to report `compute_cap`.
fn infer_cc_from_name(name: &str) -> Option<(u32, u32)> {
    let n = name.to_ascii_uppercase();
    let has = |needle: &str| n.contains(needle);
    if has("RTX") || has("GTX 16") || has(" T4") || has("QUADRO T") || has(" T1000") {
        return Some((7, 5));
    }
    let datacenter = [
        " A100", " A10", " A30", " A40", " A16", " A2 ", " H100", " H200", " L4", " L40",
    ];
    if datacenter.iter().any(|d| has(d)) {
        return Some((8, 0));
    }
    if has("V100") || has("TITAN V") {
        return Some((7, 0));
    }
    if has("GTX 10") || has("P100") || has("P40") || has("TITAN X") || has("QUADRO P") {
        return Some((6, 1));
    }
    if has("GTX 9") || has("GTX 750") || has("M40") || has("QUADRO M") {
        return Some((5, 2));
    }
    if has("GTX 7") || has("GTX 6") || has("K80") || has("K40") || has("QUADRO K") {
        return Some((3, 5));
    }
    None
}

/// Numeric, component-wise version comparison (`576.02` ≥ `528.33`).
pub fn version_at_least(found: &str, required: &str) -> bool {
    let parse = |s: &str| -> Vec<u64> {
        s.trim()
            .split(['.', '-'])
            .map(|p| {
                p.chars()
                    .take_while(char::is_ascii_digit)
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0)
            })
            .collect()
    };
    let (a, b) = (parse(found), parse(required));
    for i in 0..a.len().max(b.len()) {
        let (x, y) = (
            a.get(i).copied().unwrap_or(0),
            b.get(i).copied().unwrap_or(0),
        );
        if x != y {
            return x > y;
        }
    }
    true
}

// ---------------------------------------------------------------- NVIDIA

fn nvidia_smi_candidates() -> Vec<PathBuf> {
    let mut list: Vec<PathBuf> = find_in_path("nvidia-smi").into_iter().collect();
    if cfg!(windows) {
        if let Some(root) = std::env::var_os("SystemRoot") {
            list.push(PathBuf::from(root).join("System32").join("nvidia-smi.exe"));
        }
        if let Some(pf) = std::env::var_os("ProgramFiles") {
            list.push(
                PathBuf::from(pf)
                    .join("NVIDIA Corporation")
                    .join("NVSMI")
                    .join("nvidia-smi.exe"),
            );
        }
    } else {
        list.push(PathBuf::from("/usr/bin/nvidia-smi"));
    }
    list.dedup();
    list.into_iter().filter(|p| p.is_file()).collect()
}

async fn nvidia_smi() -> Vec<GpuReport> {
    let Some(exe) = nvidia_smi_candidates().into_iter().next() else {
        return Vec::new();
    };
    for with_cc in [true, false] {
        let fields = if with_cc {
            "name,memory.total,driver_version,compute_cap"
        } else {
            "name,memory.total,driver_version"
        };
        let mut cmd = hidden_command(&exe);
        cmd.arg(format!("--query-gpu={fields}"))
            .arg("--format=csv,noheader,nounits");
        match run_capture(cmd, Duration::from_secs(15)).await {
            Ok(out) if out.success => {
                let gpus = parse_nvidia_csv(&out.stdout);
                if !gpus.is_empty() {
                    return gpus;
                }
            }
            Ok(out) => log::info!(
                "nvidia-smi ({fields}) failed: {} {}",
                out.stdout.trim(),
                out.stderr.trim()
            ),
            Err(err) => log::info!("nvidia-smi failed: {err}"),
        }
    }
    Vec::new()
}

fn parse_nvidia_csv(text: &str) -> Vec<GpuReport> {
    let clean = |s: &str| -> Option<String> {
        let s = s.trim();
        (!s.is_empty() && !s.starts_with('[')).then(|| s.to_string())
    };
    text.lines()
        .filter_map(|line| {
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 3 {
                return None;
            }
            let name = clean(cols[0])?;
            Some(GpuReport {
                vendor: Vendor::Nvidia,
                name,
                vram: clean(cols[1])
                    .and_then(|v| v.parse::<f64>().ok())
                    .map(|mib| (mib * MIB as f64) as u64),
                driver: clean(cols[2]),
                compute_capability: cols.get(3).and_then(|c| clean(c)),
            })
        })
        .collect()
}

// ---------------------------------------------------------------- other GPUs

#[cfg(windows)]
async fn other_gpus() -> Vec<GpuReport> {
    let script = "Get-CimInstance Win32_VideoController | \
                  Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID | \
                  ConvertTo-Json -Compress";
    let mut cmd = hidden_command("powershell.exe");
    cmd.args([
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        script,
    ]);
    match run_capture(cmd, Duration::from_secs(15)).await {
        Ok(out) if out.success => parse_cim_json(&out.stdout),
        Ok(out) => {
            log::info!("CIM GPU query failed: {}", out.stderr.trim());
            Vec::new()
        }
        Err(err) => {
            log::info!("CIM GPU query failed: {err}");
            Vec::new()
        }
    }
}

#[cfg(not(windows))]
async fn other_gpus() -> Vec<GpuReport> {
    Vec::new()
}

#[cfg_attr(not(windows), allow(dead_code))]
fn parse_cim_json(text: &str) -> Vec<GpuReport> {
    let Ok(value) = serde_json::from_str::<Value>(text.trim()) else {
        return Vec::new();
    };
    let items = match value {
        Value::Array(items) => items,
        single @ Value::Object(_) => vec![single],
        _ => return Vec::new(),
    };
    items
        .iter()
        .filter_map(|item| {
            let name = item.get("Name")?.as_str()?.trim().to_string();
            let pnp = item
                .get("PNPDeviceID")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_ascii_uppercase();
            let upper = name.to_ascii_uppercase();
            let vendor = if pnp.contains("VEN_10DE") || upper.contains("NVIDIA") {
                Vendor::Nvidia
            } else if pnp.contains("VEN_1002") || upper.contains("AMD") || upper.contains("RADEON")
            {
                Vendor::Amd
            } else if pnp.contains("VEN_8086") || upper.contains("INTEL") {
                Vendor::Intel
            } else {
                Vendor::Other
            };
            let windows_driver = item.get("DriverVersion").and_then(Value::as_str);
            let driver = match vendor {
                Vendor::Nvidia => windows_driver.and_then(nvidia_from_windows_driver),
                _ => windows_driver.map(str::to_string),
            };
            Some(GpuReport {
                vendor,
                name,
                // AdapterRAM is a uint32 and saturates at 4 GB; only trust smaller values.
                vram: item
                    .get("AdapterRAM")
                    .and_then(Value::as_u64)
                    .filter(|v| *v > 0 && *v < u64::from(u32::MAX)),
                driver,
                compute_capability: None,
            })
        })
        .collect()
}

/// `32.0.15.6094` → `560.94` (the NVIDIA version is the last five digits).
fn nvidia_from_windows_driver(version: &str) -> Option<String> {
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    let digits: String = parts[parts.len() - 2..].concat();
    if digits.len() < 5 || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let last5 = &digits[digits.len() - 5..];
    let major: u32 = last5[..3].parse().ok()?;
    Some(format!("{major}.{}", &last5[3..]))
}

// ---------------------------------------------------------------- macOS

#[derive(Default)]
struct MacInfo {
    arm64: bool,
    cpu_brand: Option<String>,
    memsize: Option<u64>,
}

async fn mac_sysctl() -> MacInfo {
    async fn sysctl(key: &str) -> Option<String> {
        let exe = if std::path::Path::new("/usr/sbin/sysctl").exists() {
            "/usr/sbin/sysctl"
        } else {
            "sysctl"
        };
        let mut cmd = hidden_command(exe);
        cmd.args(["-n", key]);
        let out = run_capture(cmd, Duration::from_secs(5)).await.ok()?;
        out.success
            .then(|| out.stdout.trim().to_string())
            .filter(|s| !s.is_empty())
    }
    let (arm64, brand, mem) = tokio::join!(
        sysctl("hw.optional.arm64"),
        sysctl("machdep.cpu.brand_string"),
        sysctl("hw.memsize")
    );
    MacInfo {
        arm64: arm64.as_deref() == Some("1"),
        cpu_brand: brand,
        memsize: mem.and_then(|m| m.parse().ok()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nvidia(name: &str, vram_gb: u64, driver: &str, cc: Option<&str>) -> GpuReport {
        GpuReport {
            vendor: Vendor::Nvidia,
            name: name.into(),
            vram: Some(vram_gb * 1024 * MIB),
            driver: Some(driver.into()),
            compute_capability: cc.map(Into::into),
        }
    }

    fn codes(a: &Assessment) -> Vec<&str> {
        a.notes.iter().map(|n| n.code.as_str()).collect()
    }

    #[test]
    fn parses_nvidia_smi_output() {
        let gpus = parse_nvidia_csv(
            "NVIDIA GeForce RTX 4090, 24564, 576.02, 8.9\nTesla K80, 11441, [N/A], [N/A]\n",
        );
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].compute_capability.as_deref(), Some("8.9"));
        assert_eq!(gpus[0].vram, Some(24564 * MIB));
        assert_eq!(gpus[1].driver, None);
        let old = parse_nvidia_csv("GeForce GTX 1060 6GB, 6144, 456.71\n");
        assert_eq!(old[0].compute_capability, None);
    }

    #[test]
    fn variant_rules() {
        let a = assess(
            "windows",
            false,
            &[nvidia("RTX 4070", 12, "576.02", Some("8.9"))],
        );
        assert_eq!(a.variant, TorchVariant::Cu128);
        assert!(a.can_train);
        assert!(a.notes.is_empty());

        let a = assess(
            "windows",
            false,
            &[nvidia("GTX 1060", 6, "560.94", Some("6.1"))],
        );
        assert_eq!(a.variant, TorchVariant::Cu126);
        assert!(a.can_train);

        let a = assess("windows", false, &[nvidia("GTX 1050", 4, "456.71", None)]);
        assert_eq!(a.variant, TorchVariant::Cu126);
        assert!(!a.can_train);
        assert_eq!(codes(&a), vec!["gpu.low_vram", "driver.too_old"]);

        let a = assess(
            "linux",
            false,
            &[nvidia("RTX 3090", 24, "525.60.13", Some("8.6"))],
        );
        assert!(a.can_train);

        let a = assess("windows", false, &[]);
        assert_eq!(a.variant, TorchVariant::Cpu);
        assert_eq!(codes(&a), vec!["gpu.no_nvidia"]);

        let a = assess("macos", true, &[]);
        assert_eq!(a.variant, TorchVariant::Mac);
        assert_eq!(codes(&a), vec!["mac.train_unsupported"]);

        let a = assess("macos", false, &[]);
        assert_eq!(codes(&a), vec!["mac.intel_unsupported"]);
    }

    #[test]
    fn driver_versions() {
        assert!(version_at_least("528.33", "528.33"));
        assert!(version_at_least("576.02", "528.33"));
        assert!(!version_at_least("527.99", "528.33"));
        assert!(version_at_least("535.104.05", "525.60.13"));
        assert!(!version_at_least("525.60.12", "525.60.13"));
        assert_eq!(
            nvidia_from_windows_driver("32.0.15.6094").as_deref(),
            Some("560.94")
        );
        assert_eq!(
            nvidia_from_windows_driver("31.0.15.2833").as_deref(),
            Some("528.33")
        );
    }

    #[test]
    fn cim_json() {
        let gpus = parse_cim_json(
            r#"[{"Name":"AMD Radeon RX 6600","AdapterRAM":4293918720,"DriverVersion":"31.0.21001.45002","PNPDeviceID":"PCI\\VEN_1002&DEV_73FF"},
                {"Name":"NVIDIA GeForce RTX 3060","AdapterRAM":null,"DriverVersion":"32.0.15.6094","PNPDeviceID":"PCI\\VEN_10DE&DEV_2504"}]"#,
        );
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].vendor, Vendor::Amd);
        assert_eq!(gpus[1].vendor, Vendor::Nvidia);
        assert_eq!(gpus[1].driver.as_deref(), Some("560.94"));
        assert_eq!(infer_cc_from_name(&gpus[1].name), Some((7, 5)));
    }
}

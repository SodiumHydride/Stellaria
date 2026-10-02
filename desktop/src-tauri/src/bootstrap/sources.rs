//! Mirror ordering: explicit regions keep manifest order (preferred region first, the rest
//! as fallbacks); `auto` measures latency (HEAD, 3 s timeout) and sorts by it.

use std::time::{Duration, Instant};

use tokio::task::JoinSet;

use crate::manifest::{Source, SourceKind};
use crate::types::Region;

const PROBE_TIMEOUT: Duration = Duration::from_secs(3);
const GITHUB_PYTHON_BUILDS: &str = "https://github.com/astral-sh/python-build-standalone/releases";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceList {
    PythonInstall,
    Pypi,
    Torch,
}

/// URL probed to measure a source's latency (`None`: nothing to probe, e.g. "same as PyPI").
pub fn probe_url(list: SourceList, source: &Source) -> Option<String> {
    let base = source.url.as_deref().map(|u| u.trim_end_matches('/'));
    match list {
        SourceList::PythonInstall => Some(match base {
            Some(url) => format!("{url}/"),
            None => GITHUB_PYTHON_BUILDS.to_string(),
        }),
        // A small page; the simple-index root of PyPI is tens of megabytes.
        SourceList::Pypi => base.map(|url| format!("{url}/pip/")),
        SourceList::Torch => match source.kind {
            Some(SourceKind::Index) => base.map(|url| format!("{url}/torchaudio/")),
            Some(SourceKind::FindLinks) => base.map(|url| format!("{url}/")),
            Some(SourceKind::Pypi) | None => None,
        },
    }
}

pub type Latencies = Vec<(SourceList, String, Option<u64>)>;

/// Probes every source of every list concurrently; the whole pass takes at most ~3 s.
pub async fn probe_all(client: &reqwest::Client, lists: &[(SourceList, &[Source])]) -> Latencies {
    let mut set = JoinSet::new();
    for (list, sources) in lists {
        for source in *sources {
            let Some(url) = probe_url(*list, source) else {
                continue;
            };
            let client = client.clone();
            let (list, id) = (*list, source.id.clone());
            set.spawn(async move { (list, id, probe(&client, &url).await) });
        }
    }
    let mut out = Vec::new();
    while let Some(result) = set.join_next().await {
        if let Ok(item) = result {
            out.push(item);
        }
    }
    out
}

async fn probe(client: &reqwest::Client, url: &str) -> Option<u64> {
    let started = Instant::now();
    let response = client.head(url).timeout(PROBE_TIMEOUT).send().await.ok()?;
    // Any answer short of a server error means the host is reachable and serving.
    (!response.status().is_server_error()).then(|| started.elapsed().as_millis() as u64)
}

/// Orders `sources` for `region`. With `auto`, measured sources come first by latency,
/// unreachable ones last (manifest order is the tie-breaker everywhere).
pub fn order(
    sources: &[Source],
    region: Region,
    list: SourceList,
    latencies: &Latencies,
) -> Vec<Source> {
    let mut indexed: Vec<(usize, &Source)> = sources.iter().enumerate().collect();
    match region {
        Region::Cn | Region::Global => {
            let preferred = if region == Region::Cn { "cn" } else { "global" };
            indexed.sort_by_key(|(i, s)| {
                let rank = u8::from(!(s.region == preferred || s.region == "any"));
                (rank, *i)
            });
        }
        Region::Auto => {
            let latency = |s: &Source| {
                latencies
                    .iter()
                    .find(|(l, id, _)| *l == list && *id == s.id)
                    .and_then(|(_, _, ms)| *ms)
            };
            let has_probe = |s: &Source| probe_url(list, s).is_some();
            indexed.sort_by_key(|(i, s)| match (has_probe(s), latency(s)) {
                // Not probeable (e.g. "use the PyPI mirror"): neutral, keep manifest order.
                (false, _) => (0u8, 0u64, *i),
                (true, Some(ms)) => (0, ms, *i),
                (true, None) => (1, 0, *i),
            });
        }
    }
    indexed.into_iter().map(|(_, s)| s.clone()).collect()
}

/// Human summary for the progress detail, e.g. `pypi 180 ms · tuna 32 ms · aliyun –`.
pub fn summary(latencies: &Latencies, list: SourceList) -> String {
    latencies
        .iter()
        .filter(|(l, _, _)| *l == list)
        .map(|(_, id, ms)| match ms {
            Some(ms) => format!("{id} {ms} ms"),
            None => format!("{id} –"),
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(id: &str, region: &str) -> Source {
        Source {
            id: id.into(),
            region: region.into(),
            url: Some(format!("https://{id}.example/simple")),
            kind: Some(SourceKind::Index),
        }
    }

    #[test]
    fn explicit_regions_prefer_their_sources() {
        let sources = vec![src("pypi", "global"), src("tuna", "cn"), src("ustc", "cn")];
        let cn = order(&sources, Region::Cn, SourceList::Pypi, &Vec::new());
        let ids: Vec<_> = cn.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["tuna", "ustc", "pypi"]);
        let global = order(&sources, Region::Global, SourceList::Pypi, &Vec::new());
        assert_eq!(global[0].id, "pypi");
    }

    #[test]
    fn auto_sorts_by_latency() {
        let sources = vec![src("pypi", "global"), src("tuna", "cn"), src("ustc", "cn")];
        let lat = vec![
            (SourceList::Pypi, "pypi".to_string(), None),
            (SourceList::Pypi, "tuna".to_string(), Some(80)),
            (SourceList::Pypi, "ustc".to_string(), Some(30)),
        ];
        let ids: Vec<_> = order(&sources, Region::Auto, SourceList::Pypi, &lat)
            .into_iter()
            .map(|s| s.id)
            .collect();
        assert_eq!(ids, ["ustc", "tuna", "pypi"]);
    }
}

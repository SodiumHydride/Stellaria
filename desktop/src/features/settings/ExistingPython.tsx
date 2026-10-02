import clsx from 'clsx'
import { AlertTriangle, CheckCircle2, ChevronRight, FileSearch, FolderOpen, Layers, Loader2 } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { confirmDialog, Progress } from '../../components/ui'
import { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { useLatest } from '../../lib/useEvent'
import { listen, pickExecutable, pickFolder, shell, type BootstrapProgress, type PythonCandidate, type PythonCheck } from '../../shell/bridge'
import './existing-python.css'

/** How many of the environments found are checked in the background (each imports torch). */
const PROBE_LIMIT = 6

/**
 * Checks one interpreter (version, torch, the app's packages) and offers to use it, or to add
 * the packages it lacks — into Stellaria's own layer over it: the environment itself is never
 * changed. Should a layer not work with it, the way out is the app's own runtime (`onManaged`).
 */
export function PythonCheckPanel({
  path,
  initial,
  onUse,
  onCancel,
  onChecked,
  onManaged,
  busy,
}: {
  path: string
  /** a check that was just made (the list's background check), so it is not run twice */
  initial?: PythonCheck | null
  onUse: (path: string) => void
  onCancel: () => void
  /** every fresh check, so a list showing this interpreter stays current */
  onChecked?: (check: PythonCheck) => void
  /** use Stellaria's own runtime instead (offered when a layer does not work with this one) */
  onManaged?: () => void
  busy?: boolean
}) {
  const t = useT()
  const [check, setCheck] = useState<PythonCheck | null>(initial ?? null)
  const [installing, setInstalling] = useState<BootstrapProgress | null>(null)
  const [failure, setFailure] = useState<BootstrapProgress['error']>(null)
  const [layerFailed, setLayerFailed] = useState(false)
  // -1: `initial` is current; every step up checks again
  const [round, setRound] = useState(initial ? -1 : 0)
  const onCheckedRef = useLatest(onChecked)

  useEffect(() => {
    if (round < 0) return
    let live = true
    setCheck(null)
    void attempt(() => shell.checkPython(path)).then((c) => {
      if (!live) return
      setCheck(c ?? null)
      if (c) onCheckedRef.current?.(c)
    })
    return () => {
      live = false
    }
  }, [path, round, onCheckedRef])

  useEffect(() => {
    let off: (() => void) | undefined
    void listen<BootstrapProgress>('bootstrap://progress', (p) => {
      if (!['done', 'error', 'cancelled'].includes(p.stage)) {
        setInstalling(p)
        return
      }
      setInstalling(null)
      if (p.stage === 'error') {
        setFailure(p.error)
        if (p.error?.code === 'runtime.layer_failed') setLayerFailed(true)
      }
      // check again once the install is over
      setRound((r) => Math.max(r, 0) + 1)
    }).then((f) => (off = f))
    return () => off?.()
  }, [])

  const install = async () => {
    if (!check) return
    const n = check.missing.length
    const ok = await confirmDialog({ title: t('rt.layer_title', { n }), body: t('rt.layer_body', { path }), confirm: t('rt.layer_go'), cancel: t('common.cancel') })
    if (!ok) return
    setFailure(null)
    setLayerFailed(false)
    await attempt(() => shell.installRuntimeRequirements(path))
  }

  return (
    <div className="rt-check">
      <div className="mono rt-path">{path}</div>
      {!check ? (
        <div className="row muted">
          <Loader2 className="spin" />
          {t('rt.checking')}
        </div>
      ) : (
        <>
          <div className="rt-facts">
            <span className={check.version ? 'good-text' : 'critical-text'}>
              {check.version ? <CheckCircle2 /> : <AlertTriangle />} Python {check.version ?? '–'}
            </span>
            <span className={check.torch ? 'good-text' : 'warn-text'}>
              {check.torch ? <CheckCircle2 /> : <AlertTriangle />} torch {check.torch?.version ?? t('rt.missing')}
              {check.torch && (check.torch.cudaAvailable ? ` · CUDA ${check.torch.cuda}` : ` · ${t('rt.cpu_only')}`)}
            </span>
            <span className={check.missing.length === 0 ? 'good-text' : 'warn-text'}>
              {check.missing.length === 0 ? <CheckCircle2 /> : <AlertTriangle />}
              {check.missing.length === 0 ? t('rt.packages_ok') : t('rt.packages_missing', { n: check.missing.length })}
            </span>
          </div>
          {check.missing.length > 0 && <div className="mono muted rt-missing">{check.missing.join(' · ')}</div>}
          {check.layered && check.ok && (
            <div className="rt-note">
              <Layers />
              {t('rt.layered_note')}
            </div>
          )}
          {check.error && check.error.code !== 'python.packages_missing' && <div className="critical-text">{t.msg(check.error)}</div>}
          {failure && <div className="critical-text">{t.msg({ code: failure.code }, failure.message)}</div>}
          {installing ? (
            <div className="col" style={{ gap: 6 }}>
              <span className="muted">{installing.detail ?? installing.message}</span>
              <Progress value={installing.progress} />
            </div>
          ) : (
            <>
              <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
                <button className={clsx('btn btn-sm', check.ok && 'btn-primary')} onClick={() => onUse(path)} disabled={!check.ok || busy}>
                  {busy ? <Loader2 className="spin" /> : <CheckCircle2 />}
                  {t('rt.use')}
                </button>
                {check.version && check.missing.length > 0 && !layerFailed && (
                  <button className={clsx('btn btn-sm', check.error?.code === 'python.packages_missing' && 'btn-primary')} onClick={install}>
                    <Layers />
                    {t('rt.layer')}
                  </button>
                )}
                {layerFailed && onManaged && (
                  <button className="btn btn-sm btn-primary" onClick={onManaged} disabled={busy}>
                    {t('rt.use_managed')}
                  </button>
                )}
                <button className="btn btn-sm btn-ghost" onClick={onCancel} disabled={busy}>
                  {t('common.cancel')}
                </button>
              </div>
              {check.version && check.missing.length > 0 && !layerFailed && <span className="muted rt-hint">{t('rt.layer_hint')}</span>}
            </>
          )}
        </>
      )}
    </div>
  )
}

/** The best environment found: a usable one with a CUDA torch, else any with torch. */
function bestFound(nearby: PythonCandidate[] | null, probes: Record<string, PythonCheck | null>) {
  const checked = (nearby ?? []).map((c) => ({ c, check: probes[c.path] })).filter((x): x is { c: PythonCandidate; check: PythonCheck } => !!x.check?.torch && !!x.check.version)
  const supported = checked.filter((x) => !x.check.error || x.check.error.code === 'python.packages_missing')
  return supported.find((x) => x.check.torch?.cudaAvailable) ?? supported[0] ?? null
}

/**
 * The environments on this computer (conda first) and, in the background, a check of the first
 * few: Python version, torch and whether it sees the GPU. Checking only reads them.
 */
export function useFoundPythons() {
  const [nearby, setNearby] = useState<PythonCandidate[] | null>(null)
  const [probes, setProbes] = useState<Record<string, PythonCheck | null>>({})

  useEffect(() => {
    let live = true
    shell
      .findPythons(null)
      .then((list) => live && setNearby(list))
      .catch(() => live && setNearby([]))
    return () => {
      live = false
    }
  }, [])

  useEffect(() => {
    if (!nearby?.length) return
    let live = true
    void (async () => {
      for (const c of nearby.slice(0, PROBE_LIMIT)) {
        if (!live) return
        setProbes((p) => ({ ...p, [c.path]: null }))
        const check = await shell.checkPython(c.path).catch(() => undefined)
        if (!live) return
        setProbes((p) => {
          const next = { ...p }
          if (check) next[c.path] = check
          else delete next[c.path]
          return next
        })
      }
    })()
    return () => {
      live = false
    }
  }, [nearby])

  const best = useMemo(() => bestFound(nearby, probes), [nearby, probes])
  const update = useCallback((path: string, check: PythonCheck) => setProbes((p) => (path in p ? { ...p, [path]: check } : p)), [])
  return { nearby, probes, best, update }
}

export type FoundPythons = ReturnType<typeof useFoundPythons>

/** What the background check found, as chips on a list row. */
function ProbeTags({ check, best }: { check: PythonCheck | null | undefined; best: boolean }) {
  const t = useT()
  if (check === undefined) return null
  if (check === null) return <Loader2 className="spin xp-probe" aria-label={t('xp.checking')} />
  if (!check.version) return <span className="chip">{t('xp.unusable')}</span>
  if (check.error?.code === 'python.version_unsupported') return <span className="chip">{t('xp.old_python', { version: check.version })}</span>
  if (!check.torch) return <span className="chip">{t('xp.no_torch')}</span>
  return (
    <>
      {best && <span className="chip accent">{t('common.recommended')}</span>}
      <span className="chip good">
        torch {check.torch.version.split('+')[0]} · {check.torch.cudaAvailable ? 'CUDA' : 'CPU'}
      </span>
      {check.missing.length > 0 && <span className="chip warning">{t('xp.missing_n', { n: check.missing.length })}</span>}
    </>
  )
}

/**
 * "I already have DDSP-SVC": point at its folder (or a bundle's) and use the Python inside, or
 * pick one of the environments found on this computer, or the executable itself.
 */
export function ExistingPython({
  onUse,
  onManaged,
  busy,
  onFolder,
  found,
}: {
  onUse: (path: string) => void
  /** use Stellaria's own runtime instead (offered when a layer does not work with a Python) */
  onManaged?: () => void
  busy?: boolean
  onFolder?: (folder: string) => void
  /** environments already being looked for by the caller (onboarding starts before this opens) */
  found?: FoundPythons
}) {
  const own = useFoundPythons()
  const { nearby, probes, best, update } = found ?? own
  return <ExistingPythonView nearby={nearby} probes={probes} best={best?.c.path ?? null} onChecked={update} onUse={onUse} onManaged={onManaged} busy={busy} onFolder={onFolder} />
}

function ExistingPythonView({
  nearby,
  probes,
  best,
  onChecked,
  onUse,
  onManaged,
  busy,
  onFolder,
}: {
  nearby: PythonCandidate[] | null
  probes: Record<string, PythonCheck | null>
  best: string | null
  onChecked: (path: string, check: PythonCheck) => void
  onUse: (path: string) => void
  onManaged?: () => void
  busy?: boolean
  onFolder?: (folder: string) => void
}) {
  const t = useT()
  const [folder, setFolder] = useState<{ path: string; found: PythonCandidate[] } | null>(null)
  const [chosen, setChosen] = useState<string | null>(null)

  const chooseFolder = async () => {
    const path = await pickFolder()
    if (!path) return
    onFolder?.(path)
    const found = (await attempt(() => shell.findPythons(path))) ?? []
    setFolder({ path, found })
    setChosen(found.length === 1 ? found[0].path : null)
  }

  const chooseFile = async () => {
    const path = await pickExecutable()
    if (path) setChosen(path)
  }

  const list = (items: PythonCandidate[], probed: boolean) => (
    <div className="xp-list">
      {items.map((c) => (
        <button key={c.path} className={clsx('xp-item', chosen === c.path && 'on')} onClick={() => setChosen(c.path)}>
          <span className="xp-item-name">{c.label}</span>
          <span className="xp-item-tags">
            <span className="chip">{t(`xp.source.${c.source}` as const)}</span>
            {probed && <ProbeTags check={probes[c.path]} best={best === c.path} />}
          </span>
          <span className="mono muted xp-item-path">{c.path}</span>
          <ChevronRight />
        </button>
      ))}
    </div>
  )

  return (
    <div className="xp">
      <div className="xp-actions">
        <button className="btn" onClick={chooseFolder} disabled={busy}>
          <FolderOpen />
          {t('xp.pick_folder')}
        </button>
        <button className="btn btn-ghost" onClick={chooseFile} disabled={busy}>
          <FileSearch />
          {t('xp.pick_file')}
        </button>
      </div>
      {folder && (
        <div className="xp-group">
          <span className="eyebrow">{t('xp.in_folder')}</span>
          <span className="mono muted xp-folder">{folder.path}</span>
          {folder.found.length ? list(folder.found, false) : <p className="muted xp-empty">{t('xp.none_in_folder')}</p>}
        </div>
      )}
      {chosen && (
        <PythonCheckPanel
          key={chosen}
          path={chosen}
          initial={probes[chosen]}
          onChecked={(check) => onChecked(chosen, check)}
          onUse={onUse}
          onManaged={onManaged}
          onCancel={() => setChosen(null)}
          busy={busy}
        />
      )}
      <div className="xp-group">
        <span className="eyebrow">{t('xp.nearby')}</span>
        {nearby === null ? (
          <div className="row muted">
            <Loader2 className="spin" />
            {t('xp.searching')}
          </div>
        ) : nearby.length ? (
          <>
            {list(nearby, true)}
            <span className="muted xp-note">{t('xp.read_only')}</span>
          </>
        ) : (
          <p className="muted xp-empty">{t('xp.none_nearby')}</p>
        )}
      </div>
    </div>
  )
}

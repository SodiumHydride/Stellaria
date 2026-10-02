import { AlertTriangle, Layers, Loader2, RotateCcw } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { Progress } from '../components/ui'
import { useT } from '../i18n'
import { attempt } from '../lib/actions'
import { listen, shell, type BootstrapProgress, type PythonCheck } from '../shell/bridge'
import { useBoot } from './boot'

type Diagnosis = 'checking' | 'gone' | 'version' | 'torch' | 'packages' | 'fine' | 'other'

function diagnose(check: PythonCheck | null | undefined): Diagnosis {
  if (check === undefined) return 'checking'
  if (check === null) return 'other'
  switch (check.error?.code) {
    case undefined:
      return 'fine'
    case 'python.not_found':
      return 'gone'
    case 'python.version_unsupported':
      return 'version'
    case 'python.torch_missing':
      return 'torch'
    case 'python.packages_missing':
      return 'packages'
    default:
      return 'other'
  }
}

/**
 * The backend did not start on the user's own Python: what changed in it, and one-click ways
 * back — add what it lacks again (into Stellaria's folder, never theirs), the app's own runtime,
 * or another Python. Their environment is only ever read.
 */
export function RuntimeRepair({ copyLog, log }: { copyLog: () => void; log: string | null }) {
  const t = useT()
  const shellState = useBoot((s) => s.shell)
  const setShell = useBoot((s) => s.setShell)
  const retry = useBoot((s) => s.retry)
  const connect = useBoot((s) => s.connect)
  const chooseRuntime = useBoot((s) => s.chooseRuntime)
  const path = shellState?.runtime.pythonPath ?? ''
  const managedReady = !!shellState?.managedReady
  // undefined while checking, null when the check itself could not run
  const [check, setCheck] = useState<PythonCheck | null | undefined>(undefined)
  const [round, setRound] = useState(0)
  const [progress, setProgress] = useState<BootstrapProgress | null>(null)
  const [failure, setFailure] = useState<BootstrapProgress['error']>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!path) return
    let live = true
    setCheck(undefined)
    shell
      .checkPython(path)
      .then((c) => live && setCheck(c))
      .catch(() => live && setCheck(null))
    return () => {
      live = false
    }
  }, [path, round])

  useEffect(() => {
    let off: (() => void) | undefined
    void listen<BootstrapProgress>('bootstrap://progress', (p) => {
      if (!['done', 'error', 'cancelled'].includes(p.stage)) {
        setProgress(p)
        return
      }
      setProgress(null)
      if (p.stage === 'error') setFailure(p.error)
      if (p.stage === 'done') void retry()
      else setRound((r) => r + 1)
    }).then((f) => (off = f))
    return () => off?.()
  }, [retry])

  const run = (fn: () => Promise<unknown>) =>
    void attempt(async () => {
      setBusy(true)
      setFailure(null)
      try {
        await fn()
      } finally {
        setBusy(false)
      }
    })
  const refill = () => run(() => shell.installRuntimeRequirements(path))
  const useManaged = () =>
    run(async () => {
      setShell(await shell.setRuntimePython(null))
      await connect()
    })
  const restart = () => run(() => retry())

  const kind = diagnose(check)
  const working = busy || !!progress
  const body =
    kind === 'checking'
      ? t('repair.checking')
      : kind === 'gone'
        ? t('repair.gone', { path })
        : kind === 'version'
          ? t('repair.version', { version: check?.version ?? '?' })
          : kind === 'torch'
            ? t('repair.torch')
            : kind === 'packages'
              ? t('repair.packages', { n: check?.missing.length ?? 0 })
              : kind === 'fine'
                ? t('repair.fine')
                : check?.error
                  ? t.msg(check.error)
                  : t('repair.unknown')

  return (
    <>
      <BootCardTitle>{kind === 'fine' ? t('conn.failed_title') : t('repair.title')}</BootCardTitle>
      <p className="secondary" style={{ marginTop: 10 }}>
        {kind === 'checking' && <Loader2 className="spin repair-spin" />}
        {body}
      </p>
      <p className="mono muted repair-path">{path}</p>
      {kind !== 'checking' && kind !== 'fine' && <p className="muted repair-note">{t('repair.never_changed')}</p>}
      {progress && (
        <div className="col" style={{ gap: 6, marginTop: 12 }}>
          <span className="muted">{progress.detail ?? progress.message}</span>
          <Progress value={progress.progress} />
        </div>
      )}
      {failure && <div className="critical-text" style={{ marginTop: 10 }}>{t.msg({ code: failure.code }, failure.message)}</div>}
      <div className="row" style={{ marginTop: 18, gap: 8, flexWrap: 'wrap' }}>
        {(kind === 'packages' || kind === 'fine') && (
          <button className="btn btn-primary" disabled={working} onClick={refill}>
            {working ? <Loader2 className="spin" /> : <Layers />}
            {kind === 'packages' ? t('repair.refill') : t('repair.rebuild')}
          </button>
        )}
        {managedReady && (
          <button className={kind === 'packages' || kind === 'fine' ? 'btn' : 'btn btn-primary'} disabled={working} onClick={useManaged}>
            {t('repair.use_managed')}
          </button>
        )}
        <button className={!managedReady && kind !== 'packages' && kind !== 'fine' ? 'btn btn-primary' : 'btn'} disabled={working} onClick={chooseRuntime}>
          {managedReady ? t('repair.other_python') : t('repair.choose_runtime')}
        </button>
        <span className="spacer" />
        <button className="btn btn-ghost" disabled={working} onClick={restart}>
          <RotateCcw />
          {t('conn.restart')}
        </button>
        <button className="btn btn-ghost" onClick={copyLog}>
          {t('conn.copy_log')}
        </button>
      </div>
      {log !== null && <div className="boot-log">{log}</div>}
    </>
  )
}

export function BootCardTitle({ children }: { children: ReactNode }) {
  return (
    <div className="row" style={{ gap: 12 }}>
      <AlertTriangle style={{ color: 'var(--warning)', width: 24, height: 24 }} />
      <h2 className="section-title" style={{ fontSize: 'var(--fs-xl)' }}>
        {children}
      </h2>
    </div>
  )
}

import { RotateCcw } from 'lucide-react'
import { useState } from 'react'
import { Backdrop } from '../components/Backdrop'
import { WindowChrome } from '../components/WindowChrome'
import { AppTile } from '../components/Logo'
import { useT } from '../i18n'
import { isTauri, shell } from '../shell/bridge'
import { useBoot } from './boot'
import { BootCardTitle, RuntimeRepair } from './RuntimeRepair'
import { useSetupCloseRequests } from './useCloseRequests'

/** Shown while connecting to the backend, and when it can't be reached. */
export function BootScreen() {
  const t = useT()
  const phase = useBoot((s) => s.phase)
  const error = useBoot((s) => s.error)
  const retry = useBoot((s) => s.retry)
  const [log, setLog] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  // the user's own Python: diagnose it and offer the ways back
  const custom = useBoot((s) => s.shell?.runtime.mode === 'custom')
  useSetupCloseRequests()

  const copyLog = async () => {
    const lines = isTauri ? await shell.backendLogs(400).catch(() => []) : []
    const text = [error ?? '', ...lines].join('\n')
    setLog(text)
    await navigator.clipboard.writeText(text).catch(() => {})
  }

  return (
    <>
      <Backdrop />
      <WindowChrome />
      <div className="boot">
        {phase !== 'failed' ? (
          <>
            <AppTile size={84} running />
            <div className="boot-text shimmer-text">{t('conn.connecting')}</div>
          </>
        ) : (
          <div className="glass boot-card">
            {custom ? (
              <RuntimeRepair copyLog={copyLog} log={log === null ? null : log || error} />
            ) : (
              <>
                <BootCardTitle>{t('conn.failed_title')}</BootCardTitle>
                <p className="secondary" style={{ marginTop: 10 }}>
                  {t('conn.failed_body')}
                </p>
                <div className="row" style={{ marginTop: 18 }}>
                  <button
                    className="btn btn-primary"
                    disabled={busy}
                    onClick={async () => {
                      setBusy(true)
                      await retry()
                      setBusy(false)
                    }}
                  >
                    <RotateCcw className={busy ? 'spin' : undefined} />
                    {t('conn.restart')}
                  </button>
                  <button className="btn" onClick={copyLog}>
                    {t('conn.copy_log')}
                  </button>
                </div>
                {log !== null && <div className="boot-log">{log || error}</div>}
              </>
            )}
          </div>
        )}
      </div>
    </>
  )
}

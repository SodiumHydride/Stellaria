import { AlertTriangle, ArrowLeft, Copy, RotateCcw } from 'lucide-react'
import { useNavigate, useRouteError } from 'react-router-dom'
import { useT } from '../i18n'
import { toast } from '../state/ui'

/** Something on this screen crashed: say so calmly and offer the ways out. */
export function RouteError() {
  const t = useT()
  const nav = useNavigate()
  const error = useRouteError()
  const text = error instanceof Error ? `${error.message}\n\n${error.stack ?? ''}` : String(error)
  return (
    <div className="route-error">
      <div className="route-error-card glass-strong">
        <AlertTriangle className="route-error-icon" />
        <h2>{t('crash.title')}</h2>
        <p className="muted">{t('crash.body')}</p>
        <pre className="route-error-text mono">{error instanceof Error ? error.message : String(error)}</pre>
        <div className="row" style={{ gap: 8, justifyContent: 'center' }}>
          <button className="btn btn-primary" onClick={() => window.location.reload()}>
            <RotateCcw />
            {t('crash.reload')}
          </button>
          <button
            className="btn"
            onClick={() => {
              nav('/voices')
              window.setTimeout(() => window.location.reload(), 0)
            }}
          >
            <ArrowLeft />
            {t('crash.home')}
          </button>
          <button
            className="btn btn-ghost"
            onClick={() => {
              void navigator.clipboard?.writeText(text)
              toast({ kind: 'success', title: t('crash.copied') })
            }}
          >
            <Copy />
            {t('crash.copy')}
          </button>
        </div>
      </div>
    </div>
  )
}

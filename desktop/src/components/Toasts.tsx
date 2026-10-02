import clsx from 'clsx'
import { AlertTriangle, CheckCircle2, Info, X } from 'lucide-react'
import { useState } from 'react'
import { useT } from '../i18n'
import { useUi } from '../state/ui'
import './toasts.css'

export function Toasts() {
  const toasts = useUi((s) => s.toasts)
  const dismiss = useUi((s) => s.dismiss)
  const t = useT()
  const [open, setOpen] = useState<number | null>(null)
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((x) => (
        <div key={x.id} className={clsx('toast glass-strong', x.kind)}>
          <span className="toast-icon">{x.kind === 'error' ? <AlertTriangle /> : x.kind === 'success' ? <CheckCircle2 /> : <Info />}</span>
          <div className="toast-body">
            <div className="toast-title">{x.title}</div>
            {x.body && <div className="toast-text">{x.body}</div>}
            {x.detail && open === x.id && <pre className="toast-detail">{x.detail}</pre>}
            <div className="row" style={{ gap: 6, marginTop: x.action || x.detail ? 6 : 0 }}>
              {x.action && (
                <button
                  className="btn btn-sm"
                  onClick={() => {
                    x.action!.run()
                    dismiss(x.id)
                  }}
                >
                  {x.action.label}
                </button>
              )}
              {x.detail && (
                <>
                  <button className="btn btn-sm btn-ghost" onClick={() => setOpen(open === x.id ? null : x.id)}>
                    {open === x.id ? t('common.hide_details') : t('common.show_details')}
                  </button>
                  <button className="btn btn-sm btn-ghost" onClick={() => navigator.clipboard.writeText(`${x.title}\n${x.body ?? ''}\n${x.detail}`)}>
                    {t('common.copy_error')}
                  </button>
                </>
              )}
            </div>
          </div>
          <button className="icon-btn sm" onClick={() => dismiss(x.id)} aria-label={t('common.close')}>
            <X />
          </button>
        </div>
      ))}
    </div>
  )
}

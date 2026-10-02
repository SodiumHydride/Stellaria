import { AlertTriangle, ChevronRight, X } from 'lucide-react'
import { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useResources } from '../api/hooks'
import { useT } from '../i18n'

/**
 * For people who chose "later" at setup: a quiet reminder that the required models are still
 * missing, with the way to get them (download, or import from a folder they have).
 */
export function MissingModelsBar() {
  const t = useT()
  const nav = useNavigate()
  const { pathname } = useLocation()
  const { data: resources } = useResources()
  const [hidden, setHidden] = useState(false)
  const missing = (resources ?? []).filter((r) => r.required && r.status !== 'ready')
  const busy = missing.some((r) => r.status === 'downloading' || r.status === 'installing')
  if (hidden || busy || missing.length === 0 || pathname.startsWith('/resources') || pathname.startsWith('/settings')) return null
  return (
    <div className="missing-models glass-strong" role="status">
      <AlertTriangle />
      <span>{t('mm.text', { n: missing.length })}</span>
      <button className="btn btn-sm" onClick={() => nav('/resources')}>
        {t('mm.go')}
        <ChevronRight />
      </button>
      <button className="icon-btn sm" onClick={() => setHidden(true)} aria-label={t('common.close')}>
        <X />
      </button>
    </div>
  )
}

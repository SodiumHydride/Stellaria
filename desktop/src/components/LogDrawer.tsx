import { Copy, Search, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { api } from '../api/client'
import { onServerEvent, subscribeTopic } from '../api/live'
import type { Job } from '../api/types'
import { useT } from '../i18n'
import './log-drawer.css'

/** Readable, searchable, copyable log of a job (live for running jobs). Replaces any console window. */
export function LogDrawer({ job, title, source, onClose }: { job?: Job; title?: string; source?: () => Promise<string[]>; onClose: () => void }) {
  const t = useT()
  const [lines, setLines] = useState<string[]>([])
  const [filter, setFilter] = useState('')
  const [copied, setCopied] = useState(false)
  const bodyRef = useRef<HTMLDivElement>(null)
  const stick = useRef(true)

  useEffect(() => {
    if (!source) return
    let cancelled = false
    const load = () => source().then((l) => !cancelled && setLines(l)).catch(() => {})
    load()
    const timer = window.setInterval(load, 3000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [source])

  useEffect(() => {
    if (!job || source) return
    let cancelled = false
    api.get<{ lines: string[] }>(`/jobs/${job.id}/log`, { tail: 2000 }).then((r) => !cancelled && setLines(r.lines)).catch(() => {})
    const off = subscribeTopic(`job.log:${job.id}`)
    const offEv = onServerEvent((ev) => {
      if (ev.type === 'job.log' && ev.data.jobId === job.id) setLines((prev) => [...prev, ...ev.data.lines].slice(-5000))
    })
    return () => {
      cancelled = true
      off()
      offEv()
    }
  }, [job, source])

  useEffect(() => {
    const el = bodyRef.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [lines])

  const shown = useMemo(() => (filter ? lines.filter((l) => l.toLowerCase().includes(filter.toLowerCase())) : lines), [lines, filter])

  const copy = async () => {
    const header = job ? `${t.msg(job.title, job.type)} (${job.id})\n${job.error ? `${job.error.code}: ${job.error.message}\n${job.error.detail}\n` : ''}` : ''
    await navigator.clipboard.writeText(header + lines.join('\n'))
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }

  return createPortal(
    <div className="log-drawer glass-strong" role="dialog" aria-label={t('dock.view_log')}>
      <div className="log-head">
        <span className="section-title">{title ?? (job ? t.msg(job.title, job.type) : t('dock.view_log'))}</span>
        <span className="spacer" />
        <div className="input-wrap" style={{ width: 200 }}>
          <Search />
          <input className="input" style={{ height: 30 }} placeholder={t('common.search')} value={filter} onChange={(e) => setFilter(e.target.value)} />
        </div>
        <button className="btn btn-sm" onClick={copy}>
          <Copy />
          {copied ? t('common.copied') : t('common.copy')}
        </button>
        <button className="icon-btn sm" onClick={onClose} aria-label={t('common.close')}>
          <X />
        </button>
      </div>
      {job?.error && (
        <div className="log-error">
          <strong>{t.msg({ code: job.error.code, params: job.error.params }, job.error.message)}</strong>
        </div>
      )}
      <div
        className="log-body mono"
        ref={bodyRef}
        onScroll={(e) => {
          const el = e.currentTarget
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
        }}
      >
        {shown.map((l, i) => (
          <div key={i} className={/error|traceback|exception/i.test(l) ? 'log-line err' : /warn/i.test(l) ? 'log-line warn' : 'log-line'}>
            {l}
          </div>
        ))}
      </div>
    </div>,
    document.body,
  )
}

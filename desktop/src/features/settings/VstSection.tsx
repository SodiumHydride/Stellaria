import { useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { AlertTriangle, AudioLines, CheckCircle2, ChevronDown, Download, FolderPlus, Loader2, RefreshCw, Search, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import { useActiveJobs, useResources, useVstPlugins, useVstStatus } from '../../api/hooks'
import { keys } from '../../api/keys'
import type { Job, Resource, VstPlugin, VstStatus } from '../../api/types'
import { Progress, Tip } from '../../components/ui'
import { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { percent, relativeTime } from '../../lib/format'
import { waitForJob } from '../../lib/jobs'
import { isTauri, pickFolder } from '../../shell/bridge'
import { toast } from '../../state/ui'
import { Group, Row } from './parts'

/** Settings → 效果器插件 (VST3 / AU): the host add-on, plugin folders, scanning, what was found. */
export function VstSection() {
  const t = useT()
  const qc = useQueryClient()
  const { data: status } = useVstStatus()
  const { data: plugins } = useVstPlugins()
  const { data: resources } = useResources()
  const scanJob = useActiveJobs((j) => j.type === 'vst_scan')[0] ?? null
  const host = resources?.find((r) => r.id === 'vsthost') ?? null
  const [running, setRunning] = useState(false)
  const hostReady = host?.status === 'ready'

  const refresh = () => {
    void qc.invalidateQueries({ queryKey: keys.vstStatus })
    void qc.invalidateQueries({ queryKey: keys.vstPlugins })
  }

  // the add-on just finished installing: the host's status changes with it
  useEffect(() => {
    if (!hostReady) return
    void qc.invalidateQueries({ queryKey: keys.vstStatus })
  }, [hostReady, qc])

  const scan = async (full: boolean) => {
    setRunning(true)
    const job = await attempt(() => api.post<Job>('/vst/scan', { full }))
    if (!job) {
      setRunning(false)
      return
    }
    const done = await attempt(() => waitForJob(job.id))
    setRunning(false)
    refresh()
    if (done?.status === 'succeeded') {
      const r = (done.result ?? {}) as { found?: number; failed?: number }
      const found = r.found ?? 0
      toast({ kind: 'success', title: r.failed ? t('vst.scan_done_failed', { found, failed: r.failed }) : t('vst.scan_done', { found }) })
    } else if (done?.status === 'failed') {
      toast({ kind: 'error', title: t.msg({ code: done.error?.code ?? 'common.error' }, done.error?.message ?? t('common.error')), detail: done.error?.detail || undefined })
    }
  }

  const head = (body: React.ReactNode) => (
    <div id="vst" className="set-anchor">
      <Group icon={<AudioLines />} title={t('vst.title')}>
        {body}
      </Group>
    </div>
  )

  if (status === undefined) return head(<div className="vst-loading skeleton" />)
  if (status === null) return head(<p className="muted set-text">{t('vst.unsupported')}</p>)

  const scanning = status.scanning || !!scanJob || running
  return head(
    <>
      <HostRow status={status} host={host} />
      <FoldersRow status={status} />
      {status.hostInstalled && (
      <Row label={t('vst.scan')} hint={scanning ? undefined : status.lastScan ? t('vst.last_scan', { time: relativeTime(status.lastScan, t.lang) }) : t('vst.never_scanned')}>
        <div className="row" style={{ gap: 8 }}>
          <Tip label={t('vst.scan_full_hint')}>
            <span className="set-tip-wrap">
              <button className="btn btn-sm btn-ghost" onClick={() => scan(true)} disabled={scanning}>
                {t('vst.scan_full')}
              </button>
            </span>
          </Tip>
          <Tip label={t('vst.scan_hint')}>
            <span className="set-tip-wrap">
              <button className="btn btn-sm" onClick={() => scan(false)} disabled={scanning}>
                {scanning ? <Loader2 className="spin" /> : <RefreshCw />}
                {scanning ? t('vst.scanning') : t('vst.scan')}
              </button>
            </span>
          </Tip>
        </div>
      </Row>
      )}
      {scanning && (
        <div className="vst-progress">
          <Progress value={scanJob?.progress ?? null} />
          {(scanJob?.stage || scanJob?.progress != null) && (
            <span className="muted tabular vst-progress-text">
              {scanJob.stage ? t.msg(scanJob.stage) : ''}
              {scanJob.progress != null && `${scanJob.stage ? ' · ' : ''}${percent(scanJob.progress)}`}
            </span>
          )}
        </div>
      )}
      {status.hostInstalled && <PluginList plugins={plugins ?? []} />}
    </>,
  )
}

function HostRow({ status, host }: { status: VstStatus; host: Resource | null }) {
  const t = useT()
  const busy = host?.status === 'downloading' || host?.status === 'installing'
  const install = () => attempt(() => api.post('/resources/vsthost/install', {}))

  if (status.hostInstalled) {
    return (
      <Row label={t('vst.host')} hint={`pedalboard${status.hostVersion ? ` ${status.hostVersion}` : ''} · GPL-3.0`}>
        <div className="row" style={{ gap: 6 }}>
          {status.formats.map((f) => (
            <span key={f} className="chip">
              {f === 'au' ? 'AU' : 'VST3'}
            </span>
          ))}
          <span className="chip good">
            <CheckCircle2 />
            {t('vst.host_ready')}
          </span>
        </div>
      </Row>
    )
  }
  return (
    <Row label={t('vst.host')} hint={host?.status === 'failed' && host.error ? <span className="critical-text">{t.msg(host.error)}</span> : t('vst.host_about')}>
      {busy ? (
        <div className="vst-install-progress">
          <Progress value={host?.progress ?? null} />
          <span className="muted tabular">{host?.progress != null ? percent(host.progress) : t(`res.status.${host!.status}` as const)}</span>
        </div>
      ) : (
        <button className="btn btn-sm btn-primary" onClick={install}>
          <Download />
          {host?.status === 'failed' ? t('common.retry') : t('vst.install')}
        </button>
      )}
    </Row>
  )
}

function FoldersRow({ status }: { status: VstStatus }) {
  const t = useT()
  const qc = useQueryClient()
  const defaults = status.defaultFolders
  const custom = status.folders.filter((f) => !defaults.includes(f))

  const save = async (folders: string[]) => {
    const next = await attempt(() => api.patch<VstStatus>('/vst/settings', { folders }))
    if (next) qc.setQueryData(keys.vstStatus, next)
  }
  const add = async () => {
    const picked = isTauri ? await pickFolder() : window.prompt(t('vst.folder_path'), '')
    const path = picked?.trim()
    if (!path || defaults.includes(path) || custom.includes(path)) return
    await save([...custom, path])
  }

  return (
    <div className="vst-folders">
      <div className="vst-folders-head">
        <Tip label={t('vst.folders_hint')}>
          <span className="has-hint">{t('vst.folders')}</span>
        </Tip>
        <span className="spacer" />
        <button className="btn btn-sm" onClick={add}>
          <FolderPlus />
          {t('vst.folder_add')}
        </button>
      </div>
      <ul className="vst-folder-list">
        {defaults.map((f) => (
          <li key={f} className="vst-folder">
            <span className="vst-folder-path mono" title={f}>
              {f}
            </span>
            <span className="vst-folder-tag">{t('vst.folder_default')}</span>
          </li>
        ))}
        {custom.map((f) => (
          <li key={f} className="vst-folder">
            <span className="vst-folder-path mono" title={f}>
              {f}
            </span>
            <Tip label={t('fx.remove')}>
              <button className="icon-btn sm" onClick={() => save(custom.filter((x) => x !== f))} aria-label={t('fx.remove')}>
                <X />
              </button>
            </Tip>
          </li>
        ))}
      </ul>
    </div>
  )
}

function PluginList({ plugins }: { plugins: VstPlugin[] }) {
  const t = useT()
  const [query, setQuery] = useState('')
  const [failedOpen, setFailedOpen] = useState(false)
  const [atEnd, setAtEnd] = useState(false)
  const ok = plugins
    .filter((p) => p.status === 'ok')
    .sort((a, b) => (a.category === b.category ? a.name.localeCompare(b.name) : a.category === 'effect' ? -1 : 1))
  const failed = plugins.filter((p) => p.status === 'failed').sort((a, b) => a.name.localeCompare(b.name))
  const needle = query.trim().toLowerCase()
  const shown = needle ? ok.filter((p) => p.name.toLowerCase().includes(needle) || (p.vendor ?? '').toLowerCase().includes(needle)) : ok

  return (
    <div className="vst-plugins">
      {ok.length === 0 ? (
        <p className="muted vst-empty">{t('vst.empty')}</p>
      ) : (
        <>
          <div className="vst-plugins-head">
            <span className="eyebrow">{t('vst.count', { n: ok.length })}</span>
            <span className="spacer" />
            <div className="input-wrap vst-search">
              <Search />
              <input className="input" value={query} placeholder={t('vst.search')} onChange={(e) => setQuery(e.target.value)} aria-label={t('vst.search')} />
            </div>
          </div>
          <div
            className={clsx('vst-list', !atEnd && shown.length > 8 && 'more')}
            onScroll={(e) => setAtEnd(e.currentTarget.scrollTop + e.currentTarget.clientHeight >= e.currentTarget.scrollHeight - 4)}
          >
            {shown.length === 0 && <p className="muted vst-empty">{t('vst.no_match')}</p>}
            {shown.map((p) => (
              <div key={p.id} className={clsx('vst-plugin', p.category === 'instrument' && 'is-instrument')}>
                <span className="vst-fmt">{p.format === 'au' ? 'AU' : 'VST3'}</span>
                <div className="vst-plugin-main">
                  <span className="vst-plugin-name">{p.name}</span>
                  <span className="vst-plugin-meta">{[p.vendor, p.version].filter(Boolean).join(' · ')}</span>
                </div>
                {p.category === 'instrument' ? (
                  <Tip label={t('vst.instrument_hint')}>
                    <span className="chip vst-instrument">{t('vst.kind_instrument')}</span>
                  </Tip>
                ) : (
                  <span className="muted tabular vst-plugin-params">{t('vst.params', { n: p.params.length })}</span>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {failed.length > 0 && (
        <div className={clsx('vst-failed', failedOpen && 'open')}>
          <button className="vst-failed-toggle" onClick={() => setFailedOpen(!failedOpen)} aria-expanded={failedOpen}>
            <ChevronDown className="vst-failed-chevron" />
            <AlertTriangle className="vst-failed-icon" />
            <span>{t('vst.failed', { n: failed.length })}</span>
          </button>
          {failedOpen && (
            <div className="vst-failed-list">
              {failed.map((p) => (
                <div key={p.id} className="vst-failed-item">
                  <div className="row" style={{ gap: 8 }}>
                    <span className="vst-fmt">{p.format === 'au' ? 'AU' : 'VST3'}</span>
                    <strong className="vst-plugin-name">{p.name}</strong>
                    {p.vendor && <span className="muted vst-plugin-meta">{p.vendor}</span>}
                  </div>
                  <div className="vst-failed-error">{t.msg(p.error, t('vst.failed_unknown'))}</div>
                  <div className="mono muted vst-failed-path" title={p.path}>
                    {p.path}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

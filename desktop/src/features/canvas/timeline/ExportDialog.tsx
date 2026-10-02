import clsx from 'clsx'
import { Check, Download } from 'lucide-react'
import { useState, type CSSProperties } from 'react'
import { api, mediaUrl } from '../../../api/client'
import { useActiveJobs } from '../../../api/hooks'
import type { Job, MixdownRequest, Timeline } from '../../../api/types'
import { LevelMark } from '../../../components/Logo'
import { Dialog, Segmented, Switch } from '../../../components/ui'
import { useT } from '../../../i18n'
import { attempt } from '../../../lib/actions'
import { percent } from '../../../lib/format'
import { waitForJob } from '../../../lib/jobs'
import { isTauri, pickFolder, pickSavePath, revealPath } from '../../../shell/bridge'
import { toast } from '../../../state/ui'
import { useCanvas } from '../context'
import { clock } from './Ruler'

type Format = 'wav' | 'flac' | 'mp3'
type LevelMode = 'none' | 'peak' | 'loudness'

interface Options {
  format: Format
  rate: '44100' | '48000'
  depth: '16' | '24' | '32'
  bitrate: '192' | '320'
  range: 'all' | 'loop'
  level: LevelMode
  lufs: '-14' | '-16' | '-23'
  stems: boolean
}

const KEY = 'stellaria.tl.export'
const DEFAULTS: Options = { format: 'wav', rate: '44100', depth: '24', bitrate: '320', range: 'all', level: 'peak', lufs: '-14', stems: false }

function loadOptions(): Options {
  try {
    return { ...DEFAULTS, ...(JSON.parse(localStorage.getItem(KEY) ?? '{}') as Partial<Options>) }
  } catch {
    return DEFAULTS
  }
}

/** 导出成品: format, sample rate, bit depth / bitrate, the range, level, stems. */
export function ExportDialog({ open, onOpenChange, tl }: { open: boolean; onOpenChange: (o: boolean) => void; tl: Timeline }) {
  const t = useT()
  const { project } = useCanvas()
  const [o, setO] = useState<Options>(loadOptions)
  const [busy, setBusy] = useState(false)
  // the export's job, so the button can show its progress; then a moment of "exported" before closing
  const [jobId, setJobId] = useState<string | null>(null)
  const [finished, setFinished] = useState(false)
  const progress = useActiveJobs((j) => j.id === jobId)[0]?.progress ?? null
  const loop = tl.loop && tl.loop.end > tl.loop.start ? tl.loop : null
  const range = o.range === 'loop' && loop ? 'loop' : 'all'
  const set = <K extends keyof Options>(k: K, v: Options[K]) => {
    const next = { ...o, [k]: v }
    setO(next)
    try {
      localStorage.setItem(KEY, JSON.stringify(next))
    } catch {
      /* private mode */
    }
  }

  const go = async () => {
    let path: string | undefined
    if (isTauri) {
      const picked = o.stems ? await pickFolder() : await pickSavePath(`${project.name}.${o.format}`, [o.format])
      if (!picked) return
      path = picked
    }
    const body: MixdownRequest = {
      format: o.format,
      path,
      sampleRate: Number(o.rate) as 44100 | 48000,
      bitDepth: (o.format === 'flac' && o.depth === '32' ? 24 : Number(o.depth)) as 16 | 24 | 32,
      mp3Bitrate: Number(o.bitrate) as 192 | 320,
      normalize: o.level === 'peak',
      loudness: o.level === 'loudness' ? Number(o.lufs) : null,
      stems: o.stems,
      timeline: tl,
      ...(range === 'loop' && loop ? { start: loop.start, end: loop.end } : {}),
    }
    setBusy(true)
    const job = await attempt(() => api.post<Job>(`/projects/${project.id}/mixdown`, body))
    if (job) setJobId(job.id)
    const done = job ? await attempt(() => waitForJob(job.id)) : undefined
    setBusy(false)
    setJobId(null)
    if (!done) return
    if (done.status !== 'succeeded') {
      if (done.status === 'failed') toast({ kind: 'error', title: t.msg(done.error ? { code: done.error.code, params: done.error.params } : null, t('tl.export_failed')), detail: done.error?.detail ?? undefined })
      return
    }
    setFinished(true)
    await new Promise((r) => window.setTimeout(r, 700))
    setFinished(false)
    onOpenChange(false)
    const r = done.result as { path?: string; downloadUrl?: string; paths?: string[]; folder?: string } | null
    if (!isTauri && r?.downloadUrl) {
      window.open(mediaUrl(r.downloadUrl))
      return
    }
    const where = o.stems ? r?.folder : r?.path
    toast({
      kind: 'success',
      title: o.stems ? t('tl.exported_stems', { n: r?.paths?.length ?? 0 }) : t('tl.exported_mix'),
      body: where,
      action: where ? { label: t('common.open_folder'), run: () => void revealPath(o.stems ? (r?.paths?.[0] ?? where) : where) } : undefined,
    })
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => !busy && !finished && onOpenChange(v)}
      title={t('tl.export_title')}
      description={t('tl.export_desc')}
      actions={
        <>
          <button className="btn btn-ghost" onClick={() => onOpenChange(false)} disabled={busy || finished}>
            {t('common.cancel')}
          </button>
          <button
            className={clsx('btn btn-primary', busy && 'btn-busy', finished && 'btn-done')}
            style={busy && progress != null ? ({ '--p': progress } as CSSProperties) : undefined}
            onClick={go}
            disabled={busy || finished || tl.length <= 0}
          >
            {finished ? <Check className="tick" /> : busy ? <LevelMark size={16} running /> : <Download />}
            {finished ? t('tl.export_done_btn') : busy ? (progress != null ? t('tl.exporting_pct', { p: percent(progress) }) : t('tl.exporting')) : o.stems ? t('tl.export_stems_go') : t('tl.export_go')}
          </button>
        </>
      }
    >
      <div className="tl-export">
        <div className="field">
          <span className="field-label">{t('tl.export_format')}</span>
          <Segmented<Format> value={o.format} onChange={(v) => set('format', v)} options={(['wav', 'flac', 'mp3'] as const).map((f) => ({ value: f, label: f.toUpperCase() }))} />
        </div>
        <div className="tl-export-row">
          <div className="field">
            <span className="field-label">{t('tl.export_rate')}</span>
            <Segmented<Options['rate']>
              size="sm"
              value={o.rate}
              onChange={(v) => set('rate', v)}
              options={[
                { value: '44100', label: '44.1 kHz' },
                { value: '48000', label: '48 kHz' },
              ]}
            />
          </div>
          {o.format === 'mp3' ? (
            <div className="field">
              <span className="field-label">{t('tl.export_bitrate')}</span>
              <Segmented<Options['bitrate']>
                size="sm"
                value={o.bitrate}
                onChange={(v) => set('bitrate', v)}
                options={[
                  { value: '192', label: '192 kbps' },
                  { value: '320', label: '320 kbps' },
                ]}
              />
            </div>
          ) : (
            <div className="field">
              <span className="field-label">{t('tl.export_depth')}</span>
              <Segmented<Options['depth']>
                size="sm"
                value={o.format === 'flac' && o.depth === '32' ? '24' : o.depth}
                onChange={(v) => set('depth', v)}
                options={[
                  { value: '16', label: '16 bit' },
                  { value: '24', label: '24 bit' },
                  ...(o.format === 'wav' ? [{ value: '32' as const, label: t('tl.export_float') }] : []),
                ]}
              />
            </div>
          )}
        </div>
        <div className="field">
          <span className="field-label">{t('tl.export_range')}</span>
          <Segmented<'all' | 'loop'>
            size="sm"
            value={range}
            onChange={(v) => set('range', v)}
            options={[
              { value: 'all', label: t('tl.export_range_all') },
              { value: 'loop', label: loop ? `${t('tl.export_range_loop')} ${clock(loop.start)} – ${clock(loop.end)}` : t('tl.export_range_loop'), title: loop ? undefined : t('tl.export_no_loop') },
            ]}
          />
          {!loop && o.range === 'loop' && <span className="field-hint">{t('tl.export_no_loop')}</span>}
        </div>
        <div className="field">
          <span className="field-label">{t('tl.export_level')}</span>
          <Segmented<LevelMode>
            size="sm"
            value={o.level}
            onChange={(v) => set('level', v)}
            options={[
              { value: 'none', label: t('tl.export_level_none') },
              { value: 'peak', label: t('tl.export_level_peak') },
              { value: 'loudness', label: t('tl.export_level_lufs') },
            ]}
          />
          {o.level === 'loudness' && (
            <div className="tl-export-lufs">
              <Segmented<Options['lufs']>
                size="sm"
                value={o.lufs}
                onChange={(v) => set('lufs', v)}
                options={[
                  { value: '-14', label: '−14 LUFS', title: t('tl.export_lufs14') },
                  { value: '-16', label: '−16 LUFS', title: t('tl.export_lufs16') },
                  { value: '-23', label: '−23 LUFS', title: t('tl.export_lufs23') },
                ]}
              />
            </div>
          )}
          <span className="field-hint">{t(o.level === 'none' ? 'tl.export_level_none_hint' : o.level === 'peak' ? 'tl.export_level_peak_hint' : 'tl.export_level_lufs_hint')}</span>
        </div>
        <label className="tl-export-switch">
          <Switch checked={o.stems} onChange={(v) => set('stems', v)} label={t('tl.export_stems')} />
          <span className="col" style={{ gap: 2 }}>
            <span>{t('tl.export_stems')}</span>
            <span className="field-hint">{t(isTauri ? 'tl.export_stems_hint' : 'tl.export_stems_hint_web')}</span>
          </span>
        </label>
      </div>
    </Dialog>
  )
}

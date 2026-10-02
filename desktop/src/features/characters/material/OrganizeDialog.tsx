import clsx from 'clsx'
import { ChevronDown, Disc3, Mic, Scissors, SquareStack } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../../api/client'
import { useCleanupModels, useOrganizeDefaults, useResources } from '../../../api/hooks'
import type { Character, OrganizeOptions, RawImport } from '../../../api/types'
import { Dialog, Slider, Switch } from '../../../components/ui'
import { useT } from '../../../i18n'
import { attempt } from '../../../lib/actions'
import { cssVar, waveColor } from '../../../lib/color'
import { useTheme } from '../../../state/theme'
import { CleanupGetButton } from '../../resources/SeparationModels'

type Kind = 'dry' | 'song' | 'clips'

interface Preview {
  duration: number
  peaks: number[]
  segments: { start: number; end: number; dropped: null | 'too_short' | 'silent' | 'clipping' }[]
}

/** Organize = format + loudness + slicing + cleanup + naming + validation pick + health, in one go. */
export function OrganizeDialog({ character, raw, open, onOpenChange }: { character: Character; raw: RawImport[]; open: boolean; onOpenChange: (o: boolean) => void }) {
  const t = useT()
  const nav = useNavigate()
  const { data: defaults } = useOrganizeDefaults(character.id)
  const { data: resources } = useResources()
  const { data: cleanup } = useCleanupModels()
  const [opts, setOpts] = useState<OrganizeOptions | null>(null)
  const [kind, setKind] = useState<Kind>('dry')
  const [scope, setScope] = useState<'new' | 'all'>('new')
  const [advanced, setAdvanced] = useState(false)
  const separatorReady = resources?.find((r) => r.id === 'separator')?.status === 'ready'
  // dereverb uses the default cleanup model: there is none while nothing that removes reverb is ready
  const cleanupReady = cleanup?.defaultModel != null
  const pending = raw.filter((r) => r.status === 'imported' || r.status === 'failed')
  const [previewId, setPreviewId] = useState<string | null>(null)

  // start from the defaults each time the dialog opens (state set while rendering, not in an effect)
  const [started, setStarted] = useState(false)
  if (open && defaults && !started) {
    setStarted(true)
    setOpts(defaults)
    setKind(defaults.separateVocals ? 'song' : defaults.slice.enabled ? 'dry' : 'clips')
    setScope(pending.length > 0 ? 'new' : 'all')
    setPreviewId((pending[0] ?? raw[0])?.id ?? null)
  }
  if (!open && started) setStarted(false)

  if (!opts) return null
  const set = <K extends keyof OrganizeOptions>(k: K, v: OrganizeOptions[K]) => setOpts({ ...opts, [k]: v })
  const setSlice = <K extends keyof OrganizeOptions['slice']>(k: K, v: OrganizeOptions['slice'][K]) => setOpts({ ...opts, slice: { ...opts.slice, [k]: v } })
  const chooseKind = (k: Kind) => {
    setKind(k)
    const enabled = k !== 'clips'
    // the same slice object while it doesn't change, so the preview isn't fetched again
    setOpts({ ...opts, separateVocals: k === 'song', slice: enabled === opts.slice.enabled ? opts.slice : { ...opts.slice, enabled } })
  }

  const start = async () => {
    const body: OrganizeOptions = { ...opts, separateVocals: opts.separateVocals && separatorReady, dereverb: opts.dereverb && cleanupReady, rawIds: scope === 'new' ? pending.map((r) => r.id) : null }
    const job = await attempt(() => api.post(`/characters/${character.id}/organize`, body))
    if (job) onOpenChange(false)
  }

  const kinds: { id: Kind; icon: React.ReactNode; label: string }[] = [
    { id: 'dry', icon: <Mic />, label: t('mat.kind.dry') },
    { id: 'song', icon: <Disc3 />, label: t('mat.kind.song') },
    { id: 'clips', icon: <SquareStack />, label: t('mat.kind.clips') },
  ]

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={t('mat.organize.title')}
      wide
      actions={
        <>
          {pending.length > 0 && raw.length > pending.length && (
            <label className="org-scope-toggle">
              <Switch checked={scope === 'all'} onChange={(v) => setScope(v ? 'all' : 'new')} label={t('mat.organize.scope_all', { n: raw.length })} />
              <span>{t('mat.organize.scope_all', { n: raw.length })}</span>
            </label>
          )}
          <span className="spacer" />
          <button className="btn btn-ghost" onClick={() => onOpenChange(false)}>
            {t('common.cancel')}
          </button>
          <button className="btn btn-primary" onClick={start} disabled={(scope === 'new' && pending.length === 0) || (kind === 'song' && !separatorReady)}>
            <Scissors />
            {scope === 'new' ? t('mat.organize.start_n', { n: pending.length }) : t('mat.organize.start')}
          </button>
        </>
      }
    >
      <div className="org">
        <div className="org-kinds" role="radiogroup">
          {kinds.map((k) => (
            <button key={k.id} role="radio" aria-checked={kind === k.id} className={clsx('org-kind', kind === k.id && 'on')} onClick={() => chooseKind(k.id)}>
              {k.icon}
              <span>{k.label}</span>
            </button>
          ))}
        </div>

        {kind === 'song' && !separatorReady && (
          <div className="org-note">
            <span>{t('mat.organize.separate_missing')}</span>
            <span className="spacer" />
            <button
              className="btn btn-sm"
              onClick={() => {
                onOpenChange(false)
                nav('/resources')
              }}
            >
              {t('mat.organize.install')}
            </button>
          </div>
        )}

        {opts.slice.enabled && previewId && (
          <SlicePreview
            characterId={character.id}
            raw={raw}
            rawId={previewId}
            onRawId={setPreviewId}
            slice={opts.slice}
            maxLength={opts.maxLength}
            color={character.color}
          />
        )}

        {opts.slice.enabled && (
          <div className="org-sliders">
            <Num label={t('mat.organize.threshold')} unit="dB" value={opts.slice.thresholdDb} min={-70} max={-20} step={1} onChange={(v) => setSlice('thresholdDb', v)} />
            <Num label={t('mat.organize.min_length')} unit="s" value={opts.slice.minLength} min={2} max={15} step={0.5} onChange={(v) => setSlice('minLength', v)} />
            <Num label={t('mat.organize.max_length')} unit="s" value={opts.maxLength} min={5} max={40} step={1} onChange={(v) => set('maxLength', v)} />
          </div>
        )}

        <div className="org-checks">
          {(
            [
              ['normalize', 'mat.organize.normalize'],
              ['dropClipping', 'mat.organize.drop_clipping'],
              ['dropSilent', 'mat.organize.drop_silent'],
              ['dedupe', 'mat.organize.dedupe'],
            ] as const
          ).map(([k, label]) => (
            <button key={k} className={clsx('org-check', opts[k] && 'on')} onClick={() => set(k, !opts[k])} aria-pressed={opts[k]}>
              {t(label)}
            </button>
          ))}
        </div>

        <div className="row" style={{ gap: 'var(--s-2-5)' }}>
          <button
            className={clsx('org-check', opts.dereverb && cleanupReady && 'on')}
            style={cleanupReady ? undefined : { opacity: 0.55 }}
            onClick={() => set('dereverb', !opts.dereverb)}
            aria-pressed={opts.dereverb && cleanupReady}
            disabled={!cleanupReady}
          >
            {t('mat.organize.dereverb')}
          </button>
          <span className="muted" style={{ flex: 1, minWidth: 0 }}>
            {cleanupReady ? t('mat.organize.dereverb_hint') : t('mat.organize.dereverb_missing')}
          </span>
          {!cleanupReady && <CleanupGetButton />}
        </div>

        <label className="field">
          <span className="field-label">{t('mat.organize.name_pattern')}</span>
          <div className="row" style={{ gap: 10 }}>
            <input className="input mono" style={{ maxWidth: 260 }} value={opts.namePattern} onChange={(e) => set('namePattern', e.target.value)} />
            <span className="muted mono">→ {opts.namePattern.replace(/\{name\}/g, character.name).replace(/\{index(?::0?(\d)d?)?\}/g, (_, w) => '1'.padStart(w ? Number(w) : 1, '0')).replace(/\{source\}/g, 'song')}</span>
          </div>
        </label>

        <button className="org-adv-toggle" onClick={() => setAdvanced(!advanced)}>
          <ChevronDown style={{ transform: advanced ? 'rotate(180deg)' : undefined }} />
          {t('common.advanced')}
        </button>
        {advanced && (
          <div className="org-adv">
            <Num label={t('mat.organize.min_interval')} unit="s" value={opts.slice.minInterval} min={0.1} max={2} step={0.05} onChange={(v) => setSlice('minInterval', v)} disabled={!opts.slice.enabled} />
            <Num label={t('mat.organize.max_sil')} unit="s" value={opts.slice.maxSilKept} min={0.1} max={2} step={0.05} onChange={(v) => setSlice('maxSilKept', v)} disabled={!opts.slice.enabled} />
            <Num label={t('mat.organize.val_count')} unit="" value={opts.valCount} min={2} max={20} step={1} onChange={(v) => set('valCount', v)} />
          </div>
        )}
      </div>
    </Dialog>
  )
}

/** Where the slicer would cut one source file, refreshed as the settings change. */
function SlicePreview({
  characterId,
  raw,
  rawId,
  onRawId,
  slice,
  maxLength,
  color,
}: {
  characterId: string
  raw: RawImport[]
  rawId: string
  onRawId: (id: string) => void
  slice: OrganizeOptions['slice']
  maxLength: number
  color: string
}) {
  const t = useT()
  const [preview, setPreview] = useState<Preview | null>(null)
  const [state, setState] = useState<'loading' | 'ready' | 'unavailable'>('loading')
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const theme = useTheme((s) => s.resolved)

  useEffect(() => {
    let cancelled = false
    setState((s) => (s === 'unavailable' ? s : 'loading'))
    const timer = window.setTimeout(() => {
      api
        .post<Preview>(`/characters/${characterId}/organize/preview`, { rawId, slice, maxLength, n: 600 })
        .then((p) => {
          if (cancelled) return
          setPreview(p)
          setState('ready')
        })
        .catch(() => !cancelled && setState('unavailable'))
    }, 280)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [characterId, rawId, slice, maxLength])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !preview) return
    const w = canvas.clientWidth
    const h = 64
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = w * dpr
    canvas.height = h * dpr
    const ctx = canvas.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    const x = (sec: number) => (sec / preview.duration) * w
    // segments: kept ones tinted, dropped ones faint
    for (const s of preview.segments) {
      ctx.fillStyle = s.dropped ? cssVar('--critical-soft') : cssVar('--surface-2')
      ctx.fillRect(x(s.start), 0, Math.max(1, x(s.end) - x(s.start)), h)
    }
    const ink = waveColor(color)
    const rest = cssVar('--wave-rest')
    const n = preview.peaks.length
    for (let i = 0; i < n; i++) {
      const t0 = (i / n) * preview.duration
      const inKept = preview.segments.some((s) => !s.dropped && t0 >= s.start && t0 < s.end)
      const v = Math.max(1, preview.peaks[i] * (h - 8) * 0.5)
      ctx.fillStyle = inKept ? ink : rest
      ctx.fillRect((i / n) * w, h / 2 - v, Math.max(1, w / n - 0.5), v * 2)
    }
    // cut marks
    ctx.fillStyle = cssVar('--text-1')
    for (const s of preview.segments) ctx.fillRect(Math.round(x(s.start)), 0, 1, h)
  }, [preview, color, theme])

  if (state === 'unavailable') return null
  const kept = preview?.segments.filter((s) => !s.dropped).length ?? 0
  const dropped = (preview?.segments.length ?? 0) - kept
  return (
    <div className="org-preview">
      <div className="row" style={{ gap: 8 }}>
        {raw.length > 1 ? (
          <select className="input org-preview-file" value={rawId} onChange={(e) => onRawId(e.target.value)}>
            {raw.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name}
              </option>
            ))}
          </select>
        ) : (
          <span className="secondary">{raw[0]?.name}</span>
        )}
        <span className="spacer" />
        <span className={clsx('muted tabular', state === 'loading' && 'pulse')}>
          {preview ? t('mat.preview.count', { kept, dropped }) : t('common.loading')}
        </span>
      </div>
      <canvas ref={canvasRef} className="org-preview-canvas" style={{ opacity: state === 'loading' && preview ? 0.6 : 1 }} />
    </div>
  )
}

function Num({ label, unit, value, min, max, step, onChange, disabled }: { label: string; unit: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; disabled?: boolean }) {
  return (
    <div className={clsx('field', disabled && 'disabled')}>
      <div className="row">
        <span className="field-label">{label}</span>
        <span className="spacer" />
        <span className="tabular muted">
          {Number.isInteger(step) ? value : value.toFixed(step < 0.1 ? 2 : 1)} {unit}
        </span>
      </div>
      <Slider value={value} min={min} max={max} step={step} onChange={onChange} disabled={disabled} label={label} />
    </div>
  )
}

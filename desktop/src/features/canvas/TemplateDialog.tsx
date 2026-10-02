import clsx from 'clsx'
import { Check, Workflow } from 'lucide-react'
import { useState } from 'react'
import { api } from '../../api/client'
import { useFxEffects } from '../../api/hooks'
import type { CanvasNode, FxSlot, MediaSections, MixData, PortRef, RenderData } from '../../api/types'
import { Avatar } from '../../components/Avatar'
import { Dialog, Slider, Switch } from '../../components/ui'
import { useT } from '../../i18n'
import { signed } from '../../lib/format'
import { rememberedParams } from '../../lib/voiceParams'
import { useCanvas } from './context'
import { byIdOf, COLUMN_W, newMix, newMixInput, newProcess, newTake, stemMedia, type SourceNode } from './model'
import { newId } from './useProject'
import { latestRelease, voiceModelFor } from './voices'

type TemplateId = 'cover' | 'duet' | 'premix' | 'dry' | 'key'

const TEMPLATES: { id: TemplateId; voices: [number, number] }[] = [
  { id: 'cover', voices: [1, 6] },
  { id: 'duet', voices: [2, 2] },
  { id: 'premix', voices: [1, 1] },
  { id: 'key', voices: [1, 1] },
  { id: 'dry', voices: [1, 6] },
]

const ROW = 176

/**
 * A whole chain in one go, wired and ready for "Run all": nothing renders until it runs, and
 * every node stays editable afterwards.
 */
export function TemplateDialog({ source, onClose, onApply }: { source: SourceNode; onClose: () => void; onApply: (nodes: CanvasNode[], select: string[]) => void }) {
  const t = useT()
  const { project, voices, models, media } = useCanvas()
  const { data: effects } = useFxEffects()
  const [template, setTemplate] = useState<TemplateId>('cover')
  const [picked, setPicked] = useState<string[]>(() => voices.slice(0, 1).map((c) => c.id))
  const [separate, setSeparate] = useState(true)
  const [key, setKey] = useState(0)
  const [busy, setBusy] = useState(false)
  const spec = TEMPLATES.find((x) => x.id === template)!
  const [min, max] = spec.voices
  const chosen = voices.filter((c) => picked.includes(c.id)).slice(0, max)
  // 升降调 with no shift would make a copy of the original: ask for an amount first
  const ok = chosen.length >= min && !(template === 'key' && key === 0)

  const toggle = (id: string) => {
    if (picked.includes(id)) setPicked(picked.filter((x) => x !== id))
    else setPicked(max === 1 ? [id] : [...picked, id].slice(-max))
  }

  const build = async () => {
    setBusy(true)
    const out: CanvasNode[] = []
    const { x: sx, y: sy } = source.position
    const col = (i: number) => sx + COLUMN_W * i
    const input: PortRef = { nodeId: source.id, port: 'vocals' }
    const inst: PortRef = { nodeId: source.id, port: 'inst' }
    const allModels = Array.from(models.values())
    const take = (characterId: string, row: number, colIndex = 1, k = key) => {
      const model = voiceModelFor(voices.find((c) => c.id === characterId)!, allModels)!
      const n = newTake(newId('nd'), input, characterId, model.id, latestRelease(model)?.id ?? null, { ...rememberedParams(characterId), key: k }, { x: col(colIndex), y: sy + row * ROW })
      out.push(n)
      return n
    }
    const exportNode = (from: PortRef, row: number, colIndex: number, label: string) => {
      const n = newProcess('export', newId('nd'), { x: col(colIndex), y: sy + row * ROW }, [from], { fileName: `{project} - ${label}` })
      out.push(n)
      return n
    }
    const mixOf = (refs: { ref: PortRef; role?: 'lead' | 'harmony' | 'backing' | 'other' }[], row: number, colIndex: number, processor = 'builtin') => {
      const byId = byIdOf([...project.canvas.nodes, ...out])
      const inputs = refs.reduce<ReturnType<typeof newMixInput>[]>((acc, r) => [...acc, newMixInput(byId, r.ref, acc, newId('mi'), r.role)], [])
      const n = newMix(newId('nd'), { x: col(colIndex), y: sy + row * ROW }, inputs, processor)
      out.push(n)
      return n
    }
    const name = (id: string) => voices.find((c) => c.id === id)?.name ?? ''

    if (template === 'cover') {
      chosen.forEach((c, i) => {
        const tk = take(c.id, i)
        const mx = mixOf([{ ref: { nodeId: tk.id, port: 'out' } }, { ref: inst, role: 'backing' }], i, 2)
        exportNode({ nodeId: mx.id, port: 'out' }, i, 3, name(c.id))
      })
    } else if (template === 'duet') {
      const [a, b] = chosen
      const ta = take(a.id, 0)
      const tb = take(b.id, 1)
      // alternate sections between the two voices; the comp editor can change it later
      let segments: { start: number; end: number; input: number }[] = []
      const vocals = stemMedia(source.data, 'vocals') ?? source.data.mediaId
      const s = await api.get<MediaSections>(`/projects/${project.id}/media/${vocals}/sections`).catch(() => null)
      if (s && s.sections.length > 1) segments = s.sections.map((sec, i) => ({ start: sec.start, end: sec.end, input: i % 2 }))
      else {
        const d = media.get(source.data.mediaId)?.duration ?? 0
        if (d > 0) segments = [{ start: 0, end: d / 2, input: 0 }, { start: d / 2, end: d, input: 1 }]
      }
      const comp = newProcess('comp', newId('nd'), { x: col(2), y: sy }, [{ nodeId: ta.id, port: 'out' }, { nodeId: tb.id, port: 'out' }], { segments })
      comp.data.label = t('tpl.duet.comp', { a: a.name, b: b.name })
      out.push(comp)
      const mx = mixOf([{ ref: { nodeId: comp.id, port: 'out' }, role: 'lead' }, { ref: inst, role: 'backing' }], 0, 3)
      exportNode({ nodeId: mx.id, port: 'out' }, 0, 4, `${a.name} & ${b.name}`)
    } else if (template === 'premix') {
      const c = chosen[0]
      const tk = take(c.id, 0)
      const bus = mixOf(
        (['drums', 'bass', 'other'] as const).map((p) => ({ ref: { nodeId: source.id, port: p }, role: 'backing' as const })),
        1,
        1,
        'bus',
      )
      bus.data.label = t('tpl.premix.bus')
      const mx = mixOf([{ ref: { nodeId: tk.id, port: 'out' } }, { ref: { nodeId: bus.id, port: 'out' }, role: 'backing' }], 0, 2)
      exportNode({ nodeId: mx.id, port: 'out' }, 0, 3, name(c.id))
    } else if (template === 'key') {
      const c = chosen[0]
      const tk = take(c.id, 0)
      const pitch = newProcess('pitch', newId('nd'), { x: col(1), y: sy + ROW }, [inst], { semitones: key })
      out.push(pitch)
      const mx = mixOf([{ ref: { nodeId: tk.id, port: 'out' } }, { ref: { nodeId: pitch.id, port: 'out' }, role: 'backing' }], 0, 2)
      exportNode({ nodeId: mx.id, port: 'out' }, 0, 3, `${name(c.id)} ${signed(key)}`)
    } else if (template === 'dry') {
      const chain: FxSlot[] = ['eq', 'compressor', 'deesser']
        .filter((id) => effects?.some((e) => e.id === id))
        .map((effect) => ({ id: newId('fx'), kind: 'builtin', effect, params: {}, bypass: false, mix: 1 }))
      chosen.forEach((c, i) => {
        const tk = take(c.id, i)
        const fx = newProcess('fx', newId('nd'), { x: col(2), y: sy + i * ROW }, [{ nodeId: tk.id, port: 'out' }], { chain })
        out.push(fx)
        exportNode({ nodeId: fx.id, port: 'out' }, i, 3, `${name(c.id)} ${t('tpl.dry.suffix')}`)
      })
    }

    // without separation, voices listen to the whole song and there's no accompaniment port to use
    const result = separate
      ? out
      : out.map((n): CanvasNode => {
          if (n.type === 'render') return { ...n, data: { ...(n.data as RenderData), input: { nodeId: source.id, port: 'mix' } } }
          if (n.type === 'mix') return { ...n, data: { ...(n.data as MixData), inputs: (n.data as MixData).inputs.filter((i) => i.ref.nodeId !== source.id) } }
          return n
        })
    setBusy(false)
    onApply(result, result.filter((n) => n.type === 'render').map((n) => n.id))
  }

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && !busy && onClose()}
      title={t('tpl.title', { name: source.data.name })}
      wide
      actions={
        <>
          <span className="muted tpl-note">{t('tpl.note')}</span>
          <span className="spacer" />
          <button className="btn btn-ghost" onClick={onClose} disabled={busy}>
            {t('common.cancel')}
          </button>
          <button className="btn btn-primary" onClick={() => void build()} disabled={!ok || busy}>
            <Workflow />
            {t('tpl.apply')}
          </button>
        </>
      }
    >
      <div className="tpl">
        <div className="tpl-list">
          {TEMPLATES.map((x) => (
            <button key={x.id} className={clsx('tpl-card', template === x.id && 'on')} onClick={() => setTemplate(x.id)}>
              <strong>{t(`tpl.${x.id}` as const)}</strong>
              <span className="muted">{t(`tpl.${x.id}.desc` as const)}</span>
              <span className="tpl-flow mono">{t(`tpl.${x.id}.flow` as const)}</span>
            </button>
          ))}
        </div>
        <div className="tpl-side">
          <div className="field-label">{min === max ? t('tpl.voices_n', { n: min }) : t('tpl.voices_range', { min, max })}</div>
          <div className="batch-voices">
            {voices.map((c) => {
              const on = chosen.some((x) => x.id === c.id)
              return (
                <button key={c.id} className={clsx('batch-voice', on && 'on')} onClick={() => toggle(c.id)} aria-pressed={on}>
                  <Avatar character={c} size={30} />
                  <span className="batch-voice-name">{c.name}</span>
                  <span className="batch-check">{on && <Check />}</span>
                </button>
              )
            })}
          </div>
          {voices.length === 0 && <p className="muted">{t('create.no_voices')}</p>}
          {(template === 'cover' || template === 'key' || template === 'premix') && (
            <div className="tpl-row">
              <span className="field-label">{template === 'key' ? t('tpl.key_both') : t('insp.key')}</span>
              <Slider value={key} min={-12} max={12} step={1} centerMark onChange={setKey} label={t('insp.key')} />
              <span className="tabular insp-val">{signed(key)}</span>
            </div>
          )}
          {template === 'key' && key === 0 && <p className="muted insp-hint">{t('tpl.key_pick')}</p>}
          {template !== 'premix' && template !== 'key' && (
            <label className="tpl-row">
              <Switch checked={separate} onChange={setSeparate} label={t('tpl.separate')} />
              <span>{t('tpl.separate')}</span>
            </label>
          )}
        </div>
      </div>
    </Dialog>
  )
}

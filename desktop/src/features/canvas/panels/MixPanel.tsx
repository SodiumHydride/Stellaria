import clsx from 'clsx'
import { AlertTriangle, Download, Headphones, ListPlus, VolumeX, X } from 'lucide-react'
import type { MixData, MixInput, MixRole, PortRef } from '../../../api/types'
import { Avatar } from '../../../components/Avatar'
import { LevelMark } from '../../../components/Logo'
import { ParamControls } from '../../../components/ParamControls'
import { Segmented, Slider, Tip } from '../../../components/ui'
import { Waveform } from '../../../components/Waveform'
import { useT } from '../../../i18n'
import { signed } from '../../../lib/format'
import { entryOf } from '../catalog'
import { mediaAudioPath, mediaPeaksPath, useCanvas } from '../context'
import { byIdOf, descendants, isContent, markStale, MAX_MIX_INPUTS, newMixInput, outputPorts, voiceOf, type MixNode } from '../model'
import { useNodeName, usePortLabel, useRefLabel } from '../nodes'
import { newId } from '../useProject'
import { RUN_KEYS } from '../../../shell/bridge'
import { toast } from '../../../state/ui'
import type { MixPreset } from '../presets'
import { MixPresetsMenu } from './MixPresets'

const ROLES: MixRole[] = ['lead', 'harmony', 'backing', 'other']

/** A mix: what goes in (each with a role, level, pan), how it's mixed, and the results. */
export function MixPanel({ node }: { node: MixNode }) {
  const t = useT()
  const refLabel = useRefLabel()
  const nodeName = useNodeName()
  const portLabel = usePortLabel()
  const { project, characters, processors, actions } = useCanvas()
  const d = node.data
  const nodes = project.canvas.nodes
  const byId = byIdOf(nodes)
  const bus = d.processor === 'bus'
  const processor = processors.find((p) => p.id === d.processor)
  const busy = d.status === 'queued' || d.status === 'running'
  const selected = d.outputs[d.selected]
  const Icon = entryOf(bus ? 'bus' : 'mix').icon
  const downstream = descendants(nodes, node.id)
  const candidates: { ref: PortRef; label: string }[] = nodes
    .filter((n) => isContent(n) && n.id !== node.id && !downstream.has(n.id))
    .flatMap((n) => outputPorts(n).map((port) => ({ ref: { nodeId: n.id, port }, label: n.type === 'source' || outputPorts(n).length > 1 ? `${nodeName(n)} · ${portLabel(port, n)}` : nodeName(n) })))
    .filter((c) => !d.inputs.some((i) => i.ref.nodeId === c.ref.nodeId && i.ref.port === c.ref.port))
  const maxVoices = bus ? MAX_MIX_INPUTS : (processor?.maxVocals ?? 4)
  const voiceCount = d.inputs.filter((i) => i.role === 'lead' || i.role === 'harmony').length

  const setData = (patch: Partial<MixData>, sound = true, key?: string) =>
    actions.patchNode(
      node.id,
      (n) => {
        const next = { ...n, data: { ...(n.data as MixData), ...patch } }
        return sound ? markStale(next) : next
      },
      key ? { history: `${node.id}:${key}` } : undefined,
    )
  const presetValues = processor?.presets.find((p) => p.id === (d.preset ?? processor.presets[0]?.id))?.params
  const describePreset = (p: MixPreset) => {
    const proc = processors.find((x) => x.id === p.processor)
    if (!proc) return p.processor
    const base = proc.presets.find((x) => x.id === p.preset)
    return base ? `${t.label(proc.name)} · ${t.label(base.name)}` : t.label(proc.name)
  }
  const loadPreset = (p: MixPreset) => {
    const proc = processors.find((x) => x.id === p.processor)
    if (!proc?.available && p.processor !== 'bus') {
      toast({ kind: 'error', title: t('mix.presets.unavailable', { name: p.name }) })
      return
    }
    const preset = proc?.presets.some((x) => x.id === p.preset) ? p.preset : (proc?.presets[0]?.id ?? null)
    setData({ processor: p.processor, preset, params: { ...p.params } }, true, 'preset')
  }
  const patchInput = (id: string, patch: Partial<MixInput>, key?: string) => setData({ inputs: d.inputs.map((i) => (i.id === id ? { ...i, ...patch } : i)) }, true, key ? `${id}:${key}` : undefined)

  return (
    <div className="insp-body">
      <div className="insp-title">
        <Icon className="insp-title-icon" />
        <input className="insp-title-input" value={d.label} placeholder={bus ? t('node.bus') : t('node.mix')} onChange={(e) => setData({ label: e.target.value }, false, 'label')} />
      </div>

      {selected && !busy && (
        <div className="insp-result">
          <Waveform id={`node:${node.id}`} peaksPath={mediaPeaksPath(project.id, selected.mediaId)} audioPath={mediaAudioPath(project.id, selected.mediaId)} height={56} bars={120} showButton />
          {d.outputs.length > 1 && (
            <div className="row" style={{ gap: 6 }}>
              {d.outputs.map((o, i) => (
                <button key={o.mediaId} className={clsx('chip button', i === d.selected && 'accent')} onClick={() => setData({ selected: i }, false)}>
                  {String.fromCharCode(65 + i)}
                  {o.lufs != null && <span className="muted tabular"> {o.lufs.toFixed(1)} LUFS</span>}
                </button>
              ))}
            </div>
          )}
          <div className="row" style={{ gap: 6 }}>
            <button className="btn btn-sm" onClick={() => void actions.exportMedia(selected.mediaId, d.label || project.name)}>
              <Download />
              {t('mix.export')}
            </button>
            <button className="btn btn-sm btn-ghost" onClick={() => actions.addToTimeline(node.id)}>
              <ListPlus />
              {t('insp.add_to_timeline')}
            </button>
          </div>
        </div>
      )}
      {d.status === 'failed' && d.error && (
        <div className="chip critical insp-error">
          <AlertTriangle />
          {t.msg(d.error)}
        </div>
      )}

      <section className="insp-section">
        <div className="insp-section-head">
          <span className="eyebrow">{t('mix.inputs')}</span>
          <span className="spacer" />
          <span className="muted tabular">{d.inputs.length}</span>
        </div>
        {d.inputs.length === 0 && <p className="muted insp-hint">{t('node.mix_connect')}</p>}
        {d.inputs.map((inp) => {
          const cid = voiceOf(byId, inp.ref)
          const c = cid ? characters.get(cid) : undefined
          return (
            <div key={inp.id} className={clsx('mix-vocal', inp.mute && 'muted-input')}>
              <div className="row" style={{ gap: 8 }}>
                {c && <Avatar character={c} size={22} />}
                <strong className="mix-vocal-name">{refLabel(inp.ref)}</strong>
                <span className="spacer" />
                <Tip label={t('mix.solo')}>
                  <button className={clsx('icon-btn sm', inp.solo && 'on')} onClick={() => patchInput(inp.id, { solo: !inp.solo })} aria-pressed={inp.solo}>
                    <Headphones />
                  </button>
                </Tip>
                <Tip label={t('mix.mute')}>
                  <button className={clsx('icon-btn sm', inp.mute && 'on')} onClick={() => patchInput(inp.id, { mute: !inp.mute })} aria-pressed={inp.mute}>
                    <VolumeX />
                  </button>
                </Tip>
                <Tip label={t('mix.remove_input')}>
                  <button className="icon-btn sm" onClick={() => setData({ inputs: d.inputs.filter((x) => x.id !== inp.id) })}>
                    <X />
                  </button>
                </Tip>
              </div>
              {!bus && (
                <Segmented<MixRole> size="sm" value={inp.role} onChange={(role) => patchInput(inp.id, { role })} options={ROLES.map((r) => ({ value: r, label: t(`mix.role.${r}` as const) }))} />
              )}
              <div className="mix-vocal-sliders">
                <span className="muted">{t('mix.gain')}</span>
                <Slider value={inp.gainDb} min={-24} max={12} step={0.5} centerMark onChange={(gainDb) => patchInput(inp.id, { gainDb }, 'gain')} label={t('mix.gain')} />
                <span className="tabular muted">{signed(inp.gainDb)}</span>
                <span className="muted">{t('mix.pan')}</span>
                <Slider value={inp.pan} min={-1} max={1} step={0.05} centerMark onChange={(pan) => patchInput(inp.id, { pan }, 'pan')} label={t('mix.pan')} />
                <span className="tabular muted">{inp.pan === 0 ? 'C' : inp.pan < 0 ? `L${Math.round(-inp.pan * 100)}` : `R${Math.round(inp.pan * 100)}`}</span>
              </div>
            </div>
          )
        })}
        {candidates.length > 0 && d.inputs.length < (processor?.maxInputs ?? MAX_MIX_INPUTS) && (
          <select
            className="input insp-select"
            value=""
            onChange={(e) => {
              const c = candidates[Number(e.target.value)]
              if (c) setData({ inputs: [...d.inputs, newMixInput(byId, c.ref, d.inputs, newId('mi'))] })
            }}
            aria-label={t('mix.add_input')}
          >
            <option value="">＋ {t('mix.add_input')}</option>
            {candidates.map((c, i) => (
              <option key={`${c.ref.nodeId}:${c.ref.port}`} value={i}>
                {c.label}
              </option>
            ))}
          </select>
        )}
        {!bus && voiceCount > maxVoices && <p className="chip warning insp-error">{t.msg({ code: 'plugin.too_many_vocals', params: { max: maxVoices } })}</p>}
      </section>

      <section className="insp-section">
        <div className="insp-section-head">
          <span className="eyebrow">{t('mix.processor')}</span>
          <span className="spacer" />
          <MixPresetsMenu current={{ processor: d.processor, preset: d.preset, params: d.params }} describe={describePreset} onLoad={loadPreset} />
        </div>
        <select
          className="input insp-select"
          value={d.processor}
          onChange={(e) => {
            const id = e.target.value
            const next = processors.find((p) => p.id === id)
            setData({ processor: id, params: {}, preset: next?.presets[0]?.id ?? null })
          }}
        >
          {processors.map((p) => (
            <option key={p.id} value={p.id} disabled={!p.available}>
              {t.label(p.name)}
              {!p.available ? ` · ${t('mix.unavailable')}` : ''}
            </option>
          ))}
          {!processors.some((p) => p.id === 'bus') && <option value="bus">{t('mix.bus')}</option>}
        </select>
        {processor && !processor.available && processor.error && <p className="muted insp-hint">{t.msg(processor.error)}</p>}
        {processor && processor.presets.length > 0 && (
          <Segmented<string> size="sm" value={d.preset ?? processor.presets[0].id} onChange={(preset) => setData({ preset, params: {} })} options={processor.presets.map((p) => ({ value: p.id, label: t.label(p.name) }))} />
        )}
        {processor && processor.params.length > 0 && (
          <ParamControls
            specs={processor.params}
            values={d.params}
            defaults={presetValues}
            groupLabel={(g) => t.msg({ code: `mix.group.${g}` }, g)}
            onReset={(ids) => setData({ params: Object.fromEntries(Object.entries(d.params).filter(([k]) => !ids.includes(k))) }, true, 'params:reset')}
            onChange={(id, v) => setData({ params: { ...d.params, [id]: v as number | string | boolean } }, true, `param:${id}`)}
          />
        )}
      </section>

      {!bus && (
        <section className="insp-section">
          <div className="insp-section-head">
            <Tip label={t('mix.variants_hint')}>
              <span className="eyebrow has-hint">{t('mix.variants')}</span>
            </Tip>
          </div>
          <Segmented<string> size="sm" value={String(d.variants)} onChange={(v) => setData({ variants: Number(v) }, false)} options={['1', '2', '3', '4'].map((v) => ({ value: v, label: v }))} />
        </section>
      )}

      <div className="insp-actions">
        <button className="btn btn-primary" onClick={() => void actions.runNode(node.id)} disabled={busy || d.inputs.length === 0}>
          {busy ? <LevelMark size={16} running /> : <Icon />}
          {busy ? t('node.mixing') : d.outputs.length ? t('mix.remix') : t('node.mix_go')}
          {!busy && <span className="kbd">{RUN_KEYS}</span>}
        </button>
      </div>
    </div>
  )
}

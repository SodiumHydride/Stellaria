import clsx from 'clsx'
import { ArrowDown, ArrowUp, CircleSlash, Download, ListPlus, Loader2, Sparkles, Wand2, X } from 'lucide-react'
import { useSections } from '../../../api/hooks'
import type { CompSegment, PitchParams, ProcessData, ProcessParamsByOp, SplitParams, TrimParams } from '../../../api/types'
import { Segmented, Slider, Switch, Tip } from '../../../components/ui'
import { useT } from '../../../i18n'
import { RUN_KEYS } from '../../../shell/bridge'
import { VOICE_COLORS } from '../../../lib/color'
import { clock, signed } from '../../../lib/format'
import { PlayButton, Waveform } from '../../../components/Waveform'
import { CleanupModelPicker, SeparationModelPicker } from '../../resources/SeparationModels'
import { entryOf } from '../catalog'
import { mediaAudioPath, mediaPeaksPath, useCanvas } from '../context'
import { FxChainEditor } from '../fx/FxChainEditor'
import { byIdOf, isSource, isTake, markStale, opPorts, resolvePort, rootSourceId, stemMedia, voiceOf, type ProcessNode } from '../model'
import { usePortLabel, useRefLabel } from '../nodes'
import { runnable } from '../runner'
import { ExportPanel } from './ExportPanel'
import { RegionEditor } from './RegionEditor'
import { RunNotice, Section } from './parts'

/** Inspector for cut / split / pitch / effects / separate / dereverb / comp / concat / export nodes. */
export function ProcessPanel({ node }: { node: ProcessNode }) {
  const t = useT()
  const portLabel = usePortLabel()
  const { project, media, actions } = useCanvas()
  const d = node.data
  const byId = byIdOf(project.canvas.nodes)
  const Icon = entryOf(d.op).icon
  const busy = d.status === 'queued' || d.status === 'running'
  const bypassable = d.op === 'fx' || d.op === 'pitch'
  const inputMedia = d.inputs[0] ? resolvePort(byId, d.inputs[0]) : null
  const inputDuration = inputMedia ? (media.get(inputMedia)?.duration ?? 0) : 0

  const setData = (patch: Partial<ProcessData>, sound = true, key?: string) =>
    actions.patchNode(node.id, (n) => {
      const next = { ...n, data: { ...(n.data as ProcessData), ...patch } as ProcessData }
      return sound ? markStale(next) : next
    }, key ? { history: `${node.id}:${key}` } : undefined)
  const setParams = <Op extends ProcessData['op']>(patch: Partial<ProcessParamsByOp[Op]>, key = 'params') => setData({ params: { ...d.params, ...patch } as ProcessData['params'] }, true, key)

  const outputs = opPorts(d).filter((p) => d.outputs?.[p])

  return (
    <div className="insp-body">
      <div className="insp-title">
        <Icon className="insp-title-icon" />
        <input className="insp-title-input" value={d.label} placeholder={t(`op.${d.op}` as const)} onChange={(e) => setData({ label: e.target.value }, false, 'label')} />
        {bypassable && (
          <Tip label={`${t('ctx.bypass')} (B)`}>
            <button className={clsx('icon-btn sm', d.bypass && 'on')} onClick={() => setData({ bypass: !d.bypass }, false)} aria-pressed={d.bypass}>
              <CircleSlash />
            </button>
          </Tip>
        )}
      </div>

      {d.inputs.length === 0 && <p className="muted insp-hint">{t('node.connect_input')}</p>}

      {outputs.length > 0 && !busy && d.op !== 'export' && (
        <div className="insp-result">
          {outputs.map((p) => (
            <div key={p} className="insp-stem">
              {outputs.length > 1 && <span className="insp-stem-label">{portLabel(p, node)}</span>}
              <Waveform id={`node:${node.id}:${p}`} peaksPath={mediaPeaksPath(project.id, d.outputs[p]!)} audioPath={mediaAudioPath(project.id, d.outputs[p]!)} height={outputs.length > 1 ? 28 : 48} bars={120} showButton />
            </div>
          ))}
          {(d.result as { engine?: string } | null)?.engine && <span className="muted insp-hint">{t('op.pitch.engine', { engine: String((d.result as { engine?: string }).engine) })}</span>}
        </div>
      )}
      <RunNotice error={d.status === 'failed' ? d.error : null} stale={d.stale && d.status === 'done'} staleLabel={t('node.process_stale')} msg={(m) => t.msg(m)} />

      {d.op === 'trim' && inputMedia && <TrimEditor node={node as ProcessNode<'trim'>} mediaId={inputMedia} duration={inputDuration} setParams={setParams} />}
      {d.op === 'split' && inputMedia && <SplitEditor node={node as ProcessNode<'split'>} mediaId={inputMedia} duration={inputDuration} setParams={setParams} />}
      {d.op === 'pitch' && <PitchEditor node={node as ProcessNode<'pitch'>} setParams={setParams} />}
      {d.op === 'fx' && (
        <FxChainEditor
          value={(d as ProcessData<'fx'>).params.chain}
          onChange={(chain) => setParams<'fx'>({ chain }, 'chain')}
          bpm={(() => {
            const root = d.inputs[0] ? rootSourceId(byId, d.inputs[0].nodeId) : null
            const src = root ? byId.get(root) : undefined
            return isSource(src) ? (src.data.bpm ?? null) : null
          })()}
        />
      )}
      {d.op === 'separate' && <SeparateEditor node={node as ProcessNode<'separate'>} setParams={setParams} />}
      {d.op === 'dereverb' && <DereverbEditor node={node as ProcessNode<'dereverb'>} setParams={setParams} />}
      {d.op === 'comp' && <CompEditor node={node as ProcessNode<'comp'>} setParams={setParams} />}
      {d.op === 'concat' && <ConcatEditor node={node as ProcessNode<'concat'>} setData={setData} setParams={setParams} />}
      {d.op === 'export' && <ExportPanel params={(d as ProcessData<'export'>).params} onChange={(p) => setData({ params: p }, true, 'params')} result={d.status === 'done' ? (d.result as { path?: string } | null) : null} />}

      <div className="insp-actions">
        <button className="btn btn-primary" onClick={() => void actions.runNode(node.id)} disabled={busy || !runnable(node)}>
          {busy ? <Loader2 className="spin" /> : d.op === 'export' ? <Download /> : <Wand2 />}
          {busy ? t('node.processing') : d.op === 'export' ? t('op.export.run') : d.status === 'done' ? t('node.process_again') : t('node.process_go')}
          {!busy && <span className="kbd">{RUN_KEYS}</span>}
        </button>
        {outputs.length > 0 && d.op !== 'export' && (
          <button className="btn btn-ghost" onClick={() => actions.addToTimeline(node.id)}>
            <ListPlus />
            {t('insp.add_to_timeline')}
          </button>
        )}
      </div>
    </div>
  )
}

type SetParams = <Op extends ProcessData['op']>(patch: Partial<ProcessParamsByOp[Op]>, key?: string) => void

/** Snap points for a media: the phrase edges and downbeats of its song. */
function useSnaps(mediaId: string | null) {
  const { project } = useCanvas()
  const { data } = useSections(project.id, mediaId)
  if (!data) return { snaps: [] as number[], sections: data }
  return { snaps: [...data.phrases.flatMap((p) => [p.start, p.end]), ...data.downbeats], sections: data }
}

function TrimEditor({ node, mediaId, duration, setParams }: { node: ProcessNode<'trim'>; mediaId: string; duration: number; setParams: SetParams }) {
  const t = useT()
  const { project } = useCanvas()
  const p = node.data.params
  const { snaps } = useSnaps(mediaId)
  const end = Math.min(p.end, duration || p.end)
  return (
    <>
      <Section title={t('op.trim.region')}>
        <RegionEditor
          projectId={project.id}
          mediaId={mediaId}
          duration={duration}
          snaps={snaps}
          expandable
          title={t('op.trim')}
          mode={{ kind: 'range', start: p.start, end, onChange: (start, e) => setParams<'trim'>({ start, end: e }, 'region') }}
        />
      </Section>
      <Section title={t('op.trim.fades')} aside={<span className="tabular insp-val">{Math.round(p.fadeIn * 1000)} / {Math.round(p.fadeOut * 1000)} ms</span>}>
        <div className="insp-pair">
          <Slider value={p.fadeIn} min={0} max={2} step={0.01} onChange={(fadeIn) => setParams<'trim'>({ fadeIn } as Partial<TrimParams>, 'fadeIn')} label={t('op.trim.fade_in')} />
          <Slider value={p.fadeOut} min={0} max={2} step={0.01} onChange={(fadeOut) => setParams<'trim'>({ fadeOut } as Partial<TrimParams>, 'fadeOut')} label={t('op.trim.fade_out')} />
        </div>
      </Section>
    </>
  )
}

function SplitEditor({ node, mediaId, duration, setParams }: { node: ProcessNode<'split'>; mediaId: string; duration: number; setParams: SetParams }) {
  const t = useT()
  const { project } = useCanvas()
  const p = node.data.params
  const { snaps, sections } = useSnaps(mediaId)
  const parts = p.cuts.length + 1
  const setLabels = (labels: string[]) => setParams<'split'>({ labels } as Partial<SplitParams>, 'labels')
  return (
    <>
      <Section
        title={t('op.split.cuts')}
        aside={
          sections && sections.sections.length > 1 ? (
            <button
              className="chip button accent"
              onClick={() =>
                setParams<'split'>({
                  cuts: sections.sections.slice(1).map((s) => s.start),
                  labels: sections.sections.map((s) => (s.label ? t(`section.${s.label}` as const) : '')),
                })
              }
            >
              <Sparkles />
              {t('op.split.by_sections')}
            </button>
          ) : undefined
        }
      >
        <RegionEditor projectId={project.id} mediaId={mediaId} duration={duration} snaps={snaps} expandable title={t('op.split')} mode={{ kind: 'cuts', cuts: p.cuts, onChange: (cuts) => setParams<'split'>({ cuts }, 'cuts') }} />
        <p className="muted insp-hint">{t('region.help.cuts')}</p>
      </Section>
      <Section title={t('op.split.parts', { n: parts })}>
        <div className="split-labels">
          {Array.from({ length: parts }, (_, i) => (
            <input
              key={i}
              className="input"
              value={p.labels[i] ?? ''}
              placeholder={t('port.part', { n: i + 1 })}
              onChange={(e) => {
                const labels = Array.from({ length: parts }, (_, j) => p.labels[j] ?? '')
                labels[i] = e.target.value
                setLabels(labels)
              }}
            />
          ))}
        </div>
      </Section>
    </>
  )
}

function PitchEditor({ node, setParams }: { node: ProcessNode<'pitch'>; setParams: SetParams }) {
  const t = useT()
  const { project } = useCanvas()
  const p = node.data.params
  const set = (patch: Partial<PitchParams>, key: string) => setParams<'pitch'>(patch, key)
  // keys the voices sing this song in: one click makes the backing follow
  const byId = byIdOf(project.canvas.nodes)
  const root = node.data.inputs[0] ? rootSourceId(byId, node.data.inputs[0].nodeId) : null
  const keys = Array.from(new Set(project.canvas.nodes.filter(isTake).filter((n) => root && rootSourceId(byId, n.id) === root).map((n) => n.data.params.key))).filter((k) => k !== 0)
  return (
    <>
      <Section title={t('op.pitch.semitones')} aside={<span className="tabular insp-val">{signed(p.semitones)}</span>}>
        <Slider value={p.semitones} min={-12} max={12} step={1} centerMark onChange={(semitones) => set({ semitones }, 'semitones')} label={t('op.pitch.semitones')} />
        {keys.length > 0 && (
          <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
            {keys.map((k) => (
              <button key={k} className={clsx('chip button', p.semitones === k && 'accent')} onClick={() => set({ semitones: k }, 'semitones')}>
                {t('op.pitch.follow', { key: signed(k) })}
              </button>
            ))}
          </div>
        )}
      </Section>
      <Section title={t('op.pitch.cents')} aside={<span className="tabular insp-val">{signed(p.cents)}</span>}>
        <Slider value={p.cents} min={-100} max={100} step={1} centerMark onChange={(cents) => set({ cents }, 'cents')} label={t('op.pitch.cents')} />
      </Section>
      <Section title={t('op.pitch.tempo')} aside={<span className="tabular insp-val">{p.tempo.toFixed(2)}×</span>}>
        <Slider value={p.tempo} min={0.5} max={2} step={0.01} centerMark onChange={(tempo) => set({ tempo }, 'tempo')} label={t('op.pitch.tempo')} />
      </Section>
      <Section title={t('op.pitch.quality')}>
        <Segmented<PitchParams['quality']>
          size="sm"
          value={p.quality}
          onChange={(quality) => set({ quality }, 'quality')}
          options={[
            { value: 'high', label: t('op.pitch.quality.high') },
            { value: 'fast', label: t('op.pitch.quality.fast') },
          ]}
        />
        <label className="row insp-switch">
          <Switch checked={p.preserveFormant} onChange={(preserveFormant) => set({ preserveFormant }, 'formant')} label={t('op.pitch.formant')} />
          <Tip label={t('op.pitch.formant_hint')}>
            <span className="has-hint">{t('op.pitch.formant')}</span>
          </Tip>
        </label>
      </Section>
    </>
  )
}

function SeparateEditor({ node, setParams }: { node: ProcessNode<'separate'>; setParams: SetParams }) {
  const t = useT()
  const p = node.data.params
  return (
    <>
      <Section title={t('op.separate.stems')}>
        <Segmented<string>
          size="sm"
          value={String(p.stems)}
          onChange={(v) => setParams<'separate'>({ stems: Number(v) as 2 | 4 }, 'stems')}
          options={[
            { value: '2', label: t('op.separate.two') },
            { value: '4', label: t('op.separate.four') },
          ]}
        />
      </Section>
      <Section title={t('op.separate.model')} hint={t('sep.model_hint')}>
        <SeparationModelPicker compact value={p.model ?? 'auto'} onChange={(model) => setParams<'separate'>({ model }, 'model')} />
      </Section>
    </>
  )
}

function DereverbEditor({ node, setParams }: { node: ProcessNode<'dereverb'>; setParams: SetParams }) {
  const t = useT()
  return (
    <Section title={t('op.dereverb.model')} hint={t('cln.model_hint')}>
      <CleanupModelPicker compact value={node.data.params.model ?? 'auto'} onChange={(model) => setParams<'dereverb'>({ model }, 'model')} />
    </Section>
  )
}

function CompEditor({ node, setParams }: { node: ProcessNode<'comp'>; setParams: SetParams }) {
  const t = useT()
  const refLabel = useRefLabel()
  const { project, characters, media } = useCanvas()
  const byId = byIdOf(project.canvas.nodes)
  const p = node.data.params
  const root = node.data.inputs[0] ? rootSourceId(byId, node.data.inputs[0].nodeId) : null
  const src = root ? byId.get(root) : undefined
  // show the song's vocals (or the song) as the map to cut along
  const refMedia = isSource(src) ? (stemMedia(src.data, 'vocals') ?? src.data.mediaId) : node.data.inputs[0] ? resolvePort(byId, node.data.inputs[0]) : null
  const songMedia = isSource(src) ? src.data.mediaId : refMedia
  const duration = songMedia ? (media.get(songMedia)?.duration ?? 0) : 0
  const { snaps, sections } = useSnaps(refMedia)
  const lanes = node.data.inputs.map((ref, i) => {
    const cid = voiceOf(byId, ref)
    return { label: refLabel(ref), color: (cid && characters.get(cid)?.color) || VOICE_COLORS[i % VOICE_COLORS.length] }
  })
  const segs: CompSegment[] = p.segments.length ? p.segments : duration ? [{ start: 0, end: duration, input: 0 }] : []
  const set = (segments: CompSegment[]) => setParams<'comp'>({ segments }, 'segments')

  const alternate = () => {
    const base = sections && sections.sections.length > 1 ? sections.sections : null
    if (base) set(base.map((s, i) => ({ start: s.start, end: s.end, input: i % Math.max(1, lanes.length) })))
    else if (duration) {
      const n = Math.max(2, lanes.length)
      set(Array.from({ length: n }, (_, i) => ({ start: (duration * i) / n, end: (duration * (i + 1)) / n, input: i % lanes.length })))
    }
  }

  if (lanes.length < 2) return <p className="muted insp-hint">{t('op.comp.need_two')}</p>
  return (
    <>
      <Section
        title={t('op.comp.segments')}
        aside={
          <button className="chip button accent" onClick={alternate}>
            <Sparkles />
            {t('op.comp.alternate')}
          </button>
        }
      >
        {refMedia ? (
          <RegionEditor projectId={project.id} mediaId={refMedia} duration={duration} snaps={snaps} expandable title={t('op.comp')} height={72} mode={{ kind: 'segments', segments: segs, lanes, onChange: set }} />
        ) : (
          <p className="muted insp-hint">{t('node.connect_input')}</p>
        )}
        <div className="comp-legend">
          {lanes.map((l) => (
            <span key={l.label} className="comp-lane" style={{ ['--seg' as string]: l.color }}>
              {l.label}
            </span>
          ))}
        </div>
        <p className="muted insp-hint">{t('region.help.segments')}</p>
        {segs.length > 1 && (
          <div className="comp-segs" role="list">
            {segs.map((sg, i) => (
              <div key={`${sg.start}-${i}`} className="comp-seg" role="listitem">
                <span className="tabular comp-seg-time">
                  {clock(sg.start)}–{clock(sg.end)}
                </span>
                <div className="comp-seg-lanes" role="radiogroup" aria-label={t('op.comp.who', { n: i + 1 })}>
                  {lanes.map((l, li) => (
                    <button
                      key={l.label}
                      role="radio"
                      aria-checked={sg.input === li}
                      className={clsx('comp-seg-lane', sg.input === li && 'on')}
                      style={{ ['--seg' as string]: l.color }}
                      title={l.label}
                      onClick={() => set(segs.map((x, j) => (j === i ? { ...x, input: li } : x)))}
                    >
                      {l.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </Section>
      <Section title={t('op.comp.crossfade')} aside={<span className="tabular insp-val">{Math.round(p.crossfade * 1000)} ms</span>}>
        <Slider value={p.crossfade} min={0} max={0.5} step={0.01} onChange={(crossfade) => setParams<'comp'>({ crossfade }, 'crossfade')} label={t('op.comp.crossfade')} />
      </Section>
    </>
  )
}

function ConcatEditor({ node, setData, setParams }: { node: ProcessNode<'concat'>; setData: (patch: Partial<ProcessData>, sound?: boolean, key?: string) => void; setParams: SetParams }) {
  const t = useT()
  const refLabel = useRefLabel()
  const { project, media } = useCanvas()
  const byId = byIdOf(project.canvas.nodes)
  const p = node.data.params
  const inputs = node.data.inputs
  const mids = inputs.map((r) => resolvePort(byId, r))
  const lengths = mids.map((mid) => (mid ? (media.get(mid)?.duration ?? null) : null))
  const known = lengths.every((x) => x != null)
  const total = known ? lengths.reduce<number>((a, b) => a + (b ?? 0), 0) + Math.max(0, inputs.length - 1) * (p.gap - p.crossfade) : null
  const remove = (i: number) => setData({ inputs: inputs.filter((_, j) => j !== i) })
  const move = (i: number, dir: -1 | 1) => {
    const next = inputs.slice()
    const j = i + dir
    if (j < 0 || j >= next.length) return
    ;[next[i], next[j]] = [next[j], next[i]]
    setData({ inputs: next })
  }
  return (
    <>
      <Section title={t('op.concat.order')} aside={total != null ? <span className="tabular insp-val">{t('op.concat.total', { time: clock(total) })}</span> : undefined}>
        <div className="concat-list">
          {inputs.map((r, i) => (
            <div key={`${r.nodeId}:${r.port}:${i}`} className="concat-row">
              <span className="tabular muted">{i + 1}</span>
              {mids[i] ? <PlayButton id={`concat:${node.id}:${i}`} audioPath={mediaAudioPath(project.id, mids[i]!)} size={24} /> : <span className="concat-gap" />}
              <span className="concat-name">{refLabel(r)}</span>
              {lengths[i] != null && <span className="tabular muted concat-len">{clock(lengths[i]!)}</span>}
              <button className="icon-btn sm" onClick={() => move(i, -1)} disabled={i === 0} aria-label={t('op.concat.earlier')} title={t('op.concat.earlier')}>
                <ArrowUp />
              </button>
              <button className="icon-btn sm" onClick={() => move(i, 1)} disabled={i === inputs.length - 1} aria-label={t('op.concat.later')} title={t('op.concat.later')}>
                <ArrowDown />
              </button>
              <button className="icon-btn sm" onClick={() => remove(i)} aria-label={t('op.concat.remove')} title={t('op.concat.remove')}>
                <X />
              </button>
            </div>
          ))}
        </div>
      </Section>
      <Section title={t('op.concat.gap')} aside={<span className="tabular insp-val">{p.gap.toFixed(1)} s</span>}>
        <Slider value={p.gap} min={0} max={5} step={0.1} onChange={(gap) => setParams<'concat'>({ gap }, 'gap')} label={t('op.concat.gap')} />
      </Section>
      <Section title={t('op.concat.crossfade')} aside={<span className="tabular insp-val">{p.crossfade.toFixed(1)} s</span>}>
        <Slider value={p.crossfade} min={0} max={5} step={0.1} onChange={(crossfade) => setParams<'concat'>({ crossfade }, 'crossfade')} label={t('op.concat.crossfade')} />
      </Section>
    </>
  )
}

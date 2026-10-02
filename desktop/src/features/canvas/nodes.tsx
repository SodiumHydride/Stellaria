import { Handle, NodeResizer, Position, type NodeProps } from '@xyflow/react'
import clsx from 'clsx'
import { AlertTriangle, ChevronDown, CircleSlash, Heart, Loader2, Music, Play, RefreshCw, Scissors, SlidersHorizontal, Star, Users } from 'lucide-react'
import { memo, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { mediaUrl } from '../../api/client'
import { useActiveJobs, useResources } from '../../api/hooks'
import type { AudioData, CanvasNode, Character, GroupData, MixData, NodeLook, NoteData, PortRef, ProcessData, RenderData, SourceData } from '../../api/types'
import { Avatar } from '../../components/Avatar'
import { LevelMark, Logo } from '../../components/Logo'
import { Waveform } from '../../components/Waveform'
import { useT } from '../../i18n'
import { VOICE_COLORS } from '../../lib/color'
import { clock, percent, signed } from '../../lib/format'
import { usePlayer } from '../../state/player'
import { catalogIdOf, entryOf } from './catalog'
import { mediaAudioPath, mediaPeaksPath, useCanvas } from './context'
import { byIdOf, EXTRA_STEMS, inputsOf, isSource, isTake, opPorts, playableMedia, voiceOf } from './model'
import { runnable } from './runner'
import { ensureSeparator, separatorState } from './separator'

type RFData<T> = { node: CanvasNode & { data: T } }

function NodeWave({ mediaId, color, nodeId, height = 34 }: { mediaId: string; color?: string; nodeId: string; height?: number }) {
  const { project } = useCanvas()
  return (
    <div className="cn-wave nodrag">
      <Waveform id={`node:${nodeId}`} peaksPath={mediaPeaksPath(project.id, mediaId)} audioPath={mediaAudioPath(project.id, mediaId)} height={height} bars={80} color={color} showButton />
    </div>
  )
}

/** Colour tag and collapsed state, shared by every content node. */
function look(d: NodeLook): { className: string | false; style: CSSProperties | undefined } {
  return {
    className: clsx(d.tag && 'tagged', d.collapsed && 'collapsed'),
    style: d.tag ? ({ ['--tag' as string]: d.tag } as CSSProperties) : undefined,
  }
}

/**
 * An output port. Hovering it shows what comes out (a waveform you can play); Alt+click plays
 * it without starting a connection.
 */
function OutHandle({ id, nodeId, mediaId, label, className, playId }: { id: string; nodeId: string; mediaId: string | null; label: string; className?: string; playId?: string }) {
  const { project } = useCanvas()
  const ref = useRef<HTMLDivElement>(null)
  const [at, setAt] = useState<{ x: number; y: number } | null>(null)
  const timer = useRef<number | undefined>(undefined)
  const pid = playId ?? `node:${nodeId}:${id}`
  const show = () => {
    window.clearTimeout(timer.current)
    if (!mediaId) return
    timer.current = window.setTimeout(() => {
      const r = ref.current?.getBoundingClientRect()
      if (r) setAt({ x: r.right + 10, y: r.top + r.height / 2 })
    }, 380)
  }
  const hide = () => {
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setAt(null), 220)
  }
  useEffect(() => () => window.clearTimeout(timer.current), [])
  useEffect(() => {
    if (!at) return
    const close = () => setAt(null)
    window.addEventListener('wheel', close, { passive: true })
    return () => window.removeEventListener('wheel', close)
  }, [at])
  return (
    <>
      <Handle
        ref={ref}
        type="source"
        id={id}
        position={Position.Right}
        className={clsx('cn-handle', className)}
        onMouseEnter={show}
        onMouseLeave={hide}
        onMouseDownCapture={(e) => {
          if (!e.altKey || !mediaId) return
          e.stopPropagation()
          e.preventDefault()
          usePlayer.getState().toggle(pid, mediaUrl(mediaAudioPath(project.id, mediaId)))
        }}
      />
      {at &&
        mediaId &&
        createPortal(
          <div className="cn-peek glass-strong" style={{ left: at.x, top: at.y }} onMouseEnter={show} onMouseLeave={hide}>
            <span className="cn-peek-label">{label}</span>
            <Waveform id={pid} peaksPath={mediaPeaksPath(project.id, mediaId)} audioPath={mediaAudioPath(project.id, mediaId)} height={36} bars={64} showButton />
          </div>,
          document.body,
        )}
    </>
  )
}

/** A node's title; double-click (or 重命名 in the menu) edits it in place. */
function Title({ n, text, placeholder, children }: { n: CanvasNode; text: string; placeholder: string; children?: ReactNode }) {
  const { renaming, actions } = useCanvas()
  const field = n.type === 'source' || n.type === 'audio' ? 'name' : 'label'
  if (renaming === n.id) {
    const commit = (value: string) => {
      const v = value.trim()
      if (field === 'name' && !v) return actions.rename(null)
      actions.patchNode(n.id, (x) => ({ ...x, data: { ...x.data, [field]: v } as CanvasNode['data'] }))
      actions.rename(null)
    }
    return (
      <input
        className="cn-title-input nodrag"
        defaultValue={text}
        placeholder={placeholder}
        autoFocus
        onFocus={(e) => e.currentTarget.select()}
        onBlur={(e) => commit(e.currentTarget.value)}
        onKeyDown={(e) => {
          e.stopPropagation()
          if (e.key === 'Enter') commit(e.currentTarget.value)
          if (e.key === 'Escape') actions.rename(null)
        }}
      />
    )
  }
  return (
    <span
      className="cn-title"
      title={text || placeholder}
      onDoubleClick={(e) => {
        e.stopPropagation()
        actions.rename(n.id)
      }}
    >
      {text || placeholder}
      {children}
    </span>
  )
}

function useJobProgress(jobId: string | null) {
  const job = useActiveJobs((j) => j.id === jobId)[0]
  return job?.progress ?? null
}

/** Human name of an output port. */
export function usePortLabel() {
  const t = useT()
  return (port: string, node?: CanvasNode) => {
    const part = /^part(\d+)$/.exec(port)
    if (part) {
      const custom = node?.type === 'process' ? (node.data as ProcessData<'split'>).params.labels?.[Number(part[1]) - 1] : ''
      return custom || t('port.part', { n: part[1] })
    }
    if (port === 'mix' || port === 'vocals' || port === 'inst' || port === 'drums' || port === 'bass' || port === 'other' || port === 'out') return t(`port.${port}` as const)
    if (port === 'dry' || port === 'reverb') return t(`port.${port}` as const)
    return port
  }
}

/** A node's display name: its label, or what it is. */
export function nodeDisplayName(n: CanvasNode | undefined, characters: Map<string, Character>, t: ReturnType<typeof useT>): string {
  if (!n) return '?'
  switch (n.type) {
    case 'source':
    case 'audio':
      return (n.data as SourceData | AudioData).name
    case 'render': {
      const d = n.data as RenderData
      return d.label || characters.get(d.characterId)?.name || t('node.missing_character')
    }
    case 'mix': {
      const d = n.data as MixData
      return d.label || (d.processor === 'bus' ? t('node.bus') : t('node.mix'))
    }
    case 'process': {
      const d = n.data as ProcessData
      return d.label || t(`op.${d.op}` as const)
    }
    default:
      return ''
  }
}

export function useNodeName() {
  const t = useT()
  const { characters } = useCanvas()
  return (n: CanvasNode | undefined): string => nodeDisplayName(n, characters, t)
}

/** "Song · Vocals", "Lin #2", "Effects": where an input comes from. */
export function useRefLabel() {
  const portLabel = usePortLabel()
  const nodeName = useNodeName()
  const { project } = useCanvas()
  return (ref: PortRef) => {
    const n = project.canvas.nodes.find((x) => x.id === ref.nodeId)
    if (!n) return '?'
    if (isSource(n) || /^part\d+$/.test(ref.port) || (n.type === 'process' && opPorts(n.data as ProcessData).length > 1)) return `${nodeName(n)} · ${portLabel(ref.port, n)}`
    return nodeName(n)
  }
}

/** Output ports something reads from, as `nodeId:port` — computed once per graph version. */
const linkCache = new WeakMap<CanvasNode[], Set<string>>()
function linkedPorts(nodes: CanvasNode[]): Set<string> {
  let s = linkCache.get(nodes)
  if (!s) {
    s = new Set(nodes.flatMap((x) => inputsOf(x).map((r) => `${r.nodeId}:${r.port}`)))
    linkCache.set(nodes, s)
  }
  return s
}

function useLinked(nodeId: string, port: string): boolean {
  const { project } = useCanvas()
  return linkedPorts(project.canvas.nodes).has(`${nodeId}:${port}`)
}

/**
 * The input lane: the socket sits inside the card on the left, next to what is plugged in.
 * The cable comes out from under the card's edge.
 */
function InLane({ linked, children }: { linked: boolean; children: ReactNode }) {
  return (
    <div className={clsx('cn-port in', linked && 'linked')}>
      <Handle type="target" id="in" position={Position.Left} className={clsx('cn-handle in', linked && 'linked')} />
      {children}
    </div>
  )
}

/**
 * An output lane: label, what comes out (a waveform, or what to do to get it) and the socket
 * inside the card's right edge.
 */
function OutLane({
  id,
  nodeId,
  mediaId,
  label,
  peek,
  signal,
  playId,
  tall,
  children,
}: {
  id: string
  nodeId: string
  mediaId: string | null
  label?: string
  peek: string
  signal?: string
  playId?: string
  tall?: boolean
  children?: ReactNode
}) {
  const linked = useLinked(nodeId, id)
  return (
    <div className={clsx('cn-port', signal, !mediaId && 'pending', linked && 'linked', tall && 'tall')}>
      {label && <span className="cn-port-label">{label}</span>}
      {children}
      <OutHandle id={id} nodeId={nodeId} mediaId={mediaId} label={peek} className={clsx(signal ?? 'out', linked && 'linked')} playId={playId} />
    </div>
  )
}

/** What fills an output lane before there is audio: progress, the error, or the button that makes it.
 *  While it works the brand mark runs; mixing and exporting show the level meter instead. */
function StatusLine({ status, jobId, error, label, onRun, runLabel, disabled, mark = 'brand' }: { status: string; jobId: string | null; error: { code: string } | null; label: string; onRun: () => void; runLabel: string; disabled?: boolean; mark?: 'brand' | 'level' }) {
  const t = useT()
  const busy = status === 'running' || status === 'queued'
  const running = status === 'running'
  const progress = useJobProgress(busy ? jobId : null)
  if (busy) {
    return (
      <div className="cn-inline-status" style={progress != null ? ({ ['--p' as string]: progress } as CSSProperties) : undefined}>
        {mark === 'level' ? <LevelMark size={14} running={running} /> : <Logo size={14} running={running} />}
        <span className={clsx(running && 'shimmer-text')}>{running ? label : t('node.status.queued')}</span>
        <span className="spacer" />
        <span className="tabular">{progress != null ? percent(progress) : ''}</span>
      </div>
    )
  }
  if (status === 'failed') {
    return (
      <button className="cn-inline-status failed nodrag" onClick={onRun} title={error ? t.msg(error) : undefined}>
        <AlertTriangle />
        <span className="cn-error-text">{error ? t.msg(error) : t('node.status.failed')}</span>
        <RefreshCw />
      </button>
    )
  }
  return (
    <button className="cn-action primary nodrag" onClick={onRun} disabled={disabled}>
      <Play />
      {runLabel}
    </button>
  )
}

function StaleBar({ onRun, label }: { onRun: () => void; label: string }) {
  return (
    <button className="cn-stale nodrag" onClick={onRun}>
      <RefreshCw />
      {label}
    </button>
  )
}

// ---------------------------------------------------------------------------------------------

type StemPortId = 'mix' | 'vocals' | 'inst' | 'drums' | 'bass' | 'other'

export const SourceNode = memo(function SourceNode({ data, selected }: NodeProps & { data: RFData<SourceData> }) {
  const t = useT()
  const { media, actions, project } = useCanvas()
  const { data: resources } = useResources()
  const portLabel = usePortLabel()
  const [more, setMore] = useState(false)
  const n = data.node
  const d = n.data
  const m = media.get(d.mediaId)
  const stems = d.stems
  const busy = stems?.status === 'queued' || stems?.status === 'running'
  const progress = useJobProgress(busy ? stems.jobId : null)
  const links = linkedPorts(project.canvas.nodes)
  const hasExtra = EXTRA_STEMS.some((p) => stems?.[p])
  const showExtra = more || hasExtra || EXTRA_STEMS.some((p) => links.has(`${n.id}:${p}`))
  const sep = separatorState(resources)
  const separated = !!(stems?.vocals && stems?.inst)
  const wantSplit = !busy && (!separated || stems?.status === 'failed' || (showExtra && !hasExtra))

  const split = async () => {
    if (await ensureSeparator(resources, t)) await actions.separate(n.id, showExtra ? 4 : 2)
  }

  const lane = (p: StemPortId) => {
    const mid = p === 'mix' ? d.mediaId : (stems?.[p] ?? null)
    const empty = busy ? t('node.separating') : sep === 'installing' ? t('node.stem_installing') : sep === 'missing' ? t('node.stem_needs_addon') : t('node.stem_auto')
    return (
      <OutLane key={p} id={p} nodeId={n.id} mediaId={mid} label={portLabel(p)} peek={`${d.name} · ${portLabel(p)}`} signal={p} tall={p === 'mix'}>
        {mid ? <NodeWave mediaId={mid} nodeId={`${n.id}:${p}`} height={p === 'mix' ? 34 : 22} /> : <span className="cn-port-empty">{empty}</span>}
      </OutLane>
    )
  }

  return (
    <div className={clsx('cn cn-source', selected && 'selected', look(d).className)} style={look(d).style}>
      <div className="cn-head">
        <span className="cn-icon source">
          <Music />
        </span>
        <Title n={n} text={d.name} placeholder={d.name} />
        <span className="cn-sub tabular">{m ? clock(m.duration) : ''}</span>
      </div>
      {(d.bpm || d.key) && (
        <div className="cn-meta tabular">
          {[d.bpm ? `${Math.round(d.bpm)} BPM` : '', d.key ?? ''].filter(Boolean).join(' · ')}
        </div>
      )}
      <div className="cn-lanes">
        {lane('mix')}
        {lane('vocals')}
        {lane('inst')}
        {showExtra && EXTRA_STEMS.map(lane)}
      </div>
      {!showExtra && (
        <button className="cn-more nodrag" onClick={() => setMore(true)}>
          <ChevronDown />
          {t('node.more_stems')}
        </button>
      )}
      {busy && (
        <div className="cn-inline-status" style={progress != null ? ({ ['--p' as string]: progress } as CSSProperties) : undefined}>
          <Logo size={14} running={stems?.status === 'running'} />
          <span className={clsx(stems?.status === 'running' && 'shimmer-text')}>{t('node.separating')}</span>
          <span className="spacer" />
          <span className="tabular">{progress != null ? percent(progress) : ''}</span>
        </div>
      )}
      <div className="cn-foot">
        {wantSplit && (
          <button className="cn-action nodrag" onClick={() => void split()} disabled={sep === 'installing'}>
            {stems?.status === 'failed' ? <RefreshCw /> : sep === 'installing' ? <Loader2 className="spin" /> : <Scissors />}
            {stems?.status === 'failed' ? t('node.separate_retry') : sep === 'installing' ? t('node.stem_installing') : t('node.separate_short')}
            {sep === 'missing' && stems?.status !== 'failed' && <span className="cn-tag">{t('node.needs_install')}</span>}
          </button>
        )}
        <button className="cn-action ghost nodrag" onClick={() => actions.openBatch(n.id)}>
          <Users />
          {t('batch.open')}
        </button>
      </div>
    </div>
  )
})

// ---------------------------------------------------------------------------------------------

export const AudioNode = memo(function AudioNode({ data, selected }: NodeProps & { data: RFData<AudioData> }) {
  const t = useT()
  const { media } = useCanvas()
  const n = data.node
  const d = n.data
  const m = media.get(d.mediaId)
  const Icon = entryOf('audio').icon
  return (
    <div className={clsx('cn cn-audio', selected && 'selected', look(d).className)} style={look(d).style}>
      <div className="cn-head">
        <span className="cn-icon audio">
          <Icon />
        </span>
        <Title n={n} text={d.name} placeholder={d.name} />
        <span className="cn-sub tabular">{m ? clock(m.duration) : ''}</span>
      </div>
      <div className="cn-lanes">
        <OutLane id="out" nodeId={n.id} mediaId={d.mediaId} label={t(`audio.role.${d.role}` as const)} peek={d.name} playId={`node:${n.id}`} tall>
          <NodeWave mediaId={d.mediaId} nodeId={n.id} height={30} />
        </OutLane>
      </div>
    </div>
  )
})

// ---------------------------------------------------------------------------------------------

export const TakeNode = memo(function TakeNode({ data, selected }: NodeProps & { data: RFData<RenderData> }) {
  const t = useT()
  const { characters, models, project, actions } = useCanvas()
  const refLabel = useRefLabel()
  const n = data.node
  const d = n.data
  const c = characters.get(d.characterId)
  const model = models.get(d.modelId)
  const release = model?.releases.find((r) => r.id === d.releaseId)
  const quality = d.params.method === 'rk4' ? t('insp.quality.fine') : d.params.steps <= 25 ? t('insp.quality.fast') : t('insp.quality.standard')
  const color = c?.color ?? VOICE_COLORS[4]
  const index = project.canvas.nodes.filter((x) => isTake(x) && x.data.characterId === d.characterId && x.data.input.nodeId === d.input.nodeId).findIndex((x) => x.id === n.id)
  const done = d.status === 'done' && !!d.mediaId

  return (
    <div className={clsx('cn cn-take', `st-${d.status}`, d.stale && 'stale', selected && 'selected', look(d).className)} style={{ ['--voice' as string]: color, ...look(d).style }}>
      <div className="cn-head">
        {c ? <Avatar character={c} size={26} /> : <span className="cn-icon"><AlertTriangle /></span>}
        <Title n={n} text={d.label} placeholder={c?.name || t('node.missing_character')}>
          {!d.label && index > 0 && <span className="cn-index tabular"> #{index + 1}</span>}
        </Title>
        {d.favorite && <Heart className="cn-fav" />}
        {d.rating > 0 && (
          <span className="cn-stars" aria-label={`${d.rating}`}>
            {Array.from({ length: d.rating }).map((_, i) => (
              <Star key={i} />
            ))}
          </span>
        )}
      </div>
      <div className="cn-lanes">
        <InLane linked>
          <span className="cn-port-label">{t('node.input')}</span>
          <span className="cn-ref">{refLabel(d.input)}</span>
        </InLane>
        <div className="cn-params">
          <span className="chip tabular">{signed(d.params.key)}</span>
          <span className="chip">{quality}</span>
          <span className="chip">{release ? release.name : t('node.latest_ckpt')}</span>
          {d.params.region && (
            <span className="chip tabular">
              {clock(d.params.region.start)}–{clock(d.params.region.end)}
            </span>
          )}
        </div>
        <OutLane id="out" nodeId={n.id} mediaId={done ? d.mediaId : null} peek={d.label || c?.name || ''} signal="out" playId={`node:${n.id}`} tall>
          {done ? (
            <NodeWave mediaId={d.mediaId!} color={color} nodeId={n.id} />
          ) : (
            <StatusLine status={d.status} jobId={d.jobId} error={d.error} label={t('node.status.running')} onRun={() => void actions.runNode(n.id)} runLabel={t('insp.render')} />
          )}
        </OutLane>
      </div>
      {d.stale && d.status === 'done' && <StaleBar onRun={() => void actions.runNode(n.id)} label={t('node.stale')} />}
    </div>
  )
})

// ---------------------------------------------------------------------------------------------

export const MixNode = memo(function MixNode({ data, selected }: NodeProps & { data: RFData<MixData> }) {
  const t = useT()
  const { project, characters, actions, processors } = useCanvas()
  const refLabel = useRefLabel()
  const n = data.node
  const d = n.data
  const byId = byIdOf(project.canvas.nodes)
  const processor = processors.find((p) => p.id === d.processor)
  const busy = d.status === 'running' || d.status === 'queued'
  const out = d.outputs[d.selected] ?? d.outputs[0]
  const bus = d.processor === 'bus'
  const Icon = entryOf(bus ? 'bus' : 'mix').icon
  const voices = d.inputs.filter((i) => i.role === 'lead' || i.role === 'harmony')
  const others = d.inputs.filter((i) => i.role === 'backing' || i.role === 'other')
  const name = d.label || (bus ? t('node.bus') : t('node.mix'))

  return (
    <div className={clsx('cn cn-mix', bus && 'bus', `st-${d.status}`, d.stale && 'stale', selected && 'selected', look(d).className)} style={look(d).style}>
      <div className="cn-head">
        <span className="cn-icon mix">
          <Icon />
        </span>
        <Title n={n} text={d.label} placeholder={bus ? t('node.bus') : t('node.mix')} />
        {!bus && (
          <span className="chip cn-processor">
            <SlidersHorizontal />
            {processor ? t.label(processor.name) : t('node.mix_builtin')}
          </span>
        )}
      </div>
      <div className="cn-lanes">
        <InLane linked={d.inputs.length > 0}>
          {d.inputs.length === 0 ? (
            <span className="cn-port-empty">{t('node.mix_connect')}</span>
          ) : (
            <div className="cn-inputs">
              {voices.length > 0 && (
                <span className="cn-port-voices">
                  {voices.map((i) => {
                    const cid = voiceOf(byId, i.ref)
                    const c = cid ? characters.get(cid) : undefined
                    return c ? <Avatar key={i.id} character={c} size={18} /> : null
                  })}
                  <span className="cn-port-names">{voices.map((i) => refLabel(i.ref)).join(' · ')}</span>
                </span>
              )}
              {others.map((i) => (
                <span key={i.id} className={clsx('cn-port-names backing', i.mute && 'muted-input')}>
                  {refLabel(i.ref)}
                </span>
              ))}
            </div>
          )}
        </InLane>
        <OutLane id="out" nodeId={n.id} mediaId={out && !busy ? out.mediaId : null} peek={name} signal="out" playId={`node:${n.id}`} tall>
          {out && !busy ? (
            <NodeWave mediaId={out.mediaId} nodeId={n.id} />
          ) : (
            <StatusLine status={d.status} jobId={d.jobId} error={d.error} label={t('node.mixing')} onRun={() => void actions.runNode(n.id)} runLabel={t('node.mix_go')} disabled={d.inputs.length === 0} mark="level" />
          )}
        </OutLane>
      </div>
      {out && !busy && d.outputs.length > 1 && (
        <div className="cn-variants nodrag">
          {d.outputs.map((o, i) => (
            <button key={o.mediaId} className={clsx('chip button', i === d.selected && 'accent')} onClick={() => actions.patchNode(n.id, (x) => ({ ...x, data: { ...(x.data as MixData), selected: i } }))}>
              {String.fromCharCode(65 + i)}
            </button>
          ))}
        </div>
      )}
      {d.stale && out && !busy && <StaleBar onRun={() => void actions.runNode(n.id)} label={t('node.mix_stale')} />}
    </div>
  )
})

// ---------------------------------------------------------------------------------------------

/** One-line summary of what a process node does, shown under its title. */
export function useOpSummary() {
  const t = useT()
  return (d: ProcessData): string => {
    switch (d.op) {
      case 'trim': {
        const p = (d as ProcessData<'trim'>).params
        return `${clock(p.start)} – ${clock(p.end)}`
      }
      case 'split':
        return t('op.split.summary', { n: (d as ProcessData<'split'>).params.cuts.length + 1 })
      case 'pitch': {
        const p = (d as ProcessData<'pitch'>).params
        const st = p.semitones + p.cents / 100
        return [st ? `${signed(Math.round(st * 100) / 100)} ${t('op.pitch.st')}` : '', p.tempo !== 1 ? `${p.tempo.toFixed(2)}×` : ''].filter(Boolean).join(' · ') || t('op.pitch.none')
      }
      case 'fx': {
        const chain = (d as ProcessData<'fx'>).params.chain
        return chain.length ? t('op.fx.summary', { n: chain.filter((s) => !s.bypass).length }) : t('op.fx.empty')
      }
      case 'separate':
        return t('op.separate.summary', { n: (d as ProcessData<'separate'>).params.stems })
      case 'dereverb':
        return t('op.dereverb.summary')
      case 'comp':
        return t('op.comp.summary', { n: (d as ProcessData<'comp'>).params.segments.length, inputs: d.inputs.length })
      case 'concat':
        return t('op.concat.summary', { n: d.inputs.length })
      case 'export': {
        const p = (d as ProcessData<'export'>).params
        return `${p.format.toUpperCase()} · ${p.sampleRate / 1000} kHz${p.loudness != null ? ` · ${p.loudness} LUFS` : ''}`
      }
    }
  }
}

export const ProcessNode = memo(function ProcessNode({ data, selected }: NodeProps & { data: RFData<ProcessData> }) {
  const t = useT()
  const { actions, project } = useCanvas()
  const portLabel = usePortLabel()
  const refLabel = useRefLabel()
  const summary = useOpSummary()
  const n = data.node
  const d = n.data
  const entry = entryOf(catalogIdOf(n) ?? 'fx')
  const Icon = entry.icon
  const ports = opPorts(d)
  const bypassable = d.op === 'fx' || d.op === 'pitch'
  const byId = byIdOf(project.canvas.nodes)
  const busy = d.status === 'running' || d.status === 'queued'
  const main = playableMedia(n, byId)
  const canRun = runnable(n)
  const name = d.label || t(`op.${d.op}` as const)
  const exported = d.op === 'export' && d.status === 'done' ? (d.result as { path?: string } | null)?.path : null
  const needsRun = !d.bypass && (!main || busy || d.status === 'failed' || d.op === 'export') && !(d.op === 'export' && d.status === 'done' && !d.stale)
  const status = (
    <StatusLine
      status={d.status}
      jobId={d.jobId}
      error={d.error}
      label={t('node.processing')}
      onRun={() => void actions.runNode(n.id)}
      runLabel={d.op === 'export' ? t('op.export.run') : t('node.process_go')}
      disabled={!canRun}
      mark={d.op === 'export' ? 'level' : 'brand'}
    />
  )

  return (
    <div className={clsx('cn cn-process', `op-${d.op}`, `st-${d.status}`, d.stale && 'stale', d.bypass && 'bypassed', selected && 'selected', look(d).className)} style={look(d).style}>
      <div className="cn-head">
        <span className="cn-icon process">
          <Icon />
        </span>
        <Title n={n} text={d.label} placeholder={t(`op.${d.op}` as const)} />
        {d.bypass && (
          <span className="chip cn-bypass">
            <CircleSlash />
            {t('node.bypassed')}
          </span>
        )}
      </div>
      {d.inputs.length > 0 && <div className="cn-summary">{summary(d)}</div>}
      <div className="cn-lanes">
        <InLane linked={d.inputs.length > 0}>
          {d.inputs.length === 0 ? (
            <span className="cn-port-empty">{t('node.connect_input')}</span>
          ) : (
            <span className="cn-ref">{d.inputs.map(refLabel).join(' · ')}</span>
          )}
        </InLane>
        {ports.length === 1 ? (
          <OutLane id={ports[0]} nodeId={n.id} mediaId={busy ? null : main} peek={name} signal="out" playId={`node:${n.id}`} tall>
            {main && !busy && !(needsRun && d.status === 'failed') ? <NodeWave mediaId={main} nodeId={n.id} height={30} /> : needsRun ? status : <span className="cn-port-empty" />}
          </OutLane>
        ) : (
          ports.map((p) => {
            const mid = d.outputs?.[p]
            return (
              <OutLane key={p} id={p} nodeId={n.id} mediaId={busy ? null : (mid ?? null)} label={portLabel(p, n)} peek={`${name} · ${portLabel(p, n)}`} signal="out">
                {mid && !busy ? <NodeWave mediaId={mid} nodeId={`${n.id}:${p}`} height={22} /> : <span className="cn-port-empty" />}
              </OutLane>
            )
          })
        )}
      </div>
      {exported && <div className="cn-exported mono" title={exported}>{exported.split(/[\\/]/).pop()}</div>}
      {needsRun && ports.length !== 1 && status}
      {d.stale && main && !busy && !d.bypass && <StaleBar onRun={() => void actions.runNode(n.id)} label={t('node.process_stale')} />}
      {bypassable && d.bypass && <div className="cn-summary muted">{t('node.bypass_hint')}</div>}
    </div>
  )
})


// ---------------------------------------------------------------------------------------------

export const NoteNode = memo(function NoteNode({ data, selected }: NodeProps & { data: RFData<NoteData> }) {
  const t = useT()
  const { actions } = useCanvas()
  const n = data.node
  return (
    <div className={clsx('cn-note', selected && 'selected')} style={{ ['--note' as string]: n.data.color }}>
      <NodeResizer isVisible={selected} minWidth={160} minHeight={90} lineClassName="cn-resize-line" handleClassName="cn-resize-handle" />
      <textarea
        className="cn-note-text nodrag nowheel"
        value={n.data.text}
        placeholder={t('node.note_placeholder')}
        onChange={(e) => {
          const text = e.target.value
          actions.patchNode(n.id, (x) => ({ ...x, data: { ...(x.data as NoteData), text } }), { history: `note:${n.id}` })
        }}
      />
    </div>
  )
})

export const GroupNode = memo(function GroupNode({ data, selected }: NodeProps & { data: RFData<GroupData> }) {
  const t = useT()
  const { actions } = useCanvas()
  const n = data.node
  return (
    <div className={clsx('cn-group', selected && 'selected')} style={{ ['--group' as string]: n.data.color }}>
      <NodeResizer isVisible={selected} minWidth={240} minHeight={160} lineClassName="cn-resize-line" handleClassName="cn-resize-handle" />
      <input
        className="cn-group-label nodrag"
        value={n.data.label}
        placeholder={t('node.group_label')}
        onChange={(e) => {
          const label = e.target.value
          actions.patchNode(n.id, (x) => ({ ...x, data: { ...(x.data as GroupData), label } }), { history: `group:${n.id}` })
        }}
      />
    </div>
  )
})

export const nodeTypes = {
  source: SourceNode,
  audio: AudioNode,
  render: TakeNode,
  mix: MixNode,
  process: ProcessNode,
  note: NoteNode,
  group: GroupNode,
}

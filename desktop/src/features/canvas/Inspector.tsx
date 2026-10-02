import clsx from 'clsx'
import { AudioLines, ChevronDown, Copy, Cpu, Disc3, Download, FileAudio, Heart, ListPlus, Loader2, Pause, Repeat, Scissors, Sparkles, Trash2, Upload, Users, Wand2 } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { mediaUrl } from '../../api/client'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '../../api/client'
import { useJobs, usePreferences, useResources } from '../../api/hooks'
import { setPreference } from '../resources/ResourcesView'
import type { AudioData, CanvasNode, GroupData, KeySuggestion, NoteData, RenderData, RenderParams, StemPort } from '../../api/types'
import { Avatar } from '../../components/Avatar'
import { Logo } from '../../components/Logo'
import { PlayButton, Waveform } from '../../components/Waveform'
import { Segmented, Slider, Switch, Tip } from '../../components/ui'
import { useT } from '../../i18n'
import { VOICE_COLORS } from '../../lib/color'
import { clock, clockPrecise, hzToNote, signed, steps as fmtSteps } from '../../lib/format'
import { usePlayer } from '../../state/player'
import { CLIP_DRAG_TYPE, mediaAudioPath, mediaPeaksPath, useCanvas } from './context'
import { byIdOf, EXTRA_STEMS, isAudio, isMix, isProcess, isSource, isTake, markStale, outputPorts, playableMedia, QUALITY, qualityOf, resolvePort, STEM_PORTS, type AudioNode, type Quality, type SourceNode, type TakeNode } from './model'
import { usePitchEditor } from './pitch/store'
import { usePortLabel, useRefLabel } from './nodes'
import { MixPanel } from './panels/MixPanel'
import { RateAndFavorite, Section } from './panels/parts'
import { ProcessPanel } from './panels/ProcessPanel'
import { ensureSeparator, separatorState } from './separator'
import { SeparationModelPicker } from '../resources/SeparationModels'

export function Inspector({ selection, suggestions, onSelect }: { selection: string[]; suggestions: Record<string, KeySuggestion>; onSelect: (ids: string[]) => void }) {
  const t = useT()
  const { project, actions } = useCanvas()
  const nodes = selection.map((id) => project.canvas.nodes.find((n) => n.id === id)).filter((n): n is CanvasNode => !!n)

  let body: React.ReactNode
  const takes = nodes.filter(isTake)
  if (nodes.length === 0) body = <Versions onSelect={onSelect} />
  else if (takes.length >= 2 && takes.length === nodes.length) body = <CompareGroup takes={takes} />
  else if (nodes.length > 1)
    body = (
      <div className="insp-multi">
        <span className="section-title">{t('common.items', { n: nodes.length })}</span>
        {nodes.some(isSource) && (
          <button className="btn btn-sm btn-primary" onClick={() => actions.openBatch(nodes.find(isSource)!.id)}>
            <Users />
            {t('batch.many_songs', { n: nodes.filter(isSource).length })}
          </button>
        )}
        <button className="btn btn-sm" onClick={() => void actions.run(nodes.map((n) => n.id))}>
          <Wand2 />
          {t('ctx.run_selected', { n: nodes.length })}
        </button>
        {nodes.some((n) => outputPorts(n).length > 0) && (
          <button className="btn btn-sm" onClick={() => actions.createMix(nodes.filter((n) => outputPorts(n).length > 0 && !isSource(n)).map((n) => ({ nodeId: n.id, port: outputPorts(n)[0] })))}>
            <Disc3 />
            {t('qa.mix_take')}
          </button>
        )}
        <button
          className="btn btn-danger btn-sm"
          onClick={() => {
            actions.removeNodes(nodes.map((n) => n.id))
            onSelect([])
          }}
        >
          <Trash2 />
          {t('insp.delete')}
        </button>
      </div>
    )
  else {
    const n = nodes[0]
    if (isSource(n)) body = <SourcePanel node={n} />
    else if (isAudio(n)) body = <AudioPanel node={n} />
    else if (isTake(n)) body = <RenderPanel key={n.id} node={n} suggestion={suggestions[n.id]} />
    else if (isMix(n)) body = <MixPanel key={n.id} node={n} />
    else if (isProcess(n)) body = <ProcessPanel key={n.id} node={n} />
    else body = <NotePanel node={n} />
  }

  return <aside className="inspector glass-strong">{body}</aside>
}

/** Nothing selected: every take on this canvas, best first, playable and one click from selection. */
/** A finished take or mix can be dragged into the arrangement. */
function dragClip(mediaId: string | null | undefined, nodeId: string, name: string) {
  if (!mediaId) return {}
  return {
    draggable: true,
    onDragStart: (e: React.DragEvent) => {
      e.dataTransfer.setData(CLIP_DRAG_TYPE, JSON.stringify({ mediaId, nodeId, name }))
      e.dataTransfer.effectAllowed = 'copy'
    },
  }
}

function Versions({ onSelect }: { onSelect: (ids: string[]) => void }) {
  const t = useT()
  const { project, characters } = useCanvas()
  const renders = project.canvas.nodes
    .filter((n): n is CanvasNode & { data: RenderData } => n.type === 'render')
    .sort((a, b) => Number(b.data.favorite) - Number(a.data.favorite) || b.data.rating - a.data.rating)
  const mixes = project.canvas.nodes.filter(isMix).filter((n) => n.data.outputs.length > 0)
  if (renders.length === 0) return <div className="insp-empty">{t('insp.nothing')}</div>
  return (
    <div className="insp-versions">
      {mixes.length > 0 && (
        <>
          <div className="insp-section-head">
            <span className="eyebrow">{t('insp.finished')}</span>
          </div>
          {mixes.map((n) => {
            const mid = playableMedia(n)
            return (
              <div key={n.id} className="insp-version" role="button" tabIndex={0} onClick={() => onSelect([n.id])} title={mid ? t('insp.drag_to_timeline') : undefined} {...dragClip(mid, n.id, n.data.label || t('node.mix'))}>
                {mid ? <PlayButton id={`node:${n.id}`} audioPath={mediaAudioPath(project.id, mid)} size={26} /> : <span className="insp-version-state" />}
                <Disc3 className="insp-version-icon" />
                <span className="insp-version-name">{n.data.label || t('node.mix')}</span>
              </div>
            )
          })}
          <div className="insp-versions-gap" />
        </>
      )}
      <div className="insp-section-head">
        <span className="eyebrow">{t('insp.versions')}</span>
        <span className="spacer" />
        <span className="muted tabular">{renders.length}</span>
      </div>
      {renders.map((n, i) => {
        const d = n.data
        const c = characters.get(d.characterId)
        return (
          <div
            key={n.id}
            className="insp-version"
            role="button"
            tabIndex={0}
            onClick={() => onSelect([n.id])}
            title={d.mediaId && d.status === 'done' ? t('insp.drag_to_timeline') : undefined}
            {...dragClip(d.status === 'done' ? d.mediaId : null, n.id, d.label || c?.name || '')}
          >
            {d.mediaId && d.status === 'done' ? (
              <PlayButton id={`node:${n.id}`} audioPath={mediaAudioPath(project.id, d.mediaId)} size={26} />
            ) : (
              <span className="insp-version-state">{d.status === 'running' || d.status === 'queued' ? <Logo size={14} running={d.status === 'running'} /> : null}</span>
            )}
            {c && <Avatar character={c} size={22} />}
            <span className="insp-version-name">
              {d.label || c?.name || t('node.missing_character')}
              {!d.label && <span className="muted tabular"> #{i + 1}</span>}
            </span>
            <span className="muted tabular insp-version-key">{signed(d.params.key)}</span>
            {d.favorite ? <Heart className="insp-version-fav" /> : d.rating > 0 ? <span className="muted tabular">{'★'.repeat(d.rating)}</span> : null}
          </div>
        )
      })}
    </div>
  )
}

function SourcePanel({ node }: { node: SourceNode }) {
  const t = useT()
  const nav = useNavigate()
  const portLabel = usePortLabel()
  const { project, media, voices, actions } = useCanvas()
  const { data: resources } = useResources()
  const separatorReady = resources?.find((r) => r.id === 'separator')?.status === 'ready'
  const { data: prefs } = usePreferences()
  const qc = useQueryClient()
  const d = node.data
  const m = media.get(d.mediaId)
  const stems = d.stems
  const separating = stems?.status === 'queued' || stems?.status === 'running'
  const present = STEM_PORTS.filter((p) => stems?.[p])
  const provided = new Set(stems?.provided ?? [])
  const four = EXTRA_STEMS.some((p) => stems?.[p])
  const [quickVoice, setQuickVoice] = useState<string | null>(null)
  const quickPick = voices.find((c) => c.id === quickVoice) ?? (voices.length === 1 ? voices[0] : undefined)
  const [quickBusy, setQuickBusy] = useState(false)
  const quick = async () => {
    if (!quickPick) return
    setQuickBusy(true)
    try {
      if (await ensureSeparator(resources, t)) await actions.quickSong(node.id, quickPick.id)
    } finally {
      setQuickBusy(false)
    }
  }

  return (
    <div className="insp-body">
      <div className="insp-title">
        <span className="section-title">{d.name}</span>
        <span className="muted tabular">
          {d.bpm ? `${Math.round(d.bpm)} BPM · ` : ''}
          {d.key ? `${d.key} · ` : ''}
          {m ? clock(m.duration) : ''}
        </span>
      </div>
      <div className="insp-wave">
        <Waveform id={`node:${node.id}:mix`} peaksPath={mediaPeaksPath(project.id, d.mediaId)} audioPath={mediaAudioPath(project.id, d.mediaId)} height={52} bars={120} showButton />
      </div>

      {voices.length > 0 && (
        <Section title={t('quick.title')} hint={t('quick.hint')}>
          <div className="insp-voices" role="radiogroup" aria-label={t('quick.title')}>
            {voices.map((c) => (
              <button key={c.id} role="radio" aria-checked={quickPick?.id === c.id} className={clsx('insp-voice', quickPick?.id === c.id && 'on')} onClick={() => setQuickVoice(c.id)}>
                <Avatar character={c} size={34} />
                <span>{c.name}</span>
              </button>
            ))}
          </div>
          <button className="btn btn-primary insp-quick" onClick={() => void quick()} disabled={!quickPick || quickBusy}>
            {quickBusy ? <Logo size={16} running /> : <Sparkles />}
            {quickPick ? t('quick.go', { name: quickPick.name }) : t('quick.pick')}
          </button>
        </Section>
      )}

      <Section title={t('insp.separate')} hint={t('insp.separate_hint')}>
        {present.map((p) => (
          <div key={p} className="insp-stem">
            <span className="insp-stem-label">
              {portLabel(p)}
              {provided.has(p) && <span className="muted"> · {t('insp.stem_provided')}</span>}
            </span>
            <Waveform id={`node:${node.id}:${p}`} peaksPath={mediaPeaksPath(project.id, stems[p]!)} audioPath={mediaAudioPath(project.id, stems[p]!)} height={28} bars={90} showButton />
          </div>
        ))}
        {!separatorReady ? (
          <>
            <p className="muted insp-hint">{t('insp.separate_install')}</p>
            <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
              <button
                className="btn btn-sm"
                disabled={separatorState(resources) === 'installing'}
                onClick={async () => {
                  if (await ensureSeparator(resources, t)) await actions.separate(node.id, 2)
                }}
              >
                {separatorState(resources) === 'installing' ? <Loader2 className="spin" /> : <Scissors />}
                {separatorState(resources) === 'installing' ? t('node.stem_installing') : t('sep.install_go')}
              </button>
              <button className="btn btn-sm btn-ghost" onClick={() => nav('/resources')}>
                {t('fix.install_addon')}
              </button>
            </div>
          </>
        ) : (
          <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
            <button className="btn btn-sm" onClick={() => actions.separate(node.id, 2)} disabled={separating}>
              {separating ? <Logo size={16} running /> : <Scissors />}
              {separating ? t('node.separating') : present.length ? t('insp.separate_again') : t('insp.separate')}
            </button>
            {!four && (
              <Tip label={t('insp.separate4_hint')}>
                <button className="btn btn-sm btn-ghost" onClick={() => actions.separate(node.id, 4)} disabled={separating}>
                  {t('ctx.separate4')}
                </button>
              </Tip>
            )}
          </div>
        )}
        {stems?.status === 'failed' && stems.error && <div className="chip critical insp-error">{t.msg(stems.error)}</div>}
        {separatorReady && (
          <div className="insp-sep-model">
            <span className="field-label">{t('sep.model')}</span>
            <SeparationModelPicker compact value={prefs?.separationModel ?? 'auto'} onChange={(id) => void setPreference(qc, { separationModel: id === 'auto' ? null : id })} />
          </div>
        )}
        <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
          {(['inst', 'vocals'] as StemPort[]).map((p) => (
            <button key={p} className="btn btn-sm btn-ghost" onClick={() => void actions.importStem(node.id, p)}>
              <Upload />
              {t(p === 'inst' ? 'insp.import_inst' : 'insp.import_vocals')}
            </button>
          ))}
        </div>
      </Section>

      <Section title={t('insp.sing_with')}>
        {voices.length === 0 ? (
          <p className="muted insp-hint">{t('create.no_voices')}</p>
        ) : (
          <>
            <div className="insp-voices">
              {voices.map((c) => (
                <button key={c.id} className="insp-voice" onClick={() => actions.sing({ nodeId: node.id, port: stems?.vocals ? 'vocals' : 'mix' }, c.id)}>
                  <Avatar character={c} size={34} />
                  <span>{c.name}</span>
                </button>
              ))}
            </div>
            {voices.length > 1 && (
              <button className="btn btn-sm" onClick={() => actions.openBatch(node.id)}>
                <Users />
                {t('batch.open')}
              </button>
            )}
          </>
        )}
      </Section>

      <button className="btn btn-sm btn-ghost" onClick={() => actions.addToTimeline(node.id)}>
        <ListPlus />
        {t('insp.add_to_timeline')}
      </button>
    </div>
  )
}

function AudioPanel({ node }: { node: AudioNode }) {
  const t = useT()
  const { project, media, actions } = useCanvas()
  const d = node.data
  const m = media.get(d.mediaId)
  const setRole = (role: AudioData['role']) => actions.patchNode(node.id, (n) => ({ ...n, data: { ...(n.data as AudioData), role } }))
  return (
    <div className="insp-body">
      <div className="insp-title">
        <FileAudio className="insp-title-icon" />
        <input className="insp-title-input" value={d.name} onChange={(e) => actions.patchNode(node.id, (n) => ({ ...n, data: { ...(n.data as AudioData), name: e.target.value } }), { history: `${node.id}:name` })} />
        <span className="muted tabular">{m ? clock(m.duration) : ''}</span>
      </div>
      <div className="insp-wave">
        <Waveform id={`node:${node.id}`} peaksPath={mediaPeaksPath(project.id, d.mediaId)} audioPath={mediaAudioPath(project.id, d.mediaId)} height={52} bars={120} showButton />
      </div>
      <Section title={t('audio.role')} hint={t('audio.role_hint')}>
        <Segmented<AudioData['role']>
          size="sm"
          value={d.role}
          onChange={setRole}
          options={(['backing', 'vocal', 'other'] as const).map((r) => ({ value: r, label: t(`audio.role.${r}` as const) }))}
        />
      </Section>
      <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
        <button className="btn btn-sm" onClick={() => actions.createMix([{ nodeId: node.id, port: 'out' }])}>
          <Disc3 />
          {t('qa.mix_with_backing')}
        </button>
        <button className="btn btn-sm btn-ghost" onClick={() => actions.addToTimeline(node.id)}>
          <ListPlus />
          {t('insp.add_to_timeline')}
        </button>
        <button className="btn btn-sm btn-ghost" onClick={() => void actions.exportMedia(d.mediaId, d.name)}>
          <Download />
          {t('ctx.export_audio')}
        </button>
      </div>
    </div>
  )
}

function RenderPanel({ node, suggestion }: { node: TakeNode; suggestion?: KeySuggestion }) {
  const t = useT()
  const portLabel = usePortLabel()
  const { project, characters, models, actions, media } = useCanvas()
  const [advanced, setAdvanced] = useState(false)
  const d = node.data
  const c = characters.get(d.characterId)
  const model = models.get(d.modelId)
  const refLabel = useRefLabel()
  const byId = byIdOf(project.canvas.nodes)
  const input = byId.get(d.input.nodeId)
  const inputMedia = resolvePort(byId, d.input)
  const ports = input ? outputPorts(input).filter((p) => p === d.input.port || (isSource(input) ? p === 'mix' || p === 'vocals' || !!resolvePort(byId, { nodeId: input.id, port: p }) : true)) : []
  const inputDuration = inputMedia ? (media.get(inputMedia)?.duration ?? 0) : 0
  const busy = d.status === 'queued' || d.status === 'running'
  const compare = usePlayer((s) => s.compare)

  const setData = (patch: Partial<RenderData>) => actions.patchNode(node.id, (n) => ({ ...n, data: { ...(n.data as RenderData), ...patch } }))
  /** a change that alters the sound marks a finished take as out of date */
  const setSound = (patch: Partial<RenderData>) => actions.patchNode(node.id, (n) => markStale({ ...n, data: { ...(n.data as RenderData), ...patch } }))
  const setParams = (patch: Partial<RenderParams>) => setSound({ params: { ...d.params, ...patch } })
  const constellation = model && model.kind === 'constellation'
  // the formant shift only acts on models trained with pitch augmentation (imported ones may lack it)
  const noFormant = !!model && model.config.usePitchAug === false
  const openPitch = usePitchEditor((s) => s.open)
  const pitchEdited = d.params.pitchEdit?.segments.length ?? 0

  return (
    <div className="insp-body">
      <div className="insp-voice-head" style={{ ['--voice' as string]: c?.color ?? VOICE_COLORS[4] }}>
        {c && <Avatar character={c} size={44} />}
        <div className="col" style={{ gap: 2, minWidth: 0 }}>
          <strong className="insp-voice-name">{c?.name ?? t('node.missing_character')}</strong>
          <select className="input insp-select" value={d.releaseId ?? ''} onChange={(e) => setSound({ releaseId: e.target.value || null })} aria-label={t('insp.version')}>
            {(model?.releases ?? [])
              .slice()
              .sort((a, b) => b.step - a.step)
              .map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name}
                  {r.notes ? ` · ${r.notes}` : ''}
                </option>
              ))}
            <option value="">
              {t('insp.version_latest')}
              {model ? ` · ${fmtSteps(model.step)}` : ''}
              {model?.status === 'training' ? ` · ${t('stage.training')}` : ''}
            </option>
          </select>
        </div>
      </div>

      {d.status === 'done' && d.mediaId && (
        <div className="insp-result">
          <Waveform id={`node:${node.id}`} peaksPath={mediaPeaksPath(project.id, d.mediaId)} audioPath={mediaAudioPath(project.id, d.mediaId)} height={56} bars={120} color={c?.color} showButton />
          <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
            {inputMedia && (
              <button
                className="btn btn-sm btn-ghost"
                onClick={() =>
                  compare(
                    { id: `node:${node.id}`, url: mediaUrl(mediaAudioPath(project.id, d.mediaId!)) },
                    { id: `node:${input!.id}:${d.input.port}`, url: mediaUrl(mediaAudioPath(project.id, inputMedia)) },
                  )
                }
              >
                <Repeat />
                {t('insp.compare_input')} <span className="kbd">X</span>
              </button>
            )}
            <span className="spacer" />
            <RateAndFavorite rating={d.rating} favorite={d.favorite} onRate={(rating) => setData({ rating })} onFavorite={() => setData({ favorite: !d.favorite })} />
          </div>
        </div>
      )}

      {d.status === 'failed' && d.error && <div className="chip critical insp-error">{t.msg(d.error)}</div>}
      {d.stale && d.status === 'done' && <div className="chip warning insp-error">{t('node.stale')}</div>}

      <Section title={t('batch.input')} aside={<span className="muted insp-input-name">{refLabel(d.input)}</span>}>
        {ports.length > 1 && input && (
          <Segmented<string> size="sm" wrap value={d.input.port} onChange={(port) => setSound({ input: { ...d.input, port } })} options={ports.map((p) => ({ value: p, label: portLabel(p, input) }))} />
        )}
        {!inputMedia && <p className="muted insp-hint">{t('insp.input_pending')}</p>}
      </Section>
      {d.status === 'queued' && <GpuWait jobId={d.jobId} />}

      <Section
        title={t('insp.key')}
        aside={
          suggestion && suggestion.key !== d.params.key ? (
            <Tip label={t('insp.key_suggest_hint', { char: `${hzToNote(suggestion.characterRange.low)}–${hzToNote(suggestion.characterRange.high)}`, song: `${hzToNote(suggestion.sourceRange.low)}–${hzToNote(suggestion.sourceRange.high)}` })}>
              <button className="chip button accent" onClick={() => setParams({ key: suggestion.key })}>
                <Sparkles />
                {t('insp.key_suggest', { key: signed(suggestion.key) })}
              </button>
            </Tip>
          ) : (
            <span className="tabular insp-val">{signed(d.params.key)}</span>
          )
        }
      >
        <Slider value={d.params.key} min={-12} max={12} step={1} centerMark onChange={(key) => setParams({ key })} label={t('insp.key')} />
      </Section>

      <Section
        title={t('pitch.section')}
        aside={<span className="muted insp-input-name">{pitchEdited ? t('pitch.edited_parts', { n: pitchEdited }) : t('pitch.none')}</span>}
      >
        <Tip label={t('pitch.open_hint')}>
          <button className="btn btn-sm" disabled={!inputMedia} onClick={() => openPitch(node.id)}>
            <AudioLines />
            {t('pitch.open')}
          </button>
        </Tip>
      </Section>

      <Section title={t('insp.quality')}>
        <Segmented<Quality>
          size="sm"
          value={qualityOf(d.params)}
          onChange={(q) => setParams(QUALITY[q])}
          options={[
            { value: 'fast', label: t('insp.quality.fast') },
            { value: 'standard', label: t('insp.quality.standard') },
            { value: 'fine', label: t('insp.quality.fine') },
          ]}
        />
      </Section>

      {constellation && model && (
        <Section title={t('insp.mix')}>
          {model.characters.map((mc) => {
            const cc = characters.get(mc.characterId)
            const w = d.params.spkMix?.[mc.characterId] ?? (mc.characterId === d.characterId ? 1 : 0)
            return (
              <div key={mc.characterId} className="insp-mix-row">
                {cc && <Avatar character={cc} size={22} />}
                <span className="insp-mix-name">{cc?.name}</span>
                <div style={{ flex: 1 }}>
                  <Slider
                    value={w}
                    min={0}
                    max={1}
                    step={0.05}
                    onChange={(v) => {
                      const base = d.params.spkMix ?? Object.fromEntries(model.characters.map((x) => [x.characterId, x.characterId === d.characterId ? 1 : 0]))
                      setParams({ spkMix: { ...base, [mc.characterId]: v } })
                    }}
                    label={cc?.name}
                  />
                </div>
                <span className="tabular insp-val">{Math.round(w * 100)}%</span>
              </div>
            )
          })}
        </Section>
      )}

      <button className="insp-adv-toggle" onClick={() => setAdvanced(!advanced)}>
        <ChevronDown style={{ transform: advanced ? 'rotate(180deg)' : undefined }} />
        {t('common.advanced')}
      </button>
      {advanced && (
        <div className="insp-adv">
          <Section title={t('insp.formant')} hint={t('insp.formant_hint')} aside={<span className="tabular insp-val">{signed(d.params.formant)}</span>}>
            <Slider value={d.params.formant} min={-6} max={6} step={1} centerMark disabled={noFormant} onChange={(formant) => setParams({ formant })} label={t('insp.formant')} />
            {noFormant && <p className="muted insp-hint">{t('insp.formant_unsupported')}</p>}
          </Section>
          <Section title={t('insp.register')} hint={t('insp.register_hint')} aside={<span className="tabular insp-val">{signed(d.params.register)}</span>}>
            <Slider value={d.params.register} min={-6} max={6} step={1} centerMark onChange={(register) => setParams({ register })} label={t('insp.register')} />
          </Section>
          <Section title={t('insp.f0')}>
            <select className="input insp-select" value={d.params.f0Extractor} onChange={(e) => setParams({ f0Extractor: e.target.value as RenderParams['f0Extractor'] })}>
              {['rmvpe', 'fcpe', 'parselmouth', 'harvest', 'dio', 'crepe'].map((x) => (
                <option key={x} value={x}>
                  {x.toUpperCase()}
                </option>
              ))}
            </select>
          </Section>
          <Section title={t('insp.f0_range')} hint={t('insp.f0_range_hint')}>
            <div className="row" style={{ gap: 8 }}>
              <input
                className="input tabular"
                type="number"
                min={20}
                max={2000}
                placeholder={t('insp.f0_min')}
                value={d.params.f0Min ?? ''}
                onChange={(e) => setParams({ f0Min: e.target.value ? Number(e.target.value) : null })}
              />
              <span className="muted">–</span>
              <input
                className="input tabular"
                type="number"
                min={20}
                max={2000}
                placeholder={t('insp.f0_max')}
                value={d.params.f0Max ?? ''}
                onChange={(e) => setParams({ f0Max: e.target.value ? Number(e.target.value) : null })}
              />
              <span className="muted">Hz</span>
            </div>
          </Section>
          <Section title={t('insp.threshold')} aside={<span className="tabular insp-val">{d.params.threshold} dB</span>}>
            <Slider value={d.params.threshold} min={-90} max={-30} step={1} onChange={(threshold) => setParams({ threshold })} label={t('insp.threshold')} />
          </Section>
          <Section title={t('insp.tstart')} hint={t('insp.tstart_hint')} aside={<span className="tabular insp-val">{d.params.tStart.toFixed(2)}</span>}>
            <Slider value={d.params.tStart} min={0} max={0.9} step={0.05} onChange={(tStart) => setParams({ tStart })} label={t('insp.tstart')} />
          </Section>
          <Section title={t('insp.region')} aside={<Switch checked={!!d.params.region} onChange={(on) => setParams({ region: on ? { start: 0, end: Math.max(1, Math.min(inputDuration, 30)) } : null })} label={t('insp.region')} />}>
            {d.params.region && (
              <div className="row" style={{ gap: 8 }}>
                <label className="field" style={{ flex: 1 }}>
                  <span className="field-hint">{t('insp.region_from')} · {clockPrecise(d.params.region.start)}</span>
                  <input className="input tabular" type="number" min={0} max={inputDuration} step={0.1} value={d.params.region.start} onChange={(e) => setParams({ region: { ...d.params.region!, start: Math.max(0, Number(e.target.value)) } })} />
                </label>
                <label className="field" style={{ flex: 1 }}>
                  <span className="field-hint">{t('insp.region_to')} · {clockPrecise(d.params.region.end)}</span>
                  <input className="input tabular" type="number" min={0} max={inputDuration} step={0.1} value={d.params.region.end} onChange={(e) => setParams({ region: { ...d.params.region!, end: Math.min(inputDuration || Infinity, Number(e.target.value)) } })} />
                </label>
              </div>
            )}
          </Section>
        </div>
      )}

      <label className="field">
        <span className="eyebrow">{t('insp.label')}</span>
        <input className="input" value={d.label} placeholder={t('insp.label_placeholder')} onChange={(e) => setData({ label: e.target.value })} />
      </label>

      <div className="insp-actions">
        <button className="btn btn-primary" onClick={() => actions.runNode(node.id)} disabled={busy}>
          {busy ? <Logo size={16} running={d.status === 'running'} /> : <Wand2 />}
          {busy ? t(`node.status.${d.status}` as const) : d.status === 'done' ? t('insp.rerender') : t('insp.render')}
          {!busy && <span className="kbd">⏎</span>}
        </button>
        <Tip label={`${t('insp.variant')} (Ctrl/⌘ D)`}>
          <button className="btn" onClick={() => actions.variant(node.id)}>
            <Copy />
            {t('insp.variant')}
          </button>
        </Tip>
        {d.status === 'done' && (
          <>
            <Tip label={`${t('qa.mix_take')} (M)`}>
              <button className="btn" onClick={() => actions.createMix([{ nodeId: node.id, port: 'out' }])}>
                <Disc3 />
                {t('node.mix')}
              </button>
            </Tip>
            <button className="btn btn-ghost" onClick={() => actions.addToTimeline(node.id)}>
              <ListPlus />
              {t('insp.add_to_timeline')}
            </button>
          </>
        )}
      </div>
    </div>
  )
}

/** A render waiting behind training: say so, and offer the two ways out. */
function GpuWait({ jobId }: { jobId: string | null }) {
  const t = useT()
  const qc = useQueryClient()
  const { data: jobs } = useJobs()
  const job = jobs?.find((j) => j.id === jobId)
  const training = jobs?.filter((j) => j.type === 'train' && j.status === 'running') ?? []
  if (job?.status !== 'waiting_gpu' || training.length === 0) return null
  return (
    <div className="insp-wait">
      <div className="row" style={{ gap: 8 }}>
        <Cpu />
        <strong>{t('insp.gpu_wait')}</strong>
      </div>
      <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
        <button className="btn btn-sm" onClick={() => setPreference(qc, { gpuPolicy: 'smart' })}>
          {t('insp.gpu_wait_smart')}
        </button>
        <button className="btn btn-sm btn-ghost" onClick={() => training.forEach((j) => j.target.modelId && api.post(`/models/${j.target.modelId}/pause`).catch(() => {}))}>
          <Pause />
          {t('insp.gpu_wait_pause')}
        </button>
      </div>
    </div>
  )
}

/** Several takes selected: listen to them at the same position, switch with 1–9 or X, rate. */
function CompareGroup({ takes }: { takes: TakeNode[] }) {
  const t = useT()
  const { project, characters, actions } = useCanvas()
  const compareGroup = usePlayer((s) => s.compareGroup)
  const switchTo = usePlayer((s) => s.switchTo)
  const current = usePlayer((s) => s.currentId)
  const group = usePlayer((s) => s.group)
  const ready = takes.filter((n) => n.data.status === 'done' && n.data.mediaId)
  const items = ready.map((n) => ({ id: `node:${n.id}`, url: mediaUrl(mediaAudioPath(project.id, n.data.mediaId!)) }))
  const active = !!group && group.length === items.length && group.every((g, i) => g.id === items[i].id)
  const favorites = takes.filter((n) => n.data.favorite)

  return (
    <div className="insp-body">
      <div className="insp-title">
        <span className="section-title">{t('insp.compare.title')}</span>
        <span className="muted tabular">{takes.length}</span>
      </div>
      <div className="cmp-list">
        {takes.map((n) => {
          const d = n.data
          const c = characters.get(d.characterId)
          const i = ready.indexOf(n)
          const on = current === `node:${n.id}`
          return (
            <div
              key={n.id}
              className={clsx('cmp-row', on && 'on', i < 0 && 'pending')}
              style={{ ['--voice' as string]: c?.color ?? VOICE_COLORS[4] }}
              role="button"
              tabIndex={0}
              onClick={() => (i < 0 ? undefined : active ? switchTo(i) : compareGroup(items, i))}
            >
              <span className="cmp-badge">{i < 0 ? '·' : i + 1}</span>
              {c && <Avatar character={c} size={26} />}
              <div className="col" style={{ gap: 0, minWidth: 0, flex: 1 }}>
                <strong className="cmp-name">{d.label || c?.name}</strong>
                <span className="muted tabular" style={{ fontSize: 'var(--fs-xs)' }}>
                  {signed(d.params.key)} · {t(`insp.quality.${qualityOf(d.params)}` as const)}
                  {i < 0 ? ` · ${t(`node.status.${d.status}` as const)}` : ''}
                </span>
              </div>
              <div onClick={(e) => e.stopPropagation()}>
                <RateAndFavorite
                  rating={d.rating}
                  favorite={d.favorite}
                  onRate={(rating) => actions.patchNode(n.id, (x) => ({ ...x, data: { ...(x.data as RenderData), rating } }))}
                  onFavorite={() => actions.patchNode(n.id, (x) => ({ ...x, data: { ...(x.data as RenderData), favorite: !d.favorite } }))}
                />
              </div>
            </div>
          )
        })}
      </div>
      <div className="insp-actions">
        <button className="btn btn-primary" disabled={items.length === 0} onClick={() => compareGroup(items, 0)}>
          <Sparkles />
          {t('insp.compare.play')}
        </button>
        <span className="muted insp-hint">
          <span className="kbd">1</span>–<span className="kbd">{Math.min(9, Math.max(1, items.length))}</span> · <span className="kbd">X</span>
        </span>
      </div>
      <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
        {takes.some((n) => n.data.status !== 'done' && n.data.status !== 'running' && n.data.status !== 'queued') && (
          <button className="btn btn-sm" onClick={() => void actions.run(takes.map((n) => n.id))}>
            <Wand2 />
            {t('insp.render_all')}
          </button>
        )}
        <button className="btn btn-sm" disabled={takes.length === 0} onClick={() => actions.createMix((favorites.length ? favorites : takes).map((n) => ({ nodeId: n.id, port: 'out' })))}>
          <Disc3 />
          {favorites.length ? t('insp.mix_favorites', { n: favorites.length }) : t('qa.mix_take')}
        </button>
      </div>
    </div>
  )
}

function NotePanel({ node }: { node: CanvasNode }) {
  const t = useT()
  const { actions } = useCanvas()
  const color = (node.data as NoteData | GroupData).color
  return (
    <div className="insp-body">
      <Section title={t('insp.note_color')}>
        <div className="swatches">
          {VOICE_COLORS.map((col) => (
            <button
              key={col}
              className={clsx('swatch', col === color && 'on')}
              style={{ background: col }}
              onClick={() => actions.patchNode(node.id, (n) => ({ ...n, data: { ...(n.data as NoteData), color: col } }))}
              aria-label={col}
            />
          ))}
        </div>
      </Section>
      <button className="btn btn-danger btn-sm" onClick={() => actions.removeNodes([node.id])}>
        <Trash2 />
        {t('insp.delete')}
      </button>
    </div>
  )
}

import clsx from 'clsx'
import {
  ArrowDown,
  ArrowUp,
  ChevronDown,
  ClipboardPaste,
  Copy,
  CopyPlus,
  Crosshair,
  Download,
  FileAudio,
  LocateFixed,
  Magnet,
  Pause,
  Pencil,
  Play,
  Plus,
  Repeat,
  RotateCcw,
  Rows3,
  Scissors,
  SkipBack,
  SlidersVertical,
  StretchHorizontal,
  Trash2,
  Volume2,
  VolumeX,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { api as http, mediaUrl, sendFiles } from '../../../api/client'
import type { CanvasNode, Job, Media, Project, SourceData, Timeline as TimelineData, TimelineClip, Track } from '../../../api/types'
import { confirmDialog, Segmented, Tip } from '../../../components/ui'
import { useT } from '../../../i18n'
import { attempt } from '../../../lib/actions'
import { VOICE_COLORS } from '../../../lib/color'
import { waitForJob } from '../../../lib/jobs'
import { useEvent, useLatest } from '../../../lib/useEvent'
import { pickFiles, platform } from '../../../shell/bridge'
import { usePlayer } from '../../../state/player'
import { useTheme } from '../../../state/theme'
import { toast } from '../../../state/ui'
import { mediaAudioPath, useCanvas } from '../context'
import { expectImport } from '../imports'
import { ContextMenu, type MenuItem } from '../menus'
import { byIdOf, playableMedia } from '../model'
import { newId, type UpdateOptions } from '../useProject'
import { Arrange, type ArrangeControls } from './Arrange'
import type { ClipLook } from './ClipView'
import { ExportDialog } from './ExportDialog'
import { MixEngine } from './mixEngine'
import { Mixer } from './Mixer'
import {
  allClips,
  clamp,
  clipEnd,
  clipsAt,
  copyClips,
  duplicateClips,
  duplicateTrack,
  findClip,
  loopOn,
  moveClips,
  moveTrack,
  patchClips,
  pasteClips,
  placeClip,
  removeClips,
  removeTrack,
  splitClips,
  withLength,
  type ClipBoard,
  type GridMode,
} from './ops'
import { barClock, clock } from './Ruler'
import { palette, Signal, TimelineContext, useStored, type MenuKind, type TimelineApi, type TimelineShared } from './state'
import './timeline.css'

/** copied clips (one arrangement clipboard for the app; pasting checks the project) */
let board: ClipBoard | null = null

const isNode = (n: CanvasNode | undefined, mediaId: string) => {
  if (!n) return false
  if (n.type === 'source') {
    const d = n.data as SourceData
    return d.mediaId === mediaId || Object.values(d.stems).includes(mediaId)
  }
  return playableMedia(n) === mediaId
}

/**
 * The arrangement under the canvas (编排) and its mixer (调音台): tracks of clips taken from the
 * canvas, played with WebAudio and exported by the backend. Edits are undoable (one step per
 * gesture); keys work while focus is anywhere in it.
 */
export function Timeline({ open, onToggle, update }: { open: boolean; onToggle: () => void; update: (fn: (p: Project) => Project, opts?: UpdateOptions) => void }) {
  const t = useT()
  const { project, media, characters, actions } = useCanvas()
  const theme = useTheme((s) => s.resolved)
  const engine = useMemo(() => new MixEngine(), [])
  const pos = useMemo(() => new Signal(0), [])
  const frame = useMemo(() => new Signal(0), [])
  const [playing, setPlaying] = useState(false)
  const [view, setView] = useStored<'arrange' | 'mixer'>('stellaria.tl.view', 'arrange', (v) => v === 'arrange' || v === 'mixer')
  const [snap, setSnap] = useStored<boolean>('stellaria.tl.snap', true)
  const [gridPref, setGridMode] = useStored<GridMode>('stellaria.tl.grid', 'sec', (v) => v === 'sec' || v === 'bars')
  const [follow, setFollow] = useStored<boolean>('stellaria.tl.follow', true)
  const [pps, setPps] = useStored<number>('stellaria.tlZoom', 40, (v) => typeof v === 'number' && v > 0)
  const [selection, setSelection] = useState<string[]>([])
  const [focusTrack, setFocusTrack] = useState<string | null>(null)
  const [draft, setDraft] = useState<TimelineData | null>(null)
  const [menu, setMenu] = useState<{ x: number; y: number; items: MenuItem[]; title?: string } | null>(null)
  const [exportOpen, setExportOpen] = useState(false)
  const draftRef = useRef<TimelineData | null>(null)
  const playStart = useRef(0)
  const hoverTrack = useRef<string | null>(null)
  const arrange = useRef<ArrangeControls | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  /** the user was last working in the arrangement (or its menu) */
  const inside = useRef(false)
  const menuOpen = useLatest(!!menu)
  const tl = draft ?? project.timeline
  const projectRef = useLatest(project)
  const mod = platform === 'macos' ? '⌘' : 'Ctrl+'

  const bpm = useMemo(() => {
    const song = project.canvas.nodes.find((n) => n.type === 'source')
    const v = song ? (song.data as SourceData).bpm : null
    return v && v > 20 ? v : null
  }, [project.canvas.nodes])
  const gridMode: GridMode = bpm ? gridPref : 'sec'

  // ---------- what clips and tracks look like ----------
  const wave = palette(theme).wave
  const mixLabel = t('node.mix')
  const lookFor = useMemo(() => {
    const byId = byIdOf(project.canvas.nodes)
    const cache = new Map<string, ClipLook>()
    return (c: TimelineClip): ClipLook => {
      const key = `${c.nodeId}|${c.mediaId}`
      let look = cache.get(key)
      if (!look) {
        const n = c.nodeId ? byId.get(c.nodeId) : undefined
        if (n?.type === 'render') {
          const d = n.data as { characterId: string; label: string }
          const ch = characters.get(d.characterId)
          look = { label: d.label || ch?.name || media.get(c.mediaId)?.name || '', color: ch?.color ?? VOICE_COLORS[4] }
        } else if (n?.type === 'mix') {
          // a mix's audio is named generically; the node's label (or 成品) says which one it is
          look = { label: (n.data as { label: string }).label || mixLabel, color: wave }
        } else look = { label: media.get(c.mediaId)?.name ?? '', color: wave }
        cache.set(key, look)
      }
      return look
    }
  }, [project.canvas.nodes, characters, media, wave, mixLabel])
  const trackColor = useCallback(
    (tr: Track): string | null => {
      if (tr.kind !== 'vocal') return null
      for (const c of tr.clips) {
        const n = c.nodeId ? project.canvas.nodes.find((x) => x.id === c.nodeId) : undefined
        if (n?.type === 'render') return characters.get((n.data as { characterId: string }).characterId)?.color ?? null
      }
      return null
    },
    [project.canvas.nodes, characters],
  )
  const mediaDuration = useCallback((mediaId: string) => media.get(mediaId)?.duration, [media])

  // ---------- edits ----------
  const current = useEvent(() => draftRef.current ?? projectRef.current.timeline)
  const edit = useEvent((fn: (x: TimelineData) => TimelineData, key?: string) => {
    if (draftRef.current) {
      draftRef.current = null
      setDraft(null)
    }
    update((p) => {
      const next = fn(p.timeline)
      return next === p.timeline ? p : { ...p, timeline: withLength(next) }
    }, key ? { history: `tl:${key}` } : undefined)
  })
  const preview = useEvent((fn: (x: TimelineData) => TimelineData) => {
    draftRef.current = fn(draftRef.current ?? projectRef.current.timeline)
    setDraft(draftRef.current)
  })
  const commit = useEvent((key?: string) => {
    const d = draftRef.current
    if (!d) return
    draftRef.current = null
    setDraft(null)
    if (d !== projectRef.current.timeline) update((p) => ({ ...p, timeline: d }), key ? { history: `tl:${key}` } : undefined)
  })
  const cancel = useEvent(() => {
    draftRef.current = null
    setDraft(null)
  })
  // a slider let go somewhere without telling us: its preview is still committed
  useEffect(() => {
    const onUp = () => window.setTimeout(() => commit(), 0)
    window.addEventListener('pointerup', onUp)
    return () => window.removeEventListener('pointerup', onUp)
  }, [commit])

  const newTrack = useEvent((name?: string, kind: Track['kind'] = 'other'): Track => ({
    id: newId('trk'),
    name: name || t('tl.track_n', { n: current().tracks.length + 1 }),
    kind,
    gainDb: 0,
    pan: 0,
    mute: false,
    solo: false,
    clips: [],
  }))
  /** a take, mix or any media dropped on the lanes: a clip there (a new track past the last one, named after it) */
  const placeMedia = useEvent((item: { mediaId: string; nodeId: string | null; name?: string }, trackIndex: number, start: number): string | null => {
    const duration = media.get(item.mediaId)?.duration
    if (!duration) return null
    const node = item.nodeId ? projectRef.current.canvas.nodes.find((n) => n.id === item.nodeId) : undefined
    const clip: TimelineClip = { id: newId('tc'), mediaId: item.mediaId, nodeId: node ? node.id : null, start: Math.max(0, Math.round(start * 1000) / 1000), offset: 0, duration, gainDb: 0, fadeIn: 0.01, fadeOut: 0.05 }
    edit((x) => placeClip(x, trackIndex, clip, () => newTrack(item.name || media.get(item.mediaId)?.name, node?.type === 'render' ? 'vocal' : 'other')))
    setSelection([clip.id])
    return clip.id
  })

  // ---------- transport ----------
  const urlFor = useCallback((mid: string) => mediaUrl(mediaAudioPath(project.id, mid)), [project.id])
  const playingRef = useLatest(playing)
  const stop = useEvent(() => {
    engine.stop()
    setPlaying(false)
    pos.set(engine.position())
  })
  const play = useEvent(async (from: number = pos.value) => {
    usePlayer.getState().pause()
    playStart.current = from
    setPlaying(true)
    await engine.play(current(), from, urlFor)
  })
  const togglePlay = useEvent(() => (playingRef.current ? stop() : void play()))
  const seek = useEvent((s: number) => {
    const target = Math.max(0, s)
    pos.set(target)
    if (playingRef.current) {
      playStart.current = target
      void engine.play(current(), target, urlFor)
    } else engine.seek(target)
  })

  useEffect(() => () => engine.dispose(), [engine])
  // the canvas's player and the arrangement don't play over each other
  useEffect(() => usePlayer.subscribe((s, prev) => s.playing && !prev.playing && engine.playing && stop()), [engine, stop])

  // the playhead and meters run on one animation frame clock; at the end playback goes back to where it started
  useEffect(() => {
    let raf = 0
    if (!playing) {
      const until = performance.now() + 1500
      const tail = (now: number) => {
        frame.set(now)
        if (now < until) raf = requestAnimationFrame(tail)
      }
      raf = requestAnimationFrame(tail)
      return () => cancelAnimationFrame(raf)
    }
    const tick = (now: number) => {
      if (engine.playing) {
        const p = engine.position()
        const cur = current()
        if (p > Math.max(cur.length, loopOn(cur)?.end ?? 0) + 0.3) {
          engine.stop()
          engine.seek(playStart.current)
          pos.set(playStart.current)
          setPlaying(false)
          return
        }
        pos.set(p)
      }
      frame.set(now)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, engine, pos, frame, current])

  // mixer settings follow live; clip or loop changes are rescheduled from where playback is
  useEffect(() => engine.setMix(tl), [engine, tl])
  const layout = useMemo(() => JSON.stringify([tl.tracks.map((tr) => [tr.id, tr.clips]), loopOn(tl)]), [tl])
  useEffect(() => {
    if (!engine.playing) return
    const id = window.setTimeout(() => void engine.reschedule(current()), 120)
    return () => window.clearTimeout(id)
  }, [layout, engine, current])

  // ---------- selection ----------
  const existing = useMemo(() => new Set(allClips(tl).map((x) => x.clip.id)), [tl])
  const liveSelection = useMemo(() => {
    const kept = selection.filter((id) => existing.has(id))
    return kept.length === selection.length ? selection : kept
  }, [selection, existing])
  const selected = useEvent(() => selection.filter((id) => existing.has(id)))
  const targetTrack = useEvent((): number => {
    const cur = current()
    const first = liveSelection.length ? findClip(cur, liveSelection[0]) : undefined
    const id = focusTrack ?? first?.track.id ?? hoverTrack.current
    return Math.max(0, cur.tracks.findIndex((tr) => tr.id === id))
  })

  // ---------- commands ----------
  const split = useEvent(() => {
    const at = pos.value
    const cur = current()
    // nothing selected: the clips under the playhead on the track under the pointer (or the focused one)
    const ids = selected().length ? selected() : clipsAt(cur, at, hoverTrack.current ?? focusTrack)
    if (!clipsAt(cur, at).some((id) => ids.includes(id))) return false
    let made: string[] = []
    edit((x) => {
      const r = splitClips(x, ids, at, () => newId('tc'))
      made = r.ids
      return r.tl
    })
    setSelection(made)
    return true
  })
  const duplicate = useEvent(() => {
    if (!selected().length) return
    let made: string[] = []
    edit((x) => {
      const r = duplicateClips(x, selected(), () => newId('tc'))
      made = r.ids
      return r.tl
    })
    setSelection(made)
  })
  const copy = useEvent(() => {
    const b = copyClips(current(), selected(), project.id)
    if (b) board = b
    return !!b
  })
  const remove = useEvent(() => {
    if (!selected().length) return
    edit((x) => removeClips(x, selected()))
    setSelection([])
  })
  const cut = useEvent(() => copy() && remove())
  const paste = useEvent((at: number = pos.value, trackIndex: number = targetTrack()) => {
    if (!board || board.projectId !== project.id) return
    const b = board
    let made: string[] = []
    edit((x) => {
      const r = pasteClips(x, b, at, trackIndex, () => newId('tc'), () => newTrack())
      made = r.ids
      return r.tl
    })
    setSelection(made)
  })
  const selectAll = useEvent(() => setSelection(allClips(current()).map((x) => x.clip.id)))
  const toggleLoop = useEvent(() => {
    const cur = current()
    if (cur.loop && cur.loop.end > cur.loop.start) {
      edit((x) => ({ ...x, loop: { ...cur.loop!, on: cur.loop!.on === false } }))
      return
    }
    // no loop range yet: the selection's span, else the whole song
    const sel = allClips(cur).filter((x) => selected().includes(x.clip.id))
    const start = sel.length ? Math.min(...sel.map((x) => x.clip.start)) : 0
    const end = sel.length ? Math.max(...sel.map((x) => clipEnd(x.clip))) : cur.length
    if (end - start > 0.05) edit((x) => ({ ...x, loop: { start, end, on: true } }))
  })
  const addTrack = useEvent(() => {
    const tr = newTrack()
    edit((x) => ({ ...x, tracks: [...x.tracks, tr] }))
    setFocusTrack(tr.id)
  })
  const deleteTrack = useEvent(async (id: string) => {
    const tr = current().tracks.find((x) => x.id === id)
    if (!tr) return
    if (tr.clips.length) {
      const ok = await confirmDialog({ title: t('tl.delete_track_q', { name: tr.name }), body: t('tl.delete_track_body', { n: tr.clips.length }), confirm: t('common.delete'), cancel: t('common.cancel'), danger: true })
      if (!ok) return
    }
    edit((x) => removeTrack(x, id))
  })
  const renameTrack = (id: string) => {
    if (view !== 'arrange') setView('arrange')
    window.setTimeout(() => document.querySelector<HTMLInputElement>(`input[data-track-name="${CSS.escape(id)}"]`)?.select(), 30)
  }
  const reveal = useEvent((clipId: string) => {
    const c = findClip(current(), clipId)?.clip
    if (!c) return
    const nodes = projectRef.current.canvas.nodes
    const n = (c.nodeId ? nodes.find((x) => x.id === c.nodeId) : undefined) ?? nodes.find((x) => isNode(x, c.mediaId))
    if (!n) {
      toast({ kind: 'info', title: t('tl.reveal_missing') })
      return
    }
    actions.select([n.id])
    actions.reveal([n.id])
  })
  const importAt = useEvent(async (files: string[] | File[], trackIndex: number, time: number) => {
    if (!files.length) return
    const pid = project.id
    const job = await attempt(() => sendFiles<Job>(`/projects/${pid}/media`, files))
    if (!job) return
    // the canvas also gets them, as audio cards
    expectImport(job.id, { kind: 'audio' })
    const done = await attempt(() => waitForJob(job.id))
    if (!done) return
    if (done.status !== 'succeeded') {
      if (done.status === 'failed') toast({ kind: 'error', title: t.msg(done.error ? { code: done.error.code, params: done.error.params } : null, t('tl.import_failed')), detail: done.error?.detail ?? undefined })
      return
    }
    const ids = (done.result as { mediaIds?: string[] } | null)?.mediaIds ?? []
    const list = (await attempt(() => http.get<Media[]>(`/projects/${pid}/media`))) ?? []
    if (projectRef.current.id !== pid) return
    const clips: TimelineClip[] = []
    let at = Math.max(0, time)
    for (const mid of ids) {
      const m = list.find((x) => x.id === mid)
      if (!m || m.kind === 'stem' || !m.duration) continue
      clips.push({ id: newId('tc'), mediaId: mid, nodeId: null, start: Math.round(at * 1000) / 1000, offset: 0, duration: m.duration, gainDb: 0, fadeIn: 0.01, fadeOut: 0.05 })
      at += m.duration
    }
    if (!clips.length) return
    const name = list.find((x) => x.id === clips[0].mediaId)?.name
    edit((x) => clips.reduce((acc, c) => placeClip(acc, Math.min(trackIndex, acc.tracks.length), c, () => newTrack(name)), x))
    setSelection(clips.map((c) => c.id))
  })
  const importPicked = async (trackIndex: number, time: number) => {
    const picked = await pickFiles({ multiple: true })
    if (picked?.length) void importAt(picked, trackIndex, time)
  }

  // ---------- context menus ----------
  const openMenu = useEvent((e: { clientX: number; clientY: number }, what: MenuKind) => {
    const cur = current()
    const canPaste = !!board && board.projectId === project.id
    let items: MenuItem[] = []
    let title: string | undefined
    if (what.kind === 'clip') {
      const ids = selected().includes(what.clipId) ? selected() : [what.clipId]
      const clips = allClips(cur).filter((x) => ids.includes(x.clip.id))
      const muted = clips.every((x) => x.clip.mute)
      const canSplit = clipsAt(cur, pos.value).some((id) => ids.includes(id))
      if (!clips.length) return
      title = ids.length > 1 ? t('tl.clips_n', { n: ids.length }) : lookFor(clips[0].clip).label || t('tl.clip')
      items = [
        { icon: <Scissors />, label: t('tl.split'), shortcut: 'S', disabled: !canSplit, onClick: split },
        { icon: <Scissors />, label: t('tl.cut'), shortcut: `${mod}X`, divider: true, onClick: cut },
        { icon: <Copy />, label: t('tl.copy'), shortcut: `${mod}C`, onClick: () => void copy() },
        { icon: <ClipboardPaste />, label: t('tl.paste'), shortcut: `${mod}V`, disabled: !canPaste, onClick: () => paste() },
        { icon: <CopyPlus />, label: t('tl.duplicate'), shortcut: `${mod}D`, onClick: duplicate },
        { icon: muted ? <Volume2 /> : <VolumeX />, label: muted ? t('tl.clip_unmute') : t('tl.clip_mute'), divider: true, onClick: () => edit((x) => patchClips(x, ids, { mute: !muted })) },
        { icon: <RotateCcw />, label: t('tl.reset_fades'), disabled: clips.every((x) => !x.clip.fadeIn && !x.clip.fadeOut), onClick: () => edit((x) => patchClips(x, ids, { fadeIn: 0, fadeOut: 0 })) },
        { icon: <RotateCcw />, label: t('tl.reset_gain'), disabled: clips.every((x) => !x.clip.gainDb), onClick: () => edit((x) => patchClips(x, ids, { gainDb: 0 })) },
        { icon: <Crosshair />, label: t('tl.reveal'), divider: true, disabled: ids.length !== 1, onClick: () => reveal(what.clipId) },
        { icon: <Trash2 />, label: t('tl.delete_clips'), shortcut: 'Del', danger: true, divider: true, onClick: remove },
      ]
    } else if (what.kind === 'track') {
      const i = cur.tracks.findIndex((x) => x.id === what.trackId)
      const tr = cur.tracks[i]
      if (!tr) return
      setFocusTrack(tr.id)
      title = tr.name
      items = [
        { icon: <Pencil />, label: t('tl.rename_track'), onClick: () => renameTrack(tr.id) },
        { icon: <CopyPlus />, label: t('tl.duplicate_track'), onClick: () => edit((x) => duplicateTrack(x, tr.id, (kind) => newId(kind === 'track' ? 'trk' : 'tc'), t('tl.track_copy_name', { name: tr.name })).tl) },
        { icon: <ArrowUp />, label: t('tl.move_up'), divider: true, disabled: i === 0, onClick: () => edit((x) => moveTrack(x, tr.id, i - 1)) },
        { icon: <ArrowDown />, label: t('tl.move_down'), disabled: i === cur.tracks.length - 1, onClick: () => edit((x) => moveTrack(x, tr.id, i + 1)) },
        { icon: <Plus />, label: t('tl.add_track'), divider: true, onClick: addTrack },
        { icon: <Trash2 />, label: t('tl.delete_track'), danger: true, divider: true, onClick: () => void deleteTrack(tr.id) },
      ]
    } else {
      const { trackIndex, time } = what
      items = [
        { icon: <ClipboardPaste />, label: t('tl.paste_here'), shortcut: `${mod}V`, disabled: !canPaste, onClick: () => paste(time, trackIndex) },
        { icon: <FileAudio />, label: t('tl.import_here'), onClick: () => void importPicked(trackIndex, time) },
        { icon: <Plus />, label: t('tl.add_track'), divider: true, onClick: addTrack },
      ]
    }
    setMenu({ x: e.clientX, y: e.clientY, items, title })
  })

  // ---------- keys (anywhere in the arrangement; undo / redo stay with the canvas) ----------
  const onKeyDown = (e: React.KeyboardEvent) => {
    const target = e.target as HTMLElement
    if (e.defaultPrevented || target.closest('input, textarea, select, [contenteditable="true"], [role="dialog"], [role="menu"]')) return
    const modKey = e.metaKey || e.ctrlKey
    const key = e.key.toLowerCase()
    if (modKey && (key === 'z' || key === 'y')) return
    // faders and sliders keep their arrows and Home / End (the ruler is the playhead's slider: those move it)
    const onSlider = target.matches('[role="slider"]:not(.tl-clip):not(.tl-ruler)')
    const step = e.shiftKey ? 10 : e.altKey ? 0.1 : 1
    if (key === ' ') {
      if (playingRef.current || current().length > 0) togglePlay()
    } else if (key === 'home' && !onSlider) seek(0)
    else if (key === 'end' && !onSlider) seek(current().length)
    else if (key === 's' && !modKey && !e.altKey) split()
    // Ctrl / Cmd+K splits too, but opens the command palette when there is nothing to split
    else if (modKey && key === 'k') {
      if (!split()) return
    }
    else if (modKey && key === 'd') duplicate()
    else if (modKey && key === 'c') copy()
    else if (modKey && key === 'x') cut()
    else if (modKey && key === 'v') paste()
    else if (modKey && key === 'a') selectAll()
    else if ((key === 'delete' || key === 'backspace') && !modKey) remove()
    else if (key === 'escape' && (selection.length || menu)) {
      setSelection([])
      setMenu(null)
    } else if (key === 'n' && !modKey) setSnap(!snap)
    else if (key === 'l' && !modKey) toggleLoop()
    else if (key === 'm' && !modKey && !e.altKey) setView(view === 'arrange' ? 'mixer' : 'arrange')
    else if ((key === '+' || key === '=') && !modKey) arrange.current?.zoomBy(1.5)
    else if ((key === '-' || key === '_') && !modKey) arrange.current?.zoomBy(1 / 1.5)
    else if ((key === 'arrowleft' || key === 'arrowright') && !onSlider && !modKey) {
      const dir = key === 'arrowleft' ? -1 : 1
      // a selection is nudged, otherwise the playhead moves
      if (selected().length) edit((x) => moveClips(x, selected(), dir * step * 0.1, 0).tl, 'nudge')
      else seek(pos.value + dir * step)
    } else return
    e.preventDefault()
    e.stopPropagation()
  }

  const shared: TimelineShared = useMemo(() => {
    const api: TimelineApi = {
      edit,
      preview,
      commit,
      cancel,
      select: setSelection,
      focusTrack: setFocusTrack,
      seek,
      current,
      openMenu,
      mediaDuration: (id) => media.get(id)?.duration,
      newTrack,
      placeMedia,
      importAt: (files, ti, time) => void importAt(files, ti, time),
    }
    return { engine, pos, frame, api, playing, snap, grid: gridMode, bpm }
  }, [engine, pos, frame, edit, preview, commit, cancel, seek, current, openMenu, media, newTrack, placeMedia, importAt, playing, snap, gridMode, bpm])

  // a clip that moved to another track, or was deleted / undone away, took the focus with it:
  // it comes back to the arrangement so its keys keep working (and don't reach the canvas)
  useEffect(() => {
    const on = (e: Event) => {
      const el = e.target as HTMLElement | null
      // our context menu is portalled out of the arrangement: it counts as inside while it is open
      inside.current = !!el && (!!rootRef.current?.contains(el) || (menuOpen.current && !!el.closest?.('.ctx-menu')))
    }
    document.addEventListener('pointerdown', on, true)
    document.addEventListener('focusin', on, true)
    return () => {
      document.removeEventListener('pointerdown', on, true)
      document.removeEventListener('focusin', on, true)
    }
  }, [menuOpen])
  useLayoutEffect(() => {
    const a = document.activeElement
    if (inside.current && (!a || a === document.body)) rootRef.current?.focus({ preventScroll: true })
  })

  const playable = tl.tracks.some((tr) => tr.clips.length > 0)
  const openExport = useCallback(() => setExportOpen(true), [])
  const loopActive = !!loopOn(tl)

  return (
    <TimelineContext.Provider value={shared}>
      <div ref={rootRef} className={clsx('timeline glass-strong', open && 'open')} tabIndex={-1} onKeyDown={onKeyDown} role="region" aria-label={t('canvas.timeline')}>
        <span id="tl-keys" className="sr-only">
          {t('tl.keys.timeline')}
        </span>
        <Bar
          open={open}
          onToggle={onToggle}
          playing={playing}
          playable={playable}
          loopActive={loopActive}
          length={tl.length}
          canExport={project.timeline.length > 0}
          view={view}
          setView={setView}
          snap={snap}
          setSnap={setSnap}
          bpm={bpm}
          gridMode={gridMode}
          setGridMode={setGridMode}
          follow={follow}
          setFollow={setFollow}
          pos={pos}
          onSeek={seek}
          onPlay={togglePlay}
          onLoop={toggleLoop}
          onAddTrack={addTrack}
          onExport={openExport}
          arrange={arrange}
        />

        {open && (
          <div className="tl-body">
            {view === 'arrange' ? (
              <Arrange
                tl={tl}
                pps={clamp(pps, 2, 480)}
                setPps={setPps}
                selection={liveSelection}
                focusTrack={focusTrack}
                follow={follow}
                gridMode={gridMode}
                controls={arrange}
                onHoverTrack={(id) => (hoverTrack.current = id)}
                lookFor={lookFor}
                trackColor={trackColor}
                mediaDuration={mediaDuration}
                projectId={project.id}
              />
            ) : (
              <Mixer tl={tl} focusTrack={focusTrack} trackColor={trackColor} />
            )}
          </div>
        )}
        {menu && <ContextMenu {...menu} onClose={() => setMenu(null)} />}
        {exportOpen && <ExportDialog open onOpenChange={setExportOpen} tl={tl} />}
      </div>
    </TimelineContext.Provider>
  )
}

/** "1:05.32 / 3:30.00" (+ "12.3.2" on the bars grid); updated every frame without a render. */
function TimeReadout({ pos, length, bpm }: { pos: Signal<number>; length: number; bpm: number | null }) {
  const now = useRef<HTMLSpanElement>(null)
  const bars = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    const show = (p: number) => {
      if (now.current) now.current.textContent = clock(p)
      if (bars.current && bpm) bars.current.textContent = barClock(p, bpm)
    }
    show(pos.value)
    return pos.subscribe(show)
  }, [pos, bpm])
  return (
    <span className="tl-time tabular">
      <span ref={now} className="tl-time-now" />
      <span className="tl-time-len"> / {clock(length)}</span>
      {bpm && <span ref={bars} className="tl-time-bars" />}
    </span>
  )
}

interface BarProps {
  open: boolean
  onToggle: () => void
  playing: boolean
  playable: boolean
  loopActive: boolean
  length: number
  canExport: boolean
  view: 'arrange' | 'mixer'
  setView: (v: 'arrange' | 'mixer') => void
  snap: boolean
  setSnap: (v: boolean) => void
  bpm: number | null
  gridMode: GridMode
  setGridMode: (v: GridMode) => void
  follow: boolean
  setFollow: (v: boolean) => void
  pos: Signal<number>
  onSeek: (t: number) => void
  onPlay: () => void
  onLoop: () => void
  onAddTrack: () => void
  onExport: () => void
  arrange: React.RefObject<ArrangeControls | null>
}

/** Transport, view and tools; it doesn't render again while a clip is being dragged. */
const Bar = memo(function Bar({
  open,
  onToggle,
  playing,
  playable,
  loopActive,
  length,
  canExport,
  view,
  setView,
  snap,
  setSnap,
  bpm,
  gridMode,
  setGridMode,
  follow,
  setFollow,
  pos,
  onSeek,
  onPlay,
  onLoop,
  onAddTrack,
  onExport,
  arrange,
}: BarProps) {
  const t = useT()
  return (
      <div className="tl-bar">
        <button className="tl-toggle" onClick={onToggle} aria-expanded={open}>
          <ChevronDown className="tl-chevron" />
          <span className="section-title">{t('canvas.timeline')}</span>
        </button>
        <div className="tl-transport">
          <Tip label={`${t('tl.to_start')} (Home)`}>
            <button className="icon-btn sm" onClick={() => onSeek(0)} aria-label={t('tl.to_start')}>
              <SkipBack />
            </button>
          </Tip>
          <Tip label={`${playing ? t('common.pause') : t('common.play')} (${t('tl.key_space')})`}>
            <button className={clsx('tl-play', playing && 'on')} onClick={onPlay} disabled={!playable && !playing} aria-label={playing ? t('common.pause') : t('common.play')} aria-describedby="tl-keys">
              {playing ? <Pause /> : <Play />}
            </button>
          </Tip>
          <Tip label={`${t('tl.loop')} (L) · ${t('tl.loop_hint')}`}>
            <button className={clsx('icon-btn sm', loopActive && 'active')} onClick={onLoop} aria-pressed={loopActive} aria-label={t('tl.loop')}>
              <Repeat />
            </button>
          </Tip>
          <TimeReadout pos={pos} length={length} bpm={gridMode === 'bars' ? bpm : null} />
        </div>
        <span className="spacer" />
        {open && (
          <div className="tl-tools">
            {view === 'arrange' && (
              <>
                <Tip label={`${t('tl.snap')} (N) · ${t('tl.snap_hint')}`}>
                  <button className={clsx('icon-btn sm', snap && 'active')} onClick={() => setSnap(!snap)} aria-pressed={snap} aria-label={t('tl.snap')}>
                    <Magnet />
                  </button>
                </Tip>
                {bpm && (
                  <Segmented<GridMode>
                    size="sm"
                    value={gridMode}
                    onChange={setGridMode}
                    options={[
                      { value: 'sec', label: t('tl.grid_sec'), title: t('tl.grid_sec_hint') },
                      { value: 'bars', label: t('tl.grid_bars'), title: t('tl.grid_bars_title', { bpm: Math.round(bpm) }) },
                    ]}
                  />
                )}
                <Tip label={t('tl.follow')}>
                  <button className={clsx('icon-btn sm', follow && 'active')} onClick={() => setFollow(!follow)} aria-pressed={follow} aria-label={t('tl.follow')}>
                    <LocateFixed />
                  </button>
                </Tip>
                <span className="tl-sep" />
                <Tip label={`${t('tl.zoom_out')} (−)`}>
                  <button className="icon-btn sm" onClick={() => arrange.current?.zoomBy(1 / 1.5)} aria-label={t('tl.zoom_out')}>
                    <ZoomOut />
                  </button>
                </Tip>
                <Tip label={`${t('tl.zoom_in')} (+) · ${t('tl.zoom_wheel')}`}>
                  <button className="icon-btn sm" onClick={() => arrange.current?.zoomBy(1.5)} aria-label={t('tl.zoom_in')}>
                    <ZoomIn />
                  </button>
                </Tip>
                <Tip label={t('tl.zoom_fit')}>
                  <button className="icon-btn sm" onClick={() => arrange.current?.fit()} aria-label={t('tl.zoom_fit')}>
                    <StretchHorizontal />
                  </button>
                </Tip>
                <span className="tl-sep" />
              </>
            )}
            <Segmented<'arrange' | 'mixer'>
              size="sm"
              value={view}
              onChange={setView}
              options={[
                { value: 'arrange', label: t('tl.view_arrange'), icon: <Rows3 /> },
                { value: 'mixer', label: t('tl.mixer'), icon: <SlidersVertical />, title: `${t('tl.mixer')} (M)` },
              ]}
            />
            {/* the arrangement has its own, under the last track */}
            {view === 'mixer' && (
              <button className="btn btn-sm btn-ghost" onClick={onAddTrack}>
                <Plus />
                {t('tl.add_track')}
              </button>
            )}
          </div>
        )}
        <button className="btn btn-sm btn-primary" onClick={onExport} disabled={!canExport}>
          <Download />
          {t('tl.export')}
        </button>
      </div>
  )
})

import clsx from 'clsx'
import { ArrowDown, ArrowLeft, ArrowUp, Eye, EyeOff, Loader2, Magnet, Minus, MousePointer2, Music2, Pause, PenLine, Play, Plus, Redo2, RotateCcw, Sparkles, Undo2, Waves } from 'lucide-react'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { onBeforeQuit } from '../../../app/beforeQuit'
import { api, ApiError, mediaUrl } from '../../../api/client'
import type { Job, RenderData } from '../../../api/types'
import { Avatar } from '../../../components/Avatar'
import { confirmDialog, Segmented, Slider, Tip } from '../../../components/ui'
import { useT } from '../../../i18n'
import { attempt, errorText } from '../../../lib/actions'
import { cssVar, mix } from '../../../lib/color'
import { clockPrecise } from '../../../lib/format'
import { waitForJob } from '../../../lib/jobs'
import { useLatest } from '../../../lib/useEvent'
import { modKey } from '../../../shell/bridge'
import { usePlayer } from '../../../state/player'
import { useTheme } from '../../../state/theme'
import { toast } from '../../../state/ui'
import { mediaAudioPath, useCanvas } from '../context'
import { byIdOf, isTake, markStale, resolvePort, type TakeNode } from '../model'
import * as M from './model'
import { usePitchEditor } from './store'
import './pitch.css'

const KEY_W = 56
const RULER_H = 24
const UNDO_LIMIT = 100
const PREVIEW_MAX = 60

interface View {
  /** seconds at the plot's left edge */
  t0: number
  /** pixels per second */
  pps: number
  /** MIDI at the plot's top edge */
  top: number
  /** pixels per semitone */
  pst: number
}

type Drag =
  | { kind: 'move'; y0: number; base: M.Offsets; notes: M.Note[]; anchor: number; moved: boolean }
  | { kind: 'pen'; base: M.Offsets; buf: M.Offsets; lastI: number; lastM: number; lo: number; hi: number }
  | { kind: 'marquee'; x0: number; y0: number; x1: number; y1: number; additive: boolean }
  | { kind: 'ruler' }

/** Mounted once in the creation workspace: the editor of the sing node `usePitchEditor` names. */
export function PitchEditorHost() {
  const nodeId = usePitchEditor((s) => s.nodeId)
  const close = usePitchEditor((s) => s.close)
  const { project } = useCanvas()
  const node = nodeId ? project.canvas.nodes.find((n) => n.id === nodeId) : undefined
  useEffect(() => {
    if (nodeId && !isTake(node)) close()
  }, [nodeId, node, close])
  if (!isTake(node)) return null
  return createPortal(<PitchEditor key={node.id} node={node} onClose={close} />, document.body)
}

function PitchEditor({ node, onClose }: { node: TakeNode; onClose: () => void }) {
  const t = useT()
  const { project, characters, actions } = useCanvas()
  const d = node.data
  const voice = characters.get(d.characterId)
  const color = voice?.color ?? '#8a9bb5'
  const inputMedia = resolvePort(byIdOf(project.canvas.nodes), d.input)
  const base = `/projects/${project.id}/nodes/${node.id}/pitch`
  const ids = useMemo(() => ({ input: `pitch:${node.id}:in`, take: `pitch:${node.id}:take`, preview: `pitch:${node.id}:preview` }), [node.id])
  const rootRef = useRef<HTMLDivElement>(null)

  // ---------- the pitch line ----------
  const [analysis, setAnalysis] = useState<M.PitchAnalysis | null>(null)
  const [phase, setPhase] = useState<'loading' | 'analyzing' | 'ready' | 'failed'>('loading')
  const [failure, setFailure] = useState<string | null>(null)
  const [round, setRound] = useState(0)
  useEffect(() => {
    if (!inputMedia) return
    let live = true
    const load = async () => {
      setPhase('loading')
      let a: M.PitchAnalysis | null = null
      try {
        a = await api.get<M.PitchAnalysis>(base)
      } catch (e) {
        if (!(e instanceof ApiError && e.code === 'pitch.not_analyzed')) throw e
      }
      if (a?.current) return a
      if (!live) return null
      setPhase('analyzing')
      const job = await api.post<Job>(`${base}/analyze`, {})
      const done = await waitForJob(job.id)
      if (done.status !== 'succeeded') throw new ApiError(500, done.error?.code ?? 'pitch.failed', done.error?.message ?? 'failed', '', done.error?.params ?? {})
      return api.get<M.PitchAnalysis>(base)
    }
    load()
      .then((a) => {
        if (!live || !a) return
        setAnalysis(a)
        setPhase('ready')
      })
      .catch((e) => {
        if (!live) return
        setFailure(errorText(e).title)
        setPhase('failed')
      })
    return () => {
      live = false
    }
  }, [base, inputMedia, round])

  const notes = useMemo(() => (analysis ? M.notesOf(analysis) : []), [analysis])

  // ---------- edits, undo, saving ----------
  const initialEdit = useRef(d.params.pitchEdit)
  const [hist, setHist] = useState<{ stack: M.Offsets[]; index: number } | null>(null)
  const [draft, setDraft] = useState<M.Offsets | null>(null)
  useEffect(() => {
    if (analysis) setHist({ stack: [M.offsetsFromEdit(initialEdit.current, analysis.midi.length, analysis.frameS)], index: 0 })
  }, [analysis])
  const off = hist ? hist.stack[hist.index] : null
  const shown = draft ?? off
  const dirty = useRef(false)
  const commit = useCallback((next: M.Offsets) => {
    dirty.current = true
    setHist((h) => {
      if (!h) return h
      const stack = [...h.stack.slice(0, h.index + 1), next].slice(-UNDO_LIMIT)
      return { stack, index: stack.length - 1 }
    })
  }, [])
  const undo = () => {
    dirty.current = true
    setHist((h) => (h && h.index > 0 ? { ...h, index: h.index - 1 } : h))
  }
  const redo = () => {
    dirty.current = true
    setHist((h) => (h && h.index < h.stack.length - 1 ? { ...h, index: h.index + 1 } : h))
  }

  const save = useCallback(
    (o: M.Offsets, frameS: number) => {
      const edit = M.editFromOffsets(o, frameS)
      actions.patchNode(
        node.id,
        (n) => {
          const data = n.data as RenderData
          return markStale({ ...n, data: { ...data, params: { ...data.params, pitchEdit: edit } } })
        },
        { history: `${node.id}:pitch` },
      )
    },
    [actions, node.id],
  )
  // saving changes the canvas (and so `save`): flush through a ref, so the debounce below only
  // restarts when the edits themselves change
  const saveRef = useLatest(save)
  const pending = useRef<{ o: M.Offsets; frameS: number } | null>(null)
  const flush = useCallback(() => {
    if (!pending.current) return
    saveRef.current(pending.current.o, pending.current.frameS)
    pending.current = null
  }, [saveRef])
  useEffect(() => {
    if (!off || !analysis || !dirty.current) return
    pending.current = { o: off, frameS: analysis.frameS }
    const timer = window.setTimeout(flush, 400)
    return () => window.clearTimeout(timer)
  }, [off, analysis, flush])
  useEffect(() => flush, [flush])
  useEffect(() => onBeforeQuit(flush, 0), [flush])

  // ---------- selection, tools, view ----------
  const [sel, setSel] = useState<Set<number>>(() => new Set())
  const [tool, setTool] = useState<'select' | 'pen'>('select')
  const [showOriginal, setShowOriginal] = useState(true)
  const [amount, setAmount] = useState(0.6)
  const [keySel, setKeySel] = useState<{ tonic: number; mode: 'major' | 'minor' } | null>(null)
  const key = keySel ?? (analysis?.key ? { tonic: analysis.key.tonic, mode: analysis.key.mode } : { tonic: 0, mode: 'major' as const })
  const [cursor, setCursor] = useState(0)
  const [size, setSize] = useState({ w: 800, h: 400 })
  const [view, setView] = useState<View | null>(null)
  const [drag, setDrag] = useState<Drag | null>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useLayoutEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    setSize({ w: el.clientWidth, h: el.clientHeight })
    return () => ro.disconnect()
  }, [phase])

  // fit the sung range and ~15 s once the line is there
  useEffect(() => {
    if (!analysis || view) return
    const voiced = analysis.midi.filter((m): m is number => m != null).sort((a, b) => a - b)
    const lo = voiced.length ? voiced[Math.floor(voiced.length * 0.03)] : 55
    const hi = voiced.length ? voiced[Math.floor(voiced.length * 0.97)] : 72
    const plotH = Math.max(200, size.h - RULER_H)
    const span = Math.max(14, hi - lo + 8)
    const pst = Math.max(8, Math.min(28, plotH / span))
    const plotW = Math.max(200, size.w - KEY_W)
    const firstNote = notes[0] ? notes[0].first * analysis.frameS : 0
    setView({ t0: Math.max(0, firstNote - 0.5), pps: plotW / Math.min(15, Math.max(4, analysis.duration)), top: Math.ceil((lo + hi) / 2 + plotH / pst / 2), pst })
  }, [analysis, view, size, notes])

  const centers = useMemo(() => (analysis && shown ? notes.map((n) => M.centerOf(analysis, shown, n)) : []), [analysis, shown, notes])
  const selectedNotes = useMemo(() => notes.filter((n) => sel.has(n.id)), [notes, sel])
  const changed = analysis && off ? M.changedCount(off, notes) : 0

  // ---------- geometry ----------
  const xOf = (s: number, v: View) => KEY_W + (s - v.t0) * v.pps
  const yOf = (m: number, v: View) => RULER_H + (v.top - m) * v.pst
  const tOf = (x: number, v: View) => v.t0 + (x - KEY_W) / v.pps
  const mOf = (y: number, v: View) => v.top - (y - RULER_H) / v.pst
  const noteAt = (x: number, y: number): M.Note | null => {
    if (!analysis || !view) return null
    for (let k = notes.length - 1; k >= 0; k--) {
      const n = notes[k]
      const x0 = xOf(n.first * analysis.frameS, view)
      const x1 = xOf((n.last + 1) * analysis.frameS, view)
      const yc = yOf(centers[k], view)
      if (x >= x0 - 2 && x <= x1 + 2 && Math.abs(y - yc) <= view.pst * 0.55 + 2) return n
    }
    return null
  }
  const clampView = useCallback(
    (v: View): View => {
      const dur = analysis?.duration ?? 60
      const plotW = Math.max(100, size.w - KEY_W)
      const pps = Math.max(Math.min(plotW / Math.max(dur, 1), 40), Math.min(800, v.pps))
      const pst = Math.max(5, Math.min(48, v.pst))
      const t0 = Math.max(0, Math.min(Math.max(0, dur - plotW / pps), v.t0))
      const visible = (size.h - RULER_H) / pst
      const top = Math.max(24 + visible, Math.min(108, v.top))
      return { t0, pps, top, pst }
    },
    [analysis, size],
  )

  // ---------- drawing ----------
  const theme = useTheme((s) => s.resolved)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !analysis || !shown || !view) return
    const dpr = window.devicePixelRatio || 1
    const { w, h } = size
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(h * dpr)
    }
    const g = canvas.getContext('2d')
    if (!g) return
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    const bg = cssVar('--bg', theme === 'dark' ? '#1b1a18' : '#e9e5df')
    const ink = cssVar('--text-1', theme === 'dark' ? '#ece8e1' : '#2a2622')
    const text3 = cssVar('--text-3')
    const hair = cssVar('--hairline')
    const accent = cssVar('--accent', '#6f8db3')
    const voiceLine = theme === 'dark' ? mix(color, '#ffffff', 0.25) : mix(color, '#1e1b18', 0.35)
    g.clearRect(0, 0, w, h)
    g.fillStyle = bg
    g.fillRect(0, 0, w, h)

    // rows
    const mLo = Math.floor(mOf(h, view))
    const mHi = Math.ceil(view.top)
    for (let m = mLo; m <= mHi; m++) {
      const y = yOf(m + 0.5, view)
      if (M.isBlackKey(m)) {
        g.fillStyle = mix(bg.startsWith('#') ? bg : '#888888', ink.startsWith('#') ? ink : '#000000', 0.045)
        g.fillRect(KEY_W, y, w - KEY_W, view.pst)
      }
      g.fillStyle = (((m % 12) + 12) % 12 === 0 ? text3 : hair) as string
      g.fillRect(KEY_W, Math.round(y + view.pst) - 0.5, w - KEY_W, ((m % 12) + 12) % 12 === 0 ? 1 : 0.5)
    }
    // the key's scale, faintly
    for (let m = mLo; m <= mHi; m++) {
      const pc = (((m - key.tonic) % 12) + 12) % 12
      if (M.nearestInKey(m, key.tonic, key.mode) !== m || pc !== 0) continue
      g.fillStyle = mix('#888888', accent.startsWith('#') ? accent : '#6f8db3', 0.6) + '18'
      g.fillRect(KEY_W, yOf(m + 0.5, view), w - KEY_W, view.pst)
    }

    // time grid + ruler
    const tA = tOf(KEY_W, view)
    const tB = tOf(w, view)
    const step = [0.25, 0.5, 1, 2, 5, 10, 15, 30, 60].find((s) => s * view.pps >= 70) ?? 120
    g.fillStyle = hair
    for (let s = Math.ceil(tA / step) * step; s <= tB; s += step) g.fillRect(Math.round(xOf(s, view)) - 0.5, RULER_H, 0.5, h - RULER_H)

    // notes
    const fs = analysis.frameS
    notes.forEach((n, k) => {
      const x0 = xOf(n.first * fs, view)
      const x1 = xOf((n.last + 1) * fs, view)
      if (x1 < KEY_W || x0 > w) return
      const yc = yOf(centers[k], view)
      const hh = Math.max(6, view.pst * 0.86)
      const selected = sel.has(n.id)
      g.beginPath()
      g.roundRect(Math.max(KEY_W, x0), yc - hh / 2, Math.max(2, x1 - Math.max(KEY_W, x0)), hh, Math.min(6, hh / 2))
      g.fillStyle = color + (selected ? '8c' : '45')
      g.fill()
      if (selected) {
        g.lineWidth = 1.5
        g.strokeStyle = accent
        g.stroke()
      }
    })

    // the original line where it was changed, then the line that will be sung
    const line = (values: (i: number) => number | null, style: string, width: number, dash: number[], only?: (i: number) => boolean) => {
      g.beginPath()
      g.setLineDash(dash)
      g.lineWidth = width
      g.strokeStyle = style
      let open = false
      const i0 = Math.max(0, Math.floor(tA / fs) - 1)
      const i1 = Math.min(analysis.midi.length - 1, Math.ceil(tB / fs) + 1)
      for (let i = i0; i <= i1; i++) {
        const m = values(i)
        if (m == null || (only && !only(i))) {
          open = false
          continue
        }
        const x = xOf(i * fs, view)
        const y = yOf(m, view)
        if (open) g.lineTo(x, y)
        else g.moveTo(x, y)
        open = true
      }
      g.stroke()
      g.setLineDash([])
    }
    if (showOriginal)
      line(
        (i) => analysis.midi[i],
        text3,
        1,
        [3, 3],
        (i) => Math.abs(shown[i]) > 1e-3 || (i > 0 && Math.abs(shown[i - 1]) > 1e-3) || (i + 1 < shown.length && Math.abs(shown[i + 1]) > 1e-3),
      )
    line((i) => M.editedMidi(analysis, shown, i), voiceLine, 1.8, [])

    // marquee
    if (drag?.kind === 'marquee') {
      g.fillStyle = accent + '22'
      g.strokeStyle = accent
      g.lineWidth = 1
      const x = Math.min(drag.x0, drag.x1)
      const y = Math.min(drag.y0, drag.y1)
      g.fillRect(x, y, Math.abs(drag.x1 - drag.x0), Math.abs(drag.y1 - drag.y0))
      g.strokeRect(x + 0.5, y + 0.5, Math.abs(drag.x1 - drag.x0), Math.abs(drag.y1 - drag.y0))
    }

    // cursor (where playback starts)
    const cx = xOf(cursor, view)
    if (cx >= KEY_W && cx <= w) {
      g.fillStyle = accent
      g.fillRect(Math.round(cx) - 0.5, RULER_H, 1, h - RULER_H)
    }

    // ruler
    g.fillStyle = bg
    g.fillRect(KEY_W, 0, w - KEY_W, RULER_H)
    g.fillStyle = hair
    g.fillRect(KEY_W, RULER_H - 1, w - KEY_W, 1)
    g.fillStyle = text3
    g.font = '11px system-ui, sans-serif'
    g.textBaseline = 'middle'
    for (let s = Math.ceil(tA / step) * step; s <= tB; s += step) {
      const x = xOf(s, view)
      g.fillRect(Math.round(x) - 0.5, RULER_H - 6, 1, 6)
      g.fillText(`${Math.floor(s / 60)}:${(s % 60).toFixed(step < 1 ? 2 : 0).padStart(step < 1 ? 5 : 2, '0')}`, x + 4, RULER_H / 2)
    }
    if (cx >= KEY_W && cx <= w) {
      g.fillStyle = accent
      g.beginPath()
      g.moveTo(cx - 5, 0)
      g.lineTo(cx + 5, 0)
      g.lineTo(cx, 7)
      g.fill()
    }

    // keyboard
    g.fillStyle = bg
    g.fillRect(0, 0, KEY_W, h)
    for (let m = mLo; m <= mHi; m++) {
      const y = yOf(m + 0.5, view)
      const black = M.isBlackKey(m)
      g.fillStyle = black ? (theme === 'dark' ? '#0f0e0d' : '#3a3632') : theme === 'dark' ? '#d9d4cc' : '#fbf9f6'
      g.fillRect(0, y, black ? KEY_W * 0.62 : KEY_W - 1, view.pst - 0.5)
      if (((m % 12) + 12) % 12 === 0 && view.pst >= 9) {
        g.fillStyle = theme === 'dark' ? '#3a3632' : '#6b655e'
        g.fillText(M.noteName(m), KEY_W - 28, y + view.pst / 2)
      }
    }
    g.fillStyle = hair
    g.fillRect(KEY_W - 1, 0, 1, h)
    g.fillStyle = bg
    g.fillRect(0, 0, KEY_W, RULER_H)
  })

  // ---------- wheel: scroll and zoom ----------
  const viewRef = useLatest(view)
  useEffect(() => {
    const el = canvasRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      const v = viewRef.current
      if (!v) return
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      if (e.ctrlKey || e.metaKey) {
        const f = Math.exp(-e.deltaY * 0.0018)
        const at = v.t0 + (x - KEY_W) / v.pps
        const pps = v.pps * f
        setView(clampView({ ...v, pps, t0: at - (x - KEY_W) / pps }))
      } else if (e.altKey) {
        const f = Math.exp(-e.deltaY * 0.0018)
        const at = v.top - (y - RULER_H) / v.pst
        const pst = v.pst * f
        setView(clampView({ ...v, pst, top: at + (y - RULER_H) / pst }))
      } else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        const dx = e.shiftKey ? e.deltaY : e.deltaX
        setView(clampView({ ...v, t0: v.t0 + dx / v.pps }))
      } else {
        setView(clampView({ ...v, top: v.top - e.deltaY / v.pst / 2 }))
      }
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [clampView, viewRef, phase])

  // ---------- pointer ----------
  const local = (e: React.PointerEvent | React.MouseEvent) => {
    const r = canvasRef.current!.getBoundingClientRect()
    return { x: e.clientX - r.left, y: e.clientY - r.top }
  }
  const frameAt = (x: number) => (analysis && view ? Math.max(0, Math.min(analysis.midi.length - 1, Math.round(tOf(x, view) / analysis.frameS))) : 0)

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!analysis || !off || !view || e.button !== 0) return
    const { x, y } = local(e)
    rootRef.current?.focus()
    e.currentTarget.setPointerCapture(e.pointerId)
    if (y < RULER_H) {
      if (x >= KEY_W) {
        setCursor(Math.max(0, tOf(x, view)))
        setDrag({ kind: 'ruler' })
      }
      return
    }
    if (x < KEY_W) return
    if (tool === 'pen') {
      const buf = new Float32Array(off)
      const i = frameAt(x)
      const m = e.altKey ? mOf(y, view) : mOf(y, view)
      M.drawLine(analysis, buf, i, m, i, m)
      setDraft(new Float32Array(buf))
      setDrag({ kind: 'pen', base: off, buf, lastI: i, lastM: m, lo: i, hi: i })
      return
    }
    const hit = noteAt(x, y)
    if (hit) {
      let next = sel
      if (e.shiftKey || e.metaKey || e.ctrlKey) {
        next = new Set(sel)
        if (next.has(hit.id)) next.delete(hit.id)
        else next.add(hit.id)
        setSel(next)
        return
      }
      if (!sel.has(hit.id)) {
        next = new Set([hit.id])
        setSel(next)
      }
      const moving = notes.filter((n) => next.has(n.id))
      setDrag({ kind: 'move', y0: y, base: off, notes: moving, anchor: centers[notes.indexOf(hit)], moved: false })
      return
    }
    setDrag({ kind: 'marquee', x0: x, y0: y, x1: x, y1: y, additive: e.shiftKey })
  }

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drag || !analysis || !view) {
      if (canvasRef.current && analysis && view) {
        const { x, y } = local(e)
        canvasRef.current.style.cursor = tool === 'pen' && x > KEY_W && y > RULER_H ? 'crosshair' : y > RULER_H && noteAt(x, y) ? 'ns-resize' : 'default'
      }
      return
    }
    const { x, y } = local(e)
    if (drag.kind === 'ruler') {
      setCursor(Math.max(0, Math.min(analysis.duration, tOf(x, view))))
    } else if (drag.kind === 'move') {
      let by = (drag.y0 - y) / view.pst
      if (!e.altKey) by = Math.round(drag.anchor + by) - drag.anchor
      if (Math.abs(by) < 1e-3 && !drag.moved) return
      setDraft(M.shiftNotes(drag.base, drag.notes, by, analysis.frameS))
      if (!drag.moved) setDrag({ ...drag, moved: true })
    } else if (drag.kind === 'pen') {
      const i = frameAt(x)
      const m = mOf(y, view)
      M.drawLine(analysis, drag.buf, drag.lastI, drag.lastM, i, m)
      setDraft(new Float32Array(drag.buf))
      setDrag({ ...drag, lastI: i, lastM: m, lo: Math.min(drag.lo, i), hi: Math.max(drag.hi, i) })
    } else if (drag.kind === 'marquee') {
      setDrag({ ...drag, x1: x, y1: y })
    }
  }

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drag || !analysis || !view) return
    e.currentTarget.releasePointerCapture?.(e.pointerId)
    if (drag.kind === 'move') {
      if (drag.moved && draft) commit(draft)
    } else if (drag.kind === 'pen') {
      M.blendEdges(drag.buf, drag.base, drag.lo, drag.hi, analysis.frameS)
      commit(new Float32Array(drag.buf))
    } else if (drag.kind === 'marquee') {
      const small = Math.abs(drag.x1 - drag.x0) < 4 && Math.abs(drag.y1 - drag.y0) < 4
      if (small) {
        if (!drag.additive) setSel(new Set())
        setCursor(Math.max(0, tOf(drag.x0, view)))
      } else {
        const tA = tOf(Math.min(drag.x0, drag.x1), view)
        const tB = tOf(Math.max(drag.x0, drag.x1), view)
        const mA = mOf(Math.max(drag.y0, drag.y1), view)
        const mB = mOf(Math.min(drag.y0, drag.y1), view)
        const next = new Set(drag.additive ? sel : [])
        notes.forEach((n, k) => {
          const s = n.first * analysis.frameS
          const en = (n.last + 1) * analysis.frameS
          if (en >= tA && s <= tB && centers[k] + 0.5 >= mA && centers[k] - 0.5 <= mB) next.add(n.id)
        })
        setSel(next)
      }
    }
    setDraft(null)
    setDrag(null)
  }

  const onDoubleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!analysis || !off || tool !== 'select') return
    const { x, y } = local(e)
    const hit = noteAt(x, y)
    if (hit) commit(M.snapToSemitone(analysis, off, [hit]))
  }

  // ---------- operations ----------
  const apply = (fn: (o: M.Offsets) => M.Offsets) => {
    if (!off || !analysis || selectedNotes.length === 0) return
    commit(fn(off))
  }
  const shiftBy = (st: number) => apply((o) => M.shiftNotes(o, selectedNotes, st, analysis!.frameS))
  const resetAll = async () => {
    if (!off || !analysis) return
    const ok = await confirmDialog({ title: t('pitch.reset_all_confirm'), confirm: t('pitch.reset_all'), cancel: t('common.cancel'), danger: true })
    if (ok) commit(new Float32Array(off.length))
  }
  const zoom = (f: number) => view && setView(clampView({ ...view, pps: view.pps * f, t0: view.t0 + ((size.w - KEY_W) / 2) * (1 / view.pps - 1 / (view.pps * f)) }))

  // ---------- playback ----------
  const play = usePlayer((s) => s.play)
  const pause = usePlayer((s) => s.pause)
  const currentId = usePlayer((s) => s.currentId)
  const playing = usePlayer((s) => s.playing)
  const ours = currentId === ids.input || currentId === ids.take || currentId === ids.preview
  const previewStart = useRef(0)
  const [previewing, setPreviewing] = useState(false)
  const takeOffset = d.params.region?.start ?? 0
  const playInput = () => inputMedia && play(ids.input, mediaUrl(mediaAudioPath(project.id, inputMedia)), cursor)
  const playTake = () => d.mediaId && play(ids.take, mediaUrl(mediaAudioPath(project.id, d.mediaId)), Math.max(0, cursor - takeOffset))
  const togglePlay = () => (ours && playing ? pause() : playInput())

  const previewRange = (): [number, number] | null => {
    if (!analysis || !view) return null
    if (selectedNotes.length) {
      const a = Math.min(...selectedNotes.map((n) => n.first)) * analysis.frameS - 0.15
      const b = (Math.max(...selectedNotes.map((n) => n.last)) + 1) * analysis.frameS + 0.15
      return [Math.max(0, a), Math.min(analysis.duration, b, Math.max(0, a) + PREVIEW_MAX)]
    }
    const a = Math.max(0, tOf(KEY_W, view))
    return [a, Math.min(analysis.duration, tOf(size.w, view), a + PREVIEW_MAX)]
  }
  const runPreview = async () => {
    const range = previewRange()
    if (!range || !off || !analysis || previewing) return
    setPreviewing(true)
    const job = await attempt(() => api.post<Job>(`${base}/preview`, { start: range[0], end: range[1], pitchEdit: M.editFromOffsets(off, analysis.frameS) }))
    if (!job) {
      setPreviewing(false)
      return
    }
    const done = await waitForJob(job.id).catch(() => null)
    setPreviewing(false)
    const url = done?.status === 'succeeded' ? (done.result?.url as string | undefined) : undefined
    if (!url) {
      toast({ kind: 'error', title: t('pitch.preview'), body: done?.error ? t.msg({ code: done.error.code, params: done.error.params }, done.error.message) : undefined })
      return
    }
    previewStart.current = range[0]
    play(ids.preview, mediaUrl(url), 0)
  }
  const busy = d.status === 'queued' || d.status === 'running'
  const singAll = async () => {
    flush()
    await actions.runNode(node.id)
  }

  // ---------- keys ----------
  /** The editor's shortcuts; true when the key was one of them. */
  const onKey = (e: KeyboardEvent): boolean => {
    const mod = e.metaKey || e.ctrlKey
    const k = e.key.toLowerCase()
    if (mod && k === 'z') e.shiftKey ? redo() : undo()
    else if (mod && k === 'y') redo()
    else if (mod && k === 'a') setSel(new Set(notes.map((n) => n.id)))
    else if (e.key === ' ') togglePlay()
    else if (e.key === 'Escape') {
      if (sel.size) setSel(new Set())
      else close()
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') shiftBy((e.key === 'ArrowUp' ? 1 : -1) * (e.shiftKey ? 12 : e.altKey ? 0.1 : 1))
    else if (e.key === 'Delete' || e.key === 'Backspace') apply((o) => M.resetNotes(o, selectedNotes))
    else if (mod) return false
    else if (k === 'v') setTool('select')
    else if (k === 'p' || k === 'b') setTool('pen')
    else if (k === 's') apply((o) => M.snapToSemitone(analysis!, o, selectedNotes))
    else if (k === 'k') apply((o) => M.snapToKey(analysis!, o, selectedNotes, key.tonic, key.mode))
    else if (e.key === '=' || e.key === '+') zoom(1.25)
    else if (e.key === '-') zoom(0.8)
    else return false
    e.preventDefault()
    return true
  }
  const onKeyRef = useLatest(onKey)
  // On the window (capture), so keys work wherever focus ended up (a disabled button drops it to
  // the page) and never reach the canvas underneath; other dialogs and text fields keep theirs.
  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      const root = rootRef.current
      const target = e.target as HTMLElement | null
      if (!root) return
      const onPage = !target || target === document.body || target === document.documentElement
      if (!onPage && !root.contains(target)) return
      if (target?.closest?.('input, select, textarea, [role="slider"]') && e.key !== 'Escape') return
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') return
      if (onKeyRef.current(e) || onPage) e.stopPropagation()
    }
    window.addEventListener('keydown', listener, true)
    return () => window.removeEventListener('keydown', listener, true)
  }, [onKeyRef])
  useEffect(() => {
    rootRef.current?.focus()
  }, [])

  const close = () => {
    flush()
    if (ours) pause()
    onClose()
  }

  const keyOptions = useMemo(
    () =>
      (['major', 'minor'] as const).flatMap((mode) =>
        M.NOTE_NAMES.map((name, tonic) => ({ value: `${tonic}:${mode}`, label: `${name} ${t(mode === 'major' ? 'pitch.major' : 'pitch.minor')}` })),
      ),
    [t],
  )
  const has = selectedNotes.length > 0

  return (
    <div className="pitch-editor" role="dialog" aria-label={t('pitch.title', { name: voice?.name ?? '' })} tabIndex={-1} ref={rootRef}>
      <header className="pe-head">
        <button className="btn btn-ghost btn-sm" onClick={close}>
          <ArrowLeft />
          {t('pitch.back')}
        </button>
        <div className="pe-title" style={{ ['--voice' as string]: color }}>
          {voice && <Avatar character={voice} size={26} />}
          <strong>{t('pitch.title', { name: voice?.name ?? '' })}</strong>
          {analysis && <span className="muted">{changed ? t('pitch.edited', { n: changed }) : t('pitch.none')}</span>}
        </div>
        <span className="spacer" />
        <Tip label={`${t('pitch.undo')} · ${modKey('Z')}`}>
          <button className="icon-btn" aria-label={t('pitch.undo')} disabled={!hist || hist.index === 0} onClick={undo}>
            <Undo2 />
          </button>
        </Tip>
        <Tip label={`${t('pitch.redo')} · ${modKey('Y')}`}>
          <button className="icon-btn" aria-label={t('pitch.redo')} disabled={!hist || hist.index >= hist.stack.length - 1} onClick={redo}>
            <Redo2 />
          </button>
        </Tip>
        <Tip label={t('pitch.preview_hint')}>
          <button className="btn btn-sm" disabled={!analysis || previewing || !d.modelId} onClick={() => void runPreview()}>
            {previewing ? <Loader2 className="spin" /> : <Sparkles />}
            {previewing ? t('pitch.previewing') : t('pitch.preview')}
          </button>
        </Tip>
        <button className="btn btn-sm btn-primary" disabled={!analysis || busy || !d.modelId} onClick={() => void singAll()}>
          {busy ? <Loader2 className="spin" /> : <Music2 />}
          {busy ? t('pitch.applying') : t('pitch.apply')}
        </button>
      </header>

      <div className="pe-tools">
        <Segmented<'select' | 'pen'>
          size="sm"
          value={tool}
          onChange={setTool}
          options={[
            { value: 'select', label: t('pitch.tool.select'), icon: <MousePointer2 /> },
            { value: 'pen', label: t('pitch.tool.pen'), icon: <PenLine /> },
          ]}
        />
        <span className="pe-sep" />
        <span className={clsx('pe-sel', !has && 'muted')}>{has ? t('pitch.selected', { n: selectedNotes.length }) : t('pitch.select_hint')}</span>
        <Tip label={`${t('pitch.up')} · ↑`}>
          <button className="icon-btn" aria-label={t('pitch.up')} disabled={!has} onClick={() => shiftBy(1)}>
            <ArrowUp />
          </button>
        </Tip>
        <Tip label={`${t('pitch.down')} · ↓`}>
          <button className="icon-btn" aria-label={t('pitch.down')} disabled={!has} onClick={() => shiftBy(-1)}>
            <ArrowDown />
          </button>
        </Tip>
        <Tip label={`${t('pitch.snap')} · S`}>
          <button className="btn btn-sm btn-ghost" disabled={!has} onClick={() => apply((o) => M.snapToSemitone(analysis!, o, selectedNotes))}>
            <Magnet />
            {t('pitch.snap')}
          </button>
        </Tip>
        <Tip label={`${t('pitch.snap_key')} · K`}>
          <button className="btn btn-sm btn-ghost" disabled={!has} onClick={() => apply((o) => M.snapToKey(analysis!, o, selectedNotes, key.tonic, key.mode))}>
            <Magnet />
            {t('pitch.snap_key')}
          </button>
        </Tip>
        <Tip label={t('pitch.straighten_hint')}>
          <button className="btn btn-sm btn-ghost" disabled={!has} onClick={() => apply((o) => M.straighten(analysis!, o, selectedNotes, amount))}>
            <Minus />
            {t('pitch.straighten')}
          </button>
        </Tip>
        <div className="pe-amount">
          <Slider value={amount} min={0.1} max={1} step={0.05} onChange={setAmount} label={t('pitch.straighten')} />
          <span className="tabular muted">{Math.round(amount * 100)}%</span>
        </div>
        <button className="btn btn-sm btn-ghost" disabled={!has} onClick={() => apply((o) => M.addVibrato(o, selectedNotes, 0.35, 5.5, analysis!.frameS))}>
          <Waves />
          {t('pitch.vibrato')}
        </button>
        <button className="btn btn-sm btn-ghost" disabled={!has} onClick={() => apply((o) => M.resetNotes(o, selectedNotes))}>
          <RotateCcw />
          {t('pitch.reset')}
        </button>
        <span className="spacer" />
        <label className="pe-key">
          <span className="muted">{t('pitch.key')}</span>
          <select className="input" value={`${key.tonic}:${key.mode}`} onChange={(e) => {
            const [tonic, mode] = e.target.value.split(':')
            setKeySel({ tonic: Number(tonic), mode: mode as 'major' | 'minor' })
          }}>
            {keyOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
                {analysis?.key && o.value === `${analysis.key.tonic}:${analysis.key.mode}` ? ` ${t('pitch.key_detected')}` : ''}
              </option>
            ))}
          </select>
        </label>
        <Tip label={t('pitch.show_original')}>
          <button className={clsx('icon-btn', showOriginal && 'on')} aria-pressed={showOriginal} aria-label={t('pitch.show_original')} onClick={() => setShowOriginal((v) => !v)}>
            {showOriginal ? <Eye /> : <EyeOff />}
          </button>
        </Tip>
        <Tip label={t('pitch.zoom_out')}>
          <button className="icon-btn" aria-label={t('pitch.zoom_out')} onClick={() => zoom(0.8)}>
            <Minus />
          </button>
        </Tip>
        <Tip label={t('pitch.zoom_in')}>
          <button className="icon-btn" aria-label={t('pitch.zoom_in')} onClick={() => zoom(1.25)}>
            <Plus />
          </button>
        </Tip>
        <button className="btn btn-sm btn-ghost" disabled={!changed} onClick={() => void resetAll()}>
          {t('pitch.reset_all')}
        </button>
      </div>

      <div className="pe-stage" ref={wrapRef}>
        {phase === 'ready' && analysis && view ? (
          <>
            <canvas
              ref={canvasRef}
              className="pe-canvas"
              style={{ width: size.w, height: size.h }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={() => {
                setDraft(null)
                setDrag(null)
              }}
              onDoubleClick={onDoubleClick}
            />
            <Playhead ids={ids} view={view} width={size.w} previewStart={previewStart} takeOffset={takeOffset} />
          </>
        ) : (
          <div className="pe-empty">
            {!inputMedia ? (
              <p className="muted">{t('pitch.no_input')}</p>
            ) : phase === 'failed' ? (
              <>
                <strong>{t('pitch.failed')}</strong>
                {failure && <p className="muted">{failure}</p>}
                <button className="btn btn-sm" onClick={() => setRound((r) => r + 1)}>
                  {t('pitch.retry')}
                </button>
              </>
            ) : (
              <>
                <Loader2 className="spin" />
                <strong>{t('pitch.analyzing')}</strong>
                <p className="muted">{t('pitch.analyzing_hint')}</p>
              </>
            )}
          </div>
        )}
      </div>

      {analysis && view && <Overview analysis={analysis} notes={notes} centers={centers} color={color} view={view} plotW={size.w - KEY_W} onMove={(t0) => setView(clampView({ ...view, t0 }))} />}

      <footer className="pe-foot">
        <button className="btn btn-sm" onClick={togglePlay} disabled={!inputMedia}>
          {ours && playing ? <Pause /> : <Play />}
          {t('pitch.play_input')}
        </button>
        <button className="btn btn-sm btn-ghost" onClick={playTake} disabled={!d.mediaId}>
          <Play />
          {t('pitch.play_take')}
        </button>
        <span className="tabular muted pe-time">{clockPrecise(cursor)}</span>
        <span className="spacer" />
        <span className="muted pe-hints">{t('pitch.hints')}</span>
      </footer>
    </div>
  )
}

/** The moving playhead, apart from the canvas so playback doesn't redraw it. */
function Playhead({ ids, view, width, previewStart, takeOffset }: { ids: { input: string; take: string; preview: string }; view: View; width: number; previewStart: React.RefObject<number>; takeOffset: number }) {
  const currentId = usePlayer((s) => s.currentId)
  const time = usePlayer((s) => s.time)
  const offset = currentId === ids.input ? 0 : currentId === ids.take ? takeOffset : currentId === ids.preview ? previewStart.current : null
  if (offset == null) return null
  const x = KEY_W + (offset + time - view.t0) * view.pps
  if (x < KEY_W || x > width) return null
  return <div className="pe-playhead" style={{ transform: `translateX(${x}px)` }} />
}

/** The whole song small, with the visible part framed; click or drag to go somewhere. */
function Overview({ analysis, notes, centers, color, view, plotW, onMove }: { analysis: M.PitchAnalysis; notes: M.Note[]; centers: number[]; color: string; view: View; plotW: number; onMove: (t0: number) => void }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const [w, setW] = useState(600)
  const h = 34
  useLayoutEffect(() => {
    const el = ref.current?.parentElement
    if (!el) return
    const ro = new ResizeObserver(() => setW(el.clientWidth))
    ro.observe(el)
    setW(el.clientWidth)
    return () => ro.disconnect()
  }, [])
  const voiced = useMemo(() => {
    const v = centers.filter((c) => Number.isFinite(c))
    return v.length ? [Math.min(...v) - 2, Math.max(...v) + 2] : [48, 72]
  }, [centers])
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const dpr = window.devicePixelRatio || 1
    c.width = Math.round(w * dpr)
    c.height = Math.round(h * dpr)
    const g = c.getContext('2d')
    if (!g) return
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, w, h)
    const sx = (w - KEY_W) / Math.max(analysis.duration, 1)
    const [lo, hi] = voiced
    g.fillStyle = color + 'aa'
    notes.forEach((n, k) => {
      const x = KEY_W + n.first * analysis.frameS * sx
      const y = 3 + (1 - (centers[k] - lo) / Math.max(1, hi - lo)) * (h - 8)
      g.fillRect(x, y, Math.max(1, (n.last - n.first + 1) * analysis.frameS * sx), 2)
    })
    g.strokeStyle = cssVar('--accent', '#6f8db3')
    g.lineWidth = 1.5
    g.strokeRect(KEY_W + view.t0 * sx + 0.5, 1, Math.max(4, (plotW / view.pps) * sx), h - 2)
  }, [analysis, notes, centers, color, view, plotW, w, voiced])
  const go = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const sx = (w - KEY_W) / Math.max(analysis.duration, 1)
    onMove((e.clientX - r.left - KEY_W) / sx - plotW / view.pps / 2)
  }
  return (
    <div className="pe-overview">
      <canvas
        ref={ref}
        style={{ width: w, height: h }}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          go(e)
        }}
        onPointerMove={(e) => e.buttons === 1 && go(e)}
      />
    </div>
  )
}

import { useReactFlow } from '@xyflow/react'
import clsx from 'clsx'
import { ArrowLeft, FileAudio, Frame, LayoutDashboard, Map as MapIcon, Maximize, Music, PanelRight, Plus, Redo2, StickyNote, Undo2, Upload } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { CanvasNode, Character, PortRef, Project } from '../../api/types'
import { Avatar } from '../../components/Avatar'
import { Tip } from '../../components/ui'
import { useT } from '../../i18n'
import { toast } from '../../state/ui'
import type { CanvasActions } from './context'
import type { AddMenuState } from './menus'
import { isContent, isMix } from './model'
import { FIT, type CanvasEdit } from './useCanvasActions'
import type { CanvasTools } from './useCanvasTools'
import type { SaveState, UpdateOptions } from './useProject'
import { singablePort } from './voices'

const MINIMAP_KEY = 'stellaria.canvas.minimap'
const INSPECTOR_KEY = 'stellaria.canvas.inspector'

/** An on/off view preference remembered across sessions. */
function useStoredToggle(key: string, fallback: boolean) {
  const [on, setOn] = useState(() => {
    try {
      const v = localStorage.getItem(key)
      return v === null ? fallback : v === '1'
    } catch {
      return fallback
    }
  })
  const toggle = () => {
    setOn(!on)
    try {
      localStorage.setItem(key, on ? '0' : '1')
    } catch {
      /* private mode */
    }
  }
  return [on, toggle] as const
}

/** A panel size (px) remembered across sessions; `reset` goes back to the default. */
export function useStoredSize(key: string, fallback: number) {
  const [size, setSize] = useState(() => {
    try {
      const v = Number(localStorage.getItem(key))
      return Number.isFinite(v) && v > 0 ? v : fallback
    } catch {
      return fallback
    }
  })
  const save = (v: number | null) => {
    setSize(v ?? fallback)
    try {
      if (v === null) localStorage.removeItem(key)
      else localStorage.setItem(key, String(v))
    } catch {
      /* private mode */
    }
  }
  return [size, (v: number) => save(v), () => save(null)] as const
}

export const useMinimapPref = () => useStoredToggle(MINIMAP_KEY, false)

/** The side panel (inspector) can be put away for a wider canvas. */
export const useInspectorPref = () => useStoredToggle(INSPECTOR_KEY, true)

/** Back, the project name, save state, undo / redo. */
export function CanvasTopBar({
  project,
  update,
  saveState,
  undo,
  redo,
  canUndo,
  canRedo,
}: {
  project: Project
  update: (fn: (p: Project) => Project, opts?: UpdateOptions) => void
  saveState: SaveState
  undo: () => void
  redo: () => void
  canUndo: boolean
  canRedo: boolean
}) {
  const t = useT()
  const nav = useNavigate()
  return (
    <div className="canvas-topbar glass-strong">
      <Tip label={t('canvas.back')}>
        <button className="icon-btn sm" onClick={() => nav('/create')}>
          <ArrowLeft />
        </button>
      </Tip>
      <input
        className="canvas-name"
        value={project.name}
        onChange={(e) => {
          const name = e.target.value
          update((p) => ({ ...p, name }), { history: false })
        }}
        aria-label={t('common.rename')}
      />
      <span className={clsx('canvas-save', saveState)}>{saveState === 'saving' || saveState === 'dirty' ? t('canvas.saving') : t('canvas.saved')}</span>
      <span className="canvas-topbar-sep" />
      <Tip label={`${t('canvas.undo')} (⌘Z)`}>
        <button className="icon-btn sm" onClick={undo} disabled={!canUndo}>
          <Undo2 />
        </button>
      </Tip>
      <Tip label={`${t('canvas.redo')} (⇧⌘Z)`}>
        <button className="icon-btn sm" onClick={redo} disabled={!canRedo}>
          <Redo2 />
        </button>
      </Tip>
    </div>
  )
}

/** The tool rail on the left: add, import, note, frame, tidy up, fit, minimap, side panel. */
export function CanvasToolRail({
  selection,
  actions,
  edit,
  tools,
  openAddMenu,
  minimap,
  toggleMinimap,
  inspector,
  toggleInspector,
}: {
  selection: string[]
  actions: CanvasActions
  edit: CanvasEdit
  tools: CanvasTools
  openAddMenu: (state: AddMenuState) => void
  minimap: boolean
  toggleMinimap: () => void
  inspector: boolean
  toggleInspector: () => void
}) {
  const t = useT()
  const rf = useReactFlow()
  const { importFiles } = edit
  return (
    <div className="canvas-tools glass-strong">
      <Tip label={`${t('ctx.add_node')} (Tab)`} side="right">
        <button
          className="icon-btn accent-btn"
          onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect()
            openAddMenu({ x: r.right + 10, y: r.top, flow: rf.screenToFlowPosition({ x: window.innerWidth / 2, y: window.innerHeight / 2 }) })
          }}
        >
          <Plus />
        </button>
      </Tip>
      <Tip label={t('canvas.import')} side="right">
        <button className="icon-btn" onClick={() => void importFiles('song')}>
          <Upload />
        </button>
      </Tip>
      <Tip label={t('canvas.import_audio')} side="right">
        <button className="icon-btn" onClick={() => void importFiles('audio')}>
          <FileAudio />
        </button>
      </Tip>
      <Tip label={`${t('canvas.note')} (N)`} side="right">
        <button className="icon-btn" onClick={() => void actions.addNode('note', { position: rf.screenToFlowPosition({ x: window.innerWidth / 2, y: window.innerHeight / 2 }) })}>
          <StickyNote />
        </button>
      </Tip>
      <Tip label={`${t('canvas.group')} (G)`} side="right">
        <button className="icon-btn" onClick={() => (selection.length ? tools.frameSelection() : void actions.addNode('group', { position: rf.screenToFlowPosition({ x: window.innerWidth / 2 - 260, y: window.innerHeight / 2 - 170 }) }))}>
          <Frame />
        </button>
      </Tip>
      <span className="canvas-tools-sep" />
      <Tip label={`${t('canvas.arrange')} (L)`} side="right">
        <button className="icon-btn" onClick={tools.arrange}>
          <LayoutDashboard />
        </button>
      </Tip>
      <Tip label={`${t('canvas.fit')} (F)`} side="right">
        <button className="icon-btn" onClick={() => rf.fitView(FIT)}>
          <Maximize />
        </button>
      </Tip>
      <Tip label={t('canvas.minimap')} side="right">
        <button className={clsx('icon-btn', minimap && 'on')} onClick={toggleMinimap}>
          <MapIcon />
        </button>
      </Tip>
      <Tip label={`${t('canvas.inspector')} (I)`} side="right">
        <button className={clsx('icon-btn', inspector && 'on')} onClick={toggleInspector} aria-pressed={inspector} aria-label={t('canvas.inspector')}>
          <PanelRight />
        </button>
      </Tip>
    </div>
  )
}

/** An empty canvas: import a song or other audio. */
export function CanvasEmpty({ importFiles }: { importFiles: CanvasEdit['importFiles'] }) {
  const t = useT()
  return (
    <div className="canvas-empty">
      <div className="canvas-empty-orb">
        <Music />
      </div>
      <h2>{t('canvas.empty.title')}</h2>
      <div className="row" style={{ gap: 8 }}>
        <button className="btn btn-primary" onClick={() => void importFiles('song')}>
          <Upload />
          {t('canvas.import')}
        </button>
        <button className="btn" onClick={() => void importFiles('audio')}>
          <FileAudio />
          {t('canvas.import_audio')}
        </button>
      </div>
      <p className="muted canvas-empty-hint">{t('canvas.empty.hint')}</p>
    </div>
  )
}

/** The voices that can sing: click to sing the selection, or drag onto a node. */
export function VoiceDock({
  voices,
  highlight,
  selection,
  findNode,
  sing,
}: {
  voices: Character[]
  /** a voice to point out (opened from its page) */
  highlight: string | null
  selection: string[]
  findNode: CanvasEdit['findNode']
  sing: CanvasActions['sing']
}) {
  const t = useT()
  const nav = useNavigate()
  return (
    <div className="voice-dock glass-strong">
      <span className="voice-dock-label">{t('canvas.voices')}</span>
      {voices.length === 0 ? (
        <button className="chip button" onClick={() => nav('/voices')}>
          {t('create.go_voices')}
        </button>
      ) : (
        voices.map((c) => (
          <Tip key={c.id} label={`${c.name} · ${t('canvas.voices_hint')}`}>
            <button
              className={clsx('voice', highlight === c.id && 'highlight')}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData('application/x-stellaria-character', c.id)
                e.dataTransfer.effectAllowed = 'copy'
              }}
              onClick={() => {
                const targets = selection
                  .map(findNode)
                  .filter((n): n is CanvasNode => !!n && isContent(n) && !isMix(n))
                  .map(singablePort)
                  .filter((r): r is PortRef => !!r)
                const unique = targets.filter((r, i) => targets.findIndex((x) => x.nodeId === r.nodeId && x.port === r.port) === i)
                if (unique.length) unique.forEach((r) => void sing(r, c.id, { focus: unique.length === 1 }))
                else toast({ kind: 'info', title: t('canvas.drop_on_song') })
              }}
            >
              <Avatar character={c} size={36} />
            </button>
          </Tip>
        ))
      )}
    </div>
  )
}

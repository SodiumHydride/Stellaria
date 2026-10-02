import { Background, BackgroundVariant, MiniMap, ReactFlow, ReactFlowProvider, useNodesInitialized, type Viewport } from '@xyflow/react'
import clsx from 'clsx'
import { Loader2 } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useCharacters, useMedia, useMixProcessors, useModels } from '../../api/hooks'
import type { CanvasNode } from '../../api/types'
import { useT } from '../../i18n'
import { cssVar } from '../../lib/color'
import { useTheme } from '../../state/theme'
import { toast } from '../../state/ui'
import { Splitter } from '../../components/Splitter'
import { BatchDialog } from './BatchDialog'
import { CanvasEmpty, CanvasToolRail, CanvasTopBar, useInspectorPref, useMinimapPref, useStoredSize, VoiceDock } from './CanvasChrome'
import { CanvasContext } from './context'
import { Inspector } from './Inspector'
import { AddMenu, ContextMenu } from './menus'
import { isContent, isMix, isSource, separateOverlaps } from './model'
import { nodeTypes } from './nodes'
import { NowPlaying } from './NowPlaying'
import { RunButton } from './RunButton'
import { TemplateDialog } from './TemplateDialog'
import { PitchEditorHost } from './pitch/PitchEditor'
import { Timeline } from './timeline/Timeline'
import { useCanvasActions } from './useCanvasActions'
import { useCanvasKeys, usePointer } from './useCanvasKeys'
import { useAddMenu, useCanvasMenus } from './useCanvasMenus'
import { useCanvasTools } from './useCanvasTools'
import { useFlowBridge } from './useFlowBridge'
import { useProject } from './useProject'
import { singablePort, voiceModelFor } from './voices'
import './canvas.css'

export function CanvasWorkspace({ projectId }: { projectId: string }) {
  return (
    <ReactFlowProvider>
      <Workspace projectId={projectId} />
    </ReactFlowProvider>
  )
}

/** default side panel width and timeline height (px); both can be dragged and are remembered */
const INSPECTOR_W = typeof window !== 'undefined' && window.innerWidth <= 1280 ? 300 : 330
const TIMELINE_H = 210

function Workspace({ projectId }: { projectId: string }) {
  const t = useT()
  const [params] = useSearchParams()
  const theme = useTheme((s) => s.resolved)
  const { project, getProject, update, flush, saveState, undo, redo, canUndo, canRedo } = useProject(projectId)
  const { data: mediaList } = useMedia(projectId)
  const { data: characterList } = useCharacters()
  const { data: modelList } = useModels()
  const { data: processors } = useMixProcessors()
  const [selection, setSelection] = useState<string[]>([])
  const [timelineChoice, setTimelineChoice] = useState<boolean | null>(null)
  const [batchFor, setBatchFor] = useState<string[] | null>(null)
  const [templateFor, setTemplateFor] = useState<string | null>(null)
  const [renaming, setRenaming] = useState<string | null>(null)
  const [minimap, toggleMinimap] = useMinimapPref()
  const [inspector, toggleInspector] = useInspectorPref()
  const [inspectorWidth, setInspectorWidth, resetInspectorWidth] = useStoredSize('stellaria.canvas.inspectorWidth', INSPECTOR_W)
  const [timelineHeight, setTimelineHeight, resetTimelineHeight] = useStoredSize('stellaria.canvas.timelineHeight', TIMELINE_H)
  const { addMenu, setAddMenu, openAddAt } = useAddMenu()
  const mouse = usePointer()

  const media = useMemo(() => new Map((mediaList ?? []).map((m) => [m.id, m])), [mediaList])
  const characters = useMemo(() => new Map((characterList ?? []).map((c) => [c.id, c])), [characterList])
  const models = useMemo(() => new Map((modelList ?? []).map((m) => [m.id, m])), [modelList])
  const voices = useMemo(() => (characterList ?? []).filter((c) => voiceModelFor(c, modelList ?? [])), [characterList, modelList])

  const timelineOpen = timelineChoice ?? (project?.timeline.tracks.some((tr) => tr.clips.length > 0) ?? false)
  const showTimeline = useCallback(() => setTimelineChoice(true), [])

  const edit = useCanvasActions({
    projectId,
    project,
    getProject,
    update,
    flush,
    mediaList,
    media,
    characters,
    modelList,
    selection,
    setSelection,
    setRenaming,
    setBatchFor,
    showTimeline,
    mouse,
  })
  const { actions, suggestions, findNode, setNodes, reveal } = edit
  const tools = useCanvasTools(projectId, edit, selection, setSelection)
  const flow = useFlowBridge({ project, characters, selection, setSelection, edit, removeNodes: actions.removeNodes, openAddMenu: setAddMenu })
  const menus = useCanvasMenus({
    projectId,
    projectName: project?.name ?? '',
    characters,
    selection,
    setSelection,
    actions,
    edit,
    tools,
    flowEdges: flow.flowEdges,
    openAddAt,
    openTemplate: setTemplateFor,
  })
  useCanvasKeys({ selection, actions, edit, tools, undo, redo, openAddAt, mouse, toggleInspector })

  // once per opened project, when every card has its size: cards that grew into the one below push it down
  const nodesReady = useNodesInitialized()
  const settled = useRef<string | null>(null)
  useEffect(() => {
    if (!nodesReady || !project || settled.current === projectId) return
    settled.current = projectId
    const moved = separateOverlaps(project.canvas.nodes, edit.measure)
    if (moved.size) setNodes((ns) => ns.map((n) => (moved.has(n.id) ? { ...n, position: moved.get(n.id)! } : n)), { history: false })
  }, [nodesReady, project, projectId, edit.measure, setNodes])

  // drag a voice from the dock onto a node: that voice sings the node's audio
  const onDrop = (e: React.DragEvent) => {
    const characterId = e.dataTransfer.getData('application/x-stellaria-character')
    if (!characterId) return
    e.preventDefault()
    const el = document.elementFromPoint(e.clientX, e.clientY)?.closest<HTMLElement>('.react-flow__node')
    const target = el?.dataset.id ? findNode(el.dataset.id) : undefined
    const ref = target && isContent(target) && !isMix(target) ? singablePort(target) : null
    if (ref) void actions.sing(ref, characterId)
    else {
      const sel = selection.map(findNode).find((n): n is CanvasNode => !!n && isContent(n) && !isMix(n))
      const selRef = sel ? singablePort(sel) : null
      if (selRef) void actions.sing(selRef, characterId)
      else toast({ kind: 'info', title: t('canvas.drop_on_song') })
    }
  }

  if (!project) {
    return (
      <div className="canvas-loading">
        <Loader2 className="spin" />
      </div>
    )
  }

  const empty = project.canvas.nodes.length === 0
  const nodeById = (id: string) => project.canvas.nodes.find((n) => n.id === id)
  const batchSources = (batchFor ?? []).map(nodeById).filter(isSource)
  const templateSource = templateFor ? nodeById(templateFor) : undefined

  return (
    <CanvasContext.Provider value={{ project, media, characters, models, processors: processors ?? [], voices, actions, renaming }}>
      <div
        className={clsx('canvas-ws', timelineOpen && 'with-timeline', !inspector && 'no-inspector')}
        style={{ ['--insp-w' as string]: `${inspectorWidth}px`, ['--tl-h' as string]: `${timelineHeight}px` }}
      >
        <div
          className="canvas-area"
          onDragOver={(e) => e.dataTransfer.types.includes('application/x-stellaria-character') && e.preventDefault()}
          onDrop={onDrop}
          onDoubleClick={(e) => {
            if ((e.target as HTMLElement).classList.contains('react-flow__pane')) openAddAt(e.clientX, e.clientY)
          }}
        >
          <ReactFlow
            nodes={flow.rfNodes}
            edges={flow.rfEdges}
            nodeTypes={nodeTypes}
            onNodesChange={flow.onNodesChange}
            onEdgesChange={flow.onEdgesChange}
            onConnect={flow.onConnect}
            onConnectEnd={flow.onConnectEnd}
            onNodeDragStop={flow.onNodeDragStop}
            onNodeContextMenu={menus.nodeMenu}
            onEdgeContextMenu={menus.edgeMenu}
            onPaneContextMenu={menus.paneMenu}
            isValidConnection={flow.isValidConnection}
            defaultViewport={project.canvas.viewport}
            onMoveEnd={(_, vp: Viewport) => update((p) => ({ ...p, canvas: { ...p.canvas, viewport: vp } }), { history: false })}
            onPaneClick={() => {
              setAddMenu(null)
              menus.closeCtxMenu()
            }}
            minZoom={0.15}
            maxZoom={2.5}
            zoomOnDoubleClick={false}
            deleteKeyCode={['Delete', 'Backspace']}
            multiSelectionKeyCode={['Meta', 'Control']}
            selectionKeyCode="Shift"
            panActivationKeyCode={null}
            connectionRadius={28}
            snapToGrid
            snapGrid={[8, 8]}
            zoomOnPinch
            proOptions={{ hideAttribution: true }}
            fitView={empty ? false : project.canvas.viewport.zoom === 1 && project.canvas.viewport.x === 0}
            fitViewOptions={{ padding: 0.2, maxZoom: 1 }}
            colorMode={theme}
            connectionLineStyle={{ stroke: cssVar('--text-3'), strokeWidth: 1.8 }}
          >
            <Background variant={BackgroundVariant.Dots} gap={24} size={1} />
            {minimap && <MiniMap pannable zoomable className="canvas-minimap" nodeColor={(n) => (n.type === 'group' ? 'transparent' : cssVar('--well-strong'))} nodeStrokeColor={() => cssVar('--text-4')} maskColor={cssVar('--overlay')} />}
          </ReactFlow>

          <NowPlaying />

          <CanvasTopBar project={project} update={update} saveState={saveState} undo={undo} redo={redo} canUndo={canUndo} canRedo={canRedo} />

          <RunButton projectId={projectId} nodes={project.canvas.nodes} selection={selection} onRun={actions.run} />

          <CanvasToolRail selection={selection} actions={actions} edit={edit} tools={tools} openAddMenu={setAddMenu} minimap={minimap} toggleMinimap={toggleMinimap} inspector={inspector} toggleInspector={toggleInspector} />

          {empty && <CanvasEmpty importFiles={edit.importFiles} />}

          <VoiceDock voices={voices} highlight={params.get('with')} selection={selection} findNode={findNode} sing={actions.sing} />

          {addMenu && <AddMenu state={addMenu} onClose={() => setAddMenu(null)} />}
          {menus.ctxMenu && <ContextMenu {...menus.ctxMenu} onClose={menus.closeCtxMenu} />}
        </div>

        {inspector && (
          <div className="insp-col">
            <Splitter axis="x" grow={-1} value={inspectorWidth} min={260} max={560} onChange={setInspectorWidth} onReset={resetInspectorWidth} label={t('canvas.resize_panel')} className="insp-splitter" />
            <Inspector selection={selection} suggestions={suggestions} onSelect={setSelection} />
          </div>
        )}

        <div className="tl-col">
          {timelineOpen && <Splitter axis="y" grow={-1} value={timelineHeight} min={120} max={Math.max(160, window.innerHeight - 440)} onChange={setTimelineHeight} onReset={resetTimelineHeight} label={t('canvas.resize_timeline')} className="tl-splitter" />}
          <Timeline open={timelineOpen} onToggle={() => setTimelineChoice(!timelineOpen)} update={update} />
        </div>
      </div>
      {batchSources.length > 0 && (
        <BatchDialog
          sources={batchSources}
          voices={voices}
          onClose={() => setBatchFor(null)}
          onCreated={(ids) => {
            setBatchFor(null)
            setSelection(ids)
            reveal([...batchSources.map((s) => s.id), ...ids])
          }}
        />
      )}
      {isSource(templateSource) && (
        <TemplateDialog
          source={templateSource}
          onClose={() => setTemplateFor(null)}
          onApply={(nodes, select) => {
            setTemplateFor(null)
            setNodes((ns) => [...ns, ...nodes])
            setSelection(select)
            reveal([templateSource.id, ...nodes.map((n) => n.id)])
          }}
        />
      )}
      <PitchEditorHost />
    </CanvasContext.Provider>
  )
}

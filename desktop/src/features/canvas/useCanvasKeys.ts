import { useReactFlow } from '@xyflow/react'
import { useEffect, useRef, type RefObject } from 'react'
import type { CanvasNode } from '../../api/types'
import { useEvent } from '../../lib/useEvent'
import { usePlayer } from '../../state/player'
import type { CanvasActions } from './context'
import { isContent, isProcess, isTake, outputPorts } from './model'
import { FIT, type CanvasEdit, type Point } from './useCanvasActions'
import type { useAddMenu } from './useCanvasMenus'
import type { CanvasTools } from './useCanvasTools'

/** Where the pointer last was on screen: Tab, paste and new notes go there. */
export function usePointer() {
  const mouse = useRef<Point>({ x: window.innerWidth / 2, y: window.innerHeight / 2 })
  useEffect(() => {
    const onMove = (e: MouseEvent) => (mouse.current = { x: e.clientX, y: e.clientY })
    window.addEventListener('mousemove', onMove)
    return () => window.removeEventListener('mousemove', onMove)
  }, [])
  return mouse
}

export interface CanvasKeyArgs {
  selection: string[]
  actions: CanvasActions
  edit: CanvasEdit
  tools: CanvasTools
  undo: () => void
  redo: () => void
  openAddAt: ReturnType<typeof useAddMenu>['openAddAt']
  mouse: RefObject<Point>
  toggleInspector: () => void
}

/** The canvas's keyboard shortcuts (the 创作画布 section of the shortcuts dialog). */
export function useCanvasKeys({ selection, actions, edit, tools, undo, redo, openAddAt, mouse, toggleInspector }: CanvasKeyArgs) {
  const rf = useReactFlow()

  const onKey = useEvent((e: KeyboardEvent) => {
    const target = e.target as HTMLElement
    if (target.closest('input, textarea, select, [contenteditable="true"], [role="dialog"], [role="menu"]')) return
    const mod0 = e.metaKey || e.ctrlKey
    // the arrangement and the side panel have their own keys; only undo / redo stay global
    if (target.closest('.timeline, .inspector') && !(mod0 && /^[zy]$/i.test(e.key))) return
    // a focused control keeps Tab (move focus) and Space (press it); the canvas gets them otherwise
    const control = !!target.closest('button, a[href], [role="slider"], [role="button"], [role="switch"], [role="tab"], [tabindex]:not(.react-flow__node)')
    const onCanvas = target === document.body || (!!target.closest('.react-flow') && !control)
    const { findNode, nodesNow, flowAtMouse } = edit
    const mod = e.metaKey || e.ctrlKey
    const key = e.key.toLowerCase()
    const selected = selection.map(findNode).filter((n): n is CanvasNode => !!n)
    if (mod && key === 'z') {
      e.preventDefault()
      if (e.shiftKey) redo()
      else undo()
    } else if (mod && key === 'y') {
      e.preventDefault()
      redo()
    } else if (mod && key === 'c' && selected.length) {
      actions.copy(selected.map((n) => n.id))
    } else if (mod && key === 'v') {
      e.preventDefault()
      actions.paste()
    } else if (mod && key === 'd' && selected.length) {
      e.preventDefault()
      if (selected.length === 1 && isTake(selected[0])) actions.variant(selected[0].id)
      else actions.duplicate(selected.map((n) => n.id))
    } else if (mod && key === 'a') {
      e.preventDefault()
      actions.select(nodesNow().map((n) => n.id))
    } else if (e.key === 'Tab' && !e.shiftKey && onCanvas) {
      e.preventDefault()
      const one = selected.length === 1 ? selected[0] : undefined
      const from = one && outputPorts(one).length ? { nodeId: one.id, handle: outputPorts(one)[0] } : undefined
      openAddAt(mouse.current.x, mouse.current.y, from ? { from } : undefined)
    } else if (e.key === ' ' && selected[0] && !control) {
      e.preventDefault()
      tools.playNode(selected[0])
    } else if (mod && e.key === 'Enter') {
      e.preventDefault()
      void actions.run(e.shiftKey || selected.length === 0 ? null : selected.map((n) => n.id))
    } else if (!mod && key === 'm') {
      const voiced = selected.filter((n) => isTake(n) || (isProcess(n) && outputPorts(n).length > 0))
      if (voiced.length) actions.createMix(voiced.map((n) => ({ nodeId: n.id, port: outputPorts(n)[0] })))
    } else if (!mod && key === 'b') {
      tools.toggleBypass(selection)
    } else if (!mod && key === 'g' && selected.length) {
      tools.frameSelection()
    } else if (!mod && key === 'x' && !e.defaultPrevented) {
      usePlayer.getState().flip()
    } else if (!mod && /^[1-9]$/.test(e.key) && usePlayer.getState().group) {
      usePlayer.getState().switchTo(Number(e.key) - 1)
    } else if (!mod && key === 'f') {
      rf.fitView(FIT)
    } else if (!mod && key === 'n') {
      void actions.addNode('note', { position: flowAtMouse() })
    } else if (!mod && key === 'l') {
      tools.arrange()
    } else if (!mod && key === 'i') {
      toggleInspector()
    } else if (e.key === 'F2' && selected.length === 1 && isContent(selected[0])) {
      e.preventDefault()
      actions.rename(selected[0].id)
    }
  })

  useEffect(() => {
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onKey])
}

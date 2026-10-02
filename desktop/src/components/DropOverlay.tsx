import clsx from 'clsx'
import { FileAudio, FolderPlus, Sparkles, UserPlus } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { api, sendFiles } from '../api/client'
import { useCharacter, useCharacters } from '../api/hooks'
import type { Character, Job, Project } from '../api/types'
import { useT } from '../i18n'
import { attempt } from '../lib/actions'
import { DROP_FILES_EVENT, dropZoneAt } from '../lib/dropZones'
import { useEvent } from '../lib/useEvent'
import { pickVoiceColor } from '../lib/color'
import { isTauri, onNativeDrop } from '../shell/bridge'
import { importItems } from '../features/characters/importMaterial'
import { expectImport } from '../features/canvas/imports'
import { Avatar } from './Avatar'
import './drop-overlay.css'

type Target = 'character' | 'new-character' | 'project' | 'project-audio' | 'new-project'

function targetAt(x: number, y: number): Target | null {
  const el = document.elementFromPoint(x, y)?.closest<HTMLElement>('[data-drop-target]')
  return (el?.dataset.dropTarget as Target) ?? null
}

/**
 * Drop audio anywhere: the overlay offers the targets that make sense for the current screen
 * (this character's material, a new character, this canvas, a new project).
 */
export function DropOverlay() {
  const t = useT()
  const nav = useNavigate()
  const { pathname } = useLocation()
  const [active, setActive] = useState(false)
  const [hover, setHover] = useState<Target | null>(null)
  const depth = useRef(0)
  const characterId = pathname.match(/^\/voices\/([^/]+)/)?.[1]
  const projectId = pathname.match(/^\/create\/([^/]+)/)?.[1]
  const onCreate = pathname.startsWith('/create')
  const { data: character } = useCharacter(characterId)
  const { data: characters } = useCharacters()

  const targets: Target[] = onCreate ? (projectId ? ['project', 'project-audio', 'new-project'] : ['new-project']) : characterId ? ['character', 'new-character'] : ['new-character']

  const deliver = async (target: Target, items: string[] | File[]) => {
    if (items.length === 0) return
    if (target === 'character' && characterId) {
      await importItems(characterId, items)
      nav(`/voices/${characterId}/material`)
    } else if (target === 'new-character') {
      const used = (characters ?? []).map((c) => c.color)
      const c = await attempt(() => api.post<Character>('/characters', { name: t.lang === 'zh-CN' ? '新角色' : 'New voice', color: pickVoiceColor(String(Date.now()), used) }))
      if (c) {
        await importItems(c.id, items)
        nav(`/voices/${c.id}/material?rename=1`)
      }
    } else if ((target === 'project' || target === 'project-audio') && projectId) {
      const job = await attempt(() => sendFiles<Job>(`/projects/${projectId}/media`, items))
      if (job) expectImport(job.id, { kind: target === 'project' ? 'song' : 'audio' })
    } else if (target === 'new-project') {
      const p = await attempt(() => api.post<Project>('/projects', {}))
      if (p) {
        await attempt(() => sendFiles<Job>(`/projects/${p.id}/media`, items))
        nav(`/create/${p.id}`)
      }
    }
  }

  /** files let go at (x, y): to the target under the pointer, else this screen's first one */
  const drop = useEvent((x: number, y: number, items: string[] | File[]) => {
    setActive(false)
    setHover(null)
    // a part of the window that takes files itself (the arrangement) gets them there
    const zone = dropZoneAt(x, y)
    if (zone) {
      zone.dispatchEvent(new CustomEvent(DROP_FILES_EVENT, { detail: { x, y, items } }))
      return
    }
    void deliver(targetAt(x, y) ?? targets[0], items)
  })

  // native (Tauri) drops carry absolute paths
  useEffect(() => {
    if (!isTauri) return
    let off: (() => void) | undefined
    let gone = false
    void onNativeDrop((e) => {
      if (e.kind === 'enter') setActive(!dropZoneAt(e.x, e.y))
      else if (e.kind === 'over') {
        // over a drop zone the overlay steps aside so the zone can show where the files land
        const zone = !!dropZoneAt(e.x, e.y)
        setActive(!zone)
        setHover(zone ? null : targetAt(e.x, e.y))
      } else if (e.kind === 'leave') {
        setActive(false)
        setHover(null)
      } else drop(e.x, e.y, e.paths)
    }).then((f) => (gone ? f() : (off = f)))
    return () => {
      gone = true
      off?.()
    }
  }, [drop])

  // browser drops carry File objects
  useEffect(() => {
    if (isTauri) return
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes('Files')
    const onEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return
      depth.current += 1
      setActive(!dropZoneAt(e.clientX, e.clientY))
    }
    const onLeave = (e: DragEvent) => {
      if (!hasFiles(e)) return
      depth.current = Math.max(0, depth.current - 1)
      if (depth.current === 0) setActive(false)
    }
    const onOver = (e: DragEvent) => {
      if (!hasFiles(e)) return
      if (dropZoneAt(e.clientX, e.clientY)) {
        // the zone handles its own dragover and drop; the overlay steps aside
        setActive(false)
        setHover(null)
        return
      }
      e.preventDefault()
      setActive(true)
      setHover(targetAt(e.clientX, e.clientY))
    }
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return
      depth.current = 0
      if (dropZoneAt(e.clientX, e.clientY)) {
        // the zone took it (or will): nothing to import here
        setActive(false)
        setHover(null)
        return
      }
      e.preventDefault()
      drop(e.clientX, e.clientY, Array.from(e.dataTransfer?.files ?? []))
    }
    // capture phase: a drop zone stops its own events from bubbling, and the overlay still has
    // to know the drag moved there or ended there
    window.addEventListener('dragenter', onEnter, true)
    window.addEventListener('dragleave', onLeave, true)
    window.addEventListener('dragover', onOver, true)
    window.addEventListener('drop', onDrop, true)
    return () => {
      window.removeEventListener('dragenter', onEnter, true)
      window.removeEventListener('dragleave', onLeave, true)
      window.removeEventListener('dragover', onOver, true)
      window.removeEventListener('drop', onDrop, true)
    }
  }, [drop])

  if (!active) return null
  return (
    <div className="drop-overlay">
      <div className="drop-title">{t('drop.title')}</div>
      <div className="drop-targets">
        {targets.map((tg) => (
          <div key={tg} data-drop-target={tg} className={clsx('drop-target glass', hover === tg && 'hover', tg.startsWith('new') && 'secondary-target')}>
            <span className="drop-target-icon">
              {tg === 'character' && character ? (
                <Avatar character={character} size={56} />
              ) : tg === 'new-character' ? (
                <UserPlus />
              ) : tg === 'project' ? (
                <Sparkles />
              ) : tg === 'project-audio' ? (
                <FileAudio />
              ) : (
                <FolderPlus />
              )}
            </span>
            <span className="drop-target-label">
              {tg === 'character'
                ? t('drop.to_character', { name: character?.name ?? '' })
                : tg === 'new-character'
                  ? t('drop.new_character')
                  : tg === 'project'
                    ? t('drop.to_project')
                    : tg === 'project-audio'
                      ? t('drop.to_project_audio')
                      : t('drop.new_project')}
            </span>
          </div>
        ))}
      </div>
      <div className="drop-hint">{t('drop.hint')}</div>
    </div>
  )
}

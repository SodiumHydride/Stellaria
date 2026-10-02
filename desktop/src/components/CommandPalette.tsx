import { Command } from 'cmdk'
import { Download, Keyboard, Pause, Plus, Settings, Sparkles, Telescope, Upload } from 'lucide-react'
import { useEffect } from 'react'
import { createPortal } from 'react-dom'
import { useLocation, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { useActiveJobs, useCharacters, useModels, useProjects } from '../api/hooks'
import type { Character, Project } from '../api/types'
import { useT, type StringKey } from '../i18n'
import { attempt } from '../lib/actions'
import { coverColor, pickVoiceColor } from '../lib/color'
import { platform } from '../shell/bridge'
import { useUi } from '../state/ui'
import { importMaterial } from '../features/characters/importMaterial'
import { Avatar } from './Avatar'
import { Dialog } from './ui'
import './palette.css'

export function CommandPalette() {
  const t = useT()
  const open = useUi((s) => s.paletteOpen)
  const setOpen = useUi((s) => s.setPalette)
  const setShortcuts = useUi((s) => s.setShortcuts)
  const nav = useNavigate()
  const { pathname } = useLocation()
  const { data: characters } = useCharacters()
  const { data: projects } = useProjects()
  const { data: models } = useModels()
  const trainJobs = useActiveJobs((j) => j.type === 'train' && j.status === 'running')
  const currentCharacterId = pathname.match(/^\/voices\/([^/]+)/)?.[1]

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setOpen(!useUi.getState().paletteOpen)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [setOpen])

  const run = (fn: () => void) => {
    setOpen(false)
    fn()
  }

  const newCharacter = async () => {
    const used = (characters ?? []).map((c) => c.color)
    const c = await attempt(() => api.post<Character>('/characters', { name: t.lang === 'zh-CN' ? '新角色' : 'New voice', color: pickVoiceColor(String(Date.now()), used) }))
    if (c) nav(`/voices/${c.id}/material?rename=1`)
  }

  const newProject = async () => {
    const p = await attempt(() => api.post<Project>('/projects', {}))
    if (p) nav(`/create/${p.id}`)
  }

  if (!open) return null
  return createPortal(
    <div className="palette-overlay" onMouseDown={(e) => e.target === e.currentTarget && setOpen(false)}>
      <Command className="palette glass-strong" label="Command palette" loop>
        <Command.Input autoFocus placeholder={t('palette.placeholder')} className="palette-input" onKeyDown={(e) => e.key === 'Escape' && setOpen(false)} />
        <Command.List className="palette-list">
          <Command.Empty className="palette-empty">{t('palette.empty')}</Command.Empty>

          <Command.Group heading={t('palette.group.actions')}>
            <Command.Item onSelect={() => run(newCharacter)}>
              <Plus />
              {t('palette.new_character')}
            </Command.Item>
            {currentCharacterId && (
              <Command.Item onSelect={() => run(() => importMaterial(currentCharacterId))}>
                <Upload />
                {t('palette.import_material')}
              </Command.Item>
            )}
            <Command.Item onSelect={() => run(newProject)}>
              <Sparkles />
              {t('palette.new_project')}
            </Command.Item>
            {trainJobs.map((j) => {
              const m = models?.find((x) => x.id === j.target.modelId)
              return (
                <Command.Item key={j.id} onSelect={() => run(() => attempt(() => api.post(`/models/${j.target.modelId}/pause`)))}>
                  <Pause />
                  {t('palette.pause_training', { name: m?.name ?? '' })}
                </Command.Item>
              )
            })}
            <Command.Item onSelect={() => run(() => setShortcuts(true))}>
              <Keyboard />
              {t('palette.shortcuts')}
            </Command.Item>
          </Command.Group>

          {characters && characters.length > 0 && (
            <Command.Group heading={t('palette.group.characters')}>
              {characters.map((c) => (
                <Command.Item key={c.id} value={`character ${c.name} ${c.tags.join(' ')}`} onSelect={() => run(() => nav(`/voices/${c.id}`))}>
                  <Avatar character={c} size={20} />
                  {c.name}
                  {c.tags.length > 0 && <span className="palette-meta">{c.tags.join(' · ')}</span>}
                </Command.Item>
              ))}
            </Command.Group>
          )}

          {projects && projects.length > 0 && (
            <Command.Group heading={t('palette.group.projects')}>
              {projects.map((p) => (
                <Command.Item key={p.id} value={`project ${p.name}`} onSelect={() => run(() => nav(`/create/${p.id}`))}>
                  <span className="palette-dot" style={{ background: coverColor(p) }} />
                  {p.name}
                </Command.Item>
              ))}
            </Command.Group>
          )}

          <Command.Group heading={t('palette.group.go')}>
            <Command.Item onSelect={() => run(() => nav('/voices'))}>
              <Telescope />
              {t('space.voices')}
            </Command.Item>
            <Command.Item onSelect={() => run(() => nav('/create'))}>
              <Sparkles />
              {t('space.create')}
            </Command.Item>
            <Command.Item onSelect={() => run(() => nav('/resources'))}>
              <Download />
              {t('space.resources')}
            </Command.Item>
            <Command.Item onSelect={() => run(() => nav('/settings'))}>
              <Settings />
              {t('palette.open_settings')}
            </Command.Item>
          </Command.Group>
        </Command.List>
      </Command>
    </div>,
    document.body,
  )
}

type Scope = 'global' | 'canvas' | 'timeline' | 'material'
type Shortcut = [keys: string[], label: StringKey]

/** The page's own shortcuts: the canvas of a project, a character's material tab. */
function pageScope(pathname: string): Scope | null {
  if (/^\/create\/[^/]+/.test(pathname)) return 'canvas'
  const voice = pathname.match(/^\/voices\/[^/]+(?:\/([^/]+))?/)
  if (voice && (!voice[1] || voice[1] === 'material')) return 'material'
  return null
}

/** Shortcuts grouped by where they work; opened from a page, that page's group comes first. */
export function ShortcutsDialog() {
  const t = useT()
  const open = useUi((s) => s.shortcutsOpen)
  const setOpen = useUi((s) => s.setShortcuts)
  const { pathname } = useLocation()
  const mod = platform === 'macos' ? '⌘' : 'Ctrl'
  const sections: Record<Scope, { title: StringKey; rows: Shortcut[] }> = {
    global: {
      title: 'shortcuts.scope.global',
      rows: [
        [[mod, 'K'], 'shortcuts.palette'],
        [[mod, '1 / 2 / 3'], 'shortcuts.spaces'],
        [[mod, ','], 'palette.open_settings'],
      ],
    },
    canvas: {
      title: 'shortcuts.scope.canvas',
      rows: [
        [['Space'], 'shortcuts.play'],
        [['Tab'], 'shortcuts.add'],
        [[mod, 'Enter'], 'shortcuts.render'],
        [[mod, '⇧', 'Enter'], 'shortcuts.run_all'],
        [[mod, 'Z'], 'shortcuts.undo'],
        [[mod, '⇧', 'Z'], 'shortcuts.redo'],
        [[mod, 'C / V'], 'shortcuts.copy_paste'],
        [[mod, 'D'], 'shortcuts.variant'],
        [['M'], 'shortcuts.mix'],
        [['B'], 'shortcuts.bypass'],
        [['G'], 'shortcuts.frame'],
        [['F2'], 'shortcuts.rename'],
        [[platform === 'macos' ? '⌥' : 'Alt', t('shortcuts.click')], 'shortcuts.port_play'],
        [['X'], 'shortcuts.compare'],
        [['1 – 9'], 'shortcuts.compare_n'],
        [['L'], 'shortcuts.arrange'],
        [['F'], 'shortcuts.fit'],
        [['I'], 'shortcuts.inspector'],
        [['Delete'], 'shortcuts.delete'],
      ],
    },
    timeline: {
      title: 'shortcuts.scope.timeline',
      rows: [
        [['Space'], 'shortcuts.play'],
        [['S', `${mod} K`], 'shortcuts.tl.split'],
        [[mod, 'D'], 'shortcuts.tl.duplicate'],
        [[mod, 'X / C / V'], 'shortcuts.tl.clipboard'],
        [[mod, 'A'], 'shortcuts.tl.select_all'],
        [['←', '→'], 'shortcuts.tl.nudge'],
        [['Home', 'End'], 'shortcuts.tl.ends'],
        [['L'], 'shortcuts.tl.loop'],
        [['N'], 'shortcuts.tl.snap'],
        [['M'], 'shortcuts.tl.mixer'],
        [['+', '−'], 'shortcuts.tl.zoom'],
        [[platform === 'macos' ? '⌥' : 'Alt', t('shortcuts.drag')], 'shortcuts.tl.alt_drag'],
        [['⇧', t('shortcuts.drag')], 'shortcuts.tl.shift_drag'],
        [['Delete'], 'shortcuts.delete'],
      ],
    },
    material: {
      title: 'shortcuts.scope.material',
      rows: [
        [['Space'], 'shortcuts.play'],
        [['←', '→'], 'shortcuts.review'],
        [['D'], 'shortcuts.trash'],
        [['V'], 'shortcuts.val'],
        [['S'], 'shortcuts.split'],
        [['Delete'], 'shortcuts.delete'],
      ],
    },
  }
  const here = pageScope(pathname)
  const all: Scope[] = ['global', 'canvas', 'timeline', 'material']
  // on a canvas, its arrangement comes right after it
  const first: Scope[] = here === 'canvas' ? ['canvas', 'timeline'] : here ? [here] : []
  const order: Scope[] = [...first, ...all.filter((s) => !first.includes(s))]
  return (
    <Dialog open={open} onOpenChange={setOpen} title={t('shortcuts.title')}>
      {order.map((scope, i) => (
        <section key={scope} aria-label={t(sections[scope].title)} style={i > 0 ? { marginTop: 18 } : undefined}>
          <h3 className="eyebrow" style={{ margin: '0 0 6px' }}>
            {t(sections[scope].title)}
          </h3>
          <div className="shortcut-grid">
            {sections[scope].rows.map(([k, label]) => (
              <div key={label} className="shortcut-row">
                <span className="secondary">{t(label)}</span>
                <span className="row" style={{ gap: 4 }}>
                  {k.map((x) => (
                    <span key={x} className="kbd">
                      {x}
                    </span>
                  ))}
                </span>
              </div>
            ))}
          </div>
        </section>
      ))}
    </Dialog>
  )
}

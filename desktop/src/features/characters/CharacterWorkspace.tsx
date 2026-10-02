import * as RPopover from '@radix-ui/react-popover'
import { useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import {
  Archive,
  ArrowRight,
  Brain,
  Camera,
  Check,
  Loader2,
  MoreHorizontal,
  Package,
  PenLine,
  Pin,
  PinOff,
  Plus,
  ShieldCheck,
  ShieldQuestion,
  Sparkles,
  Trash2,
  Upload,
  X,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../../api/client'
import { useActiveJobs, useCharacter, useModels } from '../../api/hooks'
import { keys } from '../../api/keys'
import type { Character, Job, Project } from '../../api/types'
import { Avatar } from '../../components/Avatar'
import { confirmDialog, Dialog, Menu, Switch, Tip } from '../../components/ui'
import { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { VOICE_COLORS } from '../../lib/color'
import { duration, percent, steps } from '../../lib/format'
import { deliverExport } from '../../lib/exportJob'
import { isTauri, pickSavePath } from '../../shell/bridge'
import { importMaterial } from './importMaterial'
import { MaterialView } from './material/MaterialView'
import { ModelsView } from './models/ModelsView'
import { TrainingView } from './training/TrainingView'
import './workspace.css'

type Tab = 'material' | 'training' | 'models'

export function patchCharacter(qc: ReturnType<typeof useQueryClient>, id: string, patch: Partial<Character>) {
  return attempt(async () => {
    const c = await api.patch<Character>(`/characters/${id}`, patch)
    qc.setQueryData(keys.character(id), c)
    return c
  })
}

export function CharacterWorkspace({ characterId, tab: tabParam }: { characterId: string; tab?: string }) {
  const t = useT()
  const nav = useNavigate()
  const { data: c } = useCharacter(characterId)
  const { data: models } = useModels(characterId)
  const jobs = useActiveJobs((j) => j.target.characterId === characterId || (!!j.target.modelId && (c?.modelIds ?? []).includes(j.target.modelId)))
  const tab: Tab = tabParam === 'training' || tabParam === 'models' ? tabParam : 'material'

  useEffect(() => {
    void api.post(`/characters/${characterId}/opened`).catch(() => {})
  }, [characterId])

  if (!c) return <div className="ws-loading skeleton" />

  const primary = models?.find((m) => m.id === c.primaryModelId) ?? models?.[0]
  const trainJob = jobs.find((j) => j.type === 'train')
  const organizing = jobs.some((j) => j.type === 'organize' || j.type === 'import')
  const releases = (models ?? []).reduce((n, m) => n + m.releases.length, 0)

  const setTab = (next: Tab) => nav(`/voices/${characterId}/${next}`)

  const createWith = async () => {
    const p = await attempt(() => api.post<Project>('/projects', { name: c.name }))
    if (p) nav(`/create/${p.id}?with=${c.id}`)
  }

  const next = nextStep(c, trainJob, organizing)

  return (
    <div className="ws">
      <WorkspaceHeader
        c={c}
        action={
          next && (
            <button className={clsx('btn', next.primary ? 'btn-primary' : '')} onClick={() => next.run({ nav, setTab, characterId, createWith })}>
              {next.icon}
              {t(next.label)}
              <ArrowRight />
            </button>
          )
        }
      />

      <nav className="ws-tabs" role="tablist">
        <TabButton active={tab === 'material'} onClick={() => setTab('material')} label={t('ws.tab.material')}
          sub={c.stats.clipCount > 0 ? duration(c.stats.activeDuration, t.lang) : c.stats.rawCount > 0 ? String(c.stats.rawCount) : ''}
          done={c.health != null && !c.health.stale && c.health.level !== 'blocked'} busy={organizing} />
        <TabButton active={tab === 'training'} onClick={() => setTab('training')} label={t('ws.tab.training')}
          sub={trainJob ? (trainJob.progress != null ? percent(trainJob.progress) : '') : primary ? `${steps(primary.step)} / ${steps(primary.targetSteps)}` : ''}
          done={!!primary && primary.step >= primary.targetSteps} busy={!!trainJob} />
        <TabButton active={tab === 'models'} onClick={() => setTab('models')} label={t('ws.tab.models')} sub={releases > 0 ? String(releases) : ''} done={releases > 0} />
      </nav>

      <div className="ws-body">
        {tab === 'material' && <MaterialView character={c} />}
        {tab === 'training' && <TrainingView character={c} />}
        {tab === 'models' && <ModelsView character={c} onCreate={createWith} />}
      </div>
    </div>
  )
}

type NextCtx = { nav: ReturnType<typeof useNavigate>; setTab: (t: Tab) => void; characterId: string; createWith: () => void }

function nextStep(c: Character, trainJob: Job | undefined, organizing: boolean) {
  if (organizing) return null
  if (c.stage === 'empty')
    return { label: 'ws.next.import' as const, icon: <Upload />, primary: true, run: (x: NextCtx) => importMaterial(x.characterId) }
  if (c.stage === 'material') {
    if (c.stats.clipCount === 0 && c.stats.rawCount > 0)
      return { label: 'ws.next.organize' as const, icon: <Sparkles />, primary: true, run: (x: NextCtx) => { x.setTab('material'); window.dispatchEvent(new CustomEvent('stellaria:organize')) } }
    return { label: 'ws.next.fix' as const, icon: <ShieldCheck />, primary: false, run: (x: NextCtx) => x.setTab('material') }
  }
  if (c.stage === 'ready') return { label: 'ws.next.train' as const, icon: <Brain />, primary: true, run: (x: NextCtx) => x.setTab('training') }
  if (c.stage === 'training' || trainJob) return { label: 'ws.next.watch' as const, icon: <Brain />, primary: false, run: (x: NextCtx) => x.setTab('training') }
  return { label: 'ws.next.create' as const, icon: <Sparkles />, primary: true, run: (x: NextCtx) => x.createWith() }
}

function TabButton({ active, onClick, label, sub, done, busy }: { active: boolean; onClick: () => void; label: string; sub: string; done?: boolean; busy?: boolean }) {
  return (
    <button className={clsx('ws-tab', active && 'active', done && 'done', busy && 'busy')} onClick={onClick} role="tab" aria-selected={active}>
      <span className="ws-tab-label">{label}</span>
      {busy ? <Loader2 className="ws-tab-mark spin" /> : done ? <Check className="ws-tab-mark" /> : null}
      {sub && <span className="ws-tab-sub tabular">{sub}</span>}
    </button>
  )
}

function WorkspaceHeader({ c, action }: { c: Character; action: React.ReactNode }) {
  const t = useT()
  const nav = useNavigate()
  const qc = useQueryClient()
  const [params, setParams] = useSearchParams()
  const [editing, setEditing] = useState(params.get('rename') === '1')
  const [name, setName] = useState(c.name)
  const [addingTag, setAddingTag] = useState(false)
  const [consentOpen, setConsentOpen] = useState(false)
  const nameRef = useRef<HTMLInputElement>(null)
  const avatarInput = useRef<HTMLInputElement>(null)

  useEffect(() => setName(c.name), [c.name])
  useEffect(() => {
    if (editing) {
      nameRef.current?.focus()
      nameRef.current?.select()
    }
  }, [editing])

  const commitName = async () => {
    setEditing(false)
    if (params.get('rename')) {
      params.delete('rename')
      setParams(params, { replace: true })
    }
    const trimmed = name.trim()
    if (trimmed && trimmed !== c.name) await patchCharacter(qc, c.id, { name: trimmed })
    else setName(c.name)
  }

  const uploadAvatar = async (file: File) => {
    await attempt(async () => {
      const form = new FormData()
      form.append('file', file, file.name)
      const { token, url } = (await import('../../api/client')).useConnection.getState()
      const res = await fetch(`${url}/api/characters/${c.id}/avatar`, { method: 'PUT', body: form, headers: { Authorization: `Bearer ${token}` } })
      if (!res.ok) throw new Error(await res.text())
      qc.setQueryData(keys.character(c.id), await res.json())
    })
  }

  const exportPack = async () => {
    const path = isTauri ? await pickSavePath(`${c.name}.stella`, ['stella']) : undefined
    if (isTauri && !path) return
    const job = await attempt(() => api.post<Job>(`/characters/${c.id}/export`, { includeMaterial: false, path: path ?? undefined }))
    await deliverExport(job, { done: t('common.exported'), failed: t('common.error') })
  }

  const remove = async () => {
    const ok = await confirmDialog({ title: t('ws.delete_title', { name: c.name }), body: t('ws.delete_body'), confirm: t('common.delete'), cancel: t('common.cancel'), danger: true })
    if (!ok) return
    await attempt(() => api.del(`/characters/${c.id}`, { deleteModels: 1 }))
    nav('/voices')
  }

  const removeTag = (tag: string) => patchCharacter(qc, c.id, { tags: c.tags.filter((x) => x !== tag) })

  return (
    <header className="ws-header" style={{ ['--voice' as string]: c.color }}>
      <RPopover.Root>
        <RPopover.Trigger asChild>
          <button className="ws-avatar" aria-label={t('ws.change_avatar')}>
            <Avatar character={c} size={52} />
            <span className="ws-avatar-edit">
              <Camera />
            </span>
          </button>
        </RPopover.Trigger>
        <RPopover.Portal>
          <RPopover.Content className="popover ws-avatar-pop" sideOffset={8} align="start">
            <button className="btn btn-sm" onClick={() => avatarInput.current?.click()}>
              <Camera />
              {t('ws.change_avatar')}
            </button>
            <div className="eyebrow" style={{ marginTop: 12 }}>
              {t('ws.color')}
            </div>
            <div className="swatches">
              {VOICE_COLORS.map((col) => (
                <button key={col} className={clsx('swatch', col === c.color && 'on')} style={{ background: col }} onClick={() => patchCharacter(qc, c.id, { color: col })} aria-label={col} />
              ))}
            </div>
          </RPopover.Content>
        </RPopover.Portal>
      </RPopover.Root>
      <input ref={avatarInput} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={(e) => e.target.files?.[0] && uploadAvatar(e.target.files[0])} />

      <div className="ws-id">
        {editing ? (
          <input
            ref={nameRef}
            className="ws-name-input"
            value={name}
            placeholder={t('ws.rename_hint')}
            onChange={(e) => setName(e.target.value)}
            onBlur={commitName}
            onKeyDown={(e) => {
              if (e.key === 'Enter') e.currentTarget.blur()
              if (e.key === 'Escape') {
                setName(c.name)
                setEditing(false)
              }
            }}
          />
        ) : (
          <h1 className="ws-name" onDoubleClick={() => setEditing(true)} title={t('common.rename')}>
            {c.name}
          </h1>
        )}
        <div className="ws-meta">
          {!c.consent.authorized && (
            <button className="chip button warning" onClick={() => setConsentOpen(true)}>
              <ShieldQuestion />
              {t('ws.consent_missing')}
            </button>
          )}
          {c.tags.map((tag) => (
            <span key={tag} className="chip ws-tag">
              {tag}
              <button onClick={() => removeTag(tag)} aria-label={t('common.delete')}>
                <X />
              </button>
            </span>
          ))}
          {addingTag ? (
            <input
              className="input ws-tag-input"
              autoFocus
              placeholder={t('ws.tag_placeholder')}
              onBlur={() => setAddingTag(false)}
              onKeyDown={(e) => {
                const v = e.currentTarget.value.trim()
                if (e.key === 'Enter' && v) {
                  if (!c.tags.includes(v)) void patchCharacter(qc, c.id, { tags: [...c.tags, v] })
                  e.currentTarget.value = ''
                }
                if (e.key === 'Escape') setAddingTag(false)
              }}
            />
          ) : (
            <Tip label={t('ws.add_tag')}>
              <button className="chip button ws-tag-add" onClick={() => setAddingTag(true)} aria-label={t('ws.add_tag')}>
                <Plus />
              </button>
            </Tip>
          )}
        </div>
      </div>

      <span className="spacer" />

      {action}
      <Menu
        trigger={
          <button className="icon-btn" aria-label={t('common.more')}>
            <MoreHorizontal />
          </button>
        }
        items={[
          { label: t('common.rename'), icon: <PenLine />, onSelect: () => setEditing(true) },
          { label: t('ws.consent'), icon: <ShieldCheck />, onSelect: () => setConsentOpen(true) },
          { label: c.pinned ? t('ws.unpin') : t('ws.pin'), icon: c.pinned ? <PinOff /> : <Pin />, onSelect: () => patchCharacter(qc, c.id, { pinned: !c.pinned }) },
          { label: t('ws.export'), icon: <Package />, onSelect: exportPack, disabled: c.stage !== 'trained' },
          { label: c.archived ? t('ws.unarchive') : t('ws.archive'), icon: <Archive />, onSelect: () => patchCharacter(qc, c.id, { archived: !c.archived }) },
          { separator: true, label: '' },
          { label: t('ws.delete'), icon: <Trash2 />, danger: true, onSelect: remove },
        ]}
      />

      <ConsentDialog c={c} open={consentOpen} onOpenChange={setConsentOpen} />
    </header>
  )
}

function ConsentDialog({ c, open, onOpenChange }: { c: Character; open: boolean; onOpenChange: (o: boolean) => void }) {
  const t = useT()
  const qc = useQueryClient()
  const [source, setSource] = useState(c.consent.source)
  const [authorized, setAuthorized] = useState(c.consent.authorized)
  const [note, setNote] = useState(c.consent.note)
  useEffect(() => {
    if (open) {
      setSource(c.consent.source)
      setAuthorized(c.consent.authorized)
      setNote(c.consent.note)
    }
  }, [open, c.consent])
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={t('ws.consent_title')}
      description={t('ws.consent_body')}
      actions={
        <>
          <button className="btn btn-ghost" onClick={() => onOpenChange(false)}>
            {t('common.cancel')}
          </button>
          <button
            className="btn btn-primary"
            onClick={async () => {
              await patchCharacter(qc, c.id, { consent: { source, authorized, note } })
              onOpenChange(false)
            }}
          >
            {t('common.save')}
          </button>
        </>
      }
    >
      <div className="col" style={{ gap: 14 }}>
        <label className="field">
          <span className="field-label">{t('ws.consent_source')}</span>
          <input className="input" value={source} onChange={(e) => setSource(e.target.value)} />
        </label>
        <label className="row" style={{ gap: 10 }}>
          <Switch checked={authorized} onChange={setAuthorized} label={t('ws.consent_check')} />
          <span>{t('ws.consent_check')}</span>
        </label>
        <label className="field">
          <span className="field-label">{t('ws.consent_note')}</span>
          <textarea className="input" value={note} onChange={(e) => setNote(e.target.value)} />
        </label>
      </div>
    </Dialog>
  )
}

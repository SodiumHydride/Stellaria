import clsx from 'clsx'
import { Archive, ArrowLeft, FileBox, Pin, Plus, Search, UserPlus } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../api/client'
import { useActiveJobs, useCharacters } from '../../api/hooks'
import type { Character } from '../../api/types'
import { Avatar } from '../../components/Avatar'
import { Menu, Segmented } from '../../components/ui'
import { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { pickVoiceColor } from '../../lib/color'
import { duration, percent } from '../../lib/format'
import { ImportModelDialog } from './ImportModelDialog'

type Filter = 'all' | 'pinned' | 'training' | 'ready'

export async function createCharacter(name: string, used: string[]): Promise<Character | undefined> {
  return attempt(() => api.post<Character>('/characters', { name, color: pickVoiceColor(name + Date.now(), used) }))
}

export function CharacterSidebar({ selectedId }: { selectedId?: string }) {
  const t = useT()
  const nav = useNavigate()
  const [showArchived, setShowArchived] = useState(false)
  const [importing, setImporting] = useState(false)
  const { data: active } = useCharacters(false)
  const { data: archived } = useCharacters(true)
  const [filter, setFilter] = useState<Filter>('all')
  const [query, setQuery] = useState('')
  const trainJobs = useActiveJobs((j) => j.type === 'train')

  const list = useMemo(() => {
    const source = (showArchived ? archived : active) ?? []
    const q = query.trim().toLowerCase()
    return source
      .filter((c) => !q || c.name.toLowerCase().includes(q) || c.tags.some((tag) => tag.toLowerCase().includes(q)))
      .filter((c) => {
        if (showArchived || filter === 'all') return true
        if (filter === 'pinned') return c.pinned
        if (filter === 'training') return c.stage === 'training'
        return c.stage === 'trained'
      })
      .sort((a, b) => Number(b.pinned) - Number(a.pinned) || (b.lastOpenedAt ?? b.updatedAt).localeCompare(a.lastOpenedAt ?? a.updatedAt))
  }, [active, archived, showArchived, filter, query])

  const tags = useMemo(() => Array.from(new Set((active ?? []).flatMap((c) => c.tags))).slice(0, 12), [active])

  const onNew = async () => {
    const c = await createCharacter(t.lang === 'zh-CN' ? '新角色' : 'New voice', (active ?? []).map((x) => x.color))
    if (c) nav(`/voices/${c.id}/material?rename=1`)
  }

  const trainingProgress = (c: Character) => {
    const j = trainJobs.find((x) => c.modelIds.includes(x.target.modelId ?? ''))
    return j?.progress ?? null
  }

  return (
    <aside className="voices-sidebar glass">
      <div className="sb-head">
        <span className="sb-title">{t('space.voices')}</span>
        {(active?.length ?? 0) > 0 && <span className="sb-count tabular">{active!.length}</span>}
        <span className="spacer" />
        <Menu
          trigger={
            <button className="btn btn-sm sb-new" aria-label={t('voices.new')}>
              <Plus />
            </button>
          }
          items={[
            { label: t('voices.new'), icon: <UserPlus />, onSelect: onNew },
            { label: t('imp.title'), icon: <FileBox />, onSelect: () => setImporting(true) },
          ]}
        />
      </div>

      <div className="input-wrap sb-search">
        <Search />
        <input className="input" placeholder={t('voices.search')} value={query} onChange={(e) => setQuery(e.target.value)} />
      </div>

      {!showArchived ? (
        <Segmented<Filter>
          size="sm"
          value={filter}
          onChange={setFilter}
          options={[
            { value: 'all', label: t('voices.filter.all') },
            { value: 'pinned', label: t('voices.filter.pinned') },
            { value: 'training', label: t('voices.filter.training') },
            { value: 'ready', label: t('voices.filter.ready') },
          ]}
        />
      ) : (
        <button className="btn btn-ghost btn-sm" onClick={() => setShowArchived(false)} style={{ alignSelf: 'flex-start' }}>
          <ArrowLeft />
          {t('voices.back_to_active')}
        </button>
      )}

      {tags.length > 0 && !showArchived && (
        <div className="sb-tags">
          {tags.map((tag) => (
            <button key={tag} className={clsx('chip button', query === tag && 'accent')} onClick={() => setQuery(query === tag ? '' : tag)}>
              {tag}
            </button>
          ))}
        </div>
      )}

      <div className="sb-list">
        {list.map((c) => {
          const p = c.stage === 'training' ? trainingProgress(c) : null
          return (
            <button key={c.id} className={clsx('sb-item', c.id === selectedId && 'selected')} onClick={() => nav(`/voices/${c.id}`)}>
              <Avatar character={c} size={34} />
              <span className="sb-item-text">
                <span className="sb-item-name">
                  {c.name}
                  {c.pinned && <Pin className="sb-pin" />}
                </span>
                <span className="sb-item-sub">
                  {c.stage === 'training' && p != null
                    ? `${t('stage.training')} · ${percent(p)}`
                    : c.stats.activeDuration > 0 && c.stage !== 'trained'
                      ? `${t(`stage.${c.stage}` as const)} · ${duration(c.stats.activeDuration, t.lang)}`
                      : t(`stage.${c.stage}` as const)}
                </span>
              </span>
              <span className={clsx('status-dot', c.stage === 'trained' && 'good', c.stage === 'training' && 'active')} />
              {p != null && <span className="sb-item-progress" style={{ width: `${p * 100}%` }} />}
            </button>
          )
        })}
        {list.length === 0 && (active?.length ?? 0) > 0 && <div className="sb-empty">{t('voices.none_match')}</div>}
      </div>

      {!showArchived && (archived?.length ?? 0) > 0 && (
        <button className="btn btn-ghost btn-sm sb-archived" onClick={() => setShowArchived(true)}>
          <Archive />
          {t('voices.archived', { n: archived!.length })}
        </button>
      )}
      {importing && <ImportModelDialog onClose={() => setImporting(false)} />}
    </aside>
  )
}

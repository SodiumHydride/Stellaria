import { useQueryClient } from '@tanstack/react-query'
import { Copy, MoreHorizontal, Plus, Sparkles, Trash2 } from 'lucide-react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../../api/client'
import { useCharacters, useProjects } from '../../api/hooks'
import { keys } from '../../api/keys'
import type { Project } from '../../api/types'
import { Avatar } from '../../components/Avatar'
import { confirmDialog, Empty, Menu } from '../../components/ui'
import { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { relativeTime } from '../../lib/format'
import { coverColor } from '../../lib/color'
import { CanvasWorkspace } from './CanvasWorkspace'
import './canvas.css'

/** 创作: a gallery of projects; each opens as an infinite canvas with a timeline. */
export default function CreateSpace() {
  const { projectId } = useParams()
  if (projectId) return <CanvasWorkspace key={projectId} projectId={projectId} />
  return <Gallery />
}

function Gallery() {
  const t = useT()
  const nav = useNavigate()
  const qc = useQueryClient()
  const { data: projects, isLoading } = useProjects()
  const { data: characters } = useCharacters()

  const create = async () => {
    const p = await attempt(() => api.post<Project>('/projects', { name: t('create.untitled') }))
    if (p) nav(`/create/${p.id}`)
  }

  const remove = async (id: string, name: string) => {
    const ok = await confirmDialog({ title: t('create.delete_title', { name }), body: t('create.delete_body'), confirm: t('common.delete'), cancel: t('common.cancel'), danger: true })
    if (!ok) return
    await attempt(() => api.del(`/projects/${id}`))
    qc.invalidateQueries({ queryKey: keys.projects })
  }

  const duplicate = async (id: string) => {
    await attempt(() => api.post<Project>(`/projects/${id}/duplicate`))
    qc.invalidateQueries({ queryKey: keys.projects })
  }

  return (
    <div className="page">
      <div className="page-inner">
        <div className="page-header">
          <h1 className="page-title">{t('create.title')}</h1>
          <span className="spacer" />
          <button className="btn btn-primary btn-lg" onClick={create}>
            <Plus />
            {t('create.new')}
          </button>
        </div>

        {!isLoading && (projects?.length ?? 0) === 0 ? (
          <div className="glass">
            <Empty icon={<Sparkles />} title={t('create.empty.title')} body={t('create.empty.body')}>
              <button className="btn btn-primary" onClick={create}>
                <Plus />
                {t('create.new')}
              </button>
            </Empty>
          </div>
        ) : (
          <div className="gallery">
            <button className="gallery-new" onClick={create}>
              <Plus />
              <span>{t('create.new')}</span>
            </button>
            {(projects ?? []).map((p) => (
              <div key={p.id} className="gallery-card glass" style={{ ['--cover' as string]: coverColor(p) }} onClick={() => nav(`/create/${p.id}`)} role="button" tabIndex={0}>
                <div className="gallery-cover">
                  <div className="gallery-voices">
                    {p.characterIds.slice(0, 5).map((cid) => {
                      const c = characters?.find((x) => x.id === cid)
                      return c ? <Avatar key={cid} character={c} size={28} /> : null
                    })}
                  </div>
                </div>
                <div className="gallery-info">
                  <div className="gallery-name">{p.name}</div>
                  <div className="muted gallery-meta">
                    {t('create.renders', { n: p.renderCount })} · {relativeTime(p.updatedAt, t.lang)}
                  </div>
                </div>
                <div className="gallery-menu" onClick={(e) => e.stopPropagation()}>
                  <Menu
                    trigger={
                      <button className="icon-btn sm" aria-label={t('common.more')}>
                        <MoreHorizontal />
                      </button>
                    }
                    items={[
                      { label: t('create.duplicate'), icon: <Copy />, onSelect: () => duplicate(p.id) },
                      { separator: true, label: '' },
                      { label: t('common.delete'), icon: <Trash2 />, danger: true, onSelect: () => remove(p.id, p.name) },
                    ]}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

import clsx from 'clsx'
import { FileAudio, Layers, Trash2 } from 'lucide-react'
import { useMemo } from 'react'
import { api } from '../../../api/client'
import type { Clip, RawImport } from '../../../api/types'
import { confirmDialog, Tip } from '../../../components/ui'
import { useT } from '../../../i18n'
import { attempt } from '../../../lib/actions'
import { clock } from '../../../lib/format'

/** The files material came from. Picking one narrows the clip list to what was cut from it. */
export function SourceList({
  characterId,
  raw,
  clips,
  selected,
  onSelect,
}: {
  characterId: string
  raw: RawImport[]
  clips: Clip[]
  selected: string | null
  onSelect: (id: string | null) => void
}) {
  const t = useT()
  const counts = useMemo(() => {
    const m = new Map<string, number>()
    for (const c of clips) if (c.status === 'active' && c.rawId) m.set(c.rawId, (m.get(c.rawId) ?? 0) + 1)
    return m
  }, [clips])
  const active = clips.filter((c) => c.status === 'active').length

  const remove = async (r: RawImport) => {
    const ok = await confirmDialog({ title: t('mat.raw.remove_title', { name: r.name }), body: t('mat.raw.remove_body'), confirm: t('common.delete'), cancel: t('common.cancel'), danger: true })
    if (!ok) return
    if (selected === r.id) onSelect(null)
    await attempt(() => api.del(`/characters/${characterId}/raw/${r.id}`, { withClips: 1 }))
  }

  return (
    <aside className="sources glass">
      <div className="sources-head">
        <span className="section-title">{t('mat.raw.title')}</span>
        <span className="muted tabular">{raw.length}</span>
      </div>
      <div className="sources-list">
        <button className={clsx('src', selected == null && 'on')} onClick={() => onSelect(null)}>
          <Layers className="src-icon" />
          <span className="src-name">{t('mat.raw.all')}</span>
          <span className="src-count tabular">{active}</span>
        </button>
        {raw.map((r) => {
          const pending = r.status === 'imported' || r.status === 'failed'
          return (
            <div key={r.id} className={clsx('src', selected === r.id && 'on', pending && 'pending')} role="button" tabIndex={0} onClick={() => onSelect(selected === r.id ? null : r.id)}>
              <FileAudio className="src-icon" />
              <span className="src-text">
                <span className="src-name" title={r.name}>
                  {r.name}
                </span>
                <span className="src-sub tabular">
                  {r.duration != null ? clock(r.duration) : ''}
                  {r.status === 'organized' ? '' : ` · ${t(`mat.raw.status.${r.status}` as const)}`}
                </span>
              </span>
              {r.status === 'organized' && <span className="src-count tabular">{counts.get(r.id) ?? 0}</span>}
              <Tip label={t('mat.raw.remove')}>
                <button
                  className="icon-btn sm src-del"
                  onClick={(e) => {
                    e.stopPropagation()
                    void remove(r)
                  }}
                >
                  <Trash2 />
                </button>
              </Tip>
            </div>
          )
        })}
      </div>
    </aside>
  )
}

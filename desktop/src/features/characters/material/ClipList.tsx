import { useQueryClient } from '@tanstack/react-query'
import { useVirtualizer } from '@tanstack/react-virtual'
import clsx from 'clsx'
import { Check, Headphones, ListOrdered, MoreHorizontal, PenLine, RotateCcw, Scissors, Shuffle, Trash2, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api, mediaUrl } from '../../../api/client'
import type { Character, Clip, ClipFlag } from '../../../api/types'
import { PlayButton, Waveform } from '../../../components/Waveform'
import { confirmDialog, Dialog, Menu, Segmented, Tip } from '../../../components/ui'
import { useT } from '../../../i18n'
import { attempt } from '../../../lib/actions'
import { useEvent } from '../../../lib/useEvent'
import { clock, hzToNote } from '../../../lib/format'
import { usePlayer } from '../../../state/player'

type Filter = 'all' | 'issues' | 'val' | 'unreviewed' | 'trash'

const SEVERE: ClipFlag[] = ['too_short', 'clipping', 'silent', 'f0_failed', 'duplicate', 'music']

export function flagSeverity(f: ClipFlag): 'critical' | 'warning' | 'info' {
  if (f === 'too_short') return 'critical'
  if (SEVERE.includes(f)) return 'warning'
  return 'info'
}

export function clipAudioPath(characterId: string, clip: Clip) {
  return `/characters/${characterId}/clips/${clip.id}/audio`
}

export function clipPeaksPath(characterId: string, clip: Clip) {
  return `/characters/${characterId}/clips/${clip.id}/peaks`
}

export async function patchClips(characterId: string, ids: string[], set: Partial<Pick<Clip, 'status' | 'split' | 'reviewed' | 'name'>>) {
  return attempt(() => api.patch<Clip[]>(`/characters/${characterId}/clips`, { ids, set }))
}

export function ClipList({
  character,
  clips,
  loading,
  focusIds,
  onClearFocus,
  onReview,
}: {
  character: Character
  clips: Clip[]
  loading: boolean
  focusIds: string[] | null
  onClearFocus: () => void
  onReview: () => void
}) {
  const t = useT()
  const qc = useQueryClient()
  const cid = character.id
  const [filter, setFilter] = useState<Filter>('all')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const anchor = useRef<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [renaming, setRenaming] = useState<string | null>(null)
  const [renameAll, setRenameAll] = useState(false)
  const playerId = usePlayer((s) => s.currentId)
  const playerTime = usePlayer((s) => s.time)

  const counts = useMemo(() => {
    const active = clips.filter((c) => c.status === 'active')
    return {
      all: active.length,
      issues: active.filter((c) => c.flags.length > 0).length,
      val: active.filter((c) => c.split === 'val').length,
      unreviewed: active.filter((c) => !c.reviewed).length,
      trash: clips.length - active.length,
    }
  }, [clips])

  const shown = useMemo(() => {
    if (focusIds) return clips.filter((c) => focusIds.includes(c.id))
    return clips.filter((c) => {
      if (filter === 'trash') return c.status === 'trashed'
      if (c.status !== 'active') return false
      if (filter === 'issues') return c.flags.length > 0
      if (filter === 'val') return c.split === 'val'
      if (filter === 'unreviewed') return !c.reviewed
      return true
    })
  }, [clips, filter, focusIds])

  useEffect(() => setSelected(new Set()), [filter, focusIds])

  const rowVirtualizer = useVirtualizer({ count: shown.length, getScrollElement: () => scrollRef.current, estimateSize: () => 54, overscan: 8 })

  const select = (id: string, e: React.MouseEvent) => {
    const next = new Set(selected)
    if (e.shiftKey && anchor.current) {
      const a = shown.findIndex((c) => c.id === anchor.current)
      const b = shown.findIndex((c) => c.id === id)
      const [lo, hi] = a < b ? [a, b] : [b, a]
      for (let i = lo; i <= hi; i++) next.add(shown[i].id)
    } else if (e.metaKey || e.ctrlKey) {
      if (next.has(id)) next.delete(id)
      else next.add(id)
      anchor.current = id
    } else {
      if (next.size === 1 && next.has(id)) next.clear()
      else {
        next.clear()
        next.add(id)
      }
      anchor.current = id
    }
    setSelected(next)
  }

  const ids = Array.from(selected)
  const single = ids.length === 1 ? shown.find((c) => c.id === ids[0]) : undefined
  const canSplit = !!single && playerId === single.id && playerTime > 0.3 && playerTime < single.duration - 0.3
  const splitAtPlayhead = async () => {
    if (!single || !canSplit) return
    await attempt(() => api.post(`/characters/${cid}/clips/${single.id}/split`, { at: usePlayer.getState().time }))
    usePlayer.getState().stop()
    setSelected(new Set())
  }
  const bulk = async (set: Parameters<typeof patchClips>[2]) => {
    await patchClips(cid, ids, set)
    setSelected(new Set())
    qc.invalidateQueries({ queryKey: ['clips', cid] })
  }

  // keyboard: space plays the selected clip, delete trashes the selection
  const onKey = useEvent((e: KeyboardEvent) => {
    if ((e.target as HTMLElement).closest('input, textarea, [role="dialog"]') || document.querySelector('.review')) return
    if (selected.size === 0) return
    if (e.key === ' ') {
      e.preventDefault()
      const clip = shown.find((c) => selected.has(c.id))
      if (clip) usePlayer.getState().toggle(clip.id, mediaUrl(clipAudioPath(cid, clip)))
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault()
      void bulk({ status: filter === 'trash' ? 'active' : 'trashed' })
    } else if (e.key === 'Escape') setSelected(new Set())
    else if (e.key === 'F2' && single) {
      e.preventDefault()
      setRenaming(single.id)
    } else if (e.key.toLowerCase() === 's' && !e.metaKey && !e.ctrlKey) {
      e.preventDefault()
      void splitAtPlayhead()
    }
  })
  useEffect(() => {
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onKey])

  const emptyTrash = async () => {
    const ok = await confirmDialog({ title: t('mat.action.empty_trash_title', { n: counts.trash }), confirm: t('common.delete'), cancel: t('common.cancel'), danger: true })
    if (ok) await attempt(() => api.post(`/characters/${cid}/clips/empty-trash`))
  }

  return (
    <div className="clips glass">
      <div className="clips-toolbar">
        {focusIds ? (
          <button className="btn btn-sm" onClick={onClearFocus}>
            <X />
            {t('mat.filter.all')}
          </button>
        ) : (
          <Segmented<Filter>
            size="sm"
            value={filter}
            onChange={setFilter}
            options={[
              { value: 'all', label: <>{t('mat.filter.all')} <span className="seg-count">{counts.all}</span></> },
              { value: 'issues', label: <>{t('mat.filter.issues')} <span className="seg-count">{counts.issues}</span></> },
              { value: 'val', label: <>{t('mat.filter.val')} <span className="seg-count">{counts.val}</span></> },
              { value: 'unreviewed', label: <>{t('mat.filter.unreviewed')} <span className="seg-count">{counts.unreviewed}</span></> },
              { value: 'trash', label: <>{t('mat.filter.trash')} <span className="seg-count">{counts.trash}</span></> },
            ]}
          />
        )}
        <span className="spacer" />
        <button className="btn btn-sm" onClick={onReview} disabled={counts.all === 0}>
          <Headphones />
          {t('mat.review')}
        </button>
        <Menu
          trigger={
            <button className="icon-btn" aria-label={t('common.more')}>
              <MoreHorizontal />
            </button>
          }
          items={[
            { label: t('mat.action.repick_val'), icon: <Shuffle />, onSelect: () => attempt(() => api.post(`/characters/${cid}/clips/pick-val`, {})) },
            { label: t('mat.action.rename_all'), icon: <ListOrdered />, onSelect: () => setRenameAll(true) },
            { separator: true, label: '' },
            { label: t('mat.action.empty_trash'), icon: <Trash2 />, danger: true, disabled: counts.trash === 0, onSelect: emptyTrash },
          ]}
        />
      </div>

      {selected.size > 0 && (
        <div className="clips-bulk">
          <span>{t('mat.selected', { n: selected.size })}</span>
          <span className="spacer" />
          {filter === 'trash' ? (
            <button className="btn btn-sm" onClick={() => bulk({ status: 'active' })}>
              <RotateCcw />
              {t('mat.action.restore')}
            </button>
          ) : (
            <>
              {single && (
                <>
                  <Tip label="F2">
                    <button className="btn btn-sm btn-ghost" onClick={() => setRenaming(single.id)}>
                      <PenLine />
                      {t('common.rename')}
                    </button>
                  </Tip>
                  <Tip label={canSplit ? 'S' : t('mat.action.split_hint')}>
                    <span>
                      <button className="btn btn-sm btn-ghost" onClick={splitAtPlayhead} disabled={!canSplit}>
                        <Scissors />
                        {t('mat.action.split')}
                      </button>
                    </span>
                  </Tip>
                </>
              )}
              <button className="btn btn-sm btn-ghost" onClick={() => bulk({ reviewed: true })}>
                <Check />
                {t('mat.action.mark_reviewed')}
              </button>
              <button className="btn btn-sm btn-ghost" onClick={() => bulk({ split: 'val' })}>
                {t('mat.action.to_val')}
              </button>
              <button className="btn btn-sm btn-ghost" onClick={() => bulk({ split: 'train' })}>
                {t('mat.action.to_train')}
              </button>
              <button className="btn btn-sm btn-danger" onClick={() => bulk({ status: 'trashed' })}>
                <Trash2 />
                {t('mat.action.trash')}
              </button>
            </>
          )}
          <button className="icon-btn sm" onClick={() => setSelected(new Set())} aria-label={t('mat.clear_selection')}>
            <X />
          </button>
        </div>
      )}

      <div className="clips-scroll" ref={scrollRef}>
        {loading ? (
          <div className="col" style={{ padding: 12 }}>
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="skeleton" style={{ height: 40 }} />
            ))}
          </div>
        ) : shown.length === 0 ? (
          <div className="clips-empty">{counts.all === 0 && filter === 'all' ? t('mat.no_clips') : t('mat.empty_filter')}</div>
        ) : (
          <div style={{ height: rowVirtualizer.getTotalSize(), position: 'relative' }}>
            {rowVirtualizer.getVirtualItems().map((vi) => {
              const clip = shown[vi.index]
              return (
                <ClipRow
                  key={clip.id}
                  characterId={cid}
                  color={character.color}
                  clip={clip}
                  selected={selected.has(clip.id)}
                  renaming={renaming === clip.id}
                  onRenameStart={() => setRenaming(clip.id)}
                  onRenameEnd={() => setRenaming(null)}
                  onSelect={(e) => select(clip.id, e)}
                  style={{ transform: `translateY(${vi.start}px)` }}
                />
              )
            })}
          </div>
        )}
      </div>
      <RenameAllDialog character={character} clips={clips} open={renameAll} onOpenChange={setRenameAll} />
    </div>
  )
}

/** Live preview of a naming pattern, mirroring the backend's tokens. */
function applyPattern(pattern: string, name: string, index: number, source: string | null) {
  return pattern
    .replace(/\{index(?::0?(\d)d?)?\}/g, (_, w) => String(index).padStart(w ? Number(w) : 1, '0'))
    .replace(/\{name\}/g, name)
    .replace(/\{source\}/g, (source ?? '').replace(/\.[^.]+$/, ''))
}

function RenameAllDialog({ character, clips, open, onOpenChange }: { character: Character; clips: Clip[]; open: boolean; onOpenChange: (o: boolean) => void }) {
  const t = useT()
  const [pattern, setPattern] = useState('{name}_{index:04d}')
  const active = clips.filter((c) => c.status === 'active')
  const apply = async () => {
    const ok = await attempt(() => api.post(`/characters/${character.id}/clips/rename`, { pattern }))
    if (ok) onOpenChange(false)
  }
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={t('mat.rename.title', { n: active.length })}
      actions={
        <>
          <button className="btn btn-ghost" onClick={() => onOpenChange(false)}>
            {t('common.cancel')}
          </button>
          <button className="btn btn-primary" onClick={apply} disabled={!pattern.includes('{index')}>
            {t('common.rename')}
          </button>
        </>
      }
    >
      <div className="col" style={{ gap: 12 }}>
        <input className="input mono" value={pattern} onChange={(e) => setPattern(e.target.value)} aria-label={t('mat.organize.name_pattern')} autoFocus />
        <div className="rename-tokens">
          {['{name}', '{index:04d}', '{source}'].map((tok) => (
            <button key={tok} className="chip button mono" onClick={() => setPattern((p) => p + tok)}>
              {tok}
            </button>
          ))}
        </div>
        <div className="rename-preview">
          {active.slice(0, 4).map((c, i) => (
            <div key={c.id} className="row">
              <span className="muted">{c.name}</span>
              <span className="muted">→</span>
              <strong>{applyPattern(pattern, character.name, i + 1, c.rawName)}</strong>
            </div>
          ))}
        </div>
      </div>
    </Dialog>
  )
}

function ClipRow({
  characterId,
  color,
  clip,
  selected,
  renaming,
  onRenameStart,
  onRenameEnd,
  onSelect,
  style,
}: {
  characterId: string
  color: string
  clip: Clip
  selected: boolean
  renaming: boolean
  onRenameStart: () => void
  onRenameEnd: () => void
  onSelect: (e: React.MouseEvent) => void
  style: React.CSSProperties
}) {
  const t = useT()
  const trashed = clip.status === 'trashed'
  const m = clip.metrics
  const commit = async (value: string) => {
    onRenameEnd()
    const v = value.trim()
    if (v && v !== clip.name) await patchClips(characterId, [clip.id], { name: v })
  }
  return (
    <div className={clsx('clip-row', selected && 'selected', trashed && 'trashed')} style={style} onClick={onSelect}>
      <PlayButton id={clip.id} audioPath={clipAudioPath(characterId, clip)} />
      <div className="clip-name">
        {renaming ? (
          <input
            className="input clip-rename"
            defaultValue={clip.name}
            autoFocus
            onFocus={(e) => e.currentTarget.select()}
            onClick={(e) => e.stopPropagation()}
            onBlur={(e) => void commit(e.currentTarget.value)}
            onKeyDown={(e) => {
              e.stopPropagation()
              if (e.key === 'Enter') e.currentTarget.blur()
              if (e.key === 'Escape') onRenameEnd()
            }}
          />
        ) : (
          <span className="clip-title" onDoubleClick={onRenameStart}>
            {clip.name}
            {clip.reviewed && <Check className="clip-reviewed" />}
          </span>
        )}
        {clip.rawName && (
          <span className="clip-source">
            {clip.rawName}
            {clip.start != null && ` · ${clock(clip.start)}`}
          </span>
        )}
      </div>
      <div className="clip-wave">
        <Waveform id={clip.id} peaksPath={clipPeaksPath(characterId, clip)} audioPath={clipAudioPath(characterId, clip)} height={30} bars={110} color={color} />
      </div>
      <span className="clip-dur tabular">{clip.duration.toFixed(1)}s</span>
      <span className="clip-pitch tabular">{m?.f0Low ? `${hzToNote(m.f0Low)}–${hzToNote(m.f0High)}` : '–'}</span>
      <span className="clip-flags">
        {clip.flags.slice(0, 2).map((f) => (
          <span key={f} className={clsx('chip', flagSeverity(f))}>
            {t(`flag.${f}` as const)}
          </span>
        ))}
        {clip.flags.length > 2 && <span className="chip">+{clip.flags.length - 2}</span>}
      </span>
      <button
        className={clsx('chip button clip-split', clip.split === 'val' && 'accent')}
        onClick={(e) => {
          e.stopPropagation()
          void patchClips(characterId, [clip.id], { split: clip.split === 'val' ? 'train' : 'val' })
        }}
      >
        {clip.split === 'val' ? t('mat.split.val') : t('mat.split.train')}
      </button>
      <Tip label={trashed ? t('mat.action.restore') : t('mat.action.trash')}>
        <button
          className="icon-btn sm clip-trash"
          onClick={(e) => {
            e.stopPropagation()
            void patchClips(characterId, [clip.id], { status: trashed ? 'active' : 'trashed' })
          }}
        >
          {trashed ? <RotateCcw /> : <Trash2 />}
        </button>
      </Tip>
    </div>
  )
}

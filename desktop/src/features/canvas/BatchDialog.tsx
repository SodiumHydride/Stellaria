import clsx from 'clsx'
import { Check, Loader2, Users } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { Character, KeySuggestion, Model, SourcePort } from '../../api/types'
import { Avatar } from '../../components/Avatar'
import { Dialog, Segmented, Switch } from '../../components/ui'
import { useT } from '../../i18n'
import { signed } from '../../lib/format'
import { useCanvas } from './context'
import { QUALITY, slotRightOf, stemMedia, type Quality, type SourceNode } from './model'
import { usePortLabel } from './nodes'
import { newId } from './useProject'
import { latestRelease, voiceModelFor } from './voices'

const KEY_CHOICES = [-5, -3, -2, -1, 0, 1, 2, 3, 5]
type Versions = 'latest' | 'all'

/** Which versions of a voice's model to sing with: the newest release, or every release (+ the training checkpoint). */
function versionsOf(model: Model | undefined, mode: Versions): (string | null)[] {
  if (!model) return []
  const newest = latestRelease(model)
  if (mode === 'latest') return [newest?.id ?? null]
  const ids: (string | null)[] = model.releases.slice().sort((a, b) => b.step - a.step).map((r) => r.id)
  if (model.status === 'training' || model.status === 'paused' || ids.length === 0) ids.push(null)
  return ids
}

/**
 * 多人试唱: songs × voices × versions × keys in one go. Takes land as tidy stacks next to each
 * song and render one after another; they come back selected for comparing.
 */
export function BatchDialog({ sources, voices, onClose, onCreated }: { sources: SourceNode[]; voices: Character[]; onClose: () => void; onCreated: (ids: string[]) => void }) {
  const t = useT()
  const portLabel = usePortLabel()
  const { project, models, actions } = useCanvas()
  const allSeparated = sources.every((s) => stemMedia(s.data, 'vocals'))
  // unseparated songs get split automatically when the takes run
  const [port, setPort] = useState<SourcePort>('vocals')
  const [picked, setPicked] = useState<Set<string>>(() => new Set(voices.slice(0, 3).map((c) => c.id)))
  const [versions, setVersions] = useState<Versions>('latest')
  const [keyMode, setKeyMode] = useState<'auto' | 'fixed'>('auto')
  const [keys, setKeys] = useState<Set<number>>(new Set([0]))
  const [quality, setQuality] = useState<Quality>('standard')
  const [renderNow, setRenderNow] = useState(true)
  const [suggested, setSuggested] = useState<Record<string, number | null>>({})
  const [busy, setBusy] = useState(false)
  const modelList = Array.from(models.values())
  const first = sources[0]

  const portFor = (_s: SourceNode): SourcePort => port
  /** what to measure the key on: the vocals when they exist */
  const keyMedia = (s: SourceNode) => stemMedia(s.data, 'vocals') ?? s.data.mediaId

  // the key each voice would sing the first song in (shown next to each voice)
  useEffect(() => {
    const mid = keyMedia(first)
    let cancelled = false
    for (const c of voices) {
      api
        .get<KeySuggestion>(`/projects/${project.id}/suggest-key`, { mediaId: mid, characterId: c.id })
        .then((s) => !cancelled && setSuggested((prev) => ({ ...prev, [c.id]: s.key })))
        .catch(() => !cancelled && setSuggested((prev) => ({ ...prev, [c.id]: null })))
    }
    return () => {
      cancelled = true
    }
  }, [first, port, voices, project.id])

  const chosen = voices.filter((v) => picked.has(v.id))
  const keyList = keyMode === 'fixed' ? Array.from(keys).sort((a, b) => a - b) : [null]
  const versionCount = chosen.reduce((n, c) => n + Math.max(1, versionsOf(voiceModelFor(c, modelList), versions).length), 0)
  const total = sources.length * versionCount * keyList.length

  const toggle = <T,>(set: Set<T>, v: T, setter: (s: Set<T>) => void) => {
    const next = new Set(set)
    if (next.has(v)) next.delete(v)
    else next.add(v)
    setter(next)
  }

  const suggestFor = async (s: SourceNode, characterId: string): Promise<number> => {
    if (s.id === first.id && suggested[characterId] != null) return suggested[characterId]!
    const mid = keyMedia(s)
    const r = await api.get<KeySuggestion>(`/projects/${project.id}/suggest-key`, { mediaId: mid, characterId }).catch(() => null)
    return r?.key ?? 0
  }

  const create = async () => {
    setBusy(true)
    const batchId = newId('bt')
    const ids: string[] = []
    for (const s of sources) {
      let slot = slotRightOf(project.canvas.nodes, s.id)
      for (const c of chosen) {
        const model = voiceModelFor(c, modelList)
        const releaseIds = versionsOf(model, versions)
        for (const releaseId of releaseIds.length ? releaseIds : [undefined]) {
          for (const k of keyList) {
            const key = k ?? (await suggestFor(s, c.id))
            const id = await actions.sing({ nodeId: s.id, port: portFor(s) }, c.id, { position: slot, params: { key, ...QUALITY[quality] }, batchId, releaseId, focus: false })
            if (id) ids.push(id)
            slot = { x: slot.x, y: slot.y + 158 }
          }
        }
      }
    }
    onCreated(ids)
    if (renderNow) await actions.run(ids)
    setBusy(false)
  }

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && !busy && onClose()}
      title={sources.length > 1 ? t('batch.title_many', { n: sources.length }) : t('batch.title', { name: first.data.name })}
      wide
      actions={
        <>
          <label className="batch-render-now">
            <Switch checked={renderNow} onChange={setRenderNow} label={t('batch.render_now')} />
            <span>{t('batch.render_now')}</span>
          </label>
          <span className="spacer" />
          <button className="btn btn-ghost" onClick={onClose} disabled={busy}>
            {t('common.cancel')}
          </button>
          <button className="btn btn-primary" onClick={create} disabled={busy || total === 0}>
            {busy ? <Loader2 className="spin" /> : <Users />}
            {t('batch.create', { n: total })}
          </button>
        </>
      }
    >
      <div className="batch">
        {sources.length > 1 && (
          <div className="batch-songs">
            {sources.map((s) => (
              <span key={s.id} className="chip">
                {s.data.name}
              </span>
            ))}
          </div>
        )}

        {(
          <div className="batch-row">
            <span className="field-label">{t('batch.input')}</span>
            <Segmented<SourcePort>
              size="sm"
              value={port}
              onChange={setPort}
              options={[
                { value: 'vocals', label: portLabel('vocals') },
                { value: 'mix', label: portLabel('mix') },
              ]}
            />
            {!allSeparated && port === 'vocals' && <span className="muted batch-note">{t('batch.some_unseparated')}</span>}
          </div>
        )}

        <div className="batch-voices">
          {voices.map((c) => {
            const on = picked.has(c.id)
            const s = suggested[c.id]
            return (
              <button key={c.id} className={clsx('batch-voice', on && 'on')} onClick={() => toggle(picked, c.id, setPicked)} aria-pressed={on}>
                <Avatar character={c} size={34} />
                <span className="batch-voice-name">{c.name}</span>
                {keyMode === 'auto' && <span className="muted tabular batch-voice-key">{s === undefined ? '…' : s === null ? '±0' : signed(s)}</span>}
                <span className="batch-check">{on && <Check />}</span>
              </button>
            )
          })}
        </div>

        <div className="batch-row">
          <span className="field-label">{t('batch.versions')}</span>
          <Segmented<Versions>
            size="sm"
            value={versions}
            onChange={setVersions}
            options={[
              { value: 'latest', label: t('batch.versions_latest') },
              { value: 'all', label: t('batch.versions_all') },
            ]}
          />
        </div>

        <div className="batch-row">
          <span className="field-label">{t('insp.key')}</span>
          <Segmented<'auto' | 'fixed'>
            size="sm"
            value={keyMode}
            onChange={setKeyMode}
            options={[
              { value: 'auto', label: t('batch.key_auto') },
              { value: 'fixed', label: t('batch.key_fixed') },
            ]}
          />
          {keyMode === 'fixed' && (
            <div className="batch-keys">
              {KEY_CHOICES.map((k) => (
                <button key={k} className={clsx('chip button tabular', keys.has(k) && 'accent')} onClick={() => toggle(keys, k, setKeys)}>
                  {signed(k)}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="batch-row">
          <span className="field-label">{t('insp.quality')}</span>
          <Segmented<Quality>
            size="sm"
            value={quality}
            onChange={setQuality}
            options={[
              { value: 'fast', label: t('insp.quality.fast') },
              { value: 'standard', label: t('insp.quality.standard') },
              { value: 'fine', label: t('insp.quality.fine') },
            ]}
          />
        </div>
      </div>
    </Dialog>
  )
}

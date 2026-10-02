import * as RPopover from '@radix-ui/react-popover'
import clsx from 'clsx'
import { BadgeCheck, Flag, Headphones, Trash2, Upload } from 'lucide-react'
import { useState } from 'react'
import { api } from '../../../api/client'
import type { Character, Checkpoint, Model, Release } from '../../../api/types'
import { useT } from '../../../i18n'
import { attempt } from '../../../lib/actions'
import { bytes, relativeTime, steps } from '../../../lib/format'

/** Checkpoints as marks along the training timeline; publish the best one as a version. */
export function CheckpointTrack({ model, character, onListen }: { model: Model; character: Character; onListen: (step: number) => void }) {
  const t = useT()
  const max = Math.max(model.targetSteps, model.step, ...model.checkpoints.map((c) => c.step), 1)
  const pct = (s: number) => `${(s / max) * 100}%`
  return (
    <div className="track glass" style={{ ['--voice' as string]: character.color }}>
      <div className="row">
        <Flag className="track-icon" />
        <span className="section-title">{t('ckpt.title')}</span>
      </div>
      <div className="track-rail">
        <div className="track-line" />
        <div className="track-progress" style={{ width: pct(model.step) }} />
        {model.checkpoints.map((c) => (
          <CheckpointStar key={c.step} model={model} character={character} ckpt={c} left={pct(c.step)} release={model.releases.find((r) => r.id === c.releaseId)} onListen={onListen} />
        ))}
        <div className="track-end tabular" style={{ left: '100%' }}>
          {steps(max)}
        </div>
      </div>
    </div>
  )
}

function CheckpointStar({ model, character, ckpt, left, release, onListen }: { model: Model; character: Character; ckpt: Checkpoint; left: string; release?: Release; onListen: (s: number) => void }) {
  const t = useT()
  const [name, setName] = useState(t('ckpt.release_default', { name: character.name, steps: steps(ckpt.step) }))
  const publish = () => attempt(() => api.post<Release>(`/models/${model.id}/releases`, { step: ckpt.step, name }))
  const remove = () => attempt(() => api.del(`/models/${model.id}/checkpoints/${ckpt.step}`))
  return (
    <RPopover.Root>
      <RPopover.Trigger asChild>
        <button className={clsx('track-star', ckpt.kind, release && 'released')} style={{ left }} aria-label={`${ckpt.step}`}>
          <span className="track-dot" />
          <span className="track-star-label tabular">{steps(ckpt.step)}</span>
        </button>
      </RPopover.Trigger>
      <RPopover.Portal>
        <RPopover.Content className="popover track-pop" sideOffset={10}>
          <div className="row">
            <strong className="tabular">{t('val.step', { step: ckpt.step.toLocaleString() })}</strong>
            <span className="spacer" />
            <span className="chip">{t(`ckpt.kind.${ckpt.kind}` as const)}</span>
          </div>
          <div className="muted" style={{ fontSize: 'var(--fs-xs)', margin: '4px 0 10px' }}>
            {relativeTime(ckpt.createdAt, t.lang)} · {bytes(ckpt.size)}
          </div>
          <button className="btn btn-sm" style={{ width: '100%' }} onClick={() => onListen(ckpt.step)}>
            <Headphones />
            {t('ckpt.listen')}
          </button>
          {release ? (
            <div className="chip good" style={{ marginTop: 10 }}>
              <BadgeCheck />
              {t('ckpt.published')} · {release.name}
            </div>
          ) : (
            <div className="col" style={{ marginTop: 10, gap: 6 }}>
              <input className="input" value={name} onChange={(e) => setName(e.target.value)} aria-label={t('ckpt.release_name')} />
              <RPopover.Close asChild>
                <button className="btn btn-primary btn-sm" onClick={publish}>
                  <Upload />
                  {t('ckpt.publish')}
                </button>
              </RPopover.Close>
              {ckpt.kind !== 'regular' && (
                <RPopover.Close asChild>
                  <button className="btn btn-ghost btn-sm" onClick={remove}>
                    <Trash2 />
                    {t('ckpt.delete')}
                  </button>
                </RPopover.Close>
              )}
            </div>
          )}
        </RPopover.Content>
      </RPopover.Portal>
    </RPopover.Root>
  )
}

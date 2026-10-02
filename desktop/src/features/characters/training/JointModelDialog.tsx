import clsx from 'clsx'
import { Stars } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../../../api/client'
import { useCharacters } from '../../../api/hooks'
import type { Character, Model } from '../../../api/types'
import { Avatar } from '../../../components/Avatar'
import { Dialog } from '../../../components/ui'
import { useT } from '../../../i18n'
import { attempt } from '../../../lib/actions'

/** Pick characters to train together as one multi-speaker model; material stays separate. */
export function JointModelDialog({ character, open, onOpenChange, onCreated }: { character: Character; open: boolean; onOpenChange: (o: boolean) => void; onCreated: (m: Model) => void }) {
  const t = useT()
  const { data: characters } = useCharacters()
  const [picked, setPicked] = useState<string[]>([])
  const [name, setName] = useState('')
  const candidates = (characters ?? []).filter((c) => c.id !== character.id && c.health && c.health.level !== 'blocked')

  useEffect(() => {
    if (open) {
      setPicked([])
      setName('')
    }
  }, [open])

  const names = [character, ...candidates.filter((c) => picked.includes(c.id))].map((c) => c.name)

  const create = async () => {
    const m = await attempt(() => api.post<Model>('/models', { name: name.trim() || names.join(' · '), characterIds: [character.id, ...picked] }))
    if (m) {
      onCreated(m)
      onOpenChange(false)
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={t('train.const.title')}
      description={t('train.const.body')}
      actions={
        <>
          <button className="btn btn-ghost" onClick={() => onOpenChange(false)}>
            {t('common.cancel')}
          </button>
          <button className="btn btn-primary" disabled={picked.length === 0} onClick={create}>
            <Stars />
            {t('train.const.create')}
          </button>
        </>
      }
    >
      <div className="col" style={{ gap: 14 }}>
        <span className="field-label">{t('train.const.pick')}</span>
        {candidates.length === 0 ? (
          <span className="muted">{t('train.const.none')}</span>
        ) : (
          <div className="const-grid">
            <div className="const-item on fixed">
              <Avatar character={character} size={40} />
              <span>{character.name}</span>
            </div>
            {candidates.map((c) => (
              <button
                key={c.id}
                className={clsx('const-item', picked.includes(c.id) && 'on')}
                onClick={() => setPicked(picked.includes(c.id) ? picked.filter((x) => x !== c.id) : [...picked, c.id])}
              >
                <Avatar character={c} size={40} />
                <span>{c.name}</span>
              </button>
            ))}
          </div>
        )}
        <label className="field">
          <span className="field-label">{t('train.const.name')}</span>
          <input className="input" value={name} placeholder={names.join(' · ')} onChange={(e) => setName(e.target.value)} />
        </label>
      </div>
    </Dialog>
  )
}

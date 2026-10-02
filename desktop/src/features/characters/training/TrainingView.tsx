import clsx from 'clsx'
import { Cpu, Package, Plus, Users } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, sendFiles } from '../../../api/client'
import { useCharacters, useModels, useSystem } from '../../../api/hooks'
import type { Character, Job, Model } from '../../../api/types'
import { useT } from '../../../i18n'
import { attempt } from '../../../lib/actions'
import { pickFiles } from '../../../shell/bridge'
import { JointModelDialog } from './JointModelDialog'
import { TrainDashboard } from './TrainDashboard'
import { TrainSetup } from './TrainSetup'
import './training.css'

export function modelLabel(m: Model, characters: Character[] | undefined, t: ReturnType<typeof useT>): string {
  if (m.kind === 'single') return t('train.model.single')
  const names = m.characters.map((x) => characters?.find((c) => c.id === x.characterId)?.name ?? '?').join(' · ')
  return t('train.model.constellation', { names })
}

export function hasStarted(m: Model): boolean {
  return m.step > 0 || m.checkpoints.length > 0 || ['preparing', 'preprocessing', 'training', 'stopping'].includes(m.status)
}

export async function importPack() {
  const picked = await pickFiles({ multiple: false, extensions: ['stella'] })
  if (!picked || picked.length === 0) return
  await attempt(() => sendFiles<Job>('/packs/import', picked.slice(0, 1) as string[] | File[]))
}

export function TrainingView({ character }: { character: Character }) {
  const t = useT()
  const { data: system } = useSystem()
  const { data: models } = useModels(character.id)
  const { data: characters } = useCharacters()
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [constOpen, setConstOpen] = useState(false)

  const list = (models ?? []).slice().sort((a, b) => (a.kind === 'single' ? -1 : 1) - (b.kind === 'single' ? -1 : 1))
  const selected = list.find((m) => m.id === selectedId) ?? list.find((m) => m.id === character.primaryModelId) ?? list[0]

  useEffect(() => {
    if (selected && selected.id !== selectedId) setSelectedId(selected.id)
  }, [selected, selectedId])

  if (system && !system.canTrain && !(selected && hasStarted(selected))) {
    return (
      <div className="train-cannot glass">
        <div className="empty-icon">
          <Cpu />
        </div>
        <h2>{t('train.cannot.title')}</h2>
        <p className="secondary">{system.trainBlockers.map((b) => t.msg(b)).join(' ') || t.msg({ code: 'train.need_nvidia' })}</p>
        <button className="btn btn-primary" onClick={importPack}>
          <Package />
          {t('train.cannot.import')}
        </button>
      </div>
    )
  }

  return (
    <div className="training">
      {list.length > 0 && (
        <div className="train-models">
          {list.map((m) => (
            <button key={m.id} className={clsx('chip button train-model-chip', m.id === selected?.id && 'accent')} onClick={() => setSelectedId(m.id)}>
              {m.kind === 'constellation' && <Users />}
              {modelLabel(m, characters, t)}
            </button>
          ))}
          <button className="chip button" onClick={() => setConstOpen(true)}>
            <Plus />
            {t('train.model.new_constellation')}
          </button>
        </div>
      )}

      {selected && hasStarted(selected) ? (
        <TrainDashboard model={selected} character={character} />
      ) : (
        <TrainSetup character={character} model={selected ?? null} />
      )}

      <JointModelDialog
        character={character}
        open={constOpen}
        onOpenChange={setConstOpen}
        onCreated={(m) => {
          setSelectedId(m.id)
          void attempt(() => api.get(`/models/${m.id}`))
        }}
      />
    </div>
  )
}

import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '../../../api/client'
import { keys } from '../../../api/keys'
import type { Model, TrainConfig } from '../../../api/types'
import { Dialog } from '../../../components/ui'
import { useT } from '../../../i18n'
import { attempt } from '../../../lib/actions'
import { TrainParams } from './TrainParams'

/** Change training parameters between runs; the network shape is fixed once checkpoints exist. */
export function TrainParamsDialog({ model, running, onClose }: { model: Model; running: boolean; onClose: () => void }) {
  const t = useT()
  const qc = useQueryClient()
  const [config, setConfig] = useState<TrainConfig>(model.config)
  const locked = model.checkpoints.length > 0
  const changed = (Object.keys(config) as (keyof TrainConfig)[]).filter((k) => JSON.stringify(config[k]) !== JSON.stringify(model.config[k]))

  const save = async () => {
    const patch = Object.fromEntries(changed.map((k) => [k, config[k]]))
    const m = await attempt(() => api.patch<Model>(`/models/${model.id}`, { config: patch }))
    if (m) {
      qc.setQueryData(keys.model(model.id), m)
      qc.invalidateQueries({ queryKey: ['models'] })
      onClose()
    }
  }

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title={t('tp.title')}
      description={running ? t('tp.pause_first') : locked ? t('tp.locked_trained') : undefined}
      wide
      actions={
        <>
          <button className="btn btn-ghost" onClick={onClose}>
            {t('common.cancel')}
          </button>
          <button className="btn btn-primary" onClick={save} disabled={running || changed.length === 0}>
            {t('common.save')}
          </button>
        </>
      }
    >
      <fieldset className="tp-fieldset" disabled={running}>
        <TrainParams config={config} onChange={setConfig} locked={locked} />
      </fieldset>
    </Dialog>
  )
}

import { useQueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'
import { api } from '../api/client'
import { keys } from '../api/keys'
import { onServerEvent } from '../api/live'
import type { Character, Model, Release } from '../api/types'
import { translate, useLang } from '../i18n'
import { steps } from '../lib/format'

/**
 * When a training run reaches its target and the model has no published version yet,
 * publish the final checkpoint so the voice is immediately usable for creating.
 */
export function useAutoRelease() {
  const qc = useQueryClient()
  useEffect(
    () =>
      onServerEvent(async (ev) => {
        if (ev.type !== 'job' || ev.data.type !== 'train' || ev.data.status !== 'succeeded') return
        if (ev.data.result?.paused === true || !ev.data.target.modelId) return
        const model = await api.get<Model>(`/models/${ev.data.target.modelId}`).catch(() => null)
        if (!model || model.releases.length > 0 || model.step < model.targetSteps) return
        const last = model.checkpoints.reduce<number | null>((best, c) => (best == null || c.step > best ? c.step : best), null)
        if (last == null) return
        const character = qc.getQueryData<Character>(keys.character(model.characters[0]?.characterId ?? ''))
        const name = translate(useLang.getState().lang, 'ckpt.release_default', { name: character?.name ?? model.name, steps: steps(last) })
        await api.post<Release>(`/models/${model.id}/releases`, { step: last, name }).catch(() => null)
      }),
    [qc],
  )
}

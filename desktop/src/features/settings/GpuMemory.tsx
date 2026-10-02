import { useQueryClient } from '@tanstack/react-query'
import { Loader2 } from 'lucide-react'
import { useState } from 'react'
import { api } from '../../api/client'
import { usePreferences, useRenderStatus } from '../../api/hooks'
import { keys } from '../../api/keys'
import type { RenderStatus } from '../../api/types'
import { Segmented } from '../../components/ui'
import { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { toast } from '../../state/ui'
import { setPreference } from '../resources/ResourcesView'
import { Row } from './parts'

const IDLE_OPTIONS = [5, 10, 30, 60, 0] as const

/** Frees the GPU (or memory) the singing models hold; true when something was freed. */
export async function releaseRenderMemory(qc: ReturnType<typeof useQueryClient>): Promise<boolean> {
  const out = await api.post<RenderStatus & { released: boolean }>('/system/render/release')
  const { released, ...status } = out
  qc.setQueryData<RenderStatus>(keys.renderStatus, status)
  return released
}

/** "Training & GPU": when the singing models leave the GPU, and freeing it now. */
export function GpuMemoryRows() {
  const t = useT()
  const qc = useQueryClient()
  const { data: prefs } = usePreferences()
  const { data: status } = useRenderStatus()
  const [busy, setBusy] = useState(false)
  if (!prefs) return null

  const device = status?.device === 'cuda' ? t('set.gpu_mem.on_gpu') : status?.device === 'mps' ? t('set.gpu_mem.on_mps') : t('set.gpu_mem.on_cpu')
  const state = !status?.loaded
    ? t('set.gpu_mem.not_loaded')
    : status.busy
      ? t('set.gpu_mem.in_use', { where: device })
      : t('set.gpu_mem.loaded', { where: device, minutes: Math.floor((status.idleSeconds ?? 0) / 60) })

  const release = () =>
    attempt(async () => {
      setBusy(true)
      try {
        const released = await releaseRenderMemory(qc)
        toast({ kind: released ? 'success' : 'info', title: released ? t('set.gpu_mem.released') : t('set.gpu_mem.busy') })
      } finally {
        setBusy(false)
      }
    })

  return (
    <>
      <Row label={t('set.gpu_mem.idle')} hint={t('set.gpu_mem.idle_hint')}>
        <Segmented<string>
          size="sm"
          value={String(prefs.gpuIdleReleaseMin)}
          onChange={(v) => setPreference(qc, { gpuIdleReleaseMin: Number(v) })}
          options={IDLE_OPTIONS.map((m) => ({
            value: String(m),
            label: m === 0 ? t('set.gpu_mem.never') : m === 60 ? t('set.gpu_mem.hour') : t('set.gpu_mem.minutes', { n: m }),
          }))}
        />
      </Row>
      <Row label={t('set.gpu_mem.now')} hint={state}>
        <button className="btn btn-sm" disabled={busy || !status?.loaded || status.busy} onClick={() => void release()}>
          {busy && <Loader2 className="spin" />}
          {t('set.gpu_mem.release')}
        </button>
      </Row>
    </>
  )
}

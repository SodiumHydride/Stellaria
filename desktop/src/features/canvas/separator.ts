import { api } from '../../api/client'
import type { Job, Resource } from '../../api/types'
import { confirmDialog } from '../../components/ui'
import type { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { bytes } from '../../lib/format'
import { waitForJob } from '../../lib/jobs'
import { toast } from '../../state/ui'

export const SEPARATOR = 'separator'

export type SeparatorState = 'ready' | 'installing' | 'missing'

export function separatorState(resources: Resource[] | undefined): SeparatorState {
  const r = resources?.find((x) => x.id === SEPARATOR)
  if (!r) return 'missing'
  if (r.status === 'ready') return 'ready'
  return r.status === 'downloading' || r.status === 'installing' ? 'installing' : 'missing'
}

/**
 * Makes sure the separation add-on is there before splitting a song: offers to install it,
 * waits for the install and resolves true once separation can start.
 */
export async function ensureSeparator(resources: Resource[] | undefined, t: ReturnType<typeof useT>): Promise<boolean> {
  const state = separatorState(resources)
  if (state === 'ready') return true
  const r = resources?.find((x) => x.id === SEPARATOR)
  if (state === 'missing') {
    const ok = await confirmDialog({
      title: t('sep.install_title'),
      body: t('sep.install_body', { size: bytes(r?.size ?? null) }),
      confirm: t('sep.install_go'),
      cancel: t('common.cancel'),
    })
    if (!ok) return false
  }
  const job = await attempt(() => api.post<Job>(`/resources/${SEPARATOR}/install`, {}))
  if (!job) return false
  toast({ kind: 'info', title: t('sep.installing') })
  const done = await waitForJob(job.id).catch(() => null)
  if (done?.status === 'succeeded') return true
  toast({ kind: 'error', title: t('sep.install_failed'), body: done?.error ? t.msg({ code: done.error.code, params: done.error.params }, done.error.message) : undefined })
  return false
}

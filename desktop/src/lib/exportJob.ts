import { mediaUrl } from '../api/client'
import type { Job } from '../api/types'
import { toast } from '../state/ui'
import { attempt } from './actions'
import { waitForJob } from './jobs'

/** Waits for an export job; in the browser opens its download, in the app confirms where it went. */
export async function deliverExport(job: Job | undefined, labels: { done: string; failed: string }) {
  if (!job) return
  const done = await attempt(() => waitForJob(job.id))
  if (!done) return
  if (done.status !== 'succeeded') {
    if (done.status === 'failed') toast({ kind: 'error', title: labels.failed, detail: done.error?.detail ?? done.error?.message })
    return
  }
  const result = done.result as { path?: string; downloadUrl?: string } | null
  if (result?.downloadUrl && !result.path) window.open(mediaUrl(result.downloadUrl))
  else toast({ kind: 'success', title: labels.done, body: result?.path })
}

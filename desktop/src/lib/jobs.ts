import { ApiError, api } from '../api/client'
import { onServerEvent, useLive } from '../api/live'
import type { Job } from '../api/types'
import { translate, useLang } from '../i18n'

const FINAL = new Set(['succeeded', 'failed', 'cancelled'])
/** how often to ask over HTTP while the live connection is down, and (as a safety net) while it is up */
const POLL_OFFLINE_MS = 2000
const POLL_ONLINE_MS = 10_000

export interface WaitOptions {
  /** stop waiting: rejects with the signal's reason (an AbortError unless given another) */
  signal?: AbortSignal
  /** give up after this many milliseconds: rejects with ApiError `job.wait_timeout` */
  timeoutMs?: number
}

/**
 * Resolves with the job once it has finished (succeeded, failed or cancelled). Live events
 * answer first; the job is also asked for over HTTP right away, after a reconnect (events sent
 * while the socket was down are gone), every few seconds while the connection is down and now
 * and then as a safety net. Rejects (ApiError `job.lost`) when the backend no longer knows the
 * job, which happens when it restarted, so it will never finish; also on timeout or abort.
 */
export function waitForJob(jobId: string, opts: WaitOptions = {}): Promise<Job> {
  const { signal, timeoutMs } = opts
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortReason(signal))
      return
    }
    let settled = false
    let poll: number | undefined
    let deadline: number | undefined

    const settle = (done: () => void) => {
      if (settled) return
      settled = true
      offEvents()
      offLive()
      window.clearTimeout(poll)
      window.clearTimeout(deadline)
      signal?.removeEventListener('abort', onAbort)
      done()
    }
    const finish = (job: Job) => settle(() => resolve(job))
    const fail = (error: unknown) => settle(() => reject(error))
    const onAbort = () => fail(abortReason(signal!))

    const check = () =>
      api.get<Job>(`/jobs/${jobId}`).then(
        (job) => {
          if (FINAL.has(job.status)) finish(job)
        },
        (e) => {
          if (e instanceof ApiError && e.status === 404) fail(waitError('lost', jobId))
          // unreachable or not connected yet: the next poll asks again
        },
      )
    const schedule = () => {
      window.clearTimeout(poll)
      if (settled) return
      poll = window.setTimeout(() => void check().then(schedule), useLive.getState().connected ? POLL_ONLINE_MS : POLL_OFFLINE_MS)
    }

    const offEvents = onServerEvent((ev) => {
      if (ev.type === 'job' && ev.data.id === jobId && FINAL.has(ev.data.status)) finish(ev.data)
    })
    const offLive = useLive.subscribe((s, prev) => {
      if (s.connected === prev.connected) return
      if (s.connected) void check()
      schedule()
    })
    signal?.addEventListener('abort', onAbort, { once: true })
    if (timeoutMs !== undefined) deadline = window.setTimeout(() => fail(waitError('timeout', jobId)), timeoutMs)
    void check().then(schedule)
  })
}

function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException('Aborted', 'AbortError')
}

function waitError(kind: 'lost' | 'timeout', jobId: string): ApiError {
  const lang = useLang.getState().lang
  return kind === 'lost'
    ? new ApiError(404, 'job.lost', translate(lang, 'jobs.lost'), `job ${jobId}`)
    : new ApiError(0, 'job.wait_timeout', translate(lang, 'jobs.wait_timeout'), `job ${jobId}`)
}

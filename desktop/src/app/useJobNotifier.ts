import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef } from 'react'
import { keys } from '../api/keys'
import { onServerEvent } from '../api/live'
import type { Job, JobType, Preferences } from '../api/types'
import { isActiveJob } from '../api/types'
import { translate, translateMsg, useLang } from '../i18n'
import { notify } from '../shell/bridge'
import { toast } from '../state/ui'

const ANNOUNCE: JobType[] = ['organize', 'train', 'render', 'separate', 'mixdown', 'download', 'install_addon', 'import_pack', 'export', 'slim', 'import']
const SYSTEM_NOTIFY: JobType[] = ['organize', 'train', 'mixdown', 'download', 'install_addon', 'separate']

/** Toasts (and OS notifications when the window is in the background) when jobs finish. */
export function useJobNotifier() {
  const qc = useQueryClient()
  const seen = useRef(new Map<string, Job['status']>())

  useEffect(() => {
    for (const j of qc.getQueryData<Job[]>(keys.jobs) ?? []) seen.current.set(j.id, j.status)
    return onServerEvent((ev) => {
      if (ev.type !== 'job') return
      const job = ev.data
      const before = seen.current.get(job.id)
      seen.current.set(job.id, job.status)
      if (before === undefined && !isActiveJob(job)) return
      if (before !== undefined && !isActiveJob({ ...job, status: before })) return
      if (isActiveJob(job) || job.status === 'cancelled') return
      if (!ANNOUNCE.includes(job.type)) return

      const lang = useLang.getState().lang
      const title = translateMsg(lang, job.title, job.type)
      const paused = job.type === 'train' && job.result?.paused === true
      if (job.status === 'succeeded') {
        if (paused) return
        toast({ kind: 'success', title: translate(lang, 'toast.job_done', { title }) })
      } else if (job.status === 'failed') {
        toast({
          kind: 'error',
          title: translate(lang, 'toast.job_failed', { title }),
          body: job.error ? translateMsg(lang, { code: job.error.code, params: job.error.params }, job.error.message) : undefined,
          detail: job.error?.detail || undefined,
        })
      }
      const prefs = qc.getQueryData<Preferences>(keys.preferences)
      if ((prefs?.notifyOnFinish ?? true) && SYSTEM_NOTIFY.includes(job.type) && (document.hidden || !document.hasFocus())) {
        void notify(translate(lang, 'app.name'), translate(lang, job.status === 'succeeded' ? 'toast.job_done' : 'toast.job_failed', { title }))
      }
    })
  }, [qc])
}

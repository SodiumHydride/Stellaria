import { useQueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'
import { api } from '../api/client'
import { keys } from '../api/keys'
import type { Job } from '../api/types'
import { confirmDialog } from '../components/ui'
import { releaseRenderMemory } from '../features/settings/GpuMemory'
import { translate, useLang } from '../i18n'
import { isTauri, listen, shell, windowAction } from '../shell/bridge'
import { toast } from '../state/ui'
import { runBeforeQuit } from './beforeQuit'

/** Quit once the edits still waiting to be saved are with the backend. */
async function saveAndQuit() {
  await runBeforeQuit()
  await shell.quitApp(false)
}

/**
 * The shell asks before closing: with work in progress the window hides to the tray
 * (training keeps running) unless the user chooses to pause and quit.
 */
export function useCloseRequests() {
  const qc = useQueryClient()
  useEffect(() => {
    if (!isTauri) return
    const offs: (() => void)[] = []
    void listen<{ activeJobs: boolean }>('app://close-requested', async ({ activeJobs }) => {
      if (!activeJobs) {
        await saveAndQuit()
        return
      }
      const lang = useLang.getState().lang
      const hide = await confirmDialog({
        title: translate(lang, 'close.title'),
        body: translate(lang, 'close.body'),
        confirm: translate(lang, 'close.hide'),
        cancel: translate(lang, 'close.quit'),
      })
      if (hide) await windowAction('hide')
      else await saveAndQuit()
    }).then((f) => offs.push(f))
    void listen<{ action: string }>('tray://action', async ({ action }) => {
      if (action === 'quit') {
        // the shell waits a few seconds for this before quitting on its own
        await saveAndQuit()
        return
      }
      if (action === 'release_memory') {
        const lang = useLang.getState().lang
        const released = await releaseRenderMemory(qc).catch(() => false)
        toast({ kind: released ? 'success' : 'info', title: translate(lang, released ? 'set.gpu_mem.released' : 'set.gpu_mem.nothing') })
        return
      }
      if (action !== 'pause_training') return
      const jobs = qc.getQueryData<Job[]>(keys.jobs) ?? []
      for (const j of jobs) {
        if (j.type === 'train' && j.status === 'running' && j.target.modelId) await api.post(`/models/${j.target.modelId}/pause`).catch(() => {})
      }
    }).then((f) => offs.push(f))
    return () => offs.forEach((f) => f())
  }, [qc])
}

/**
 * Close requests before the main window exists (setup, connecting): nothing runs in the
 * background yet, so the window quits — after asking while the runtime is being installed.
 */
export function useSetupCloseRequests() {
  useEffect(() => {
    if (!isTauri) return
    const offs: (() => void)[] = []
    let gone = false
    const keep = (f: () => void) => (gone ? f() : offs.push(f))
    // nothing to save yet: the tray's Quit needn't wait
    void listen<{ action: string }>('tray://action', async ({ action }) => {
      if (action === 'quit') await shell.quitApp(false)
    }).then(keep)
    void listen<{ activeJobs: boolean }>('app://close-requested', async ({ activeJobs }) => {
      if (activeJobs) {
        const lang = useLang.getState().lang
        const quit = await confirmDialog({
          title: translate(lang, 'close.setup_title'),
          body: translate(lang, 'close.setup_body'),
          confirm: translate(lang, 'close.setup_quit'),
          cancel: translate(lang, 'close.setup_stay'),
        })
        if (!quit) return
      }
      await shell.quitApp(false)
    }).then(keep)
    return () => {
      gone = true
      offs.forEach((f) => f())
    }
  }, [])
}

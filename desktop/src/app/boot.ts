import { create } from 'zustand'
import { api, useConnection } from '../api/client'
import type { Resource } from '../api/types'
import { useLang } from '../i18n'
import { browserBackend, isTauri, listen, shell, type BackendStatus, type ShellState } from '../shell/bridge'

export type BootPhase = 'init' | 'onboarding' | 'connecting' | 'ready' | 'failed'

/** Set when the user chose to get the required models later: starting up then doesn't insist. */
export const MODELS_LATER_KEY = 'stellaria.models.later'

function modelsLater(): boolean {
  try {
    return localStorage.getItem(MODELS_LATER_KEY) === '1'
  } catch {
    return false
  }
}

interface BootState {
  phase: BootPhase
  shell: ShellState | null
  backend: BackendStatus | null
  error: string | null
  /** required models are missing (onboarding insists unless the user chose "later") */
  missingModels: boolean
  /** onboarding opens at choosing the runtime (the user's Python broke and they want another) */
  runtimeSetup: boolean
  setShell: (s: ShellState) => void
  start: () => Promise<void>
  connect: () => Promise<void>
  retry: () => Promise<void>
  finishOnboarding: () => Promise<void>
  /** back to choosing the runtime: the app's own, or another Python */
  chooseRuntime: () => void
}

async function waitForHealth(url: string, tries = 40): Promise<boolean> {
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url + '/api/health')
      if (res.ok) return true
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 500))
  }
  return false
}

async function requiredModelsMissing(): Promise<boolean> {
  try {
    const list = await api.get<Resource[]>('/resources')
    return list.some((r) => r.required && r.status !== 'ready')
  } catch {
    return false
  }
}

let unlistenBackend: (() => void) | null = null

export const useBoot = create<BootState>((set, get) => ({
  phase: 'init',
  shell: null,
  backend: null,
  error: null,
  missingModels: false,
  runtimeSetup: false,

  setShell: (s) => set({ shell: s }),

  start: async () => {
    if (!isTauri) {
      await get().connect()
      return
    }
    try {
      const state = await shell.getState()
      set({ shell: state })
      if (state.language) useLang.getState().setLang(state.language)
      if (!unlistenBackend) {
        unlistenBackend = await listen<BackendStatus>('backend://status', (b) => {
          set({ backend: b })
          if (b.status === 'ready' && b.url && b.token) {
            useConnection.getState().setBackend(b.url, b.token)
          } else if (b.status === 'crashed' && get().phase !== 'onboarding') {
            set({ phase: 'failed', error: b.error })
          }
        })
      }
      if (state.firstRunDone && state.runtime.mode === 'custom' && state.runtime.status === 'broken') {
        // their own Python changed since: say what happened and how to get going again
        set({ phase: 'failed', error: 'runtime.custom_broken' })
        return
      }
      if (!state.firstRunDone || state.runtime.status !== 'ready') {
        set({ phase: 'onboarding' })
        return
      }
      await get().connect()
    } catch (e) {
      set({ phase: 'failed', error: String(e) })
    }
  },

  /** Make sure the backend is running and reachable, then decide ready vs onboarding. */
  connect: async () => {
    set({ phase: get().phase === 'onboarding' ? 'onboarding' : 'connecting', error: null })
    let url: string
    let token: string
    if (isTauri) {
      let status = get().backend ?? (await shell.getState()).backend
      if (status.status !== 'ready') status = await shell.backendStart()
      const deadline = Date.now() + 90_000
      while (status.status !== 'ready' && Date.now() < deadline) {
        if (status.status === 'crashed') break
        await new Promise((r) => setTimeout(r, 400))
        status = get().backend ?? status
      }
      if (status.status !== 'ready' || !status.url || !status.token) {
        set({ phase: 'failed', error: status.error ?? 'backend did not start' })
        return
      }
      url = status.url
      token = status.token
    } else {
      ;({ url, token } = browserBackend())
    }
    if (!(await waitForHealth(url, isTauri ? 20 : 6))) {
      set({ phase: 'failed', error: `no response from ${url}` })
      return
    }
    useConnection.getState().setBackend(url, token)
    const missing = await requiredModelsMissing()
    set({ missingModels: missing, phase: missing && !modelsLater() ? 'onboarding' : 'ready' })
  },

  retry: async () => {
    if (isTauri) {
      try {
        set({ backend: await shell.backendRestart() })
      } catch (e) {
        set({ error: String(e) })
      }
    }
    await get().connect()
  },

  chooseRuntime: () => set({ phase: 'onboarding', runtimeSetup: true, error: null }),

  finishOnboarding: async () => {
    if (isTauri) set({ shell: await shell.finishOnboarding() })
    set({ phase: 'ready', missingModels: false, runtimeSetup: false })
  },
}))

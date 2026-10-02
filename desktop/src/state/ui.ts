import { create } from 'zustand'

export interface Toast {
  id: number
  kind: 'info' | 'success' | 'error'
  title: string
  body?: string
  detail?: string
  action?: { label: string; run: () => void }
}

interface UiState {
  paletteOpen: boolean
  dockOpen: boolean
  shortcutsOpen: boolean
  toasts: Toast[]
  /** true while a training job runs and the user asked for low-power effects */
  lowPower: boolean
  setPalette: (open: boolean) => void
  setDock: (open: boolean) => void
  setShortcuts: (open: boolean) => void
  setLowPower: (on: boolean) => void
  toast: (t: Omit<Toast, 'id'>, ttl?: number) => number
  dismiss: (id: number) => void
}

let nextId = 1

export const useUi = create<UiState>((set, get) => ({
  paletteOpen: false,
  dockOpen: false,
  shortcutsOpen: false,
  toasts: [],
  lowPower: false,
  setPalette: (paletteOpen) => set({ paletteOpen }),
  setDock: (dockOpen) => set({ dockOpen }),
  setShortcuts: (shortcutsOpen) => set({ shortcutsOpen }),
  setLowPower: (lowPower) => set({ lowPower }),
  toast: (t, ttl = t.kind === 'error' ? 9000 : 4200) => {
    const id = nextId++
    set({ toasts: [...get().toasts.slice(-3), { ...t, id }] })
    if (ttl > 0) window.setTimeout(() => get().dismiss(id), ttl)
    return id
  },
  dismiss: (id) => set({ toasts: get().toasts.filter((x) => x.id !== id) }),
}))

export const toast = (t: Omit<Toast, 'id'>, ttl?: number) => useUi.getState().toast(t, ttl)

import { create } from 'zustand'

export type ThemePref = 'system' | 'light' | 'dark'
const KEY = 'stellaria.theme'
const media = typeof window !== 'undefined' ? window.matchMedia('(prefers-color-scheme: dark)') : null

function readPref(): ThemePref {
  try {
    const v = localStorage.getItem(KEY)
    if (v === 'light' || v === 'dark' || v === 'system') return v
  } catch {
    /* storage unavailable: fall back to the system theme */
  }
  return 'system'
}

function resolve(pref: ThemePref): 'light' | 'dark' {
  return pref === 'dark' || (pref === 'system' && !!media?.matches) ? 'dark' : 'light'
}

function apply(pref: ThemePref) {
  const resolved = resolve(pref)
  document.documentElement.dataset.theme = resolved
  if (useTheme.getState().resolved !== resolved) useTheme.setState({ resolved })
}

interface ThemeState {
  pref: ThemePref
  /** what is on screen; canvases that paint colors redraw when it changes */
  resolved: 'light' | 'dark'
  setPref: (pref: ThemePref) => void
}

/** Theme lives on the device (it must apply before the backend is reachable). */
export const useTheme = create<ThemeState>((set) => ({
  pref: readPref(),
  resolved: resolve(readPref()),
  setPref: (pref) => {
    try {
      localStorage.setItem(KEY, pref)
    } catch {
      /* not persisted; still applies for this session */
    }
    apply(pref)
    set({ pref })
  },
}))

export function initTheme() {
  apply(useTheme.getState().pref)
  media?.addEventListener('change', () => apply(useTheme.getState().pref))
}

export const isDark = () => document.documentElement.dataset.theme === 'dark'

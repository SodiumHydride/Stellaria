import { create } from 'zustand'
import type { Msg } from '../api/types'
import { strings, type StringKey } from './strings'
import { messages } from './messages'

export type Lang = 'zh-CN' | 'en'

function initialLang(): Lang {
  try {
    const saved = localStorage.getItem('stellaria.lang')
    if (saved === 'zh-CN' || saved === 'en') return saved
  } catch {
    /* storage unavailable */
  }
  return navigator.language.toLowerCase().startsWith('zh') ? 'zh-CN' : 'en'
}

export const useLang = create<{ lang: Lang; setLang: (lang: Lang) => void }>((set) => ({
  lang: initialLang(),
  setLang: (lang) => {
    try {
      localStorage.setItem('stellaria.lang', lang)
    } catch {
      /* storage unavailable */
    }
    document.documentElement.lang = lang
    set({ lang })
  },
}))

type Params = Record<string, string | number>

function interpolate(text: string, params?: Params): string {
  if (!params) return text
  return text.replace(/\{(\w+)\}/g, (_, k) => (params[k] !== undefined ? String(params[k]) : `{${k}}`))
}

export function translate(lang: Lang, key: StringKey, params?: Params): string {
  const entry = strings[key]
  return interpolate(lang === 'zh-CN' ? entry[0] : entry[1], params)
}

/** Localize a backend message ({code, params}); unknown codes fall back to the code itself. */
export function translateMsg(lang: Lang, msg: Msg | null | undefined, fallback?: string): string {
  if (!msg) return fallback ?? ''
  // a stable `reason` (or `resource`) param picks a more specific message when there is one
  // (dsp.bad_params.trim.range, addon.missing.separator)
  const reason = msg.params?.reason ?? msg.params?.resource
  const entry = (typeof reason === 'string' && messages[`${msg.code}.${reason}`]) || messages[msg.code]
  if (!entry) {
    // the backend may ship its own translations inline: {code, params: {'zh-CN': …, en: …}}
    const inline = msg.params?.[lang] ?? msg.params?.[lang === 'zh-CN' ? 'en' : 'zh-CN']
    if (typeof inline === 'string') return inline
    return fallback ?? msg.code
  }
  return interpolate(lang === 'zh-CN' ? entry[0] : entry[1], msg.params)
}

export function useT() {
  const lang = useLang((s) => s.lang)
  const t = (key: StringKey, params?: Params) => translate(lang, key, params)
  t.msg = (msg: Msg | null | undefined, fallback?: string) => translateMsg(lang, msg, fallback)
  /** Labels that may be plain text (plugin manifests) or a translatable Msg (built-in specs). */
  t.label = (x: Msg | string | null | undefined) => (typeof x === 'string' ? x : translateMsg(lang, x))
  t.lang = lang
  return t
}

export type { StringKey }

import type { Lang } from '../i18n'

/** 83.4 → "1:23"; 3723 → "1:02:03" */
export function clock(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds)) return '–:––'
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}` : `${m}:${String(sec).padStart(2, '0')}`
}

/** precise clock for editors: "1:23.4" */
export function clockPrecise(seconds: number): string {
  const s = Math.max(0, seconds)
  const m = Math.floor(s / 60)
  const rest = s - m * 60
  return `${m}:${rest.toFixed(1).padStart(4, '0')}`
}

/** Human duration: "42 分钟", "1 小时 5 分", "35 秒" */
export function duration(seconds: number | null | undefined, lang: Lang): string {
  if (seconds == null || !isFinite(seconds)) return '–'
  const s = Math.round(seconds)
  const zh = lang === 'zh-CN'
  if (s < 60) return zh ? `${s} 秒` : `${s} s`
  const m = Math.round(s / 60)
  if (m < 60) return zh ? `${m} 分钟` : `${m} min`
  const h = Math.floor(s / 3600)
  const mm = Math.round((s % 3600) / 60)
  if (mm === 0) return zh ? `${h} 小时` : `${h} h`
  return zh ? `${h} 小时 ${mm} 分` : `${h} h ${mm} min`
}

export function bytes(n: number | null | undefined): string {
  if (n == null || !isFinite(n)) return '–'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`
}

export function speed(bytesPerSec: number | null | undefined): string {
  if (!bytesPerSec) return ''
  return `${bytes(bytesPerSec)}/s`
}

/** 32000 → "32k", 1250 → "1.3k" */
export function steps(n: number): string {
  if (n < 1000) return String(n)
  const k = n / 1000
  return `${k >= 100 || Number.isInteger(k) ? Math.round(k) : k.toFixed(1)}k`
}

export function number(n: number, digits = 0): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

const NOTE_NAMES = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']

export function hzToMidi(hz: number): number {
  return 69 + 12 * Math.log2(hz / 440)
}

export function midiToNote(midi: number): string {
  const m = Math.round(midi)
  return `${NOTE_NAMES[((m % 12) + 12) % 12]}${Math.floor(m / 12) - 1}`
}

export function hzToNote(hz: number | null | undefined): string {
  if (!hz || hz <= 0) return '–'
  return midiToNote(hzToMidi(hz))
}

export function isBlackKey(midi: number): boolean {
  return [1, 3, 6, 8, 10].includes(((Math.round(midi) % 12) + 12) % 12)
}

export function relativeTime(iso: string | null | undefined, lang: Lang): string {
  if (!iso) return ''
  const then = new Date(iso).getTime()
  const diff = (Date.now() - then) / 1000
  const zh = lang === 'zh-CN'
  if (diff < 45) return zh ? '刚刚' : 'just now'
  if (diff < 3600) {
    const m = Math.round(diff / 60)
    return zh ? `${m} 分钟前` : `${m} min ago`
  }
  if (diff < 86400) {
    const h = Math.round(diff / 3600)
    return zh ? `${h} 小时前` : `${h} h ago`
  }
  if (diff < 86400 * 7) {
    const d = Math.round(diff / 86400)
    return zh ? `${d} 天前` : `${d} d ago`
  }
  return new Date(iso).toLocaleDateString(lang)
}

export function signed(n: number, digits = 0): string {
  const v = digits ? n.toFixed(digits) : String(Math.round(n))
  return n > 0 ? `+${v}` : v
}

export function percent(p: number | null | undefined): string {
  if (p == null) return ''
  return `${Math.round(p * 100)}%`
}

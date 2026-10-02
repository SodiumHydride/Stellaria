/** Muted identity colors a character can wear (Morandi). Identity only — never status. */
export const VOICE_COLORS = [
  '#9aaabd', // 雾蓝
  '#c9a6a0', // 豆沙
  '#a7b59f', // 鼠尾草
  '#cdbb9f', // 燕麦
  '#b0a3bb', // 灰紫
  '#95b0ad', // 灰青
  '#d0adb9', // 藕粉
  '#bca48e', // 驼色
  '#b3b494', // 橄榄灰
  '#d5b39a', // 杏灰
]

export function rgba(hex: string, alpha: number): string {
  const h = hex.replace('#', '')
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h
  const n = parseInt(full, 16)
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`
}

/** Deterministic 0..1 values from a string (stable colors). */
export function hashUnit(input: string, salt = 0): number {
  let h = 2166136261 ^ salt
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  h ^= h >>> 13
  h = Math.imul(h, 0x5bd1e995)
  h ^= h >>> 15
  return (h >>> 0) / 4294967295
}

/** Project covers use the same refined palette, whatever the stored color is. */
export function coverColor(p: { id: string; coverColor?: string }): string {
  if (p.coverColor && VOICE_COLORS.includes(p.coverColor.toLowerCase())) return p.coverColor
  return VOICE_COLORS[Math.floor(hashUnit(p.id, 7) * VOICE_COLORS.length) % VOICE_COLORS.length]
}

export function pickVoiceColor(seed: string, used: string[] = []): string {
  const free = VOICE_COLORS.filter((c) => !used.includes(c))
  const pool = free.length > 0 ? free : VOICE_COLORS
  return pool[Math.floor(hashUnit(seed) * pool.length) % pool.length]
}

function hexRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h
  const n = parseInt(full, 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

/** Linear mix of two hex colors, t = share of b. */
export function mix(a: string, b: string, t: number): string {
  const x = hexRgb(a)
  const y = hexRgb(b)
  return '#' + x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, '0')).join('')
}

export function cssVar(name: string, fallback = '#888888'): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
}

const dark = () => document.documentElement.dataset.theme === 'dark'

/** Ink for drawing a voice's waveform: on paper the muted hue is deepened so bars stay legible. */
export function waveColor(hex?: string): string {
  const base = hex && hex.startsWith('#') ? hex : cssVar('--wave')
  return dark() ? base : mix(base, '#3a3530', 0.38)
}

type Stops = [number, [number, number, number]][]
/** Spectrogram colormaps with monotonic lightness: quiet = the surface, loud = ink (light) or paper (dark). */
const MEL_LIGHT: Stops = [
  [0, [246, 243, 238]],
  [0.3, [214, 205, 198]],
  [0.55, [160, 166, 182]],
  [0.78, [96, 106, 128]],
  [1, [40, 42, 52]],
]
const MEL_DARK: Stops = [
  [0, [27, 26, 24]],
  [0.3, [52, 56, 70]],
  [0.55, [104, 98, 112]],
  [0.78, [176, 150, 132]],
  [1, [246, 238, 224]],
]

function buildLut(stops: Stops): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 4)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    let k = 0
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++
    const [t0, c0] = stops[k]
    const [t1, c1] = stops[k + 1]
    const f = (t - t0) / (t1 - t0)
    lut[i * 4] = c0[0] + (c1[0] - c0[0]) * f
    lut[i * 4 + 1] = c0[1] + (c1[1] - c0[1]) * f
    lut[i * 4 + 2] = c0[2] + (c1[2] - c0[2]) * f
    lut[i * 4 + 3] = 255
  }
  return lut
}

const LUTS = { light: buildLut(MEL_LIGHT), dark: buildLut(MEL_DARK) }
export const melLut = () => (dark() ? LUTS.dark : LUTS.light)

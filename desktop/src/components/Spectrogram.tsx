import { useEffect, useRef } from 'react'
import { melLut } from '../lib/color'
import { useTheme } from '../state/theme'

function decode(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

/** Draws a quantized mel spectrogram (frames × bins, row-major) with the theme's colormap. */
export function Spectrogram({ data, frames, bins, height = 96 }: { data: string; frames: number; bins: number; height?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const theme = useTheme((s) => s.resolved)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !data || frames <= 0 || bins <= 0) return
    const values = decode(data)
    canvas.width = frames
    canvas.height = bins
    const ctx = canvas.getContext('2d')!
    const img = ctx.createImageData(frames, bins)
    const MEL_LUT = melLut()
    for (let f = 0; f < frames; f++) {
      for (let b = 0; b < bins; b++) {
        const v = values[f * bins + b] ?? 0
        const o = ((bins - 1 - b) * frames + f) * 4
        img.data[o] = MEL_LUT[v * 4]
        img.data[o + 1] = MEL_LUT[v * 4 + 1]
        img.data[o + 2] = MEL_LUT[v * 4 + 2]
        img.data[o + 3] = 255
      }
    }
    ctx.putImageData(img, 0, 0)
  }, [data, frames, bins, theme])
  return <canvas ref={ref} className="spectrogram" style={{ width: '100%', height, imageRendering: 'auto', borderRadius: 10, display: 'block' }} />
}

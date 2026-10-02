import type { RenderParams } from '../api/types'

const KEY = (characterId: string) => `stellaria.voiceParams.${characterId}`
/** Per-song choices (key, section, blend) are not carried over to the next song. */
const SONG_SPECIFIC: (keyof RenderParams)[] = ['key', 'region', 'spkMix']

/** The settings a voice was last rendered with, so a new take starts where the last one ended. */
export function rememberedParams(characterId: string): Partial<RenderParams> {
  try {
    const raw = localStorage.getItem(KEY(characterId))
    return raw ? (JSON.parse(raw) as Partial<RenderParams>) : {}
  } catch {
    return {}
  }
}

export function rememberParams(characterId: string, params: RenderParams) {
  const kept = Object.fromEntries(Object.entries(params).filter(([k]) => !SONG_SPECIFIC.includes(k as keyof RenderParams)))
  try {
    localStorage.setItem(KEY(characterId), JSON.stringify(kept))
  } catch {
    /* storage unavailable: new takes fall back to defaults */
  }
}

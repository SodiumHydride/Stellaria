import { sendFiles } from '../../api/client'
import type { Job } from '../../api/types'
import { attempt } from '../../lib/actions'
import { pickFiles } from '../../shell/bridge'

/** Send dropped paths (desktop) or File objects (browser) into a character's raw material. */
export function importItems(characterId: string, items: string[] | File[]): Promise<Job | undefined> {
  if (items.length === 0) return Promise.resolve(undefined)
  return attempt(() => sendFiles<Job>(`/characters/${characterId}/raw`, items))
}

/** Open the file picker, then import. */
export async function importMaterial(characterId: string, opts?: { directory?: boolean }): Promise<Job | undefined> {
  const picked = await pickFiles({ multiple: true, directory: opts?.directory })
  if (!picked || picked.length === 0) return undefined
  return importItems(characterId, picked)
}

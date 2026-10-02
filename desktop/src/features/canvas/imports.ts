// Imports run as jobs; the canvas turns finished imports into nodes. This remembers what an
// import was for (a song to split, or a plain audio track) and where it should land.

export interface ImportIntent {
  kind: 'song' | 'audio'
  position?: { x: number; y: number }
}

const pending = new Map<string, ImportIntent>()

export function expectImport(jobId: string, intent: ImportIntent) {
  pending.set(jobId, intent)
}

/** The intent of a finished import job (songs by default), forgotten once read. */
export function takeImport(jobId: string): ImportIntent {
  const intent = pending.get(jobId) ?? { kind: 'song' }
  pending.delete(jobId)
  return intent
}

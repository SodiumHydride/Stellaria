import type { CanvasNode, Character, Model, PortRef } from '../../api/types'
import { isSource, isTake, outputPorts, preferredPort } from './model'

export function latestRelease(m: Model | undefined) {
  if (!m || m.releases.length === 0) return null
  return m.releases.reduce((a, b) => (b.step > a.step ? b : a))
}

/** Pick the model a character sings with: its own model with the newest release, else any trained one. */
export function voiceModelFor(c: Character, models: Model[]): Model | undefined {
  const mine = models.filter((m) => m.characters.some((x) => x.characterId === c.id) && (m.releases.length > 0 || m.checkpoints.length > 0))
  return mine.find((m) => m.id === c.primaryModelId && m.releases.length > 0) ?? mine.find((m) => m.releases.length > 0) ?? mine[0]
}

/** Where a voice should listen on a node: a song's vocals, else the node's first output. */
export function singablePort(n: CanvasNode): PortRef | null {
  if (isSource(n)) return { nodeId: n.id, port: preferredPort(n.data) }
  if (isTake(n)) return n.data.input
  const ports = outputPorts(n)
  return ports.length ? { nodeId: n.id, port: ports[0] } : null
}

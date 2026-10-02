/**
 * Work that has to finish before the app quits: edits still waiting in a debounce. Editors that
 * feed the project (stage 0) flush first, then the project saves themselves (stage 1).
 */
type Hook = { fn: () => void | Promise<void>; stage: number }

const hooks = new Set<Hook>()

export function onBeforeQuit(fn: () => void | Promise<void>, stage: 0 | 1 = 1): () => void {
  const hook = { fn, stage }
  hooks.add(hook)
  return () => {
    hooks.delete(hook)
  }
}

/** Runs every hook (stage by stage); gives up waiting after `timeoutMs` so quitting never hangs. */
export async function runBeforeQuit(timeoutMs = 4000): Promise<void> {
  const run = async () => {
    for (const stage of [0, 1]) {
      await Promise.all([...hooks].filter((h) => h.stage === stage).map((h) => Promise.resolve().then(h.fn).catch(() => {})))
    }
  }
  let timer: number | undefined
  await Promise.race([run(), new Promise<void>((resolve) => (timer = window.setTimeout(resolve, timeoutMs)))])
  window.clearTimeout(timer)
}

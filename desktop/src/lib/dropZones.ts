/**
 * Parts of the window that take dropped files themselves (the arrangement's lanes) mark
 * themselves with `data-drop-zone`. The global drop overlay stays out of their way: browser
 * drops reach them directly, and desktop-shell drops (which only report a position) are handed
 * over as this event, with detail `{x, y, items}`.
 */
export const DROP_FILES_EVENT = 'stellaria:drop-files'

/** The drop zone under a window position, if any (by bounds: the overlay may cover it). */
export function dropZoneAt(x: number, y: number): HTMLElement | null {
  for (const el of document.querySelectorAll<HTMLElement>('[data-drop-zone]')) {
    const r = el.getBoundingClientRect()
    if (r.width > 0 && x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) return el
  }
  return null
}

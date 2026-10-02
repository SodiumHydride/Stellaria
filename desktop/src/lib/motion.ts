// One-shot feedback animations (docs/architecture.md, "Brand and motion"). They follow the system's
// reduced-motion setting; the low-power training mode only drops the glass blur, so these still play.

export function motionAllowed(): boolean {
  return !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}

/** A small pop for an icon that was just switched on (a favourite, a star). */
export function pop(el: Element | null | undefined): void {
  if (!el || !motionAllowed()) return
  el.animate(
    [{ transform: 'scale(1)' }, { transform: 'scale(0.8)', offset: 0.25 }, { transform: 'scale(1.22)', offset: 0.6 }, { transform: 'scale(1)' }],
    { duration: 420, easing: 'ease-out' },
  )
}

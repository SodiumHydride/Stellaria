import './backdrop.css'

/** The app's paper: a still wash of muted color under the frosted panels, plus fine grain.
 *  Pure CSS, painted once; nothing animates, nothing runs per frame. */
export function Backdrop() {
  return (
    <div className="backdrop" aria-hidden>
      <div className="backdrop-mist" />
      <div className="backdrop-grain" />
    </div>
  )
}

import { Check, Copy, ExternalLink } from 'lucide-react'
import { useId, useState } from 'react'
import { useT } from '../i18n'
import { OFFICIAL_REPO, OFFICIAL_REPO_LABEL } from '../lib/official'
import { openUrl } from '../shell/bridge'
import './free-notice.css'

/** A cinnabar seal reading 免费 (FREE), as if stamped onto the page. Small ones carry one 免 (or
 *  the brand star in English): two characters would not survive the size. */
function Seal({ size = 64 }: { size?: number }) {
  const t = useT()
  const id = useId().replace(/:/g, '')
  const zh = t.lang !== 'en'
  const small = size < 40
  return (
    <svg className="seal" width={size} height={size} viewBox="0 0 64 64" aria-hidden>
      <defs>
        {/* a gently uneven edge, and a few patches where the ink did not take */}
        <filter id={`ink-${id}`} x="-8%" y="-8%" width="116%" height="116%">
          <feTurbulence type="fractalNoise" baseFrequency="0.05" numOctaves="2" seed="3" result="wobble" />
          <feDisplacementMap in="SourceGraphic" in2="wobble" scale="2.2" xChannelSelector="R" yChannelSelector="G" result="rough" />
          <feTurbulence type="fractalNoise" baseFrequency={small ? '0.18' : '0.22'} numOctaves="3" seed="8" result="grain" />
          <feColorMatrix in="grain" type="matrix" values={`0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 -9 ${small ? '7.3' : '6.9'}`} result="worn" />
          <feComposite in="rough" in2="worn" operator="in" />
        </filter>
      </defs>
      <g filter={`url(#ink-${id})`}>
        <rect x="4" y="4" width="56" height="56" rx="7" className="seal-ink" />
        {!small && <rect x="8.5" y="8.5" width="47" height="47" rx="4" className="seal-rule" />}
        {small ? (
          zh ? (
            <text x="32" y="44" className="seal-text one">
              免
            </text>
          ) : (
            <path className="seal-star" transform="translate(32 32) scale(0.064)" d="M0 -330C22 -86 86 -22 330 0C86 22 22 86 0 330C-22 86 -86 22 -330 0C-86 -22 -22 -86 0 -330Z" />
          )
        ) : zh ? (
          <>
            <text x="32" y="30" className="seal-text">
              免
            </text>
            <text x="32" y="52" className="seal-text">
              费
            </text>
          </>
        ) : (
          <text x="32" y="38" className="seal-text en">
            FREE
          </text>
        )}
      </g>
    </svg>
  )
}

/** "星籁是免费的": whoever paid for a copy learns they were scammed, and where the real one lives. */
export function FreeNotice() {
  const t = useT()
  const [copied, setCopied] = useState(false)
  const copy = () =>
    navigator.clipboard.writeText(OFFICIAL_REPO).then(
      () => {
        setCopied(true)
        window.setTimeout(() => setCopied(false), 1600)
      },
      () => {},
    )
  return (
    <section className="free-card" aria-label={t('free.title')}>
      <Seal />
      <p className="free-title">{t('free.title')}</p>
      <p className="free-body">{t('free.body')}</p>
      <div className="free-foot">
        <span className="free-label">{t('free.official')}</span>
        <span className="free-url">{OFFICIAL_REPO_LABEL}</span>
        <span className="spacer" />
        <button className="btn btn-sm btn-ghost" onClick={copy}>
          {copied ? <Check /> : <Copy />}
          {copied ? t('free.copied') : t('free.copy')}
        </button>
        <button className="btn btn-sm" onClick={() => void openUrl(OFFICIAL_REPO)}>
          <ExternalLink />
          {t('free.open')}
        </button>
      </div>
    </section>
  )
}

/** The same in one line, for the welcome screen. */
export function FreeLine() {
  const t = useT()
  return (
    <p className="free-line">
      <Seal size={30} />
      <span>{t('free.line')}</span>
    </p>
  )
}

import clsx from 'clsx'
import { Heart, Star } from 'lucide-react'
import { useState } from 'react'
import { Tip } from '../../../components/ui'
import { useT } from '../../../i18n'
import { pop } from '../../../lib/motion'

export function Section({ title, hint, children, aside }: { title: string; hint?: string; children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <section className="insp-section">
      <div className="insp-section-head">
        {hint ? (
          <Tip label={hint}>
            <span className="eyebrow has-hint">{title}</span>
          </Tip>
        ) : (
          <span className="eyebrow">{title}</span>
        )}
        <span className="spacer" />
        {aside}
      </div>
      {children}
    </section>
  )
}

export function Rating({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const t = useT()
  const [hover, setHover] = useState(0)
  return (
    <div className="rating" onMouseLeave={() => setHover(0)}>
      {[1, 2, 3, 4, 5].map((i) => (
        <button
          key={i}
          className={clsx((hover || value) >= i && 'on')}
          onMouseEnter={() => setHover(i)}
          onClick={(e) => {
            if (value !== i) pop(e.currentTarget.querySelector('svg'))
            onChange(value === i ? 0 : i)
          }}
          aria-label={t('insp.rate', { n: i })}
          aria-pressed={value === i}
        >
          <Star />
        </button>
      ))}
    </div>
  )
}

/** A take's stars and heart, kept together so a narrow panel wraps them as one group. */
export function RateAndFavorite({ rating, favorite, onRate, onFavorite }: { rating: number; favorite: boolean; onRate: (v: number) => void; onFavorite: () => void }) {
  const t = useT()
  const label = favorite ? t('insp.unfavorite') : t('insp.favorite')
  return (
    <div className="row insp-rate">
      <Rating value={rating} onChange={onRate} />
      <Tip label={label}>
        <button
          className={clsx('icon-btn sm', favorite && 'fav-on')}
          onClick={(e) => {
            if (!favorite) pop(e.currentTarget.querySelector('svg'))
            onFavorite()
          }}
          aria-label={label}
          aria-pressed={favorite}
        >
          <Heart />
        </button>
      </Tip>
    </div>
  )
}

/** Short status row for a node's last run: error or out-of-date notice. */
export function RunNotice({ error, stale, staleLabel, msg }: { error: { code: string } | null; stale: boolean; staleLabel: string; msg: (m: { code: string }) => string }) {
  if (error) return <div className="chip critical insp-error">{msg(error)}</div>
  if (stale) return <div className="chip warning insp-error">{staleLabel}</div>
  return null
}

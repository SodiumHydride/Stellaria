import clsx from 'clsx'
import { GripVertical } from 'lucide-react'
import { memo } from 'react'
import type { Track } from '../../../api/types'
import { useT } from '../../../i18n'
import { LiveSlider, MuteSolo, usePanText } from './controls'
import { Meter } from './Meter'
import { faderDb, faderPos, fmtDb, moveTrack, patchTrack } from './ops'
import { HEADER_W, useTimeline } from './state'

interface HeaderProps {
  track: Track
  index: number
  color: string | null
  focused: boolean
  onReorderStart: (e: React.PointerEvent, trackId: string) => void
}

/**
 * A track's header in the arrangement: colour strip, grip (drag or ↑ ↓ to reorder), name,
 * mute / solo, volume in dB and pan (double-click either to reset), a small level meter.
 */
export const TrackHeader = memo(function TrackHeader({ track, index, color, focused, onReorderStart }: HeaderProps) {
  const t = useT()
  const panText = usePanText()
  const { api } = useTimeline()
  const id = track.id
  const patch = (p: Partial<Track>, key?: string) => api.edit((tl) => patchTrack(tl, id, p), key)
  const live = (p: Partial<Track>) => api.preview((tl) => patchTrack(tl, id, p))

  return (
    <div
      className={clsx('tl-head', focused && 'focused', track.mute && 'is-muted')}
      style={{ width: HEADER_W, ['--trk' as string]: color ?? 'var(--text-4)' }}
      data-track-head={id}
      onPointerDown={() => api.focusTrack(id)}
      onContextMenu={(e) => {
        e.preventDefault()
        api.openMenu(e, { kind: 'track', trackId: id })
      }}
    >
      <span className="tl-head-strip" />
      <div className="tl-head-main">
        <div className="tl-head-row">
          <button
            className="tl-grip"
            onPointerDown={(e) => onReorderStart(e, id)}
            onKeyDown={(e) => {
              if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
              e.preventDefault()
              e.stopPropagation()
              api.edit((tl) => moveTrack(tl, id, index + (e.key === 'ArrowUp' ? -1 : 1)))
            }}
            title={t('tl.track_reorder')}
            aria-label={`${t('tl.track_reorder')} · ${track.name}`}
          >
            <GripVertical />
          </button>
          <input
            className="tl-name"
            data-track-name={id}
            value={track.name}
            spellCheck={false}
            onChange={(e) => patch({ name: e.target.value }, `name:${id}`)}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === 'Escape') && e.currentTarget.blur()}
            aria-label={t('tl.track_name')}
          />
          <MuteSolo track={track} onPatch={(p) => patch(p)} />
        </div>
        <div className="tl-head-row" title={`${t('tl.volume')} ${fmtDb(track.gainDb)} dB · ${t('tl.dblclick_reset')}`}>
          <span className="tl-head-key">{t('tl.vol_short')}</span>
          <LiveSlider
            value={Math.round(faderPos(track.gainDb) * 1000)}
            min={0}
            max={1000}
            step={5}
            reset={800}
            label={`${t('tl.volume')} · ${track.name}`}
            onLive={(v) => live({ gainDb: faderDb(v / 1000) })}
            onCommit={() => api.commit(`vol:${id}`)}
            onReset={() => patch({ gainDb: 0 })}
          />
          <span className="tl-head-val tabular">{fmtDb(track.gainDb)}</span>
        </div>
        <div className="tl-head-row" title={`${t('tl.pan')} ${panText(track.pan)} · ${t('tl.dblclick_center')}`}>
          <span className="tl-head-key">{t('tl.pan_short')}</span>
          <LiveSlider
            value={Math.round(track.pan * 100)}
            min={-100}
            max={100}
            step={1}
            reset={0}
            centerMark
            label={`${t('tl.pan')} · ${track.name}`}
            onLive={(v) => live({ pan: v / 100 })}
            onCommit={() => api.commit(`pan:${id}`)}
            onReset={() => patch({ pan: 0 })}
          />
          <span className="tl-head-val tabular">{panText(track.pan)}</span>
        </div>
      </div>
      <Meter id={id} compact className="tl-head-meter" />
    </div>
  )
}, sameHeader)

/** a header shows no clips: moving clips on its track doesn't render it again */
function sameHeader(a: HeaderProps, b: HeaderProps): boolean {
  const ta = a.track
  const tb = b.track
  return (
    a.index === b.index &&
    a.color === b.color &&
    a.focused === b.focused &&
    a.onReorderStart === b.onReorderStart &&
    ta.id === tb.id &&
    ta.name === tb.name &&
    ta.gainDb === tb.gainDb &&
    ta.pan === tb.pan &&
    ta.mute === tb.mute &&
    ta.solo === tb.solo
  )
}

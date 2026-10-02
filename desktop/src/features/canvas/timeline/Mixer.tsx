import clsx from 'clsx'
import { memo } from 'react'
import type { Timeline, Track } from '../../../api/types'
import { useT } from '../../../i18n'
import { DbInput, DbScale, Fader, LiveSlider, MuteSolo, usePanText } from './controls'
import { Meter } from './Meter'
import { patchTrack } from './ops'
import { useTimeline } from './state'

/** 调音台: one channel strip per track and the master; the same tracks as the arrangement. */
export function Mixer({ tl, focusTrack, trackColor }: { tl: Timeline; focusTrack: string | null; trackColor: (tr: Track) => string | null }) {
  const t = useT()
  const { api } = useTimeline()
  const master = tl.masterGainDb ?? 0
  return (
    <div className="tl-mixer">
      <div className="tl-strips" role="group" aria-label={t('tl.mixer')}>
        {tl.tracks.length === 0 && <div className="tl-mixer-empty muted">{t('tl.mixer_empty')}</div>}
        {tl.tracks.map((tr) => (
          <Strip key={tr.id} track={tr} color={trackColor(tr)} focused={focusTrack === tr.id} />
        ))}
      </div>
      <div className="tl-strip master" role="group" aria-label={t('tl.master')}>
        <div className="tl-strip-name">{t('tl.master')}</div>
        <div className="tl-strip-gap" />
        <div className="tl-strip-body">
          <DbScale />
          <Fader
            db={master}
            label={t('tl.master')}
            onLive={(db) => api.preview((x) => ({ ...x, masterGainDb: db }))}
            onCommit={(db) => {
              api.preview((x) => ({ ...x, masterGainDb: db }))
              api.commit('master')
            }}
          />
          <Meter id="master" />
        </div>
        <DbInput db={master} label={t('tl.master')} onCommit={(db) => api.edit((x) => ({ ...x, masterGainDb: db }), 'master')} />
      </div>
    </div>
  )
}

const Strip = memo(function Strip({ track, color, focused }: { track: Track; color: string | null; focused: boolean }) {
  const t = useT()
  const panText = usePanText()
  const { api } = useTimeline()
  const id = track.id
  const patch = (p: Partial<Track>, key?: string) => api.edit((x) => patchTrack(x, id, p), key)
  const setGain = (db: number) => {
    api.preview((x) => patchTrack(x, id, { gainDb: db }))
    api.commit(`vol:${id}`)
  }
  return (
    <div
      className={clsx('tl-strip', focused && 'focused', track.mute && 'is-muted')}
      style={{ ['--trk' as string]: color ?? 'var(--text-4)' }}
      role="group"
      aria-label={track.name}
      onPointerDown={() => api.focusTrack(id)}
      onContextMenu={(e) => {
        e.preventDefault()
        api.openMenu(e, { kind: 'track', trackId: id })
      }}
    >
      <div className="tl-strip-name" title={track.name}>
        {track.name}
      </div>
      <div className="tl-strip-pan" title={`${t('tl.pan')} ${panText(track.pan)} · ${t('tl.dblclick_center')}`}>
        <LiveSlider
          value={Math.round(track.pan * 100)}
          min={-100}
          max={100}
          step={1}
          reset={0}
          centerMark
          label={`${t('tl.pan')} · ${track.name}`}
          onLive={(v) => api.preview((x) => patchTrack(x, id, { pan: v / 100 }))}
          onCommit={() => api.commit(`pan:${id}`)}
          onReset={() => patch({ pan: 0 })}
        />
        <span className="tl-strip-pan-val tabular">{panText(track.pan)}</span>
      </div>
      <div className="tl-strip-ms">
        <MuteSolo track={track} onPatch={(p) => patch(p)} />
      </div>
      <div className="tl-strip-body">
        <DbScale />
        <Fader db={track.gainDb} label={`${t('tl.volume')} · ${track.name}`} onLive={(db) => api.preview((x) => patchTrack(x, id, { gainDb: db }))} onCommit={setGain} />
        <Meter id={id} />
      </div>
      <DbInput db={track.gainDb} label={`${t('tl.volume')} · ${track.name}`} onCommit={setGain} />
    </div>
  )
})

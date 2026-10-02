import { useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Check, ChevronLeft, ChevronRight, Pause, Play, Scissors, Sparkles, Trash2, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { api, mediaUrl } from '../../../api/client'
import type { Character, Clip } from '../../../api/types'
import { Waveform } from '../../../components/Waveform'
import { Progress } from '../../../components/ui'
import { useT } from '../../../i18n'
import { attempt } from '../../../lib/actions'
import { clock, hzToNote } from '../../../lib/format'
import { useEvent } from '../../../lib/useEvent'
import { usePlayback, usePlayer } from '../../../state/player'
import { clipAudioPath, clipPeaksPath, flagSeverity, patchClips } from './ClipList'

/**
 * Focused, keyboard-first listening pass. Space plays, ←/→ move, D trashes, V toggles
 * validation, S splits at the playhead, Enter keeps (marks reviewed) and advances.
 */
export function ReviewMode({ character, clips, onExit }: { character: Character; clips: Clip[]; onExit: () => void }) {
  const t = useT()
  const qc = useQueryClient()
  const cid = character.id
  // the order is fixed for the session so items don't jump while reviewing:
  // unreviewed first, flagged ones first among them
  const [queue] = useState(() =>
    clips.filter((c) => c.status === 'active').sort((a, b) => Number(a.reviewed) - Number(b.reviewed) || b.flags.length - a.flags.length),
  )
  const [index, setIndex] = useState(0)
  const live = useMemo(() => new Map(clips.map((c) => [c.id, c])), [clips])
  const clip = queue[index] ? (live.get(queue[index].id) ?? queue[index]) : undefined
  const reviewed = queue.filter((c) => live.get(c.id)?.reviewed || live.get(c.id)?.status === 'trashed').length
  const { playing, time } = usePlayback(clip?.id ?? '')
  const toggle = usePlayer((s) => s.toggle)
  const play = usePlayer((s) => s.play)
  const stop = usePlayer((s) => s.stop)
  const autoplay = useRef(true)

  const go = useCallback(
    (delta: number) => {
      setIndex((i) => Math.max(0, Math.min(queue.length, i + delta)))
    },
    [queue.length],
  )

  // a new clip plays from the start (edits to the current one don't restart it)
  const startClip = useEvent(() => {
    if (clip && autoplay.current) play(clip.id, mediaUrl(clipAudioPath(cid, clip)), 0)
  })
  const clipId = clip?.id
  useEffect(() => {
    if (clipId) startClip()
  }, [clipId, startClip])

  useEffect(() => () => stop(), [stop])

  const refresh = () => qc.invalidateQueries({ queryKey: ['clips', cid] })

  const keep = async () => {
    if (!clip) return
    await patchClips(cid, [clip.id], { reviewed: true })
    refresh()
    go(1)
  }
  const trash = async () => {
    if (!clip) return
    await patchClips(cid, [clip.id], { status: 'trashed', reviewed: true })
    refresh()
    go(1)
  }
  const toggleVal = async () => {
    if (!clip) return
    await patchClips(cid, [clip.id], { split: clip.split === 'val' ? 'train' : 'val' })
    refresh()
  }
  const split = async () => {
    if (!clip || time <= 0.3 || time >= clip.duration - 0.3) return
    await attempt(() => api.post(`/characters/${cid}/clips/${clip.id}/split`, { at: time }))
    refresh()
    go(1)
  }

  const onKey = useEvent((e: KeyboardEvent) => {
    if ((e.target as HTMLElement).closest('input, textarea')) return
    const k = e.key.toLowerCase()
    if (k === ' ') {
      e.preventDefault()
      if (clip) toggle(clip.id, mediaUrl(clipAudioPath(cid, clip)))
    } else if (k === 'arrowright') go(1)
    else if (k === 'arrowleft') go(-1)
    else if (k === 'd' || k === 'delete') void trash()
    else if (k === 'v') void toggleVal()
    else if (k === 's') void split()
    else if (k === 'enter') void keep()
    else if (k === 'escape') onExit()
    else return
    e.preventDefault()
  })
  useEffect(() => {
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onKey])

  return createPortal(
    <div className="review" role="dialog" aria-label={t('mat.review')}>
      <div className="review-top">
        <span className="section-title">{t('mat.review')}</span>
        <span className="muted tabular">{t('mat.review.progress', { done: reviewed, total: queue.length })}</span>
        <div style={{ width: 220 }}>
          <Progress value={queue.length ? reviewed / queue.length : 0} />
        </div>
        <span className="spacer" />
        <button className="btn btn-sm" onClick={onExit}>
          <X />
          {t('mat.review.exit')}
        </button>
      </div>

      {clip ? (
        <div className="review-stage">
          <div className="review-card glass" style={{ ['--voice' as string]: character.color }}>
            <div className="row" style={{ gap: 10 }}>
              <span className="review-index tabular">
                {index + 1} / {queue.length}
              </span>
              <h2 className="review-name">{clip.name}</h2>
              {clip.reviewed && <span className="chip good"><Check />{t('mat.action.mark_reviewed')}</span>}
              <span className="spacer" />
              <button className={clsx('chip button', clip.split === 'val' && 'accent')} onClick={toggleVal} aria-pressed={clip.split === 'val'}>
                {clip.split === 'val' ? t('mat.split.val') : t('mat.split.train')} <span className="kbd">V</span>
              </button>
            </div>
            {clip.rawName && (
              <div className="muted" style={{ marginTop: 4 }}>
                {t('mat.review.from', { name: clip.rawName, at: clip.start != null ? clock(clip.start) : '' })}
              </div>
            )}
            <div className="review-wave">
              <Waveform id={clip.id} peaksPath={clipPeaksPath(cid, clip)} audioPath={clipAudioPath(cid, clip)} height={120} bars={260} color={character.color} />
            </div>
            <div className="review-facts">
              <span className="tabular">{clip.duration.toFixed(1)} s</span>
              {clip.metrics?.f0Low && (
                <span className="tabular">
                  {hzToNote(clip.metrics.f0Low)} – {hzToNote(clip.metrics.f0High)}
                </span>
              )}
              {clip.metrics?.snrDb != null && <span className="tabular">SNR {Math.round(clip.metrics.snrDb)} dB</span>}
              {clip.flags.map((f) => (
                <span key={f} className={clsx('chip', flagSeverity(f))}>
                  {t(`flag.${f}` as const)}
                </span>
              ))}
            </div>
            <div className="review-actions">
              <button className="btn btn-lg" onClick={() => go(-1)} disabled={index === 0}>
                <ChevronLeft />
                <span className="kbd">←</span>
              </button>
              <button className="btn btn-lg btn-danger" onClick={trash}>
                <Trash2 />
                {t('mat.action.trash')}
                <span className="kbd">D</span>
              </button>
              <button className="review-play" onClick={() => toggle(clip.id, mediaUrl(clipAudioPath(cid, clip)))} aria-label={playing ? 'pause' : 'play'}>
                {playing ? <Pause /> : <Play />}
              </button>
              <button className="btn btn-lg" onClick={split} disabled={time <= 0.3}>
                <Scissors />
                {t('mat.review.split_here')}
                <span className="kbd">S</span>
              </button>
              <button className="btn btn-lg btn-primary" onClick={keep}>
                <Check />
                {t('mat.review.keep')}
                <span className="kbd">↵</span>
              </button>
              <button className="btn btn-lg" onClick={() => go(1)}>
                <span className="kbd">→</span>
                <ChevronRight />
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="review-stage">
          <div className="review-done">
            <Sparkles />
            <h2>{t('mat.review.done_all')}</h2>
            <button className="btn btn-primary" onClick={onExit}>
              {t('common.done')}
            </button>
          </div>
        </div>
      )}
    </div>,
    document.body,
  )
}

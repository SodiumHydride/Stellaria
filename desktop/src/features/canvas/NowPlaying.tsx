import clsx from 'clsx'
import { useEffect, useRef } from 'react'
import { Pause, Play, X } from 'lucide-react'
import { mediaUrl } from '../../api/client'
import type { CanvasNode } from '../../api/types'
import { LevelMeter } from '../../components/LevelMeter'
import { Segmented, Slider, Tip } from '../../components/ui'
import { useT } from '../../i18n'
import { clockPrecise } from '../../lib/format'
import { usePlayer } from '../../state/player'
import { mediaAudioPath, useCanvas } from './context'
import { byIdOf, isProcess, isTake, playableMedia, resolvePort, type ById } from './model'
import { nodeDisplayName, usePortLabel } from './nodes'

type Partner = { kind: 'original' | 'bypass'; mediaId: string }

/**
 * What a node's sound can be compared with at the same position: a take with what it sang
 * (原唱), an effect or pitch shift (same tempo) with its input (A/B against bypass).
 */
function partnerOf(n: CanvasNode | undefined, byId: ById): Partner | null {
  if (isTake(n) && n.data.status === 'done' && n.data.mediaId) {
    const mid = resolvePort(byId, n.data.input)
    return mid ? { kind: 'original', mediaId: mid } : null
  }
  if (isProcess(n) && !n.data.bypass && n.data.inputs[0] && (n.data.op === 'fx' || (n.data.op === 'pitch' && (n.data.params as { tempo?: number }).tempo === 1))) {
    const mid = resolvePort(byId, n.data.inputs[0])
    return mid ? { kind: 'bypass', mediaId: mid } : null
  }
  return null
}

/** The canvas transport: what is playing, where, how loud, and a one-key comparison. */
export function NowPlaying() {
  const t = useT()
  const { project, characters } = useCanvas()
  const portLabel = usePortLabel()
  const currentId = usePlayer((s) => s.currentId)
  const playing = usePlayer((s) => s.playing)
  const time = usePlayer((s) => s.time)
  const duration = usePlayer((s) => s.duration)
  const url = usePlayer((s) => s.url)
  const group = usePlayer((s) => s.group)
  // X starts the comparison when nothing is being compared yet (the canvas flips an existing one)
  const startCompare = useRef<(() => void) | null>(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() !== 'x' || e.metaKey || e.ctrlKey || e.altKey) return
      if ((e.target as HTMLElement).closest('input, textarea, select, [contenteditable="true"]')) return
      if (usePlayer.getState().group || !startCompare.current) return
      e.preventDefault() // the canvas's X (flip) leaves it alone
      startCompare.current()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  if (!currentId || !url) {
    startCompare.current = null
    return null
  }

  const byId = byIdOf(project.canvas.nodes)
  const m = /^(node|in):([^:]+)(?::(.+))?$/.exec(currentId)
  const node = m ? byId.get(m[2]) : undefined
  const port = m?.[3]
  const name = node ? `${nodeDisplayName(node, characters, t)}${port && port !== 'out' ? ` · ${portLabel(port, node)}` : ''}` : t('np.playing')
  const partner = node && !port ? partnerOf(node, byId) : null
  const own = node ? playableMedia(node, byId) : null
  const pair =
    node && partner && own
      ? [
          { id: `node:${node.id}`, url: mediaUrl(mediaAudioPath(project.id, own)) },
          { id: `in:${node.id}`, url: mediaUrl(mediaAudioPath(project.id, partner.mediaId)) },
        ]
      : null
  const comparing = !!pair && !!group && group.length === 2 && group[0].id === pair[0].id && group[1].id === pair[1].id
  const side = currentId.startsWith('in:') ? 'b' : 'a'
  const player = usePlayer.getState()
  startCompare.current = pair && !comparing ? () => player.compareGroup(pair, 1) : null

  return (
    <div className="now-playing glass-strong" role="region" aria-label={t('np.playing')}>
      <button className="icon-btn sm" onClick={() => (playing ? player.pause() : player.play(currentId, url))} aria-label={playing ? t('np.pause') : t('np.play')}>
        {playing ? <Pause /> : <Play />}
      </button>
      <span className="now-playing-name" title={name}>
        {name}
      </span>
      <span className="now-playing-time tabular">
        {clockPrecise(time)} / {clockPrecise(duration)}
      </span>
      <div className="now-playing-scrub">
        <Slider value={Math.min(time, duration || 0)} min={0} max={Math.max(0.01, duration)} step={0.05} onChange={(v) => player.seek(v)} label={t('np.position')} />
      </div>
      <LevelMeter />
      {pair && (
        <Tip label={t('np.compare_hint')}>
          <span className={clsx('now-playing-ab', comparing && 'on')}>
            <Segmented<'a' | 'b'>
              size="sm"
              value={comparing ? side : 'a'}
              onChange={(v) => {
                if (!comparing) player.compareGroup(pair, v === 'a' ? 0 : 1)
                else player.switchTo(v === 'a' ? 0 : 1)
              }}
              options={[
                { value: 'a', label: partner!.kind === 'original' ? t('np.converted') : t('np.processed') },
                { value: 'b', label: partner!.kind === 'original' ? t('np.original') : t('np.bypassed') },
              ]}
            />
            <span className="kbd">X</span>
          </span>
        </Tip>
      )}
      <Tip label={t('np.stop')}>
        <button className="icon-btn sm" onClick={() => player.stop()} aria-label={t('np.stop')}>
          <X />
        </button>
      </Tip>
    </div>
  )
}

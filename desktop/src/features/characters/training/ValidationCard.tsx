import clsx from 'clsx'
import { Headphones, Pause, Play, Repeat } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, mediaUrl } from '../../../api/client'
import { useValidation, useValidationSteps } from '../../../api/hooks'
import type { Character, MelData, Model, ValidationSample } from '../../../api/types'
import { Spectrogram } from '../../../components/Spectrogram'
import { Tip } from '../../../components/ui'
import { useT } from '../../../i18n'
import { steps as fmtSteps } from '../../../lib/format'
import { usePlayer } from '../../../state/player'

const melCache = new Map<string, Promise<MelData>>()

function useMel(url: string | null) {
  const [mel, setMel] = useState<MelData | null>(null)
  useEffect(() => {
    if (!url) return
    let p = melCache.get(url)
    if (!p) {
      p = api.get<MelData>(url.replace(/^\/api/, ''))
      p.catch(() => melCache.delete(url))
      melCache.set(url, p)
    }
    let cancelled = false
    p.then((m) => !cancelled && setMel(m)).catch(() => {})
    return () => {
      cancelled = true
    }
  }, [url])
  return mel
}

/** Listen to validation clips at any preview step: original vs. synthesized, A/B with one key. */
export function ValidationCard({ model, character, step, onStep }: { model: Model; character: Character; step: number | 'latest'; onStep: (s: number | 'latest') => void }) {
  const t = useT()
  const { data: stepsList } = useValidationSteps(model.id)
  const { data: snap } = useValidation(model.id, step)
  const [open, setOpen] = useState<string | null>(null)

  const recent = (stepsList ?? []).slice(-6).reverse()

  return (
    <div className="valcard glass">
      <div className="row">
        <Headphones className="valcard-icon" />
        <span className="section-title">{t('val.title')}</span>
        <span className="spacer" />
        {snap && <span className="chip accent tabular">{t('val.step', { step: fmtSteps(snap.step) })}</span>}
      </div>
      {recent.length > 0 && (
        <div className="valcard-steps">
          <button className={clsx('chip button', step === 'latest' && 'accent')} onClick={() => onStep('latest')}>
            {t('val.latest')}
          </button>
          {recent.map((s) => (
            <button key={s} className={clsx('chip button tabular', step === s && 'accent')} onClick={() => onStep(s)}>
              {fmtSteps(s)}
            </button>
          ))}
        </div>
      )}
      {!snap || snap.samples.length === 0 ? (
        <div className="valcard-empty muted">{t('val.none', { step: fmtSteps(model.config.intervalVal) })}</div>
      ) : (
        <div className="valcard-list">
          {snap.samples.map((s) => (
            <SampleRow key={s.name} sample={s} step={snap.step} color={character.color} open={open === s.name} onToggle={() => setOpen(open === s.name ? null : s.name)} />
          ))}
        </div>
      )}
    </div>
  )
}

function SampleRow({ sample, step, color, open, onToggle }: { sample: ValidationSample; step: number; color: string; open: boolean; onToggle: () => void }) {
  const t = useT()
  const idGt = `val:${step}:${sample.name}:gt`
  const idPred = `val:${step}:${sample.name}:pred`
  const current = usePlayer((s) => s.currentId)
  const playing = usePlayer((s) => s.playing)
  const group = usePlayer((s) => s.group)
  const compare = usePlayer((s) => s.compare)
  const flip = usePlayer((s) => s.flip)
  const toggle = usePlayer((s) => s.toggle)
  const mel = useMel(open ? sample.melUrl : null)
  const isMine = current === idGt || current === idPred
  const inAb = !!group?.some((x) => x.id === idGt)

  const play = (which: 'gt' | 'pred') => toggle(which === 'gt' ? idGt : idPred, mediaUrl(which === 'gt' ? sample.gtUrl : sample.predUrl))

  return (
    <div className={clsx('sample', open && 'open', isMine && playing && 'playing')} style={{ ['--voice' as string]: color }}>
      <div className="sample-head">
        <button className="sample-name" onClick={onToggle} title={sample.name}>
          {sample.name}
        </button>
        <div className="sample-ab">
          <button className={clsx('sample-btn', current === idGt && playing && 'on')} onClick={() => play('gt')}>
            {current === idGt && playing ? <Pause /> : <Play />}
            {t('val.gt')}
          </button>
          <button className={clsx('sample-btn pred', current === idPred && playing && 'on')} onClick={() => play('pred')}>
            {current === idPred && playing ? <Pause /> : <Play />}
            {t('val.pred')}
          </button>
          <Tip label={`${t('val.ab')} (X)`}>
            <button
              className={clsx('icon-btn sm', inAb && 'active')}
              onClick={() => (inAb ? flip() : compare({ id: idGt, url: mediaUrl(sample.gtUrl) }, { id: idPred, url: mediaUrl(sample.predUrl) }))}
            >
              <Repeat />
            </button>
          </Tip>
        </div>
      </div>
      {open && (
        <div className="sample-mels">
          {mel ? (
            <>
              <div>
                <span className="eyebrow">{t('val.gt')}</span>
                <Spectrogram data={mel.gtData} frames={mel.frames} bins={mel.bins} height={70} />
              </div>
              <div>
                <span className="eyebrow">{t('val.pred')}</span>
                <Spectrogram data={mel.data} frames={mel.frames} bins={mel.bins} height={70} />
              </div>
            </>
          ) : (
            <div className="skeleton" style={{ height: 160 }} />
          )}
        </div>
      )}
    </div>
  )
}

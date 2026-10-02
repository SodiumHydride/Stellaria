import clsx from 'clsx'
import { AlertTriangle, ChevronDown, Cpu, Rocket, ShieldCheck, Waves } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../../api/client'
import { useSystem, useTrainPresets } from '../../../api/hooks'
import type { Character, Model, TrainConfig, TrainPresetId } from '../../../api/types'
import { Segmented } from '../../../components/ui'
import { useT } from '../../../i18n'
import { attempt } from '../../../lib/actions'
import { bytes, duration, steps } from '../../../lib/format'
import { TrainParams } from './TrainParams'

type TargetChoice = 'quick' | 'standard' | 'fine' | 'custom'
const TARGETS: Record<Exclude<TargetChoice, 'custom'>, number> = { quick: 20000, standard: 100000, fine: 200000 }

const DEFAULT_CONFIG: TrainConfig = {
  preset: 'vram8',
  targetSteps: 100000,
  batchSize: 48,
  lr: 0.0005,
  ampDtype: 'fp16',
  cacheDevice: 'cpu',
  numWorkers: 2,
  intervalVal: 2000,
  intervalForceSave: 10000,
  f0Extractor: 'rmvpe',
  encoder: 'contentvec768l12tta2x',
  nLayers: 6,
  nChans: 1024,
  nAuxLayers: 6,
  nAuxChans: 512,
  usePitchAug: true,
  useFusedKernels: 'auto',
}

/** Choose a preset and a length, then start. Everything else hides under "Advanced". */
export function TrainSetup({ character, model }: { character: Character; model: Model | null }) {
  const t = useT()
  const nav = useNavigate()
  const { data: presets } = useTrainPresets()
  const { data: system } = useSystem()
  const [config, setConfig] = useState<TrainConfig>(model?.config ?? DEFAULT_CONFIG)
  const [target, setTarget] = useState<TargetChoice>('standard')
  const [advanced, setAdvanced] = useState(false)
  const [busy, setBusy] = useState(false)

  const recommended = presets?.find((p) => p.recommended)
  // a new model starts from the recommended preset, once it is known (refetches don't reset edits)
  const applied = useRef<string | null>(null)
  useEffect(() => {
    if (model || !recommended || applied.current === recommended.id) return
    applied.current = recommended.id
    setConfig((c) => ({ ...c, ...recommended.config, preset: recommended.id }))
  }, [model, recommended])

  const gpu = system?.gpus[0]
  const vramGb = gpu ? gpu.vramTotal / 1024 ** 3 : 0
  const blocked = character.health?.level === 'blocked' || !character.health
  const targetSteps = target === 'custom' ? config.targetSteps : TARGETS[target]

  const choosePreset = (id: TrainPresetId) => {
    const p = presets?.find((x) => x.id === id)
    if (p) setConfig((c) => ({ ...c, ...p.config, preset: id }))
  }


  const start = async () => {
    setBusy(true)
    const body = { ...config, targetSteps }
    let id = model?.id
    if (!id) {
      const m = await attempt(() => api.post<Model>('/models', { name: character.name, characterIds: [character.id], config: body }))
      id = m?.id
    } else {
      await attempt(() => api.patch<Model>(`/models/${id}`, { config: body, targetSteps }))
    }
    if (id) await attempt(() => api.post(`/models/${id}/train`, {}))
    setBusy(false)
  }

  return (
    <div className="setup">
      <div className="setup-hero glass">
        <div className="setup-hero-text">
          <h2>{t('train.setup.title', { name: character.name })}</h2>
        </div>
        <div className="setup-facts">
          <div className="fact">
            <Waves />
            <div>
              <span className="eyebrow">{t('train.setup.data')}</span>
              <strong>{t('train.setup.data_value', { duration: duration(character.stats.activeDuration, t.lang), clips: character.stats.clipCount })}</strong>
            </div>
          </div>
          <div className="fact">
            <ShieldCheck />
            <div>
              <span className="eyebrow">{t('health.title')}</span>
              <strong className={clsx(blocked && 'warn-text')}>
                {character.health && !blocked ? t('train.setup.health', { score: character.health.score }) : t('train.setup.health_blocked')}
              </strong>
            </div>
          </div>
          <div className="fact">
            <Cpu />
            <div>
              <span className="eyebrow">{t('train.setup.gpu')}</span>
              <strong>{gpu ? `${gpu.name.replace('NVIDIA ', '')} · ${bytes(gpu.vramTotal)}` : t('gpu.none')}</strong>
            </div>
          </div>
        </div>
        {blocked && (
          <div className="setup-warn">
            <AlertTriangle />
            <span>{t('train.setup.health_blocked')}</span>
            <span className="spacer" />
            <button className="btn btn-sm" onClick={() => nav(`/voices/${character.id}/material`)}>
              {t('train.setup.go_fix')}
            </button>
          </div>
        )}
      </div>

      <div className="setup-section">
        <div className="setup-label">
          <span className="section-title">{t('train.setup.preset')}</span>
        </div>
        <div className="presets">
          {(presets ?? []).filter((p) => p.id !== 'custom').map((p) => (
            <button key={p.id} className={clsx('preset glass', config.preset === p.id && 'on')} onClick={() => choosePreset(p.id)}>
              {p.recommended && <span className="chip accent preset-badge">{t('common.recommended')}</span>}
              <span className="preset-vram">{t('train.setup.preset_label', { vram: p.vram })}</span>
              <span className="muted">{t('train.setup.preset_sub', { batch: p.config.batchSize ?? '–' })}</span>
              {vramGb > 0 && p.vram > vramGb + 0.5 && <span className="chip warning">{t('train.setup.too_big')}</span>}
            </button>
          ))}
        </div>
      </div>

      <div className="setup-section">
        <div className="setup-label">
          <span className="section-title">{t('train.setup.target')}</span>
        </div>
        <div className="row" style={{ gap: 12, flexWrap: 'wrap' }}>
          <Segmented<TargetChoice>
            value={target}
            onChange={setTarget}
            options={[
              { value: 'quick', label: `${t('train.setup.target.quick')} · ${steps(TARGETS.quick)}` },
              { value: 'standard', label: `${t('train.setup.target.standard')} · ${steps(TARGETS.standard)}` },
              { value: 'fine', label: `${t('train.setup.target.fine')} · ${steps(TARGETS.fine)}` },
              { value: 'custom', label: t('train.setup.target.custom') },
            ]}
          />
          {target === 'custom' && (
            <input
              className="input tabular"
              style={{ width: 140 }}
              type="number"
              min={1000}
              step={1000}
              value={config.targetSteps}
              onChange={(e) => setConfig((c) => ({ ...c, targetSteps: Math.max(1000, Number(e.target.value) || 0) }))}
            />
          )}
        </div>
      </div>

      <button className="setup-adv-toggle" onClick={() => setAdvanced(!advanced)}>
        <ChevronDown style={{ transform: advanced ? 'rotate(180deg)' : undefined }} />
        {t('common.advanced')}
      </button>
      {advanced && (
        <div className="setup-params glass">
          <TrainParams config={config} onChange={setConfig} />
        </div>
      )}

      <div className="setup-go">
        <button className="btn btn-primary btn-lg" onClick={start} disabled={busy || blocked || !system?.canTrain}>
          <Rocket />
          {t('train.setup.start')}
          <span className="setup-go-steps">· {t('train.setup.target_steps', { steps: steps(targetSteps) })}</span>
        </button>
      </div>
    </div>
  )
}

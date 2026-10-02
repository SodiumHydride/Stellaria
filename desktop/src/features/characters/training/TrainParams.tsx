import { Lock } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useBaseModels, useResources } from '../../../api/hooks'
import type { BaseModel, TrainConfig } from '../../../api/types'
import { Segmented, Switch, Tip } from '../../../components/ui'
import { useT } from '../../../i18n'
import { steps } from '../../../lib/format'

const STRUCTURAL: (keyof TrainConfig)[] = ['encoder', 'nLayers', 'nChans', 'nAuxLayers', 'nAuxChans', 'baseModelId']
export const isStructural = (k: keyof TrainConfig) => STRUCTURAL.includes(k)

/** Copy a base model's structure into a config: a base model decides the network shape. */
export function applyBase(config: TrainConfig, base: BaseModel | undefined): TrainConfig {
  if (!base) return { ...config, baseModelId: null }
  return { ...config, baseModelId: base.id, encoder: base.encoder, nLayers: base.nLayers, nChans: base.nChans, nAuxLayers: base.nAuxLayers, nAuxChans: base.nAuxChans }
}

/**
 * Every training parameter, grouped. `locked` freezes the network structure (after the first
 * checkpoint the shape can't change); with a base model the structure follows the base.
 */
export function TrainParams({ config, onChange, locked = false }: { config: TrainConfig; onChange: (next: TrainConfig) => void; locked?: boolean }) {
  const t = useT()
  const nav = useNavigate()
  const { data: resources } = useResources()
  const { data: bases } = useBaseModels()
  const hubertReady = resources?.find((r) => r.id === 'hubertsoft')?.status === 'ready'
  const base = bases?.find((b) => b.id === config.baseModelId)
  const structureLocked = locked || !!base
  const set = <K extends keyof TrainConfig>(k: K, v: TrainConfig[K]) => onChange({ ...config, [k]: v, preset: 'custom' })
  const num = (k: keyof TrainConfig, fallback: number, min = 0) => (e: React.ChangeEvent<HTMLInputElement>) => set(k, Math.max(min, Number(e.target.value) || fallback) as never)

  const lockTip = locked ? t('tp.locked_trained') : t('tp.locked_base')
  const Label = ({ text, hint, structural }: { text: string; hint?: string; structural?: boolean }) => (
    <span className="field-label tp-label">
      {hint ? (
        <Tip label={hint}>
          <span className="has-hint">{text}</span>
        </Tip>
      ) : (
        text
      )}
      {structural && structureLocked && (
        <Tip label={lockTip}>
          <Lock className="tp-lock" />
        </Tip>
      )}
    </span>
  )

  return (
    <div className="tp">
      <section className="tp-group">
        <div className="tp-title">{t('tp.base')}</div>
        <div className="tp-base">
          <select
            className="input"
            value={config.baseModelId ?? ''}
            disabled={locked}
            onChange={(e) => onChange(applyBase(config, bases?.find((b) => b.id === e.target.value)))}
          >
            <option value="">{t('tp.base_none')}</option>
            {(bases ?? []).map((b) => (
              <option key={b.id} value={b.id} disabled={b.status !== 'ready'}>
                {b.name} · {b.nLayers}×{b.nChans} · {steps(b.step)}
                {b.status !== 'ready' ? ` · ${t('tp.base_not_ready')}` : ''}
              </option>
            ))}
          </select>
          <button className="btn btn-sm btn-ghost" onClick={() => nav('/resources#base-models')}>
            {t('tp.base_manage')}
          </button>
        </div>
        {base?.notes && <p className="muted tp-note">{base.notes}</p>}
      </section>

      <section className="tp-group">
        <div className="tp-title">{t('tp.basics')}</div>
        <div className="tp-grid">
          <label className="field">
            <Label text={t('train.setup.adv.batch')} hint={t('tp.batch_hint')} />
            <input className="input tabular" type="number" min={1} max={512} value={config.batchSize} onChange={num('batchSize', 1, 1)} />
          </label>
          <label className="field">
            <Label text={t('train.setup.adv.lr')} hint={t('tp.lr_hint')} />
            <input className="input tabular" type="number" step={0.00005} min={0.000001} value={config.lr} onChange={num('lr', 0.0005, 0.000001)} />
          </label>
          <label className="field">
            <Label text={t('train.setup.adv.interval_val')} />
            <input className="input tabular" type="number" step={500} min={100} value={config.intervalVal} onChange={num('intervalVal', 2000, 100)} />
          </label>
          <label className="field">
            <Label text={t('train.setup.adv.interval_save')} />
            <input className="input tabular" type="number" step={1000} min={1000} value={config.intervalForceSave} onChange={num('intervalForceSave', 10000, 1000)} />
          </label>
        </div>
      </section>

      <section className="tp-group">
        <div className="tp-title">{t('tp.features')}</div>
        <div className="tp-grid">
          <label className="field">
            <Label text={t('train.setup.adv.encoder')} structural />
            <select className="input" value={config.encoder} disabled={structureLocked} onChange={(e) => set('encoder', e.target.value as TrainConfig['encoder'])}>
              <option value="contentvec768l12tta2x">ContentVec TTA ×2 ({t('common.recommended')})</option>
              <option value="contentvec768l12">ContentVec</option>
              <option value="hubertsoft" disabled={!hubertReady}>
                HuBERT-Soft{!hubertReady ? ` — ${t('train.setup.adv.encoder_missing')}` : ''}
              </option>
            </select>
          </label>
          <label className="field">
            <Label text={t('train.setup.adv.f0')} hint={t('tp.f0_hint')} />
            <select className="input" value={config.f0Extractor} onChange={(e) => set('f0Extractor', e.target.value as TrainConfig['f0Extractor'])}>
              {['rmvpe', 'fcpe', 'parselmouth', 'harvest', 'dio', 'crepe'].map((x) => (
                <option key={x} value={x}>
                  {x.toUpperCase()}
                  {x === 'rmvpe' ? ` (${t('common.recommended')})` : ''}
                </option>
              ))}
            </select>
          </label>
          <label className="row tp-switch">
            <Switch checked={config.usePitchAug} onChange={(v) => set('usePitchAug', v)} label={t('train.setup.adv.pitch_aug')} />
            <span>{t('train.setup.adv.pitch_aug')}</span>
          </label>
        </div>
      </section>

      <section className="tp-group">
        <div className="tp-title">{t('tp.structure')}</div>
        <div className="tp-grid">
          <label className="field">
            <Label text={t('tp.layers')} structural />
            <select className="input tabular" value={config.nLayers} disabled={structureLocked} onChange={(e) => set('nLayers', Number(e.target.value))}>
              {[3, 4, 6, 8, 10, 12].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <Label text={t('tp.channels')} structural />
            <select className="input tabular" value={config.nChans} disabled={structureLocked} onChange={(e) => set('nChans', Number(e.target.value))}>
              {[256, 384, 512, 768, 1024, 1536].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <Label text={t('tp.aux_layers')} hint={t('tp.aux_hint')} structural />
            <select className="input tabular" value={config.nAuxLayers} disabled={structureLocked} onChange={(e) => set('nAuxLayers', Number(e.target.value))}>
              {[2, 3, 4, 6, 8].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <Label text={t('tp.aux_channels')} structural />
            <select className="input tabular" value={config.nAuxChans} disabled={structureLocked} onChange={(e) => set('nAuxChans', Number(e.target.value))}>
              {[256, 384, 512, 768, 1024].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        </div>
      </section>

      <section className="tp-group">
        <div className="tp-title">{t('tp.performance')}</div>
        <div className="tp-grid">
          <label className="field">
            <Label text={t('train.setup.adv.amp')} hint={t('tp.amp_hint')} />
            <Segmented<TrainConfig['ampDtype']>
              size="sm"
              value={config.ampDtype}
              onChange={(v) => set('ampDtype', v)}
              options={[
                { value: 'fp16', label: 'FP16' },
                { value: 'bf16', label: 'BF16' },
                { value: 'fp32', label: 'FP32' },
              ]}
            />
          </label>
          <label className="field">
            <Label text={t('train.setup.adv.cache')} hint={t('tp.cache_hint')} />
            <Segmented<TrainConfig['cacheDevice']>
              size="sm"
              value={config.cacheDevice}
              onChange={(v) => set('cacheDevice', v)}
              options={[
                { value: 'cpu', label: t('tp.cache_ram') },
                { value: 'cuda', label: t('tp.cache_vram') },
              ]}
            />
          </label>
          <label className="field">
            <Label text={t('tp.workers')} />
            <input className="input tabular" type="number" min={0} max={16} value={config.numWorkers} onChange={(e) => set('numWorkers', Math.max(0, Number(e.target.value) || 0))} />
          </label>
          <label className="field">
            <Label text={t('tp.fused')} hint={t('tp.fused_hint')} />
            <Segmented<string>
              size="sm"
              value={String(config.useFusedKernels)}
              onChange={(v) => set('useFusedKernels', v === 'auto' ? 'auto' : v === 'true')}
              options={[
                { value: 'auto', label: t('tp.auto') },
                { value: 'true', label: t('common.on') },
                { value: 'false', label: t('common.off') },
              ]}
            />
          </label>
        </div>
      </section>
    </div>
  )
}

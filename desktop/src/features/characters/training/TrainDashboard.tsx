import * as RPopover from '@radix-ui/react-popover'
import clsx from 'clsx'
import { Check, FileText, Layers, Mic2, MoreHorizontal, Pause, Play, RotateCcw, SlidersHorizontal, Target, Wand2 } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../../../api/client'
import { useActiveJobs, useMetrics } from '../../../api/hooks'
import { subscribeTopic, useLive } from '../../../api/live'
import type { Character, Model } from '../../../api/types'
import { LogDrawer } from '../../../components/LogDrawer'
import { Logo } from '../../../components/Logo'
import { MetricChart } from '../../../components/MetricChart'
import { confirmDialog, Menu, Ring, Slider, Tip } from '../../../components/ui'
import { translate, useLang, useT, type StringKey } from '../../../i18n'
import { attempt } from '../../../lib/actions'
import { bytes, duration, number, steps } from '../../../lib/format'
import { TryDialog } from '../models/ModelsView'
import { CheckpointTrack } from './CheckpointTrack'
import { TrainParamsDialog } from './TrainParamsDialog'
import { ValidationCard } from './ValidationCard'

type ChartSeries = { tag: string; label: StringKey; colorVar: string }
const CHARTS: Record<'loss' | 'val' | 'snr', ChartSeries[]> = {
  loss: [
    { tag: 'train/loss', label: 'chart.series.total', colorVar: '--series-1' },
    { tag: 'train/ddsp_loss', label: 'chart.series.ddsp', colorVar: '--series-2' },
    { tag: 'train/reflow_loss', label: 'chart.series.reflow', colorVar: '--series-3' },
  ],
  val: [
    { tag: 'validation/loss', label: 'chart.series.total', colorVar: '--series-1' },
    { tag: 'validation/ddsp_loss', label: 'chart.series.ddsp', colorVar: '--series-2' },
    { tag: 'validation/reflow_loss', label: 'chart.series.reflow', colorVar: '--series-3' },
  ],
  snr: [
    { tag: 'validation/mel_val_snr', label: 'chart.series.snr', colorVar: '--series-1' },
    { tag: 'validation/mel_val_sisnr', label: 'chart.series.sisnr', colorVar: '--series-2' },
  ],
}

export function TrainDashboard({ model, character }: { model: Model; character: Character }) {
  const t = useT()
  const lang = useLang((s) => s.lang)
  // stable per language, so the charts only rebuild when their series really change
  const charts = useMemo(() => {
    const label = (list: ChartSeries[]) => list.map((s) => ({ ...s, label: translate(lang, s.label) }))
    return { loss: label(CHARTS.loss), val: label(CHARTS.val), snr: label(CHARTS.snr) }
  }, [lang])
  const job = useActiveJobs((j) => j.type === 'train' && j.target.modelId === model.id)[0]
  const { series } = useMetrics(model.id)
  const [smoothing, setSmoothing] = useState(() => Number(localStorage.getItem('stellaria.smoothing') ?? 0.6))
  const [showLog, setShowLog] = useState(false)
  const [listenStep, setListenStep] = useState<number | 'latest'>('latest')
  const [trying, setTrying] = useState(false)
  const [editing, setEditing] = useState(false)

  useEffect(() => localStorage.setItem('stellaria.smoothing', String(smoothing)), [smoothing])

  const running = model.status === 'training' || model.status === 'preparing' || model.status === 'preprocessing' || model.status === 'stopping' || !!job
  const progress = model.targetSteps > 0 ? Math.min(1, model.step / model.targetSteps) : 0
  const stage: 'prepare' | 'preprocess' | 'train' = model.status === 'preparing' ? 'prepare' : model.status === 'preprocessing' ? 'preprocess' : 'train'

  const pause = () => attempt(() => api.post(`/models/${model.id}/pause`))
  const resume = () => attempt(() => api.post(`/models/${model.id}/train`, {}))
  const fresh = async () => {
    const ok = await confirmDialog({ title: t('train.fresh_title'), body: t('train.fresh_body'), confirm: t('train.fresh'), cancel: t('common.cancel'), danger: true })
    if (ok) await attempt(() => api.post(`/models/${model.id}/train`, { fresh: true }))
  }
  const setTarget = (v: number) => attempt(() => api.patch(`/models/${model.id}`, { targetSteps: v }))
  const loadLog = useCallback(() => api.get<{ lines: string[] }>(`/models/${model.id}/log`, { tail: 1500 }).then((r) => r.lines), [model.id])

  return (
    <div className="dash">
      <div className="dash-status glass">
        <Ring value={progress} size={86} stroke={7}>
          <div className="dash-ring-text">
            {model.status === 'training' ? <span className="tabular">{Math.floor(progress * 100)}%</span> : running ? <Logo size={28} running /> : model.status === 'done' ? <Check /> : <span className="tabular">{Math.floor(progress * 100)}%</span>}
          </div>
        </Ring>
        <div className="dash-status-main">
          <div className="row" style={{ gap: 10 }}>
            <span className={clsx('status-dot', model.status === 'training' && 'active', model.status === 'done' && 'good', model.status === 'failed' && 'critical')} />
            <strong className="dash-status-label">{t(`train.status.${model.status}` as const)}</strong>
            {model.dataset.status === 'stale' && !running && <span className="chip warning">{t('train.dataset_stale')}</span>}
          </div>
          <div className="dash-steps tabular">
            {t('train.steps_of', { step: number(model.step), target: number(model.targetSteps) })}
          </div>
          <div className="dash-bar">
            <div className="dash-bar-fill" style={{ width: `${progress * 100}%` }}>
              <span className="dash-bar-star" />
            </div>
          </div>
          <div className="dash-meta tabular">
            {model.speed != null && model.status === 'training' && <span>{t('train.speed', { v: model.speed.toFixed(2) })}</span>}
            {model.eta != null && model.status === 'training' && <span>{t('train.eta', { time: duration(model.eta, t.lang) })}</span>}
            {model.trainingTime > 0 && <span>{t('train.elapsed', { time: duration(model.trainingTime, t.lang) })}</span>}
            {model.diskUsage && <span>{bytes(model.diskUsage.total)}</span>}
          </div>
          {model.error && <div className="dash-error">{t.msg(model.error)}</div>}
        </div>
        <div className="dash-actions">
          {model.status === 'training' ? (
            <Tip label={t('train.pause_hint')}>
              <button className="btn btn-lg" onClick={pause}>
                <Pause />
                {t('train.pause')}
              </button>
            </Tip>
          ) : running ? (
            <button className="btn btn-lg" disabled>
              <Logo size={18} running />
              {t(`train.status.${model.status}` as const)}
            </button>
          ) : (
            <button className="btn btn-primary btn-lg" onClick={model.step >= model.targetSteps ? () => setTarget(model.targetSteps + 50000).then(resume) : resume}>
              <Play />
              {model.step >= model.targetSteps ? t('train.more') : t('train.resume')}
            </button>
          )}
          <Tip label={t('train.try_hint')}>
            <span>
              <button className="btn btn-lg" onClick={() => setTrying(true)} disabled={model.checkpoints.length === 0}>
                <Mic2 />
                {t('models.try')}
              </button>
            </span>
          </Tip>
          <TargetPopover value={model.targetSteps} min={Math.max(1000, model.step + 1000)} onSave={setTarget} />
          <Tip label={t('tp.title')}>
            <button className="icon-btn" onClick={() => setEditing(true)}>
              <SlidersHorizontal />
            </button>
          </Tip>
          <Tip label={t('train.log')}>
            <button className="icon-btn" onClick={() => setShowLog(true)}>
              <FileText />
            </button>
          </Tip>
          <Menu
            trigger={
              <button className="icon-btn" aria-label={t('common.more')}>
                <MoreHorizontal />
              </button>
            }
            items={[{ label: t('train.fresh'), icon: <RotateCcw />, danger: true, onSelect: fresh, disabled: running }]}
          />
        </div>
      </div>

      {(model.status === 'preparing' || model.status === 'preprocessing') && (
        <div className="dash-stages glass">
          {(['prepare', 'preprocess', 'train'] as const).map((s, i) => (
            <div key={s} className={clsx('dash-stage', s === stage && 'on', ['prepare', 'preprocess', 'train'].indexOf(stage) > i && 'done')}>
              <span className="dash-stage-dot">{['prepare', 'preprocess', 'train'].indexOf(stage) > i ? <Check /> : s === 'train' ? <Wand2 /> : <Layers />}</span>
              <span>{t(`train.stage.${s}` as const)}</span>
              {s === stage && job?.progress != null && <span className="muted tabular">{Math.round(job.progress * 100)}%</span>}
            </div>
          ))}
        </div>
      )}

      <div className="dash-grid">
        <div className="dash-charts">
          <MetricChart
            title={t('chart.loss')}
            hint={t('chart.loss_hint')}
            smoothing={smoothing}
            data={series}
            emptyText={t('chart.waiting')}
            height={210}
            headerExtra={
              <div className="dash-smooth">
                <span className="muted">{t('chart.smoothing')}</span>
                <div style={{ width: 110 }}>
                  <Slider value={smoothing} min={0} max={0.98} step={0.01} onChange={setSmoothing} label={t('chart.smoothing')} />
                </div>
                <span className="tabular muted" style={{ width: 30 }}>
                  {smoothing.toFixed(2)}
                </span>
              </div>
            }
            series={charts.loss}
          />
          <div className="dash-charts-row">
            <MetricChart
              title={t('chart.val')}
              smoothing={0}
              data={series}
              emptyText={t('val.none', { step: steps(model.config.intervalVal) })}
              height={150}
              series={charts.val}
            />
            <MetricChart
              title={t('chart.snr')}
              hint={t('chart.snr_hint')}
              smoothing={0}
              unit="dB"
              data={series}
              emptyText={t('val.none', { step: steps(model.config.intervalVal) })}
              height={150}
              series={charts.snr}
            />
          </div>
        </div>
        <div className="dash-side">
          <GpuCard active={model.status === 'training'} />
          <ValidationCard model={model} character={character} step={listenStep} onStep={setListenStep} />
        </div>
      </div>

      <CheckpointTrack model={model} character={character} onListen={(s) => setListenStep(s)} />

      {editing && <TrainParamsDialog model={model} running={running} onClose={() => setEditing(false)} />}
      {trying && <TryDialog character={character} model={model} release={null} onClose={() => setTrying(false)} />}
      {showLog && <LogDrawer title={t('train.log')} source={loadLog} onClose={() => setShowLog(false)} />}
    </div>
  )
}

function TargetPopover({ value, min, onSave }: { value: number; min: number; onSave: (v: number) => void }) {
  const t = useT()
  const [v, setV] = useState(value)
  useEffect(() => setV(value), [value])
  return (
    <RPopover.Root>
      <Tip label={t('train.set_target')}>
        <RPopover.Trigger asChild>
          <button className="icon-btn">
            <Target />
          </button>
        </RPopover.Trigger>
      </Tip>
      <RPopover.Portal>
        <RPopover.Content className="popover" sideOffset={6} align="end" style={{ width: 240 }}>
          <div className="field">
            <span className="field-label">{t('train.target')}</span>
            <input className="input tabular" type="number" step={5000} min={min} value={v} onChange={(e) => setV(Number(e.target.value))} />
          </div>
          <div className="row" style={{ gap: 6, marginTop: 10, flexWrap: 'wrap' }}>
            {[50000, 100000, 150000, 200000].filter((x) => x >= min).map((x) => (
              <button key={x} className="chip button" onClick={() => setV(x)}>
                {steps(x)}
              </button>
            ))}
          </div>
          <RPopover.Close asChild>
            <button className="btn btn-primary btn-sm" style={{ marginTop: 12, width: '100%' }} onClick={() => onSave(Math.max(min, v))}>
              {t('common.save')}
            </button>
          </RPopover.Close>
        </RPopover.Content>
      </RPopover.Portal>
    </RPopover.Root>
  )
}

function GpuCard({ active }: { active: boolean }) {
  const t = useT()
  const gpus = useLive((s) => s.gpus)
  useEffect(() => subscribeTopic('gpu'), [])
  const g = gpus[0]
  if (!g) {
    return (
      <div className="gpu glass">
        <span className="section-title">{t('gpu.title')}</span>
        <span className="muted">{t('gpu.none')}</span>
      </div>
    )
  }
  const vram = g.vramUsed != null && g.vramTotal ? g.vramUsed / g.vramTotal : 0
  return (
    <div className={clsx('gpu glass', active && 'active')}>
      <div className="row">
        <span className="section-title">{t('gpu.title')}</span>
        <span className="spacer" />
        <span className="muted" style={{ fontSize: 'var(--fs-xs)' }}>
          {g.name.replace('NVIDIA ', '')}
        </span>
      </div>
      <div className="gpu-meters">
        <Meter label={t('gpu.util')} value={(g.utilization ?? 0) / 100} text={`${g.utilization ?? 0}%`} />
        <Meter label={t('gpu.vram')} value={vram} text={`${bytes(g.vramUsed ?? 0)}`} />
        <Meter label={t('gpu.temp')} value={Math.min(1, (g.temperature ?? 0) / 95)} text={g.temperature != null ? `${g.temperature}°C` : '–'} warn={(g.temperature ?? 0) >= 83} />
      </div>
    </div>
  )
}

function Meter({ label, value, text, warn }: { label: string; value: number; text: string; warn?: boolean }) {
  return (
    <div className="meter">
      <Ring value={value} size={54} stroke={5} color={warn ? 'var(--warning)' : undefined}>
        <span className="meter-text tabular">{text}</span>
      </Ring>
      <span className="muted">{label}</span>
    </div>
  )
}

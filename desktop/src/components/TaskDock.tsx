import { useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronUp,
  Cpu,
  Download,
  FileAudio,
  FileText,
  Layers,
  Pause,
  Scissors,
  ShieldCheck,
  Sparkles,
  Square,
  Trash2,
  Wand2,
  X,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { useJobs } from '../api/hooks'
import { keys } from '../api/keys'
import { onServerEvent, subscribeTopic, useLive } from '../api/live'
import type { Job, JobType, Model } from '../api/types'
import { isActiveJob } from '../api/types'
import { useT } from '../i18n'
import { attempt } from '../lib/actions'
import { bytes, duration, percent } from '../lib/format'
import { useUi } from '../state/ui'
import { LevelMark, Logo } from './Logo'
import { LogDrawer } from './LogDrawer'
import { Progress, Tip } from './ui'
import './dock.css'

const ICONS: Partial<Record<JobType, typeof Cpu>> = {
  import: FileAudio,
  organize: Scissors,
  health: ShieldCheck,
  prepare: Layers,
  preprocess: Layers,
  train: Sparkles,
  render: Wand2,
  separate: Layers,
  mixdown: FileAudio,
  download: Download,
  install_addon: Download,
}

/** mixing and exporting audio show the level meter; everything else the brand mark */
const MIXING: JobType[] = ['mix', 'mixdown']

function jobRank(j: Job): number {
  if (j.type === 'train' && j.status === 'running') return 0
  if (j.status === 'running') return 1
  if (j.status === 'stopping') return 2
  return 3
}

/** Tiny live sparkline of train/loss for the dock capsule. */
function useLossSpark(modelId: string | undefined) {
  const [points, setPoints] = useState<number[]>([])
  useEffect(() => {
    setPoints([])
    if (!modelId) return
    return onServerEvent((ev) => {
      if (ev.type === 'metrics' && ev.data.modelId === modelId && ev.data.scalars['train/loss'] !== undefined) {
        setPoints((p) => [...p.slice(-59), ev.data.scalars['train/loss']])
      }
    })
  }, [modelId])
  return points
}

function Spark({ values }: { values: number[] }) {
  if (values.length < 3) return null
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const d = values.map((v, i) => `${i === 0 ? 'M' : 'L'}${(i / (values.length - 1)) * 64},${18 - ((v - min) / span) * 16}`).join(' ')
  return (
    <svg className="dock-spark" width="64" height="20" viewBox="0 0 64 20" aria-hidden>
      <path d={d} fill="none" stroke="var(--series-1)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function TaskDock() {
  const t = useT()
  const nav = useNavigate()
  const qc = useQueryClient()
  const { data: jobs } = useJobs()
  const open = useUi((s) => s.dockOpen)
  const setOpen = useUi((s) => s.setDock)
  const gpus = useLive((s) => s.gpus)
  const [logJob, setLogJob] = useState<Job | null>(null)
  const ref = useRef<HTMLDivElement>(null)

  const active = useMemo(() => (jobs ?? []).filter(isActiveJob).sort((a, b) => jobRank(a) - jobRank(b)), [jobs])
  const recent = useMemo(() => (jobs ?? []).filter((j) => !isActiveJob(j)).slice(0, 12), [jobs])
  const hero = active[0]
  const heroModelId = hero?.type === 'train' ? hero.target.modelId : undefined
  const spark = useLossSpark(heroModelId)
  const queued = active.filter((j) => j.status === 'queued' || j.status === 'waiting_gpu').length

  useEffect(() => {
    if (!open) return
    const off = subscribeTopic('gpu')
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node) && !(e.target as HTMLElement).closest('.menu, .dialog, .log-drawer')) setOpen(false)
    }
    window.addEventListener('mousedown', onDown)
    return () => {
      off()
      window.removeEventListener('mousedown', onDown)
    }
  }, [open, setOpen])

  const goTo = (j: Job) => {
    const tab = j.type === 'train' || j.type === 'prepare' || j.type === 'preprocess' ? 'training' : 'material'
    if (j.target.projectId) nav(`/create/${j.target.projectId}`)
    else if (j.target.characterId) nav(`/voices/${j.target.characterId}/${tab}`)
    else if (j.target.modelId) {
      const m = qc.getQueryData<Model>(keys.model(j.target.modelId))
      const cid = m?.characters[0]?.characterId
      if (cid) nav(`/voices/${cid}/training`)
    } else if (j.target.resourceId) nav('/resources')
    setOpen(false)
  }

  const pauseTraining = (j: Job) => j.target.modelId && attempt(() => api.post(`/models/${j.target.modelId}/pause`))
  const cancel = (j: Job) => attempt(() => api.post(`/jobs/${j.id}/cancel`))

  const label = (j: Job) => t.msg(j.title, j.type)

  return (
    <div className={clsx('dock-wrap', open && 'open')} ref={ref}>
      {open && (
        <div className="dock-panel glass-strong">
          <div className="dock-panel-head">
            <span className="section-title">{t('dock.title')}</span>
            <span className="spacer" />
            {gpus.length > 0 && (
              <div className="dock-gpu">
                <Cpu />
                <span className="tabular">
                  {gpus[0].name.replace(/NVIDIA |GeForce /g, '')} · {gpus[0].utilization ?? 0}% · {bytes(gpus[0].vramUsed ?? 0)} / {bytes(gpus[0].vramTotal)}
                  {gpus[0].temperature != null && ` · ${gpus[0].temperature}°C`}
                </span>
              </div>
            )}
            {recent.length > 0 && (
              <button className="btn btn-sm btn-ghost" onClick={() => attempt(() => api.post('/jobs/clear')).then(() => qc.invalidateQueries({ queryKey: keys.jobs }))}>
                <Trash2 />
                {t('dock.clear')}
              </button>
            )}
          </div>
          <div className="dock-list">
            {active.length === 0 && recent.length === 0 && <div className="dock-empty">{t('dock.empty')}</div>}
            {[...active, ...recent].map((j) => {
              const Icon = ICONS[j.type] ?? Cpu
              const isActive = isActiveJob(j)
              return (
                <div key={j.id} className={clsx('dock-job', j.status)}>
                  <span className="dock-job-icon">
                    {j.status === 'failed' ? <AlertTriangle /> : j.status === 'succeeded' ? <CheckCircle2 /> : <Icon />}
                  </span>
                  <div className="dock-job-main" onClick={() => goTo(j)} role="button" tabIndex={0}>
                    <div className="row" style={{ gap: 6 }}>
                      <span className="dock-job-title">{label(j)}</span>
                      <span className="spacer" />
                      <span className="dock-job-meta tabular">
                        {j.status === 'running' && j.progress != null ? percent(j.progress) : t(`job.status.${j.status}` as const)}
                        {j.status === 'running' && j.eta != null && ` · ${t('dock.eta', { time: duration(j.eta, t.lang) })}`}
                      </span>
                    </div>
                    {j.stage && isActive && <div className="dock-job-stage">{t.msg(j.stage)}</div>}
                    {j.error && <div className="dock-job-error">{t.msg({ code: j.error.code, params: j.error.params }, j.error.message)}</div>}
                    {isActive && <Progress value={j.status === 'running' ? j.progress : j.status === 'stopping' ? null : 0} className="dock-job-bar" />}
                  </div>
                  <div className="dock-job-actions">
                    {j.type === 'train' && j.status === 'running' && (
                      <Tip label={t('common.pause')}>
                        <button className="icon-btn sm" onClick={() => pauseTraining(j)}>
                          <Pause />
                        </button>
                      </Tip>
                    )}
                    {(j.error || j.type === 'train' || j.type === 'preprocess') && (
                      <Tip label={t('dock.view_log')}>
                        <button className="icon-btn sm" onClick={() => setLogJob(j)}>
                          <FileText />
                        </button>
                      </Tip>
                    )}
                    {isActive && j.type !== 'train' && (
                      <Tip label={t('dock.cancel')}>
                        <button className="icon-btn sm" onClick={() => cancel(j)}>
                          <X />
                        </button>
                      </Tip>
                    )}
                    {isActive && j.type === 'train' && (
                      <Tip label={t('common.stop')}>
                        <button className="icon-btn sm" onClick={() => cancel(j)}>
                          <Square />
                        </button>
                      </Tip>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      <button className={clsx('dock glass-strong', hero && 'busy')} onClick={() => setOpen(!open)} aria-expanded={open}>
        {hero ? (
          <>
            <span className="dock-orb">
              {hero.status !== 'running' ? <Pause /> : MIXING.includes(hero.type) ? <LevelMark size={18} running /> : <Logo size={20} running />}
            </span>
            <span className="dock-text">
              <span className="dock-title">{label(hero)}</span>
              <span className={clsx('dock-sub tabular', hero.status === 'running' && 'shimmer-text')}>
                {hero.status === 'waiting_gpu'
                  ? t('dock.waiting_gpu')
                  : hero.stage
                    ? t.msg(hero.stage)
                    : t(`job.status.${hero.status}` as const)}
                {hero.progress != null && hero.status === 'running' && ` · ${percent(hero.progress)}`}
                {hero.eta != null && hero.status === 'running' && ` · ${t('dock.eta', { time: duration(hero.eta, t.lang) })}`}
              </span>
            </span>
            <Spark values={spark} />
            {active.length > 1 && <span className="chip accent">+{active.length - 1}</span>}
            {hero.progress != null && <span className="dock-progress" style={{ width: `${hero.progress * 100}%` }} />}
          </>
        ) : (
          <>
            <span className="dock-orb idle">
              <Logo size={18} />
            </span>
            <span className="dock-text">
              <span className="dock-title">{t('dock.idle')}</span>
            </span>
          </>
        )}
        {queued > 0 && !hero && <span className="chip">{t('dock.queued', { n: queued })}</span>}
        <ChevronUp className="dock-chevron" />
      </button>
      {logJob && <LogDrawer job={logJob} onClose={() => setLogJob(null)} />}
    </div>
  )
}

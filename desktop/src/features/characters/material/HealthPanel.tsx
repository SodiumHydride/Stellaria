import clsx from 'clsx'
import { AlertOctagon, AlertTriangle, CheckCircle2, Info, Loader2, RefreshCw, ShieldCheck } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../../api/client'
import type { Character, HealthIssue, HealthReport } from '../../../api/types'
import { Ring, Switch, Tip } from '../../../components/ui'
import { useT } from '../../../i18n'
import { healthDetails } from '../../../i18n/messages'
import { attempt } from '../../../lib/actions'
import { waveColor } from '../../../lib/color'
import { duration, hzToMidi, isBlackKey, midiToNote, relativeTime } from '../../../lib/format'
import { importMaterial } from '../importMaterial'

export const LEVEL_COLOR: Record<HealthReport['level'], string> = {
  excellent: 'var(--good)',
  good: 'var(--info)',
  fair: 'var(--warning)',
  poor: 'var(--serious)',
  blocked: 'var(--critical)',
}

/** Runs a health issue's one-click fix. */
export function useHealthFix(cid: string, onOrganize: () => void) {
  const nav = useNavigate()
  return async (issue: HealthIssue) => {
    const action = issue.fix?.action
    if (action === 'trash_clips') await attempt(() => api.patch(`/characters/${cid}/clips`, { ids: issue.clipIds, set: { status: 'trashed' } }))
    else if (action === 'pick_val') await attempt(() => api.post(`/characters/${cid}/clips/pick-val`, {}))
    else if (action === 'organize') onOrganize()
    else if (action === 'add_material') await importMaterial(cid)
    else if (action === 'install_addon') nav('/resources')
  }
}

/** The full health report: score, pitch range, and every issue with its fix. */
export function HealthReportView({
  character,
  report,
  running,
  runningProgress,
  onShowClips,
  onOrganize,
}: {
  character: Character
  report: HealthReport | null
  running: boolean
  runningProgress: number | null
  onShowClips: (ids: string[]) => void
  onOrganize: () => void
}) {
  const t = useT()
  const [deep, setDeep] = useState(false)
  const cid = character.id
  const stale = character.health?.stale ?? false
  const fix = useHealthFix(cid, onOrganize)

  const run = () => attempt(() => api.post(`/characters/${cid}/health`, { advanced: deep }))

  const errors = report?.issues.filter((i) => i.severity === 'error').length ?? 0

  return (
    <div className="health">
      <div className="row">
        <ShieldCheck className="health-title-icon" />
        <span className="section-title">{t('health.title')}</span>
        <span className="spacer" />
        <Tip label={t('health.recheck')}>
          <button className="icon-btn sm" onClick={run} disabled={running}>
            {running ? <Loader2 className="spin" /> : <RefreshCw />}
          </button>
        </Tip>
      </div>

      {!report ? (
        <div className="health-never">
          <p className="muted">{running ? t('health.running') : t('health.never')}</p>
          {!running && character.stats.clipCount > 0 && (
            <button className="btn btn-sm btn-primary" onClick={run}>
              {t('health.run')}
            </button>
          )}
        </div>
      ) : (
        <>
          <div className="health-score">
            <Ring value={running ? (runningProgress ?? 0) : report.score / 100} size={92} stroke={8} color={running ? undefined : LEVEL_COLOR[report.level]}>
              <div className="health-score-num">
                {running ? <Loader2 className="spin" style={{ width: 22, height: 22, color: 'var(--text-2)' }} /> : <span>{report.score}</span>}
              </div>
            </Ring>
            <div className="col" style={{ gap: 4 }}>
              <strong className="health-level" style={{ color: LEVEL_COLOR[report.level] }}>
                {t(`health.level.${report.level}` as const)}
              </strong>
              <span className="muted" style={{ fontSize: 'var(--fs-sm)' }}>
                {report.blocking ? t('health.blocking', { n: errors }) : report.issues.every((i) => i.severity === 'info') ? t('health.all_good') : ''}
              </span>
              <span className="muted" style={{ fontSize: 'var(--fs-xs)' }}>
                {stale ? t('health.stale') : t('health.checked', { time: relativeTime(report.checkedAt, t.lang) })}
              </span>
            </div>
          </div>

          {report.pitch.histogram.length > 0 && <PitchHistogram report={report} color={character.color} />}

          <div className="health-issues">
            {report.issues.map((issue) => (
              <IssueRow key={issue.code} issue={issue} onFix={() => fix(issue)} onShow={() => onShowClips(issue.clipIds)} />
            ))}
            {report.issues.length === 0 && (
              <div className="issue info">
                <CheckCircle2 className="issue-icon good" />
                <span>{t('health.all_good')}</span>
              </div>
            )}
          </div>
        </>
      )}

      <label className="health-deep">
        <Switch checked={deep} onChange={setDeep} label={t('health.deep')} />
        <Tip label={t('health.deep_hint')}>
          <span className="health-deep-label">{t('health.deep')}</span>
        </Tip>
      </label>
    </div>
  )
}

function IssueRow({ issue, onFix, onShow }: { issue: HealthIssue; onFix: () => void; onShow: () => void }) {
  const t = useT()
  const Icon = issue.severity === 'error' ? AlertOctagon : issue.severity === 'warning' ? AlertTriangle : Info
  const detail = healthDetails[issue.code]
  return (
    <div className={clsx('issue', issue.severity)}>
      <Icon className="issue-icon" />
      <div className="issue-body">
        {detail ? (
          <Tip label={t.lang === 'zh-CN' ? detail[0] : detail[1]}>
            <div className="issue-title has-detail">{t.msg({ code: issue.code, params: { count: issue.clipIds.length, ...issue.params } })}</div>
          </Tip>
        ) : (
          <div className="issue-title">{t.msg({ code: issue.code, params: { count: issue.clipIds.length, ...issue.params } })}</div>
        )}
        {(issue.fix || issue.clipIds.length > 0) && (
          <div className="row" style={{ gap: 6, marginTop: 8 }}>
            {issue.fix && (
              <button className={clsx('btn btn-sm', issue.severity === 'error' ? 'btn-primary' : '')} onClick={onFix}>
                {t(`fix.${issue.fix.action}` as const)}
              </button>
            )}
            {issue.clipIds.length > 0 && (
              <button className="btn btn-sm btn-ghost" onClick={onShow}>
                {t('fix.show_clips')} ({issue.clipIds.length})
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/** Voiced seconds per semitone, on a piano-key axis. */
function PitchHistogram({ report, color }: { report: HealthReport; color: string }) {
  const t = useT()
  const hist = report.pitch.histogram
  const lo = Math.min(...hist.map((h) => h.midi))
  const hi = Math.max(...hist.map((h) => h.midi))
  const start = Math.max(28, lo - 2)
  const end = Math.min(96, hi + 2)
  const max = Math.max(...hist.map((h) => h.seconds), 1e-6)
  const byMidi = new Map(hist.map((h) => [h.midi, h.seconds]))
  const keys = Array.from({ length: end - start + 1 }, (_, i) => start + i)
  const median = report.pitch.median ? Math.round(hzToMidi(report.pitch.median)) : null
  const [hover, setHover] = useState<number | null>(null)
  return (
    <div className="pitch">
      <div className="row">
        <span className="eyebrow">{t('health.pitch')}</span>
        <span className="spacer" />
        <span className="muted tabular" style={{ fontSize: 'var(--fs-xs)' }}>
          {hover != null
            ? `${midiToNote(hover)} · ${duration(byMidi.get(hover) ?? 0, t.lang)}`
            : report.pitch.low && report.pitch.high
              ? `${midiToNote(hzToMidi(report.pitch.low))} – ${midiToNote(hzToMidi(report.pitch.high))}`
              : ''}
        </span>
      </div>
      <div className="pitch-bars" onMouseLeave={() => setHover(null)}>
        {keys.map((k) => {
          const v = (byMidi.get(k) ?? 0) / max
          return (
            <div key={k} className={clsx('pitch-col', isBlackKey(k) && 'black', k === median && 'median')} onMouseEnter={() => setHover(k)}>
              <div className="pitch-bar" style={{ height: `${Math.max(v > 0 ? 6 : 0, v * 100)}%`, background: v > 0 ? waveColor(color) : undefined }} />
              <div className="pitch-key" />
            </div>
          )
        })}
      </div>
      <div className="pitch-axis">
        {keys
          .filter((k) => k % 12 === 0)
          .map((k) => (
            <span key={k} style={{ left: `${((k - start + 0.5) / keys.length) * 100}%` }}>
              {midiToNote(k)}
            </span>
          ))}
      </div>
    </div>
  )
}

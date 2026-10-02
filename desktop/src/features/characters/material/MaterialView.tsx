import * as RPopover from '@radix-ui/react-popover'
import clsx from 'clsx'
import { AlertOctagon, AlertTriangle, ChevronDown, FolderOpen, Loader2, Music4, Scissors, Upload } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useActiveJobs, useClips, useHealth, useRaw } from '../../../api/hooks'
import { onServerEvent } from '../../../api/live'
import type { Character, HealthIssue } from '../../../api/types'
import { Empty, Menu, Progress } from '../../../components/ui'
import { useT } from '../../../i18n'
import { duration, hzToMidi, midiToNote, percent } from '../../../lib/format'
import { importMaterial } from '../importMaterial'
import { ClipList } from './ClipList'
import { HealthReportView, LEVEL_COLOR, useHealthFix } from './HealthPanel'
import { OrganizeDialog } from './OrganizeDialog'
import { ReviewMode } from './ReviewMode'
import { SourceList } from './SourceList'
import './material.css'

/** Recommended amount of clean singing; the meter fills toward it. */
const TARGET_SECONDS = 30 * 60
const MIN_SECONDS = 5 * 60

/**
 * 素材: one summary bar (how much, how healthy, what to do next), the files it came from on the
 * left, and the clips on the right. Problems that need attention surface as one-line banners.
 */
export function MaterialView({ character }: { character: Character }) {
  const t = useT()
  const id = character.id
  const { data: raw } = useRaw(id)
  const { data: clips } = useClips(id, 'all')
  const { data: health } = useHealth(id)
  const jobs = useActiveJobs((j) => j.target.characterId === id)
  const organizeJob = jobs.find((j) => j.type === 'organize' || j.type === 'import')
  const healthJob = jobs.find((j) => j.type === 'health')
  const [organizeOpen, setOrganizeOpen] = useState(false)
  const [reviewing, setReviewing] = useState(false)
  const [focusIds, setFocusIds] = useState<string[] | null>(null)
  const [source, setSource] = useState<string | null>(null)

  useEffect(() => {
    const open = () => setOrganizeOpen(true)
    window.addEventListener('stellaria:organize', open)
    return () => window.removeEventListener('stellaria:organize', open)
  }, [])

  // first import for a voice with no clips yet: offer to organize right away
  const noClips = (clips?.length ?? 0) === 0
  useEffect(
    () =>
      onServerEvent((ev) => {
        if (ev.type === 'job' && ev.data.type === 'import' && ev.data.status === 'succeeded' && ev.data.target.characterId === id && noClips) setOrganizeOpen(true)
      }),
    [id, noClips],
  )

  const pending = (raw ?? []).filter((r) => r.status === 'imported' || r.status === 'failed').length
  const hasAnything = (raw?.length ?? 0) > 0 || (clips?.length ?? 0) > 0
  const shownClips = useMemo(() => (source ? (clips ?? []).filter((c) => c.rawId === source) : (clips ?? [])), [clips, source])

  if (raw && clips && !hasAnything && !organizeJob) {
    return (
      <div className="mat-hero glass">
        <Empty icon={<Music4 />} title={t('mat.drop.title')} body={t('mat.drop.body')}>
          <button className="btn btn-primary" onClick={() => importMaterial(id)}>
            <Upload />
            {t('mat.import')}
          </button>
          <button className="btn" onClick={() => importMaterial(id, { directory: true })}>
            <FolderOpen />
            {t('mat.import_folder')}
          </button>
        </Empty>
      </div>
    )
  }

  const banners = (health?.issues ?? []).filter((i) => i.severity !== 'info' && (i.fix || i.clipIds.length > 0))

  return (
    <div className="mat">
      <div className="mat-summary glass">
        <Meter seconds={character.stats.activeDuration} />
        <Stat label={t('health.clips')} value={String(character.stats.clipCount)} />
        <Stat label={t('health.val')} value={String(health?.totals.valCount ?? '–')} />
        <Stat
          label={t('mat.stat.range')}
          value={health?.pitch.low && health.pitch.high ? `${midiToNote(hzToMidi(health.pitch.low))}–${midiToNote(hzToMidi(health.pitch.high))}` : '–'}
        />
        <HealthChip character={character} running={!!healthJob} runningProgress={healthJob?.progress ?? null} onShowClips={setFocusIds} onOrganize={() => setOrganizeOpen(true)} />
        <span className="spacer" />
        {organizeJob ? (
          <div className="mat-busy">
            <div className="row" style={{ gap: 8 }}>
              <Loader2 className="spin" />
              <span className="mat-busy-text">{organizeJob.stage ? t.msg(organizeJob.stage) : t.msg(organizeJob.title, t('mat.organizing'))}</span>
              <span className="muted tabular">{organizeJob.progress != null ? percent(organizeJob.progress) : ''}</span>
            </div>
            <Progress value={organizeJob.progress} />
          </div>
        ) : (
          <>
            <Menu
              trigger={
                <button className="btn">
                  <Upload />
                  {t('common.import')}
                  <ChevronDown className="btn-caret" />
                </button>
              }
              items={[
                { label: t('mat.import'), icon: <Upload />, onSelect: () => importMaterial(id) },
                { label: t('mat.import_folder'), icon: <FolderOpen />, onSelect: () => importMaterial(id, { directory: true }) },
              ]}
            />
            <button className={clsx('btn', pending > 0 && 'btn-primary')} onClick={() => setOrganizeOpen(true)} disabled={(raw?.length ?? 0) === 0}>
              <Scissors />
              {pending > 0 ? t('mat.organize_n', { n: pending }) : t('mat.organize')}
            </button>
          </>
        )}
      </div>

      {banners.length > 0 && !focusIds && <IssueBanners character={character} issues={banners} onShowClips={setFocusIds} onOrganize={() => setOrganizeOpen(true)} />}

      <div className="mat-body">
        <SourceList characterId={id} raw={raw ?? []} clips={clips ?? []} selected={source} onSelect={setSource} />
        <ClipList
          character={character}
          clips={shownClips}
          loading={!clips}
          focusIds={focusIds}
          onClearFocus={() => setFocusIds(null)}
          onReview={() => setReviewing(true)}
        />
      </div>

      <OrganizeDialog character={character} raw={raw ?? []} open={organizeOpen} onOpenChange={setOrganizeOpen} />
      {reviewing && clips && <ReviewMode character={character} clips={shownClips} onExit={() => setReviewing(false)} />}
    </div>
  )
}

function Meter({ seconds }: { seconds: number }) {
  const t = useT()
  const frac = Math.min(1, seconds / TARGET_SECONDS)
  const level = seconds < MIN_SECONDS ? 'low' : seconds < TARGET_SECONDS ? 'ok' : 'full'
  return (
    <div className="mat-meter">
      <div className="mat-meter-top">
        <strong className="tabular">{duration(seconds, t.lang)}</strong>
        <span className="muted">{t('mat.meter.target', { duration: duration(TARGET_SECONDS, t.lang) })}</span>
      </div>
      <div className={clsx('mat-meter-bar', level)}>
        <span style={{ width: `${frac * 100}%` }} />
        <i style={{ left: `${(MIN_SECONDS / TARGET_SECONDS) * 100}%` }} />
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="mat-stat">
      <span className="eyebrow">{label}</span>
      <strong className="tabular">{value}</strong>
    </div>
  )
}

/** Score chip; the full report (pitch range, every issue, deep check) opens in a popover. */
function HealthChip({
  character,
  running,
  runningProgress,
  onShowClips,
  onOrganize,
}: {
  character: Character
  running: boolean
  runningProgress: number | null
  onShowClips: (ids: string[]) => void
  onOrganize: () => void
}) {
  const t = useT()
  const { data: report } = useHealth(character.id)
  const [open, setOpen] = useState(false)
  const color = report ? LEVEL_COLOR[report.level] : 'var(--text-3)'
  return (
    <RPopover.Root open={open} onOpenChange={setOpen}>
      <RPopover.Trigger asChild>
        <button className="mat-health" style={{ ['--level' as string]: color }}>
          <span className="eyebrow">{t('health.title')}</span>
          <span className="mat-health-value">
            {running ? (
              <Loader2 className="spin" />
            ) : report ? (
              <>
                <span className="mat-health-dot" />
                <strong className="tabular">{report.score}</strong>
                <span className="mat-health-level">{t(`health.level.${report.level}` as const)}</span>
              </>
            ) : (
              <span className="muted">–</span>
            )}
            <ChevronDown className="btn-caret" />
          </span>
        </button>
      </RPopover.Trigger>
      <RPopover.Portal>
        <RPopover.Content className="popover health-pop" sideOffset={8} align="start" collisionPadding={16} onOpenAutoFocus={(e) => e.preventDefault()}>
          <HealthReportView
            character={character}
            report={report ?? null}
            running={running}
            runningProgress={runningProgress}
            onShowClips={(ids) => {
              setOpen(false)
              onShowClips(ids)
            }}
            onOrganize={() => {
              setOpen(false)
              onOrganize()
            }}
          />
        </RPopover.Content>
      </RPopover.Portal>
    </RPopover.Root>
  )
}

function IssueBanners({ character, issues, onShowClips, onOrganize }: { character: Character; issues: HealthIssue[]; onShowClips: (ids: string[]) => void; onOrganize: () => void }) {
  const t = useT()
  const fix = useHealthFix(character.id, onOrganize)
  return (
    <div className="mat-banners">
      {issues.slice(0, 3).map((issue) => {
        const Icon = issue.severity === 'error' ? AlertOctagon : AlertTriangle
        return (
          <div key={issue.code} className={clsx('mat-banner', issue.severity)}>
            <Icon />
            <span className="mat-banner-text">{t.msg({ code: issue.code, params: { count: issue.clipIds.length, ...issue.params } })}</span>
            {issue.clipIds.length > 0 && (
              <button className="btn btn-sm btn-ghost" onClick={() => onShowClips(issue.clipIds)}>
                {t('fix.show_clips')}
              </button>
            )}
            {issue.fix && (
              <button className="btn btn-sm" onClick={() => fix(issue)}>
                {t(`fix.${issue.fix.action}` as const)}
              </button>
            )}
          </div>
        )
      })}
    </div>
  )
}

import clsx from 'clsx'
import { Brain, HardDrive, Loader2, Mic2, MoreHorizontal, Package, Scissors, Sparkles, Trash2, Upload, Users } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, sendFiles } from '../../../api/client'
import { useCharacters, useJobs, useModels } from '../../../api/hooks'
import type { Character, Job, Model, Release } from '../../../api/types'
import { Waveform } from '../../../components/Waveform'
import { confirmDialog, Dialog, Empty, Menu, Slider } from '../../../components/ui'
import { useT } from '../../../i18n'
import { attempt } from '../../../lib/actions'
import { deliverExport } from '../../../lib/exportJob'
import { bytes, relativeTime, signed, steps } from '../../../lib/format'
import { isTauri, pickFiles, pickSavePath } from '../../../shell/bridge'
import { modelLabel } from '../training/TrainingView'
import './models.css'

export function ModelsView({ character, onCreate }: { character: Character; onCreate: () => void }) {
  const t = useT()
  const nav = useNavigate()
  const { data: models } = useModels(character.id)
  const { data: characters } = useCharacters()
  const [tryRelease, setTryRelease] = useState<{ model: Model; release: Release } | null>(null)
  const [slimModel, setSlimModel] = useState<Model | null>(null)
  const withReleases = (models ?? []).filter((m) => m.releases.length > 0)

  const exportRelease = async (m: Model, r: Release) => {
    const path = isTauri ? await pickSavePath(`${r.name}.stella`, ['stella']) : undefined
    if (isTauri && !path) return
    const job = await attempt(() => api.post<Job>(`/models/${m.id}/export`, { releaseId: r.id, path: path ?? undefined }))
    await deliverExport(job, { done: t('common.exported'), failed: t('common.error') })
  }

  const removeRelease = async (m: Model, r: Release) => {
    const ok = await confirmDialog({ title: t('models.delete_title', { name: r.name }), body: t('models.delete_body'), confirm: t('common.delete'), cancel: t('common.cancel'), danger: true })
    if (ok) await attempt(() => api.del(`/models/${m.id}/releases/${r.id}`))
  }

  return (
    <div className="models">
      <div className="models-head">
        <h2 className="section-title" style={{ fontSize: 'var(--fs-xl)' }}>
          {t('models.title')}
        </h2>
      </div>

      {withReleases.length === 0 ? (
        <div className="glass" style={{ marginTop: 16 }}>
          <Empty icon={<Sparkles />} title={t('models.none')} body={t('models.none_body')}>
            <button className="btn btn-primary" onClick={() => nav(`/voices/${character.id}/training`)}>
              <Brain />
              {t('models.go_training')}
            </button>
          </Empty>
        </div>
      ) : (
        withReleases.map((m) => (
          <section key={m.id} className="models-group">
            {(models ?? []).length > 1 && (
              <div className="models-group-title">
                {m.kind === 'constellation' && <Users />}
                {modelLabel(m, characters, t)}
              </div>
            )}
            <div className="releases">
              {m.releases
                .slice()
                .sort((a, b) => b.step - a.step)
                .map((r, i) => (
                  <div key={r.id} className={clsx('release glass', i === 0 && 'newest')} style={{ ['--voice' as string]: character.color }}>
                    <div className="release-orb">
                      <Sparkles />
                    </div>
                    <div className="release-main">
                      <div className="release-name">{r.name}</div>
                      <div className="muted release-meta tabular">
                        {t('models.step', { steps: steps(r.step) })} · {relativeTime(r.createdAt, t.lang)} · {bytes(r.size)}
                      </div>
                      <ReleaseNotes model={m} release={r} />
                    </div>
                    <button className="btn btn-sm" onClick={() => setTryRelease({ model: m, release: r })}>
                      <Mic2 />
                      {t('models.try')}
                    </button>
                    <button className="btn btn-sm btn-primary" onClick={onCreate}>
                      <Sparkles />
                      {t('models.create')}
                    </button>
                    <Menu
                      trigger={
                        <button className="icon-btn sm" aria-label={t('common.more')}>
                          <MoreHorizontal />
                        </button>
                      }
                      items={[
                        { label: t('models.export'), icon: <Package />, onSelect: () => exportRelease(m, r) },
                        { separator: true, label: '' },
                        { label: t('common.delete'), icon: <Trash2 />, danger: true, onSelect: () => removeRelease(m, r) },
                      ]}
                    />
                  </div>
                ))}
            </div>
          </section>
        ))
      )}

      {(models ?? []).some((m) => m.diskUsage) && (
        <section className="models-group">
          <div className="models-group-title">
            <HardDrive />
            {t('models.storage')}
          </div>
          {(models ?? []).map((m) =>
            m.diskUsage ? (
              <div key={m.id} className="storage glass">
                <span className="storage-name">{modelLabel(m, characters, t)}</span>
                <StorageBar usage={m.diskUsage} />
                <span className="tabular">{bytes(m.diskUsage.total)}</span>
                <button className="btn btn-sm" onClick={() => setSlimModel(m)}>
                  <Scissors />
                  {t('models.slim')}
                </button>
              </div>
            ) : null,
          )}
        </section>
      )}

      {tryRelease && <TryDialog character={character} model={tryRelease.model} release={tryRelease.release} onClose={() => setTryRelease(null)} />}
      {slimModel && <SlimDialog model={slimModel} name={modelLabel(slimModel, characters, t)} onClose={() => setSlimModel(null)} />}
    </div>
  )
}

function StorageBar({ usage }: { usage: NonNullable<Model['diskUsage']> }) {
  const t = useT()
  const total = Math.max(1, usage.total)
  const parts = [
    { k: 'dataset', v: usage.dataset, label: t('models.storage.dataset'), color: 'var(--series-1)' },
    { k: 'checkpoints', v: usage.checkpoints, label: t('models.storage.checkpoints'), color: 'var(--series-3)' },
    { k: 'releases', v: usage.releases, label: t('models.storage.releases'), color: 'var(--series-2)' },
  ]
  return (
    <div className="storage-bar-wrap">
      <div className="storage-bar">
        {parts.map((p) => (
          <span key={p.k} style={{ width: `${(p.v / total) * 100}%`, background: p.color }} title={`${p.label} · ${bytes(p.v)}`} />
        ))}
      </div>
      <div className="storage-legend">
        {parts.map((p) => (
          <span key={p.k}>
            <i style={{ background: p.color }} />
            {p.label} <span className="tabular muted">{bytes(p.v)}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

/** Quick "try it": pick a short clip, convert with this version, listen. */
/** Quick render of one clip with a release — or, with `release` null, the newest checkpoint (works mid-training). */
export function TryDialog({ character, model, release, onClose }: { character: Character; model: Model; release: Release | null; onClose: () => void }) {
  const t = useT()
  const { data: jobs } = useJobs()
  const [key, setKey] = useState(0)
  const [jobId, setJobId] = useState<string | null>(null)
  const job = jobs?.find((j) => j.id === jobId)
  const resultUrl = job?.status === 'succeeded' ? (job.result as { url?: string } | null)?.url : null

  useEffect(() => setJobId(null), [release?.id])

  const pick = async () => {
    const picked = await pickFiles({ multiple: false })
    if (!picked || picked.length === 0) return
    const extra = { characterId: character.id, releaseId: release?.id ?? null, params: { key } }
    const j = await attempt(() =>
      typeof picked[0] === 'string'
        ? api.post<Job>(`/models/${model.id}/quick-render`, { path: picked[0], ...extra })
        : sendFiles<Job>(`/models/${model.id}/quick-render`, picked as File[], extra),
    )
    if (j) setJobId(j.id)
  }

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title={release ? t('models.try_title', { name: release.name }) : t('models.try_latest_title', { steps: steps(model.step) })}
      description={t('models.try_body')}
    >
      <div className="col" style={{ gap: 16 }}>
        <div className="field">
          <div className="row">
            <span className="field-label">{t('models.try_key')}</span>
            <span className="spacer" />
            <span className="tabular">{signed(key)}</span>
          </div>
          <Slider value={key} min={-12} max={12} step={1} onChange={setKey} centerMark label={t('models.try_key')} />
        </div>
        <button className="btn btn-primary" onClick={pick} disabled={!!job && job.status !== 'succeeded' && job.status !== 'failed'}>
          {job && (job.status === 'running' || job.status === 'queued' || job.status === 'waiting_gpu') ? <Loader2 className="spin" /> : <Upload />}
          {job && job.status === 'running' ? t('models.try_running') : t('models.try_pick')}
        </button>
        {job && job.status !== 'succeeded' && job.status !== 'failed' && job.stage && <div className="muted try-stage">{t.msg(job.stage)}</div>}
        {job?.status === 'failed' && <div className="chip critical">{t.msg({ code: job.error?.code ?? '' }, job.error?.message)}</div>}
        {resultUrl && (
          <div className="col" style={{ gap: 6 }}>
            <span className="eyebrow">{t('models.try_result')}</span>
            <Waveform id={`try:${jobId}`} peaksPath={resultUrl.replace(/^\/api/, '').replace(/\/audio$/, '/peaks')} audioPath={resultUrl} height={56} color={character.color} showButton />
          </div>
        )}
      </div>
    </Dialog>
  )
}

function SlimDialog({ model, name, onClose }: { model: Model; name: string; onClose: () => void }) {
  const t = useT()
  const [dataset, setDataset] = useState(true)
  const [ckpt, setCkpt] = useState<'none' | 'keep_latest' | 'releases_only'>('keep_latest')
  const go = async () => {
    await attempt(() => api.post(`/models/${model.id}/slim`, { dataset, checkpoints: ckpt }))
    onClose()
  }
  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title={t('models.slim_title', { name })}
      description={t('models.slim_body')}
      actions={
        <>
          <button className="btn btn-ghost" onClick={onClose}>
            {t('common.cancel')}
          </button>
          <button className="btn btn-primary" onClick={go}>
            <Scissors />
            {t('models.slim.go')}
          </button>
        </>
      }
    >
      <div className="col" style={{ gap: 10 }}>
        <label className={clsx('org-choice', dataset && 'on')}>
          <input type="checkbox" checked={dataset} onChange={(e) => setDataset(e.target.checked)} />
          {t('models.slim.dataset')}
        </label>
        {(['none', 'keep_latest', 'releases_only'] as const).map((k) => (
          <label key={k} className={clsx('org-choice', ckpt === k && 'on')}>
            <input type="radio" checked={ckpt === k} onChange={() => setCkpt(k)} />
            {t(k === 'none' ? 'models.slim.ckpt_none' : k === 'keep_latest' ? 'models.slim.ckpt_latest' : 'models.slim.ckpt_releases')}
          </label>
        ))}
      </div>
    </Dialog>
  )
}

/** A one-line note per published version ("breathier", "best for high notes"), edited in place. */
function ReleaseNotes({ model, release }: { model: Model; release: Release }) {
  const t = useT()
  const [value, setValue] = useState(release.notes)
  useEffect(() => setValue(release.notes), [release.notes])
  const save = () => {
    const notes = value.trim()
    if (notes !== release.notes) void attempt(() => api.patch(`/models/${model.id}/releases/${release.id}`, { notes }))
  }
  return (
    <input
      className="release-notes"
      value={value}
      placeholder={t('models.notes_placeholder')}
      onChange={(e) => setValue(e.target.value)}
      onBlur={save}
      onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
    />
  )
}

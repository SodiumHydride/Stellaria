import { useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { AudioLines, Boxes, CheckCircle2, Cpu, Download, FolderInput, FolderOpen, HardDrive, Layers, Loader2, Package, Plus, RotateCcw, Trash2, Wrench } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { api } from '../../api/client'
import { useBaseModels, useCharacters, useCleanupModels, useDisk, useModels, usePreferences, useProjects, useResources, useSeparationModels, useSystem } from '../../api/hooks'
import { keys } from '../../api/keys'
import type { BaseModel, Job, Preferences, Resource } from '../../api/types'
import { confirmDialog, Dialog, Progress, Segmented, Tip } from '../../components/ui'
import { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { waitForJob } from '../../lib/jobs'
import { bytes, speed, steps } from '../../lib/format'
import { isTauri, listen, pickFiles, pickFolder, shell, type BootstrapProgress } from '../../shell/bridge'
import { importModelsFromFolder } from './importFolder'
import { CleanupModelPicker, ImportSeparationButton, SeparationModelPicker } from './SeparationModels'
import { importPack } from '../characters/training/TrainingView'
import './resources.css'

export async function setPreference(qc: ReturnType<typeof useQueryClient>, patch: Partial<Preferences>) {
  const p = await attempt(() => api.put<Preferences>('/preferences', patch))
  if (p) qc.setQueryData(keys.preferences, p)
}

export default function ResourcesView() {
  const t = useT()
  const qc = useQueryClient()
  const { data: resources } = useResources()
  const { data: prefs } = usePreferences()
  const { data: separation } = useSeparationModels()
  const { data: cleanup } = useCleanupModels()
  // separation and cleanup checkpoints live in their own section, with the model pickers
  const separationIds = new Set([...(separation?.models ?? []), ...(cleanup?.models ?? [])].map((m) => m.resourceId))
  const models = (resources ?? []).filter((r) => r.kind === 'model' && !separationIds.has(r.id))
  const addons = (resources ?? []).filter((r) => r.kind === 'addon')
  const missingRequired = models.some((r) => r.required && r.status === 'missing')
  const { hash } = useLocation()
  useEffect(() => {
    if (!hash) return
    const id = hash.slice(1)
    window.setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120)
  }, [hash])

  return (
    <div className="page">
      <div className="page-inner">
        <div className="page-header">
          <h1 className="page-title">{t('res.title')}</h1>
          <span className="spacer" />
          {prefs && (
            <Tip label={t('res.region')}>
              <span>
                <Segmented<Preferences['mirrorRegion']>
                size="sm"
                value={prefs.mirrorRegion}
                onChange={(v) => setPreference(qc, { mirrorRegion: v })}
                options={[
                  { value: 'auto', label: t('res.region.auto') },
                  { value: 'cn', label: t('res.region.cn') },
                  { value: 'global', label: t('res.region.global') },
                ]}
                />
              </span>
            </Tip>
          )}
        </div>

        <RuntimeCard />

        <section className="res-section">
          <div className="res-section-head">
            <Boxes />
            <span className="section-title">{t('res.models')}</span>
            <span className="spacer" />
            {isTauri && models.some((r) => r.status !== 'ready') && (
              <Tip label={t('ob.models.folder_hint')}>
                <button
                  className="btn btn-sm btn-ghost"
                  onClick={async () => {
                    const path = await pickFolder()
                    if (path) await importModelsFromFolder(path, resources ?? [])
                  }}
                >
                  <FolderInput />
                  {t('ob.models.pick_folder')}
                </button>
              </Tip>
            )}
            {missingRequired && (
              <button className="btn btn-sm btn-primary" onClick={() => attempt(() => api.post('/resources/install-required'))}>
                <Download />
                {t('res.install_all')}
              </button>
            )}
          </div>
          <div className="res-list">
            {models.map((r) => (
              <ResourceRow key={r.id} r={r} />
            ))}
          </div>
        </section>

        <section className="res-section" id="separation">
          <div className="res-section-head">
            <AudioLines />
            <span className="section-title">{t('res.separation')}</span>
            <span className="spacer" />
            {isTauri && <ImportSeparationButton />}
          </div>
          <div className="sep-cards">
            <div className="res-card glass">
              <h3 className="sep-card-title">{t('sep.model')}</h3>
              <p className="secondary res-sep-hint">{t('sep.model_hint')}</p>
              <SeparationModelPicker value={prefs?.separationModel ?? 'auto'} onChange={(id) => void setPreference(qc, { separationModel: id === 'auto' ? null : id })} />
            </div>
            <div className="res-card glass" id="cleanup">
              <h3 className="sep-card-title">{t('cln.title')}</h3>
              <p className="secondary res-sep-hint">{t('cln.model_hint')}</p>
              <CleanupModelPicker value={prefs?.cleanupModel ?? 'auto'} onChange={(id) => void setPreference(qc, { cleanupModel: id === 'auto' ? null : id })} />
            </div>
          </div>
        </section>

        <section className="res-section">
          <div className="res-section-head">
            <Wrench />
            <span className="section-title">{t('res.addons')}</span>
          </div>
          <div className="res-list">
            {addons.map((r) => (
              <ResourceRow key={r.id} r={r} />
            ))}
          </div>
        </section>

        <section className="res-section">
          <div className="res-section-head">
            <Package />
            <span className="section-title">{t('res.packs')}</span>
          </div>
          <div className="res-card glass row" style={{ gap: 16 }}>
            <span className="secondary" style={{ flex: 1 }}>
              {t('res.packs_body')}
            </span>
            <button className="btn" onClick={importPack}>
              <Package />
              {t('res.packs_import')}
            </button>
          </div>
        </section>

        <BaseModelsSection />

        <StorageSection />
      </div>
    </div>
  )
}

function RuntimeCard() {
  const t = useT()
  const { data: system } = useSystem()
  const [progress, setProgress] = useState<BootstrapProgress | null>(null)

  useEffect(() => {
    if (!isTauri) return
    let off: (() => void) | undefined
    void listen<BootstrapProgress>('bootstrap://progress', (p) => setProgress(p)).then((f) => (off = f))
    return () => off?.()
  }, [])

  const repairing = progress && !['done', 'error', 'cancelled'].includes(progress.stage)
  const torch = system?.torch
  return (
    <section className="res-section">
      <div className="res-section-head">
        <Cpu />
        <span className="section-title">{t('res.runtime')}</span>
      </div>
      <div className="res-card glass">
        <div className="runtime-grid">
          <div>
            <span className="eyebrow">{t('res.runtime.torch')}</span>
            <strong>{torch?.version ?? '–'}</strong>
          </div>
          <div>
            <span className="eyebrow">{t('res.runtime.device')}</span>
            <strong>{torch ? t(`res.runtime.device.${torch.device}` as const) : '–'}</strong>
            {torch?.cuda && <span className="muted"> · CUDA {torch.cuda}</span>}
          </div>
          <div>
            <span className="eyebrow">{t('res.runtime.fused')}</span>
            <strong>{torch?.triton ? t('common.on') : t('common.off')}</strong>
          </div>
          <div>
            <span className="eyebrow">GPU</span>
            <strong>{system?.gpus[0]?.name ?? t('gpu.none')}</strong>
          </div>
        </div>
        {isTauri && (
          <div className="runtime-repair">
            <span className="muted" style={{ flex: 1 }}>
              {repairing ? `${t(`ob.install.stage.${progress!.stage}` as 'ob.install.stage.prepare')} · ${progress!.detail ?? progress!.message}` : ''}
            </span>
            {repairing ? (
              <div style={{ width: 200 }}>
                <Progress value={progress!.overall} />
              </div>
            ) : (
              <button className="btn btn-sm" onClick={() => shell.bootstrapRepair()}>
                <RotateCcw />
                {t('res.runtime.repair')}
              </button>
            )}
          </div>
        )}
      </div>
    </section>
  )
}

async function pickLocalPath(extensions: string[]): Promise<string | null> {
  if (!isTauri) return window.prompt('Path')
  const picked = await pickFiles({ multiple: false, extensions })
  return Array.isArray(picked) && typeof picked[0] === 'string' ? picked[0] : null
}

function ResourceRow({ r }: { r: Resource }) {
  const t = useT()
  const importLocal = async (res: Resource) => {
    const path = await pickLocalPath(['pt', 'ckpt', 'bin', 'pth', 'zip', 'onnx'])
    if (path) await attempt(() => api.post(`/resources/${res.id}/import`, { path }))
  }
  const busy = r.status === 'downloading' || r.status === 'installing'
  return (
    <div className={clsx('res-row glass', r.status)}>
      <div className="res-icon">{r.status === 'ready' ? <CheckCircle2 /> : busy ? <Loader2 className="spin" /> : <Download />}</div>
      <div className="res-main">
        <div className="row" style={{ gap: 8 }}>
          <Tip label={t.msg(r.description, '')}>
            <strong className="res-name">{t.msg(r.name, r.id)}</strong>
          </Tip>
          {r.status !== 'ready' && (
            <span className={clsx('chip', r.status === 'failed' ? 'critical' : busy ? 'accent' : '')}>{t(`res.status.${r.status}` as const)}</span>
          )}
        </div>
        {busy && (
          <div className="res-progress">
            <Progress value={r.progress} />
            <span className="muted tabular">
              {r.bytesDone != null && r.size > 0 ? `${bytes(r.bytesDone)} / ${bytes(r.size)}` : ''} {speed(r.speed)} {r.source ? `· ${t('res.source', { source: r.source })}` : ''}
            </span>
          </div>
        )}
        {r.status === 'failed' && r.error && <p className="res-error">{t.msg(r.error)}</p>}
      </div>
      <div className="res-actions">
        {r.size > 0 && !busy && <span className="muted tabular">{bytes(r.size)}</span>}
        {r.status === 'missing' || r.status === 'failed' ? (
          <>
            {r.kind === 'model' && (
              <Tip label={t('res.local_hint')}>
                <button className="btn btn-sm btn-ghost" onClick={() => importLocal(r)}>
                  <FolderOpen />
                  {t('res.local')}
                </button>
              </Tip>
            )}
            <button className="btn btn-sm btn-primary" onClick={() => attempt(() => api.post(`/resources/${r.id}/install`, {}))}>
              <Download />
              {r.status === 'failed' ? t('common.retry') : t('res.install')}
            </button>
          </>
        ) : r.status === 'ready' && !r.required ? (
          <button className="btn btn-sm btn-ghost" onClick={() => attempt(() => api.del(`/resources/${r.id}`))}>
            <Trash2 />
            {t('res.remove')}
          </button>
        ) : null}
      </div>
    </div>
  )
}

function StorageSection() {
  const t = useT()
  const { data: disk } = useDisk()
  const { data: characters } = useCharacters()
  const { data: models } = useModels()
  const { data: projects } = useProjects()
  if (!disk) return null
  const used = disk.libraryTotal - disk.libraryFree
  const sum = (xs: { bytes: number }[]) => xs.reduce((a, b) => a + b.bytes, 0)
  const groups = [
    { k: 'characters', label: t('res.storage.characters'), bytes: sum(disk.characters), color: 'var(--series-1)' },
    { k: 'models', label: t('res.storage.models'), bytes: sum(disk.models), color: 'var(--series-3)' },
    { k: 'projects', label: t('res.storage.projects'), bytes: sum(disk.projects), color: 'var(--series-2)' },
  ]
  const top = [
    ...disk.characters.map((x) => ({ id: x.id, name: characters?.find((c) => c.id === x.id)?.name ?? x.id, bytes: x.bytes, kind: t('res.storage.characters') })),
    ...disk.models.map((x) => ({ id: x.id, name: models?.find((m) => m.id === x.id)?.name ?? x.id, bytes: x.bytes, kind: t('res.storage.models') })),
    ...disk.projects.map((x) => ({ id: x.id, name: projects?.find((p) => p.id === x.id)?.name ?? x.id, bytes: x.bytes, kind: t('res.storage.projects') })),
  ]
    .sort((a, b) => b.bytes - a.bytes)
    .slice(0, 5)
  return (
    <section className="res-section">
      <div className="res-section-head">
        <HardDrive />
        <span className="section-title">{t('res.storage')}</span>
      </div>
      <div className="res-card glass">
        <div className="muted">{t('res.storage.free', { free: bytes(disk.libraryFree), total: bytes(disk.libraryTotal) })}</div>
        <div className="disk-bar">
          {groups.map((g) => (
            <span key={g.k} style={{ width: `${(g.bytes / Math.max(1, disk.libraryTotal)) * 100}%`, background: g.color }} title={`${g.label} ${bytes(g.bytes)}`} />
          ))}
          <span className="disk-other" style={{ width: `${(Math.max(0, used - sum(groups)) / Math.max(1, disk.libraryTotal)) * 100}%` }} />
        </div>
        <div className="disk-legend">
          {groups.map((g) => (
            <span key={g.k}>
              <i style={{ background: g.color }} />
              {g.label} <span className="tabular muted">{bytes(g.bytes)}</span>
            </span>
          ))}
        </div>
        {top.length > 0 && (
          <div className="disk-top">
            <span className="eyebrow">{t('res.storage.top')}</span>
            {top.map((x) => (
              <div key={x.id} className="row">
                <span>{x.name}</span>
                <span className="muted">· {x.kind}</span>
                <span className="spacer" />
                <span className="tabular muted">{bytes(x.bytes)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}

/** 底模: checkpoints new voices can start training from. Download, import a .pt, or reuse a version. */
function BaseModelsSection() {
  const t = useT()
  const qc = useQueryClient()
  const { data: bases } = useBaseModels()
  const { data: models } = useModels()
  const { data: characters } = useCharacters()
  const [fromRelease, setFromRelease] = useState(false)
  const refresh = () => qc.invalidateQueries({ queryKey: keys.baseModels })

  const importPt = async () => {
    const path = await pickLocalPath(['pt'])
    if (!path) return
    const name = path.split(/[\\/]/).pop()?.replace(/\.pt$/, '') ?? 'base'
    const job = await attempt(() => api.post<Job>('/base-models/import', { ckptPath: path, name }))
    if (job) {
      await attempt(() => waitForJob(job.id))
      refresh()
    }
  }

  const remove = async (b: BaseModel) => {
    const ok = await confirmDialog({ title: t('base.remove_title', { name: b.name }), confirm: t('common.delete'), cancel: t('common.cancel'), danger: true })
    if (ok && (await attempt(() => api.del(`/base-models/${b.id}`)))) refresh()
  }

  const releases = (models ?? []).flatMap((m) =>
    m.releases.map((r) => ({ m, r, label: `${m.characters.map((c) => characters?.find((x) => x.id === c.characterId)?.name ?? '?').join(' · ')} · ${r.name}` })),
  )

  return (
    <section className="res-section" id="base-models">
      <div className="res-section-head">
        <Layers />
        <span className="section-title">{t('base.title')}</span>
        <span className="spacer" />
        {releases.length > 0 && (
          <button className="btn btn-sm btn-ghost" onClick={() => setFromRelease(true)}>
            <Plus />
            {t('base.from_release')}
          </button>
        )}
        <button className="btn btn-sm" onClick={importPt}>
          <FolderOpen />
          {t('base.import')}
        </button>
      </div>
      <div className="res-list">
        {(bases ?? []).length === 0 && <div className="res-card glass muted">{t('base.empty')}</div>}
        {(bases ?? []).map((b) => (
          <div key={b.id} className={clsx('res-row glass', b.status)}>
            <div className="res-icon">{b.status === 'ready' ? <CheckCircle2 /> : b.status === 'downloading' ? <Loader2 className="spin" /> : <Download />}</div>
            <div className="res-main">
              <div className="row" style={{ gap: 8 }}>
                <strong>{b.name}</strong>
                <span className="chip">{t(`base.source.${b.source}` as const)}</span>
              </div>
              <span className="muted tabular res-desc">
                {b.nLayers}×{b.nChans} · {b.encoder} · {t('base.speakers', { n: b.nSpk })} · {t('models.step', { steps: steps(b.step) })}
              </span>
            </div>
            <div className="res-actions">
              {b.size > 0 && <span className="muted tabular">{bytes(b.size)}</span>}
              {b.source === 'download' && b.status === 'missing' && b.resourceId ? (
                <button className="btn btn-sm btn-primary" onClick={() => attempt(() => api.post(`/resources/${b.resourceId}/install`, {}))}>
                  <Download />
                  {t('res.install')}
                </button>
              ) : (
                <button className="btn btn-sm btn-ghost" onClick={() => remove(b)}>
                  <Trash2 />
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
      {fromRelease && (
        <FromReleaseDialog
          releases={releases}
          onClose={() => setFromRelease(false)}
          onDone={() => {
            setFromRelease(false)
            refresh()
          }}
        />
      )}
    </section>
  )
}

function FromReleaseDialog({ releases, onClose, onDone }: { releases: { m: { id: string }; r: { id: string; name: string }; label: string }[]; onClose: () => void; onDone: () => void }) {
  const t = useT()
  const [pick, setPick] = useState(0)
  const [name, setName] = useState(releases[0]?.label ?? '')
  const save = async () => {
    const x = releases[pick]
    if (x && (await attempt(() => api.post<BaseModel>('/base-models/from-release', { modelId: x.m.id, releaseId: x.r.id, name: name.trim() || x.label })))) onDone()
  }
  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title={t('base.from_release')}
      description={t('base.from_release_body')}
      actions={
        <>
          <button className="btn btn-ghost" onClick={onClose}>
            {t('common.cancel')}
          </button>
          <button className="btn btn-primary" onClick={save}>
            {t('common.save')}
          </button>
        </>
      }
    >
      <div className="col" style={{ gap: 12 }}>
        <select
          className="input"
          value={pick}
          onChange={(e) => {
            const i = Number(e.target.value)
            setPick(i)
            setName(releases[i]?.label ?? '')
          }}
        >
          {releases.map((x, i) => (
            <option key={x.r.id} value={i}>
              {x.label}
            </option>
          ))}
        </select>
        <input className="input" value={name} onChange={(e) => setName(e.target.value)} aria-label={t('plg.name')} />
      </div>
    </Dialog>
  )
}

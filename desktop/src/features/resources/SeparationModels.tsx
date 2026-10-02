import clsx from 'clsx'
import { ArrowUpCircle, Check, Download, FolderInput, Loader2, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { api } from '../../api/client'
import { useCleanupModels, useResources, useSeparationModels } from '../../api/hooks'
import type { CleanupModel, Resource, SeparationImportResult, SeparationModel } from '../../api/types'
import { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { bytes, percent } from '../../lib/format'
import { pickFolder } from '../../shell/bridge'
import { toast } from '../../state/ui'
import './separation.css'

/** What a separation or cleanup model shares for the list. */
type ListedModel = Pick<SeparationModel, 'id' | 'name' | 'source' | 'resourceId' | 'sdr' | 'ready' | 'missing' | 'size'>

/** What to install so `m` can run: the add-on (fresh or updated) or its checkpoint. */
function installTarget(m: ListedModel): string | null {
  if (m.missing === 'addon' || m.missing === 'addon_update') return 'separator'
  if (m.missing === 'model') return m.resourceId
  return null
}

function installProgress(resources: Resource[] | undefined, id: string | null): number | null {
  const r = id ? resources?.find((x) => x.id === id) : undefined
  return r && (r.status === 'downloading' || r.status === 'installing') ? (r.progress ?? 0) : null
}

/** The label of the button that gets what `m` needs. */
function useGetLabel() {
  const t = useT()
  return (m: ListedModel) =>
    m.missing === 'addon_update' ? t('sep.update') : m.missing === 'addon' ? t('sep.install_addon') : m.size ? t('sep.download_size', { size: bytes(m.size) }) : t('sep.download')
}

/** Download / install / update what `m` needs, with progress, in its row. */
function GetButton({ m, compact }: { m: ListedModel; compact?: boolean }) {
  const t = useT()
  const label = useGetLabel()(m)
  const { data: resources } = useResources()
  const target = installTarget(m)
  if (!target) return null
  const progress = installProgress(resources, target)
  return (
    <button
      className={clsx('btn btn-sm sep-get', compact && 'icon-only')}
      disabled={progress != null}
      onClick={() => void attempt(() => api.post(`/resources/${target}/install`, {}))}
      title={m.missing === 'addon_update' ? t('sep.update_hint') : compact ? label : undefined}
      aria-label={label}
    >
      {progress != null ? <Loader2 className="spin" /> : m.missing === 'addon_update' ? <ArrowUpCircle /> : <Download />}
      {progress != null ? percent(progress) : compact ? '' : label}
    </button>
  )
}

/** A radio list of models: 'automatic' first, then each model with what it needs to run. */
function ModelList<M extends ListedModel>({
  models,
  defaultModel,
  value,
  onChange,
  compact,
  label,
  deletePath,
  badge,
}: {
  models: M[]
  defaultModel: string | null
  value: string
  onChange: (id: string) => void
  compact?: boolean
  label: string
  /** where imported models are deleted (`${deletePath}/${id}`) */
  deletePath: string
  badge?: (m: M) => React.ReactNode
}) {
  const t = useT()
  const auto = models.find((m) => m.id === defaultModel)
  return (
    <div className={clsx('sep-models', compact && 'compact')} role="radiogroup" aria-label={label}>
      <button role="radio" aria-checked={value === 'auto'} className={clsx('sep-model', value === 'auto' && 'on')} onClick={() => onChange('auto')}>
        <span className="sep-radio">{value === 'auto' && <Check />}</span>
        <span className="sep-name">{t('sep.auto')}</span>
        <span className="sep-meta">{auto ? t.msg(auto.name) : defaultModel === null ? t('cln.none_ready') : ''}</span>
      </button>
      {models.map((m) => (
        <div key={m.id} className={clsx('sep-model-row', value === m.id && 'on', !m.ready && 'unready')}>
          <button role="radio" aria-checked={value === m.id} className={clsx('sep-model', value === m.id && 'on')} disabled={!m.ready} onClick={() => onChange(m.id)}>
            <span className="sep-radio">{value === m.id && <Check />}</span>
            <span className="sep-name">{t.msg(m.name, m.id)}</span>
            {!compact && badge?.(m)}
            {!compact && m.sdr != null && <span className="sep-meta tabular">SDR {m.sdr.toFixed(1)}</span>}
          </button>
          {!m.ready && <GetButton m={m} compact={compact} />}
          {m.source === 'local' && (
            <button className="icon-btn sm sep-del" aria-label={t('common.delete')} title={t('common.delete')} onClick={() => void attempt(() => api.del(`${deletePath}/${m.id}`))}>
              <Trash2 />
            </button>
          )}
        </div>
      ))}
    </div>
  )
}

/**
 * Pick a vocal separation model. `auto` follows the best installed one. Models that can't run yet
 * offer what they need (download the checkpoint, install or update the add-on) right in the row.
 */
export function SeparationModelPicker({ value, onChange, compact }: { value: string; onChange: (id: string) => void; compact?: boolean }) {
  const t = useT()
  const { data } = useSeparationModels()
  if (!data) return null
  return (
    <ModelList<SeparationModel>
      models={data.models}
      defaultModel={data.defaultModel}
      value={value}
      onChange={onChange}
      compact={compact}
      label={t('sep.model')}
      deletePath="/resources/separation"
      badge={(m) => m.quality === 'high' && <span className="chip good sep-badge">{t('sep.better')}</span>}
    />
  )
}

/** Pick a vocal cleanup (dereverb) model; `auto` follows the best installed one. */
export function CleanupModelPicker({ value, onChange, compact }: { value: string; onChange: (id: string) => void; compact?: boolean }) {
  const t = useT()
  const { data } = useCleanupModels()
  if (!data) return null
  return (
    <ModelList<CleanupModel>
      models={data.models}
      defaultModel={data.defaultModel}
      value={value}
      onChange={onChange}
      compact={compact}
      label={t('cln.model')}
      deletePath="/resources/cleanup"
      badge={(m) => <span className="chip sep-badge">{t(`cln.kind.${m.kind}` as const)}</span>}
    />
  )
}

/**
 * While no dereverb model is ready: one button that gets what the recommended one needs (the
 * add-on first, then the checkpoint). Renders nothing once a model is ready.
 */
export function CleanupGetButton({ compact }: { compact?: boolean }) {
  const { data } = useCleanupModels()
  const first = data && data.defaultModel === null ? data.models.find((m) => m.source === 'resource') : undefined
  return first ? <GetButton m={first} compact={compact} /> : null
}

/** Bring in models from a Music-Source-Separation-Training (or MSST-GUI) folder, or one checkpoint folder. */
export function ImportSeparationButton() {
  const t = useT()
  const [busy, setBusy] = useState(false)
  const run = async () => {
    const path = await pickFolder()
    if (!path) return
    setBusy(true)
    const res = await attempt(() => api.post<SeparationImportResult>('/resources/separation/import', { path }))
    setBusy(false)
    if (res) reportSeparationImport(res, t)
  }
  return (
    <button className="btn btn-sm" onClick={() => void run()} disabled={busy}>
      {busy ? <Loader2 className="spin" /> : <FolderInput />}
      {t('sep.import')}
    </button>
  )
}

/** Toast what an import found and why the rest was skipped. */
export function reportSeparationImport(res: SeparationImportResult, t: ReturnType<typeof useT>) {
  const skipped = res.skipped.slice(0, 3).map((s) => `${s.path.split(/[\\/]/).pop()}：${t.msg(s.reason)}`)
  if (res.found.length === 0) {
    toast({ kind: 'info', title: t('sep.import_none'), body: skipped.join('\n') || t('sep.import_none_body') })
    return
  }
  const cleanup = res.found.filter((f) => f.role === 'cleanup').length
  const separation = res.found.length - cleanup
  toast({
    kind: 'success',
    title: [separation && t('sep.import_found', { n: separation }), cleanup && t('cln.import_found', { n: cleanup })].filter(Boolean).join(t('cln.and')),
    body: [res.found.map((f) => f.name).join('、'), ...skipped].filter(Boolean).join('\n') || undefined,
  })
}

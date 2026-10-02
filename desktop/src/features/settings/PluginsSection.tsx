import { useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { AlertTriangle, CheckCircle2, ChevronDown, Cpu, FlaskConical, FolderOpen, Loader2, Plug, Plus, Trash2, Wrench } from 'lucide-react'
import { useState } from 'react'
import { api, ApiError, mediaUrl } from '../../api/client'
import { usePlugins } from '../../api/hooks'
import { keys } from '../../api/keys'
import type { Job, Plugin } from '../../api/types'
import { ParamControls } from '../../components/ParamControls'
import { confirmDialog, Dialog, Segmented, Switch } from '../../components/ui'
import { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { waitForJob } from '../../lib/jobs'
import { isTauri, pickExecutable, pickFolder } from '../../shell/bridge'
import { toast } from '../../state/ui'
import { Group } from './parts'

/** User plugins: folders with a script (e.g. a friend's mixing chain) that the canvas can run. */
export function PluginsSection() {
  const t = useT()
  const qc = useQueryClient()
  const { data: plugins } = usePlugins()
  const [wizard, setWizard] = useState<{ folder: string; candidates: string[] } | null>(null)
  const [manualPath, setManualPath] = useState<string | null>(null)
  const refresh = () => {
    qc.invalidateQueries({ queryKey: keys.plugins })
    qc.invalidateQueries({ queryKey: keys.mixProcessors })
  }

  const add = async (folder: string) => {
    const ok = await confirmDialog({ title: t('plg.trust_title'), body: t('plg.trust_body'), confirm: t('plg.add'), cancel: t('common.cancel') })
    if (!ok) return
    try {
      await api.post<Plugin>('/plugins', { folder })
      refresh()
    } catch (e) {
      if (e instanceof ApiError && e.code === 'plugin.no_manifest') {
        let candidates: string[] = []
        try {
          const detail = JSON.parse(e.detail || '{}') as { candidates?: string[] }
          candidates = detail.candidates ?? []
        } catch {
          /* detail isn't JSON: the wizard lets the user type the entry */
        }
        setWizard({ folder, candidates })
      } else await attempt(() => Promise.reject(e))
    }
  }

  const pick = async () => {
    if (isTauri) {
      const folder = await pickFolder()
      if (folder) await add(folder)
    } else setManualPath('')
  }

  return (
    <Group
      icon={<Plug />}
      title={t('plg.title')}
      aside={
        <button className="btn btn-sm" onClick={pick}>
          <Plus />
          {t('plg.add')}
        </button>
      }
    >
      {(plugins ?? []).length === 0 ? (
        <div className="plg-empty muted">{t('plg.empty')}</div>
      ) : (
        (plugins ?? []).map((p) => <PluginRow key={p.id} plugin={p} onChange={refresh} />)
      )}

      {manualPath !== null && (
        <Dialog
          open
          onOpenChange={(o) => !o && setManualPath(null)}
          title={t('plg.add')}
          actions={
            <>
              <button className="btn btn-ghost" onClick={() => setManualPath(null)}>
                {t('common.cancel')}
              </button>
              <button
                className="btn btn-primary"
                disabled={!manualPath.trim()}
                onClick={() => {
                  const folder = manualPath.trim()
                  setManualPath(null)
                  void add(folder)
                }}
              >
                {t('common.continue')}
              </button>
            </>
          }
        >
          <input className="input mono" autoFocus placeholder="/path/to/plugin-folder" value={manualPath} onChange={(e) => setManualPath(e.target.value)} />
        </Dialog>
      )}
      {wizard && (
        <WizardDialog
          folder={wizard.folder}
          candidates={wizard.candidates}
          onClose={() => setWizard(null)}
          onDone={() => {
            setWizard(null)
            refresh()
          }}
        />
      )}
    </Group>
  )
}

function PluginRow({ plugin: p, onChange }: { plugin: Plugin; onChange: () => void }) {
  const t = useT()
  const [open, setOpen] = useState(p.status !== 'ready')
  const [testing, setTesting] = useState(false)
  const [testUrl, setTestUrl] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const patch = async (body: Partial<Pick<Plugin, 'enabled' | 'interpreter' | 'pythonPath' | 'settingValues'>>) => {
    setBusy(true)
    await attempt(() => api.patch<Plugin>(`/plugins/${p.id}`, body))
    setBusy(false)
    onChange()
  }

  const remove = async () => {
    const ok = await confirmDialog({ title: t('plg.remove_title', { name: p.name }), body: t('plg.remove_body'), confirm: t('common.delete'), cancel: t('common.cancel'), danger: true })
    if (!ok) return
    await attempt(() => api.del(`/plugins/${p.id}`))
    onChange()
  }

  const install = async () => {
    if (p.interpreter === 'custom') {
      const ok = await confirmDialog({ title: t('plg.install_custom_title'), body: t('plg.install_custom_body', { path: p.pythonPath ?? '' }), confirm: t('plg.install_custom_go'), cancel: t('common.cancel'), danger: true })
      if (!ok) return
    }
    const job = await attempt(() => api.post<Job>(`/plugins/${p.id}/install-requirements`, { target: p.interpreter }))
    if (!job) return
    setBusy(true)
    await attempt(() => waitForJob(job.id))
    setBusy(false)
    onChange()
  }

  const test = async () => {
    setTesting(true)
    setTestUrl(null)
    const job = await attempt(() => api.post<Job>(`/plugins/${p.id}/test`))
    if (job) {
      const done = await attempt(() => waitForJob(job.id))
      if (done?.status === 'succeeded') setTestUrl((done.result as { mediaUrl?: string } | null)?.mediaUrl ?? null)
      else if (done) toast({ kind: 'error', title: t('plg.test_failed'), detail: done.error?.detail ?? done.error?.message })
    }
    setTesting(false)
  }

  const choosePython = async () => {
    const path = isTauri ? await pickExecutable() : window.prompt(t('plg.python_path'), p.pythonPath ?? '')
    if (path) await patch({ interpreter: 'custom', pythonPath: path })
  }

  const check = p.check
  return (
    <div className={clsx('plg', open && 'open')}>
      <div className="plg-head">
        <button className="plg-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
          <ChevronDown className="plg-chevron" />
          <strong>{p.name}</strong>
          <span className="chip">{t(p.kind === 'mix' ? 'plg.kind.mix' : 'plg.kind.effect')}</span>
          {p.status === 'ready' ? (
            <span className="chip good">
              <CheckCircle2 />
              {t('plg.ready')}
            </span>
          ) : (
            <span className={clsx('chip', p.status === 'error' ? 'critical' : 'warning')}>
              <AlertTriangle />
              {t(p.status === 'error' ? 'plg.error' : 'plg.needs_setup')}
            </span>
          )}
        </button>
        <span className="spacer" />
        <Switch checked={p.enabled} onChange={(enabled) => patch({ enabled })} label={t('plg.enabled')} />
      </div>
      {open && (
        <div className="plg-body">
          <div className="mono muted plg-folder">{p.folder}</div>

          <div className="plg-line">
            <span className="field-label">{t('plg.python')}</span>
            <Segmented<'builtin' | 'custom'>
              size="sm"
              value={p.interpreter}
              onChange={(interpreter) => (interpreter === 'custom' ? void choosePython() : void patch({ interpreter: 'builtin', pythonPath: null }))}
              options={[
                { value: 'builtin', label: t('plg.python.builtin') },
                { value: 'custom', label: t('plg.python.custom') },
              ]}
            />
            {p.interpreter === 'custom' && (
              <button className="btn btn-sm btn-ghost mono plg-pypath" onClick={choosePython} title={p.pythonPath ?? ''}>
                <Cpu />
                {p.pythonPath}
              </button>
            )}
          </div>

          {check && (
            <div className="plg-check">
              {check.version && <span className="muted">Python {check.version}</span>}
              {check.missing.length > 0 ? (
                <>
                  <span className="warn-text">{t('rt.packages_missing', { n: check.missing.length })}</span>
                  <span className="mono muted">{check.missing.join(' · ')}</span>
                  <button className="btn btn-sm" onClick={install} disabled={busy}>
                    {busy ? <Loader2 className="spin" /> : <Wrench />}
                    {t('plg.install')}
                  </button>
                </>
              ) : (
                check.ok && <span className="good-text">{t('rt.packages_ok')}</span>
              )}
              {check.error && <span className="critical-text">{t.msg(check.error)}</span>}
            </div>
          )}

          {p.settings.length > 0 && (
            <ParamControls
              specs={p.settings}
              values={p.settingValues}
              onChange={(id, v) => void patch({ settingValues: { ...p.settingValues, [id]: v } })}
            />
          )}

          <div className="row" style={{ gap: 8 }}>
            <button className="btn btn-sm" onClick={test} disabled={testing || !p.enabled}>
              {testing ? <Loader2 className="spin" /> : <FlaskConical />}
              {t('plg.test')}
            </button>
            {testUrl && <audio className="plg-audio" controls src={mediaUrl(testUrl)} />}
            <span className="spacer" />
            <button className="btn btn-sm btn-ghost" onClick={remove}>
              <Trash2 />
              {t('common.delete')}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

/** A folder without a manifest: describe how to call the script once, the app remembers it. */
function WizardDialog({ folder, candidates, onClose, onDone }: { folder: string; candidates: string[]; onClose: () => void; onDone: () => void }) {
  const t = useT()
  const base = folder.split(/[\\/]/).filter(Boolean).pop() ?? 'plugin'
  const [name, setName] = useState(base)
  const [entry, setEntry] = useState(candidates[0] ?? '')
  const [kind, setKind] = useState<'mix' | 'effect'>('mix')
  const [order, setOrder] = useState<'backing_vocal' | 'vocal_backing' | 'custom'>('backing_vocal')
  const [custom, setCustom] = useState('{backing} {vocal}')
  const [outputs, setOutputs] = useState('output/*.flac, output/*.wav')
  const [cwd, setCwd] = useState<'temp' | 'folder'>('temp')
  const [advanced, setAdvanced] = useState(false)

  const args = kind === 'effect' ? ['{input}', '{output}'] : order === 'backing_vocal' ? ['{backing}', '{vocal}'] : order === 'vocal_backing' ? ['{vocal}', '{backing}'] : custom.split(/\s+/).filter(Boolean)

  const save = async () => {
    const plugin = await attempt(() =>
      api.post<Plugin>('/plugins/wizard', {
        folder,
        name: name.trim() || base,
        kind,
        entry,
        args,
        outputs: outputs.split(',').map((x) => x.trim()).filter(Boolean),
        cwd,
      }),
    )
    if (plugin) onDone()
  }

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title={t('plg.wizard_title')}
      description={t('plg.wizard_body')}
      wide
      actions={
        <>
          <button className="btn btn-ghost" onClick={onClose}>
            {t('common.cancel')}
          </button>
          <button className="btn btn-primary" onClick={save} disabled={!entry}>
            {t('plg.add')}
          </button>
        </>
      }
    >
      <div className="plg-wizard">
        <label className="field">
          <span className="field-label">{t('plg.name')}</span>
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="field">
          <span className="field-label">{t('plg.entry')}</span>
          {candidates.length > 0 ? (
            <select className="input mono" value={entry} onChange={(e) => setEntry(e.target.value)}>
              {candidates.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          ) : (
            <input className="input mono" value={entry} placeholder="main.py" onChange={(e) => setEntry(e.target.value)} />
          )}
        </label>
        <div className="field">
          <span className="field-label">{t('plg.kind')}</span>
          <Segmented<'mix' | 'effect'>
            size="sm"
            value={kind}
            onChange={setKind}
            options={[
              { value: 'mix', label: t('plg.kind.mix') },
              { value: 'effect', label: t('plg.kind.effect') },
            ]}
          />
        </div>
        {kind === 'mix' && (
          <div className="field">
            <span className="field-label">{t('plg.args')}</span>
            <Segmented<'backing_vocal' | 'vocal_backing' | 'custom'>
              size="sm"
              value={order}
              onChange={setOrder}
              options={[
                { value: 'backing_vocal', label: t('plg.args.backing_vocal') },
                { value: 'vocal_backing', label: t('plg.args.vocal_backing') },
                { value: 'custom', label: t('plg.args.custom') },
              ]}
            />
            {order === 'custom' && <input className="input mono" value={custom} onChange={(e) => setCustom(e.target.value)} />}
          </div>
        )}
        <div className="mono muted plg-preview">
          python {entry || 'main.py'} {args.join(' ')}
        </div>
        <button className="org-adv-toggle" onClick={() => setAdvanced(!advanced)}>
          <ChevronDown style={{ transform: advanced ? 'rotate(180deg)' : undefined }} />
          {t('common.advanced')}
        </button>
        {advanced && (
          <>
            <label className="field">
              <span className="field-label">{t('plg.outputs')}</span>
              <input className="input mono" value={outputs} onChange={(e) => setOutputs(e.target.value)} />
            </label>
            <div className="field">
              <span className="field-label">{t('plg.cwd')}</span>
              <Segmented<'temp' | 'folder'>
                size="sm"
                value={cwd}
                onChange={setCwd}
                options={[
                  { value: 'temp', label: t('plg.cwd.temp') },
                  { value: 'folder', label: t('plg.cwd.folder') },
                ]}
              />
            </div>
          </>
        )}
        <div className="row muted" style={{ gap: 6, fontSize: 'var(--fs-xs)' }}>
          <FolderOpen style={{ width: 13, height: 13 }} />
          <span className="mono">{folder}</span>
        </div>
      </div>
    </Dialog>
  )
}

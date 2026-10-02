import { useQueryClient } from '@tanstack/react-query'
import { FolderOpen, Info, Keyboard, LifeBuoy, Loader2, Palette, RotateCcw, Settings2, Shield, Zap } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { api, mediaUrl } from '../../api/client'
import { usePreferences, useSystem } from '../../api/hooks'
import type { Job, Preferences } from '../../api/types'
import { useBoot } from '../../app/boot'
import { FreeNotice } from '../../components/FreeNotice'
import { AppTile } from '../../components/Logo'
import { confirmDialog, Segmented, Switch, Tip } from '../../components/ui'
import { useLang, useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { waitForJob } from '../../lib/jobs'
import { isTauri, pickFolder, revealPath, shell } from '../../shell/bridge'
import { useTheme, type ThemePref } from '../../state/theme'
import { toast, useUi } from '../../state/ui'
import { setPreference } from '../resources/ResourcesView'
import { TermsDialog } from '../terms/Terms'
import { GpuMemoryRows } from './GpuMemory'
import { Group, Row } from './parts'
import { PluginsSection } from './PluginsSection'
import { RuntimeSection } from './RuntimeSection'
import { VstSection } from './VstSection'
import './settings.css'

export default function SettingsView() {
  const t = useT()
  const qc = useQueryClient()
  const { data: prefs } = usePreferences()
  const { data: system } = useSystem()
  const shellState = useBoot((s) => s.shell)
  const setShell = useBoot((s) => s.setShell)
  const setLang = useLang((s) => s.setLang)
  const setShortcuts = useUi((s) => s.setShortcuts)
  const [closeToTray, setCloseToTray] = useState(true)
  const [termsOpen, setTermsOpen] = useState(false)
  const theme = useTheme((s) => s.pref)
  const setTheme = useTheme((s) => s.setPref)
  const { hash } = useLocation()
  const ready = !!prefs

  useEffect(() => {
    if (isTauri) void shell.setCloseToTray(closeToTray).catch(() => {})
  }, [closeToTray])

  // deep links like /settings#vst (from the effects chain editor)
  useEffect(() => {
    if (!hash || !ready) return
    const timer = window.setTimeout(() => document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120)
    return () => window.clearTimeout(timer)
  }, [hash, ready])

  if (!prefs) return null
  const set = (patch: Partial<Preferences>) => setPreference(qc, patch)

  const changeLibrary = async () => {
    const path = await pickFolder()
    if (!path) return
    const ok = await confirmDialog({ title: t('set.library'), body: t('set.library_change_body'), confirm: t('common.confirm'), cancel: t('common.cancel') })
    if (!ok) return
    setShell(await shell.setLibraryPath(path))
    qc.invalidateQueries()
  }

  return (
    <div className="page">
      <div className="page-inner settings">
        <div className="page-header">
          <h1 className="page-title">{t('set.title')}</h1>
        </div>

        <Group icon={<Settings2 />} title={t('set.general')}>
          <Row label={t('set.language')}>
            <Segmented<'zh-CN' | 'en'>
              size="sm"
              value={prefs.language}
              onChange={(language) => {
                setLang(language)
                void set({ language })
                if (isTauri) void shell.setLanguage(language)
              }}
              options={[
                { value: 'zh-CN', label: '简体中文' },
                { value: 'en', label: 'English' },
              ]}
            />
          </Row>
          <Row label={t('set.notify')}>
            <Switch checked={prefs.notifyOnFinish} onChange={(v) => set({ notifyOnFinish: v })} label={t('set.notify')} />
          </Row>
          {isTauri && (
            <Row label={t('set.tray')}>
              <Switch checked={closeToTray} onChange={setCloseToTray} label={t('set.tray')} />
            </Row>
          )}
          <Row label={t('set.shortcuts')}>
            <button className="btn btn-sm" onClick={() => setShortcuts(true)}>
              <Keyboard />
              {t('set.shortcuts')}
            </button>
          </Row>
        </Group>

        <Group icon={<Palette />} title={t('set.appearance')}>
          <Row label={t('set.theme')}>
            <Segmented<ThemePref>
              size="sm"
              value={theme}
              onChange={setTheme}
              options={[
                { value: 'system', label: t('set.theme.system') },
                { value: 'light', label: t('set.theme.light') },
                { value: 'dark', label: t('set.theme.dark') },
              ]}
            />
          </Row>
          <Row label={t('set.glass')}>
            <Switch checked={prefs.glass} onChange={(glass) => set({ glass })} label={t('set.glass')} />
          </Row>
          <Row label={t('set.low_power')}>
            <Switch checked={prefs.lowPowerWhileTraining} onChange={(v) => set({ lowPowerWhileTraining: v })} label={t('set.low_power')} />
          </Row>
        </Group>

        <Group icon={<Zap />} title={t('set.training')}>
          <Row label={t('set.gpu_policy')} hint={t(`set.gpu_policy_hint.${prefs.gpuPolicy}` as const)}>
            <Segmented<Preferences['gpuPolicy']>
              size="sm"
              value={prefs.gpuPolicy}
              onChange={(gpuPolicy) => set({ gpuPolicy })}
              options={[
                { value: 'smart', label: t('set.gpu_policy.smart') },
                { value: 'cpu', label: t('set.gpu_policy.cpu') },
                { value: 'queue', label: t('set.gpu_policy.queue') },
              ]}
            />
          </Row>
          <GpuMemoryRows />
          <Row label={t('set.auto_slim')}>
            <Switch checked={prefs.autoSlimAfterTraining} onChange={(v) => set({ autoSlimAfterTraining: v })} label={t('set.auto_slim')} />
          </Row>
        </Group>

        <PluginsSection />

        <VstSection />

        <RuntimeSection />

        <Group icon={<FolderOpen />} title={t('set.storage')}>
          <Row label={t('set.library')}>
            <div className="row" style={{ gap: 8, minWidth: 0 }}>
              <span className="set-path mono" title={system?.libraryPath}>
                {system?.libraryPath ?? shellState?.libraryPath}
              </span>
              {isTauri && (
                <>
                  <button className="btn btn-sm" onClick={() => system && revealPath(system.libraryPath)}>
                    <FolderOpen />
                  </button>
                  <button className="btn btn-sm" onClick={changeLibrary}>
                    {t('set.library_change')}
                  </button>
                </>
              )}
            </div>
          </Row>
          <ExportFolderRow prefs={prefs} onChange={(exportFolder) => set({ exportFolder })} />
          {isTauri && (
            <Row label={t('set.open_logs')}>
              <div className="row" style={{ gap: 8 }}>
                <button className="btn btn-sm" onClick={() => system && revealPath(system.dataPath)}>
                  {t('set.open_data')}
                </button>
                <button className="btn btn-sm" onClick={() => shell.openLogsFolder()}>
                  {t('set.open_logs')}
                </button>
              </div>
            </Row>
          )}
        </Group>

        <Group
          icon={<Shield />}
          title={t('set.disclaimer')}
          aside={
            <button className="btn btn-sm btn-ghost" onClick={() => setTermsOpen(true)}>
              {t('terms.view')}
            </button>
          }
        >
          <p className="secondary set-text">{t('set.disclaimer_body')}</p>
        </Group>
        <TermsDialog open={termsOpen} onOpenChange={setTermsOpen} />

        <Group icon={<Info />} title={t('set.about')}>
          <div className="about">
            <AppTile size={52} />
            <div className="col" style={{ gap: 2 }}>
              <strong style={{ fontSize: 'var(--fs-lg)' }}>
                {t('app.name')} · Stellaria
              </strong>
              <span className="muted">{t('set.version', { version: shellState?.appVersion ?? system?.version ?? '' })}</span>
              <span className="muted">{t('set.engine', { version: system?.engineVersion ?? '' })}</span>
            </div>
          </div>
          <FreeNotice />
          <DiagnosticsRow />
          <div className="eyebrow" style={{ marginTop: 16 }}>
            {t('set.credits')}
          </div>
          <p className="secondary set-text">{t('set.license')}</p>
          <ul className="credits">
            <li>{t('set.credit.ddsp')}</li>
            <li>{t('set.credit.vocoder')}</li>
            <li>{t('set.credit.contentvec')}</li>
            <li>{t('set.credit.rmvpe')}</li>
            <li>{t('set.credit.demucs')}</li>
            <li>{t('set.credit.roformer')}</li>
            <li>{t('set.credit.dereverb')}</li>
            <li>{t('set.credit.pedalboard')}</li>
            <li>{t('set.credit.stretch')}</li>
            <li>{t('set.credit.lameenc')}</li>
            <li>{t('set.credit.wenkai')}</li>
          </ul>
        </Group>
      </div>
    </div>
  )
}


/** Where export nodes write when they don't name a folder. */
function ExportFolderRow({ prefs, onChange }: { prefs: Preferences; onChange: (folder: string | null) => void }) {
  const t = useT()
  const folder = prefs.exportFolder ?? null
  const choose = async () => {
    const picked = isTauri ? await pickFolder() : window.prompt(t('set.export_folder'), folder ?? '')
    if (picked?.trim()) onChange(picked.trim())
  }
  return (
    <Row label={t('set.export_folder')}>
      <div className="row" style={{ gap: 8, minWidth: 0 }}>
        <span className={folder ? 'set-path mono' : 'set-path is-unset'} title={folder ?? undefined}>
          {folder ?? t('set.export_folder_unset')}
        </span>
        {folder && isTauri && (
          <button className="btn btn-sm" onClick={() => void revealPath(folder)} aria-label={t('common.open_folder')}>
            <FolderOpen />
          </button>
        )}
        <button className="btn btn-sm" onClick={choose}>
          {t('set.export_folder_choose')}
        </button>
        {folder && (
          <Tip label={t('common.reset')}>
            <button className="btn btn-sm btn-ghost" onClick={() => onChange(null)} aria-label={t('common.reset')}>
              <RotateCcw />
            </button>
          </Tip>
        )}
      </div>
    </Row>
  )
}

/** Settings → About → 导出诊断信息: logs + system report as a zip (no audio). */
function DiagnosticsRow() {
  const t = useT()
  const [running, setRunning] = useState(false)
  const [path, setPath] = useState<string | null>(null)

  const run = async () => {
    setRunning(true)
    const job = await attempt(() => api.post<Job>('/system/diagnostics'))
    if (!job) {
      setRunning(false)
      return
    }
    const done = await attempt(() => waitForJob(job.id))
    setRunning(false)
    if (!done) return
    const result = (done.result ?? {}) as { path?: string; downloadUrl?: string }
    if (done.status === 'succeeded') {
      if (result.downloadUrl && !isTauri) window.open(mediaUrl(result.downloadUrl))
      if (result.path) {
        setPath(result.path)
        const where = result.path
        toast({ kind: 'success', title: t('set.diagnostics_done'), body: where, action: isTauri ? { label: t('exp.reveal'), run: () => void revealPath(where) } : undefined })
      }
    } else if (done.status === 'failed') {
      toast({ kind: 'error', title: t.msg({ code: done.error?.code ?? 'common.error' }, t('common.error')), detail: done.error?.detail || done.error?.message })
    }
  }

  return (
    <div className="set-about-row">
      <Row
        label={
          <Tip label={t('set.diagnostics_hint')}>
            <span className="has-hint">{t('set.diagnostics')}</span>
          </Tip>
        }
        hint={path ? <span className="mono">{path}</span> : undefined}
      >
        <div className="row" style={{ gap: 8 }}>
          {path && isTauri && (
            <button className="btn btn-sm" onClick={() => void revealPath(path)}>
              <FolderOpen />
              {t('exp.reveal')}
            </button>
          )}
          <button className="btn btn-sm" onClick={run} disabled={running}>
            {running ? <Loader2 className="spin" /> : <LifeBuoy />}
            {running ? t('set.diagnostics_running') : t('set.diagnostics_export')}
          </button>
        </div>
      </Row>
    </div>
  )
}

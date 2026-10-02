import { useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { AlertTriangle, ArrowLeft, Check, CheckCircle2, ChevronRight, Cpu, Download, FolderInput, FolderOpen, HardDrive, Loader2, RotateCcw } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, useConnection } from '../../api/client'
import { useResources } from '../../api/hooks'
import { startLive } from '../../api/live'
import type { Resource } from '../../api/types'
import { MODELS_LATER_KEY, useBoot } from '../../app/boot'
import { Backdrop } from '../../components/Backdrop'
import { FreeLine } from '../../components/FreeNotice'
import { AppTile } from '../../components/Logo'
import { ConfirmHost, Progress, Ring, Segmented, Switch, TooltipProvider } from '../../components/ui'
import { useLang, useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { bytes, speed } from '../../lib/format'
import { isTauri, listen, pickFolder, shell, type BootstrapProgress, type HardwareReport, type Region, type TorchVariant } from '../../shell/bridge'
import { Toasts } from '../../components/Toasts'
import { WindowChrome } from '../../components/WindowChrome'
import { useSetupCloseRequests } from '../../app/useCloseRequests'
import { importModelsFromFolder } from '../resources/importFolder'
import { ExistingPython, useFoundPythons } from '../settings/ExistingPython'
import { TermsDialog } from '../terms/Terms'
import { acceptTerms } from '../terms/termsOfUse'
import './onboarding.css'

/** The DDSP-SVC (or bundle) folder chosen for "use what I have": the models step looks there too. */
const FOLDER_KEY = 'stellaria.ob.folder'

function stored(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function store(key: string, value: string | null) {
  try {
    if (value === null) localStorage.removeItem(key)
    else localStorage.setItem(key, value)
  } catch {
    /* storage unavailable */
  }
}

type Step = 'welcome' | 'library' | 'hardware' | 'install' | 'models' | 'done'
const STAGES = ['prepare', 'python', 'venv', 'torch', 'packages', 'verify'] as const

export function Onboarding() {
  const boot = useBoot()
  const runtimeReady = boot.shell?.runtime.status === 'ready'
  // set up before: straight to what is missing (choosing the runtime again starts at the hardware)
  const firstStep: Step = !isTauri
    ? 'models'
    : boot.shell?.firstRunDone
      ? runtimeReady && !boot.runtimeSetup
        ? 'models'
        : 'hardware'
      : 'welcome'
  const [step, setStep] = useState<Step>(firstStep)
  const [variant, setVariant] = useState<TorchVariant>('cpu')
  const [region, setRegion] = useState<Region>('auto')
  const steps: Step[] = isTauri ? ['library', 'hardware', 'install', 'models'] : ['models']
  useSetupCloseRequests()

  return (
    <TooltipProvider>
      <Backdrop />
      <WindowChrome />
      <div className="ob">
        {step !== 'welcome' && step !== 'done' && (
          <ol className="ob-steps">
            {steps.map((s, i) => (
              <li key={s} className={clsx(s === step && 'on', steps.indexOf(step) > i && 'done')}>
                <span className="ob-step-dot">{steps.indexOf(step) > i ? <Check /> : i + 1}</span>
                <StepLabel step={s} />
              </li>
            ))}
          </ol>
        )}
        <div className="ob-card glass">
          {step === 'welcome' && <Welcome onNext={() => setStep('library')} />}
          {step === 'library' && <Library onNext={() => setStep('hardware')} />}
          {step === 'hardware' && (
            <Hardware
              region={region}
              setRegion={setRegion}
              onNext={(v) => {
                setVariant(v)
                setStep(runtimeReady ? 'models' : 'install')
              }}
            />
          )}
          {step === 'install' && <Install variant={variant} region={region} setRegion={setRegion} onDone={() => setStep('models')} />}
          {step === 'models' && <Models onDone={() => setStep('done')} />}
          {step === 'done' && <Done />}
        </div>
      </div>
      <Toasts />
      <ConfirmHost />
    </TooltipProvider>
  )
}

function StepLabel({ step }: { step: Step }) {
  const t = useT()
  const key = step === 'library' ? 'ob.step.library' : step === 'hardware' ? 'ob.step.hardware' : step === 'install' ? 'ob.step.install' : 'ob.step.models'
  return <span>{t(key)}</span>
}

function Welcome({ onNext }: { onNext: () => void }) {
  const t = useT()
  const lang = useLang((s) => s.lang)
  const setLang = useLang((s) => s.setLang)
  const [agree, setAgree] = useState(false)
  const [reading, setReading] = useState(false)
  const setShell = useBoot((s) => s.setShell)
  return (
    <div className="ob-welcome">
      <div className="ob-float">
        <AppTile size={96} />
      </div>
      <h1>{t('ob.welcome.title')}</h1>
      <p className="secondary">{t('ob.welcome.body')}</p>
      <FreeLine />
      <Segmented<'zh-CN' | 'en'>
        size="sm"
        value={lang}
        onChange={(l) => {
          setLang(l)
          if (isTauri) void shell.setLanguage(l)
        }}
        options={[
          { value: 'zh-CN', label: '简体中文' },
          { value: 'en', label: 'English' },
        ]}
      />
      <label className="ob-agree">
        <Switch checked={agree} onChange={setAgree} label={t('ob.welcome.agree')} />
        <span>
          {t('ob.welcome.agree_pre')}
          <button
            type="button"
            className="link-btn"
            onClick={(e) => {
              e.preventDefault()
              setReading(true)
            }}
          >
            {t('ob.welcome.agree_link')}
          </button>
          {t('ob.welcome.agree_post')}
        </span>
      </label>
      <TermsDialog open={reading} onOpenChange={setReading} />
      <button
        className="btn btn-primary btn-lg"
        disabled={!agree}
        onClick={() =>
          void attempt(async () => {
            const next = await acceptTerms()
            if (next) setShell(next)
            onNext()
          })
        }
      >
        {t('ob.start')}
        <ChevronRight />
      </button>
    </div>
  )
}

function Library({ onNext }: { onNext: () => void }) {
  const t = useT()
  const shellState = useBoot((s) => s.shell)
  const setShell = useBoot((s) => s.setShell)
  const [space, setSpace] = useState<{ free: number; total: number } | null>(null)
  const path = shellState?.libraryPath ?? ''

  useEffect(() => {
    if (path) void shell.diskSpace(path).then(setSpace).catch(() => setSpace(null))
  }, [path])

  const change = async () => {
    const p = await pickFolder()
    if (p) setShell(await shell.setLibraryPath(p))
  }

  return (
    <div className="ob-body">
      <HardDrive className="ob-hero-icon" />
      <h2>{t('ob.library.title')}</h2>
      <p className="secondary">{t('ob.library.body')}</p>
      <div className="ob-path">
        <FolderOpen />
        <span className="mono">{path}</span>
        <span className="spacer" />
        {space && <span className={clsx('chip', space.free < 20 * 1024 ** 3 ? 'warning' : 'good')}>{t('ob.library.free', { free: bytes(space.free) })}</span>}
        <button className="btn btn-sm" onClick={change}>
          {t('set.library_change')}
        </button>
      </div>
      {space && space.free < 20 * 1024 ** 3 && <p className="ob-warn">{t('ob.library.low')}</p>}
      <div className="ob-actions">
        <button className="btn btn-primary btn-lg" onClick={onNext}>
          {t('common.next')}
          <ChevronRight />
        </button>
      </div>
    </div>
  )
}

function Hardware({ region, setRegion, onNext }: { region: Region; setRegion: (r: Region) => void; onNext: (v: TorchVariant) => void }) {
  const t = useT()
  const [hw, setHw] = useState<HardwareReport | null>(null)
  const [variant, setVariant] = useState<TorchVariant | null>(null)
  useEffect(() => {
    void shell.detectHardware().then((h) => {
      setHw(h)
      setVariant(h.recommendedVariant)
    })
  }, [])
  const blocked = hw?.notes.some((n) => n.code === 'mac.intel_unsupported')
  return (
    <div className="ob-body">
      <Cpu className="ob-hero-icon" />
      <h2>{t('ob.hardware.title')}</h2>
      {!hw ? (
        <div className="ob-scan">
          <Loader2 className="spin" />
          {t('ob.hardware.scanning')}
        </div>
      ) : (
        <>
          <div className="ob-hw">
            <div className="ob-hw-row">
              <span className="eyebrow">CPU</span>
              <span>{hw.cpu}</span>
              <span className="muted">{bytes(hw.ramTotal)} RAM</span>
            </div>
            {hw.gpus.map((g, i) => (
              <div key={i} className="ob-hw-row">
                <span className="eyebrow">GPU</span>
                <span>{g.name}</span>
                <span className="muted">{g.vram ? bytes(g.vram) : ''}</span>
              </div>
            ))}
          </div>
          <div className={clsx('ob-verdict', hw.canTrain ? 'good' : 'info')}>
            <CheckCircle2 />
            <div>
              <strong>{hw.canTrain ? t('ob.hardware.can_train') : t('ob.hardware.create_only')}</strong>
              {!hw.canTrain && <p>{t('ob.hardware.train_note')}</p>}
            </div>
          </div>
          {hw.notes.map((n) => (
            <div key={n.code} className="ob-note">
              <AlertTriangle />
              {t(`ob.note.${n.code}` as 'ob.note.gpu.no_nvidia')}
            </div>
          ))}
          <div className="ob-options">
            <div className="field">
              <span className="field-label">{t('ob.hardware.variant_label')}</span>
              <select className="input" value={variant ?? ''} onChange={(e) => setVariant(e.target.value as TorchVariant)}>
                {(['cu128', 'cu126', 'cpu', 'mac'] as const)
                  .filter((v) => (hw.appleSilicon ? v === 'mac' || v === 'cpu' : v !== 'mac'))
                  .map((v) => (
                    <option key={v} value={v}>
                      {t(`ob.variant.${v}` as const)}
                      {v === hw.recommendedVariant ? ` · ${t('common.recommended')}` : ''}
                    </option>
                  ))}
              </select>
            </div>
            <div className="field">
              <span className="field-label">{t('ob.region')}</span>
              <Segmented<Region>
                size="sm"
                value={region}
                onChange={setRegion}
                options={[
                  { value: 'auto', label: t('res.region.auto') },
                  { value: 'cn', label: t('res.region.cn') },
                  { value: 'global', label: t('res.region.global') },
                ]}
              />
            </div>
          </div>
          <div className="ob-actions">
            <button className="btn btn-primary btn-lg" disabled={!variant || blocked} onClick={() => variant && onNext(variant)}>
              {t('common.next')}
              <ChevronRight />
            </button>
          </div>
        </>
      )}
    </div>
  )
}

/** The best environment found on this computer (or what was found), under the "use existing" choice. */
function FoundSummary({ found }: { found: ReturnType<typeof useFoundPythons> }) {
  const t = useT()
  if (found.best) {
    const torch = found.best.check.torch
    return (
      <span className="good-text">
        {t('ob.rt.found_best', { name: found.best.c.label, torch: torch?.version.split('+')[0] ?? '', device: torch?.cudaAvailable ? 'CUDA' : 'CPU' })}
      </span>
    )
  }
  const conda = (found.nearby ?? []).filter((c) => c.source === 'conda')
  if (!conda.length) return null
  return <span className="muted">{t('ob.rt.found_some', { n: conda.length, names: conda.slice(0, 3).map((c) => c.label).join('、') })}</span>
}

function Install({ variant, region, setRegion, onDone }: { variant: TorchVariant; region: Region; setRegion: (r: Region) => void; onDone: () => void }) {
  const t = useT()
  const connect = useBoot((s) => s.connect)
  const setShell = useBoot((s) => s.setShell)
  const [p, setP] = useState<BootstrapProgress | null>(null)
  const [started, setStarted] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [existing, setExisting] = useState(false)
  const [switching, setSwitching] = useState(false)
  // look for conda environments and other Pythons right away, so the choice can say what is there
  const found = useFoundPythons()

  /** Run on the user's own interpreter: the backend starts on it, then the models step. */
  const switchToExisting = async (path: string) => {
    setSwitching(true)
    const next = await attempt(() => shell.setRuntimePython(path))
    if (next) {
      setShell(next)
      setConnecting(true)
      await connect()
      onDone()
    }
    setSwitching(false)
  }

  useEffect(() => {
    let off: (() => void) | undefined
    void listen<BootstrapProgress>('bootstrap://progress', setP).then((f) => (off = f))
    return () => off?.()
  }, [])

  useEffect(() => {
    // only the runtime install started here: adding packages to an existing Python reports on
    // the same channel
    if (!started || p?.stage !== 'done' || connecting) return
    setConnecting(true)
    void connect().then(onDone)
  }, [started, p?.stage, connect, onDone, connecting])

  const start = async () => {
    setStarted(true)
    setP(null)
    await shell.bootstrapStart(variant, region).catch((e) => setP({ stage: 'error', stageIndex: 0, stageCount: 6, progress: null, overall: null, message: String(e), detail: null, bytesDone: null, bytesTotal: null, speed: null, source: null, error: { code: 'bootstrap.failed', message: String(e), detail: '' } }))
  }

  const failed = p?.stage === 'error' || p?.stage === 'cancelled'
  const currentIdx = p ? STAGES.indexOf(p.stage as (typeof STAGES)[number]) : -1

  if (existing && !started) {
    return (
      <div className="ob-body">
        <FolderInput className="ob-hero-icon" />
        <h2>{t('ob.rt.existing_title')}</h2>
        <p className="secondary">{t('ob.rt.existing_intro')}</p>
        <ExistingPython
          onUse={(path) => void switchToExisting(path)}
          onManaged={() => {
            setExisting(false)
            void start()
          }}
          busy={switching}
          onFolder={(f) => store(FOLDER_KEY, f)}
          found={found}
        />
        {connecting && (
          <div className="ob-detail">
            <Loader2 className="spin" />
            {t('ob.starting_backend')}
          </div>
        )}
        <div className="ob-actions start">
          <button className="btn btn-ghost" onClick={() => setExisting(false)} disabled={switching}>
            <ArrowLeft />
            {t('ob.rt.back')}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="ob-body">
      <Download className="ob-hero-icon" />
      <h2>{started ? t('ob.install.title') : t('ob.rt.title')}</h2>
      <p className="secondary">{started ? t('ob.install.body') : t('ob.rt.body')}</p>

      {!started ? (
        <div className="ob-choices">
          <button className="ob-choice recommended" onClick={start}>
            <Download />
            <span className="ob-choice-text">
              <strong>
                {t('ob.rt.auto')}
                <span className="chip accent">{t('common.recommended')}</span>
              </strong>
              <span className="secondary">{t('ob.rt.auto_body')}</span>
              <span className="muted">{t('ob.hardware.variant', { variant: t(`ob.variant.${variant}` as const) })}</span>
            </span>
            <ChevronRight />
          </button>
          <button className="ob-choice" onClick={() => setExisting(true)}>
            <FolderInput />
            <span className="ob-choice-text">
              <strong>
                {t('ob.rt.existing')}
                {found.best && <span className="chip good">{t('ob.rt.found')}</span>}
              </strong>
              <span className="secondary">{t('ob.rt.existing_body')}</span>
              <FoundSummary found={found} />
            </span>
            <ChevronRight />
          </button>
        </div>
      ) : (
        <div className="ob-install">
          <Ring value={p?.overall ?? 0} size={120} stroke={9}>
            <div className="ob-install-pct tabular">{p?.stage === 'done' ? <Check /> : `${Math.round((p?.overall ?? 0) * 100)}%`}</div>
          </Ring>
          <div className="ob-stages">
            {STAGES.map((s, i) => (
              <div key={s} className={clsx('ob-stage', i === currentIdx && 'on', (i < currentIdx || p?.stage === 'done') && 'done')}>
                <span className="ob-stage-dot">{i < currentIdx || p?.stage === 'done' ? <Check /> : i === currentIdx ? <Loader2 className="spin" /> : null}</span>
                <span>{t(`ob.install.stage.${s}` as const)}</span>
                {i === currentIdx && p?.progress != null && <span className="muted tabular">{Math.round(p.progress * 100)}%</span>}
              </div>
            ))}
          </div>
          {p && !failed && (
            <div className="ob-detail">
              <span className="mono">{p.detail ?? p.message}</span>
              <span className="muted tabular">
                {p.bytesDone != null && p.bytesTotal ? `${bytes(p.bytesDone)} / ${bytes(p.bytesTotal)} ` : ''}
                {speed(p.speed)}
                {p.source ? ` · ${p.source}` : ''}
              </span>
            </div>
          )}
          {connecting && (
            <div className="ob-detail">
              <Loader2 className="spin" />
              {t('ob.starting_backend')}
            </div>
          )}
          {failed && (
            <div className="ob-failed">
              <div className="row" style={{ gap: 8 }}>
                <AlertTriangle />
                <strong>{t('ob.install.failed')}</strong>
              </div>
              <p className="secondary">{t('ob.install.failed_body')}</p>
              {p?.error && <pre className="ob-error mono">{`${p.error.message}\n${p.error.detail}`.trim()}</pre>}
              <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
                <Segmented<Region>
                  size="sm"
                  value={region}
                  onChange={setRegion}
                  options={[
                    { value: 'auto', label: t('res.region.auto') },
                    { value: 'cn', label: t('res.region.cn') },
                    { value: 'global', label: t('res.region.global') },
                  ]}
                />
                <button className="btn btn-primary" onClick={start}>
                  <RotateCcw />
                  {t('common.retry')}
                </button>
                <button className="btn btn-ghost" onClick={() => navigator.clipboard.writeText(`${p?.error?.message}\n${p?.error?.detail}`)}>
                  {t('common.copy_error')}
                </button>
              </div>
            </div>
          )}
          {!failed && p?.stage !== 'done' && (
            <button className="btn btn-ghost btn-sm" onClick={() => shell.bootstrapCancel()}>
              {t('ob.install.cancel')}
            </button>
          )}
        </div>
      )}
    </div>
  )
}

function Models({ onDone }: { onDone: () => void }) {
  const t = useT()
  const qc = useQueryClient()
  const url = useConnection((s) => s.url)
  const phase = useBoot((s) => s.phase)
  const connect = useBoot((s) => s.connect)
  const { data: resources } = useResources()
  const required = (resources ?? []).filter((r) => r.required)
  const allReady = required.length > 0 && required.every((r) => r.status === 'ready')
  const busy = required.some((r) => r.status === 'downloading' || r.status === 'installing')
  const [folder, setFolder] = useState(() => stored(FOLDER_KEY))
  const [importing, setImporting] = useState(false)

  const importFrom = async (path: string | null) => {
    const chosen = path ?? (await pickFolder())
    if (!chosen) return
    setFolder(chosen)
    store(FOLDER_KEY, chosen)
    setImporting(true)
    await importModelsFromFolder(chosen, resources ?? [])
    setImporting(false)
  }

  const later = () => {
    store(MODELS_LATER_KEY, '1')
    onDone()
  }

  useEffect(() => {
    if (!url) void connect()
  }, [url, connect, phase])

  useEffect(() => (url ? startLive(qc) : undefined), [url, qc])

  useEffect(() => {
    if (allReady) onDone()
  }, [allReady, onDone])

  return (
    <div className="ob-body">
      <Download className="ob-hero-icon" />
      <h2>{t('ob.models.title')}</h2>
      <p className="secondary">{t('ob.models.body')}</p>
      {!url ? (
        <div className="ob-scan">
          <Loader2 className="spin" />
          {t('ob.starting_backend')}
        </div>
      ) : (
        <>
          <div className="ob-models">
            {required.map((r) => (
              <ModelRow key={r.id} r={r} />
            ))}
          </div>
          <div className="ob-actions">
            <button className="btn btn-primary btn-lg" disabled={busy || allReady} onClick={() => attempt(() => api.post('/resources/install-required'))}>
              {busy ? <Loader2 className="spin" /> : <Download />}
              {t('ob.models.go')}
            </button>
          </div>
          {isTauri && (
            <div className="ob-have">
              <span className="secondary">{t('ob.models.have')}</span>
              <div className="row" style={{ gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
                {folder && (
                  <button className="btn btn-sm" onClick={() => void importFrom(folder)} disabled={busy || importing || allReady} title={folder}>
                    {importing ? <Loader2 className="spin" /> : <FolderInput />}
                    {t('ob.models.from_folder', { name: folder.split(/[\\/]/).filter(Boolean).pop() ?? folder })}
                  </button>
                )}
                <button className="btn btn-sm btn-ghost" onClick={() => void importFrom(null)} disabled={busy || importing || allReady}>
                  <FolderOpen />
                  {folder ? t('ob.models.other_folder') : t('ob.models.pick_folder')}
                </button>
              </div>
              <span className="muted ob-have-hint">{t('ob.models.folder_hint')}</span>
            </div>
          )}
          <button className="btn btn-ghost btn-sm ob-later" onClick={later} disabled={busy || importing}>
            {t('ob.models.later')}
          </button>
        </>
      )}
    </div>
  )
}

function ModelRow({ r }: { r: Resource }) {
  const t = useT()
  return (
    <div className="ob-model">
      <span className={clsx('ob-stage-dot', r.status === 'ready' && 'ready')}>{r.status === 'ready' ? <Check /> : r.status === 'downloading' || r.status === 'installing' ? <Loader2 className="spin" /> : null}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div className="row">
          <strong>{t.msg(r.name, r.id)}</strong>
          <span className="spacer" />
          <span className="muted tabular">
            {r.status === 'downloading' && r.bytesDone != null
              ? `${bytes(r.bytesDone)}${r.size > 0 ? ` / ${bytes(r.size)}` : ''} ${speed(r.speed)}`
              : r.size > 0
                ? bytes(r.size)
                : ''}
          </span>
        </div>
        {(r.status === 'downloading' || r.status === 'installing') && <Progress value={r.progress} />}
        {r.status === 'failed' && <span className="res-error">{t.msg(r.error)}</span>}
      </div>
    </div>
  )
}

function Done() {
  const t = useT()
  const finish = useBoot((s) => s.finishOnboarding)
  return (
    <div className="ob-welcome">
      <div className="ob-logo done">
        <Check />
      </div>
      <h1>{t('ob.done.title')}</h1>
      <p className="secondary">{t('ob.done.body')}</p>
      <button className="btn btn-primary btn-lg" onClick={() => finish()}>
        {t('ob.done.go')}
        <ChevronRight />
      </button>
    </div>
  )
}

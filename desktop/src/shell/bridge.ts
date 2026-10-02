// Bridge to the Tauri shell (docs/api.md §2) with browser fallbacks, so the UI also runs
// in a plain browser against a dev backend.

export type TorchVariant = 'cu128' | 'cu126' | 'cpu' | 'mac'
export type Region = 'auto' | 'cn' | 'global'

export interface BackendStatus {
  status: 'stopped' | 'starting' | 'ready' | 'crashed'
  url: string | null
  token: string | null
  error: string | null
  restarts: number
}

export interface ShellState {
  platform: 'windows' | 'macos' | 'linux'
  arch: string
  appVersion: string
  firstRunDone: boolean
  libraryPath: string
  dataPath: string
  language: 'zh-CN' | 'en' | null
  /** version of the terms of use the user accepted; 0 = none yet */
  termsAccepted: number
  /** the app's own runtime is installed (the way back when the user's Python breaks) */
  managedReady: boolean
  runtime: {
    status: 'missing' | 'outdated' | 'ready' | 'installing' | 'broken'
    variant: TorchVariant | null
    python: string | null
    torch: string | null
    cuda: string | null
    /** managed = the app's own runtime; custom = a Python the user picked */
    mode?: 'managed' | 'custom'
    pythonPath?: string | null
    /** the user's Python runs through Stellaria's layer (packages it lacked live there) */
    layered?: boolean
  }
  backend: BackendStatus
}

export interface PythonCheck {
  ok: boolean
  executable: string
  version: string | null
  missing: string[]
  error: { code: string; params?: Record<string, string | number> } | null
  torch: { version: string; cuda: string | null; cudaAvailable: boolean } | null
  /** checked through Stellaria's layer over this Python: what it lacked is installed there */
  layered?: boolean
}

/** An interpreter found on this computer (not checked yet). */
export interface PythonCandidate {
  path: string
  /** inside the chosen folder, a conda environment, or a system install */
  source: 'folder' | 'conda' | 'system'
  label: string
}

export interface HardwareReport {
  os: string
  osVersion: string
  cpu: string
  ramTotal: number
  gpus: {
    vendor: 'nvidia' | 'amd' | 'intel' | 'apple' | 'other'
    name: string
    vram: number | null
    driver: string | null
    computeCapability: string | null
  }[]
  appleSilicon: boolean
  recommendedVariant: TorchVariant
  canTrain: boolean
  notes: { code: string; params?: Record<string, string | number> }[]
}

export interface BootstrapProgress {
  stage: 'prepare' | 'python' | 'venv' | 'torch' | 'packages' | 'verify' | 'done' | 'error' | 'cancelled'
  stageIndex: number
  stageCount: number
  progress: number | null
  overall: number | null
  message: string
  detail: string | null
  bytesDone: number | null
  bytesTotal: number | null
  speed: number | null
  source: string | null
  error: { code: string; message: string; detail: string } | null
}

export const isTauri: boolean = typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window

export const platform: 'windows' | 'macos' | 'linux' | 'web' = (() => {
  if (!isTauri) return 'web'
  const ua = navigator.userAgent
  if (/Windows/i.test(ua)) return 'windows'
  if (/Mac OS X|Macintosh/i.test(ua)) return 'macos'
  return 'linux'
})()

/** Shortcuts as this platform writes them: ⌘C on a Mac, Ctrl+C elsewhere. */
export const modKey = (key: string) => (platform === 'macos' ? `⌘${key}` : `Ctrl+${key}`)
export const RUN_KEYS = modKey('⏎')

export async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const core = await import('@tauri-apps/api/core')
  return core.invoke<T>(cmd, args)
}

export async function listen<T>(event: string, handler: (payload: T) => void): Promise<() => void> {
  if (!isTauri) return () => {}
  const ev = await import('@tauri-apps/api/event')
  return ev.listen<T>(event, (e) => handler(e.payload))
}

export const shell = {
  getState: () => invoke<ShellState>('get_shell_state'),
  detectHardware: () => invoke<HardwareReport>('detect_hardware'),
  diskSpace: (path: string) => invoke<{ free: number; total: number }>('disk_space', { path }),
  setLibraryPath: (path: string) => invoke<ShellState>('set_library_path', { path }),
  setLanguage: (language: 'zh-CN' | 'en') => invoke<ShellState>('set_language', { language }),
  bootstrapStart: (variant: TorchVariant, region: Region) => invoke<null>('bootstrap_start', { variant, region }),
  bootstrapCancel: () => invoke<null>('bootstrap_cancel'),
  bootstrapRepair: () => invoke<null>('bootstrap_repair'),
  finishOnboarding: () => invoke<ShellState>('finish_onboarding'),
  backendStart: () => invoke<BackendStatus>('backend_start'),
  backendRestart: () => invoke<BackendStatus>('backend_restart'),
  backendLogs: (tail = 400) => invoke<string[]>('backend_logs', { tail }),
  setCloseToTray: (enabled: boolean) => invoke<null>('set_close_to_tray', { enabled }),
  acceptTerms: (version: number) => invoke<ShellState>('accept_terms', { version }),
  setTrayStatus: (text: string | null, progress: number | null) => invoke<null>('set_tray_status', { text, progress }),
  revealPath: (path: string) => invoke<null>('reveal_path', { path }),
  openLogsFolder: () => invoke<null>('open_logs_folder'),
  quitApp: (force = false) => invoke<null>('quit_app', { force }),
  checkPython: (path: string) => invoke<PythonCheck>('check_python', { path }),
  findPythons: (folder: string | null = null) => invoke<PythonCandidate[]>('find_pythons', { folder }),
  setRuntimePython: (path: string | null) => invoke<ShellState>('set_runtime_python', { path }),
  /** what `path` lacks, into Stellaria's layer over it (its own environment is never written to) */
  installRuntimeRequirements: (path: string) => invoke<null>('install_runtime_requirements', { path }),
}

/** Backend location in browser mode: ?backend=…&token=… or Vite env, else the dev default. */
export function browserBackend(): { url: string; token: string } {
  const params = new URLSearchParams(window.location.search)
  const url = params.get('backend') ?? import.meta.env.VITE_BACKEND_URL ?? 'http://127.0.0.1:8765'
  const token = params.get('token') ?? import.meta.env.VITE_BACKEND_TOKEN ?? 'dev'
  return { url: url.replace(/\/$/, ''), token }
}

// ---------- OS integration with fallbacks ----------

export async function pickFiles(opts: { multiple?: boolean; directory?: boolean; audioOnly?: boolean; extensions?: string[]; filterName?: string }): Promise<string[] | File[] | null> {
  if (isTauri) {
    const dialog = await import('@tauri-apps/plugin-dialog')
    const extensions = opts.extensions ?? AUDIO_EXTENSIONS
    const res = await dialog.open({
      multiple: opts.multiple ?? true,
      directory: opts.directory ?? false,
      filters: opts.directory ? undefined : [{ name: opts.filterName ?? 'Audio', extensions }],
    })
    if (res == null) return null
    return Array.isArray(res) ? res : [res]
  }
  return new Promise((resolve) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = opts.multiple ?? true
    const extensions = opts.extensions ?? AUDIO_EXTENSIONS
    input.accept = extensions.map((e) => '.' + e).join(',')
    input.onchange = () => resolve(input.files ? Array.from(input.files) : null)
    input.click()
  })
}

export async function pickSavePath(defaultPath: string, extensions: string[]): Promise<string | null> {
  if (!isTauri) return null
  const dialog = await import('@tauri-apps/plugin-dialog')
  return dialog.save({ defaultPath, filters: [{ name: extensions[0].toUpperCase(), extensions }] })
}

/** A Python interpreter: python.exe on Windows, any file elsewhere (bin/python3 has no extension). */
export async function pickExecutable(): Promise<string | null> {
  if (!isTauri) return null
  const dialog = await import('@tauri-apps/plugin-dialog')
  const res = await dialog.open({ multiple: false, directory: false, filters: platform === 'windows' ? [{ name: 'Python', extensions: ['exe'] }] : undefined })
  return typeof res === 'string' ? res : null
}

export async function pickFolder(): Promise<string | null> {
  if (!isTauri) return null
  const dialog = await import('@tauri-apps/plugin-dialog')
  const res = await dialog.open({ directory: true, multiple: false })
  return typeof res === 'string' ? res : null
}

/** A web page in the default browser. */
export async function openUrl(url: string): Promise<void> {
  if (isTauri) {
    const opener = await import('@tauri-apps/plugin-opener')
    await opener.openUrl(url)
    return
  }
  window.open(url, '_blank', 'noopener')
}

export async function revealPath(path: string): Promise<void> {
  if (isTauri) await shell.revealPath(path)
}

export async function notify(title: string, body: string): Promise<void> {
  if (isTauri) {
    const n = await import('@tauri-apps/plugin-notification')
    let granted = await n.isPermissionGranted()
    if (!granted) granted = (await n.requestPermission()) === 'granted'
    if (granted) n.sendNotification({ title, body })
    return
  }
  if ('Notification' in window && Notification.permission === 'granted') new Notification(title, { body })
}

export async function windowAction(action: 'minimize' | 'toggleMaximize' | 'close' | 'hide' | 'show'): Promise<void> {
  if (!isTauri) return
  const { getCurrentWindow } = await import('@tauri-apps/api/window')
  const w = getCurrentWindow()
  if (action === 'minimize') await w.minimize()
  else if (action === 'toggleMaximize') await w.toggleMaximize()
  else if (action === 'close') await w.close()
  else if (action === 'hide') await w.hide()
  else await w.show()
}

/** Tauri delivers OS file drops through the webview instead of HTML5 drop events. */
export async function onNativeDrop(handler: (e: { kind: 'enter' | 'over' | 'leave' | 'drop'; paths: string[]; x: number; y: number }) => void): Promise<() => void> {
  if (!isTauri) return () => {}
  const { getCurrentWebview } = await import('@tauri-apps/api/webview')
  return getCurrentWebview().onDragDropEvent((event) => {
    const p = event.payload
    const ratio = window.devicePixelRatio || 1
    if (p.type === 'enter') handler({ kind: 'enter', paths: p.paths, x: p.position.x / ratio, y: p.position.y / ratio })
    else if (p.type === 'over') handler({ kind: 'over', paths: [], x: p.position.x / ratio, y: p.position.y / ratio })
    else if (p.type === 'drop') handler({ kind: 'drop', paths: p.paths, x: p.position.x / ratio, y: p.position.y / ratio })
    else handler({ kind: 'leave', paths: [], x: 0, y: 0 })
  })
}

export const AUDIO_EXTENSIONS = ['wav', 'flac', 'mp3', 'm4a', 'aac', 'ogg', 'opus', 'wma', 'mp4', 'mkv', 'mov', 'webm']

import clsx from 'clsx'
import { Minus, Search, Settings, Square, X } from 'lucide-react'
import { useEffect, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useLive } from '../api/live'
import { useT } from '../i18n'
import { platform, windowAction } from '../shell/bridge'
import { useUi } from '../state/ui'
import { Logo } from './Logo'
import { Segmented, Tip } from './ui'
import './titlebar.css'

export type Space = 'voices' | 'create' | 'resources'

export function spaceOf(pathname: string): Space | 'settings' {
  if (pathname.startsWith('/create')) return 'create'
  if (pathname.startsWith('/resources')) return 'resources'
  if (pathname.startsWith('/settings')) return 'settings'
  return 'voices'
}

/** Custom titlebar: brand, the space switcher, search, settings and (Windows) window buttons. */
export function TitleBar() {
  const t = useT()
  const nav = useNavigate()
  const { pathname } = useLocation()
  const space = spaceOf(pathname)
  // where each space was left, so switching back returns there
  const lastPath = useRef<Partial<Record<Space, string>>>({})
  useEffect(() => {
    if (space !== 'settings') lastPath.current[space] = pathname
  }, [space, pathname])
  const setPalette = useUi((s) => s.setPalette)
  const connected = useLive((s) => s.connected)
  const mod = platform === 'macos' ? '⌘' : 'Ctrl'

  const go = (s: Space) => {
    const back = lastPath.current[s]
    nav(back && s !== space ? back : `/${s}`)
  }

  return (
    <header className={clsx('titlebar', platform === 'macos' && 'mac')} data-tauri-drag-region>
      <div className="tb-brand" data-tauri-drag-region>
        <span className="tb-logo">
          <Logo size={20} />
        </span>
        <span className="tb-name" data-tauri-drag-region>
          {t('app.name')}
        </span>
        {!connected && <span className="status-dot warning pulse" title={t('conn.offline')} />}
      </div>

      <div className="tb-center" data-tauri-drag-region>
        <Segmented<Space>
          value={space === 'settings' ? ('' as Space) : space}
          onChange={go}
          options={[
            { value: 'voices', label: t('space.voices') },
            { value: 'create', label: t('space.create') },
            { value: 'resources', label: t('space.resources') },
          ]}
        />
      </div>

      <div className="tb-right">
        <button className="tb-search" onClick={() => setPalette(true)}>
          <Search />
          <span>{t('titlebar.search')}</span>
          <span className="kbd">{mod}</span>
          <span className="kbd">K</span>
        </button>
        <Tip label={t('space.settings')} side="bottom">
          <button className={clsx('icon-btn', space === 'settings' && 'active')} onClick={() => nav('/settings')} aria-label={t('space.settings')}>
            <Settings />
          </button>
        </Tip>
        {platform === 'windows' && (
          <div className="tb-window">
            <button onClick={() => windowAction('minimize')} aria-label={t('titlebar.minimize')}>
              <Minus />
            </button>
            <button onClick={() => windowAction('toggleMaximize')} aria-label={t('titlebar.maximize')}>
              <Square />
            </button>
            <button className="close" onClick={() => windowAction('close')} aria-label={t('titlebar.close')}>
              <X />
            </button>
          </div>
        )}
      </div>
    </header>
  )
}

import { Minus, Square, X } from 'lucide-react'
import { useT } from '../i18n'
import { isTauri, platform, windowAction } from '../shell/bridge'
import './window-chrome.css'

/**
 * The window's own strip for screens shown before the titlebar (setup, connecting): a drag
 * area everywhere, plus minimize / maximize / close on Windows, where the system frame is off.
 */
export function WindowChrome() {
  const t = useT()
  if (!isTauri) return null
  return (
    <div className="win-chrome" data-tauri-drag-region>
      {platform === 'windows' && (
        <div className="win-chrome-buttons">
          <button onClick={() => windowAction('minimize')} aria-label={t('titlebar.minimize')} title={t('titlebar.minimize')}>
            <Minus />
          </button>
          <button onClick={() => windowAction('toggleMaximize')} aria-label={t('titlebar.maximize')} title={t('titlebar.maximize')}>
            <Square />
          </button>
          <button className="close" onClick={() => windowAction('close')} aria-label={t('titlebar.close')} title={t('titlebar.close')}>
            <X />
          </button>
        </div>
      )}
    </div>
  )
}

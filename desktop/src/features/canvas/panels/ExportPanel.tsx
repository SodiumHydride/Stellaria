import clsx from 'clsx'
import { Check, CheckCircle2, Copy, CornerDownRight, FolderOpen, X } from 'lucide-react'
import { useRef, useState } from 'react'
import { usePreferences } from '../../../api/hooks'
import type { ExportParams } from '../../../api/types'
import { Segmented, Slider, Switch, Tip } from '../../../components/ui'
import { useT, type StringKey } from '../../../i18n'
import { isTauri, pickFolder, revealPath } from '../../../shell/bridge'
import './export.css'

const TOKENS = ['project', 'song', 'voice', 'label', 'date', 'index'] as const
const DEFAULT_NAME = '{project} - {song}'
const LOUDNESS_DEFAULT = -14

/** Export node (canvas v3 §1.3): format, loudness, where and under which name the file is written. */
export function ExportPanel({ params, onChange, result }: { params: ExportParams; onChange: (next: ExportParams) => void; result?: { path?: string } | null }) {
  const t = useT()
  const { data: prefs } = usePreferences()
  const nameRef = useRef<HTMLInputElement>(null)
  const [copied, setCopied] = useState(false)
  const set = (patch: Partial<ExportParams>) => onChange({ ...params, ...patch })

  const setFormat = (format: ExportParams['format']) => set({ format, bitDepth: format === 'flac' && params.bitDepth === 32 ? 24 : params.bitDepth })

  const insertToken = (token: string) => {
    const el = nameRef.current
    const value = params.fileName
    const start = el?.selectionStart ?? value.length
    const end = el?.selectionEnd ?? value.length
    const before = value.slice(0, start)
    const glue = /\}\s*$/.test(before) && !/\s-\s*$/.test(before) ? (before.endsWith(' ') ? '- ' : ' - ') : ''
    const insert = `${glue}{${token}}`
    set({ fileName: before + insert + value.slice(end) })
    requestAnimationFrame(() => {
      el?.focus()
      el?.setSelectionRange(start + insert.length, start + insert.length)
    })
  }

  const chooseFolder = async () => {
    const folder = await pickFolder()
    if (folder) set({ folder })
  }

  const copyPath = async (path: string) => {
    try {
      await navigator.clipboard.writeText(path)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1400)
    } catch {
      /* clipboard unavailable */
    }
  }

  const example = exampleName(params.fileName || DEFAULT_NAME, params.format, (k) => t(k))
  const defaultFolder = prefs?.exportFolder || null

  return (
    <div className="exp">
      {result?.path && (
        <div className="exp-result">
          <div className="exp-result-head">
            <CheckCircle2 />
            <span>{t('exp.done')}</span>
          </div>
          <div className="exp-result-path mono" title={result.path}>
            {result.path}
          </div>
          <div className="exp-result-actions">
            {isTauri && (
              <button className="btn btn-sm" onClick={() => void revealPath(result.path!)}>
                <FolderOpen />
                {t('exp.reveal')}
              </button>
            )}
            <button className="btn btn-sm btn-ghost" onClick={() => void copyPath(result.path!)}>
              {copied ? <Check /> : <Copy />}
              {copied ? t('common.copied') : t('exp.copy_path')}
            </button>
          </div>
        </div>
      )}

      <section className="exp-sec">
        <div className="exp-sec-head">
          <span className="eyebrow">{t('exp.format')}</span>
        </div>
        <div className="exp-seg-full">
          <Segmented<ExportParams['format']>
            value={params.format}
            onChange={setFormat}
            options={[
              { value: 'wav', label: 'WAV' },
              { value: 'flac', label: 'FLAC' },
              { value: 'mp3', label: 'MP3' },
            ]}
          />
        </div>
        <div className="exp-row">
          <span className="exp-label">{t('exp.sample_rate')}</span>
          <Segmented<string>
            size="sm"
            value={String(params.sampleRate)}
            onChange={(v) => set({ sampleRate: Number(v) as ExportParams['sampleRate'] })}
            options={[
              { value: '44100', label: '44.1 kHz' },
              { value: '48000', label: '48 kHz' },
            ]}
          />
        </div>
        {params.format === 'mp3' ? (
          <div className="exp-row">
            <span className="exp-label">
              {t('exp.bitrate')} <span className="exp-unit">kbps</span>
            </span>
            <Segmented<string>
              size="sm"
              value={String(params.mp3Bitrate)}
              onChange={(v) => set({ mp3Bitrate: Number(v) as ExportParams['mp3Bitrate'] })}
              options={['192', '256', '320'].map((v) => ({ value: v, label: v }))}
            />
          </div>
        ) : (
          <div className="exp-row">
            <span className="exp-label">{t('exp.bit_depth')}</span>
            <Segmented<string>
              size="sm"
              value={String(params.bitDepth)}
              onChange={(v) => set({ bitDepth: Number(v) as ExportParams['bitDepth'] })}
              options={[
                { value: '16', label: '16' },
                { value: '24', label: '24' },
                ...(params.format === 'wav' ? [{ value: '32', label: t('exp.float'), title: t('exp.float') }] : []),
              ]}
            />
          </div>
        )}
      </section>

      <section className="exp-sec">
        <div className="exp-sec-head">
          <span className="eyebrow">{t('exp.sec_file')}</span>
        </div>
        <div className="exp-field">
          <span className="exp-label">{t('exp.folder')}</span>
          {isTauri ? (
            <div className="exp-folder">
              <button className="exp-folder-pick" onClick={chooseFolder} title={params.folder ?? defaultFolder ?? undefined}>
                <FolderOpen />
                <span className={clsx('exp-folder-path', !params.folder && 'is-default')}>{params.folder ? shortPath(params.folder) : t('exp.folder_default')}</span>
              </button>
              {params.folder && (
                <Tip label={t('exp.folder_reset')}>
                  <button className="icon-btn sm" onClick={() => set({ folder: null })} aria-label={t('exp.folder_reset')}>
                    <X />
                  </button>
                </Tip>
              )}
            </div>
          ) : (
            <input
              className="input mono exp-input"
              value={params.folder ?? ''}
              placeholder={defaultFolder ?? t('exp.folder_default')}
              onChange={(e) => set({ folder: e.target.value.trim() ? e.target.value : null })}
              aria-label={t('exp.folder')}
            />
          )}
        </div>
        <div className="exp-field">
          <Tip label={t('exp.file_name_hint')}>
            <span className="exp-label has-hint">{t('exp.file_name')}</span>
          </Tip>
          <input
            ref={nameRef}
            className="input exp-input"
            value={params.fileName}
            placeholder={DEFAULT_NAME}
            spellCheck={false}
            onChange={(e) => set({ fileName: e.target.value })}
            onBlur={() => !params.fileName.trim() && set({ fileName: DEFAULT_NAME })}
            aria-label={t('exp.file_name')}
          />
          <div className="exp-tokens">
            {TOKENS.map((k) => (
              <Tip key={k} label={`{${k}}`} delay={500}>
                <button className="exp-token" onMouseDown={(e) => e.preventDefault()} onClick={() => insertToken(k)}>
                  {t(`exp.token.${k}` as StringKey)}
                </button>
              </Tip>
            ))}
          </div>
          <div className="exp-example" title={example}>
            <CornerDownRight />
            <span>{example}</span>
          </div>
        </div>
      </section>

      <section className="exp-sec">
        <div className="exp-sec-head">
          <span className="eyebrow">{t('exp.sec_options')}</span>
        </div>
        <div className="exp-row">
          <Tip label={t('exp.loudness_hint')}>
            <span className="exp-label has-hint">{t('exp.loudness')}</span>
          </Tip>
          <Switch checked={params.loudness !== null} onChange={(on) => set({ loudness: on ? LOUDNESS_DEFAULT : null })} label={t('exp.loudness')} />
        </div>
        {params.loudness !== null && (
          <div className="exp-lufs">
            <Slider value={params.loudness} min={-24} max={-6} step={0.5} onChange={(loudness) => set({ loudness })} label={t('exp.target')} />
            <span className="exp-lufs-val tabular">{formatLufs(params.loudness)} LUFS</span>
          </div>
        )}
        <div className="exp-row">
          <Tip label={t('exp.keep_position_hint')}>
            <span className="exp-label has-hint">{t('exp.keep_position')}</span>
          </Tip>
          <Switch checked={params.keepPosition} onChange={(keepPosition) => set({ keepPosition })} label={t('exp.keep_position')} />
        </div>
        <div className="exp-row">
          <span className="exp-label">{t('exp.tags')}</span>
          <Switch checked={params.tags !== null} onChange={(on) => set({ tags: on ? { title: '', artist: '' } : null })} label={t('exp.tags')} />
        </div>
        {params.tags && (
          <div className="exp-tags">
            <input
              className="input exp-input"
              value={params.tags.title}
              placeholder={t('exp.tag_title')}
              onChange={(e) => set({ tags: { ...params.tags!, title: e.target.value } })}
              aria-label={t('exp.tag_title')}
            />
            <input
              className="input exp-input"
              value={params.tags.artist}
              placeholder={t('exp.tag_artist')}
              onChange={(e) => set({ tags: { ...params.tags!, artist: e.target.value } })}
              aria-label={t('exp.tag_artist')}
            />
          </div>
        )}
      </section>
    </div>
  )
}

/** The end of a path is what people recognise: "…/Music/Stellaria". */
function shortPath(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean)
  if (parts.length <= 3) return path
  const sep = path.includes('\\') && !path.includes('/') ? '\\' : '/'
  return `…${sep}${parts.slice(-2).join(sep)}`
}

function formatLufs(v: number): string {
  const text = Number.isInteger(v) ? String(v) : v.toFixed(1)
  return text.replace('-', '−')
}

/** What a template turns into, with sample values; mirrors the backend's filename sanitising. */
function exampleName(template: string, format: ExportParams['format'], t: (k: StringKey) => string): string {
  const d = new Date()
  const values: Record<string, string> = {
    project: t('exp.sample.project'),
    song: t('exp.sample.song'),
    voice: t('exp.sample.voice'),
    label: t('exp.sample.label'),
    date: `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`,
    index: '1',
  }
  const name = template
    .replace(/\{(\w+)\}/g, (m, k: string) => values[k] ?? m)
    .replace(/[\\/:*?"<>|]+/g, '_')
    .replace(/\s+/g, ' ')
    .trim()
  return `${name || values.project}.${format}`
}

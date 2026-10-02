import * as RPopover from '@radix-ui/react-popover'
import { Bookmark, FileDown, FileUp, Plus, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { confirmDialog } from '../../../components/ui'
import { useT } from '../../../i18n'
import { toast } from '../../../state/ui'
import '../fx/fx.css'
import { usePresets as useFxPresets } from '../fx/fxModel'
import { importPresetFile, presetFile, savePresetFile, useMixPresets, type MixPreset } from '../presets'

/** The user's own mix settings: load, save the current ones, delete, export to / import from a file. */
export function MixPresetsMenu({
  current,
  describe,
  onLoad,
}: {
  current: Pick<MixPreset, 'processor' | 'preset' | 'params'>
  /** what a saved preset is for (its processor / base preset), shown under its name */
  describe: (p: MixPreset) => string
  onLoad: (p: MixPreset) => void
}) {
  const t = useT()
  const presets = useMixPresets((s) => s.presets)
  const save = useMixPresets((s) => s.save)
  const remove = useMixPresets((s) => s.remove)
  const [open, setOpen] = useState(false)
  const [name, setName] = useState<string | null>(null)
  const exists = !!name && presets.some((p) => p.name.trim().toLowerCase() === name.trim().toLowerCase())

  const submit = () => {
    const label = name?.trim()
    if (!label) return
    const ok = save({ name: label, ...current })
    setName(null)
    setOpen(false)
    toast(ok ? { kind: 'success', title: t('fx.presets.saved', { name: label }) } : { kind: 'error', title: t('fx.presets.save_failed') })
  }

  return (
    <RPopover.Root
      open={open}
      onOpenChange={(o) => {
        setOpen(o)
        if (!o) setName(null)
      }}
    >
      <RPopover.Trigger asChild>
        <button className="btn btn-sm btn-ghost fx-presets-btn">
          <Bookmark />
          {t('fx.presets.mine')}
        </button>
      </RPopover.Trigger>
      <RPopover.Portal>
        <RPopover.Content className="popover fx-pop fx-presets-pop" align="end" sideOffset={6} collisionPadding={12}>
          <div className="fx-pop-label">{t('fx.presets.mine')}</div>
          {presets.length === 0 ? (
            <div className="fx-pop-empty">{t('mix.presets.none')}</div>
          ) : (
            presets.map((p) => (
              <div key={p.id} className="fx-pop-item has-action">
                <button
                  className="fx-pop-main"
                  onClick={() => {
                    setOpen(false)
                    onLoad(p)
                  }}
                >
                  <span className="fx-pop-name">{p.name}</span>
                  <span className="fx-pop-meta">{describe(p)}</span>
                </button>
                <button
                  className="icon-btn sm fx-pop-del"
                  aria-label={t('common.delete')}
                  onClick={async () => {
                    setOpen(false)
                    const ok = await confirmDialog({ title: t('fx.presets.delete', { name: p.name }), confirm: t('common.delete'), cancel: t('common.cancel'), danger: true })
                    if (ok) remove(p.id)
                  }}
                >
                  <Trash2 />
                </button>
              </div>
            ))
          )}

          <div className="fx-pop-sep" />
          {name === null ? (
            <button className="fx-pop-item fx-pop-save-btn" onClick={() => setName('')}>
              <Plus />
              <span className="fx-pop-name">{t('mix.presets.save')}</span>
            </button>
          ) : (
            <form
              className="fx-pop-save"
              onSubmit={(e) => {
                e.preventDefault()
                submit()
              }}
            >
              <input className="input" autoFocus value={name} maxLength={40} placeholder={t('fx.presets.name')} onChange={(e) => setName(e.target.value)} />
              <button className="btn btn-sm btn-primary" type="submit" disabled={!name.trim()}>
                {exists ? t('fx.presets.overwrite') : t('common.save')}
              </button>
            </form>
          )}
          <PresetFileItems kind="mix" onDone={() => setOpen(false)} />
        </RPopover.Content>
      </RPopover.Portal>
    </RPopover.Root>
  )
}

/** "Export my presets…" / "Import from a file…" at the bottom of a presets popover. */
export function PresetFileItems({ kind, onDone }: { kind: 'mix' | 'fx'; onDone: () => void }) {
  const t = useT()
  const mixCount = useMixPresets((s) => s.presets.length)
  const fxCount = useFxPresets((s) => s.presets.length)
  const has = (kind === 'mix' ? mixCount : fxCount) > 0
  return (
    <>
      <div className="fx-pop-sep" />
      <button
        className="fx-pop-item fx-pop-save-btn"
        disabled={!has}
        onClick={() => {
          onDone()
          void savePresetFile(presetFile(kind), t(kind === 'mix' ? 'presets.file_mix' : 'presets.file_fx'), t)
        }}
      >
        <FileDown />
        <span className="fx-pop-name">{t('presets.export')}</span>
      </button>
      <button
        className="fx-pop-item fx-pop-save-btn"
        onClick={() => {
          onDone()
          void importPresetFile(t)
        }}
      >
        <FileUp />
        <span className="fx-pop-name">{t('presets.import')}</span>
      </button>
    </>
  )
}

import * as RPopover from '@radix-ui/react-popover'
import { Command } from 'cmdk'
import { Bookmark, ChevronRight, Plug, Plus, Search, Settings2, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { EffectSpec, FxSlot, VstPlugin, VstStatus } from '../../../api/types'
import { confirmDialog } from '../../../components/ui'
import { useT } from '../../../i18n'
import { toast } from '../../../state/ui'
import { PresetFileItems } from '../panels/MixPresets'
import { CATEGORY_ORDER, newBuiltinSlot, newVstSlot, STARTERS, starterChain, usePresets, withFreshIds } from './fxModel'

/** "添加效果": built-in effects by category, then installed VST3/AU effects, searchable. */
export function AddEffect({ effects, plugins, vst, onAdd }: { effects: EffectSpec[]; plugins: VstPlugin[]; vst: VstStatus | null | undefined; onAdd: (slot: FxSlot) => void }) {
  const t = useT()
  const nav = useNavigate()
  const [open, setOpen] = useState(false)
  const groups = CATEGORY_ORDER.map((cat) => ({ cat, items: effects.filter((e) => e.category === cat) })).filter((g) => g.items.length > 0)
  const usable = plugins.filter((p) => p.status === 'ok' && p.category === 'effect').sort((a, b) => a.name.localeCompare(b.name))
  const pick = (slot: FxSlot) => {
    setOpen(false)
    onAdd(slot)
  }
  const toSettings = () => {
    setOpen(false)
    nav('/settings#vst')
  }

  return (
    <RPopover.Root open={open} onOpenChange={setOpen}>
      <RPopover.Trigger asChild>
        <button className="fx-add-btn">
          <Plus />
          {t('fx.add')}
        </button>
      </RPopover.Trigger>
      <RPopover.Portal>
        <RPopover.Content className="popover fx-pop fx-add-pop" align="start" side="bottom" sideOffset={6} collisionPadding={12}>
          <Command className="fx-cmd" loop>
            <div className="fx-cmd-search">
              <Search />
              <Command.Input autoFocus placeholder={t('fx.add_search')} />
            </div>
            <Command.List className="fx-cmd-list">
              <Command.Empty className="fx-cmd-empty">{effects.length === 0 && !vst ? t.msg({ code: 'not_found' }) : t('fx.add_none')}</Command.Empty>
              {groups.map((g) => (
                <Command.Group key={g.cat} heading={t.msg({ code: `fx.cat.${g.cat}` })}>
                  {g.items.map((e) => (
                    <Command.Item key={e.id} value={`builtin:${e.id}`} keywords={[t.label(e.name), e.id]} onSelect={() => pick(newBuiltinSlot(e))}>
                      <span className="fx-cmd-name">{t.label(e.name)}</span>
                    </Command.Item>
                  ))}
                </Command.Group>
              ))}
              {vst && (
                <Command.Group heading={t('fx.vst_group')}>
                  {!vst.hostInstalled ? (
                    <Command.Item value="vst:setup" keywords={['vst', 'au', 'plugin', t('fx.vst_setup')]} onSelect={toSettings} className="fx-cmd-quiet">
                      <Plug />
                      <span className="fx-cmd-name">{t('fx.vst_setup')}</span>
                      <ChevronRight className="fx-cmd-end" />
                    </Command.Item>
                  ) : usable.length === 0 ? (
                    <Command.Item value="vst:none" keywords={['vst', 'au', 'plugin']} onSelect={toSettings} className="fx-cmd-quiet">
                      <Plug />
                      <span className="fx-cmd-name">{t('fx.vst_none')}</span>
                      <ChevronRight className="fx-cmd-end" />
                    </Command.Item>
                  ) : (
                    usable.map((p) => (
                      <Command.Item key={p.id} value={`vst:${p.id}`} keywords={[p.name, p.vendor ?? '', p.format]} onSelect={() => pick(newVstSlot(p))}>
                        <span className="fx-cmd-name">{p.name}</span>
                        {p.vendor && <span className="fx-cmd-meta">{p.vendor}</span>}
                        <span className="fx-fmt">{p.format === 'au' ? 'AU' : 'VST3'}</span>
                      </Command.Item>
                    ))
                  )}
                </Command.Group>
              )}
            </Command.List>
            {vst?.hostInstalled && (
              <button className="fx-cmd-foot" onClick={toSettings}>
                <Settings2 />
                {t('fx.vst_manage')}
              </button>
            )}
          </Command>
        </RPopover.Content>
      </RPopover.Portal>
    </RPopover.Root>
  )
}

/** Starter chains and the user's own (saved in this browser profile / app data). */
export function PresetsMenu({ chain, specs, onLoad }: { chain: FxSlot[]; specs: Map<string, EffectSpec>; onLoad: (chain: FxSlot[], name: string) => void }) {
  const t = useT()
  const presets = usePresets((s) => s.presets)
  const save = usePresets((s) => s.save)
  const remove = usePresets((s) => s.remove)
  const [open, setOpen] = useState(false)
  const [name, setName] = useState<string | null>(null)
  const effectName = (id: string) => {
    const spec = specs.get(id)
    return spec ? t.label(spec.name) : t.msg({ code: `fx.effect.${id}` }, id)
  }
  const load = (next: FxSlot[], label: string) => {
    setOpen(false)
    onLoad(next, label)
  }
  const submit = () => {
    const label = name?.trim()
    if (!label || chain.length === 0) return
    const ok = save(label, chain)
    setName(null)
    setOpen(false)
    toast(ok ? { kind: 'success', title: t('fx.presets.saved', { name: label }) } : { kind: 'error', title: t('fx.presets.save_failed') })
  }
  const exists = !!name && presets.some((p) => p.name.trim().toLowerCase() === name.trim().toLowerCase())

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
          {t('fx.presets')}
        </button>
      </RPopover.Trigger>
      <RPopover.Portal>
        <RPopover.Content className="popover fx-pop fx-presets-pop" align="end" sideOffset={6} collisionPadding={12}>
          <div className="fx-pop-label">{t('fx.presets.starters')}</div>
          {STARTERS.map((s) => {
            if (!s.slots.some((x) => specs.has(x.effect))) return null
            return (
              <button key={s.id} className="fx-pop-item" onClick={() => load(starterChain(s, specs), t(s.name))}>
                <span className="fx-pop-name">{t(s.name)}</span>
                <span className="fx-pop-meta">{s.slots.filter((x) => specs.has(x.effect)).map((x) => effectName(x.effect)).join(' · ')}</span>
              </button>
            )
          })}

          <div className="fx-pop-label">{t('fx.presets.mine')}</div>
          {presets.length === 0 ? (
            <div className="fx-pop-empty">{t('fx.presets.none')}</div>
          ) : (
            presets.map((p) => (
              <div key={p.id} className="fx-pop-item has-action">
                <button className="fx-pop-main" onClick={() => load(withFreshIds(p.chain), p.name)}>
                  <span className="fx-pop-name">{p.name}</span>
                  <span className="fx-pop-meta">{t('fx.presets.count', { n: p.chain.length })}</span>
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
            <button className="fx-pop-item fx-pop-save-btn" disabled={chain.length === 0} onClick={() => setName('')}>
              <Plus />
              <span className="fx-pop-name">{t('fx.presets.save')}</span>
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
          <PresetFileItems kind="fx" onDone={() => setOpen(false)} />
        </RPopover.Content>
      </RPopover.Portal>
    </RPopover.Root>
  )
}

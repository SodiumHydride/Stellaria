import clsx from 'clsx'
import { ChevronRight, FolderOpen, RotateCcw } from 'lucide-react'
import { useState } from 'react'
import type { ParamSpec } from '../api/types'
import { useT } from '../i18n'
import { pickFiles, pickFolder } from '../shell/bridge'
import { Segmented, Slider, Switch } from './ui'

type Value = number | string | boolean | null | undefined

/**
 * Controls generated from a processor's or plugin's parameter specs. Params with a `group` are
 * fine-grained settings: they sit folded under "advanced", by group. `defaults` are the values
 * that apply while a param is unset (a preset's), so the controls show what will be used.
 */
export function ParamControls({
  specs,
  values,
  onChange,
  defaults,
  groupLabel,
  onReset,
}: {
  specs: ParamSpec[]
  values: Record<string, unknown>
  onChange: (id: string, v: number | string | boolean | null) => void
  defaults?: Record<string, unknown>
  groupLabel?: (group: string) => string
  /** clears the given params (back to `defaults`) */
  onReset?: (ids: string[]) => void
}) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const value = (p: ParamSpec) => (values[p.id] as Value) ?? (defaults?.[p.id] as Value) ?? p.default
  const main = specs.filter((p) => !p.group)
  const fine = specs.filter((p) => p.group)
  const groups = [...new Set(fine.map((p) => p.group as string))]
  const changed = fine.filter((p) => values[p.id] !== undefined && values[p.id] !== null).map((p) => p.id)
  return (
    <div className="param-controls">
      {main.map((p) => (
        <ParamControl key={p.id} spec={p} value={value(p)} onChange={(v) => onChange(p.id, v)} />
      ))}
      {fine.length > 0 && (
        <div className={clsx('param-advanced', open && 'open')}>
          <div className="param-advanced-head">
            <button className="param-advanced-toggle" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
              <ChevronRight />
              {t('param.advanced')}
              {changed.length > 0 && <span className="chip accent">{t('param.changed', { n: changed.length })}</span>}
            </button>
            {open && changed.length > 0 && onReset && (
              <button className="btn btn-sm btn-ghost" onClick={() => onReset(changed)}>
                <RotateCcw />
                {t('param.reset')}
              </button>
            )}
          </div>
          {open &&
            groups.map((g) => (
              <div key={g} className="param-group">
                <span className="eyebrow">{groupLabel ? groupLabel(g) : g}</span>
                {fine
                  .filter((p) => p.group === g)
                  .map((p) => (
                    <ParamControl key={p.id} spec={p} value={value(p)} changed={changed.includes(p.id)} onChange={(v) => onChange(p.id, v)} />
                  ))}
              </div>
            ))}
        </div>
      )}
    </div>
  )
}

function fmt(v: number, step?: number) {
  if (step && step >= 1) return String(Math.round(v))
  const decimals = step ? Math.min(3, Math.max(0, -Math.floor(Math.log10(step)))) : 2
  return v.toFixed(decimals)
}

export function ParamControl({ spec, value, onChange, changed }: { spec: ParamSpec; value: Value; onChange: (v: number | string | boolean | null) => void; changed?: boolean }) {
  const t = useT()
  const label = t.label(spec.label)
  if (spec.type === 'number') {
    const v = typeof value === 'number' ? value : Number(spec.default ?? spec.min ?? 0)
    const centered = (spec.min ?? 0) < 0 && (spec.max ?? 0) > 0
    return (
      <div className={clsx('field param', changed && 'changed')}>
        <div className="row">
          <span className="field-label">{label}</span>
          <span className="spacer" />
          <span className="tabular muted">
            {centered && v > 0 ? '+' : ''}
            {fmt(v, spec.step)}
            {spec.unit ? ` ${spec.unit}` : ''}
          </span>
        </div>
        <Slider value={v} min={spec.min ?? 0} max={spec.max ?? 1} step={spec.step ?? 0.01} centerMark={centered} onChange={onChange} label={label} />
      </div>
    )
  }
  if (spec.type === 'boolean') {
    return (
      <label className="row param param-bool">
        <span className="field-label" style={{ flex: 1 }}>
          {label}
        </span>
        <Switch checked={!!value} onChange={onChange} label={label} />
      </label>
    )
  }
  if (spec.type === 'choice') {
    const options = (spec.options ?? []).map((o) => ({ value: o.value, label: t.label(o.label) }))
    return (
      <div className="field param">
        <span className="field-label">{label}</span>
        {options.length <= 4 ? (
          <Segmented<string> size="sm" value={String(value ?? '')} onChange={onChange} options={options} />
        ) : (
          <select className="input" value={String(value ?? '')} onChange={(e) => onChange(e.target.value)}>
            {options.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        )}
      </div>
    )
  }
  if (spec.type === 'folder' || spec.type === 'file') {
    const browse = async () => {
      const picked = spec.type === 'folder' ? await pickFolder() : await pickFiles({ multiple: false, extensions: ['*'] })
      const path = Array.isArray(picked) ? picked[0] : picked
      if (typeof path === 'string') onChange(path)
    }
    return (
      <div className="field param">
        <span className="field-label">{label}</span>
        <div className="row" style={{ gap: 6 }}>
          <input className="input mono" value={String(value ?? '')} onChange={(e) => onChange(e.target.value || null)} />
          <button className="btn btn-sm" onClick={browse} aria-label={label}>
            <FolderOpen />
          </button>
        </div>
      </div>
    )
  }
  return (
    <label className="field param">
      <span className="field-label">{label}</span>
      <input className="input" value={String(value ?? '')} onChange={(e) => onChange(e.target.value)} />
    </label>
  )
}

import clsx from 'clsx'
import { Table2 } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import uPlot from 'uplot'
import type { ScalarSeries } from '../api/types'
import { useT } from '../i18n'
import { cssVar } from '../lib/color'
import { steps as fmtSteps } from '../lib/format'
import { useLatest } from '../lib/useEvent'
import { useTheme } from '../state/theme'
import { Tip } from './ui'
import './metric-chart.css'

export interface SeriesSpec {
  tag: string
  label: string
  /** CSS variable name, e.g. '--series-1' (resolved at draw time; text never uses it) */
  colorVar: string
}

function withAlpha(hex: string, a: number): string {
  const h = hex.replace('#', '')
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16)
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`
}

/** TensorBoard-style debiased exponential moving average. */
function ema(values: (number | null)[], weight: number): (number | null)[] {
  if (weight <= 0) return values
  let last = 0
  let n = 0
  return values.map((v) => {
    if (v == null || !isFinite(v)) return null
    last = last * weight + (1 - weight) * v
    n++
    return last / (1 - weight ** n)
  })
}

function quantile(sorted: number[], q: number) {
  if (sorted.length === 0) return 0
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))
  return sorted[i]
}

function payloadOf(a: { xs: number[]; raw: (number | null)[][]; smooth: (number | null)[][] }): uPlot.AlignedData {
  return [a.xs, ...a.raw.flatMap((r, i) => [r, a.smooth[i]])] as uPlot.AlignedData
}

function fmtValue(v: number | null | undefined): string {
  if (v == null || !isFinite(v)) return '–'
  const a = Math.abs(v)
  if (a !== 0 && (a < 0.001 || a >= 100000)) return v.toExponential(2)
  if (a < 1) return v.toFixed(4)
  if (a < 100) return v.toFixed(3)
  return v.toFixed(1)
}

/**
 * Line chart for training scalars: raw values faint, smoothed values bold, one y-axis,
 * crosshair readout in the legend, outlier-robust y-range, and a table view.
 */
export function MetricChart({
  title,
  hint,
  series,
  data,
  smoothing,
  height = 190,
  emptyText,
  unit,
  headerExtra,
}: {
  title: string
  hint?: string
  series: SeriesSpec[]
  data: Record<string, ScalarSeries['points']>
  smoothing: number
  height?: number
  emptyText: string
  unit?: string
  headerExtra?: React.ReactNode
}) {
  const t = useT()
  const wrapRef = useRef<HTMLDivElement>(null)
  const plotRef = useRef<uPlot | null>(null)
  const [cursorIdx, setCursorIdx] = useState<number | null>(null)
  const [showTable, setShowTable] = useState(false)
  const theme = useTheme((s) => s.resolved)

  const present = useMemo(() => series.filter((s) => (data[s.tag]?.length ?? 0) > 0), [series, data])

  const aligned = useMemo(() => {
    const stepSet = new Set<number>()
    for (const s of present) for (const p of data[s.tag]) stepSet.add(p[0])
    const xs = Array.from(stepSet).sort((a, b) => a - b)
    const index = new Map(xs.map((x, i) => [x, i]))
    const raw = present.map((s) => {
      const arr: (number | null)[] = new Array(xs.length).fill(null)
      for (const p of data[s.tag]) arr[index.get(p[0])!] = p[1]
      return arr
    })
    const smooth = raw.map((r) => ema(r, smoothing))
    return { xs, raw, smooth }
  }, [data, smoothing, present])

  // the plot outlives individual data updates: it reads the latest values through this
  const latest = useLatest({ aligned, present, height })
  const layoutKey = `${present.map((s) => s.tag).join()}|${height}|${showTable}|${theme}`
  const hasData = aligned.xs.length >= 2

  // build the plot once per series layout…
  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap || !hasData || !layoutKey) return
    const { aligned, present, height } = latest.current
    const colors = present.map((s) => cssVar(s.colorVar))
    const label = cssVar('--chart-label')
    const grid = cssVar('--chart-grid')
    const axis = cssVar('--chart-axis')

    const opts: uPlot.Options = {
      width: wrap.clientWidth,
      height,
      padding: [8, 8, 0, 0],
      legend: { show: false },
      cursor: {
        points: { size: 8, width: 2 },
        drag: { x: false, y: false },
        y: false,
      },
      scales: {
        x: { time: false },
        y: {
          range: (_u, min, max) => {
            const vals = latest.current.aligned.smooth.flat().filter((v): v is number => v != null && isFinite(v)).sort((a, b) => a - b)
            if (vals.length < 10) return uPlot.rangeNum(min, max, 0.1, true)
            const lo = quantile(vals, 0.02)
            const hi = quantile(vals, 0.98)
            const pad = (hi - lo) * 0.12 || Math.abs(hi) * 0.1 || 1
            return [lo - pad, hi + pad]
          },
        },
      },
      axes: [
        {
          stroke: label,
          grid: { show: false },
          ticks: { stroke: axis, width: 1, size: 4 },
          font: '11px system-ui, sans-serif',
          values: (_u, splits) => splits.map((v) => fmtSteps(Math.round(v))),
        },
        {
          stroke: label,
          grid: { stroke: grid, width: 1 },
          ticks: { show: false },
          font: '11px system-ui, sans-serif',
          size: 54,
          values: (_u, splits) => splits.map((v) => fmtValue(v)),
        },
      ],
      series: [
        {},
        ...present.flatMap((s, i) => [
          { label: `${s.label} raw`, stroke: withAlpha(colors[i], 0.22), width: 1, points: { show: false }, spanGaps: true },
          { label: s.label, stroke: colors[i], width: 2, points: { show: false }, spanGaps: true },
        ]),
      ],
      hooks: {
        setCursor: [(u) => setCursorIdx(u.cursor.idx ?? null)],
      },
    }
    plotRef.current = new uPlot(opts, payloadOf(aligned), wrap)
    const ro = new ResizeObserver(() => plotRef.current?.setSize({ width: wrap.clientWidth, height }))
    ro.observe(wrap)
    return () => {
      ro.disconnect()
      plotRef.current?.destroy()
      plotRef.current = null
    }
  }, [layoutKey, hasData, latest])

  // …and stream new points into it
  useEffect(() => {
    plotRef.current?.setData(payloadOf(aligned))
  }, [aligned])

  const idx = cursorIdx ?? aligned.xs.length - 1
  const readStep = aligned.xs[idx]

  return (
    <div className="mchart glass">
      <div className="mchart-head">
        {hint ? (
          <Tip label={hint}>
            <span className="mchart-title has-hint">{title}</span>
          </Tip>
        ) : (
          <span className="mchart-title">{title}</span>
        )}
        <span className="spacer" />
        {headerExtra}
        <Tip label={t('chart.table')}>
          <button className={clsx('icon-btn sm', showTable && 'active')} onClick={() => setShowTable(!showTable)} aria-pressed={showTable}>
            <Table2 />
          </button>
        </Tip>
      </div>
      {present.length > 0 && aligned.xs.length > 0 && (
        <div className="mchart-legend">
          {readStep != null && <span className="mchart-step tabular">{t('val.step', { step: readStep.toLocaleString() })}</span>}
          {present.map((s, i) => (
            <span key={s.tag} className="mchart-key">
              <i style={{ background: `var(${s.colorVar})` }} />
              <span className="secondary">{s.label}</span>
              <span className="tabular">
                {fmtValue(aligned.smooth[i][idx])}
                {unit ? ` ${unit}` : ''}
              </span>
            </span>
          ))}
        </div>
      )}
      {aligned.xs.length < 2 ? (
        <div className="mchart-empty" style={{ height }}>
          {emptyText}
        </div>
      ) : showTable ? (
        <div className="mchart-table" style={{ height }}>
          <table>
            <thead>
              <tr>
                <th>{t('common.steps')}</th>
                {present.map((s) => (
                  <th key={s.tag}>{s.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {aligned.xs
                .map((x, i) => ({ x, i }))
                .filter((_, k, arr) => arr.length <= 60 || k % Math.ceil(arr.length / 60) === 0 || k === arr.length - 1)
                .reverse()
                .map(({ x, i }) => (
                  <tr key={x}>
                    <td className="tabular">{x.toLocaleString()}</td>
                    {present.map((s, si) => (
                      <td key={s.tag} className="tabular">
                        {fmtValue(aligned.raw[si][i])}
                      </td>
                    ))}
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div ref={wrapRef} className="mchart-plot" style={{ height }} onMouseLeave={() => setCursorIdx(null)} />
      )}
    </div>
  )
}

import { Download, Play, X } from 'lucide-react'
import type { CanvasNode } from '../../api/types'
import { Logo } from '../../components/Logo'
import { Progress, Tip } from '../../components/ui'
import { useT } from '../../i18n'
import { platform } from '../../shell/bridge'
import { isProcess } from './model'
import { cancelRun, pendingWork, runnable, useRunner } from './runner'

/** Connect everything first, then run it in one go — or run just what the selection needs. */
export function RunButton({ projectId, nodes, selection, onRun }: { projectId: string; nodes: CanvasNode[]; selection: string[]; onRun: (targets: string[] | null) => void }) {
  const t = useT()
  const state = useRunner((s) => s.runs[projectId])
  const targets = selection.length ? selection : null
  const pending = pendingWork(nodes, targets)
  const mod = platform === 'macos' ? '⌘' : 'Ctrl'

  if (state?.running) {
    return (
      <div className="run-pill glass-strong running">
        <Logo size={16} running />
        <span className="tabular">{t('run.progress', { done: state.done + state.failed + state.skipped, total: state.total })}</span>
        <div className="run-progress">
          <Progress value={state.total ? (state.done + state.failed + state.skipped) / state.total : null} />
        </div>
        <Tip label={t('run.cancel')}>
          <button className="icon-btn sm" onClick={() => cancelRun(projectId)}>
            <X />
          </button>
        </Tip>
      </div>
    )
  }
  const exports = nodes.filter((n) => isProcess(n) && n.data.op === 'export' && runnable(n)).map((n) => n.id)
  return (
    <div className="run-group">
      {exports.length > 0 && (
        <Tip label={t('run.export_all_hint')}>
          <button className="run-pill glass-strong secondary" onClick={() => onRun(exports)}>
            <Download />
            <span>{t('run.export_all')}</span>
            <span className="run-count tabular">{exports.length}</span>
          </button>
        </Tip>
      )}
      <Tip label={`${targets ? t('run.selection_hint') : t('run.all_hint')} (${mod} ⏎)`}>
        <button className="run-pill glass-strong" onClick={() => onRun(targets)} disabled={pending === 0}>
          <Play />
          <span>{targets ? t('run.selection') : t('run.all')}</span>
          {pending > 0 && <span className="run-count tabular">{pending}</span>}
        </button>
      </Tip>
    </div>
  )
}

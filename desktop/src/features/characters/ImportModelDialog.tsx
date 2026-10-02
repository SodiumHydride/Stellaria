import { useQueryClient } from '@tanstack/react-query'
import { FileBox, Loader2 } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../api/client'
import { useCharacters } from '../../api/hooks'
import type { Job } from '../../api/types'
import { Dialog } from '../../components/ui'
import { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { pickVoiceColor, VOICE_COLORS } from '../../lib/color'
import { waitForJob } from '../../lib/jobs'
import { isTauri, pickFiles } from '../../shell/bridge'
import { toast } from '../../state/ui'

/**
 * A DDSP-SVC model trained elsewhere (.pt files with their config.yaml) becomes a voice that can
 * sing right away — no material needed. Every .pt picked becomes a version to choose from.
 */
export function ImportModelDialog({ onClose }: { onClose: () => void }) {
  const t = useT()
  const nav = useNavigate()
  const qc = useQueryClient()
  const { data: characters } = useCharacters()
  const [paths, setPaths] = useState<string[]>([])
  const [manual, setManual] = useState('')
  const [name, setName] = useState('')
  const [color, setColor] = useState(() => pickVoiceColor(String(Date.now()), (characters ?? []).map((c) => c.color)))
  const [busy, setBusy] = useState(false)

  const pick = async () => {
    const picked = await pickFiles({ multiple: true, extensions: ['pt'] })
    if (!Array.isArray(picked)) return
    const list = picked.filter((p): p is string => typeof p === 'string')
    setPaths(list)
    if (!name && list[0]) {
      const parts = list[0].split(/[\\/]/)
      setName(parts[parts.length - 2] ?? '')
    }
  }

  const ckptPaths = isTauri ? paths : manual.split('\n').map((x) => x.trim()).filter(Boolean)

  const go = async () => {
    setBusy(true)
    const job = await attempt(() => api.post<Job>('/models/import-external', { ckptPaths, newCharacter: { name: name.trim() || 'Voice', color } }))
    if (job) {
      const done = await attempt(() => waitForJob(job.id))
      if (done?.status === 'succeeded') {
        const r = done.result as { characterId?: string } | null
        qc.invalidateQueries({ queryKey: ['characters'] })
        qc.invalidateQueries({ queryKey: ['models'] })
        onClose()
        if (r?.characterId) nav(`/voices/${r.characterId}/models`)
      } else if (done) {
        toast({ kind: 'error', title: t.msg(done.error ? { code: done.error.code, params: done.error.params } : null, t('common.error')), detail: done.error?.message })
      }
    }
    setBusy(false)
  }

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && !busy && onClose()}
      title={t('imp.title')}
      description={t('imp.body')}
      actions={
        <>
          <button className="btn btn-ghost" onClick={onClose} disabled={busy}>
            {t('common.cancel')}
          </button>
          <button className="btn btn-primary" onClick={go} disabled={busy || ckptPaths.length === 0}>
            {busy ? <Loader2 className="spin" /> : <FileBox />}
            {t('imp.go')}
          </button>
        </>
      }
    >
      <div className="col" style={{ gap: 14 }}>
        {isTauri ? (
          <div className="col" style={{ gap: 6 }}>
            <button className="btn" onClick={pick}>
              <FileBox />
              {paths.length ? t('imp.picked', { n: paths.length }) : t('imp.pick')}
            </button>
            {paths.length > 0 && <div className="mono muted imp-paths">{paths.map((p) => p.split(/[\\/]/).pop()).join(' · ')}</div>}
          </div>
        ) : (
          <textarea className="input mono" rows={3} placeholder="/path/to/model_100000.pt" value={manual} onChange={(e) => setManual(e.target.value)} />
        )}
        <label className="field">
          <span className="field-label">{t('imp.name')}</span>
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <div className="field">
          <span className="field-label">{t('ws.color')}</span>
          <div className="swatches">
            {VOICE_COLORS.map((col) => (
              <button key={col} className={`swatch${col === color ? ' on' : ''}`} style={{ background: col }} onClick={() => setColor(col)} aria-label={col} />
            ))}
          </div>
        </div>
      </div>
    </Dialog>
  )
}

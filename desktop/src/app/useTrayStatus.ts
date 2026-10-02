import { useEffect } from 'react'
import { useActiveJobs } from '../api/hooks'
import { useT } from '../i18n'
import { percent } from '../lib/format'
import { isTauri, shell } from '../shell/bridge'

/** Mirrors the running training into the tray tooltip and the taskbar/dock progress. */
export function useTrayStatus() {
  const t = useT()
  const train = useActiveJobs((j) => j.type === 'train' && j.status === 'running')[0]
  const text = train ? `${t.msg(train.title, 'train')} ${percent(train.progress)}` : null
  const progress = train?.progress ?? null
  const rounded = progress == null ? null : Math.round(progress * 100) / 100
  useEffect(() => {
    if (!isTauri) return
    void shell.setTrayStatus(text, rounded).catch(() => {})
  }, [text, rounded])
}

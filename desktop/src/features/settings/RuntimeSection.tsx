import { Cpu, RotateCcw, Terminal } from 'lucide-react'
import { useState } from 'react'
import { useSystem } from '../../api/hooks'
import { useBoot } from '../../app/boot'
import { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { isTauri, shell } from '../../shell/bridge'
import { ExistingPython } from './ExistingPython'
import { Group, Row } from './parts'

/** The Python the app runs on: the managed runtime, or one the user already has (e.g. their DDSP-SVC env). */
export function RuntimeSection() {
  const t = useT()
  const { data: system } = useSystem()
  const shellState = useBoot((s) => s.shell)
  const setShell = useBoot((s) => s.setShell)
  const [picking, setPicking] = useState(false)
  const [switching, setSwitching] = useState(false)
  const runtime = shellState?.runtime
  const custom = runtime?.mode === 'custom'
  const torch = system?.torch

  const use = async (path: string | null) => {
    setSwitching(true)
    const next = await attempt(() => shell.setRuntimePython(path))
    if (next) {
      setShell(next)
      setPicking(false)
    }
    setSwitching(false)
  }

  const current = custom ? runtime?.pythonPath : t('rt.managed')
  const summary = torch ? `Python ${runtime?.python ?? ''} · torch ${torch.version}${torch.cuda ? ` · CUDA ${torch.cuda}` : ''}` : (runtime?.python ?? '')

  return (
    <Group icon={<Terminal />} title={t('rt.title')}>
      <Row
        label={custom ? t('rt.custom') : t('rt.managed')}
        hint={
          <>
            <span className="mono">{custom ? current : summary}</span>
            {custom && runtime?.layered && <span className="rt-layered">{t('rt.layered_note')}</span>}
          </>
        }
      >
        {isTauri && (
          <div className="row" style={{ gap: 8 }}>
            {custom && (
              <button className="btn btn-sm btn-ghost" onClick={() => use(null)} disabled={switching}>
                <RotateCcw />
                {t('rt.back_to_managed')}
              </button>
            )}
            <button className="btn btn-sm" onClick={() => setPicking((p) => !p)} disabled={switching}>
              <Cpu />
              {t('rt.choose')}
            </button>
          </div>
        )}
      </Row>
      {picking && (
        <div className="rt-picker">
          <ExistingPython onUse={(p) => void use(p)} onManaged={custom ? () => void use(null) : () => setPicking(false)} busy={switching} />
        </div>
      )}
    </Group>
  )
}

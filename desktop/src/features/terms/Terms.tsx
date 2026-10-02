import { useState } from 'react'
import { useBoot } from '../../app/boot'
import { Dialog } from '../../components/ui'
import { useLang, useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { isTauri, shell } from '../../shell/bridge'
import { acceptedVersion, acceptTerms, TERMS_DATE, TERMS_VERSION, termsSections } from './termsOfUse'
import './terms.css'

/** The full text, for a dialog or a page. */
export function TermsText() {
  const t = useT()
  const lang = useLang((s) => s.lang)
  return (
    <div className="terms-text">
      <p className="muted terms-meta">{t('terms.meta', { version: TERMS_VERSION, date: TERMS_DATE })}</p>
      {termsSections(lang).map((s) => (
        <section key={s.title}>
          <h3>{s.title}</h3>
          {s.body.map((p) => (
            <p key={p.slice(0, 24)}>{p}</p>
          ))}
        </section>
      ))}
    </div>
  )
}

/** Reading the terms (from the welcome step or Settings). */
export function TermsDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const t = useT()
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      wide
      title={t('terms.title')}
      actions={
        <button className="btn btn-primary" onClick={() => onOpenChange(false)} autoFocus>
          {t('common.close')}
        </button>
      }
    >
      <TermsText />
    </Dialog>
  )
}

/** Asks once for a version of the terms not accepted yet (users from before it existed, or after an update). */
export function TermsGate() {
  const t = useT()
  const shellState = useBoot((s) => s.shell)
  const setShell = useBoot((s) => s.setShell)
  const [accepted, setAccepted] = useState(() => acceptedVersion(shellState) >= TERMS_VERSION)
  if (accepted || acceptedVersion(shellState) >= TERMS_VERSION) return null
  const agree = () =>
    attempt(async () => {
      const next = await acceptTerms()
      if (next) setShell(next)
      setAccepted(true)
    })
  return (
    <Dialog
      open
      onOpenChange={() => {}}
      dismissable={false}
      wide
      title={t('terms.updated_title')}
      description={t('terms.updated_body')}
      actions={
        <>
          {isTauri && (
            <button className="btn btn-ghost" onClick={() => void shell.quitApp(false)}>
              {t('terms.decline')}
            </button>
          )}
          <button className="btn btn-primary" onClick={() => void agree()} autoFocus>
            {t('terms.accept')}
          </button>
        </>
      }
    >
      <TermsText />
    </Dialog>
  )
}

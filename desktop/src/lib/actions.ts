import { ApiError } from '../api/client'
import { translate, translateMsg, useLang } from '../i18n'
import { toast } from '../state/ui'

/** Localized message for any thrown error. */
export function errorText(e: unknown): { title: string; detail?: string } {
  const lang = useLang.getState().lang
  if (e instanceof ApiError) {
    const localized = translateMsg(lang, { code: e.code, params: e.params }, '')
    return { title: localized || e.message || translate(lang, 'common.error'), detail: e.detail || undefined }
  }
  // the desktop shell rejects with a plain {code, message, detail}; detail may itself be a {code, params} message
  if (e && typeof e === 'object' && typeof (e as { code?: unknown }).code === 'string') {
    const { code, message, detail } = e as { code: string; message?: string; detail?: string }
    let inner: string | undefined = detail || undefined
    if (detail?.startsWith('{')) {
      try {
        const m = JSON.parse(detail) as { code?: string; params?: Record<string, string | number> }
        if (m.code) inner = translateMsg(lang, { code: m.code, params: m.params }, detail)
      } catch {
        /* not JSON: show as is */
      }
    }
    return { title: translateMsg(lang, { code }, '') || message || translate(lang, 'common.error'), detail: inner }
  }
  return { title: translate(lang, 'common.error'), detail: e instanceof Error ? e.message : String(e) }
}

/** Run an API action; failures become an error toast (with copyable details). */
export async function attempt<T>(fn: () => Promise<T>, opts?: { success?: string }): Promise<T | undefined> {
  try {
    const res = await fn()
    if (opts?.success) toast({ kind: 'success', title: opts.success })
    return res
  } catch (e) {
    const { title, detail } = errorText(e)
    toast({ kind: 'error', title, detail })
    return undefined
  }
}

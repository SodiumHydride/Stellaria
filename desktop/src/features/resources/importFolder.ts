import { api } from '../../api/client'
import type { Resource, SeparationImportResult } from '../../api/types'
import { translate, translateMsg, useLang } from '../../i18n'
import { attempt } from '../../lib/actions'
import { toast } from '../../state/ui'

/**
 * Import every missing model found in a folder someone already has (their DDSP-SVC checkout or
 * bundle) and say what was found. Returns whether the request went through.
 */
export async function importModelsFromFolder(path: string, resources: Resource[]): Promise<boolean> {
  const res = await attempt(() => api.post<{ found: { id: string }[]; missing: string[]; separation?: SeparationImportResult }>('/resources/import-folder', { path }))
  if (!res) return false
  const lang = useLang.getState().lang
  const separated = res.separation?.found.length ?? 0
  if (separated) toast({ kind: 'success', title: translate(lang, 'sep.import_found', { n: separated }), body: res.separation!.found.map((f) => f.name).join(lang === 'zh-CN' ? '、' : ', ') })
  const nameOf = (id: string) => translateMsg(lang, resources.find((r) => r.id === id)?.name ?? null, id)
  const missingRequired = res.missing.filter((id) => resources.some((r) => r.id === id && r.required))
  if (res.found.length === 0) {
    toast({ kind: 'info', title: translate(lang, 'ob.models.none_found'), body: translate(lang, 'ob.models.none_found_body') })
  } else {
    toast({
      kind: missingRequired.length ? 'info' : 'success',
      title: translate(lang, 'ob.models.found', { n: res.found.length }),
      body: missingRequired.length ? translate(lang, 'ob.models.still_missing', { names: missingRequired.map(nameOf).join(lang === 'zh-CN' ? '、' : ', ') }) : undefined,
    })
  }
  return true
}

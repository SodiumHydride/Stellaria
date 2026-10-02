import { create } from 'zustand'

/** Where the backend lives. Set once the shell reports it ready (or from browser params). */
interface ConnectionState {
  url: string | null
  token: string | null
  setBackend: (url: string, token: string) => void
  clear: () => void
}

export const useConnection = create<ConnectionState>((set) => ({
  url: null,
  token: null,
  setBackend: (url, token) => set({ url: url.replace(/\/$/, ''), token }),
  clear: () => set({ url: null, token: null }),
}))

export class ApiError extends Error {
  code: string
  detail: string
  status: number
  params: Record<string, string | number>

  constructor(status: number, code: string, message: string, detail = '', params: Record<string, string | number> = {}) {
    super(message)
    this.status = status
    this.code = code
    this.detail = detail
    this.params = params
  }
}

type Query = Record<string, string | number | boolean | null | undefined>

function base(): { url: string; token: string } {
  const { url, token } = useConnection.getState()
  if (!url || !token) throw new ApiError(0, 'backend.not_connected', 'Backend not connected')
  return { url, token }
}

function buildUrl(path: string, query?: Query): string {
  const { url } = base()
  const u = new URL(url + '/api' + path)
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined && v !== null) u.searchParams.set(k, String(v))
    }
  }
  return u.toString()
}

async function request<T>(method: string, path: string, body?: unknown, query?: Query): Promise<T> {
  const { token } = base()
  const init: RequestInit = { method, headers: { Authorization: `Bearer ${token}` } }
  if (body instanceof FormData) {
    init.body = body
  } else if (body !== undefined) {
    init.body = JSON.stringify(body)
    ;(init.headers as Record<string, string>)['Content-Type'] = 'application/json'
  }
  let res: Response
  try {
    res = await fetch(buildUrl(path, query), init)
  } catch (e) {
    throw new ApiError(0, 'backend.unreachable', 'Backend unreachable', String(e))
  }
  if (!res.ok) {
    let code = `http.${res.status}`
    let message = res.statusText
    let detail = ''
    let params: Record<string, string | number> = {}
    try {
      const data = await res.json()
      if (data?.error) {
        code = data.error.code ?? code
        message = data.error.message ?? message
        detail = typeof data.error.detail === 'string' ? data.error.detail : data.error.detail ? JSON.stringify(data.error.detail) : ''
        params = data.error.params ?? {}
      } else if (data?.detail) {
        detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, code, message, detail, params)
  }
  if (res.status === 204) return undefined as T
  const type = res.headers.get('content-type') ?? ''
  return (type.includes('application/json') ? res.json() : res.text()) as Promise<T>
}

export const api = {
  get: <T>(path: string, query?: Query) => request<T>('GET', path, undefined, query),
  post: <T>(path: string, body?: unknown, query?: Query) => request<T>('POST', path, body ?? {}, query),
  put: <T>(path: string, body?: unknown) => request<T>('PUT', path, body ?? {}),
  patch: <T>(path: string, body?: unknown) => request<T>('PATCH', path, body ?? {}),
  del: <T>(path: string, query?: Query) => request<T>('DELETE', path, undefined, query),
  upload: <T>(path: string, files: File[], field = 'files', extra?: Record<string, string>) => {
    const form = new FormData()
    for (const f of files) form.append(field, f, f.name)
    if (extra) for (const [k, v] of Object.entries(extra)) form.append(k, v)
    return request<T>('POST', path, form)
  },
}

/** URL usable in <audio>/<img> (token in query, since those cannot send headers). */
export function mediaUrl(path: string, query?: Query): string {
  const { token } = base()
  return buildUrl(path.startsWith('/api') ? path.slice(4) : path, { ...query, token })
}

export function eventsUrl(): string {
  const { url, token } = base()
  return url.replace(/^http/, 'ws') + '/api/events?token=' + encodeURIComponent(token)
}

/** Import paths (desktop) or upload files (browser) to an endpoint that accepts both. */
export function sendFiles<T>(path: string, items: string[] | File[], extra?: Record<string, unknown>): Promise<T> {
  if (items.length > 0 && typeof items[0] === 'string') return api.post<T>(path, { paths: items, ...extra })
  const flat: Record<string, string> = {}
  if (extra) for (const [k, v] of Object.entries(extra)) flat[k] = typeof v === 'string' ? v : JSON.stringify(v)
  return api.upload<T>(path, items as File[], 'files', flat)
}

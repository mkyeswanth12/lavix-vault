import createDOMPurify from 'dompurify'
import hljs from 'highlight.js/lib/core'
import bash from 'highlight.js/lib/languages/bash'
import css from 'highlight.js/lib/languages/css'
import javascript from 'highlight.js/lib/languages/javascript'
import json from 'highlight.js/lib/languages/json'
import markdown from 'highlight.js/lib/languages/markdown'
import python from 'highlight.js/lib/languages/python'
import sql from 'highlight.js/lib/languages/sql'
import typescript from 'highlight.js/lib/languages/typescript'
import xml from 'highlight.js/lib/languages/xml'
import { marked } from 'marked'

for (const [name, language] of Object.entries({ bash, css, javascript, json, markdown, python, sql, typescript, xml })) {
  hljs.registerLanguage(name, language)
}

const API_URL = window.location.origin + '/api'
const _markdownPurifier = createDOMPurify(window)
const _markdownPolicy = {
  ALLOWED_TAGS: [
    'a', 'blockquote', 'br', 'code', 'del', 'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'hr', 'li', 'ol', 'p', 'pre', 'span', 'strong', 'table', 'tbody', 'td', 'th', 'thead',
    'tr', 'ul',
  ],
  ALLOWED_ATTR: ['class', 'href', 'rel', 'start', 'target', 'title'],
  ALLOW_ARIA_ATTR: false,
  ALLOW_DATA_ATTR: false,
  ALLOW_UNKNOWN_PROTOCOLS: false,
  FORBID_TAGS: ['button', 'embed', 'form', 'iframe', 'img', 'input', 'math', 'object', 'script', 'select', 'style', 'svg', 'textarea'],
  RETURN_TRUSTED_TYPE: false,
}

const _escapeAttribute = value => String(value)
  .replaceAll('&', '&amp;')
  .replaceAll('"', '&quot;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;')

export const safeExternalUrl = value => {
  if (typeof value !== 'string' || /[\u0000-\u001f\u007f]/.test(value)) return null
  const trimmed = value.trim()
  if (!trimmed || trimmed.startsWith('//')) return null
  try {
    const parsed = new URL(trimmed)
    const safeProtocol = parsed.protocol === 'https:' || parsed.protocol === 'http:'
    return safeProtocol && !parsed.username && !parsed.password ? parsed.href : null
  } catch (_error) {
    return null
  }
}

const _safeMarkdownUrl = value => {
  if (typeof value !== 'string' || /[\u0000-\u001f\u007f\\]/.test(value)) return null
  const trimmed = value.trim()
  if (!trimmed || trimmed.startsWith('//')) return null
  if (/^(?:https?:\/\/)/i.test(trimmed)) return safeExternalUrl(trimmed)
  if (/^mailto:[^\s@]+@[^\s@]+$/i.test(trimmed)) return trimmed
  if (/^(?:#|\/(?!\/)|\.\.?\/|\?)/.test(trimmed)) return trimmed
  return null
}

_markdownPurifier.addHook('afterSanitizeAttributes', node => {
  if (node.nodeName?.toLowerCase() !== 'a') return
  const href = _safeMarkdownUrl(node.getAttribute('href'))
  if (!href) {
    node.removeAttribute('href')
    node.removeAttribute('rel')
    node.removeAttribute('target')
    return
  }
  node.setAttribute('href', href)
  if (/^https?:\/\//i.test(href)) {
    node.setAttribute('target', '_blank')
    node.setAttribute('rel', 'noopener noreferrer')
  } else {
    node.removeAttribute('rel')
    node.removeAttribute('target')
  }
})

marked.setOptions({
  breaks: true,
  gfm: true,
})

const _markedRenderer = new marked.Renderer()
_markedRenderer.code = (token, lang) => {
  const codeStr = (token && typeof token === 'object') ? (token.text || '') : (token || '')
  const language = (token && typeof token === 'object') ? (token.lang || '') : (lang || '')
  if (language && hljs.getLanguage(language)) {
    try {
      const highlighted = hljs.highlight(codeStr, { language }).value
      return `<pre><code class="hljs language-${language}">${highlighted}</code></pre>`
    } catch(e) {}
  }
  try {
    const highlighted = hljs.highlightAuto(codeStr).value
    return `<pre><code class="hljs">${highlighted}</code></pre>`
  } catch(e) {}
  return `<pre><code>${codeStr}</code></pre>`
}
_markedRenderer.link = (href, title, text) => {
  const safeHref = _safeMarkdownUrl(href)
  if (!safeHref) return text
  const titleAttr = title ? ` title="${_escapeAttribute(title)}"` : ''
  const externalAttrs = /^https?:\/\//i.test(safeHref)
    ? ' target="_blank" rel="noopener noreferrer"'
    : ''
  return `<a href="${_escapeAttribute(safeHref)}"${titleAttr}${externalAttrs}>${text}</a>`
}
marked.use({ renderer: _markedRenderer })

export const stripAssistantProtocolArtifacts = text => {
  if (!text) return ''
  return String(text)
    // Older saved messages may contain the answer envelope itself. Its tags
    // are private transport syntax, while the text inside remains public.
    .replace(/<\s*\/?\s*LAVIX_ANSWER\b[^>]*>/gi, '')
    .replace(/\n?\s*<\s*\/?\s*LAVIX_ANSWER\b[^>]*$/i, '')
    // Follow-up payloads are a transport detail, including the malformed legacy
    // spelling "< LAVIX_FOLLOWUPS>" found in saved chat history.
    .replace(/\n?\s*<\s*\/?\s*LAVIX_FOLLOWUPS\b[^>]*>[\s\S]*$/i, '')
    .replace(/\n?\s*LAVIX_FOLLOWUPS\s*>?[\s\S]*$/i, '')
    .replace(/^\s*Evidence\s*:\s*(?:\[[VW]\d+\][\s,;]*)*$/gim, '')
    .replace(/\s*\{\s*"answer_id"\s*:\s*"[^"]*"[^}]*\}[\s\S]*$/i, '')
    // Some small models emit the metadata key on its own line, optionally
    // followed by an assignment. Hide it in live text and legacy history.
    .replace(/(?:^|\n)[ \t]*answer_id(?:[ \t]*[:=][^\n]*)?[ \t]*(?=\n|$)/gi, '')
    .replace(/\s*answer_id\s*=\s*\d+/gi, '')
    .replace(/\s*answer_id\s*[:=]\s*\d{8}_\d{3,}\s*/gi, '')
    .replace(/<lavix_answer_id>\d*<\/lavix_answer_id>/gi, '')
    .replace(/<answer_id>\s*<\/answer_id>/gi, '')
    .replace(/\bno_citation_needed\b/gi, '')
    .replace(/\s*answer_id\s*=\s*(?=\s*$)/gi, '')
    // Strip trailing followups JSON (quoted and unquoted keys)
    .replace(/\s*[,]?\s*"followups"\s*:\s*\[[\s\S]*?\]\s*[},]?\s*$/i, '')
    .replace(/\s*[,]?\s*followups\s*:\s*\[[\s\S]*?\]\s*[},]?\s*$/i, '')
    // Strip trailing bare date-like IDs (20230615_001)
    .replace(/\n\d{8}_\d{3,}[ \t]*$/g, '')
    // Strip trailing long bare numeric IDs (LLM answer_id without label).
    // Short numbers (totals, years, counts) are legitimate answer content.
    .replace(/(?:\n\d{6,}[ \t]*)+$/g, '')
    // Defense-in-depth: strip trailing echoed followup questions (lines ending
    // with ?) that the model repeats in the answer body. The backend already
    // handles this, but the 3B model sometimes echoes them as plain prose.
    .replace(/(?:\n[^?\n]+\?)+$/g, '')
    .trimEnd()
}

const _cleanAssistantCitations = text => stripAssistantProtocolArtifacts(text)
  .replace(/\[([VW]\d+)\]/gi, '')
  .replace(/[ \t]+([,.;!?])/g, '$1')
  .replace(/[ \t]{2,}/g, ' ')
  .replace(/\n{3,}/g, '\n\n')
  .trim()

export const assistantDisplayText = (text, sources = []) =>
  _cleanAssistantCitations(text)

export const renderMarkdown = (text) => {
  if (!text) return ''
  try {
    const cleaned = text.replace(/【[^】]*】/g, '').replace(/[ \t]{2,}/g, ' ')
    const parsed = marked.parse(cleaned)
    return _markdownPurifier.sanitize(parsed, _markdownPolicy)
  } catch(e) {
    const escaped = _escapeAttribute(text)
    return `<p>${escaped}</p>`
  }
}

export const renderAssistantMarkdown = (text, sources = [], anchorPrefix = 'source') =>
  renderMarkdown(_cleanAssistantCitations(text))

const _sessionActivityStorageKeys = ['last_activity_at', 'idle_expires_at', 'idle_timeout_seconds']
const _authStorageKeys = ['token', 'refresh_token', 'session_id', ..._sessionActivityStorageKeys]
export const AUTH_SESSION_BROADCAST_KEY = 'lavix_auth_session_event'
const SESSION_ACTIVITY_CLAIM_KEY = 'lavix_session_activity_sent_at'
let _authTerminated = false

export const readStoredSessionActivity = () => {
  const lastActivityAt = localStorage.getItem('last_activity_at')
  const idleExpiresAt = localStorage.getItem('idle_expires_at')
  const idleTimeoutSeconds = Number(localStorage.getItem('idle_timeout_seconds'))
  if (
    !lastActivityAt
    || !idleExpiresAt
    || !Number.isFinite(Date.parse(lastActivityAt))
    || !Number.isFinite(Date.parse(idleExpiresAt))
    || !Number.isFinite(idleTimeoutSeconds)
    || idleTimeoutSeconds <= 0
  ) return null
  return {
    last_activity_at: lastActivityAt,
    idle_expires_at: idleExpiresAt,
    idle_timeout_seconds: idleTimeoutSeconds,
  }
}

export const storeSessionActivity = session => {
  if (!session || typeof session !== 'object') return null
  const lastActivityMs = Date.parse(session.last_activity_at)
  const idleExpiresMs = Date.parse(session.idle_expires_at)
  const idleTimeoutSeconds = Number(session.idle_timeout_seconds)
  if (
    !Number.isFinite(lastActivityMs)
    || !Number.isFinite(idleExpiresMs)
    || !Number.isFinite(idleTimeoutSeconds)
    || idleTimeoutSeconds <= 0
  ) return null
  const activity = {
    last_activity_at: new Date(lastActivityMs).toISOString(),
    idle_expires_at: new Date(idleExpiresMs).toISOString(),
    idle_timeout_seconds: idleTimeoutSeconds,
  }
  for (const [key, value] of Object.entries(activity)) localStorage.setItem(key, String(value))
  window.dispatchEvent(new CustomEvent('session-activity-updated', { detail: activity }))
  return activity
}

export const storeAuthSession = (session, username) => {
  if (!session || typeof session !== 'object') throw new TypeError('Invalid authentication response')
  const previousSessionId = localStorage.getItem('session_id')
  const previousUsername = localStorage.getItem('username')
  const values = {
    token: session.access_token,
    refresh_token: session.refresh_token,
    session_id: session.session_id,
  }
  for (const [key, value] of Object.entries(values)) {
    if (typeof value !== 'string' || !value) throw new TypeError(`Missing ${key}`)
  }
  for (const [key, value] of Object.entries(values)) localStorage.setItem(key, value)
  storeSessionActivity(session)
  if (typeof username === 'string' && username.trim()) localStorage.setItem('username', username.trim())
  if (
    (previousSessionId && previousSessionId !== values.session_id)
    || (previousUsername && typeof username === 'string' && previousUsername !== username.trim())
  ) {
    window.dispatchEvent(new Event('lavix-preview-cache-clear'))
  }
  _authTerminated = false
  return values
}

export const clearAuthSession = ({ preserveUsername = false } = {}) => {
  for (const key of _authStorageKeys) localStorage.removeItem(key)
  localStorage.removeItem(SESSION_ACTIVITY_CLAIM_KEY)
  if (!preserveUsername) localStorage.removeItem('username')
  window.dispatchEvent(new Event('lavix-preview-cache-clear'))
}

const authDetailFromBody = body => {
  const detail = body?.detail && typeof body.detail === 'object' ? body.detail : body
  if (!detail || typeof detail !== 'object') return null
  return {
    code: detail.code || 'session_invalid',
    message: detail.message || formatApiError(body, 'Your session is no longer valid'),
    idle_timeout_seconds: Number(detail.idle_timeout_seconds) || undefined,
  }
}

const authDetailFromResponse = async response => {
  try { return authDetailFromBody(await response.clone().json()) } catch (_error) { return null }
}

const dispatchAuthFailure = (detail = {}, { broadcast = true } = {}) => {
  if (_authTerminated) return
  _authTerminated = true
  const normalized = {
    code: detail.code || 'session_invalid',
    message: detail.message || 'Your session is no longer valid',
    idle_timeout_seconds: Number(detail.idle_timeout_seconds) || undefined,
  }
  clearAuthSession({ preserveUsername: normalized.code === 'session_idle_timeout' })
  if (broadcast) {
    const id = window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`
    localStorage.setItem(AUTH_SESSION_BROADCAST_KEY, JSON.stringify({ id, detail: normalized }))
  }
  window.dispatchEvent(new CustomEvent('auth-error', { detail: normalized }))
}

export const formatApiError = (error, fallback = 'Request failed') => {
  if (!error) return fallback
  if (typeof error === 'string') return error
  if (Array.isArray(error)) {
    const messages = error.map(item => {
      if (!item || typeof item !== 'object') return String(item)
      const location = item.location || item.loc || []
      const path = Array.isArray(location)
        ? location.filter(part => part !== 'body').map(String).join('.')
        : ''
      const message = item.message || item.msg || fallback
      return path ? `${path}: ${message}` : String(message)
    }).filter(Boolean)
    return messages.join(' | ') || fallback
  }
  if (typeof error === 'object') {
    if (error.detail && error.detail !== error) return formatApiError(error.detail, fallback)
    return error.message || error.msg || error.error || fallback
  }
  return String(error)
}

;(function() {
  const _origFetch = window.fetch.bind(window)
  let refreshInFlight = null

  const requestPath = input => {
    const raw = typeof input === 'string' ? input : input?.url
    if (!raw) return ''
    try { return new URL(raw, window.location.origin).pathname } catch (_error) { return '' }
  }

  const publicAuthPaths = new Set([
    '/api/auth/capabilities',
    '/api/auth/google/token',
    '/api/auth/login',
    '/api/auth/refresh',
    '/api/auth/register',
  ])

  const rotateRefreshToken = async () => {
    const observedRefreshToken = localStorage.getItem('refresh_token')
    if (!observedRefreshToken) return null
    if (!refreshInFlight) {
      const rotate = async () => {
        const currentRefreshToken = localStorage.getItem('refresh_token')
        if (!currentRefreshToken) return null
        if (currentRefreshToken !== observedRefreshToken) return localStorage.getItem('token')
        const response = await _origFetch(`${API_URL}/auth/refresh`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: currentRefreshToken }),
        })
        if (!response.ok) {
          const error = new Error('Session refresh failed')
          error.authDetail = await authDetailFromResponse(response)
          throw error
        }
        const session = await response.json()
        storeAuthSession(session)
        window.dispatchEvent(new CustomEvent('auth-refreshed', { detail: session }))
        return session.access_token
      }
      const rotation = window.navigator?.locks?.request
        ? window.navigator.locks.request('lavix-auth-refresh', rotate)
        : rotate()
      refreshInFlight = rotation.finally(() => { refreshInFlight = null })
    }
    return refreshInFlight
  }

  const retryArguments = (args, accessToken, requestClone) => {
    const [input, options = {}] = args
    const headers = new Headers(input instanceof Request ? input.headers : options.headers)
    headers.set('Authorization', `Bearer ${accessToken}`)
    if (input instanceof Request) {
      return [new Request(requestClone, { ...options, headers })]
    }
    return [input, { ...options, headers }]
  }

  window.fetch = async function(...args) {
    const path = requestPath(args[0])
    const isApiRequest = path === '/api' || path.startsWith('/api/')
    const requestClone = args[0] instanceof Request ? args[0].clone() : null
    let res = await _origFetch(...args)
    if (res.status === 401 && isApiRequest && !publicAuthPaths.has(path)) {
      const initialDetail = await authDetailFromResponse(res)
      if (initialDetail?.code === 'session_idle_timeout') {
        dispatchAuthFailure(initialDetail)
      } else {
        try {
          const accessToken = await rotateRefreshToken()
          if (accessToken) res = await _origFetch(...retryArguments(args, accessToken, requestClone))
        } catch (error) {
          dispatchAuthFailure(error.authDetail || initialDetail || {})
        }
      }
    }
    const url = typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url) || ''
    if (res.status >= 500 && url.includes('/api') && !url.endsWith('/api/health')) {
      let reason = null
      try {
        const body = await res.clone().json()
        reason = body?.reason || body?.detail?.reason || body?.error?.reason || null
      } catch(e) {}
      if (reason === 'low_memory') {
        window.dispatchEvent(new CustomEvent('backend-error', { detail: { status: res.status, url, reason } }))
      }
    }
    if (res.status === 401 && isApiRequest && !publicAuthPaths.has(path)) {
      dispatchAuthFailure((await authDetailFromResponse(res)) || {})
    }
    return res
  }
})()

const getHeaders = (includeAuth = true) => {
  const h = {}
  if (includeAuth) {
    const token = localStorage.getItem('token')
    if (token) h['Authorization'] = `Bearer ${token}`
  }
  return h
}

export const installSessionActivityTracking = ({
  heartbeatThrottleMs = 60_000,
  now = () => Date.now(),
  setTimeoutFn = window.setTimeout.bind(window),
  clearTimeoutFn = window.clearTimeout.bind(window),
} = {}) => {
  let disposed = false
  let deadlineTimer = null
  let heartbeatTimer = null
  let heartbeatInFlight = null
  let pendingTrustedActivity = false

  const clearTimer = timer => {
    if (timer !== null) clearTimeoutFn(timer)
  }

  const scheduleDeadline = () => {
    clearTimer(deadlineTimer)
    deadlineTimer = null
    if (disposed || !localStorage.getItem('token')) return
    const activity = readStoredSessionActivity()
    if (!activity) return
    const delay = Date.parse(activity.idle_expires_at) - now()
    if (delay <= 0) {
      if (heartbeatInFlight) {
        heartbeatInFlight.then(scheduleDeadline, scheduleDeadline)
        return
      }
      dispatchAuthFailure({
        code: 'session_idle_timeout',
        message: `Session locked after ${Math.round(activity.idle_timeout_seconds / 60)} minutes of inactivity`,
        idle_timeout_seconds: activity.idle_timeout_seconds,
      })
      return
    }
    deadlineTimer = setTimeoutFn(scheduleDeadline, Math.min(delay, 2_147_000_000))
  }

  const scheduleHeartbeat = delay => {
    clearTimer(heartbeatTimer)
    heartbeatTimer = setTimeoutFn(() => {
      heartbeatTimer = null
      void sendHeartbeat().catch(() => {})
    }, Math.max(0, delay))
  }

  const postHeartbeat = async () => {
    if (
      disposed
      || !pendingTrustedActivity
      || document.visibilityState !== 'visible'
      || !localStorage.getItem('token')
    ) return null
    const sentAt = Number(localStorage.getItem(SESSION_ACTIVITY_CLAIM_KEY)) || 0
    const remaining = heartbeatThrottleMs - (now() - sentAt)
    if (remaining > 0) {
      scheduleHeartbeat(remaining)
      return null
    }

    const claim = now()
    localStorage.setItem(SESSION_ACTIVITY_CLAIM_KEY, String(claim))
    pendingTrustedActivity = false
    try {
      const response = await fetch(`${API_URL}/auth/activity`, {
        method: 'POST',
        headers: getHeaders(),
      })
      if (!response.ok) throw new Error('Session activity update failed')
      const activity = await response.json()
      storeSessionActivity(activity)
      return activity
    } catch (error) {
      if (localStorage.getItem(SESSION_ACTIVITY_CLAIM_KEY) === String(claim)) {
        localStorage.removeItem(SESSION_ACTIVITY_CLAIM_KEY)
      }
      throw error
    }
  }

  const sendHeartbeat = () => {
    if (heartbeatInFlight) return heartbeatInFlight
    const work = () => postHeartbeat()
    const request = window.navigator?.locks?.request
      ? window.navigator.locks.request('lavix-session-activity', work)
      : work()
    heartbeatInFlight = Promise.resolve(request).finally(() => { heartbeatInFlight = null })
    return heartbeatInFlight
  }

  const recordTrustedActivity = event => {
    if (
      disposed
      || event?.isTrusted !== true
      || document.visibilityState !== 'visible'
      || !localStorage.getItem('token')
    ) return Promise.resolve(null)
    pendingTrustedActivity = true
    return sendHeartbeat().catch(() => null)
  }

  const onStorage = event => {
    if (_sessionActivityStorageKeys.includes(event.key)) scheduleDeadline()
    if (event.key === AUTH_SESSION_BROADCAST_KEY && event.newValue) {
      try {
        const payload = JSON.parse(event.newValue)
        dispatchAuthFailure(payload?.detail || {}, { broadcast: false })
      } catch (_error) { /* Ignore malformed cross-tab messages. */ }
    }
  }
  const onMetadata = () => scheduleDeadline()
  const onVisibilityChange = () => {
    if (document.visibilityState !== 'visible') {
      clearTimer(heartbeatTimer)
      heartbeatTimer = null
    }
  }
  // `scroll` can be browser-generated after application code changes scrollTop
  // (for example, chat auto-scroll). A trusted wheel gesture records the same
  // user intent without allowing programmatic scrolling to renew the session.
  const trustedEvents = ['pointerdown', 'keydown', 'touchstart', 'wheel']
  for (const eventName of trustedEvents) {
    window.addEventListener(eventName, recordTrustedActivity, { passive: true, capture: true })
  }
  window.addEventListener('storage', onStorage)
  window.addEventListener('session-activity-updated', onMetadata)
  document.addEventListener('visibilitychange', onVisibilityChange)
  scheduleDeadline()

  return {
    recordTrustedActivity,
    syncDeadline: scheduleDeadline,
    dispose: () => {
      disposed = true
      clearTimer(deadlineTimer)
      clearTimer(heartbeatTimer)
      for (const eventName of trustedEvents) {
        window.removeEventListener(eventName, recordTrustedActivity, { capture: true })
      }
      window.removeEventListener('storage', onStorage)
      window.removeEventListener('session-activity-updated', onMetadata)
      document.removeEventListener('visibilitychange', onVisibilityChange)
    },
  }
}

export const api = {
  getHeaders,

  login: async (username, password) => {
    const res = await fetch(`${API_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    })
    return res.json()
  },
  register: async (username, email, password) => {
    const res = await fetch(`${API_URL}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password })
    })
    return res.json()
  },
  logout: async () => {
    try {
      if (!localStorage.getItem('token')) return { message: 'Already logged out' }
      const res = await fetch(`${API_URL}/auth/logout`, { method: 'POST', headers: getHeaders() })
      if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Logout failed'))
      return res.json()
    } finally {
      clearAuthSession()
    }
  },
  grantAIAccess: async (fileId) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/files/grant-ai-access/${fileId}`, {
      method: 'POST', headers: h
    })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail || body.error || 'Failed to grant access')
    }
    return res.json()
  },
  getFiles: async (showDeleted = false) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/files/list?show_deleted=${showDeleted}`, { headers: h })
    if (!res.ok) throw new Error('Failed to load files')
    return res.json()
  },
  getFolders: async () => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/folders`, { headers: h })
    if (!res.ok) throw new Error('Failed to load folders')
    return res.json()
  },
  createFolder: async (name, parentId = null) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/folders`, {
      method: 'POST', headers: h, body: JSON.stringify({ name, parent_id: parentId })
    })
    if (!res.ok) { const d = await res.json(); throw new Error(d.detail || 'Failed to create folder') }
    return res.json()
  },
  renameFolder: async (folderId, name) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/folders/${folderId}`, {
      method: 'PUT', headers: h, body: JSON.stringify({ name })
    })
    if (!res.ok) { const d = await res.json(); throw new Error(d.detail || 'Failed to rename folder') }
    return res.json()
  },
  moveFolder: async (folderId, parentId) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/folders/${folderId}/move`, {
      method: 'PATCH', headers: h, body: JSON.stringify({ parent_id: parentId ?? null })
    })
    if (!res.ok) { const d = await res.json(); throw new Error(d.detail || 'Failed to move folder') }
    return res.json()
  },
  deleteFolder: async (folderId) => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/folders/${folderId}`, { method: 'DELETE', headers: h })
    if (!res.ok) throw new Error('Failed to delete folder')
    return res.json()
  },
  getTrashedFolders: async () => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/folders/trash`, { headers: h })
    if (!res.ok) throw new Error('Failed to load trashed folders')
    return res.json()
  },
  restoreFolder: async (folderId) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/folders/${folderId}/restore`, { method: 'POST', headers: h })
    if (!res.ok) { const d = await res.json(); throw new Error(d.detail || 'Failed to restore folder') }
    return res.json()
  },
  hardDeleteFolder: async (folderId) => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/folders/${folderId}/hard`, { method: 'DELETE', headers: h })
    if (!res.ok) throw new Error('Failed to permanently delete folder')
    return res.json()
  },
  moveFile: async (fileId, folderId) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/files/move/${fileId}`, {
      method: 'PATCH', headers: h, body: JSON.stringify({ folder_id: folderId })
    })
    if (!res.ok) throw new Error('Failed to move file')
    return res.json()
  },
  renameFile: async (fileId, filename) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/files/rename/${fileId}`, {
      method: 'PATCH', headers: h, body: JSON.stringify({ filename })
    })
    if (!res.ok) throw new Error('Failed to rename file')
    return res.json()
  },
  copyFile: async (fileId, folderId) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/files/copy/${fileId}`, {
      method: 'POST', headers: h, body: JSON.stringify({ folder_id: folderId ?? null })
    })
    if (!res.ok) throw new Error('Failed to copy file')
    return res.json()
  },
  deleteFile: async (fileId) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/files/delete/${fileId}`, {
      method: 'DELETE', headers: h
    })
    if (!res.ok) throw new Error('Failed to move file to trash')
    return res.json()
  },
  restoreFile: async (fileId) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/files/restore/${fileId}`, {
      method: 'POST', headers: h
    })
    if (!res.ok) throw new Error('Failed to restore file')
    return res.json()
  },
  hardDeleteFile: async (fileId) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/files/hard-delete/${fileId}`, {
      method: 'DELETE', headers: h
    })
    if (!res.ok) throw new Error('Failed to hard delete file')
    return res.json()
  },
  emptyTrash: async () => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/files/trash/empty`, {
      method: 'DELETE', headers: h
    })
    if (!res.ok) throw new Error('Failed to empty trash')
    return res.json()
  },
  getAIStatus: async (fileId) => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/files/ai-status/${fileId}`, { headers: h })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail || body.error || 'Failed to load indexing status')
    }
    return res.json()
  },
  createPreviewSession: async (fileId) => {
    const numericId = Number(fileId)
    if (!Number.isInteger(numericId) || numericId <= 0) throw new Error('Invalid file ID')
    const h = getHeaders()
    const res = await fetch(`${API_URL}/files/create-session/${numericId}`, { method: 'POST', headers: h })
    if (!res.ok) {
      throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to create preview session'))
    }
    return res.json()
  },
  async chat({
    message,
    fileIds = null,
    deepSearch = false,
    webSearch = true,
    personaPrompt = null,
    signal = null,
    onToken,
    onStatus,
    onSources,
    onFollowups,
    onUsage,
    chatUploadIds = null,
    chatId = null,
    onError = null,
    onNotice = null,
    onClarification = null,
    mode = null,
    folderIds = null,
  }) {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const body = { message }
    // The backend owns the user's persisted chat-model preference. Keeping this
    // request model-free prevents stale tabs/localStorage from silently routing
    // chat through a different model than Settings reports.
    body.provider = 'ollama'
    // Null means "no explicit selection" (server may apply session scope);
    // an explicit empty array means "clear scope, search globally".
    if (fileIds !== null && fileIds !== undefined) body.file_ids = fileIds
    if (folderIds !== null && folderIds !== undefined) body.folder_ids = folderIds
    if (mode === 'casual' || mode === 'expert') body.mode = mode
    if (chatUploadIds && chatUploadIds.length > 0) body.chat_upload_ids = chatUploadIds
    if (deepSearch === true) body.deep_search = true
    body.web_search_enabled = webSearch
    if (personaPrompt) body.persona_prompt = personaPrompt
    if (chatId) body.chat_id = chatId
    const res = await fetch(`${API_URL}/ai/chat`, {
      method: 'POST', headers: h, body: JSON.stringify(body), signal
    })
    if (!res.ok) {
      const error = await res.json().catch(() => ({}))
      throw new Error(error.detail || error.error || 'Chat failed')
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let fullAnswer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6).trim()
        if (data === '[DONE]') return { answer: fullAnswer }
        try {
          const event = JSON.parse(data)
          if (event.type === 'token') {
            fullAnswer += event.content
            onToken && onToken(event.content)
          } else if (event.type === 'status') {
            onStatus && onStatus(event)
          } else if (event.type === 'sources') {
            onSources && onSources(event)
          } else if (event.type === 'followups') {
            onFollowups && onFollowups(event.questions)
          } else if (event.type === 'usage') {
            onUsage && onUsage(event)
          } else if (event.type === 'notice') {
            onNotice && onNotice(event)
          } else if (event.type === 'clarification') {
            // One-round web-clarification turn: terminal like error, but
            // rendered as a question — the next user message answers it.
            // Unknown to older clients, which ignore it gracefully.
            onClarification && onClarification(event)
            return { clarification: event }
          } else if (event.type === 'confirmation_needed') {
            return event
          } else if (event.type === 'error') {
            onError && onError(event)
            return { error: event }
          }
        } catch(e) {}
      }
    }
    return { answer: fullAnswer }
  },
  listChats: async () => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/ai/chats`, { headers: h })
    if (!res.ok) throw new Error('Failed to list chats')
    return res.json()
  },
  createChat: async (title = 'New Chat') => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/ai/chats`, { method: 'POST', headers: h, body: JSON.stringify({ title }) })
    if (!res.ok) throw new Error('Failed to create chat')
    return res.json()
  },
  getChat: async (chatId) => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/ai/chats/${chatId}`, { headers: h })
    if (!res.ok) throw new Error('Failed to get chat')
    return res.json()
  },
  patchChat: async (chatId, data) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/ai/chats/${chatId}`, { method: 'PATCH', headers: h, body: JSON.stringify(data) })
    if (!res.ok) throw new Error('Failed to update chat')
    return res.json()
  },
  setChatScope: async (chatId, fileIds = null, folderIds = null) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    // Null keys are omitted so folder-only edits can never wipe file scope
    // (and vice versa); explicit [] clears that key.
    const body = {}
    if (fileIds !== null && fileIds !== undefined) body.file_ids = fileIds
    if (folderIds !== null && folderIds !== undefined) body.folder_ids = folderIds
    const res = await fetch(`${API_URL}/ai/chats/${chatId}/scope`, { method: 'POST', headers: h, body: JSON.stringify(body) })
    if (!res.ok) throw new Error('Failed to update chat scope')
    return res.json()
  },
  deleteChat: async (chatId) => {    const h = getHeaders()
    const res = await fetch(`${API_URL}/ai/chats/${chatId}`, { method: 'DELETE', headers: h })
    if (!res.ok) throw new Error('Failed to delete chat')
    return res.json()
  },
  getChatHistory: async (sessionId = null) => {
    const h = getHeaders()
    let url = `${API_URL}/ai/chat/history`
    if (sessionId) url += `?session_id=${sessionId}`
    const res = await fetch(url, { headers: h })
    if (!res.ok) throw new Error('Failed to load chat history')
    return res.json()
  },
  clearChatHistory: async (sessionId = null) => {
    const h = getHeaders()
    const url = sessionId ? `${API_URL}/ai/chat/history/${sessionId}` : `${API_URL}/ai/chat/history`
    const res = await fetch(url, { method: 'DELETE', headers: h })
    if (!res.ok) throw new Error('Failed to delete chat history')
    return res.json()
  },
  getChatSessions: async () => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/ai/chat/sessions`, { headers: h })
    if (!res.ok) throw new Error('Failed to load chat sessions')
    return res.json()
  },
  deleteChatSession: async (sessionId) => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/ai/chat/history/${sessionId}`, { method: 'DELETE', headers: h })
    if (!res.ok) throw new Error('Failed to delete chat session')
    return res.json()
  },
  downloadFile: async (fileId, filename) => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/files/download/${fileId}`, { method: 'GET', headers: h })
    if (!res.ok) throw new Error('Download failed')
    const blob = await res.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(url)
    a.remove()
  },
  fetchFileObjectUrl: async (fileId) => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/files/download/${fileId}`, { method: 'GET', headers: h })
    if (!res.ok) throw new Error('Preview fetch failed')
    const blob = await res.blob()
    return window.URL.createObjectURL(blob)
  },
  revokeFileObjectUrl: (url) => {
    try { if (url) window.URL.revokeObjectURL(url) } catch (e) { }
  },
  uploadFile: async (file, replace = false, folderId = null, signal = null, folderName = null) => {
    const h = getHeaders()
    const formData = new FormData()
    formData.append('file', file)
    if (replace) formData.append('replace', 'true')
    if (folderId !== null) formData.append('folder_id', folderId)
    if (folderName) formData.append('folder_name', folderName)
    const res = await fetch(`${API_URL}/files/upload`, {
      method: 'POST', headers: h, body: formData, signal
    })
    if (!res.ok) {
      let errorDetail = 'Upload failed'
      let extra = {}
      try {
        const errorData = await res.json()
        if (errorData.detail && typeof errorData.detail === 'object') {
          errorDetail = errorData.detail.message || errorDetail
          if (errorData.detail.reason) errorDetail += ` (${errorData.detail.reason})`
          extra = errorData.detail
        } else {
          errorDetail = errorData.detail || errorDetail
        }
      } catch (e) { }
      const err = new Error(errorDetail)
      err.data = extra
      throw err
    }
    return res.json()
  },
  getAiStats: async () => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/ai/stats`, { headers: h })
    if (!res.ok) throw new Error('Failed to load AI stats')
    return res.json()
  },
  getMemory: async () => {
    const res = await fetch(`${API_URL}/ai/memory`, { headers: getHeaders() })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to load memory'))
    return res.json()
  },

  addMemory: async (text) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/ai/memory`, {
      method: 'POST', headers: h, body: JSON.stringify({ text })
    })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to add memory'))
    return res.json()
  },
  deleteMemoryItem: async (memoryId) => {
    const res = await fetch(`${API_URL}/ai/memory/${memoryId}`, { method: 'DELETE', headers: getHeaders() })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to delete memory'))
    return res.json()
  },

  getGraphMemory: async () => {
    const res = await fetch(`${API_URL}/ai/graph-memory`, { headers: getHeaders() })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to load relationship memory'))
    return res.json()
  },
  updateGraphMemory: async ({ enabled, retentionDays, expectedRevision }) => {
    const h = getHeaders(); h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/ai/graph-memory`, {
      method: 'PUT',
      headers: h,
      body: JSON.stringify({ enabled: Boolean(enabled), retention_days: Number(retentionDays), expected_revision: expectedRevision }),
    })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to update relationship memory'))
    return res.json()
  },
  getGraphMemoryItems: async ({ status = null, kind = null, limit = 50, offset = 0 } = {}) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
    if (status) params.set('status', status)
    if (kind) params.set('kind', kind)
    const res = await fetch(`${API_URL}/ai/graph-memory/items?${params}`, { headers: getHeaders() })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to load relationship memories'))
    return res.json()
  },
  getAllGraphMemoryItems: async ({ status = null, kind = null } = {}) => {
    const items = []
    let offset = 0
    let total = 0
    do {
      const page = await api.getGraphMemoryItems({ status, kind, limit: 100, offset })
      const pageItems = Array.isArray(page.items) ? page.items : []
      items.push(...pageItems)
      const reportedTotal = Number(page.total)
      total = Number.isFinite(reportedTotal) ? Math.max(0, reportedTotal) : items.length
      if (pageItems.length === 0) break
      offset += pageItems.length
    } while (offset < total)
    return { items, total, limit: 100, offset: 0 }
  },
  editGraphMemoryItem: async (memoryId, { subject, predicate, objectValue, expectedRevision }) => {
    const h = getHeaders(); h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/ai/graph-memory/items/${encodeURIComponent(memoryId)}`, {
      method: 'PATCH',
      headers: h,
      body: JSON.stringify({
        subject,
        predicate,
        object_value: objectValue,
        expected_revision: expectedRevision,
      }),
    })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to edit relationship memory'))
    return res.json()
  },
  approveGraphMemoryItem: async memoryId => {
    const res = await fetch(`${API_URL}/ai/graph-memory/items/${encodeURIComponent(memoryId)}/approve`, { method: 'POST', headers: getHeaders() })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to approve relationship memory'))
    return res.json()
  },
  renewGraphMemoryItem: async memoryId => {
    const res = await fetch(`${API_URL}/ai/graph-memory/items/${encodeURIComponent(memoryId)}/renew`, { method: 'POST', headers: getHeaders() })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to renew relationship memory'))
    return res.json()
  },
  deleteGraphMemoryItem: async memoryId => {
    const res = await fetch(`${API_URL}/ai/graph-memory/items/${encodeURIComponent(memoryId)}`, { method: 'DELETE', headers: getHeaders() })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to delete relationship memory'))
    return res.json()
  },
  clearGraphMemory: async () => {
    const res = await fetch(`${API_URL}/ai/graph-memory`, { method: 'DELETE', headers: getHeaders() })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to clear relationship memory'))
    return res.json()
  },
  clearPersonalMemory: async () => {
    const res = await fetch(`${API_URL}/ai/personal-memory`, { method: 'DELETE', headers: getHeaders() })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to clear personal memory'))
    return res.json()
  },
  searchVault: async (query, maxResults = 5) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/ai/search`, {
      method: 'POST',
      headers: h,
      body: JSON.stringify({ query, max_results: maxResults }),
    })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Vault search failed'))
    return res.json()
  },
  enableAllEmbeddings: async () => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/files/enable-all-embeddings`, { method: 'POST', headers: h })
    if (!res.ok) throw new Error('Failed to enable file indexing')
    return res.json()
  },
  revokeAIAccess: async (fileId) => {
    const h = getHeaders()
    const res = await fetch(`${API_URL}/files/revoke-ai-access/${fileId}`, { method: 'POST', headers: h })
    if (!res.ok) throw new Error('Failed to revoke AI access')
    return res.json()
  },
  revokeAIAccessBulk: async (fileIds) => {
    const h = { ...getHeaders(), 'Content-Type': 'application/json' }
    const res = await fetch(`${API_URL}/files/revoke-ai-access/bulk`, { method: 'POST', headers: h, body: JSON.stringify({ file_ids: fileIds }) })
    if (!res.ok) throw new Error('Failed to revoke AI access')
    return res.json()
  },
  grantAIAccessBulk: async (fileIds) => {
    const h = { ...getHeaders(), 'Content-Type': 'application/json' }
    const res = await fetch(`${API_URL}/files/grant-ai-access/bulk`, { method: 'POST', headers: h, body: JSON.stringify({ file_ids: fileIds }) })
    if (!res.ok) throw new Error('Failed to grant AI access')
    return res.json()
  },
  cancelIndexing: async (fileId = null) => {
    const h = getHeaders()
    const suffix = Number.isInteger(fileId) && fileId > 0 ? `/${fileId}` : ''
    const res = await fetch(`${API_URL}/files/cancel-indexing${suffix}`, { method: 'POST', headers: h })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to cancel indexing'))
    return res.json()
  },

  getModelConfig: async (provider) => {
    const params = provider ? `?provider=${encodeURIComponent(provider)}` : ''
    const res = await fetch(`${API_URL}/auth/model-config${params}`, { headers: getHeaders() })
    if (!res.ok) throw new Error('Failed to load model config')
    return res.json()
  },
  updateChatModel: async (model) => {
    const h = getHeaders()
    h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/auth/chat-model`, {
      method: 'PUT',
      headers: h,
      body: JSON.stringify({ model: model || null }),
    })
    if (!res.ok) {
      throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to update chat model'))
    }
    return res.json()
  },






  me: async () => {
    const res = await fetch(`${API_URL}/auth/me`, { headers: getHeaders() })
    if (!res.ok) throw new Error('Failed to fetch profile')
    return res.json()
  },
  updatePersona: async (personaPrompt) => {
    const h = getHeaders(); h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/auth/persona`, { method: 'PUT', headers: h, body: JSON.stringify({ persona_prompt: personaPrompt }) })
    if (!res.ok) throw new Error('Failed to update persona')
    return res.json()
  },
  updateUsername: async (username) => {
    const h = getHeaders(); h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/auth/username`, { method: 'PUT', headers: h, body: JSON.stringify({ username }) })
    if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || 'Failed to update username') }
    return res.json()
  },
  changePassword: async (currentPassword, newPassword) => {
    const h = getHeaders(); h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/auth/change-password`, { method: 'POST', headers: h, body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }) })
    if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || 'Failed to change password') }
    return res.json()
  },
  getSessions: async () => {
    const res = await fetch(`${API_URL}/auth/sessions`, { headers: getHeaders() })
    if (!res.ok) throw new Error('Failed to fetch sessions')
    return res.json()
  },
  revokeSession: async (sessionId) => {
    const res = await fetch(`${API_URL}/auth/sessions/${sessionId}`, { method: 'DELETE', headers: getHeaders() })
    if (!res.ok) throw new Error('Failed to revoke session')
    return res.json()
  },
  adminGetUsers: async () => {
    const res = await fetch(`${API_URL}/admin/users`, { headers: getHeaders() })
    if (!res.ok) throw new Error('Forbidden')
    return res.json()
  },
  adminUpdateUser: async (id, data) => {
    const h = getHeaders(); h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/admin/users/${id}`, { method: 'PATCH', headers: h, body: JSON.stringify(data) })
    if (!res.ok) throw new Error('Failed')
    return res.json()
  },
  adminResetPassword: async (id, newPassword) => {
    const h = getHeaders(); h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/admin/users/${id}/reset-password`, { method: 'POST', headers: h, body: JSON.stringify({ new_password: newPassword }) })
    if (!res.ok) throw new Error('Failed')
    return res.json()
  },
  adminGetSystem: async () => {
    const res = await fetch(`${API_URL}/admin/system`, { headers: getHeaders() })
    if (!res.ok) throw new Error('Failed')
    return res.json()
  },
  adminGetAiModels: async () => {
    const res = await fetch(`${API_URL}/admin/ai-models`, { headers: getHeaders() })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to load AI model configuration'))
    return res.json()
  },
  adminUpdateAiModels: async (configuration) => {
    const h = getHeaders(); h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/admin/ai-models`, {
      method: 'PUT', headers: h, body: JSON.stringify(configuration),
    })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to update AI model configuration'))
    return res.json()
  },
  adminGetPreferences: async () => {
    const res = await fetch(`${API_URL}/admin/preferences`, { headers: getHeaders() })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to load preferences'))
    return res.json()
  },
  adminUpdatePreferences: async (preferences) => {
    const h = getHeaders(); h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/admin/preferences`, {
      method: 'PUT', headers: h, body: JSON.stringify(preferences),
    })
    if (!res.ok) throw new Error(formatApiError(await res.json().catch(() => null), 'Failed to update preferences'))
    return res.json()
  },
  adminGetStorage: async () => {
    const res = await fetch(`${API_URL}/admin/storage`, { headers: getHeaders() })
    if (!res.ok) throw new Error('Failed')
    return res.json()
  },
  adminSetPermissions: async (userId, perms) => {
    const h = getHeaders(); h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/admin/users/${userId}/permissions`, { method: 'PATCH', headers: h, body: JSON.stringify(perms) })
    if (!res.ok) throw new Error('Failed')
    return res.json()
  },
  adminGetPolicies: async () => {
    const res = await fetch(`${API_URL}/admin/policies`, { headers: getHeaders() })
    if (!res.ok) throw new Error('Failed')
    return res.json()
  },
  adminClearEmbeddings: async () => {
    const res = await fetch(`${API_URL}/admin/clear-embeddings`, { method: 'POST', headers: getHeaders() })
    if (!res.ok) throw new Error('Failed to clear embeddings')
    return res.json()
  },
  adminApplyPolicy: async (userId, policyId) => {
    const h = getHeaders(); h['Content-Type'] = 'application/json'
    const res = await fetch(`${API_URL}/admin/users/${userId}/apply-policy`, { method: 'POST', headers: h, body: JSON.stringify({ policy_id: policyId }) })
    if (!res.ok) throw new Error('Failed')
    return res.json()
  },
  updateAvatar: async (avatarData) => {
    const h = { ...getHeaders(), 'Content-Type': 'application/json' }
    const res = await fetch(`${API_URL}/auth/avatar`, { method: 'PUT', headers: h, body: JSON.stringify({ avatar_data: avatarData }) })
    if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || 'Failed to update avatar') }
    return res.json()
  },
}

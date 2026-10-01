import { isPdfFile } from './frontend-workflows.js'

export const _mvThumbCache = new Map()
export const _mvSessionCache = new Map()
const _mvSessionRequests = new Map()
const _mvSessionCooldowns = new Map()
const _mvThumbnailGenerations = new Map()
const _mvThumbnailRequests = new Map()

const SESSION_REFRESH_SKEW_MS = 30_000
const SESSION_FAILURE_COOLDOWN_MS = 30_000
const MAX_BUFFERED_PREVIEW_BYTES = 30 * 1024 * 1024
const PDF_RANGE_CHUNK_BYTES = 1024 * 1024

// Backward-compatible window globals
window._mvThumbCache = _mvThumbCache
window._mvSessionCache = _mvSessionCache

export function clearMvPreviewCaches() {
  // Advance every known file before clearing requests. A render that was
  // started for the previous account/cache epoch may still finish, but it can
  // no longer repopulate the thumbnail cache.
  const knownFileIds = new Set([
    ..._mvThumbCache.keys(),
    ..._mvSessionCache.keys(),
    ..._mvSessionRequests.keys(),
    ..._mvSessionCooldowns.keys(),
    ..._mvThumbnailGenerations.keys(),
    ..._mvThumbnailRequests.keys(),
  ])
  const invalidations = []
  for (const fileId of knownFileIds) {
    const generation = (_mvThumbnailGenerations.get(fileId) || 0) + 1
    _mvThumbnailGenerations.set(fileId, generation)
    invalidations.push({ fileId, generation })
  }
  for (const request of _mvThumbnailRequests.values()) request.cancel?.()
  _mvThumbCache.clear()
  _mvSessionCache.clear()
  _mvSessionRequests.clear()
  _mvSessionCooldowns.clear()
  _mvThumbnailRequests.clear()
  // Clear already-rendered component-local data as well. `reload: false`
  // prevents account/logout cache clears from immediately creating new preview
  // sessions with credentials that are being removed.
  for (const { fileId, generation } of invalidations) {
    _emitThumbnailInvalidated(fileId, generation, { reload: false, reason: 'cache-clear' })
  }
}

function _numericFileId(fileId) {
  const numericId = Number(fileId)
  return Number.isInteger(numericId) && numericId > 0 ? numericId : null
}

function _thumbnailGeneration(fileId) {
  return _mvThumbnailGenerations.get(fileId) || 0
}

function _advanceThumbnailGeneration(fileId) {
  const generation = _thumbnailGeneration(fileId) + 1
  _mvThumbnailGenerations.set(fileId, generation)
  return generation
}

function _emitThumbnailInvalidated(fileId, generation, extra = {}) {
  window.dispatchEvent(new window.CustomEvent('lavix-thumbnail-invalidated', {
    detail: { fileId, generation, ...extra },
  }))
}

function _emitThumbnailGenerated(fileId, generation, thumbnail) {
  window.dispatchEvent(new window.CustomEvent('lavix-thumbnail-generated', {
    detail: { fileId, generation, thumbnail },
  }))
}

export function invalidateMvFileThumbnail(
  fileId,
  { reload = true, reason, failed = false, detachPendingSession = false } = {},
) {
  const numericId = _numericFileId(fileId)
  if (numericId === null) return false
  const generation = _advanceThumbnailGeneration(numericId)
  _mvThumbCache.delete(numericId)
  _mvThumbnailRequests.get(numericId)?.cancel?.()
  _mvThumbnailRequests.delete(numericId)
  if (detachPendingSession) {
    // Preserve a still-valid cached signed URL, but detach a hung/failed
    // session creation so a deliberate retry can create a genuinely fresh
    // request. The request identity check in mvGetPreviewSession fences any
    // late completion from repopulating shared state.
    _mvSessionRequests.delete(numericId)
    _mvSessionCooldowns.delete(numericId)
  }
  const extra = {}
  if (!reload) extra.reload = false
  if (reason) extra.reason = reason
  if (failed) extra.failed = true
  _emitThumbnailInvalidated(numericId, generation, extra)
  return true
}

export function invalidateMvFilePreview(fileId) {
  const numericId = _numericFileId(fileId)
  if (numericId === null) return false
  // Reset all shared state before notifying mounted components, so a listener
  // cannot synchronously adopt the preview session being invalidated.
  const generation = _advanceThumbnailGeneration(numericId)
  _mvThumbCache.delete(numericId)
  _mvThumbnailRequests.get(numericId)?.cancel?.()
  _mvThumbnailRequests.delete(numericId)
  _mvSessionCache.delete(numericId)
  _mvSessionRequests.delete(numericId)
  _mvSessionCooldowns.delete(numericId)
  _emitThumbnailInvalidated(numericId, generation)
  return true
}

window.addEventListener('lavix-preview-cache-clear', clearMvPreviewCaches)

function _sessionExpiry(ttlSeconds, now) {
  const ttlMs = Math.max(0, Number(ttlSeconds) || 0) * 1000
  const skew = Math.min(SESSION_REFRESH_SKEW_MS, Math.max(1_000, ttlMs * 0.1))
  return now + Math.max(0, ttlMs - skew)
}

export async function mvGetPreviewSession(fileId, createSession, { force = false, now = Date.now() } = {}) {
  const numericId = _numericFileId(fileId)
  if (numericId === null) throw new Error('invalid file id')
  if (typeof createSession !== 'function') throw new TypeError('createSession must be a function')

  if (force) {
    _mvSessionCache.delete(numericId)
    _mvSessionCooldowns.delete(numericId)
  }
  const cached = _mvSessionCache.get(numericId)
  if (cached?.previewUrl && cached.expiresAt > now) return cached

  const cooldown = _mvSessionCooldowns.get(numericId)
  if (cooldown && cooldown.until > now) throw cooldown.error
  const pending = _mvSessionRequests.get(numericId)
  if (pending) return pending

  let request
  request = Promise.resolve(createSession(numericId))
    .then(session => {
      const previewUrl = String(session?.preview_url || '')
      if (!previewUrl) throw new Error('preview session has no URL')
      const value = {
        ...session,
        previewUrl,
        expiresAt: _sessionExpiry(session?.expires_in_seconds, now),
      }
      // A full preview reset removes this request from the registry. Keep
      // returning its result to its original caller, but do not resurrect the
      // invalidated session in shared state.
      if (_mvSessionRequests.get(numericId) === request) {
        _mvSessionCache.set(numericId, value)
        _mvSessionCooldowns.delete(numericId)
      }
      return value
    })
    .catch(error => {
      const safeError = error instanceof Error ? error : new Error('preview session failed')
      if (_mvSessionRequests.get(numericId) === request) {
        _mvSessionCooldowns.set(numericId, { error: safeError, until: now + SESSION_FAILURE_COOLDOWN_MS })
      }
      throw safeError
    })
    .finally(() => {
      if (_mvSessionRequests.get(numericId) === request) _mvSessionRequests.delete(numericId)
    })
  _mvSessionRequests.set(numericId, request)
  return request
}

let _pdfjsPromise
let _xlsxPromise
let _mammothPromise

function _loadPdfJs() {
  if (!_pdfjsPromise) {
    _pdfjsPromise = Promise.all([
      import('pdfjs-dist'),
      import('pdfjs-dist/build/pdf.worker.min.mjs?url'),
    ]).then(([pdfjs, worker]) => {
      // The query changes the cache key once after the production server's
      // .mjs MIME fix; prior immutable responses may have cached octet-stream.
      pdfjs.GlobalWorkerOptions.workerSrc = `${worker.default}?lavix-pdf-worker=2`
      return pdfjs
    })
  }
  return _pdfjsPromise
}

function _loadXlsx() {
  if (!_xlsxPromise) _xlsxPromise = import('@e965/xlsx')
  return _xlsxPromise
}

function _loadMammoth() {
  if (!_mammothPromise) _mammothPromise = import('mammoth/mammoth.browser.js')
  return _mammothPromise
}

async function _fetchPreview(url, { signal } = {}) {
  const res = await fetch(url, { signal })
  if (!res.ok) {
    const error = new Error('fetch ' + res.status)
    error.status = res.status
    throw error
  }
  return res
}

export function mvPdfLoadMode(sizeBytes) {
  return Math.max(0, Number(sizeBytes) || 0) > MAX_BUFFERED_PREVIEW_BYTES ? 'range' : 'buffer'
}

export async function mvLoadPdfDocument(url, sizeBytes, { signal } = {}) {
  const pdfjsLib = await _loadPdfJs()
  let source
  if (mvPdfLoadMode(sizeBytes) === 'range') {
    source = {
      url: new URL(url, window.location.href).href,
      rangeChunkSize: PDF_RANGE_CHUNK_BYTES,
      disableStream: true,
      disableAutoFetch: true,
    }
  } else {
    const res = await _fetchPreview(url, { signal })
    source = { data: new Uint8Array(await res.arrayBuffer()) }
  }
  const loadingTask = pdfjsLib.getDocument({ ...source, isEvalSupported: false })
  const cancel = () => {
    try { Promise.resolve(loadingTask.destroy?.()).catch(() => {}) } catch { /* already destroyed */ }
  }
  signal?.addEventListener('abort', cancel, { once: true })
  if (signal?.aborted) cancel()
  try {
    return await loadingTask.promise
  } finally {
    signal?.removeEventListener('abort', cancel)
  }
}

export async function mvDestroyPdfDocument(pdf) {
  if (!pdf) return
  if (typeof pdf.destroy === 'function') {
    await pdf.destroy()
    return
  }
  if (typeof pdf.loadingTask?.destroy === 'function') {
    await pdf.loadingTask.destroy()
    return
  }
  if (typeof pdf.cleanup === 'function') await pdf.cleanup()
}

export async function mvRenderPdfPage(pdf, pageNumber, canvas, { maxWidth = 1200, signal } = {}) {
  if (!pdf || !canvas) throw new Error('PDF renderer is not ready')
  const page = await pdf.getPage(pageNumber)
  try {
    const baseViewport = page.getViewport({ scale: 1 })
    const displayScale = Math.max(0.25, Math.min(2, Number(maxWidth) / baseViewport.width))
    const pixelRatio = Math.max(1, Math.min(2, Number(window.devicePixelRatio) || 1))
    const viewport = page.getViewport({ scale: displayScale * pixelRatio })
    canvas.width = Math.max(1, Math.round(viewport.width))
    canvas.height = Math.max(1, Math.round(viewport.height))
    canvas.style.width = `${Math.round(baseViewport.width * displayScale)}px`
    canvas.style.height = `${Math.round(baseViewport.height * displayScale)}px`
    const renderTask = page.render({ canvasContext: canvas.getContext('2d'), viewport })
    const cancel = () => { try { renderTask.cancel?.() } catch { /* already complete */ } }
    signal?.addEventListener('abort', cancel, { once: true })
    if (signal?.aborted) cancel()
    try {
      await renderTask.promise
    } finally {
      signal?.removeEventListener('abort', cancel)
    }
    return { width: canvas.width, height: canvas.height }
  } finally {
    page.cleanup()
  }
}

async function _pdfThumb(url, sizeBytes, { signal } = {}) {
  const pdf = await mvLoadPdfDocument(url, sizeBytes, { signal })
  try {
    const page = await pdf.getPage(1)
    try {
      const vp = page.getViewport({ scale: 0.4 })
      const canvas = document.createElement('canvas')
      canvas.width = Math.round(vp.width)
      canvas.height = Math.round(vp.height)
      const renderTask = page.render({ canvasContext: canvas.getContext('2d'), viewport: vp })
      const cancel = () => { try { renderTask.cancel?.() } catch { /* already complete */ } }
      signal?.addEventListener('abort', cancel, { once: true })
      if (signal?.aborted) cancel()
      try {
        await renderTask.promise
      } finally {
        signal?.removeEventListener('abort', cancel)
      }
      return canvas.toDataURL('image/jpeg', 0.75)
    } finally {
      page.cleanup()
    }
  } finally {
    await mvDestroyPdfDocument(pdf)
  }
}

async function _sheetThumb(url, { signal } = {}) {
  const XLSX = await _loadXlsx()
  const res = await _fetchPreview(url, { signal })
  const ab = await res.arrayBuffer()
  const wb   = XLSX.read(ab, { type: 'array', sheetRows: 6 })
  const ws   = wb.Sheets[wb.SheetNames[0]]
  const rows = XLSX.utils.sheet_to_json(ws, { header: 1, range: 0, defval: '', blankrows: true })
  const W = 160, H = 120
  const canvas = document.createElement('canvas')
  canvas.width = W; canvas.height = H
  const ctx = canvas.getContext('2d')
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, W, H)
  const COLS = 4, CW = 39, CH = 21, OX = 1, OY = 1
  const visRows = Math.min(rows.length, 5)
  for (let ri = 0; ri < visRows; ri++) {
    for (let ci = 0; ci < COLS; ci++) {
      const x = OX + ci * CW, y = OY + ri * CH
      if (ri === 0) { ctx.fillStyle = '#e5e7eb'; ctx.fillRect(x, y, CW - 1, CH - 1) }
      ctx.strokeStyle = '#d1d5db'; ctx.lineWidth = 0.5
      ctx.strokeRect(x, y, CW - 1, CH - 1)
      const cell = String((rows[ri] || [])[ci] !== undefined ? (rows[ri] || [])[ci] : '').slice(0, 9)
      ctx.fillStyle = '#111827'
      ctx.font = (ri === 0 ? 'bold ' : '') + '6px Arial'
      ctx.textBaseline = 'middle'
      ctx.fillText(cell, x + 2, y + CH / 2)
    }
  }
  return canvas.toDataURL('image/jpeg', 0.85)
}

async function _docxThumb(url, { signal } = {}) {
  const mammoth = await _loadMammoth()
  const res = await _fetchPreview(url, { signal })
  const ab = await res.arrayBuffer()
  const result = await mammoth.extractRawText({ arrayBuffer: ab })
  return _textCanvas(result.value.trim().slice(0, 500), 'doc')
}

async function _textThumb(url, { signal } = {}) {
  const res = await _fetchPreview(url, { signal })
  const raw = await res.text()
  return _textCanvas(raw.slice(0, 500), 'code')
}

function _textCanvas(text, style) {
  const W = 160, H = 120, LH = 8.5, PAD = 6
  const isDoc = style === 'doc'
  const canvas = document.createElement('canvas')
  canvas.width = W; canvas.height = H
  const ctx = canvas.getContext('2d')
  ctx.fillStyle = isDoc ? '#ffffff' : '#0d1117'
  ctx.fillRect(0, 0, W, H)
  if (isDoc) {
    ctx.strokeStyle = '#f3f4f6'; ctx.lineWidth = 0.5
    for (let yl = PAD + LH; yl < H - PAD; yl += LH) {
      ctx.beginPath(); ctx.moveTo(PAD, yl); ctx.lineTo(W - PAD, yl); ctx.stroke()
    }
  }
  ctx.fillStyle = isDoc ? '#374151' : '#8b949e'
  ctx.font      = isDoc ? '6.5px Georgia,serif' : '5.5px "Courier New",monospace'
  ctx.textBaseline = 'top'
  const words = text.split(/\s+/).filter(Boolean)
  const maxW  = W - PAD * 2
  let line = '', y = PAD
  for (let i = 0; i < words.length; i++) {
    const test = line ? line + ' ' + words[i] : words[i]
    if (ctx.measureText(test).width > maxW && line) {
      ctx.fillText(line, PAD, y)
      y += LH
      if (y > H - PAD) break
      line = words[i]
    } else { line = test }
  }
  if (line && y <= H - PAD) ctx.fillText(line, PAD, y)
  return canvas.toDataURL('image/jpeg', 0.75)
}

function _thumbnailPlan(file) {
  const mime = ((file.mime_type || '') + '').toLowerCase()
  const name = ((file.filename  || '') + '').toLowerCase()
  const sz   = Number(file.file_size_bytes ?? file.size_bytes ?? 0) || 0
  if (mime.startsWith('video/') || mime.startsWith('audio/')) return { kind: null, reason: 'unsupported', sizeBytes: sz }
  if (mime.startsWith('image/')) return { kind: 'image', sizeBytes: sz }
  if (isPdfFile(file)) return { kind: 'pdf', sizeBytes: sz }
  if (sz > MAX_BUFFERED_PREVIEW_BYTES) return { kind: null, reason: 'too large', sizeBytes: sz }
  if (mime.includes('spreadsheet') || mime.includes('excel') ||
      mime === 'text/csv'          || /\.(xlsx?|csv)$/.test(name)) return { kind: 'sheet', sizeBytes: sz }
  if (mime.includes('wordprocess') || name.endsWith('.docx')) return { kind: 'docx', sizeBytes: sz }
  if (mime.startsWith('text/') ||
      /\.(txt|md|json|js|ts|py|sh|css|html|xml|yaml|toml|ini|cfg)$/.test(name)) {
    return { kind: 'text', sizeBytes: sz }
  }
  return { kind: null, reason: 'no handler', sizeBytes: sz }
}

export function mvCanGenerateThumbnail(file) {
  return _thumbnailPlan(file).kind !== null
}

export async function mvGenerateThumbnail(file, previewUrl, { signal } = {}) {
  const plan = _thumbnailPlan(file)
  if (plan.kind === 'image') return previewUrl
  if (plan.kind === 'pdf') return _pdfThumb(previewUrl, plan.sizeBytes, { signal })
  if (plan.kind === 'sheet') return _sheetThumb(previewUrl, { signal })
  if (plan.kind === 'docx') return _docxThumb(previewUrl, { signal })
  if (plan.kind === 'text') return _textThumb(previewUrl, { signal })
  throw new Error(plan.reason)
}

function _thumbnailCallArguments(generateThumbnailOrOptions, maybeOptions) {
  if (typeof generateThumbnailOrOptions === 'function') {
    return {
      generateThumbnail: generateThumbnailOrOptions,
      options: maybeOptions && typeof maybeOptions === 'object' ? maybeOptions : {},
    }
  }
  return {
    generateThumbnail: typeof maybeOptions === 'function' ? maybeOptions : mvGenerateThumbnail,
    options: generateThumbnailOrOptions && typeof generateThumbnailOrOptions === 'object'
      ? generateThumbnailOrOptions
      : {},
  }
}

async function _generateFileThumbnailForEpoch(file, fileId, generation, createSession, generateThumbnail, signal) {
  let session = await mvGetPreviewSession(fileId, createSession)
  // Invalidation while the preview session was being created makes this work
  // stale. It may still resolve for its original caller, but the generation
  // fence below prevents it from mutating shared thumbnail state.
  if (_thumbnailGeneration(fileId) !== generation) {
    return generateThumbnail(file, session.previewUrl, { signal })
  }

  let thumbnail
  try {
    thumbnail = await generateThumbnail(file, session.previewUrl, { signal })
  } catch (error) {
    if (_thumbnailGeneration(fileId) !== generation) throw error
    if (error?.status !== 403 && error?.status !== 404) throw error
    // A rejected signed URL is the sole reason thumbnail generation replaces a
    // preview session. Explicit thumbnail regeneration otherwise reuses a valid
    // cached session.
    _mvSessionCache.delete(fileId)
    session = await mvGetPreviewSession(fileId, createSession, { force: true })
    thumbnail = await generateThumbnail(file, session.previewUrl, { signal })
  }

  if (_thumbnailGeneration(fileId) === generation) {
    _mvThumbCache.set(fileId, thumbnail)
    _emitThumbnailGenerated(fileId, generation, thumbnail)
  }
  return thumbnail
}

export function mvGenerateFileThumbnail(
  file,
  createSession,
  generateThumbnailOrOptions = mvGenerateThumbnail,
  maybeOptions,
) {
  const fileId = _numericFileId(file?.file_id ?? file?.id)
  if (fileId === null) return Promise.reject(new Error('invalid file id'))

  const { generateThumbnail, options } = _thumbnailCallArguments(generateThumbnailOrOptions, maybeOptions)
  const force = Boolean(options.force)
  const supersede = Boolean(options.supersede)
  let generation = _thumbnailGeneration(fileId)
  let pending = _mvThumbnailRequests.get(fileId)

  if (force) {
    // Repeated force callers share the regeneration already underway. A force
    // call does supersede an ordinary render from the prior epoch.
    if (pending?.generation === generation && pending.forced && !supersede) return pending.promise
    invalidateMvFileThumbnail(fileId, { detachPendingSession: supersede })
    generation = _thumbnailGeneration(fileId)

    // Event subscribers can synchronously start the fresh render. Adopt that
    // request rather than creating a duplicate for the same file and epoch.
    pending = _mvThumbnailRequests.get(fileId)
    if (pending?.generation === generation) {
      pending.forced = true
      return pending.promise
    }
  } else {
    if (_mvThumbCache.has(fileId)) return Promise.resolve(_mvThumbCache.get(fileId))
    if (pending?.generation === generation) return pending.promise
  }

  const controller = new AbortController()
  const promise = _generateFileThumbnailForEpoch(
    file,
    fileId,
    generation,
    createSession,
    generateThumbnail,
    controller.signal,
  )
  const request = { generation, forced: force, promise, cancel: () => controller.abort() }
  _mvThumbnailRequests.set(fileId, request)
  promise.finally(() => {
    if (_mvThumbnailRequests.get(fileId) === request) _mvThumbnailRequests.delete(fileId)
  }).catch(() => {})
  return promise
}

window.mvGenerateThumbnail = mvGenerateThumbnail
window.mvGenerateFileThumbnail = mvGenerateFileThumbnail
window.mvGetPreviewSession = mvGetPreviewSession
window.mvLoadPdfDocument = mvLoadPdfDocument
window.mvRenderPdfPage = mvRenderPdfPage
window.clearMvPreviewCaches = clearMvPreviewCaches
window.invalidateMvFileThumbnail = invalidateMvFileThumbnail
window.invalidateMvFilePreview = invalidateMvFilePreview

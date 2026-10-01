const activeIndexStates = new Set([
  'processing',
  'queued',
  'decrypting',
  'converting',
  'parsing',
  'chunking',
  'embedding',
  'publishing',
])

const indexStateLabels = {
  queued: 'Queued',
  decrypting: 'Decrypting',
  converting: 'Converting',
  parsing: 'Parsing',
  chunking: 'Chunking',
  embedding: 'Embedding',
  publishing: 'Indexing…',
  ready: 'Indexed',
  failed: 'Failed',
  cancelled: 'Cancelled',
  unsupported: 'Unsupported',
  password_required: 'Password required',
  not_granted: 'Not indexed',
}

export function isPdfFile(file = {}) {
  const mime = String(file.mime_type || '').trim().toLowerCase().split(';', 1)[0]
  if (mime === 'application/pdf') return true
  return [file.filename, file.original_filename]
    .some(name => /\.pdf$/i.test(String(name || '').trim()))
}

export function isImageFile(file = {}) {
  const mime = String(file.mime_type || '').trim().toLowerCase().split(';', 1)[0]
  if (mime.startsWith('image/')) return true
  return [file.filename, file.original_filename]
    .some(name => /\.(png|jpe?g|webp|gif|svg)$/i.test(String(name || '').trim()))
}

export function getIndexState(file = {}) {
  const rawState = file.state || file.ingestion_state || file.ai_status || file.status
  if (rawState === 'completed') return 'ready'
  if (rawState === 'locked') return 'not_granted'
  if (rawState === 'not_supported') return 'unsupported'
  return rawState || 'not_granted'
}

export function isIndexReady(file = {}) {
  return getIndexState(file) === 'ready'
}

export function isIndexSearchable(file = {}) {
  if (typeof file.ai_searchable === 'boolean') return file.ai_searchable
  if (typeof file.embeddings_ready === 'boolean') return file.embeddings_ready
  return isIndexReady(file)
}

export function isIndexActive(file = {}) {
  return activeIndexStates.has(getIndexState(file))
}

export function isIndexUnsupported(file = {}) {
  const mime = String(file.mime_type || '').toLowerCase()
  return getIndexState(file) === 'unsupported' || mime.startsWith('video/') || mime.startsWith('audio/')
}

export function getIndexStateLabel(file = {}) {
  const state = getIndexState(file)
  return indexStateLabels[state] || state.replace(/_/g, ' ').replace(/\b\w/g, character => character.toUpperCase())
}

export function normalizeActiveChatModelConfig(data = {}) {
  const exposedAllowlist = data.allowed_chat_models || data.allowed_models
  const availableModels = Array.isArray(exposedAllowlist) && exposedAllowlist.length > 0
    ? exposedAllowlist
    : [data.active_chat_model, data.chat?.model, data.model].filter(Boolean)
  const serverModel = data.active_chat_model || data.chat?.model || data.model || ''
  const model = availableModels.includes(serverModel)
    ? serverModel
    : (availableModels[0] || serverModel)
  return {
    provider: data.provider || data.chat?.provider || 'ollama',
    model,
    available_models: [...new Set(availableModels)],
  }
}

export function mergeAdminAndAccountModelConfig(adminConfig = {}, accountConfig = {}) {
  const accountChat = accountConfig.chat || {}
  const activeModel = accountConfig.active_chat_model || accountChat.model || accountConfig.model || null
  const preferredModel = accountConfig.preferred_chat_model ?? accountChat.preferred_model ?? null
  const installedNames = (adminConfig.installed_models || [])
    .map(model => typeof model === 'string' ? model : model?.name)
    .filter(Boolean)

  return {
    ...adminConfig,
    provider: accountConfig.provider || 'ollama',
    active_chat_model: activeModel,
    preferred_chat_model: preferredModel,
    allowed_chat_models: [...(accountConfig.allowed_chat_models || adminConfig.chat?.allowed_models || [])],
    available_models: [...(accountConfig.available_models || installedNames)],
    chat: {
      ...(adminConfig.chat || {}),
      model: activeModel,
      preferred_model: preferredModel,
      active_available: accountChat.available ?? false,
      available: adminConfig.chat?.available ?? accountChat.available ?? false,
    },
  }
}

export function resolveMemoryConsent(legacyEnabled, graphEnabled) {
  const canonicalEnabled = Boolean(graphEnabled)
  const mismatch = Boolean(legacyEnabled) !== canonicalEnabled
  return {
    enabled: canonicalEnabled && !mismatch,
    mismatch,
  }
}

export function dedupeAboutMeSentences(sentences) {
  if (!Array.isArray(sentences)) return []
  const unique = new Map()

  for (const sentence of sentences) {
    if (!sentence || typeof sentence !== 'object') continue
    const text = String(sentence.text || '').normalize('NFKC').replace(/\s+/g, ' ').trim()
    if (!text) continue
    // Sentence-final punctuation is presentation, not identity. Keep the
    // first display form while treating "Claim", "Claim.", and "Claim!" as
    // the same grounded fact.
    const foldedText = text.toLocaleLowerCase()
    const key = foldedText.replace(/[.!?…。！？]+$/gu, '') || foldedText
    const current = unique.get(key)
    if (!current) {
      unique.set(key, { ...sentence, text, backing_memory_ids: [] })
    }
    const display = unique.get(key)
    const seenIds = new Set(display.backing_memory_ids.map(value => String(value).toLocaleLowerCase()))
    for (const memoryId of Array.isArray(sentence.backing_memory_ids) ? sentence.backing_memory_ids : []) {
      const id = String(memoryId || '').trim()
      const idKey = id.toLocaleLowerCase()
      if (!id || seenIds.has(idKey)) continue
      seenIds.add(idKey)
      display.backing_memory_ids.push(id)
    }
  }

  return [...unique.values()]
}

export function normalizeMatchPercentage(value, { fractional = false } = {}) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return null
  const percentage = fractional && numeric >= 0 && numeric <= 1 ? numeric * 100 : numeric
  return Math.round(Math.max(0, Math.min(100, percentage)))
}

export function resolveMatchPercentage(source = {}) {
  if (source.match_percentage !== null && source.match_percentage !== undefined) {
    return normalizeMatchPercentage(source.match_percentage)
  }
  return normalizeMatchPercentage(source.relevance_score ?? source.score, { fractional: true })
}

export function getIndexProgress(stats = {}) {
  const value = stats && typeof stats === 'object' && !Array.isArray(stats) ? stats : {}
  // Beta-dev parity: the numerator is ai_ready (published revision exists,
  // consent not required — revoke nulls the revision, so revokes drop it).
  // Older payloads fall back to `searchable`.
  const searchable = Math.max(0, Number(value.ai_ready ?? value.searchable) || 0)
  // Honest denominator: only consented, parser-supported files are indexable.
  // Older deployments without the field fall back to `supported`.
  const indexable = Math.max(searchable, Number(value.indexable ?? value.supported ?? value.total) || 0)
  const supported = Math.max(indexable, Number(value.supported ?? value.total) || 0)
  const active = Math.max(0, Number(value.active) || 0)
  // Box fraction denominator: every non-deleted file (media and
  // unsupported formats included). Older payloads fall back to `total`.
  const allFilesTotal = Math.max(0, Number(value.all_files_total ?? value.total) || 0)
  // Kept for backward compatibility: supported minus permanently
  // un-actionable files.
  const actionable = Math.max(searchable, Number(value.actionable ?? supported) || 0)
  // Eligible base: everything that could still become searchable
  // (indexable backlog + grantable-but-ungranted files, no double
  // count). Prefer the server-computed `eligible`; legacy payloads fall
  // back to `indexable`. Empty eligible means nothing to show: 0%, never
  // 100% — an empty vault is not "ready".
  const eligible = Math.max(0, Number(value.eligible ?? value.indexable ?? value.supported) || 0)
  // Percentage base is `eligible`, so the gap always equals an
  // identifiable remaining set (fresh backlog + grantable + retry).
  // Display rule: whole numbers only, always rounded DOWN while work
  // remains — 99.9% ready must read 99%, never 100%. Only a complete
  // ratio (searchable >= eligible) may display 100%.
  const rawPercentage = eligible > 0 ? Math.min(100, (searchable / eligible) * 100) : 0
  const percentage = rawPercentage >= 100 ? 100 : Math.floor(rawPercentage)
  return { searchable, supported, actionable, allFilesTotal, eligible, indexable, active, percentage }
}

// Box display state for the AI-ready pill. An empty vault is an empty state
// ("No files yet"), never "ready". All-failed (nothing searchable, nothing
// in flight, at least one failure) is a distinct failed state, never a
// silent 0% and never 100%.
export function aiBoxState(stats = {}) {
  const progress = getIndexProgress(stats)
  const failed = Math.max(0, Number(stats?.failed) || 0)
  const empty = progress.allFilesTotal === 0
  const failedAll = !empty && progress.searchable === 0 && progress.active === 0 && failed > 0
  return { empty, failed: failedAll, progress }
}

export function getServiceHealthPresentation(status, readiness = {}) {
  const readinessState = String(readiness?.status || '').toLowerCase()
  const warning = readinessState === 'busy' || readinessState === 'checking'
  const healthy = status === 'ok' && !['degraded', 'offline'].includes(readinessState)
  const label = warning
    ? readinessState.replace(/_/g, ' ').replace(/\b\w/g, letter => letter.toUpperCase())
    : String(status || 'unknown')
  return { readinessState, tone: warning ? 'warning' : healthy ? 'healthy' : 'error', label }
}

export function documentTypeLabel(value) {
  const normalized = String(value || '').trim().toLowerCase()
  if (!normalized) return ''
  return normalized
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, character => character.toUpperCase())
}

export function fileFormatLabel(file = {}) {
  const filename = String(file.filename || file.original_filename || '').trim()
  const extension = filename.match(/\.([a-z0-9]{1,10})$/i)?.[1]
  if (extension) return extension.toUpperCase()

  const mime = String(file.mime_type || '').trim().toLowerCase()
  const exact = {
    'application/pdf': 'PDF',
    'text/plain': 'TXT',
    'text/csv': 'CSV',
    'text/html': 'HTML',
    'application/json': 'JSON',
  }
  if (exact[mime]) return exact[mime]
  if (mime.includes('wordprocessingml')) return 'DOCX'
  if (mime.includes('msword')) return 'DOC'
  if (mime.includes('spreadsheetml')) return 'XLSX'
  if (mime.includes('ms-excel')) return 'XLS'
  if (mime.includes('presentationml')) return 'PPTX'
  if (mime.includes('ms-powerpoint')) return 'PPT'
  if (mime.startsWith('image/')) return mime.slice('image/'.length).replace('jpeg', 'JPG').toUpperCase()
  return 'FILE'
}

export function matchesSemanticTags(tags, query) {
  if (!Array.isArray(tags)) return false
  const normalizedQuery = String(query || '').normalize('NFKC').replace(/\s+/g, ' ').trim().toLocaleLowerCase()
  if (!normalizedQuery) return false
  const hashtag = normalizedQuery.startsWith('#')
  const needle = hashtag ? normalizedQuery.slice(1).trim() : normalizedQuery
  if (!needle) return false
  return tags.some(value => {
    const tag = String(value || '').normalize('NFKC').replace(/\s+/g, ' ').trim().toLocaleLowerCase()
    return hashtag ? tag === needle : tag.includes(needle)
  })
}

export function normalizeFollowupSuggestions(values) {
  if (!Array.isArray(values)) return []
  const seen = new Set()
  const output = []
  for (const value of values) {
    const question = String(value || '').replace(/\s+/g, ' ').trim()
    if (
      question.length < 8
      || question.length > 64
      || question.split(' ').length > 10
      || /^what does .+ say that most directly answers\b/i.test(question)
      || /^how does .+ compare with .+\bon\b/i.test(question)
      || /^(?:what (?:happened|changed)(?: immediately)? (?:before|after|next|as a result)(?: this)?|what (?:is|are) the (?:most )?(?:practical )?next steps?|what (?:trade[ -]?offs|context) (?:should we consider first|could change this conclusion)|which evidence (?:most )?directly supports this answer)\??$/i.test(question)
      || /LAVIX_FOLLOWUPS/i.test(question)
    ) continue
    const key = question.toLocaleLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    output.push(question)
    if (output.length === 2) break
  }
  return output
}

export function describeIndexingError(error, filename = 'File') {
  const raw = String(
    error?.data?.detail
      ?? error?.detail
      ?? error?.message
      ?? ''
  )
  const name = String(filename || 'File')
  if (/audio_video_disabled/i.test(raw)) {
    return `${name} attached — audio and video files aren't indexed.`
  }
  if (/unsupported_image_encoding/i.test(raw)) {
    return `${name} attached — this image format isn't supported for indexing.`
  }
  if (
    /no_parser_for_media_type|unsupported|415|not supported/i.test(raw)
    || /failed to grant access/i.test(raw)
  ) {
    return `${name} attached — AI indexing doesn't read this file type. Supported: PDF, DOCX, XLSX, PPTX, HTML, CSV, TXT and common images.`
  }
  return `${name} attached — indexing couldn't start right now. You can enable it later from the file menu.`
}

export function normalizeScopeIds(value) {
  if (!Array.isArray(value)) return []
  const output = []
  for (const item of value) {
    const id = typeof item === 'number' ? item : Number.parseInt(item, 10)
    if (!Number.isInteger(id) || id <= 0 || output.includes(id)) continue
    if (output.length >= 50) break
    output.push(id)
  }
  return output
}

export function resolveSendFileIds({ tagIds = null, sessionScope = [], scopeCleared = false } = {}) {
  const tags = normalizeScopeIds(tagIds)
  if (tags.length > 0) return tags
  if (scopeCleared && normalizeScopeIds(sessionScope).length > 0) return []
  return null
}

export function nextChatScope({ tagIds = null, sessionScope = [], scopeCleared = false } = {}) {
  const tags = normalizeScopeIds(tagIds)
  if (tags.length > 0) return tags
  if (scopeCleared) return []
  return normalizeScopeIds(sessionScope)
}

export function followupFileIds(sessionScope = []) {
  const scope = normalizeScopeIds(sessionScope)
  return scope.length > 0 ? scope : null
}

export function unionScopeIds(primary, secondary) {
  const output = normalizeScopeIds(primary)
  for (const id of normalizeScopeIds(secondary)) {
    if (!output.includes(id)) {
      if (output.length >= 50) break
      output.push(id)
    }
  }
  return output
}

const TAG_UNSUPPORTED_MIMES = [
  'application/zip',
  'application/x-zip-compressed',
  'application/x-rar-compressed',
  'application/x-7z-compressed',
  'application/x-tar',
  'application/gzip',
  'application/x-iso9660-image',
  'application/x-msdownload',
  'application/vnd.android.package-archive',
]

const TAG_UNSUPPORTED_EXTENSIONS = /\.(zip|rar|7z|tar|gz|bz2|xz|iso|dmg|exe|msi|apk|mp4|mov|avi|mkv|webm|mp3|wav|flac|ogg|m4a)$/i

export function classifyTagTarget(file = {}) {
  const name = file.original_filename || file.filename || 'File'
  const mime = String(file.mime_type || '').toLowerCase().split(';', 1)[0].trim()
  if (
    isIndexUnsupported(file)
    || TAG_UNSUPPORTED_MIMES.includes(mime)
    || TAG_UNSUPPORTED_EXTENSIONS.test(name)
  ) {
    return { verdict: 'unsupported', reason: `${name} can't be used — this file type isn't readable for AI search` }
  }
  if (!isIndexSearchable(file)) {
    return { verdict: 'unindexed', reason: `${name} isn't indexed yet — grant access when sending` }
  }
  return { verdict: 'ok', reason: '' }
}

// ── Bulk selection + folder paths (Documents toolbar, Upload/Move/Copy) ──

/**
 * Next selection after a bulk Move/Copy/Delete-style operation.
 * Full success (no failures) clears the selection so the caller can also
 * exit selection mode; any failure preserves the previous selection.
 * Returns a NEW Set; never mutates inputs.
 */
export function nextSelectionAfterBulkOp(previousIds, failedIds) {
  const failed = Array.isArray(failedIds) ? failedIds.filter(v => v !== undefined && v !== null) : []
  if (failed.length > 0) return new Set(previousIds || [])
  return new Set()
}

/**
 * Canonical absolute vault path for a folder id.
 * '/Root' for root (null/undefined/'') and for unknown ids (never invent).
 * Otherwise '/Root/<name>/.../<name>' walking parent_id with a cycle guard.
 */
export function folderAbsolutePath(folderId, folders) {
  if (folderId === null || folderId === undefined || folderId === '') return '/Root'
  const map = new Map((folders || []).map(f => [Number(f.id), f]))
  const parts = []
  const seen = new Set()
  let cur = map.get(Number(folderId))
  while (cur && !seen.has(Number(cur.id))) {
    seen.add(Number(cur.id))
    parts.unshift(String(cur.name ?? cur.id))
    const parent = cur.parent_id === null || cur.parent_id === undefined ? null : Number(cur.parent_id)
    cur = parent === null ? null : map.get(parent)
  }
  if (!parts.length) return '/Root'
  return `/Root/${parts.join('/')}`
}

import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { JSDOM } from 'jsdom'

import {
  aiBoxState,
  dedupeAboutMeSentences,
  describeIndexingError,
  folderAbsolutePath,
  getIndexState,
  getIndexStateLabel,
  getIndexProgress,
  getServiceHealthPresentation,
  isIndexActive,
  isIndexReady,
  isIndexSearchable,
  isIndexUnsupported,
  isPdfFile,
  documentTypeLabel,
  fileFormatLabel,
  matchesSemanticTags,
  mergeAdminAndAccountModelConfig,
  nextSelectionAfterBulkOp,
  normalizeMatchPercentage,
  normalizeActiveChatModelConfig,
  normalizeFollowupSuggestions,
  resolveMatchPercentage,
  resolveMemoryConsent,
} from './frontend-workflows.js'

let importCounter = 0
const freshImport = path => import(`${path}?test=${importCounter += 1}`)

function installBrowserMocks(fetchImpl = async () => new Response('{}', { status: 200 })) {
  const dom = new JSDOM('<!doctype html><html><head></head><body></body></html>', {
    url: 'http://lavix.test/',
  })
  const events = []
  const globals = {
    window: dom.window,
    document: dom.window.document,
    localStorage: dom.window.localStorage,
    Event: dom.window.Event,
    CustomEvent: dom.window.CustomEvent,
    StorageEvent: dom.window.StorageEvent,
  }
  for (const [key, value] of Object.entries(globals)) {
    Object.defineProperty(globalThis, key, { configurable: true, value, writable: true })
  }
  dom.window.fetch = fetchImpl
  globalThis.fetch = (...args) => dom.window.fetch(...args)
  dom.window.URL.createObjectURL = () => 'blob:download'
  dom.window.URL.revokeObjectURL = () => {}
  for (const type of ['auth-error', 'auth-refreshed', 'backend-error']) {
    dom.window.addEventListener(type, event => events.push(event))
  }
  return { dom, events }
}

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function streamResponse(chunks) {
  const encoder = new TextEncoder()
  return new Response(new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  }), { status: 200 })
}

function sourceBetween(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker)
  assert.notEqual(start, -1, `missing source marker: ${startMarker}`)
  const end = source.indexOf(endMarker, start + startMarker.length)
  assert.notEqual(end, -1, `missing source marker: ${endMarker}`)
  return source.slice(start, end)
}

test('production ingestion helpers preserve granular v2 states and labels', () => {
  assert.equal(getIndexState({ ingestion_state: 'chunking', ai_status: 'processing' }), 'chunking')
  assert.equal(getIndexState({ state: 'completed' }), 'ready')
  assert.equal(getIndexState({ status: 'not_supported' }), 'unsupported')
  assert.equal(getIndexState({}), 'not_granted')
  assert.equal(getIndexStateLabel({ state: 'password_required' }), 'Password required')
  assert.equal(isIndexReady({ state: 'ready' }), true)
  assert.equal(isIndexSearchable({ ingestion_state: 'embedding', ai_searchable: true }), true)
  const failedReplacement = { state: 'failed', embeddings_ready: true }
  assert.equal(isIndexReady(failedReplacement), false)
  assert.equal(isIndexActive(failedReplacement), false)
  assert.equal(isIndexSearchable(failedReplacement), true)
  assert.equal(isIndexSearchable({ state: 'ready', ai_searchable: false }), false)
  assert.equal(isIndexSearchable({ state: 'ready' }), true)
  assert.equal(isIndexActive({ state: 'embedding' }), true)
  assert.equal(isIndexUnsupported({ mime_type: 'video/mp4' }), true)
  assert.deepEqual(
    getIndexProgress({ total: 9, supported: 6, searchable: 3, active: 2 }),
    { searchable: 3, supported: 6, actionable: 6, allFilesTotal: 9, eligible: 6, indexable: 6, active: 2, percentage: 50 },
  )
  // Percentage base is `eligible`, so the gap always names its remainder.
  assert.deepEqual(
    getIndexProgress({ total: 9, supported: 6, indexable: 4, searchable: 3, active: 2 }),
    { searchable: 3, supported: 6, actionable: 6, allFilesTotal: 9, eligible: 4, indexable: 4, active: 2, percentage: 75 },
  )
  // Empty vault (or cold start with no stats) is 0%, never "ready".
  for (const coldStartValue of [null, undefined, false, '', []]) {
    assert.deepEqual(
      getIndexProgress(coldStartValue),
      { searchable: 0, supported: 0, actionable: 0, allFilesTotal: 0, eligible: 0, indexable: 0, active: 0, percentage: 0 },
    )
  }
  assert.deepEqual(
    getIndexProgress({ total: 0, all_files_total: 0, eligible: 0, searchable: 0, active: 0 }),
    { searchable: 0, supported: 0, actionable: 0, allFilesTotal: 0, eligible: 0, indexable: 0, active: 0, percentage: 0 },
  )
  // Partial progress uses the real indexed/total ratio.
  assert.equal(getIndexProgress({ eligible: 4, searchable: 1, all_files_total: 4 }).percentage, 25)
  // All done reads 100.
  assert.equal(getIndexProgress({ eligible: 3, searchable: 3, all_files_total: 3 }).percentage, 100)
  // All failed: 0 indexed of an eligible backlog is 0%, never 100%.
  assert.equal(getIndexProgress({ eligible: 2, searchable: 0, failed: 2, all_files_total: 2 }).percentage, 0)
  // Box fraction uses every file; numerator prefers ai_ready (beta parity).
  assert.deepEqual(
    getIndexProgress({ total: 104, supported: 100, actionable: 99, indexable: 100, searchable: 99, active: 0, needs_attention: 1 }),
    { searchable: 99, supported: 100, actionable: 99, allFilesTotal: 104, eligible: 100, indexable: 100, active: 0, percentage: 99 },
  )
  // ai_ready wins over searchable when both present (consent-less revision).
  // With no eligible backlog the box reads 0%, never "ready".
  assert.deepEqual(
    getIndexProgress({ all_files_total: 10, ai_ready: 5, searchable: 2, active: 0 }),
    { searchable: 5, supported: 5, actionable: 5, allFilesTotal: 10, eligible: 0, indexable: 5, active: 0, percentage: 0 },
  )
  // Legacy payload without `all_files_total` falls back to `total`.
  assert.deepEqual(
    getIndexProgress({ total: 9, supported: 4, searchable: 3, active: 0 }),
    { searchable: 3, supported: 4, actionable: 4, allFilesTotal: 9, eligible: 4, indexable: 4, active: 0, percentage: 75 },
  )
  // Eligible zero is 0% (nothing indexed), never NaN/Infinity.
  assert.equal(getIndexProgress({ supported: 0, actionable: 0, searchable: 0 }).percentage, 0)
  assert.equal(getIndexProgress({ supported: 5, actionable: 0, searchable: 0 }).percentage, 0)
  // Box states: empty vault, partial progress, complete, and all-failed.
  assert.deepEqual(aiBoxState(null), { empty: true, failed: false, progress: getIndexProgress(null) })
  assert.deepEqual(aiBoxState({ all_files_total: 0 }).empty, true)
  const partialBox = aiBoxState({ eligible: 4, searchable: 1, active: 1, all_files_total: 4 })
  assert.equal(partialBox.empty, false)
  assert.equal(partialBox.failed, false)
  assert.equal(partialBox.progress.percentage, 25)
  const doneBox = aiBoxState({ eligible: 3, searchable: 3, active: 0, all_files_total: 3 })
  assert.equal(doneBox.progress.percentage, 100)
  assert.equal(doneBox.failed, false)
  const failedBox = aiBoxState({ eligible: 2, searchable: 0, active: 0, failed: 2, all_files_total: 2 })
  assert.equal(failedBox.empty, false)
  assert.equal(failedBox.failed, true)
  assert.equal(failedBox.progress.percentage, 0)
  // In-flight failures are not the failed state yet.
  assert.equal(aiBoxState({ eligible: 2, searchable: 0, active: 1, failed: 1, all_files_total: 2 }).failed, false)
  // AI-Ready display rule: floor while incomplete, 100 only when complete.
  // Fractional progress arrives as integer counts (e.g. 999/1000 = 99.9).
  assert.equal(getIndexProgress({ eligible: 1000, indexable: 1000, searchable: 991 }).percentage, 99) // 99.1
  assert.equal(getIndexProgress({ eligible: 1000, indexable: 1000, searchable: 995 }).percentage, 99) // 99.5
  assert.equal(getIndexProgress({ eligible: 1000, indexable: 1000, searchable: 999 }).percentage, 99) // 99.9
  assert.equal(getIndexProgress({ eligible: 1000, indexable: 1000, searchable: 1000 }).percentage, 100) // complete
  assert.equal(getIndexProgress({ eligible: 250, indexable: 250, searchable: 127 }).percentage, 50) // 50.8
  assert.equal(getIndexProgress({ eligible: 500, indexable: 500, searchable: 377 }).percentage, 75) // 75.4
  assert.equal(getIndexProgress({ eligible: 1100, indexable: 1100, searchable: 1000 }).percentage, 90) // 90.9
  assert.equal(getIndexProgress({ eligible: 4, indexable: 4, searchable: 1 }).percentage, 25)
  assert.equal(getServiceHealthPresentation('ok', { status: 'ready', topology: 'private' }).tone, 'healthy')
  assert.equal(getServiceHealthPresentation('ok', { status: 'busy', topology: 'private' }).tone, 'warning')
  assert.equal(getServiceHealthPresentation('error', { status: 'degraded', topology: 'private' }).tone, 'error')
})

test('memory consent fails closed when compatibility and canonical values disagree', () => {
  assert.deepEqual(resolveMemoryConsent(false, false), { enabled: false, mismatch: false })
  assert.deepEqual(resolveMemoryConsent(true, true), { enabled: true, mismatch: false })
  assert.deepEqual(resolveMemoryConsent(true, false), { enabled: false, mismatch: true })
  assert.deepEqual(resolveMemoryConsent(false, true), { enabled: false, mismatch: true })
})

test('production model normalization uses the server preference and ignores stale browser routing', () => {
  assert.deepEqual(
    normalizeActiveChatModelConfig(
      { active_chat_model: 'local-a', allowed_chat_models: ['local-a', 'local-b'] },
      { selectedProvider: 'ollama', selectedModel: 'local-b' },
    ),
    { provider: 'ollama', model: 'local-a', available_models: ['local-a', 'local-b'] },
  )
  assert.deepEqual(
    normalizeActiveChatModelConfig(
      { model: 'local-a', allowed_chat_models: ['local-a'] },
      { selectedProvider: 'openrouter', selectedModel: 'cloud/model' },
    ),
    { provider: 'ollama', model: 'local-a', available_models: ['local-a'] },
  )
  assert.deepEqual(
    normalizeActiveChatModelConfig({ chat: { provider: 'ollama', model: 'local-b' }, allowed_chat_models: ['local-a', 'local-b'] }),
    { provider: 'ollama', model: 'local-b', available_models: ['local-a', 'local-b'] },
  )
})

test('admin model settings merge global controls with the authenticated account model', () => {
  const merged = mergeAdminAndAccountModelConfig(
    {
      revision: 7,
      installed_models: [{ name: 'local-a' }, { name: 'local-b' }],
      deployment_allowed_chat_models: ['local-a', 'local-b'],
      chat: {
        enabled: true,
        default_model: 'local-a',
        allowed_models: ['local-a', 'local-b'],
        available: true,
      },
      vision: { enabled: false, model: 'missing-vlm', available: false },
    },
    {
      revision: 7,
      provider: 'ollama',
      active_chat_model: 'local-b',
      preferred_chat_model: 'local-b',
      allowed_chat_models: ['local-a', 'local-b'],
      available_models: ['local-a', 'local-b'],
      chat: { model: 'local-b', available: true },
    },
  )

  assert.equal(merged.revision, 7)
  assert.equal(merged.chat.default_model, 'local-a')
  assert.deepEqual(merged.chat.allowed_models, ['local-a', 'local-b'])
  assert.equal(merged.chat.model, 'local-b')
  assert.equal(merged.chat.preferred_model, 'local-b')
  assert.equal(merged.chat.active_available, true)
  assert.equal(merged.active_chat_model, 'local-b')
  assert.equal(merged.preferred_chat_model, 'local-b')
  assert.deepEqual(merged.installed_models, [{ name: 'local-a' }, { name: 'local-b' }])
  assert.deepEqual(merged.vision, { enabled: false, model: 'missing-vlm', available: false })
})

test('match percentages and document type labels use the public display contract', () => {
  assert.equal(normalizeMatchPercentage(0), 0)
  assert.equal(normalizeMatchPercentage(1), 1)
  assert.equal(normalizeMatchPercentage(2), 2)
  assert.equal(normalizeMatchPercentage(63), 63)
  assert.equal(normalizeMatchPercentage(900), 100)
  assert.equal(normalizeMatchPercentage('not-a-score'), null)
  assert.equal(resolveMatchPercentage({ match_percentage: 1, score: 0.91 }), 1)
  assert.equal(resolveMatchPercentage({ score: 0.874 }), 87)
  assert.equal(resolveMatchPercentage({ relevance_score: 0.42 }), 42)
  assert.equal(documentTypeLabel('financial_report'), 'Financial Report')
  assert.equal(documentTypeLabel(''), '')
  assert.equal(fileFormatLabel({ filename: 'brief.pptx', mime_type: 'application/vnd.openxmlformats-officedocument.presentationml.presentation' }), 'PPTX')
  assert.equal(fileFormatLabel({ mime_type: 'application/pdf' }), 'PDF')
  assert.equal(fileFormatLabel({ mime_type: 'application/msword' }), 'DOC')
  assert.equal(fileFormatLabel({ mime_type: 'application/vnd.ms-excel' }), 'XLS')
  assert.equal(fileFormatLabel({ mime_type: 'application/vnd.ms-powerpoint' }), 'PPT')
  assert.equal(matchesSemanticTags(['cloud security', 'access controls'], '#cloud security'), true)
  assert.equal(matchesSemanticTags(['cloud security'], '#cloud'), false)
  assert.equal(matchesSemanticTags(['cloud security'], 'cloud'), true)
  assert.deepEqual(
    normalizeFollowupSuggestions([
      'What does A Very Long Report say that most directly answers this question?',
      'What changed after 1945?',
      'What changed after 1945?',
      'Which treaty shaped the relationship?',
      'How did every separate diplomatic compromise affect every country involved?',
      'Which later event mattered most?',
    ]),
    ['What changed after 1945?', 'Which treaty shaped the relationship?'],
  )
  assert.deepEqual(
    normalizeFollowupSuggestions([
      'What changed as a result?',
      'What happened immediately before this?',
      'Which evidence most directly supports this answer?',
      'What context could change this conclusion?',
      'Which treaty shaped the relationship?',
    ]),
    ['Which treaty shaped the relationship?'],
  )
  assert.deepEqual(
    normalizeFollowupSuggestions([
      'How did every separate diplomatic compromise affect every country involved?',
    ]),
    [],
  )
})

test('the shared PDF predicate recognizes MIME and every public filename field', () => {
  assert.equal(isPdfFile({ mime_type: 'application/pdf' }), true)
  assert.equal(isPdfFile({ mime_type: 'APPLICATION/PDF', filename: 'stored.bin' }), true)
  assert.equal(isPdfFile({ mime_type: 'application/octet-stream', filename: 'display.PDF' }), true)
  assert.equal(
    isPdfFile({
      mime_type: 'application/octet-stream',
      filename: 'opaque-storage-name',
      original_filename: 'CUSTOMER-INVOICE.PDF',
    }),
    true,
  )
  assert.equal(
    isPdfFile({
      mime_type: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
      filename: 'slides.pptx',
      original_filename: 'slides.PPTX',
    }),
    false,
  )
  assert.equal(isPdfFile({}), false)
})

test('frontend safety and availability controls stay explicit', () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  const client = readFileSync(new URL('./vault-client.js', import.meta.url), 'utf8')
  const workflow = readFileSync(new URL('./frontend-workflows.js', import.meta.url), 'utf8')
  const workflowImport = app.match(/import \{([^}]+)\} from '\.\/frontend-workflows\.js'/)
  const workflowImports = new Set((workflowImport?.[1] || '').split(',').map(value => value.trim()).filter(Boolean))
  const appWithoutWorkflowImport = workflowImport ? app.replace(workflowImport[0], '') : app

  for (const [, helper] of workflow.matchAll(/^export function (\w+)/gm)) {
    if (new RegExp(`\\b${helper}\\b`).test(appWithoutWorkflowImport)) {
      assert.ok(workflowImports.has(helper), `${helper} must be imported before App.jsx uses it`)
    }
  }

  const cancelPanel = app.match(/const handleCancelIndexingForPanel = async \(fileId\) => \{([\s\S]*?)\n            \};/)?.[1] || ''
  assert.match(cancelPanel, /await api\.cancelIndexing\(fileId\)/)
  assert.doesNotMatch(cancelPanel, /revokeAIAccess/)
  assert.match(app, /onCancelIndexing && onCancelIndexing\(file\.id\)/)
  assert.match(app, /title: 'Revoke AI access\?'/)
  assert.match(app, /onConfirm: \(close\) => revokeAccessForPanel\(fileId, close\)/)

  assert.match(app, /useState\(localStorage\.getItem\('fileSearch'\) !== 'false'\)/)
  assert.match(app, /setFileSearch\(true\);\s*localStorage\.setItem\('fileSearch', 'true'\)/)
  assert.match(app, /window\.setInterval\(loadSystem, 15000\)/)
  assert.match(app, /<button onClick=\{loadSystem\}[\s\S]*?>\s*Refresh now\s*<\/button>/)
  assert.doesNotMatch(app, /localStorage\.(?:getItem|setItem)\('selectedModel'/)
  assert.doesNotMatch(client, /disableAllEmbeddings/)
  assert.match(client, /updateChatModel: async \(model\)/)
  assert.match(app, /const \[adminConfig, accountConfig\] = await Promise\.all\(\[\s*api\.adminGetAiModels\(\),\s*api\.getModelConfig\('ollama'\)/)
  assert.ok((app.match(/mergeAdminAndAccountModelConfig\(adminConfig, accountConfig\)/g) || []).length >= 2)
  assert.match(app, /data-testid="admin-active-chat-model"/)
  // Collapsed card: preference/admin rows removed; active readout remains.
  assert.doesNotMatch(app, /data-testid="admin-preferred-chat-model"/)
  assert.match(app, /VLM · Vision/)
  assert.match(app, /Optional · Off/)
  assert.match(client, /clearGraphMemory: async \(\) => \{[\s\S]*?\/ai\/graph-memory[\s\S]*?method: 'DELETE'/)
  assert.match(app, /data-testid="graph-memory-clear-relationships"[\s\S]*?await api\.clearGraphMemory\(\)/)
  assert.match(app, /data-testid="graph-memory-clear"[\s\S]*?await api\.clearPersonalMemory\(\)/)
  assert.match(app, />\s*Clear Relationship Memory\s*<\/button>/)
  assert.match(app, />\s*Clear all personal memory\s*<\/button>/)
  assert.match(app, /data-testid="file-viewer-download-error"[\s\S]*?Retry download/)
  assert.match(app, /isImage && <img[\s\S]*?onError=\{handlePreviewError\}/)
  assert.match(app, /const supportsThumbnail = mvCanGenerateThumbnail\(file\)/)
  assert.doesNotMatch(app, /startsWith\('image\/'\)[\s\S]{0,160}invalidateMvFilePreview/)
  assert.match(app, /↑ \{m\.usage\.prompt_tokens/)
  assert.match(app, /className="no-scrollbar[^\"]*overflow-x-auto/)
  assert.match(app, /const keepActiveDocTypeVisible = React\.useCallback/)
  assert.match(app, /querySelector\('button\[aria-pressed="true"\]'\)/)
  assert.match(app, /window\.addEventListener\('resize', handleResize\)/)
  assert.match(app, /overflow-x-hidden bg="#000000"|overflow-x-hidden bg-\[#000000\]/)
  assert.match(app, /const historyLoadGenerationRef = React\.useRef\(0\)/)
  assert.match(app, /const skipHistoryLoadRef = React\.useRef\(null\)/)
  assert.match(app, /const activeChatSelectionRef = React\.useRef\(\{ sessionId, pgChatId \}\)/)
  assert.match(app, /if \(changed && processing\) abortRef\.current\?\.abort\(\)/)
  assert.match(app, /historyLoadGenerationRef\.current \+= 1;\s*setLoadingHistory\(false\)/)
  assert.match(app, /skippedSelection\.sessionId === sessionId[\s\S]*?skippedSelection\.pgChatId === pgChatId/)
  assert.match(app, /\}, \[sessionId, pgChatId, processing\]\)/)
  assert.match(app, /createdPgChat = newChat/)
  assert.match(app, /createdPgChat && selectionUnchanged/)
  assert.match(app, /onPgChatCreated\?\.\(createdPgChat\)/)
  assert.match(app, /skipHistoryLoadRef\.current = selectionAtSend;\s*\}\s*setProcessing\(false\)/)
  assert.doesNotMatch(app, /activeChatId = newChat\.id;\s*onPgChatCreated/)

  const trustedEvents = client.match(/const trustedEvents = \[([^\]]+)\]/)?.[1] || ''
  assert.match(trustedEvents, /'wheel'/)
  assert.doesNotMatch(trustedEvents, /'scroll'/)
})

test('dashboard bulk copy/move sends folder ids and refreshes the picker list', () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  const client = readFileSync(new URL('./vault-client.js', import.meta.url), 'utf8')
  // Client routes mirror the file router: move is PATCH with folder_id,
  // copy is POST with folder_id (null = root).
  assert.match(client, /moveFile: async \(fileId, folderId\) => \{[\s\S]*?\/files\/move\/\$\{fileId\}[\s\S]*?method: 'PATCH'[\s\S]*?folder_id: folderId/)
  assert.match(client, /copyFile: async \(fileId, folderId\) => \{[\s\S]*?\/files\/copy\/\$\{fileId\}[\s\S]*?method: 'POST'[\s\S]*?folder_id: folderId \?\? null/)
  // Destination picker offers root plus every folder by canonical path.
  assert.match(app, /\{\s*value: '', label: '\/Root'\s*\}, \.\.\.\(folders \|\| \[\]\)\.map\(f => \(\{\s*value: String\(f\.id\), label: folderAbsolutePath\(f\.id, folders\)\s*\}\)\)/)
  // Empty picker value means root; otherwise the numeric folder id.
  assert.match(app, /const fid = bulkMoveFolder === '' \? null : \+bulkMoveFolder/)
  assert.match(app, /const fid = val === '' \|\| val === null \? null : \+val/)
  // Per selected file, then a silent refresh so a just-created folder is
  // listed; selection clears only on full success.
  assert.match(app, /for \(const id of selFileIds\) \{\s*try \{ await api\.moveFile\(id, fid\); \}/)
  assert.match(app, /for \(const id of selFileIds\) \{\s*try \{ await api\.copyFile\(id, fid\); \}/)
  assert.match(app, /refreshFolders\(\); refresh\(\);/)
  assert.match(app, /setSelectMode\(false\)/)
  // Folder creation refreshes the same list the picker reads from.
  assert.match(app, /await api\.createFolder\(name, currentFolder \? currentFolder\.id : null\)/)
})

test('thumbnail failures fall back silently; per-file retry controls are gone and Settings owns regeneration', () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  const thumbnailGenerators = readFileSync(new URL('./thumbnail-generators.js', import.meta.url), 'utf8')
  const details = sourceBetween(app, 'function FileDetails(', 'function SkeletonFileCard(')
  const viewer = sourceBetween(app, 'function FileViewer(', '// ── LOGIN')
  const thumbnail = sourceBetween(app, 'function FileThumbnail(', '// Close "⋯ More" overflow dropdown on outside click')

  // Details, the full viewer, and list/grid eligibility must not drift into
  // separate filename/MIME interpretations again.
  assert.match(details, /const isPDF = !isFolder && isPdfFile\(file\)/)
  assert.match(viewer, /const isPDF = isPdfFile\(file\)/)
  assert.match(thumbnailGenerators, /import \{ isPdfFile \} from '\.\/frontend-workflows\.js'/)
  assert.match(thumbnailGenerators, /if \(isPdfFile\(file\)\) return \{ kind: 'pdf'/)
  assert.doesNotMatch(details, /endsWith\(['"]\.pdf/)
  assert.doesNotMatch(viewer, /endsWith\(['"]\.pdf/)

  // Per-file thumbnail recovery controls were removed on purpose: failed
  // thumbnails degrade to the type fallback icon and regeneration lives
  // only in Settings ("Regenerate all thumbnails").
  assert.doesNotMatch(thumbnail, /retryThumbnail|showThumbnailRetry|Try again|Try thumbnail again/)
  assert.doesNotMatch(app, /usePdfThumbnailRegeneration|PdfThumbnailRegenerateAction/)
  assert.doesNotMatch(app, /Regenerate thumbnail/)
  assert.match(thumbnail, /<FallbackIcon mimeType=\{file\.mime_type\} filename=\{file\.filename\} \/>/)

  // Settings keeps the single regeneration entry point.
  assert.match(app, /Regenerate all thumbnails/)
  assert.match(app, /window\.clearMvPreviewCaches\?\.\(\)/)

  assert.match(app, /<FileThumbnail file=\{file\} size="sm"/)
  assert.match(app, /<FileThumbnail file=\{file\} size="lg"/)
  assert.match(app, /aria-controls=\{thumbnailRegionId\}/)
  assert.match(app, /lavix-thumbnail-generated/)
})

test('Markdown rendering strips active content and normalizes safe links', async () => {
  installBrowserMocks()
  const { renderMarkdown, safeExternalUrl } = await freshImport('./vault-client.js')
  const html = renderMarkdown(`Answer 【12:3†source】

<script>alert(1)</script><img src=x onerror=alert(2)>

[bad](javascript:alert(3)) [data](data:text/html,x) [good](https://example.com/docs)

\`\`\`js
const x = 1
\`\`\``)

  assert.match(html, /<p>Answer\s*<\/p>/)
  assert.match(html, /class="hljs language-js"/)
  assert.match(html, /href="https:\/\/example\.com\/docs" target="_blank" rel="noopener noreferrer"/)
  assert.doesNotMatch(html, /script|img|onerror|javascript:|data:text|source/i)
  assert.equal(safeExternalUrl('https://example.com/a'), 'https://example.com/a')
  assert.equal(safeExternalUrl('javascript:alert(1)'), null)
  assert.equal(safeExternalUrl('/relative'), null)
})

test('assistant rendering hides protocol artifacts and keeps citations out of the message', async () => {
  installBrowserMocks()
  const { assistantDisplayText, renderAssistantMarkdown, stripAssistantProtocolArtifacts } = await freshImport('./vault-client.js')
  const raw = 'Supported answer [W1]. Unknown [V9].\n< LAVIX_FOLLOWUPS>{"followups":["internal"]}'
  const sources = [{ id: 'W1', kind: 'web', filename: 'Example' }]

  assert.equal(stripAssistantProtocolArtifacts(raw), 'Supported answer [W1]. Unknown [V9].')
  assert.equal(assistantDisplayText(raw, sources), 'Supported answer. Unknown.')
  const html = renderAssistantMarkdown(raw, sources, 'message-3-source')
  assert.match(html, /Supported answer\. Unknown\./)
  assert.doesNotMatch(html, /href=|W1|V9|LAVIX_FOLLOWUPS|internal/)

  const wrapped = '<LAVIX_ANSWER>Public message only.</LAVIX_ANSWER>'
  assert.equal(stripAssistantProtocolArtifacts(wrapped), 'Public message only.')
  assert.equal(stripAssistantProtocolArtifacts('Public message only.</LAVIX_ANSWER'), 'Public message only.')
  assert.equal(stripAssistantProtocolArtifacts('Public message only.\n\nanswer_id'), 'Public message only.')
  assert.equal(stripAssistantProtocolArtifacts('Public message only.\nanswer_id=12345'), 'Public message only.')
})

test('refresh rotation retries concurrent protected requests once and stores the rotated session', async () => {
  let refreshCalls = 0
  const calls = []
  const { events } = installBrowserMocks(async (url, options = {}) => {
    const path = new URL(String(url), 'http://lavix.test').pathname
    calls.push({ path, options })
    if (path === '/api/auth/refresh') {
      refreshCalls += 1
      await Promise.resolve()
      return jsonResponse({
        access_token: 'new-access',
        refresh_token: 'new-refresh',
        session_id: 'session-1',
        last_activity_at: '2026-07-15T10:00:00Z',
        idle_expires_at: '2026-07-15T12:00:00Z',
        idle_timeout_seconds: 7200,
      })
    }
    const authorization = new Headers(options.headers).get('Authorization')
    if (authorization !== 'Bearer new-access') return jsonResponse({ detail: 'expired' }, 401)
    if (path === '/api/files/list') return jsonResponse({ files: [{ id: 1 }] })
    if (path === '/api/folders') return jsonResponse([{ id: 2 }])
    return jsonResponse({})
  })
  localStorage.setItem('token', 'old-access')
  localStorage.setItem('refresh_token', 'old-refresh')
  localStorage.setItem('session_id', 'session-1')
  localStorage.setItem('username', 'ada')
  const { api } = await freshImport('./vault-client.js')

  const [files, folders] = await Promise.all([api.getFiles(), api.getFolders()])

  assert.deepEqual(files.files, [{ id: 1 }])
  assert.deepEqual(folders, [{ id: 2 }])
  assert.equal(refreshCalls, 1)
  assert.equal(localStorage.getItem('token'), 'new-access')
  assert.equal(localStorage.getItem('refresh_token'), 'new-refresh')
  assert.equal(localStorage.getItem('session_id'), 'session-1')
  assert.equal(localStorage.getItem('username'), 'ada')
  assert.equal(localStorage.getItem('last_activity_at'), '2026-07-15T10:00:00.000Z')
  assert.equal(localStorage.getItem('idle_expires_at'), '2026-07-15T12:00:00.000Z')
  assert.equal(localStorage.getItem('idle_timeout_seconds'), '7200')
  assert.equal(events.filter(event => event.type === 'auth-refreshed').length, 1)
  assert.equal(events.filter(event => event.type === 'auth-error').length, 0)
  assert.equal(calls.filter(call => call.path === '/api/auth/refresh').length, 1)
})

test('a 401 mid-upload refreshes once and retries the same file without loss', async () => {
  let refreshCalls = 0
  let uploadCalls = 0
  const { events } = installBrowserMocks(async (url, options = {}) => {
    const path = new URL(String(url), 'http://lavix.test').pathname
    if (path === '/api/auth/refresh') {
      refreshCalls += 1
      return jsonResponse({
        access_token: 'new-access',
        refresh_token: 'new-refresh',
        session_id: 'session-1',
        last_activity_at: '2026-07-15T10:00:00Z',
        idle_expires_at: '2026-07-15T12:00:00Z',
        idle_timeout_seconds: 7200,
      })
    }
    if (path === '/api/files/upload') {
      uploadCalls += 1
      const authorization = new Headers(options.headers).get('Authorization')
      if (authorization !== 'Bearer new-access') return jsonResponse({ detail: 'expired' }, 401)
      return jsonResponse({ file_id: 11, filename: 'kept.pdf' })
    }
    return jsonResponse({})
  })
  localStorage.setItem('token', 'old-access')
  localStorage.setItem('refresh_token', 'old-refresh')
  localStorage.setItem('session_id', 'session-1')
  const { api } = await freshImport('./vault-client.js')

  const result = await api.uploadFile(new File(['x'], 'kept.pdf'), false, 7)

  assert.equal(result.file_id, 11)
  assert.equal(uploadCalls, 2)
  assert.equal(refreshCalls, 1)
  assert.equal(events.filter(event => event.type === 'auth-error').length, 0)
})

test('an idle-timeout 401 bypasses refresh, preserves the username, and emits structured reauth state', async () => {
  let refreshCalls = 0
  const { events } = installBrowserMocks(async url => {
    const path = new URL(String(url), 'http://lavix.test').pathname
    if (path === '/api/auth/refresh') refreshCalls += 1
    return jsonResponse({
      detail: {
        code: 'session_idle_timeout',
        message: 'Session locked after 120 minutes of inactivity',
        idle_timeout_seconds: 7200,
      },
    }, 401)
  })
  for (const [key, value] of Object.entries({
    token: 'old-access',
    refresh_token: 'old-refresh',
    session_id: 'session-1',
    username: 'ada',
  })) localStorage.setItem(key, value)
  const { api } = await freshImport('./vault-client.js')

  await assert.rejects(() => api.getFiles(), /Failed to load files/)

  assert.equal(refreshCalls, 0)
  assert.equal(localStorage.getItem('token'), null)
  assert.equal(localStorage.getItem('refresh_token'), null)
  assert.equal(localStorage.getItem('username'), 'ada')
  const authEvent = events.find(event => event.type === 'auth-error')
  assert.equal(authEvent.detail.code, 'session_idle_timeout')
  assert.equal(authEvent.detail.idle_timeout_seconds, 7200)
})

test('idle timeout returned by refresh is inspected before retrying the protected request', async () => {
  let protectedCalls = 0
  let refreshCalls = 0
  const { events } = installBrowserMocks(async url => {
    const path = new URL(String(url), 'http://lavix.test').pathname
    if (path === '/api/auth/refresh') {
      refreshCalls += 1
      return jsonResponse({
        detail: {
          code: 'session_idle_timeout',
          message: 'Session locked after 120 minutes of inactivity',
          idle_timeout_seconds: 7200,
        },
      }, 401)
    }
    protectedCalls += 1
    return jsonResponse({ detail: 'Access token expired' }, 401)
  })
  for (const [key, value] of Object.entries({
    token: 'old-access', refresh_token: 'old-refresh', session_id: 'session-1', username: 'ada',
  })) localStorage.setItem(key, value)
  const { api } = await freshImport('./vault-client.js')

  await assert.rejects(() => api.getFiles(), /Failed to load files/)

  assert.equal(refreshCalls, 1)
  assert.equal(protectedCalls, 1)
  assert.equal(localStorage.getItem('username'), 'ada')
  assert.equal(events.filter(event => event.type === 'auth-error').length, 1)
  assert.equal(events.find(event => event.type === 'auth-error').detail.code, 'session_idle_timeout')
})

test('cross-tab refresh locking reuses a session another tab already rotated', async () => {
  let refreshCalls = 0
  installBrowserMocks(async (url, options = {}) => {
    const path = new URL(String(url), 'http://lavix.test').pathname
    if (path === '/api/auth/refresh') {
      refreshCalls += 1
      return jsonResponse({}, 500)
    }
    const authorization = new Headers(options.headers).get('Authorization')
    return authorization === 'Bearer cross-tab-access'
      ? jsonResponse({ files: [] })
      : jsonResponse({ detail: 'expired' }, 401)
  })
  localStorage.setItem('token', 'old-access')
  localStorage.setItem('refresh_token', 'old-refresh')
  localStorage.setItem('session_id', 'session-1')
  Object.defineProperty(window.navigator, 'locks', {
    configurable: true,
    value: {
      request: async (_name, callback) => {
        localStorage.setItem('token', 'cross-tab-access')
        localStorage.setItem('refresh_token', 'cross-tab-refresh')
        return callback()
      },
    },
  })
  const { api } = await freshImport('./vault-client.js')

  assert.deepEqual(await api.getFiles(), { files: [] })
  assert.equal(refreshCalls, 0)
  assert.equal(localStorage.getItem('refresh_token'), 'cross-tab-refresh')
})

test('failed refresh clears all auth state and emits one terminal auth event', async () => {
  const { events } = installBrowserMocks(async url => {
    const path = new URL(String(url), 'http://lavix.test').pathname
    if (path === '/api/auth/refresh') return jsonResponse({ detail: 'invalid refresh' }, 401)
    return jsonResponse({ detail: 'expired' }, 401)
  })
  for (const [key, value] of Object.entries({
    token: 'old-access',
    refresh_token: 'old-refresh',
    session_id: 'session-1',
    username: 'ada',
  })) localStorage.setItem(key, value)
  const { api } = await freshImport('./vault-client.js')

  await assert.rejects(() => api.getFiles(), /Failed to load files/)

  for (const key of ['token', 'refresh_token', 'session_id', 'username']) {
    assert.equal(localStorage.getItem(key), null)
  }
  assert.equal(events.filter(event => event.type === 'auth-error').length, 1)
})

test('public login failures do not rotate or erase an existing session', async () => {
  let refreshCalls = 0
  const { events } = installBrowserMocks(async url => {
    const path = new URL(String(url), 'http://lavix.test').pathname
    if (path === '/api/auth/refresh') refreshCalls += 1
    return jsonResponse({ detail: 'Incorrect username or password' }, 401)
  })
  localStorage.setItem('token', 'existing-access')
  localStorage.setItem('refresh_token', 'existing-refresh')
  localStorage.setItem('session_id', 'existing-session')
  const { api } = await freshImport('./vault-client.js')

  assert.deepEqual(await api.login('ada', 'wrong'), { detail: 'Incorrect username or password' })
  assert.equal(refreshCalls, 0)
  assert.equal(localStorage.getItem('token'), 'existing-access')
  assert.equal(events.length, 0)
})

test('session helpers require complete token triples and server logout clears local identity', async () => {
  let logoutAuthorization = null
  installBrowserMocks(async (url, options = {}) => {
    const path = new URL(String(url), 'http://lavix.test').pathname
    if (path === '/api/auth/logout') {
      logoutAuthorization = new Headers(options.headers).get('Authorization')
      return jsonResponse({ message: 'Successfully logged out from session' })
    }
    return jsonResponse({})
  })
  const { api, storeAuthSession } = await freshImport('./vault-client.js')
  assert.throws(() => storeAuthSession({ access_token: 'incomplete' }), /Missing refresh_token/)
  storeAuthSession({ access_token: 'access', refresh_token: 'refresh', session_id: 'session' }, ' ada ')

  await api.logout()

  assert.equal(logoutAuthorization, 'Bearer access')
  for (const key of ['token', 'refresh_token', 'session_id', 'username']) {
    assert.equal(localStorage.getItem(key), null)
  }
})

test('visible trusted interaction alone records activity and stores the server deadline', async () => {
  const activityCalls = []
  installBrowserMocks(async (url, options = {}) => {
    const path = new URL(String(url), 'http://lavix.test').pathname
    if (path === '/api/auth/activity') {
      activityCalls.push(options)
      return jsonResponse({
        last_activity_at: '2026-07-15T11:00:00Z',
        idle_expires_at: '2026-07-15T13:00:00Z',
        idle_timeout_seconds: 7200,
      })
    }
    return jsonResponse({})
  })
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' })
  const { installSessionActivityTracking, storeAuthSession } = await freshImport('./vault-client.js')
  storeAuthSession({
    access_token: 'access',
    refresh_token: 'refresh',
    session_id: 'session',
    last_activity_at: '2026-07-15T10:00:00Z',
    idle_expires_at: '2026-07-15T12:00:00Z',
    idle_timeout_seconds: 7200,
  }, 'ada')
  const tracker = installSessionActivityTracking({
    heartbeatThrottleMs: 0,
    now: () => Date.parse('2026-07-15T10:30:00Z'),
  })

  await tracker.recordTrustedActivity({ isTrusted: false })
  assert.equal(activityCalls.length, 0)
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' })
  await tracker.recordTrustedActivity({ isTrusted: true })
  assert.equal(activityCalls.length, 0)
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' })
  await fetch('/api/health')
  assert.equal(activityCalls.length, 0)
  await tracker.recordTrustedActivity({ isTrusted: true })

  assert.equal(activityCalls.length, 1)
  assert.equal(new Headers(activityCalls[0].headers).get('Authorization'), 'Bearer access')
  assert.equal(localStorage.getItem('last_activity_at'), '2026-07-15T11:00:00.000Z')
  assert.equal(localStorage.getItem('idle_expires_at'), '2026-07-15T13:00:00.000Z')
  tracker.dispose()
})

test('session timeout broadcasts lock another tab while preserving its reauthentication username', async () => {
  const { events } = installBrowserMocks()
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' })
  const { AUTH_SESSION_BROADCAST_KEY, installSessionActivityTracking, storeAuthSession } = await freshImport('./vault-client.js')
  storeAuthSession({
    access_token: 'access',
    refresh_token: 'refresh',
    session_id: 'session',
    last_activity_at: '2026-07-15T10:00:00Z',
    idle_expires_at: '2026-07-15T12:00:00Z',
    idle_timeout_seconds: 7200,
  }, 'ada')
  assert.equal(localStorage.getItem('token'), 'access')
  const tracker = installSessionActivityTracking({
    now: () => Date.parse('2026-07-15T10:30:00Z'),
  })

  window.dispatchEvent(new StorageEvent('storage', {
    key: AUTH_SESSION_BROADCAST_KEY,
    newValue: JSON.stringify({
      id: 'other-tab',
      detail: { code: 'session_idle_timeout', idle_timeout_seconds: 7200 },
    }),
  }))

  assert.equal(localStorage.getItem('token'), null)
  assert.equal(localStorage.getItem('username'), 'ada')
  assert.equal(events.filter(event => event.type === 'auth-error').length, 1)
  assert.equal(events.find(event => event.type === 'auth-error').detail.code, 'session_idle_timeout')
  tracker.dispose()
})

test('the local lock timer follows the server deadline instead of assuming a fixed 120 minutes', async () => {
  const { events } = installBrowserMocks()
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' })
  const timers = new Map()
  let nextTimer = 1
  let now = 1_000
  const { installSessionActivityTracking, storeAuthSession } = await freshImport('./vault-client.js')
  storeAuthSession({
    access_token: 'access',
    refresh_token: 'refresh',
    session_id: 'session',
    last_activity_at: new Date(2_000).toISOString(),
    idle_expires_at: new Date(5_000).toISOString(),
    idle_timeout_seconds: 3,
  }, 'ada')
  const tracker = installSessionActivityTracking({
    now: () => now,
    setTimeoutFn: (callback, delay) => {
      const id = nextTimer++
      timers.set(id, { callback, delay })
      return id
    },
    clearTimeoutFn: id => timers.delete(id),
  })

  const deadline = [...timers.values()][0]
  assert.equal(deadline.delay, 4_000)
  now = 5_000
  deadline.callback()

  assert.equal(localStorage.getItem('token'), null)
  assert.equal(localStorage.getItem('username'), 'ada')
  assert.equal(events.find(event => event.type === 'auth-error').detail.idle_timeout_seconds, 3)
  tracker.dispose()
})

test('API validation errors use the sanitized location/message contract', async () => {
  installBrowserMocks()
  const { formatApiError } = await freshImport('./vault-client.js')
  assert.equal(
    formatApiError({ detail: [
      { location: ['body', 'email'], message: 'Invalid email', type: 'value_error' },
      { location: ['body', 'password'], message: 'Too short', type: 'value_error' },
    ] }),
    'email: Invalid email | password: Too short',
  )
})

test('the real client sends authenticated upload and Ollama SSE chat contracts', async () => {
  const calls = []
  installBrowserMocks(async (url, options = {}) => {
    calls.push({ url: String(url), options })
    if (String(url).endsWith('/api/files/upload')) return jsonResponse({ file_id: 44, filename: 'a.pdf' })
    if (String(url).endsWith('/api/ai/search')) return jsonResponse({ results: [{ file_id: 1, match_percentage: 88 }] })
    if (String(url).endsWith('/api/files/cancel-indexing')) return jsonResponse({ cancelled_jobs: 2, removed_chunks: 0 })
    if (String(url).endsWith('/api/files/cancel-indexing/44')) return jsonResponse({ file_id: 44, cancelled_jobs: 1, index_data_preserved: true })
    if (String(url).endsWith('/api/auth/chat-model')) return jsonResponse({ preferred_chat_model: 'local-b', active_chat_model: 'local-b' })
    if (String(url).endsWith('/api/ai/chat')) {
      return streamResponse([
        'data: {"type":"status","step":"searching_vault"}\n',
        'data: {"type":"token","content":"Hi"}\n',
        'data: {"type":"sources","sources":[{"file_id":1}]}\n',
        'data: [DONE]\n',
      ])
    }
    return jsonResponse({})
  })
  localStorage.setItem('token', 'secret-token')
  localStorage.setItem('selectedProvider', 'ollama')
  localStorage.setItem('selectedModel', 'llama')
  const { api } = await freshImport('./vault-client.js')

  assert.equal((await api.uploadFile(new File(['x'], 'a.pdf'), false, 7)).file_id, 44)
  assert.equal((await api.searchVault('invoice', 5)).results[0].match_percentage, 88)
  assert.equal((await api.cancelIndexing()).removed_chunks, 0)
  assert.equal((await api.cancelIndexing(44)).index_data_preserved, true)
  assert.equal((await api.updateChatModel('local-b')).active_chat_model, 'local-b')
  const seen = { tokens: '', statuses: [], sources: [] }
  const answer = await api.chat({
    message: 'hello',
    fileIds: [1],
    deepSearch: true,
    webSearch: false,
    personaPrompt: 'persona',
    onToken: token => { seen.tokens += token },
    onStatus: status => seen.statuses.push(status.step),
    onSources: sources => { seen.sources = sources.sources },
  })

  assert.equal(calls[0].options.body.get('folder_id'), '7')
  assert.equal(new Headers(calls[0].options.headers).get('Authorization'), 'Bearer secret-token')
  assert.deepEqual(JSON.parse(calls[1].options.body), { query: 'invoice', max_results: 5 })
  assert.equal(calls[2].options.method, 'POST')
  assert.match(calls[3].url, /\/api\/files\/cancel-indexing\/44$/)
  const updateModelCall = calls.find(call => call.url.endsWith('/api/auth/chat-model'))
  assert.equal(updateModelCall.options.method, 'PUT')
  assert.deepEqual(JSON.parse(updateModelCall.options.body), { model: 'local-b' })
  const chatCall = calls.find(call => call.url.endsWith('/api/ai/chat'))
  const chatBody = JSON.parse(chatCall.options.body)
  assert.deepEqual(chatBody.file_ids, [1])
  assert.equal(chatBody.provider, 'ollama')
  assert.equal('model' in chatBody, false)
  assert.equal(chatBody.deep_search, true)
  assert.equal(chatBody.web_search_enabled, false)
  assert.equal('confirm_search' in chatBody, false)
  assert.equal('session_id' in chatBody, false)
  assert.equal(seen.tokens, 'Hi')
  assert.deepEqual(seen.statuses, ['searching_vault'])
  assert.deepEqual(seen.sources, [{ file_id: 1 }])
  assert.deepEqual(answer, { answer: 'Hi' })
})

test('the chat stream surfaces a clarification turn terminally', async () => {
  installBrowserMocks(async (url, options = {}) => {
    if (String(url).endsWith('/api/ai/chat')) {
      return streamResponse([
        'data: {"type":"status","step":"searching_web"}\n',
        'data: {"type":"clarification","question":"Just to confirm — by \'his\' do you mean Don Bradman?"}\n',
      ])
    }
    return jsonResponse({})
  })
  localStorage.setItem('token', 'secret-token')
  const { api } = await freshImport('./vault-client.js')

  const seen = { statuses: [], clarification: null }
  const result = await api.chat({
    message: 'his total centuries?',
    onStatus: status => seen.statuses.push(status.step),
    onClarification: event => { seen.clarification = event.question },
  })

  assert.deepEqual(seen.statuses, ['searching_web'])
  assert.equal(seen.clarification, "Just to confirm — by 'his' do you mean Don Bradman?")
  assert.deepEqual(result, {
    clarification: { type: 'clarification', question: "Just to confirm — by 'his' do you mean Don Bradman?" },
  })
})

test('the real client manages only explicit authenticated memory items', async () => {
  const calls = []
  installBrowserMocks(async (url, options = {}) => {
    const path = new URL(String(url), 'http://lavix.test').pathname
    calls.push({ path, options })
    if (options.method === 'POST') return jsonResponse({ item: { id: 7, text: 'Use metric units.' } }, 201)
    if (options.method === 'DELETE') return jsonResponse(path.endsWith('/7') ? { deleted_id: 7 } : { deleted_count: 1 })
    return jsonResponse({ enabled: false, items: [], max_items: 20, max_text_length: 500 })
  })
  localStorage.setItem('token', 'secret-token')
  const { api } = await freshImport('./vault-client.js')

  assert.equal((await api.getMemory()).enabled, false)
  assert.equal((await api.addMemory('Use metric units.')).item.id, 7)
  assert.equal((await api.deleteMemoryItem(7)).deleted_id, 7)

  assert.deepEqual(calls.map(call => [call.options.method || 'GET', call.path]), [
    ['GET', '/api/ai/memory'],
    ['POST', '/api/ai/memory'],
    ['DELETE', '/api/ai/memory/7'],
  ])
  assert.deepEqual(JSON.parse(calls[1].options.body), { text: 'Use metric units.' })
  assert.equal(new Headers(calls[2].options.headers).get('Authorization'), 'Bearer secret-token')
})

test('graph memory client keeps settings atomic and traverses every item page', async () => {
  const calls = []
  installBrowserMocks(async (url, options = {}) => {
    const requestUrl = new URL(String(url), 'http://lavix.test')
    calls.push({ requestUrl, options })
    if (requestUrl.pathname === '/api/ai/graph-memory/items' && !options.method) {
      const offset = Number(requestUrl.searchParams.get('offset'))
      const limit = Number(requestUrl.searchParams.get('limit'))
      const total = 205
      return jsonResponse({
        items: Array.from({ length: Math.min(limit, total - offset) }, (_, index) => ({ id: `memory-${offset + index}` })),
        total,
        limit,
        offset,
      })
    }
    if (requestUrl.pathname === '/api/ai/graph-memory' && options.method === 'PUT') {
      return jsonResponse({ enabled: true, retention_days: 90, revision: 8 })
    }
    if (requestUrl.pathname === '/api/ai/graph-memory/items/memory-7' && options.method === 'PATCH') {
      return jsonResponse({ item: { id: 'memory-7', revision: 4 } })
    }
    if (options.method === 'POST') return jsonResponse({ item: { id: 'memory-7' } })
    if (options.method === 'DELETE') return jsonResponse({ deleted_id: 'memory-7', purge_state: 'ready' })
    return jsonResponse({})
  })
  localStorage.setItem('token', 'secret-token')
  const { api } = await freshImport('./vault-client.js')

  const updated = await api.updateGraphMemory({ enabled: true, retentionDays: 90, expectedRevision: 7 })
  const page = await api.getAllGraphMemoryItems({ status: 'active', kind: 'relationship' })
  const edited = await api.editGraphMemoryItem('memory-7', {
    subject: 'I',
    predicate: 'prefer',
    objectValue: 'concise answers',
    expectedRevision: 3,
  })
  await api.approveGraphMemoryItem('memory-7')
  await api.renewGraphMemoryItem('memory-7')
  await api.deleteGraphMemoryItem('memory-7')
  await api.clearPersonalMemory()

  assert.equal(updated.revision, 8)
  assert.equal(page.items.length, 205)
  assert.equal(page.items.at(-1).id, 'memory-204')
  assert.equal(edited.item.revision, 4)
  const settingsCalls = calls.filter(call => call.requestUrl.pathname === '/api/ai/graph-memory' && call.options.method === 'PUT')
  assert.equal(settingsCalls.length, 1)
  assert.deepEqual(JSON.parse(settingsCalls[0].options.body), {
    enabled: true,
    retention_days: 90,
    expected_revision: 7,
  })
  const listCalls = calls.filter(call => call.requestUrl.pathname === '/api/ai/graph-memory/items' && !call.options.method)
  assert.deepEqual(listCalls.map(call => Number(call.requestUrl.searchParams.get('offset'))), [0, 100, 200])
  assert.ok(listCalls.every(call => call.requestUrl.searchParams.get('status') === 'active'))
  assert.ok(listCalls.every(call => call.requestUrl.searchParams.get('kind') === 'relationship'))
  const editCall = calls.find(call => call.options.method === 'PATCH')
  assert.deepEqual(JSON.parse(editCall.options.body), {
    subject: 'I',
    predicate: 'prefer',
    object_value: 'concise answers',
    expected_revision: 3,
  })
  assert.equal(new Headers(editCall.options.headers).get('Authorization'), 'Bearer secret-token')
  assert.ok(calls.some(call => call.requestUrl.pathname.endsWith('/approve') && call.options.method === 'POST'))
  assert.ok(calls.some(call => call.requestUrl.pathname.endsWith('/renew') && call.options.method === 'POST'))
  assert.ok(calls.some(call => call.requestUrl.pathname === '/api/ai/personal-memory' && call.options.method === 'DELETE'))
})

test('About Me sentences render once while preserving distinct facts and every backing memory', () => {
  const original = [
    { text: ' You work on CANVAS-4821   browser acceptance. ', backing_memory_ids: ['memory-a'] },
    { text: 'you work on canvas-4821 browser acceptance.', backing_memory_ids: ['memory-a', 'memory-b'] },
    { text: 'YOU WORK ON CANVAS-4821 BROWSER ACCEPTANCE!', backing_memory_ids: ['memory-e'] },
    { text: 'You prefer concise answers.', backing_memory_ids: ['memory-c'] },
    { text: 'You prefer detailed answers.', backing_memory_ids: ['memory-d'] },
    { text: '   ', backing_memory_ids: ['memory-empty'] },
  ]

  assert.deepEqual(dedupeAboutMeSentences(original), [
    {
      text: 'You work on CANVAS-4821 browser acceptance.',
      backing_memory_ids: ['memory-a', 'memory-b', 'memory-e'],
    },
    { text: 'You prefer concise answers.', backing_memory_ids: ['memory-c'] },
    { text: 'You prefer detailed answers.', backing_memory_ids: ['memory-d'] },
  ])
  assert.equal(original[0].text, ' You work on CANVAS-4821   browser acceptance. ')
  assert.deepEqual(dedupeAboutMeSentences(null), [])
})

test('AI Memory settings expose grounded, revision-safe graph controls without truncation', () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  const toggleMemory = app.match(/const toggleMemory = async \(\) => \{([\s\S]*?)\n            \};/)?.[1] || ''

  assert.match(toggleMemory, /api\.updateGraphMemory\(/)
  assert.equal((toggleMemory.match(/api\.updateGraphMemory\(/g) || []).length, 1)
  assert.doesNotMatch(toggleMemory, /api\.setMemoryEnabled\(/)
  assert.match(app, /resolveMemoryConsent\(data\.enabled, graph\.enabled\)/)
  assert.match(app, /setMemoryEnabled\(consent\.enabled\)/)
  assert.match(app, /data-testid="graph-memory-consent-mismatch"/)
  assert.match(app, /aria-pressed=\{memoryEnabled\}/)
  assert.match(app, /api\.getAllGraphMemoryItems\(\)/)
  assert.match(app, /api\.editGraphMemoryItem\(item\.id/)
  assert.match(app, /expectedRevision: item\.revision/)
  // About Me renders as a single grounded summary paragraph with a copy action.
  assert.match(app, /data-testid="graph-memory-about-me"/)
  assert.match(app, /graphMemory\.about_me\?\.summary \|\| 'No relationship memories yet\.'/)
  assert.match(app, /title=\{summaryCopied \? 'Copied!' : 'Copy summary'\}/)
  assert.match(app, /summaryCopied \? <Icons\.Check/)
  assert.match(app, /setSummaryCopied\(true\)/)
  assert.doesNotMatch(app, /about_me\?\.sentences/)
  assert.doesNotMatch(app, /backing_memory_ids/)
  assert.match(app, /item\.source_excerpt/)
  assert.match(app, /item\.projection_state/)
  assert.match(app, /Sensitive or uncertain statements are rejected\./)
  assert.match(app, /Eligible, non-sensitive lower-confidence candidates wait 14 days for review\./)
  assert.doesNotMatch(app, /Sensitive or uncertain candidates wait 14 days/)
  // The memory panel must render every backing memory: no list truncation.
  // Scoped to the memory tab (upload-queue caps elsewhere are intentional).
  const memoryPanel = app.slice(
    app.indexOf('Safe Automatic memory'),
    app.indexOf('── INTERFACE TAB ──'),
  )
  assert.ok(memoryPanel.length > 0, 'memory panel anchors must exist in App.jsx')
  assert.doesNotMatch(memoryPanel, /\.slice\(0,\s*12\)/)
  for (const days of [30, 90, 365]) assert.match(app, new RegExp(`\\{ value: '${days}', label: '${days} days' \\}`))
})

test('admin Preferences section exposes session timeout and registration toggle', () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  const client = readFileSync(new URL('./vault-client.js', import.meta.url), 'utf8')
  // Admin-only tab with a non-admin guard.
  assert.match(app, /\{ id: 'admin_preferences', label: 'Preferences'/)
  assert.match(app, /tab === 'admin_preferences' && userProfile\?\.is_admin/)
  assert.match(app, /\(tab === 'api_keys' \|\| tab === 'admin_preferences'\) && !userProfile\?\.is_admin/)
  // Panel controls with stable test hooks.
  assert.match(app, /data-testid="admin-preferences"/)
  assert.match(app, /testid="admin-prefs-timeout"/)
  assert.match(app, /data-testid="admin-prefs-registration"/)
  assert.match(app, /data-testid="save-admin-preferences"/)
  assert.match(app, /api\.adminGetPreferences\(\)/)
  assert.match(app, /api\.adminUpdatePreferences\(\{/)
  // All seven timeout choices are offered.
  for (const minutes of [15, 30, 60, 120, 240, 480, 1440]) {
    assert.match(app, new RegExp(`\\{ value: '${minutes}'`))
  }
  // Client hits the admin endpoints.
  assert.match(client, /adminGetPreferences: async \(\) => \{[\s\S]*?\/admin\/preferences/)
  assert.match(client, /adminUpdatePreferences: async \(preferences\) => \{[\s\S]*?\/admin\/preferences[\s\S]*?method: 'PUT'/)
  // Login keeps following the capabilities toggle.
  assert.match(app, /authCapabilities\.registration_enabled && \(/)
})

test('locally bundled spreadsheet preview parses a workbook without runtime scripts', async () => {
  const XLSX = await import('@e965/xlsx')
  const workbook = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(workbook, XLSX.utils.aoa_to_sheet([['Name', 'Value'], ['alpha', 1]]), 'Sheet1')
  const bytes = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' })
  const { dom } = installBrowserMocks(async () => new Response(bytes, { status: 200 }))
  const originalCreateElement = dom.window.document.createElement.bind(dom.window.document)
  dom.window.document.createElement = tagName => {
    if (tagName !== 'canvas') return originalCreateElement(tagName)
    return {
      width: 0,
      height: 0,
      getContext: () => ({
        beginPath() {}, fillRect() {}, fillText() {}, lineTo() {}, moveTo() {}, stroke() {}, strokeRect() {},
        measureText: value => ({ width: String(value).length * 4 }),
      }),
      toDataURL: () => 'data:image/jpeg;base64,preview',
    }
  }
  const { mvCanGenerateThumbnail, mvGenerateThumbnail } = await freshImport('./thumbnail-generators.js')

  assert.equal(
    await mvGenerateThumbnail({ mime_type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename: 'a.xlsx' }, '/preview'),
    'data:image/jpeg;base64,preview',
  )
  assert.equal(await mvGenerateThumbnail({ mime_type: 'image/png', filename: 'a.png' }, '/image'), '/image')
  assert.equal(mvCanGenerateThumbnail({ mime_type: 'application/octet-stream', filename: 'UPPER.PDF' }), true)
  assert.equal(
    mvCanGenerateThumbnail({
      mime_type: 'application/octet-stream',
      filename: 'opaque-storage-name',
      original_filename: 'UPPER-ORIGINAL.PDF',
    }),
    true,
  )
  assert.equal(mvCanGenerateThumbnail({ mime_type: 'application/vnd.ms-powerpoint', filename: 'slides.ppt' }), false)
  assert.equal(mvCanGenerateThumbnail({ mime_type: 'video/mp4', filename: 'a.mp4' }), false)
  assert.equal(mvCanGenerateThumbnail({ mime_type: 'text/plain', filename: 'huge.txt', size_bytes: 31 * 1024 * 1024 }), false)
  await assert.rejects(() => mvGenerateThumbnail({ mime_type: 'video/mp4', filename: 'a.mp4' }, '/preview'), /unsupported/)
})

test('large PDFs use bounded range loading while small previews stay buffered', async () => {
  installBrowserMocks()
  const { mvPdfLoadMode } = await freshImport('./thumbnail-generators.js')

  assert.equal(mvPdfLoadMode(5 * 1024 * 1024), 'buffer')
  assert.equal(mvPdfLoadMode(79 * 1024 * 1024), 'range')
})

test('PDF cleanup supports document proxies that expose teardown through their loading task', async () => {
  installBrowserMocks()
  const { mvDestroyPdfDocument } = await freshImport('./thumbnail-generators.js')
  let loadingTaskDestroyCalls = 0
  let legacyDestroyCalls = 0

  await mvDestroyPdfDocument({
    loadingTask: { destroy: async () => { loadingTaskDestroyCalls += 1 } },
    cleanup: async () => { throw new Error('loading-task cleanup must be preferred') },
  })
  await mvDestroyPdfDocument({ destroy: async () => { legacyDestroyCalls += 1 } })

  assert.equal(loadingTaskDestroyCalls, 1)
  assert.equal(legacyDestroyCalls, 1)
})

test('preview sessions deduplicate requests and refresh before their server expiry', async () => {
  installBrowserMocks()
  const { clearMvPreviewCaches, mvGetPreviewSession } = await freshImport('./thumbnail-generators.js')
  clearMvPreviewCaches()
  let calls = 0
  const createSession = async fileId => {
    calls += 1
    await Promise.resolve()
    return { preview_url: `/preview/${fileId}/${calls}`, expires_in_seconds: 60 }
  }

  const [first, duplicate] = await Promise.all([
    mvGetPreviewSession(7, createSession, { now: 1_000 }),
    mvGetPreviewSession(7, createSession, { now: 1_000 }),
  ])
  assert.equal(calls, 1)
  assert.equal(first.previewUrl, '/preview/7/1')
  assert.equal(duplicate.previewUrl, first.previewUrl)
  assert.equal((await mvGetPreviewSession(7, createSession, { now: 54_999 })).previewUrl, first.previewUrl)
  assert.equal(calls, 1)
  assert.equal((await mvGetPreviewSession(7, createSession, { now: 55_001 })).previewUrl, '/preview/7/2')
  assert.equal(calls, 2)
})

test('thumbnail renders deduplicate by file generation', async () => {
  installBrowserMocks()
  const thumbnailModule = await freshImport('./thumbnail-generators.js')
  thumbnailModule.clearMvPreviewCaches()
  thumbnailModule._mvSessionCache.set(8, { previewUrl: '/preview/8', expiresAt: Infinity })
  let renderCalls = 0
  let resolveRender
  let markRenderStarted
  const renderStarted = new Promise(resolve => { markRenderStarted = resolve })
  const renderResult = new Promise(resolve => { resolveRender = resolve })
  const generate = async () => {
    renderCalls += 1
    markRenderStarted()
    return renderResult
  }
  const createSession = async () => { throw new Error('valid session must be reused') }
  const file = { id: 8, mime_type: 'application/pdf', filename: 'same.pdf' }

  const first = thumbnailModule.mvGenerateFileThumbnail(file, createSession, generate)
  const duplicate = thumbnailModule.mvGenerateFileThumbnail(file, createSession, generate)
  assert.equal(first, duplicate)
  await renderStarted
  assert.equal(renderCalls, 1)

  resolveRender('thumb:same-generation')
  assert.equal(await first, 'thumb:same-generation')
  assert.equal(await duplicate, 'thumb:same-generation')
  assert.equal(thumbnailModule._mvThumbCache.get(8), 'thumb:same-generation')
})

test('forced thumbnail regeneration reuses a valid preview session and is generation-deduplicated', async () => {
  installBrowserMocks()
  const thumbnailModule = await freshImport('./thumbnail-generators.js')
  thumbnailModule.clearMvPreviewCaches()
  thumbnailModule._mvThumbCache.set(21, 'thumb:old')
  thumbnailModule._mvSessionCache.set(21, { previewUrl: '/preview/valid', expiresAt: Infinity })
  thumbnailModule._mvThumbCache.set(22, 'thumb:other')
  thumbnailModule._mvSessionCache.set(22, { previewUrl: '/preview/other', expiresAt: Infinity })
  const events = []
  const generatedEvents = []
  window.addEventListener('lavix-thumbnail-invalidated', event => events.push(event.detail))
  window.addEventListener('lavix-thumbnail-generated', event => generatedEvents.push(event.detail))
  let sessionCalls = 0
  let renderCalls = 0
  let resolveRender
  let markRenderStarted
  const renderStarted = new Promise(resolve => { markRenderStarted = resolve })
  const renderResult = new Promise(resolve => { resolveRender = resolve })
  const createSession = async () => {
    sessionCalls += 1
    return { preview_url: '/preview/replaced', expires_in_seconds: 900 }
  }
  const generate = async (_file, url) => {
    renderCalls += 1
    assert.equal(url, '/preview/valid')
    markRenderStarted()
    return renderResult
  }
  const file = { id: 21, mime_type: 'application/pdf', filename: 'force.pdf' }

  const forced = thumbnailModule.mvGenerateFileThumbnail(file, createSession, generate, { force: true })
  await renderStarted
  const duplicateForce = thumbnailModule.mvGenerateFileThumbnail(file, createSession, generate, { force: true })
  assert.equal(forced, duplicateForce)
  assert.equal(renderCalls, 1)
  assert.equal(sessionCalls, 0)
  assert.deepEqual(events, [{ fileId: 21, generation: 1 }])
  assert.equal(thumbnailModule._mvThumbCache.get(22), 'thumb:other')
  assert.equal(thumbnailModule._mvSessionCache.get(22).previewUrl, '/preview/other')

  resolveRender('thumb:fresh')
  assert.equal(await forced, 'thumb:fresh')
  assert.equal(thumbnailModule._mvThumbCache.get(21), 'thumb:fresh')
  assert.equal(thumbnailModule._mvSessionCache.get(21).previewUrl, '/preview/valid')
  assert.deepEqual(generatedEvents, [{ fileId: 21, generation: 1, thumbnail: 'thumb:fresh' }])

  // The options-only overload keeps production callers concise while the
  // injectable-generator signature above remains backward compatible.
  thumbnailModule._mvThumbCache.set(23, 'thumb:image-old')
  thumbnailModule._mvSessionCache.set(23, { previewUrl: '/preview/image', expiresAt: Infinity })
  assert.equal(
    await thumbnailModule.mvGenerateFileThumbnail(
      { id: 23, mime_type: 'image/png', filename: 'image.png' },
      createSession,
      { force: true },
    ),
    '/preview/image',
  )
  assert.equal(sessionCalls, 0)
})

test('invalidating an active thumbnail generation fences its stale cache write', async () => {
  installBrowserMocks()
  const thumbnailModule = await freshImport('./thumbnail-generators.js')
  thumbnailModule.clearMvPreviewCaches()
  thumbnailModule._mvSessionCache.set(31, { previewUrl: '/preview/31', expiresAt: Infinity })
  thumbnailModule._mvThumbCache.set(32, 'thumb:other')
  thumbnailModule._mvSessionCache.set(32, { previewUrl: '/preview/32', expiresAt: Infinity })
  let renderCalls = 0
  let resolveStaleRender
  let markStaleStarted
  const staleStarted = new Promise(resolve => { markStaleStarted = resolve })
  const staleResult = new Promise(resolve => { resolveStaleRender = resolve })
  const generate = async () => {
    renderCalls += 1
    if (renderCalls === 1) {
      markStaleStarted()
      return staleResult
    }
    return 'thumb:fresh-generation'
  }
  const file = { id: 31, mime_type: 'application/pdf', filename: 'race.pdf' }
  const createSession = async () => { throw new Error('valid session must be preserved') }

  const stale = thumbnailModule.mvGenerateFileThumbnail(file, createSession, generate)
  await staleStarted
  const fresh = thumbnailModule.mvGenerateFileThumbnail(file, createSession, generate, { force: true })
  assert.equal(await fresh, 'thumb:fresh-generation')
  assert.equal(thumbnailModule._mvThumbCache.get(31), 'thumb:fresh-generation')

  resolveStaleRender('thumb:stale-generation')
  assert.equal(await stale, 'thumb:stale-generation')
  assert.equal(thumbnailModule._mvThumbCache.get(31), 'thumb:fresh-generation')
  assert.equal(thumbnailModule._mvSessionCache.get(31).previewUrl, '/preview/31')
  assert.equal(thumbnailModule._mvThumbCache.get(32), 'thumb:other')
  assert.equal(thumbnailModule._mvSessionCache.get(32).previewUrl, '/preview/32')
})

test('manual thumbnail regeneration supersedes a hung forced request', async () => {
  installBrowserMocks()
  const thumbnailModule = await freshImport('./thumbnail-generators.js')
  thumbnailModule.clearMvPreviewCaches()
  thumbnailModule._mvSessionCache.set(41, { previewUrl: '/preview/41', expiresAt: Infinity })
  const invalidations = []
  window.addEventListener('lavix-thumbnail-invalidated', event => invalidations.push(event.detail))
  let renderCalls = 0
  let resolveHung
  let markHungStarted
  const hungStarted = new Promise(resolve => { markHungStarted = resolve })
  const hungResult = new Promise(resolve => { resolveHung = resolve })
  const generate = async () => {
    renderCalls += 1
    if (renderCalls === 1) {
      markHungStarted()
      return hungResult
    }
    return 'thumb:recovered'
  }
  const createSession = async () => { throw new Error('valid session must be preserved') }
  const file = { id: 41, mime_type: 'application/pdf', filename: 'hung.pdf' }

  const hung = thumbnailModule.mvGenerateFileThumbnail(file, createSession, generate, { force: true })
  await hungStarted
  const recovered = thumbnailModule.mvGenerateFileThumbnail(
    file,
    createSession,
    generate,
    { force: true, supersede: true },
  )
  assert.equal(await recovered, 'thumb:recovered')
  assert.equal(renderCalls, 2)
  assert.equal(thumbnailModule._mvThumbCache.get(41), 'thumb:recovered')
  assert.deepEqual(invalidations, [
    { fileId: 41, generation: 1 },
    { fileId: 41, generation: 2 },
  ])

  resolveHung('thumb:obsolete')
  assert.equal(await hung, 'thumb:obsolete')
  assert.equal(thumbnailModule._mvThumbCache.get(41), 'thumb:recovered')
})

test('superseding thumbnail work aborts the obsolete production-style renderer', async () => {
  installBrowserMocks()
  const thumbnailModule = await freshImport('./thumbnail-generators.js')
  thumbnailModule.clearMvPreviewCaches()
  thumbnailModule._mvSessionCache.set(43, { previewUrl: '/preview/43', expiresAt: Infinity })
  let renderCalls = 0
  let aborts = 0
  let markStarted
  const started = new Promise(resolve => { markStarted = resolve })
  const generate = async (_file, _url, { signal } = {}) => {
    renderCalls += 1
    if (renderCalls > 1) return 'thumb:replacement'
    markStarted()
    return new Promise((resolve, reject) => {
      signal.addEventListener('abort', () => {
        aborts += 1
        const error = new Error('obsolete thumbnail work aborted')
        error.name = 'AbortError'
        reject(error)
      }, { once: true })
    })
  }
  const createSession = async () => { throw new Error('valid session must be preserved') }
  const file = { id: 43, mime_type: 'application/pdf', filename: 'abortable.pdf' }

  const obsolete = thumbnailModule.mvGenerateFileThumbnail(file, createSession, generate)
  await started
  const replacement = thumbnailModule.mvGenerateFileThumbnail(
    file,
    createSession,
    generate,
    { force: true, supersede: true },
  )

  assert.equal(await replacement, 'thumb:replacement')
  await assert.rejects(obsolete, /obsolete thumbnail work aborted/)
  assert.equal(aborts, 1)
  assert.equal(renderCalls, 2)
  assert.equal(thumbnailModule._mvThumbCache.get(43), 'thumb:replacement')
})

test('manual thumbnail regeneration detaches a hung preview-session request', async () => {
  installBrowserMocks()
  const thumbnailModule = await freshImport('./thumbnail-generators.js')
  thumbnailModule.clearMvPreviewCaches()
  let sessionCalls = 0
  let resolveHungSession
  let markHungSessionStarted
  const hungSessionStarted = new Promise(resolve => { markHungSessionStarted = resolve })
  const createSession = () => {
    sessionCalls += 1
    if (sessionCalls === 1) {
      markHungSessionStarted()
      return new Promise(resolve => { resolveHungSession = resolve })
    }
    return Promise.resolve({ preview_url: '/preview/recovered', expires_in_seconds: 900 })
  }
  const generate = async (_file, url) => `thumb:${url}`
  const file = { id: 42, mime_type: 'application/pdf', filename: 'hung-session.pdf' }

  const hung = thumbnailModule.mvGenerateFileThumbnail(file, createSession, generate, { force: true })
  await hungSessionStarted
  const recovered = thumbnailModule.mvGenerateFileThumbnail(
    file,
    createSession,
    generate,
    { force: true, supersede: true },
  )
  assert.equal(await recovered, 'thumb:/preview/recovered')
  assert.equal(sessionCalls, 2)
  assert.equal(thumbnailModule._mvSessionCache.get(42).previewUrl, '/preview/recovered')
  assert.equal(thumbnailModule._mvThumbCache.get(42), 'thumb:/preview/recovered')

  resolveHungSession({ preview_url: '/preview/obsolete', expires_in_seconds: 900 })
  assert.equal(await hung, 'thumb:/preview/obsolete')
  assert.equal(thumbnailModule._mvSessionCache.get(42).previewUrl, '/preview/recovered')
  assert.equal(thumbnailModule._mvThumbCache.get(42), 'thumb:/preview/recovered')
})

test('a stale thumbnail URL gets exactly one forced preview-session retry', async () => {
  installBrowserMocks()
  const { clearMvPreviewCaches, mvGenerateFileThumbnail } = await freshImport('./thumbnail-generators.js')
  clearMvPreviewCaches()
  let sessionCalls = 0
  let renderCalls = 0
  const createSession = async () => ({
    preview_url: `/preview/${sessionCalls += 1}`,
    expires_in_seconds: 900,
  })
  const generate = async (_file, url) => {
    renderCalls += 1
    if (renderCalls === 1) {
      const error = new Error('stale preview')
      error.status = 404
      throw error
    }
    return `thumb:${url}`
  }

  assert.equal(
    await mvGenerateFileThumbnail({ id: 9, mime_type: 'application/pdf', filename: 'a.pdf' }, createSession, generate),
    'thumb:/preview/2',
  )
  assert.equal(sessionCalls, 2)
  assert.equal(renderCalls, 2)
})

test('an explicit thumbnail retry clears a failed preview-session cooldown', async () => {
  installBrowserMocks()
  const thumbnailModule = await freshImport('./thumbnail-generators.js')
  thumbnailModule.clearMvPreviewCaches()
  let sessionCalls = 0
  const createSession = async () => {
    sessionCalls += 1
    if (sessionCalls === 1) throw new Error('temporary preview-session failure')
    return { preview_url: '/preview/recovered', expires_in_seconds: 900 }
  }
  const file = { id: 44, mime_type: 'application/pdf', filename: 'retry.pdf' }
  const generate = async (_file, url) => `thumb:${url}`

  await assert.rejects(
    thumbnailModule.mvGenerateFileThumbnail(file, createSession, generate),
    /temporary preview-session failure/,
  )
  assert.equal(sessionCalls, 1)
  thumbnailModule.invalidateMvFileThumbnail(44, { detachPendingSession: true })
  assert.equal(
    await thumbnailModule.mvGenerateFileThumbnail(file, createSession, generate),
    'thumb:/preview/recovered',
  )
  assert.equal(sessionCalls, 2)
})

test('thumbnail-only invalidation emits its generation and full preview invalidation still resets the session', async () => {
  installBrowserMocks()
  const thumbnailModule = await freshImport('./thumbnail-generators.js')
  thumbnailModule.clearMvPreviewCaches()
  thumbnailModule._mvThumbCache.set(9, 'failed-file-thumb')
  thumbnailModule._mvThumbCache.set(10, 'other-file-thumb')
  thumbnailModule._mvSessionCache.set(9, { previewUrl: '/stale', expiresAt: Infinity })
  thumbnailModule._mvSessionCache.set(10, { previewUrl: '/valid', expiresAt: Infinity })
  const events = []
  window.addEventListener('lavix-thumbnail-invalidated', event => events.push(event.detail))

  assert.equal(window.invalidateMvFileThumbnail, thumbnailModule.invalidateMvFileThumbnail)
  assert.equal(thumbnailModule.invalidateMvFileThumbnail(9), true)
  assert.equal(thumbnailModule._mvThumbCache.has(9), false)
  assert.equal(thumbnailModule._mvSessionCache.get(9).previewUrl, '/stale')
  assert.equal(thumbnailModule._mvThumbCache.get(10), 'other-file-thumb')
  assert.equal(thumbnailModule._mvSessionCache.get(10).previewUrl, '/valid')
  assert.deepEqual(events, [{ fileId: 9, generation: 1 }])

  assert.equal(thumbnailModule.invalidateMvFilePreview(9), true)
  assert.equal(thumbnailModule._mvThumbCache.has(9), false)
  assert.equal(thumbnailModule._mvSessionCache.has(9), false)
  assert.equal(thumbnailModule._mvThumbCache.get(10), 'other-file-thumb')
  assert.equal(thumbnailModule._mvSessionCache.get(10).previewUrl, '/valid')
  assert.deepEqual(events, [
    { fileId: 9, generation: 1 },
    { fileId: 9, generation: 2 },
  ])
  assert.equal(thumbnailModule.invalidateMvFileThumbnail('invalid'), false)
  assert.equal(thumbnailModule.invalidateMvFilePreview('invalid'), false)
})

test('preview and thumbnail caches clear when the authenticated account changes', async () => {
  installBrowserMocks()
  const thumbnailModule = await freshImport('./thumbnail-generators.js')
  thumbnailModule.clearMvPreviewCaches()
  thumbnailModule._mvThumbCache.set(1, 'thumb')
  thumbnailModule._mvSessionCache.set(1, { previewUrl: '/old', expiresAt: Infinity })
  const invalidations = []
  window.addEventListener('lavix-thumbnail-invalidated', event => invalidations.push(event.detail))
  localStorage.setItem('session_id', 'old-session')
  localStorage.setItem('username', 'old-user')
  const { storeAuthSession } = await freshImport('./vault-client.js')

  storeAuthSession({
    access_token: 'new-access',
    refresh_token: 'new-refresh',
    session_id: 'new-session',
    last_activity_at: '2026-07-15T10:00:00Z',
    idle_expires_at: '2026-07-15T12:00:00Z',
    idle_timeout_seconds: 7200,
  }, 'new-user')

  assert.equal(thumbnailModule._mvThumbCache.size, 0)
  assert.equal(thumbnailModule._mvSessionCache.size, 0)
  assert.deepEqual(invalidations, [{ fileId: 1, generation: 1, reload: false, reason: 'cache-clear' }])
})

test('low-memory errors emit a backend event but health checks stay quiet', async () => {
  const { events } = installBrowserMocks(async url => jsonResponse({ reason: 'low_memory' }, 503))
  await freshImport('./vault-client.js')

  await fetch('http://lavix.test/api/low-memory')
  await fetch('http://lavix.test/api/health')

  assert.equal(events.filter(event => event.type === 'backend-error').length, 1)
})

test('unsupported-format indexing errors translate into friendly attachment copy', () => {
  const err = code => new Error(code)
  assert.equal(
    describeIndexingError(err('no_parser_for_media_type'), 'tool.exe'),
    "tool.exe attached — AI indexing doesn't read this file type. Supported: PDF, DOCX, XLSX, PPTX, HTML, CSV, TXT and common images.",
  )
  assert.equal(
    describeIndexingError(err('audio_video_disabled'), 'clip.mp4'),
    "clip.mp4 attached — audio and video files aren't indexed.",
  )
  assert.equal(
    describeIndexingError(err('unsupported_image_encoding'), 'raw.nef'),
    "raw.nef attached — this image format isn't supported for indexing.",
  )
  assert.match(describeIndexingError(new Error('Failed to grant access'), 'x.bin'), /Supported: PDF/)
  assert.equal(
    describeIndexingError(err('something unexpected'), 'a.txt'),
    "a.txt attached — indexing couldn't start right now. You can enable it later from the file menu.",
  )
})

test('chat grant-and-send polls real indexing status instead of racing the fail-fast', () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  const orchestrator = readFileSync(
    new URL('../../app/routers/chat/orchestrator.py', import.meta.url), 'utf8',
  )

  // The old flow waited a blind 800ms and fired straight into the backend's
  // instant selected_file_not_ready rejection.
  assert.doesNotMatch(app, /setTimeout\(r,\s*800\)/)
  assert.match(app, /const waitDeadline = Date\.now\(\) \+ 35_000/)
  assert.match(app, /grantedIds\.map\(id => api\.getAIStatus\(id\)/)
  assert.match(app, /allReady = states\.every\(status => !isIndexActive\(status\)\)/)
  assert.doesNotMatch(app, /let stillIndexing/)
  assert.match(app, /Still indexing — your message is kept/)

  // Backend mirrors this: freshly granted regular files wait; stale ones fail fast.
  assert.match(orchestrator, /CHAT_RECENT_GRANT_WINDOW_SECONDS = 120/)
  assert.match(orchestrator, /not states\[file_id\]\.recently_granted/)
})

test('relationship memory pages at 3 and grant access never reloads the grid', () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')

  // Relationship memory list shows 3 per page; Previous/Next already wired.
  assert.match(app, /const GRAPH_MEMORY_PAGE_SIZE = 3;/)
  assert.match(app, /aria-label="Previous relationship memory page"/)

  // Grant from the home grid is a single-card optimistic flip: no global
  // loading toggle, no skeleton flash, silent background truth-sync.
  const handler = app.match(
    /const handleGrantAccess = async \(id\) => \{([\s\S]*?)\n            \};/,
  )?.[1] || ''
  assert.match(handler, /setFiles\(prev => prev\.map\(/)
  assert.match(handler, /ai_status: 'queued'/)
  assert.match(handler, /refresh\(\{ showLoading: false \}\)/)
  assert.doesNotMatch(handler, /setLoading\(/)
})

test('@-mention menu defaults to Recent tab, then Files, then Folders', () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')

  // Default tab and reset-on-close are both Recent.
  assert.match(app, /const \[mentionTab, setMentionTab\] = useState\('recent'\)/)
  assert.match(app, /setExpandedFolderId\(null\);\s*setMentionTab\('recent'\)/)

  // Tab bar order: Recent button renders before Files, Folders stays last.
  const recentBtn = app.indexOf(">Recent</button>")
  const filesBtn = app.indexOf(">Files</button>")
  const foldersBtn = app.indexOf(">Folders</button>")
  assert.ok(recentBtn !== -1 && filesBtn !== -1 && foldersBtn !== -1)
  assert.ok(recentBtn < filesBtn && filesBtn < foldersBtn)

  // Tab key cycles all three tabs and resets the highlight index.
  assert.match(app, /setMentionTab\(prev => prev === 'recent' \? 'files' : prev === 'files' \? 'folders' : 'recent'\)/)

  // Recent list: most recently ADDED first (uploaded_at desc, the same
  // timestamp the backend list and file grid sort by), already-tagged
  // excluded, same query filter as Files, capped at 10.
  assert.match(app, /Date\.parse\(b\.uploaded_at \|\| b\.created_at/)
  assert.match(app, /recentFileItems\.forEach\(f => flatItems\.push\(f\)\)/)
  assert.match(app, /files: mentionTab === 'recent' \? recentFileItems : fileSlice/)
  assert.match(app, /folders: mentionTab === 'recent' \? \[\] : folderSlice/)

  // Files tab is an alphabetical A-Z index, never the backend upload order,
  // so Recent (chronological) and Files can never render the same list.
  assert.match(app, /const fileSlice = filteredFileItems\s*\n?\s*\.sort\(\(a, b\) => \(a\._fullPath \|\| a\.filename/)

  // Recent rows carry a visible Added-date sublabel.
  assert.match(app, /mentionTab === 'recent' && file\.uploaded_at/)

  // File-row render block serves both Files and Recent tabs.
  assert.match(app, /\(mentionTab === 'files' \|\| mentionTab === 'recent'\) && \(/)
})

test('chat scope helpers keep tag/followup/clear payloads distinct', async () => {
  const { normalizeScopeIds, resolveSendFileIds, followupFileIds } = await import('./frontend-workflows.js')
  assert.deepEqual(normalizeScopeIds([3, 1, 3, 0, -2, 'x', true, '7']), [3, 1, 7])
  assert.deepEqual(normalizeScopeIds(null), [])
  assert.deepEqual(normalizeScopeIds({}), [])
  // Fresh tags win over everything.
  assert.deepEqual(resolveSendFileIds({ tagIds: [3], sessionScope: [5] }), [3])
  // Explicit clear transmits [] so the server wipes session scope.
  assert.deepEqual(resolveSendFileIds({ tagIds: [], sessionScope: [5], scopeCleared: true }), [])
  // Untagged with no clear intent transmits null (discovery/server fallback may run).
  assert.equal(resolveSendFileIds({ tagIds: [], sessionScope: [5] }), null)
  assert.equal(resolveSendFileIds({}), null)
  // Follow-ups re-attach known scope, else null for the server fallback.
  assert.deepEqual(followupFileIds([5, 9]), [5, 9])
  assert.equal(followupFileIds([]), null)
})

test('App wires session scope through sends and seeds it on load', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // Follow-up, regenerate and skip resends re-attach session scope.
  assert.match(app, /sendRaw\(question, null, scopeIds, null, true, null, false, sessionFolderIds\(\)\)/)
  assert.match(app, /sendRaw\(userMsg, null, regenScope, null, true, null, false, sessionFolderIds\(\)\)/)
  assert.match(app, /followupFileIds\(chatScopes\[pgChatId \|\| pgChatIdRef\.current\] \|\| \[\]\)/)
  // History load seeds scope from the new projection field.
  assert.match(app, /normalizeScopeIds\(data\.scoped_file_ids\)/)
  // Scope indicator with explicit clear action.
  assert.match(app, /Scoped to \{sessionScope\.length\} file/)
  assert.match(app, /scopeClearedRef\.current = true/)
  // Server notice events surface as toasts.
  assert.match(app, /onNotice: \(event\) => \{/)
})

test('Expert default licenses length and agrees with the synthesis contract', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  const match = app.match(/const DEFAULT_PROMPT_B = `([\s\S]*?)`;/)
  assert.ok(match, 'DEFAULT_PROMPT_B found')
  const expert = match[1].replace(/\\n/g, '\n')
  assert.match(expert, /complete answers/)
  assert.match(expert, /Never truncate/)
  assert.ok(!expert.includes('Cite retrieved sources explicitly'), 'no citation-ID conflict')
  assert.match(expert, /server renders source cards automatically/)
  assert.match(expert, /presentation only/)
})

test('persona v5 migration only refreshes untouched Expert slots', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  assert.match(app, /PERSONA_VERSION = 'v5'/)
  assert.match(app, /PREVIOUS_DEFAULT_PROMPT_B/)
  assert.match(app, /localStorage\.getItem\('personaPromptB'\) === PREVIOUS_DEFAULT_PROMPT_B/)
})

test('composer chips carry a Currently reading caption while processing', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  assert.match(app, /Currently reading…/)
  assert.match(app, /processing && \(taggedFiles\.length > 0 \|\| taggedFolders\.length > 0\)/)
})

test('chat mode flag rides every send path', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  assert.match(app, /mode: activePrompt === 'B' \? 'expert' : 'casual'/)
})

test('scope union helper merges tags additively', async () => {
  const { unionScopeIds } = await import('./frontend-workflows.js')
  assert.deepEqual(unionScopeIds([3], [3, 9]), [3, 9])
  assert.deepEqual(unionScopeIds([], [5]), [5])
  assert.deepEqual(unionScopeIds(null, null), [])
})

test('scope pill owns the only manager onclick; edits PATCH scope', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // Pill body opens the manager; its X stops propagation and clears instead.
  assert.match(app, /onClick=\{\(\) => setShowScopeManager\(true\)\}/)
  assert.match(app, /e\.stopPropagation\(\);\s*clearScopeNow\(\)/)
  // Manager lists files with per-file remove + clear-all + done.
  assert.match(app, /removeScopeFile\(id\)/)
  assert.match(app, /api\.setChatScope\(key, next\)/)
  assert.match(app, /Chat scope — \{sessionScope\.length \+ sessionFolders\.length\} item/)
  assert.match(app, /Type @ in chatbar to add more files/)
  // Esc closes without changing anything.
  assert.match(app, /if \(e\.key === 'Escape'\) setShowScopeManager\(false\)/)
  // Additive sends: fresh tags union into live session scope.
  assert.match(app, /unionScopeIds\(tagIds, liveScope\)/)
  // Exactly two manager openers: folder pill + scoop pill, nothing else.
  const opens = app.match(/onClick=\{\(\) => setShowScopeManager\(true\)\}/g) || []
  assert.equal(opens.length, 2)
})

test('scope pill stays visible while composing, committed scope only', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // Pill no longer hides behind composer tags; the row renders whenever
  // files, folders, or session scope exist.
  assert.match(app, /taggedFolders\.length === 0 && !hasScope\) return null/)
  // No merged preview: pill shows committed scope, composer chips show the
  // draft. One file is never displayed twice, pill and manager agree.
  assert.ok(!app.includes('previewIds'), 'no uncommitted preview')
  assert.ok(!app.includes('+adding'), 'no adding badge')
  // Scoop pill is files-only; folders live in their own pill.
  assert.match(app, /Scoped to \{sessionScope\.length\} file/)
  assert.ok(!app.includes("parts.join(' + ')"), 'no file+folder merge in pill')
  // The two manager openers are exactly the folder pill + scoop pill.
  const opens = app.match(/onClick=\{\(\) => setShowScopeManager\(true\)\}/g) || []
  assert.equal(opens.length, 2)
})

test('scope pill renders at position 0 before composer chips', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  const folderPillAt = app.indexOf('Manage folder scope')
  const scoopPillAt = app.indexOf('Manage chat file scope')
  const chipsAt = app.indexOf('.map(tf => {')
  assert.ok(folderPillAt !== -1 && scoopPillAt !== -1 && chipsAt !== -1)
  assert.ok(folderPillAt < scoopPillAt && scoopPillAt < chipsAt, 'folder pill 0, scoop pill 1, chips after')
})

test('folder scope flows end to end through sends and manager', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // Folder tag action exists on menu rows, distinct from expand.
  assert.match(app, /selectMentionFolder\(file\)/)
  assert.match(app, /Scope whole folder/)
  // Sends carry folderIds; follow-ups re-attach session folders.
  assert.match(app, /folderIds: folderIds !== undefined \? folderIds : null/)
  assert.match(app, /sessionFolderIds\(\)/)
  // Pill shows folder counts; manager lists folders with remove.
  assert.match(app, /sessionFolders\.length/)
  assert.match(app, /removeScopeFolder\(id\)/)
  assert.match(app, /live folder/)
  // No per-folder composer chips: tagged folders render only through the
  // folder pill + manager pending section.
  assert.ok(!app.includes('{taggedFolders.map(tf => ('), 'no dumb folder chips')
  // Pending tagged folders appear in the manager with untag remove.
  assert.match(app, /pending-folder-\$\{tf\.id\}|pending-folder-/)
  // Projection seeding covers folders.
  assert.match(app, /normalizeScopeIds\(data\.scoped_folder_ids\)/)
})

test('@-mention filter trims whitespace-only queries to show all files', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  assert.match(app, /const q = mentionQuery\.toLowerCase\(\)\.trim\(\)/)
})

test('tag classifier splits unsupported / unindexed / ok', async () => {
  const { classifyTagTarget } = await import('./frontend-workflows.js')
  assert.equal(classifyTagTarget({ original_filename: 'a.zip', ai_status: 'not_granted' }).verdict, 'unsupported')
  assert.equal(classifyTagTarget({ original_filename: 'b.rar', mime_type: 'application/octet-stream' }).verdict, 'unsupported')
  assert.equal(classifyTagTarget({ original_filename: 'c.mp4', mime_type: 'video/mp4', ai_status: 'ready' }).verdict, 'unsupported')
  assert.equal(classifyTagTarget({ original_filename: 'd.mov', ai_status: 'ready' }).verdict, 'unsupported')
  assert.equal(classifyTagTarget({ original_filename: 'e.pdf', ai_status: 'not_granted' }).verdict, 'unindexed')
  assert.equal(classifyTagTarget({ original_filename: 'f.pdf' }).verdict, 'unindexed')
  assert.equal(classifyTagTarget({ original_filename: 'g.pdf', ai_status: 'ready' }).verdict, 'ok')
  assert.ok(classifyTagTarget({ original_filename: 'a.zip' }).reason.length > 10)
  assert.ok(classifyTagTarget({ original_filename: 'e.pdf', ai_status: 'not_granted' }).reason.length > 10)
})

test('unsupported files are blocked at tag time, unindexed get red pills', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // selectMention guards before attaching; popup stays open for another pick.
  assert.match(app, /const verdict = classifyTagTarget\(item\)/)
  assert.match(app, /verdict\.verdict === 'unsupported'/)
  // tag-all and vault picker filter with a skip toast.
  assert.match(app, /Skipped \$\{blocked\.length\} unreadable file/)
  // Red pill branch keyed on the warning flag, amber indexing state untouched.
  assert.match(app, /tf\._warn/)
  assert.match(app, /bg-red-500\/10 border-red-500\/30/)
  assert.match(app, /not indexed/)
})

test('folder pill carries its own clear action; transport omits null keys', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // Folder pill X clears folders only, never opens the manager.
  assert.match(app, /clearFolderScopeNow/)
  assert.match(app, /Clear folder scope — file scope untouched/)
  // Transport: absent keys omitted so folder-only edits cannot wipe files.
  const client = readFileSync(new URL('./vault-client.js', import.meta.url), 'utf8')
  assert.match(client, /if \(fileIds !== null && fileIds !== undefined\) body\.file_ids = fileIds/)
  assert.match(client, /if \(folderIds !== null && folderIds !== undefined\) body\.folder_ids = folderIds/)
})

test('composer chips hide files already named by the scoop pill', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  assert.match(app, /\.filter\(tf => !sessionScope\.includes\(tf\.id\)\)/)
})

test('whole-vault fallback notice clears the scope chip', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // The server marks whole-vault fallback notices; the client drops the
  // dead chat scope so the chip cannot claim unapplied scope.
  assert.match(app, /event\.scope_fallback/)
  assert.match(app, /delete next\[deadKey\]/)
})

test('vault picker renders above the sidebar with full title and backdrop', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // The picker overlay must layer above the sidebar (z-[200]) or it renders
  // underneath it: backdrop misses the sidebar and the title clips.
  const overlay = app.match(/\{showVaultPicker && \([\s\S]*?\n(\s*)<div className="([^"]*)" onClick=\{\(\) => setShowVaultPicker\(false\)\}/)
  assert.ok(overlay, 'vault picker overlay block found')
  const classes = overlay[2]
  assert.match(classes, /fixed inset-0/, 'overlay is viewport-fixed and full-bleed')
  assert.match(classes, /z-\[300\]/, 'overlay layers above the sidebar')
  assert.match(classes, /bg-black\/70/, 'overlay carries a dimming backdrop')
  assert.match(classes, /items-center justify-center/, 'overlay centers the dialog')
  assert.match(app, />Add from Vault<\/h3>/, 'picker title renders in full')
})

test('folder context menu escapes card stacking via portal', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // Hovered sibling cards create stacking contexts (translateY) that paint
  // over an in-card absolute menu. The open menu portals to body instead.
  assert.match(app, /createPortal\(/)
  assert.match(app, /fixed z-\[9999\]/)
  assert.match(app, /anchorRect/)
  assert.match(app, /document\.addEventListener\('scroll', scrollHandler, true\)/)
})

test('file details offers chat-about-this-file as scoped new chat', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // The CHAT cell replaces the STATUS chip in the file stat grid (status
  // already lives in the header badge). Type-agnostic file branch only.
  assert.match(app, /data-testid="chat-about-file"/)
  assert.match(app, /title="Open a new chat scoped to this file"/)
  // Handler: new chat + id via the established preTagIds path (taggedFiles
  // lives in ChatView scope — touching it here throws and the click dies).
  assert.match(app, /const handleChatAboutFile = \(f\) => \{/)
  assert.match(app, /setChatPreTagIds\(\[fid\]\)/)
  assert.doesNotMatch(app, /setTaggedFiles\(\[\{/)
  assert.match(app, /setCurrentPgChatId\(null\)/)
  assert.match(app, /setSelectedItem\(null\)/)
  assert.match(app, /onChatAboutFile=\{handleChatAboutFile\}/)
})

test('dashboard search bar focuses in theme indigo, never red', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // The main vault search must focus with the theme accent; red focus
  // implies a validation error where there is none.
  const match = app.match(/placeholder="Search files or #tags\.\.\."[\s\S]{0,400}?className="([^"]*)"/)
  assert.ok(match, 'dashboard search input found')
  assert.match(match[1], /focus:border-indigo-500\/50/)
  assert.doesNotMatch(match[1], /focus:border-red/)
})

test('scope-mismatch notice is advisory with clear action', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // Fires only on entity-bearing input with zero filename/tag overlap;
  // never blocks or auto-clears (clearScopeNow only on explicit click).
  assert.match(app, /data-testid="scope-mismatch-notice"/)
  assert.match(app, /Still scoped to/)
  assert.match(app, /may not be relevant/)
  assert.match(app, /scopeWords\.has\(e\.toLowerCase\(\)\)/)
  assert.match(app, /entities\.length === 0\) return null/)
})

test('chat model popup uses clickable allowed list, no dropdown', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // Single pattern: each Allowed entry selects immediately with ✓ on active.
  assert.match(app, /onClick=\{\(\) => updateActiveChatModel\(m\)\}/)
  assert.match(app, /'✓ '/)
  assert.doesNotMatch(app, /onChange=\{\(e\) => updateActiveChatModel\(e\.target\.value\)\}/)
  // Picking a model keeps the popup open (no auto-close on success).
  assert.doesNotMatch(app, /updateActiveChatModel[\s\S]{0,800}?setShowModelMenu\(false\)/)
  // Active Model is a read-only display (no dropdown select).
  assert.match(app, /Active Model<\/div>/)
  assert.doesNotMatch(app, /<select[^>]*updateActiveChatModel/)
})

test('provider section never references openrouter', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  assert.doesNotMatch(app, /[Oo]pen[Rr]outer/)
  assert.match(app, /Local models served via Ollama\./)
})

test('ai-ready boxes show fraction, percentage and bar', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // Fraction is searchable over every file; bar + % follow indexable.
  assert.match(app, /\{aiProgress\.searchable\}\/\{aiProgress\.allFilesTotal\}/)
  assert.match(app, />\{aiProgress\.percentage\}%</)
  assert.match(app, /bg-gradient-to-r from-blue-600 to-violet-500/)
})

test('upload and grant flows refresh ai stats', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // Upload completion moves the eligible denominator immediately, even
  // before AI access is granted; grant/revoke update it too.
  const uploadBlock = app.match(/New uploads change the eligible-files denominator[\s\S]{0,200}?refreshStats\(\);/)
  assert.ok(uploadBlock, 'upload completion refreshes stats')
  const grantBlock = app.match(/handleGrantAccessForPanel[\s\S]{0,600}?refreshStats\(\);/)
  assert.ok(grantBlock, 'grant refreshes stats')
})

test('web auto-check note renders beside sources', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // The override note rides the sources event onto the message, then
  // renders as its own banner (never inside answer prose).
  assert.match(app, /data-testid="assistant-webauto-note"/)
  assert.match(app, /webAutoNote/)
  assert.match(app, /data-testid="assistant-fallback-note"/)
})

test('admin max-context dropdown is admin-only with fixed options', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  assert.match(app, /testid="admin-max-num-ctx"/)
  assert.match(app, /userProfile\?\.is_admin && \(/)
  assert.match(app, /Max context/)
  assert.match(app, /model_max_num_ctx_options/)
})

test('chat card shows one active model plus the default dropdown', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // Collapsed card: single active-model readout + default dropdown.
  // Preference/admin-default rows and the bare fallback line are gone.
  assert.match(app, /testid="admin-active-chat-model"/)
  assert.match(app, /testid="model-role-select-chat"/)
  assert.doesNotMatch(app, /admin-preferred-chat-model/)
  assert.doesNotMatch(app, /testid="admin-default-chat-model"/)
  assert.match(app, /default_model: model,/)
})

test('allowlist chips toggle membership without reassigning the default model', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // The chip add branch must not carry default_model — selecting an
  // allowed model must never switch the admin default/active marker.
  assert.match(app, /allowed_models: \[\.\.\.current, model\] \}/)
  assert.doesNotMatch(app, /allowed_models: \[\.\.\.current, model\], default_model: model/)
  // Removing the current default from the allowlist is blocked; the
  // default must change via the explicit default dropdown first.
  assert.match(app, /if \(model === previous\.chat\.default_model\) return previous/)
  // The chip highlight tracks the admin default only (draft first,
  // saved fallback) — never the account's personal active model.
  assert.match(app, /const active = model === \(modelsDraft\?\.chat\?\.default_model \|\| modelsConfig\.chat\?\.default_model\)/)
})

test('all dropdowns share the VaultDropdown component', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  const css = readFileSync(new URL('./index.css', import.meta.url), 'utf8')
  // No native <select> ELEMENTS remain (comments excluded): every
  // dropdown renders through the shared portal-escaped component.
  const codeLines = app.split('\n').filter(line => !line.trim().startsWith('//'));
  assert.doesNotMatch(codeLines.join('\n'), /<select[\s>]/);
  assert.match(app, /createPortal\(/)
  assert.match(app, /z-\[9999\]/)
  assert.match(app, /role="listbox"/)
  assert.match(app, /aria-selected/)
  // Previous native-select sites now use it (spot check).
  assert.match(app, /<VaultDropdown/)
  assert.match(css, /\.vault-select option/)
})

test('chat card fallback row is selectable with fail-loudly option', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  assert.match(app, /testid="model-role-select-fallback"/)
  assert.match(app, /testid="admin-fallback-chat-model"/)
  assert.match(app, /None \(fail loudly\)/)
  assert.match(app, /fallback_chat_model: model \|\| null/)
})

test('model save strips nested fallback from chat object', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // fallback_chat_model lives top-level only; the nested copy inside
  // modelsDraft.chat must not ride along or the backend rejects it as
  // chat.fallback_chat_model (extra=forbid).
  assert.match(app, /\(\(\{ fallback_chat_model, \.\.\.rest \}\) => rest\)\(modelsDraft\.chat\)/)
  assert.match(app, /fallback_chat_model: modelsDraft\.chat\?\.fallback_chat_model \|\| null/)
})

test('revoke confirmation closes on accept without waiting for refresh', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // The wrapper hands every confirm handler a close() callback and
  // guarantees the modal closes even when a handler throws (previously a
  // throwing handler wedged the modal open forever).
  assert.match(app, /await config\.onConfirm\(close\)/)
  assert.match(app, /if \(!closed\) setConfirmState\(null\)/)
  // Dashboard revoke: POST, then close, then background refresh — and no
  // reference to the undefined setLoading that crashed this path.
  assert.doesNotMatch(app, /handleRevokeAccess[\s\S]{0,800}?setLoading/)
  assert.match(app, /revokeAIAccess\(id\)[\s\S]{0,400}?close\(\)[\s\S]{0,400}?refresh\(\)\.catch/)
  // Panel path forwards close and toasts failures instead of console-only.
  assert.match(app, /revokeAccessForPanel\(fileId, close\)/)
  assert.match(app, /revokeAccessForPanel[\s\S]{0,1200}?describeIndexingError/)
})

test('bulk bar offers Enable AI and Revoke AI for selected files', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  const client = readFileSync(new URL('./vault-client.js', import.meta.url), 'utf8')
  // Client methods hit the bulk endpoints with a JSON body.
  assert.match(client, /revokeAIAccessBulk: async \(fileIds\)/)
  assert.match(client, /\/files\/revoke-ai-access\/bulk/)
  assert.match(client, /grantAIAccessBulk: async \(fileIds\)/)
  assert.match(client, /\/files\/grant-ai-access\/bulk/)
  // Bar renders both buttons, files-only, disabled while a bulk op runs.
  assert.match(app, /onClick=\{handleBulkGrant\}[\s\S]{0,120}?disabled=\{bulkBusy\}/)
  assert.match(app, /onClick=\{handleBulkRevoke\}[\s\S]{0,120}?disabled=\{bulkBusy\}/)
  // Bulk revoke goes through the confirm modal and closes on accept.
  assert.match(app, /title: `Revoke AI access for \$/)
  assert.match(app, /await api\.revokeAIAccessBulk\(ids\)[\s\S]{0,200}?close\(\)/)
  // Failures produce a named summary instead of silent swallowing.
  assert.match(app, /revoked, \$\{failed\.length\} failed/)
  assert.match(app, /granted, \$\{failed\.length\} failed/)
  assert.doesNotMatch(app, /revokeAIAccessBulk\(ids\)\.catch\(\(\) => \{\}\)/)
})

test('VaultDropdown keeps interior scroll and auto-scrolls highlight', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // Capture-phase scroll closes only for scrollers outside the menu itself.
  assert.match(app, /const onScroll = \(e\) => \{ if \(menuRef\.current && !menuRef\.current\.contains\(e\.target\)\) closeMenu\(\); \}/)
  // Highlight follows into view for lists taller than the viewport.
  assert.match(app, /role="option"[\s\S]*?scrollIntoView\(\{ block: 'nearest' \}\)/)
})

test('extractive summary badge explains the fallback reason', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  assert.match(app, /intelligenceFallbackHints/)
  assert.match(app, /invalid_model_response: 'Model output was unusable/)
  assert.match(app, /title=\{intelligenceFallbackHint \|\| undefined\}/)
})

test('enable dialog merges grantable and cancelled into one awaiting bucket', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // One pending bucket with per-file reasons: grantable-but-ungranted
  // plus granted-cancelled without a revision. Never double-counted,
  // never hidden.
  assert.match(app, /const awaitingFiles = \(files \|\| \[\]\)\.flatMap\(f =>/)
  assert.match(app, /reason: 'needs grant'/)
  assert.match(app, /reason: 'cancelled — ACTIVATE retries'/)
  assert.match(app, /Awaiting action:/)
  assert.match(app, /\$\{freshNeeded\} files left/)
  assert.doesNotMatch(app, /To retry \(cancelled\):/)
})

test('enable dialog lists unsupported formats separately', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  assert.match(app, /Unsupported format:/)
  assert.match(app, /f\.parser_supported !== false/)
  assert.match(app, /No parser for this format/)
})

test('completed uploads always refresh the file list', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // The post-upload timeout must refresh in both branches — previously only
  // minimized uploads refreshed, leaving new files invisible otherwise.
  const block = app.match(/setTimeout\(async \(\) => \{([\s\S]{0,1200}?)\}, 800\)/)
  assert.ok(block, 'upload completion timeout exists')
  assert.match(block[1], /refreshFiles\(\)/)
  assert.match(block[1], /refreshStats\(\)/)
})

test('enable dialog bucket uses the serialized consent field', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // Consent is read from `ai_access_granted` (the list API field) — the
  // DB column name never reaches the client and silently broke both rows.
  assert.match(app, /!f\.ai_access_granted &&/)
  assert.match(app, /f\.parser_supported === true/)
  const bucket = app.match(/const awaitingFiles =[\s\S]{0,1500}?;/)
  assert.ok(bucket && !bucket[0].includes('user_granted_ai_access'), 'bucket must not use the DB column name')
})

test('enable dialog polls while open regardless of indexing flag', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  assert.match(app, /Dialog-open polling: the Enable\/indexing dialog must stay/)
  assert.match(app, /if \(!showDSConfirm \|\| !token \|\| sessionExpired\) return undefined;/)
})

test('dialog auto-close arms only on its own ACTIVATE success', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // No blanket armer on indexing transitions: background runs must never
  // close a manually opened dialog mid-read.
  assert.doesNotMatch(app, /if \(indexingActive\) autoCloseRef\.current = true;/)
  // ACTIVATE arms after success and disarms on failure.
  assert.match(app, /await api\.enableAllEmbeddings\(\);[\s\S]{0,200}?autoCloseRef\.current = true;/)
  assert.match(app, /Failed to enable file indexing:[\s\S]{0,200}?autoCloseRef\.current = false;/)
  // Manual opens still clear the flag.
  assert.match(app, /opening the widget manually must always clear it[\s\S]{0,200}?autoCloseRef\.current = false;/)
})

test('video/audio skipped row expands to filenames like unsupported', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  assert.match(app, /Video\/Audio \(skipped\):/)
  assert.match(app, /mime\.startsWith\('audio\/'\) \|\| mime\.startsWith\('video\/'\)/)
  assert.match(app, /Audio\/video isn't parsed/)
})

test('bulk Move/Copy clears selection on success, preserves on failure', () => {
  // Fix 1: full success → empty set (caller also exits selection mode)
  assert.deepEqual([...nextSelectionAfterBulkOp(['file:1', 'file:2'], [])], [])
  assert.deepEqual([...nextSelectionAfterBulkOp(['file:1'], null)], [])
  // Fix 1: any failure → previous selection preserved for retry
  assert.deepEqual([...nextSelectionAfterBulkOp(['file:1', 'file:2'], ['file:2'])].sort(), ['file:1', 'file:2'])
  assert.deepEqual([...nextSelectionAfterBulkOp(['file:1'], ['file:1'])], ['file:1'])
  // inputs never mutated
  const prev = ['file:1']
  nextSelectionAfterBulkOp(prev, ['file:1'])
  assert.deepEqual(prev, ['file:1'])
})

test('folderAbsolutePath renders canonical /Root paths', () => {
  // Fix 3: root and unknown ids never invent paths
  assert.equal(folderAbsolutePath(null, []), '/Root')
  assert.equal(folderAbsolutePath(undefined, []), '/Root')
  assert.equal(folderAbsolutePath('', []), '/Root')
  assert.equal(folderAbsolutePath(999, [{ id: 2, name: '2', parent_id: null }]), '/Root')
  // Nested canonical path used by Upload/Move/Copy destinations and picker
  const folders = [
    { id: 2, name: '2', parent_id: null },
    { id: 23, name: '23', parent_id: 2 },
  ]
  assert.equal(folderAbsolutePath(2, folders), '/Root/2')
  assert.equal(folderAbsolutePath(23, folders), '/Root/2/23')
  assert.equal(folderAbsolutePath('23', folders), '/Root/2/23')
  // Cycle guard terminates
  const cyclic = [
    { id: 1, name: 'a', parent_id: 2 },
    { id: 2, name: 'b', parent_id: 1 },
  ]
  assert.equal(folderAbsolutePath(1, cyclic), '/Root/b/a')
})

test('admin RAG scope controls file scope and top-k independently', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // File Scope card with mandated copy; Top-K card with mandated copy.
  assert.match(app, /testid="admin-file-scope"/)
  assert.match(app, /maximum number of distinct files that can contribute to the final RAG context/)
  assert.match(app, /testid="admin-top-k"/)
  assert.match(app, /Top-K controls retrieved chunks, not the number of files/)
  // Dropdown presets with defaults marked; Custom entry opens a free input.
  assert.match(app, /5 files \(Recommended\)/)
  assert.match(app, /20 chunks \(Recommended\)/)
  assert.match(app, /value: 'custom', label: 'Custom/)
  assert.match(app, /testid="admin-file-scope-custom"/)
  assert.match(app, /testid="admin-top-k-custom"/)
  // Admin-only gating matches the Max-context pattern.
  assert.match(app, /userProfile\?\.is_admin && \(\s*<div className="p-4/)
  // Draft defaults (5 files, 20 chunks) and save payload carry both keys.
  assert.match(app, /file_scope: Number\(config\.file_scope\) \|\| 5/)
  assert.match(app, /top_k: Number\(config\.top_k\) \|\| 20/)
  assert.match(app, /file_scope: Number\(modelsDraft\.file_scope\) \|\| null/)
  assert.match(app, /top_k: Number\(modelsDraft\.top_k\) \|\| null/)
  // Custom-input clamps mirror backend validation (1-100, 1-500).
  assert.match(app, /type="number" min=\{1\} max=\{100\}/)
  assert.match(app, /type="number" min=\{1\} max=\{500\}/)
  // Web Search depth: own section, same dropdown style, draft + save wiring.
  // Conservative is the default but no longer tagged Recommended; Pro is the
  // top tier.
  assert.match(app, /testid="admin-search-depth"/)
  assert.match(app, /\{ value: 'conservative', label: 'Conservative' \}/)
  assert.doesNotMatch(app, /Conservative \(Recommended\)/)
  assert.match(app, /\{ value: 'pro', label: 'Pro' \}/)
  assert.match(app, /search_depth: config\.search_depth \|\| 'conservative'/)
  assert.match(app, /search_depth: modelsDraft\.search_depth \|\| null/)
  // Per-user storage quota editor in the expanded admin user panel.
  assert.match(app, /Storage quota/)
  assert.match(app, /storage_quota_gb: gb/)
  assert.match(app, /quotaDraft\[u\.id\]/)
  // Quota changes refresh the dashboard STORAGE box underneath the overlay.
  assert.match(app, /storage-changed/)
  assert.match(app, /refreshFilesRef\.current\(\{ showLoading: false \}\)/)
})

test('branding: gear admin logo, mobile name, session panel layout', async () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  // 2010s admin gear replaces the shield-with-tick everywhere (single icon def).
  assert.match(app, /Shield: \(\{ size = 24, className = "" \}\) => <svg[^>]*><circle cx="12" cy="12" r="3"/)
  assert.doesNotMatch(app, /M9 12l2 2 4-4m5\.618/)
  // Mobile top bar carries the product name beside the logo.
  assert.match(app, /alt="Lavix" className="h-7 w-auto filter brightness-110" \/>/)
  assert.match(app, /text-sm font-black tracking-tighter gradient-text">LAVIX VAULT/)
  // Session timeout panel: brand logo + name, buttons side by side.
  assert.match(app, /src="\/svg\/lavix\.svg" alt="Lavix Vault" className="w-16 h-16 object-contain mb-3"/)
  assert.match(app, /text-lg font-black tracking-tight gradient-text">LAVIX VAULT/)
  assert.match(app, /<div className="flex gap-3">/)
  // Trash grid view renders folder cards (error #130 fix): Folder icon defined.
  assert.match(app, /Folder: \(\{ size = 20, className = "" \}\) => <svg/)
  // Preferences uses the sliders icon, distinct from the gear.
  assert.match(app, /id: 'prefs', label: 'Preferences', icon: Icons\.Sliders/)
  // Details panel: click-away dismiss + CHAT card descriptive text.
  assert.match(app, /ref=\{panelRef\}/)
  assert.match(app, /document\.addEventListener\('mousedown', onDocMouseDown\)/)
  assert.match(app, /Talk about this file/)
  // Folder picker row: Tag button renders after the folder name (right side),
  // not before the folder icon.
  assert.match(app, /allFolders\.map\(\(file\) => \{\s*const flatIdx = mentionResults\.flatIndexMap\[file\.id\];\s*const isExpanded = expandedFolderId === file\._folderId;\s*return \(\s*<React\.Fragment key=\{file\.id\}>\s*<button\s*data-mention-idx=\{flatIdx\}\s*onClick=\{\(\) => selectMention\(file\)\}/)
})

#!/usr/bin/env node

/**
 * Dependency-free Chromium smoke test for a running Lavix WebUI.
 *
 * This deliberately stays anonymous and read-only. It proves that the SPA can
 * mount without an uncaught browser exception and that the bundled PDF.js
 * module/worker chain is reachable with a browser-safe MIME type. It is a
 * baseline smoke check, not the authenticated release-acceptance suite.
 *
 * Usage:
 *   node scripts/test-browser-smoke.mjs [http://127.0.0.1:3005]
 *
 * Optional environment:
 *   CHROME_BIN=/path/to/chromium
 *   LAVIX_BROWSER_TIMEOUT_MS=30000
 */

import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'

const timeoutMs = Number.parseInt(process.env.LAVIX_BROWSER_TIMEOUT_MS || '30000', 10)
if (!Number.isFinite(timeoutMs) || timeoutMs < 1000) {
  throw new Error('LAVIX_BROWSER_TIMEOUT_MS must be an integer of at least 1000')
}

const requestedUrl = new URL(process.argv[2] || 'http://127.0.0.1:3005/')
if (!['http:', 'https:'].includes(requestedUrl.protocol)) {
  throw new Error('WebUI URL must use http or https')
}
if (requestedUrl.username || requestedUrl.password) {
  throw new Error('Do not put credentials in the browser-smoke URL')
}
requestedUrl.hash = ''
const displayUrl = `${requestedUrl.origin}${requestedUrl.pathname}`

function findChrome() {
  const candidates = [
    process.env.CHROME_BIN,
    '/usr/bin/google-chrome-stable',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ].filter(Boolean)
  const executable = candidates.find((candidate) => existsSync(candidate))
  if (!executable) {
    throw new Error('Chromium/Chrome was not found; set CHROME_BIN explicitly')
  }
  return executable
}

function withTimeout(promise, label) {
  let timer
  const expired = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs} ms`)), timeoutMs)
  })
  return Promise.race([promise, expired]).finally(() => clearTimeout(timer))
}

class CdpClient {
  constructor(webSocket) {
    this.webSocket = webSocket
    this.nextId = 1
    this.pending = new Map()
    this.listeners = new Set()

    webSocket.addEventListener('message', (event) => {
      const message = JSON.parse(String(event.data))
      if (message.id) {
        const pending = this.pending.get(message.id)
        if (!pending) return
        this.pending.delete(message.id)
        if (message.error) {
          pending.reject(new Error(`${pending.method}: ${message.error.message}`))
        } else {
          pending.resolve(message.result || {})
        }
        return
      }
      for (const listener of this.listeners) listener(message)
    })

    webSocket.addEventListener('close', () => {
      for (const pending of this.pending.values()) {
        pending.reject(new Error('Chrome DevTools connection closed unexpectedly'))
      }
      this.pending.clear()
    })
  }

  send(method, params = {}, sessionId = undefined) {
    const id = this.nextId++
    return withTimeout(new Promise((resolve, reject) => {
      this.pending.set(id, { method, resolve, reject })
      this.webSocket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }))
    }), method)
  }

  onEvent(listener) {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  waitFor(method, sessionId) {
    return withTimeout(new Promise((resolve) => {
      const unsubscribe = this.onEvent((message) => {
        if (message.method !== method || message.sessionId !== sessionId) return
        unsubscribe()
        resolve(message.params || {})
      })
    }), method)
  }
}

async function launchChrome(userDataDir) {
  const executable = findChrome()
  const browser = spawn(executable, [
    '--headless=new',
    '--remote-debugging-port=0',
    `--user-data-dir=${userDataDir}`,
    '--disable-background-networking',
    '--disable-component-update',
    '--disable-default-apps',
    '--disable-dev-shm-usage',
    '--disable-gpu',
    '--disable-sync',
    '--metrics-recording-only',
    '--mute-audio',
    '--no-default-browser-check',
    '--no-first-run',
    'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe'] })

  let stderr = ''
  const webSocketUrl = await withTimeout(new Promise((resolve, reject) => {
    browser.stderr.setEncoding('utf8')
    browser.stderr.on('data', (chunk) => {
      stderr = `${stderr}${chunk}`.slice(-12000)
      const match = stderr.match(/DevTools listening on (ws:\/\/\S+)/)
      if (match) resolve(match[1])
    })
    browser.once('error', reject)
    browser.once('exit', (code) => {
      reject(new Error(`Chrome exited before DevTools was ready (code ${code})\n${stderr}`))
    })
  }), 'Chrome startup')

  return { browser, executable, webSocketUrl }
}

async function connectCdp(webSocketUrl) {
  const webSocket = new WebSocket(webSocketUrl)
  await withTimeout(new Promise((resolve, reject) => {
    webSocket.addEventListener('open', resolve, { once: true })
    webSocket.addEventListener('error', () => reject(new Error('Could not connect to Chrome DevTools')), { once: true })
  }), 'Chrome DevTools connection')
  return { webSocket, cdp: new CdpClient(webSocket) }
}

function assetReferences(source, sourceUrl) {
  const references = new Set()
  const patterns = [
    /(?:src|href)=["']([^"']+\.(?:m?js))(?:\?[^"']*)?["']/g,
    /\bimport\s*(?:\(\s*)?["'`]([^"'`]+\.(?:m?js))(?:\?[^"'`]*)?["'`]/g,
    /\bfrom\s*["'`]([^"'`]+\.(?:m?js))(?:\?[^"'`]*)?["'`]/g,
    /["'`](\/assets\/pdf\.worker\.min-[^"'`/]+\.mjs)(?:\?[^"'`]*)?["'`]/g,
  ]
  for (const pattern of patterns) {
    for (const match of source.matchAll(pattern)) {
      const candidate = new URL(match[1], sourceUrl)
      if (candidate.origin === requestedUrl.origin && candidate.pathname.startsWith('/assets/')) {
        references.add(candidate.href)
      }
    }
  }
  return references
}

async function verifyPdfAssets(html) {
  const queue = [...assetReferences(html, requestedUrl)]
  const visited = new Map()

  while (queue.length) {
    if (visited.size >= 80) throw new Error('Static asset crawl exceeded its safety bound')
    const assetUrl = queue.shift()
    if (visited.has(assetUrl)) continue
    const response = await fetch(assetUrl, { redirect: 'error' })
    if (!response.ok) throw new Error(`Static asset returned HTTP ${response.status}: ${new URL(assetUrl).pathname}`)
    const body = await response.text()
    if (!body.length) throw new Error(`Static asset was empty: ${new URL(assetUrl).pathname}`)
    visited.set(assetUrl, response.headers.get('content-type') || '')
    for (const reference of assetReferences(body, assetUrl)) {
      if (!visited.has(reference)) queue.push(reference)
    }
  }

  const pdfAssets = [...visited.entries()].filter(([assetUrl]) => /\/pdf(?:\.worker\.min)?-[^/]+\.(?:m?js)$/.test(new URL(assetUrl).pathname))
  const pdfModule = pdfAssets.find(([assetUrl]) => /\/pdf-[^/]+\.js$/.test(new URL(assetUrl).pathname))
  const pdfWorker = pdfAssets.find(([assetUrl]) => /\/pdf\.worker\.min-[^/]+\.mjs$/.test(new URL(assetUrl).pathname))
  if (!pdfModule) throw new Error('The built PDF.js module could not be discovered from the application bundle')
  if (!pdfWorker) throw new Error('The built PDF.js module worker could not be discovered from the application bundle')

  const workerType = pdfWorker[1].split(';', 1)[0].trim().toLowerCase()
  if (!['application/javascript', 'text/javascript'].includes(workerType)) {
    throw new Error(`PDF.js module worker has unsafe MIME type: ${pdfWorker[1] || '(missing)'}`)
  }

  return {
    crawled: visited.size,
    module: new URL(pdfModule[0]).pathname,
    worker: new URL(pdfWorker[0]).pathname,
    workerType,
  }
}

async function verifyBrowserPage(cdp) {
  const { browserContextId } = await cdp.send('Target.createBrowserContext')
  const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank', browserContextId })
  const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true })
  const exceptions = []
  const javascriptErrors = []
  const failedStaticRequests = []
  const staticTypes = new Set(['Document', 'Script', 'Stylesheet', 'Font', 'Image', 'Worker'])
  const unsubscribe = cdp.onEvent((message) => {
    if (message.sessionId !== sessionId) return
    if (message.method === 'Runtime.exceptionThrown') {
      exceptions.push(message.params.exceptionDetails?.text || 'Uncaught browser exception')
    }
    if (message.method === 'Log.entryAdded' && message.params.entry?.level === 'error' && message.params.entry?.source === 'javascript') {
      javascriptErrors.push(message.params.entry.text || 'JavaScript log error')
    }
    if (message.method === 'Network.loadingFailed' && staticTypes.has(message.params.type) && !message.params.canceled) {
      failedStaticRequests.push(`${message.params.type}: ${message.params.errorText}`)
    }
    if (message.method === 'Network.responseReceived' && staticTypes.has(message.params.type) && message.params.response?.status >= 400) {
      const failedUrl = new URL(message.params.response.url)
      failedStaticRequests.push(`${message.params.type}: HTTP ${message.params.response.status} ${failedUrl.pathname}`)
    }
  })

  try {
    await Promise.all([
      cdp.send('Page.enable', {}, sessionId),
      cdp.send('Runtime.enable', {}, sessionId),
      cdp.send('Log.enable', {}, sessionId),
      cdp.send('Network.enable', {}, sessionId),
    ])
    const loaded = cdp.waitFor('Page.loadEventFired', sessionId)
    const navigation = await cdp.send('Page.navigate', { url: requestedUrl.href }, sessionId)
    if (navigation.errorText) throw new Error(`Browser navigation failed: ${navigation.errorText}`)
    await loaded
    await delay(2500)

    const evaluated = await cdp.send('Runtime.evaluate', {
      expression: `(() => {
        const root = document.getElementById('root')
        const text = root?.innerText?.trim() || ''
        return {
          readyState: document.readyState,
          title: document.title,
          path: location.pathname,
          rootChildren: root?.childElementCount ?? 0,
          rootTextLength: text.length,
          errorBoundaryVisible: text.includes('Application Error') && text.includes('Something went wrong'),
        }
      })()`,
      returnByValue: true,
    }, sessionId)
    if (evaluated.exceptionDetails) {
      throw new Error(`Browser state evaluation failed: ${evaluated.exceptionDetails.text}`)
    }
    const state = evaluated.result?.value
    if (!state || state.readyState !== 'complete') throw new Error('The browser document did not reach readyState=complete')
    if (state.rootChildren < 1 || state.rootTextLength < 1) throw new Error('The React root mounted no visible application UI')
    if (state.errorBoundaryVisible) throw new Error('The Lavix application error boundary is visible')
    if (exceptions.length) throw new Error(`Uncaught browser exception: ${exceptions.join(' | ')}`)
    if (javascriptErrors.length) throw new Error(`Browser JavaScript error: ${javascriptErrors.join(' | ')}`)
    if (failedStaticRequests.length) throw new Error(`Static browser request failed: ${failedStaticRequests.join(' | ')}`)
    return state
  } finally {
    unsubscribe()
    await cdp.send('Target.closeTarget', { targetId }).catch(() => {})
    await cdp.send('Target.disposeBrowserContext', { browserContextId }).catch(() => {})
  }
}

async function main() {
  const indexResponse = await fetch(requestedUrl, { redirect: 'error' })
  if (!indexResponse.ok) throw new Error(`WebUI returned HTTP ${indexResponse.status}`)
  const html = await indexResponse.text()
  if (!html.includes('id="root"')) throw new Error('WebUI response is not the Lavix application shell')

  const pdfAssets = await verifyPdfAssets(html)
  const profileDir = await mkdtemp(path.join(tmpdir(), 'lavix-browser-smoke-'))
  let browser
  let webSocket
  try {
    const launched = await launchChrome(profileDir)
    browser = launched.browser
    const connected = await connectCdp(launched.webSocketUrl)
    webSocket = connected.webSocket
    const page = await verifyBrowserPage(connected.cdp)
    console.log(`PASS anonymous Lavix browser smoke: ${displayUrl}`)
    console.log(`  page: ${page.title} (${page.path}), root text ${page.rootTextLength} characters`)
    console.log(`  PDF.js: ${pdfAssets.module}`)
    console.log(`  worker: ${pdfAssets.worker} (${pdfAssets.workerType})`)
    console.log(`  static module graph: ${pdfAssets.crawled} JavaScript assets reachable`)
  } finally {
    if (webSocket?.readyState === WebSocket.OPEN) webSocket.close()
    if (browser && browser.exitCode === null) {
      browser.kill('SIGTERM')
      await Promise.race([
        new Promise((resolve) => browser.once('exit', resolve)),
        delay(2000).then(() => browser.kill('SIGKILL')),
      ])
    }
    await rm(profileDir, { recursive: true, force: true })
  }
}

main().catch((error) => {
  console.error(`FAIL browser smoke: ${error.message}`)
  process.exitCode = 1
})

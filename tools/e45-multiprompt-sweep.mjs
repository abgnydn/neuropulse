#!/usr/bin/env node
// E45 Phase 2 multi-prompt sweep.
//
// Drives a headed Chromium via Playwright (WebGPU on macOS Metal works
// in headed mode — see tests/butterfly-sweep.spec.mjs). Uses the
// dev-only `__e45.sweep` harness exposed by src/main.ts.
//
// For each prompt, runs fixedpoint at iter ∈ DEFAULT_ITERS and writes
// one JSON artifact to tests/results/YYYY-MM-DD/E45-phase2-multiprompt.json.
//
// Why a script and not a Playwright `test`: this is a one-shot batch
// experiment, not a regression test. The artifact is the deliverable.
//
// Usage:
//   # Make sure `npm run dev` is up on :4000.
//   node tools/e45-multiprompt-sweep.mjs                 # default 16 prompts × 4 iters
//   ITERS=1,2,3,5,10  node tools/e45-multiprompt-sweep.mjs
//   PROMPTS_FILE=./my-prompts.json  node tools/e45-multiprompt-sweep.mjs
//   MAX_TOKENS=20  node tools/e45-multiprompt-sweep.mjs
//   BASE_URL=http://localhost:4000  HEADLESS=0  node tools/e45-multiprompt-sweep.mjs

import { chromium } from '@playwright/test'
import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = join(__dirname, '..')

const BASE_URL    = process.env.BASE_URL    || 'http://localhost:4000'
const MAX_TOKENS  = parseInt(process.env.MAX_TOKENS || '12', 10)
const ITERS       = (process.env.ITERS || '1,2,3,10').split(',').map((s) => parseInt(s.trim(), 10))
const HEADLESS    = process.env.HEADLESS === '1'   // default: headed (WebGPU works reliably this way on macOS)
const USER_DATA_DIR = process.env.USER_DATA_DIR || join(REPO_ROOT, '.playwright-profile-e45')
// First run downloads ~2 GB of Phi-3 weights into the OPFS cache inside the
// persistent userDataDir. Subsequent runs reuse the cache and start in ~30s.
const ENGINE_READY_TIMEOUT_MS = parseInt(process.env.ENGINE_READY_TIMEOUT_MS || '900000', 10)  // 15 min default
const PER_GEN_TIMEOUT_MS      = 120_000             // wall-clock for one engine.generate() call

// 15-prompt validation set + the Phase-2 anchor. Mirrors the in-app HF
// cross-validation suite's prompt list so we can compare apples to apples.
const DEFAULT_PROMPTS = [
  'The capital of Japan is',
  'Hello, world!',
  'What is 2 + 2?',
  'Name three colors.',
  'Write a Python function to reverse a string',
  'Explain gravity in one sentence.',
  'The quick brown fox jumps over the lazy',
  'こんにちは、元気ですか?',
  '🚀 rocket emoji test',
  'Parse JSON: {"key": 42, "list": [1,2,3]}',
  'Why is the sky blue?',
  'List 5 fruits.',
  'Translate hello to French.',
  'What is 15 percent of 80?',
  'Who wrote Hamlet?',
  'Define recursion.',
]
const PROMPTS = process.env.PROMPTS_FILE
  ? JSON.parse(readFileSync(process.env.PROMPTS_FILE, 'utf8'))
  : DEFAULT_PROMPTS

const today = new Date().toISOString().slice(0, 10)
const OUT_DIR  = join(REPO_ROOT, 'tests', 'results', today)
const OUT_PATH = join(OUT_DIR, 'E45-phase2-multiprompt.json')
mkdirSync(OUT_DIR, { recursive: true })

// Use launchPersistentContext so OPFS / Cache API survive across runs.
// First run downloads ~2 GB of Phi-3 weights; subsequent runs reuse.
// WebGPU flags: headed Chromium on macOS Metal usually exposes navigator.gpu
// out of the box, but pass --enable-unsafe-webgpu as a belt-and-braces signal.
mkdirSync(USER_DATA_DIR, { recursive: true })
console.log(`[e45-sweep] persistent userDataDir: ${USER_DATA_DIR}`)
const ctx = await chromium.launchPersistentContext(USER_DATA_DIR, {
  headless: HEADLESS,
  viewport: { width: 1280, height: 800 },
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan,WebGPU',
    '--disable-features=DialMediaRouteProvider',
  ],
})
const page = ctx.pages()[0] || (await ctx.newPage())

const pageErrors = []
let bootVerbose = true  // log every browser console line during engine init so we can see weight-loader progress
page.on('pageerror', (e) => { pageErrors.push(e.message); console.error('[pageerror]', e.message) })
page.on('console', (m) => {
  const t = m.type()
  const txt = m.text()
  if (bootVerbose) {
    if (t !== 'debug') console.log(`[browser ${t}] ${txt.slice(0, 240)}`)
  } else {
    if (t === 'error' || t === 'warning') console.log(`[browser ${t}] ${txt.slice(0, 200)}`)
    if (txt.startsWith('[E45')) console.log(`[browser]      ${txt.slice(0, 200)}`)
  }
})

const url = `${BASE_URL}/app/?attn=fixedpoint&max_iter=1&noauto`
console.log(`[e45-sweep] launching ${HEADLESS ? 'headless' : 'headed'} chromium`)
console.log(`[e45-sweep] navigating to ${url}`)
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 })

// If this profile has no cached weights yet, the page shows a "Download & run"
// gate that the user normally clicks once. In a self-driving script we click
// it for them. After the first run, OPFS holds the weights and this button
// won't appear on subsequent runs against the same userDataDir.
try {
  const dlBtn = page.getByRole('button', { name: /Download.*run/i })
  await dlBtn.waitFor({ state: 'visible', timeout: 8_000 })
  console.log('[e45-sweep] cold profile — clicking "Download & run" (first-time ~2 GB)')
  await dlBtn.click()
} catch {
  console.log('[e45-sweep] no download gate detected — weights already cached')
}

console.log(`[e45-sweep] waiting for engine + __e45 (up to ${ENGINE_READY_TIMEOUT_MS / 1000}s)…`)
// NB: page.waitForFunction signature is (fn, arg, options). Passing options as
// the second arg silently falls back to the default 30s timeout — which is
// way too short for Phi-3 first-load (~30-60s cached, ~3min cold). Always
// pass `undefined` as arg explicitly.
await page.waitForFunction(() => typeof globalThis.__e45 !== 'undefined', undefined, {
  timeout: ENGINE_READY_TIMEOUT_MS,
  polling: 1000,
})
bootVerbose = false  // engine is up; quiet down the per-token telemetry
console.log('[e45-sweep] engine ready. starting sweep…')
// Long per-generate timeout — at high max_iter the call can take ~minute.
page.setDefaultTimeout(PER_GEN_TIMEOUT_MS)

const startedAt = new Date().toISOString()
const t0 = Date.now()
const results = []

for (let i = 0; i < PROMPTS.length; i++) {
  const prompt = PROMPTS[i]
  const sweep = []
  console.log(`[e45-sweep] (${i + 1}/${PROMPTS.length}) ${JSON.stringify(prompt).slice(0, 60)}`)
  for (const iter of ITERS) {
    // NB: page.evaluate is (fn, arg) — no options. Per-call timeout is set
    // page-wide via setDefaultTimeout above the loop.
    const run = await page.evaluate(
      async ({ p, n, it }) => {
        const eng = globalThis.__e45.engine
        eng.e45Config.attentionKernel = it === 0 ? 'standard' : 'fixedpoint'
        if (it > 0) eng.e45Config.fixedPointMaxIter = it
        const tokens = []
        const t0 = performance.now()
        const text = await eng.generate(p, n, { onToken: (t, id) => tokens.push({ id, t }) })
        return { iter: it, kernel: eng.e45Config.attentionKernel, text, ms: (performance.now() - t0) | 0, tokenIds: tokens.map((x) => x.id) }
      },
      { p: prompt, n: MAX_TOKENS, it: iter },
    )
    console.log(`[e45-sweep]   iter=${iter}  ${run.ms.toString().padStart(5)}ms  ${JSON.stringify(run.text).slice(0, 70)}`)
    sweep.push(run)
  }
  results.push({ prompt, sweep })

  // Incremental save after each prompt — if the run dies mid-sweep,
  // we still have everything up to here.
  writeFileSync(
    OUT_PATH,
    JSON.stringify(
      {
        experimentId: 'E45',
        predictionId: 'P-20260526-07',
        phase: '2-multiprompt',
        baseUrl: BASE_URL,
        iters: ITERS,
        maxNewTokens: MAX_TOKENS,
        promptsTotal: PROMPTS.length,
        promptsDone: results.length,
        startedAt,
        doneAt: results.length === PROMPTS.length ? new Date().toISOString() : null,
        elapsedSec: Math.round((Date.now() - t0) / 1000),
        results,
        pageErrors,
      },
      null,
      2,
    ),
  )
}

const elapsedSec = Math.round((Date.now() - t0) / 1000)
console.log(`[e45-sweep] all ${PROMPTS.length} prompts complete in ${elapsedSec}s`)

// Compact summary table — easy eyeball check for the cliff per prompt.
console.log('')
console.log('Cliff summary (per prompt, per iter — first 60 chars):')
console.log('')
for (const r of results) {
  console.log(`  prompt: ${JSON.stringify(r.prompt).slice(0, 60)}`)
  for (const s of r.sweep) {
    console.log(`    iter=${String(s.iter).padStart(3)}  ${JSON.stringify(s.text).slice(0, 70)}`)
  }
  console.log('')
}

console.log(`[e45-sweep] artifact: ${OUT_PATH}`)

await ctx.close()

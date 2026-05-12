import { test, expect } from '@playwright/test'
import { mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

// Butterfly v2.5 — pre-registered scaling sweep (PREDICTIONS.md P-20260512-05).
// Drives the 4 built-in transcripts × N runs each, captures every verdict
// pair to results/butterfly-sweep-<ts>.json, and the localStorage tally
// for cross-check.
//
// WHY this isn't an `npm test` default: each run takes 25-50s of real
// WebGPU inference. Full default sweep (4 × 20 = 80 runs) ≈ 60 min.
// Run with:
//   RUNS_PER=2  npx playwright test butterfly-sweep    (~10 min smoke)
//   RUNS_PER=20 npx playwright test butterfly-sweep    (full sweep, ~60 min)
//
// Requires:
//   - Dev server on :4000 (or set BASE_URL).
//   - Real WebGPU. Chromium headed mode on macOS works via Metal.
//   - Phi-3 weights either cached or local-served from ~/mlc-weights.

const __dirname = dirname(fileURLToPath(import.meta.url))
const RESULTS_DIR = join(__dirname, '..', 'test-results', 'butterfly-sweep')
const RUNS_PER = parseInt(process.env.RUNS_PER || '20', 10)
const TRANSCRIPT_IDS = (process.env.TRANSCRIPTS || 'jwt-clock-race,auth-owner-pto,rate-limit-decision,cache-race-fileline').split(',')
const PER_RUN_TIMEOUT_MS = 240_000          // headed Playwright runs are slower than user's daily Chrome
const ENGINE_READY_TIMEOUT_MS = 180_000

test.describe.configure({ mode: 'serial' })

test('butterfly sweep — 4 transcripts × N runs', async ({ page, browser }, testInfo) => {
  test.setTimeout(TRANSCRIPT_IDS.length * RUNS_PER * PER_RUN_TIMEOUT_MS + ENGINE_READY_TIMEOUT_MS)

  // Capture pageerrors + browser console for postmortem. During the
  // engine-init phase we log EVERYTHING (including info/log) so the
  // sweep operator can see weight-loader progress + shader compile.
  const pageErrors = []
  let verboseConsole = true
  page.on('pageerror', (e) => { pageErrors.push(e.message); console.error('[pageerror]', e.message) })
  page.on('console', (msg) => {
    const t = msg.type()
    if (verboseConsole || t === 'error' || t === 'warning') {
      const txt = msg.text()
      // skip noisy stuff
      if (txt.includes('DevTools')) return
      console.log(`[browser ${t}] ${txt.slice(0, 240)}`)
    }
  })

  // Status-snapshot helper — dumps page state every 15s during engine init
  // so we don't fly blind through a 3-minute boot.
  const snapshot = setInterval(async () => {
    try {
      const s = await page.evaluate(() => ({
        title: document.title,
        loadingVisible: !!document.querySelector('#loadingOverlay:not([hidden])'),
        loadingText: document.querySelector('#loadingOverlay')?.innerText?.slice(0, 200) || null,
        bflyPickerExists: !!document.getElementById('bflyTranscriptPicker'),
        toggleFnExists: typeof window.__toggleButterflyPanel === 'function',
        navGpu: !!navigator.gpu,
      }))
      console.log(`[snapshot] ${JSON.stringify(s)}`)
    } catch (_e) { /* page may not be ready yet */ }
  }, 15_000)

  // ── Engine warm-up ──────────────────────────────────────────────
  console.log(`\n[sweep] opening /app/ and waiting for engine init…`)
  await page.goto('/app/', { waitUntil: 'domcontentloaded' })

  // Suppress the first-visit welcome modal up-front by setting the
  // dismissed flag in localStorage BEFORE the app boots. Without this,
  // the welcome overlay intercepts pointer events and the Run button
  // click bounces. Also handles the boot gate (#bootGoBtn).
  await page.addInitScript(() => {
    try { localStorage.setItem('np:welcome-dismissed', '1') } catch {}
  })
  // Re-navigate so the init script runs before any app code.
  await page.goto('/app/', { waitUntil: 'domcontentloaded' })

  try {
    await page.locator('#bootGoBtn').click({ timeout: 5_000 })
    console.log('[sweep] clicked first-visit gate — engine init beginning')
  } catch {
    console.log('[sweep] no first-visit gate (cache hit) — engine init beginning')
  }
  // Defensive: if the welcome overlay still rendered (e.g. flag set too
  // late), click its dismiss button.
  try {
    await page.locator('#welcome-dismiss').click({ timeout: 3_000 })
    console.log('[sweep] dismissed welcome overlay')
  } catch { /* not present or already hidden — fine */ }

  // Wait for butterfly init + the dropdown to exist
  await page.waitForFunction(
    () => typeof window.__toggleButterflyPanel === 'function' && document.getElementById('bflyTranscriptPicker'),
    { timeout: ENGINE_READY_TIMEOUT_MS },
  )

  // Open the panel
  await page.evaluate(() => window.__toggleButterflyPanel())
  await expect(page.locator('.bfly-panel')).toBeVisible({ timeout: 5_000 })

  // Engine readiness: try a Run and watch for the "Engine not ready" message.
  // The panel shows a friendly status when engine isn't up yet, so we poll
  // until the engine is genuinely loaded (status doesn't say "not ready").
  console.log(`[sweep] verifying engine is ready (waiting up to ${ENGINE_READY_TIMEOUT_MS/1000}s)…`)
  await page.waitForFunction(
    () => {
      const s = document.getElementById('bflyStatus')?.textContent || ''
      // The panel's resetUI() sets a "Ready. Press Run…" message after
      // initial bind. Engine init separately gates the actual Run handler.
      return !s.includes('not ready') && document.querySelector('#bflyRunBtn:not([disabled])')
    },
    { timeout: ENGINE_READY_TIMEOUT_MS },
  )

  // Engine + panel are ready — quiet down the console firehose.
  clearInterval(snapshot)
  verboseConsole = false
  console.log(`[sweep] engine + panel ready — switching to per-run mode\n`)

  // Reset any prior tally so this sweep is a clean baseline.
  await page.evaluate(() => localStorage.removeItem('butterfly-mode-stats-v1'))

  console.log(`[sweep] starting: ${TRANSCRIPT_IDS.length} transcripts × ${RUNS_PER} runs = ${TRANSCRIPT_IDS.length * RUNS_PER} runs`)

  const results = {
    started_at: new Date().toISOString(),
    runs_per_transcript: RUNS_PER,
    transcripts: TRANSCRIPT_IDS,
    fingerprint: await page.evaluate(() => ({
      ua: navigator.userAgent,
      ts: Date.now(),
      sha: document.getElementById('fp-sha')?.textContent || null,
      gpu: document.getElementById('fp-gpu')?.textContent || null,
    })),
    runs: [],
  }

  for (const tid of TRANSCRIPT_IDS) {
    console.log(`\n[sweep] ── transcript: ${tid} ──`)
    // Select transcript
    await page.locator('#bflyTranscriptPicker').selectOption(tid)
    await page.waitForTimeout(200)

    for (let i = 1; i <= RUNS_PER; i++) {
      const t0 = Date.now()
      console.log(`[sweep]   run ${i}/${RUNS_PER}…`)

      // Click Run, then wait for it to re-enable. The runBtn becomes
      // disabled at start and re-enables once both arms + both judges
      // are done (see butterfly-mode.ts line ~1031).
      await page.locator('#bflyRunBtn').click()
      // Brief wait so the button has a chance to enter disabled state.
      await page.waitForFunction(() => document.querySelector('#bflyRunBtn[disabled]'), { timeout: 5_000 }).catch(() => {})

      // Per-run status snapshots every 20s — tells us what stage the run
      // is in (tagging gen N, chrysalis, judging, etc). Silent runs that
      // hang past PER_RUN_TIMEOUT_MS are diagnosable from these breadcrumbs.
      const runSnapshot = setInterval(async () => {
        try {
          const s = await page.evaluate(() => document.getElementById('bflyStatus')?.textContent || '')
          console.log(`[sweep]     status: ${s.slice(0, 120)}`)
        } catch {}
      }, 20_000)

      try {
        await page.waitForFunction(
          () => document.querySelector('#bflyRunBtn:not([disabled])'),
          { timeout: PER_RUN_TIMEOUT_MS },
        )
      } finally {
        clearInterval(runSnapshot)
      }

      // Pull verdicts + status off the DOM.
      const observed = await page.evaluate(() => {
        const vb = (document.getElementById('bflyVerdictBfly')?.className || '').match(/\b(hit|partial|miss)\b/)?.[1] || ''
        const vl = (document.getElementById('bflyVerdictLastn')?.className || '').match(/\b(hit|partial|miss)\b/)?.[1] || ''
        const status = document.getElementById('bflyStatus')?.textContent || ''
        const ansBfly = document.getElementById('bflyAnsBfly')?.textContent || ''
        const ansLastn = document.getElementById('bflyAnsLastn')?.textContent || ''
        return { vb, vl, status, ansBflyLen: ansBfly.length, ansLastnLen: ansLastn.length }
      })

      const dt = ((Date.now() - t0) / 1000).toFixed(1)
      results.runs.push({
        transcript: tid,
        run: i,
        bfly: observed.vb || 'miss',
        lastn: observed.vl || 'miss',
        seconds: parseFloat(dt),
        status: observed.status.slice(0, 200),
      })
      console.log(`[sweep]     bfly=${observed.vb || '∅'} · lastN=${observed.vl || '∅'} · ${dt}s`)
    }
  }

  // Pull the localStorage tally too — cross-check against our DOM scrape.
  const localTally = await page.evaluate(() => {
    try { return JSON.parse(localStorage.getItem('butterfly-mode-stats-v1') || '[]') }
    catch { return [] }
  })
  results.local_tally = localTally
  results.finished_at = new Date().toISOString()
  results.page_errors = pageErrors

  mkdirSync(RESULTS_DIR, { recursive: true })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const outPath = join(RESULTS_DIR, `butterfly-sweep-${stamp}.json`)
  writeFileSync(outPath, JSON.stringify(results, null, 2))
  console.log(`\n[sweep] wrote ${outPath}`)
  console.log(`[sweep] ${results.runs.length} runs captured · ${pageErrors.length} pageerrors`)
})

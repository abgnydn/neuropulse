// Butterfly v2.5 sweep — paste-in-console driver.
//
// PURPOSE: programmatically launched Chrome (Playwright, raw CDP) does NOT
// expose navigator.gpu on macOS, so the Playwright sweep in
// tools/run-butterfly-sweep.mjs cannot start the engine in our env. Your
// already-running Chrome at neuropulse.live DOES have WebGPU, so we
// just drive the run loop directly inside that page.
//
// HOW TO RUN
//  1. Open https://neuropulse.live/app/  (or http://localhost:5173/app/
//     if you're testing locally) — wait until the Phi-3 engine is loaded
//     (the prompt input becomes responsive; ~10-30s on a hot cache).
//  2. Open DevTools (Cmd+Opt+I) → Console.
//  3. Paste this whole file. Then call:
//        butterflySweep({ runsPer: 20 })          // full 80-run sweep
//        butterflySweep({ runsPer: 1 })           // smoke
//        butterflySweep({ runsPer: 20, transcripts: ['jwt-clock-race'] })
//  4. Leave the tab alone (foreground!) for ~60 min on a 20-run sweep.
//     Background tabs throttle requestAnimationFrame and pause inference.
//  5. When done the function returns the results object AND triggers a
//     download of butterfly-sweep-<timestamp>.json. Paste that JSON back
//     to me and I'll grade it against PREDICTIONS.md P-20260512-05.

(function () {
  const TRANSCRIPT_IDS = ['jwt-clock-race', 'auth-owner-pto', 'rate-limit-decision', 'cache-race-fileline']

  function waitFor(predicate, { timeout = 120000, interval = 200 } = {}) {
    const start = Date.now()
    return new Promise((resolve, reject) => {
      const tick = () => {
        try {
          const v = predicate()
          if (v) return resolve(v)
        } catch (_e) { /* ignore */ }
        if (Date.now() - start > timeout) return reject(new Error(`waitFor timeout after ${timeout}ms`))
        setTimeout(tick, interval)
      }
      tick()
    })
  }

  async function openPanel() {
    if (typeof window.__toggleButterflyPanel !== 'function') {
      throw new Error('Butterfly panel not initialized. Refresh the app page and wait for engine ready before pasting.')
    }
    if (!document.querySelector('.bfly-panel.open')) window.__toggleButterflyPanel()
    await waitFor(() => document.getElementById('bflyTranscriptPicker'), { timeout: 10000 })
  }

  function pickTranscript(id) {
    const sel = document.getElementById('bflyTranscriptPicker')
    sel.value = id
    sel.dispatchEvent(new Event('change', { bubbles: true }))
  }

  function readVerdict(el) {
    const m = (el?.className || '').match(/\b(hit|partial|miss)\b/)
    return m ? m[1] : ''
  }

  async function singleRun(transcriptId) {
    pickTranscript(transcriptId)
    await new Promise((r) => setTimeout(r, 200))
    const runBtn = document.getElementById('bflyRunBtn')
    if (!runBtn || runBtn.disabled) throw new Error('Run button disabled — engine still loading or another run in flight')
    runBtn.click()
    // Wait for runBtn to enter disabled state…
    await waitFor(() => runBtn.disabled, { timeout: 5000 }).catch(() => {})
    // …then re-enable.
    await waitFor(() => !runBtn.disabled, { timeout: 180000 })
    return {
      bfly: readVerdict(document.getElementById('bflyVerdictBfly')) || 'miss',
      lastn: readVerdict(document.getElementById('bflyVerdictLastn')) || 'miss',
      ansBflyLen: (document.getElementById('bflyAnsBfly')?.textContent || '').length,
      ansLastnLen: (document.getElementById('bflyAnsLastn')?.textContent || '').length,
      status: (document.getElementById('bflyStatus')?.textContent || '').slice(0, 240),
    }
  }

  window.butterflySweep = async function butterflySweep({ runsPer = 20, transcripts = TRANSCRIPT_IDS, resetTally = true } = {}) {
    if (transcripts.some((t) => !TRANSCRIPT_IDS.includes(t))) {
      throw new Error(`Unknown transcript id. Valid: ${TRANSCRIPT_IDS.join(', ')}`)
    }
    await openPanel()
    if (resetTally) localStorage.removeItem('butterfly-mode-stats-v1')

    const results = {
      started_at: new Date().toISOString(),
      runs_per_transcript: runsPer,
      transcripts,
      fingerprint: {
        ua: navigator.userAgent,
        sha: document.getElementById('fp-sha')?.textContent || null,
        gpu: document.getElementById('fp-gpu')?.textContent || null,
      },
      runs: [],
    }

    const total = transcripts.length * runsPer
    let done = 0
    console.log(`[sweep] starting ${total} runs (${transcripts.length} transcripts × ${runsPer})`)

    for (const tid of transcripts) {
      console.log(`[sweep] ── ${tid} ──`)
      for (let i = 1; i <= runsPer; i++) {
        const t0 = performance.now()
        try {
          const r = await singleRun(tid)
          const dt = ((performance.now() - t0) / 1000).toFixed(1)
          results.runs.push({ transcript: tid, run: i, ...r, seconds: parseFloat(dt) })
          done++
          console.log(`[sweep] ${tid} ${i}/${runsPer}  bfly=${r.bfly}  lastN=${r.lastn}  ${dt}s  (${done}/${total} overall)`)
        } catch (e) {
          console.error(`[sweep] ${tid} ${i}/${runsPer} ERROR: ${e.message}`)
          results.runs.push({ transcript: tid, run: i, error: e.message })
        }
      }
    }

    results.local_tally = JSON.parse(localStorage.getItem('butterfly-mode-stats-v1') || '[]')
    results.finished_at = new Date().toISOString()

    // Trigger a download so the user has a file they can paste back.
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `butterfly-sweep-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
    document.body.appendChild(a); a.click(); a.remove()
    URL.revokeObjectURL(url)
    console.log(`[sweep] DONE — ${results.runs.length} runs captured. Download triggered.`)
    return results
  }

  console.log('[butterfly-sweep] loaded. Call:  butterflySweep({ runsPer: 20 })')
})();

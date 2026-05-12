import { defineConfig } from '@playwright/test'

// Dedicated config for the Butterfly v2.5 scaling sweep
// (PREDICTIONS.md P-20260512-05).
//
// ── Environment notes ──────────────────────────────────────────────
// WebGPU is gated on a SECURE CONTEXT — `navigator.gpu` is undefined
// on `about:blank`, `data:` URLs, and other non-secure origins. It IS
// exposed on `http://localhost` and `https://`. So the probe and the
// sweep both navigate to /app/ (served by vite on 127.0.0.1:4000)
// before touching navigator.gpu. See tests/webgpu-probe.spec.mjs.
//
// Playwright's default chrome-headless-shell uses SwiftShader (software
// GL) and does not expose WebGPU even on a secure context. So this
// config uses the `chrome` channel (your system Chrome install),
// headed mode, plus --enable-unsafe-webgpu. A small Chrome window
// appears during the sweep.
//
// ── Known limitation (2026-05-12, Chrome 148 + macOS 25.4) ─────────
// Even with everything below — secure context, system Chrome, headed
// mode, --enable-unsafe-webgpu, --disable-background-timer-throttling,
// --disable-renderer-backgrounding — Phi-3 inference HANGS at the
// first tagger call. The page boots correctly (pipelines compile,
// buffers allocate, panel opens, transcript loads), but engine.generate()
// never returns. Verified across 4 increasingly-aggressive configs.
//
// The same neuropulse build runs fine in the user's daily Chrome
// session at neuropulse.live. Cause unknown — possibly a Metal-context
// difference between a fresh tempdir profile and a real user profile.
//
// Until that's resolved, the sweep runs via `tools/console-sweep.js`
// inside a normal Chrome tab. The Playwright config is kept here so
// the failure mode is regression-testable and the harness is ready
// the moment Chrome/Playwright behavior changes.
//
// ── How to run ─────────────────────────────────────────────────────
//
//   # one-shot full sweep (4 transcripts × 20 runs ≈ 60 min)
//   npm run sweep:butterfly
//
//   # smaller smoke (1 run per transcript ≈ 3-5 min)
//   RUNS_PER=1 npm run sweep:butterfly
//
//   # specific transcript
//   TRANSCRIPTS=jwt-clock-race npm run sweep:butterfly

export default defineConfig({
  testDir: './tests',
  testMatch: /butterfly-sweep|webgpu-probe/,
  timeout: 3_600_000,            // sweep itself can take 60+ min
  fullyParallel: false,
  retries: 0,                    // every run scored honestly; no auto-retry
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://127.0.0.1:4000',
    trace: 'off',
    headless: false,             // ⬅ MUST be false; see env note above
    channel: 'chrome',
    launchOptions: {
      args: [
        '--enable-unsafe-webgpu',
        '--no-first-run',
        '--no-default-browser-check',
        // Defeat Chrome's background-tab throttling — without these,
        // an unfocused window stalls Phi-3 inference indefinitely
        // (verified: tagging hangs at "Gen 1/3 tagging 8 messages"
        // for 7+ minutes when the window is offscreen).
        '--disable-background-timer-throttling',
        '--disable-backgrounding-occluded-windows',
        '--disable-renderer-backgrounding',
        '--disable-features=CalculateNativeWinOcclusion',
      ],
    },
  },
})

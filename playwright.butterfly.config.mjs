import { defineConfig } from '@playwright/test'

// Dedicated config for the Butterfly v2.5 scaling sweep
// (PREDICTIONS.md P-20260512-05).
//
// ── Critical environment note ─────────────────────────────────────
// Chrome 148 on macOS — when launched programmatically (Playwright,
// raw CDP, or otherwise) with a synthetic --user-data-dir — does NOT
// expose `navigator.gpu`, regardless of flag combinations:
//   --enable-unsafe-webgpu        (no effect)
//   --headless=new                (no effect)
//   --enable-features=WebGPU      (no effect)
//   --ignore-gpu-blocklist        (no effect)
//   --use-angle=metal             (no effect)
//   ignoreDefaultArgs:true        (no effect)
//
// The same Chrome run from a regular user session (Cmd-Tab open) DOES
// expose WebGPU and Phi-3 inference works fine — confirmed by the
// neuropulse demo URL itself.
//
// So this config is intended for: run from a TTY, expects a real
// display, expects the user to have closed any chrome://settings
// privacy-blocking-WebGPU policies. The Chrome window will appear
// briefly during the sweep.
//
// ── How to run ───────────────────────────────────────────────────
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
      ],
    },
  },
})

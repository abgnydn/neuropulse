// Diagnostic: does a Chrome launched on :9222 expose navigator.gpu?
// Usage:
//   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
//     --user-data-dir=/tmp/cdp-chrome --remote-debugging-port=9222 \
//     --no-first-run --enable-unsafe-webgpu http://localhost:4000 &
//   node tools/cdp-probe.mjs
//
// NOTE: WebGPU is gated on a SECURE CONTEXT. `navigator.gpu` is
// undefined on `about:blank` and `data:` URLs, even when the GPU
// adapter is fully available. Always probe against `http://localhost`
// or `https://`. The original draft of this file probed about:blank
// and got false-negative `hasNavGpu: false` results — corrected.
import { chromium } from '@playwright/test'
const browser = await chromium.connectOverCDP('http://127.0.0.1:9222')
const ctx = browser.contexts()[0] || await browser.newContext()
const page = ctx.pages()[0] || await ctx.newPage()
await page.goto('about:blank')
const info = await page.evaluate(async () => {
  const out = { hasNavGpu: !!navigator.gpu, ua: navigator.userAgent }
  if (navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter()
      out.adapter = !!a
      if (a) {
        out.info = {
          vendor: a.info?.vendor, architecture: a.info?.architecture,
          device: a.info?.device, description: a.info?.description,
        }
      }
    } catch (e) { out.err = String(e) }
  }
  return out
})
console.log(JSON.stringify(info, null, 2))
await browser.close()

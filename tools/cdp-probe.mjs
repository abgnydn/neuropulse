// Diagnostic: does a Chrome launched on :9222 expose navigator.gpu?
// Usage:
//   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
//     --user-data-dir=/tmp/cdp-chrome --remote-debugging-port=9222 \
//     --no-first-run --enable-unsafe-webgpu about:blank &
//   node tools/cdp-probe.mjs
//
// Empirical finding (2026-05-12): hasNavGpu is false in every config
// attempted — headless=new, fully headed, ignoreDefaultArgs:true. The
// same Chrome under the user's regular session does expose WebGPU.
// See E37 in the research vault for the full env note.
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

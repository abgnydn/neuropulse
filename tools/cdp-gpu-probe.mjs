import { chromium } from '@playwright/test'
const browser = await chromium.connectOverCDP('http://127.0.0.1:9222')
const ctx = browser.contexts()[0] || await browser.newContext()
const page = ctx.pages()[0] || await ctx.newPage()
await page.goto('chrome://gpu', { waitUntil: 'domcontentloaded' })
const summary = await page.evaluate(() => {
  // Pull the feature status table — looking for "WebGPU" and "Vulkan" rows
  const text = document.body.innerText
  const lines = text.split('\n')
  const out = []
  for (const l of lines) {
    if (/WebGPU|Vulkan|GPU compositing|Hardware|Metal/i.test(l)) out.push(l.trim())
  }
  return out.slice(0, 30).join('\n')
})
console.log('--- chrome://gpu summary ---')
console.log(summary)
console.log()
await page.goto('about:blank')
const probe = await page.evaluate(async () => {
  const o = { hasNavGpu: !!navigator.gpu }
  if (navigator.gpu) {
    try { o.adapter = !!(await navigator.gpu.requestAdapter()) }
    catch (e) { o.err = String(e) }
  }
  return o
})
console.log('--- nav.gpu ---')
console.log(JSON.stringify(probe, null, 2))
await browser.close()

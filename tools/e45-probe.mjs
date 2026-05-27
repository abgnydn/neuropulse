#!/usr/bin/env node
// Quick diagnostic: launch the same playwright chromium and probe whether
// WebGPU is available, what console output the engine init produces, and
// what state the page is in after 20s.

import { chromium } from '@playwright/test'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = join(__dirname, '..')
const USER_DATA_DIR = join(REPO_ROOT, '.playwright-profile-e45')

const ctx = await chromium.launchPersistentContext(USER_DATA_DIR, {
  headless: false,
  viewport: { width: 1280, height: 800 },
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU'],
})
const page = ctx.pages()[0] || (await ctx.newPage())

page.on('pageerror', (e) => console.error('[pageerror]', e.message))
page.on('console', (m) => {
  if (m.type() !== 'debug') console.log(`[browser ${m.type()}] ${m.text().slice(0, 240)}`)
})
page.on('requestfailed', (r) => console.log(`[reqfailed] ${r.url().slice(0, 100)} — ${r.failure()?.errorText}`))
page.on('response', (r) => {
  if (r.status() >= 400) console.log(`[resp ${r.status()}] ${r.url().slice(0, 120)}`)
})

await page.goto('http://localhost:4000/app/?attn=fixedpoint&max_iter=1&noauto', { waitUntil: 'domcontentloaded' })

console.log('--- WebGPU probe at t=2s ---')
await page.waitForTimeout(2000)
const probe = await page.evaluate(async () => {
  const o = { hasNavGpu: !!navigator.gpu, ua: navigator.userAgent.slice(0, 100) }
  if (navigator.gpu) {
    try { const ad = await navigator.gpu.requestAdapter(); o.adapter = !!ad; o.adapterName = ad?.info?.architecture ?? null }
    catch (e) { o.adapterErr = String(e) }
  }
  return o
})
console.log(JSON.stringify(probe, null, 2))

console.log('--- waiting 30s for engine init, watching console ---')
await page.waitForTimeout(30_000)

const status = await page.evaluate(() => ({
  hasE45: typeof globalThis.__e45 !== 'undefined',
  loadingText: document.querySelector('[class*="loading"], [id*="loading"]')?.textContent?.replace(/\s+/g, ' ').trim().slice(0, 200) ?? null,
  errorText: document.querySelector('[class*="error"]')?.textContent?.slice(0, 200) ?? null,
}))
console.log('--- status at t=32s ---')
console.log(JSON.stringify(status, null, 2))

await page.screenshot({ path: join(REPO_ROOT, 'tests', 'results', 'e45-probe.png') })
console.log('screenshot: tests/results/e45-probe.png')

await ctx.close()

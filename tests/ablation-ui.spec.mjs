import { test, expect } from '@playwright/test'

// UI-only smoke. Engine is NOT initialized (?bypass=1 skips engine init):
// we only verify the visualizer's ablation picking + panel wiring. One
// consolidated test avoids the per-test page.goto flake seen when
// Chromium's swiftshader + Three.js init is repeated in a loop.

test('ablation panel: toggle, multi-select, clear, invalid', async ({ page }) => {
  page.on('pageerror', e => console.log('[pageerror]', e.message))

  await page.goto('/app/?noauto&bypass=1', { waitUntil: 'domcontentloaded' })
  await page.waitForFunction(() => typeof window.__testToggleAblation === 'function', { timeout: 15_000 })

  const panel = page.locator('.ablate-panel')
  const status = page.locator('#ablateStatus')

  // 1. Starts hidden.
  await expect(panel).toBeHidden()

  // 2. Toggle one head → visible with count "1 head ablated across 1 layer".
  expect(await page.evaluate(() => window.__testToggleAblation(12, 5))).toBe(true)
  await expect(panel).toBeVisible()
  await expect(status).toHaveText('1 head ablated across 1 layer')

  // 3. Toggle the same head again → hidden.
  expect(await page.evaluate(() => window.__testToggleAblation(12, 5))).toBe(true)
  await expect(panel).toBeHidden()

  // 4. Multiple heads across layers → correct plurals + layer count.
  await page.evaluate(() => {
    window.__testToggleAblation(0, 0)
    window.__testToggleAblation(0, 1)
    window.__testToggleAblation(15, 3)
    window.__testToggleAblation(31, 7)
  })
  await expect(status).toHaveText('4 heads ablated across 3 layers')

  // 5. Run-ablated button present + enabled.
  const runBtn = page.locator('#ablateRunBtn')
  await expect(runBtn).toBeVisible()
  await expect(runBtn).toBeEnabled()
  await expect(runBtn).toHaveText('Run ablated')

  // 6. Clear resets.
  await page.evaluate(() => document.getElementById('ablateClearBtn')?.click())
  await expect(panel).toBeHidden()

  // 7. Invalid layer/head returns false, panel stays hidden.
  const bogus = await page.evaluate(() => window.__testToggleAblation(999, 0))
  expect(bogus).toBe(false)
  await expect(panel).toBeHidden()
})

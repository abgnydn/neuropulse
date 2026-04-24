import { test, expect } from '@playwright/test'

// UI-only smoke. Engine is NOT initialized (?bypass=1 skips engine init):
// we only verify the visualizer's ablation picking + panel wiring. One
// consolidated test avoids the per-test page.goto flake seen when
// Chromium's swiftshader + Three.js init is repeated in a loop.

test('ablation panel: toggle, multi-select, clear, invalid, sweep UI present', async ({ page }) => {
  page.on('pageerror', e => console.log('[pageerror]', e.message))

  await page.goto('/app/?noauto&bypass=1', { waitUntil: 'domcontentloaded' })
  await page.waitForFunction(() => typeof window.__testToggleAblation === 'function', { timeout: 15_000 })

  const panel = page.locator('.ablate-panel')
  const status = page.locator('#ablateStatus')
  const runBtn = page.locator('#ablateRunBtn')
  const sweepBtn = page.locator('#ablateSweepBtn')

  // 1. Panel always visible now (it hosts sweep controls too).
  await expect(panel).toBeVisible()
  await expect(status).toContainText('No heads ablated')
  // Run-ablated starts disabled when nothing is selected.
  await expect(runBtn).toBeDisabled()
  // Sweep controls are present and enabled.
  await expect(sweepBtn).toBeEnabled()
  await expect(page.locator('#ablateSweepLayer')).toHaveValue('31')

  // 2. Toggle one head → status updates, Run button enables.
  expect(await page.evaluate(() => window.__testToggleAblation(12, 5))).toBe(true)
  await expect(status).toHaveText('1 head ablated across 1 layer')
  await expect(runBtn).toBeEnabled()

  // 3. Toggle the same head again → status reverts, Run disables.
  expect(await page.evaluate(() => window.__testToggleAblation(12, 5))).toBe(true)
  await expect(status).toContainText('No heads ablated')
  await expect(runBtn).toBeDisabled()

  // 4. Multiple heads across layers → plurals + layer count.
  await page.evaluate(() => {
    window.__testToggleAblation(0, 0)
    window.__testToggleAblation(0, 1)
    window.__testToggleAblation(15, 3)
    window.__testToggleAblation(31, 7)
  })
  await expect(status).toHaveText('4 heads ablated across 3 layers')

  // 5. Run-ablated button label + enabled state.
  await expect(runBtn).toHaveText('Run ablated')
  await expect(runBtn).toBeEnabled()

  // 6. Clear resets selection back to empty status.
  await page.evaluate(() => document.getElementById('ablateClearBtn')?.click())
  await expect(status).toContainText('No heads ablated')
  await expect(runBtn).toBeDisabled()

  // 7. Invalid layer/head returns false; status stays empty.
  const bogus = await page.evaluate(() => window.__testToggleAblation(999, 0))
  expect(bogus).toBe(false)
  await expect(status).toContainText('No heads ablated')
})

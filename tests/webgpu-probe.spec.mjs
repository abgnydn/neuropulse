import { test } from '@playwright/test'

// Fast diagnostic: does the launched browser actually expose a WebGPU
// adapter? If not, the butterfly sweep cannot run here regardless of
// how long we wait.
test('webgpu adapter probe', async ({ page }, _testInfo) => {
  test.setTimeout(30_000)
  // WebGPU is gated on a secure context — navigator.gpu is NOT exposed
  // on about:blank. Probe against http://localhost (a secure context).
  await page.goto('/app/', { waitUntil: 'domcontentloaded' })
  const info = await page.evaluate(async () => {
    const out = { hasNavGpu: !!navigator.gpu, ua: navigator.userAgent }
    if (!navigator.gpu) return out
    try {
      const adapter = await navigator.gpu.requestAdapter()
      out.adapter = !!adapter
      if (adapter) {
        out.adapterInfo = {
          vendor: adapter.info?.vendor || null,
          architecture: adapter.info?.architecture || null,
          device: adapter.info?.device || null,
          description: adapter.info?.description || null,
        }
        const features = []
        for (const f of adapter.features) features.push(f)
        out.features = features
      }
    } catch (e) {
      out.adapterError = String(e)
    }
    return out
  })
  console.log('[probe]', JSON.stringify(info, null, 2))
})

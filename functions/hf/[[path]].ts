/**
 * CF Pages Function — proxies HuggingFace with aggressive edge caching.
 *
 * Every /hf/<path> request maps to https://huggingface.co/<path> and the
 * response is stored in Cloudflare's runtime cache for 1 year, marked
 * `immutable`. First user in each CF region pays the full HF latency;
 * subsequent users in that region read from the edge (300+ POPs worldwide).
 *
 * Typical effect for a 2 GB model: 2-5× geography-dependent speedup for
 * users outside US/EU (HuggingFace's CDN has far fewer POPs).
 *
 * Weight loader uses this as a tier (between Cache-API and direct HF), so
 * if this function errors, downloads still succeed via the HF fallback.
 *
 * CORS: same-origin for normal use, but we add * headers anyway so the file
 * can be fetched from a Worker or another subdomain during experiments.
 */

interface PagesContext {
  request: Request
  waitUntil: (p: Promise<unknown>) => void
}

export const onRequestGet = async (ctx: PagesContext): Promise<Response> => {
  const url = new URL(ctx.request.url)
  const pathAfter = url.pathname.replace(/^\/hf\//, '')
  if (!pathAfter) return new Response('Missing path', { status: 400 })

  const hfUrl = 'https://huggingface.co/' + pathAfter + url.search

  // CF runtime cache lookup — free, part of the Workers platform.
  const cache = (caches as unknown as { default: Cache }).default
  const cacheKey = new Request(hfUrl)
  let resp = await cache.match(cacheKey)

  if (resp) {
    // Cache hit — reconstruct a Response that allows * origin.
    const headers = new Headers(resp.headers)
    headers.set('Access-Control-Allow-Origin', '*')
    headers.set('X-Neuropulse-Cache', 'HIT')
    return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers })
  }

  // Miss — fetch from HF. `cf.cacheEverything` + long `cacheTtl` lets
  // Cloudflare hold even uncacheable-by-default responses.
  const hfResp = await fetch(hfUrl, {
    cf: { cacheEverything: true, cacheTtl: 31_536_000 },
  } as RequestInit)

  if (!hfResp.ok) {
    // Don't cache failures.
    const headers = new Headers(hfResp.headers)
    headers.set('Access-Control-Allow-Origin', '*')
    headers.set('X-Neuropulse-Cache', 'MISS-ERR')
    return new Response(hfResp.body, { status: hfResp.status, statusText: hfResp.statusText, headers })
  }

  // Clone headers to stamp our own Cache-Control. Weight files are content-
  // addressed at HF (filename + branch), so `immutable` is safe.
  const headers = new Headers(hfResp.headers)
  headers.set('Cache-Control', 'public, max-age=31536000, immutable')
  headers.set('Access-Control-Allow-Origin', '*')
  headers.set('X-Neuropulse-Cache', 'MISS')
  headers.delete('Set-Cookie')

  // Tee the body: one stream flows to the client, one to the cache.
  const toCache = new Response(hfResp.body, {
    status: hfResp.status,
    statusText: hfResp.statusText,
    headers,
  })
  ctx.waitUntil(cache.put(cacheKey, toCache.clone()))
  return toCache
}

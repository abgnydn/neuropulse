/**
 * WEIGHT LOADER — Phi-3 MLC weights on WebGPU.
 *
 * Ported from zero-tvm's `fcefff3` rewrite (Apr 2026), adapted to keep
 * Neuropulse's rich per-byte progress UI.
 *
 * Flow:
 *   1. Open OPFS (Origin Private File System) — graceful no-op if unavailable
 *   2. Fetch ndarray-cache.json manifest via tiered cache
 *   3. Group records by shard (parameters share one shard file)
 *   4. **In parallel**: for each shard, try tiered cache → stream-download
 *      with byte-level progress → write to OPFS + Cache API → upload to
 *      GPU. Shards upload to GPU as they arrive, not after all download.
 *   5. Assemble `LoadedWeights` by resolving named parameters from the
 *      populated `GPUBuffer` map
 *
 * Tiered cache (fastest → slowest):
 *   0. DEV-only Vite mirror at `/local-weights/*`
 *   1. OPFS — persistent per-origin, ~2× faster than Cache API for 200+ MB blobs
 *   2. Browser Cache API — scans *any* cache name, so WebLLM-prepopulated
 *      caches count too, and old `neural-pulse-phi3-weights` caches migrate
 *      transparently
 *   3. HuggingFace network — last resort, stream-fetched with retry
 *
 * Second visit = instant cold-start from OPFS (no network).
 */

import { WEIGHT_REVISION, SHARD_SHA256 } from './weight-manifest'

// ============================================================
// Model URL + cache name
// ============================================================

// Pinned to an immutable HF git revision (not `main`) so the bytes always
// match the SHA-256s in weight-manifest.ts / parity.json. This is what makes
// the download content-addressed and verifiable.
export const PHI3_MODEL_BASE =
  `https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC/resolve/${WEIGHT_REVISION}/`

export const CACHE_NAME = 'neuropulse-phi3-weights'
// Older sessions used 'neural-pulse-phi3-weights'. The anyCacheMatch tier
// scans all cache names, so an old cache is picked up and promoted into
// `CACHE_NAME` + OPFS transparently on first load — no re-download.

const OPFS_DIR = 'neuropulse-weights'

// Dev-only Vite mirror — see vite.config.ts; `~/.cache/huggingface/hub` proxied
// to `/local-weights/*` so cold-start e2e testing doesn't pay 2 GB.
const LOCAL_MIRROR_BASE = (import.meta as { env?: { DEV?: boolean } }).env?.DEV
  ? '/local-weights/'
  : null

// Prod edge-cached proxy — see functions/hf/[[path]].ts (CF Pages Function).
// First user in each CF region pays full HF latency; subsequent users in that
// region read from the edge (300+ POPs). Gracefully falls through to direct
// HF if the function 4xx/5xx's (e.g., not deployed, cold start error).
const CF_PROXY_BASE =
  typeof window !== 'undefined' && !LOCAL_MIRROR_BASE
    ? `/hf/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC/resolve/${WEIGHT_REVISION}/`
    : null

// ============================================================
// ndarray-cache.json types
// ============================================================

interface FlatRecord {
  name: string
  shape: number[]
  dtype: string
  format: string
  dataPath: string
  byteOffset: number
  nbytes: number
}

interface ShardGroup {
  dataPath: string
  format: string
  byteOffset: number
  nbytes: number
  records: FlatRecord[]
}

interface NDArrayCache {
  records: (FlatRecord | ShardGroup)[]
}

// ============================================================
// Progress callback — preserves the rich interface main.ts expects
// ============================================================

export interface LoadProgress {
  phase: 'manifest' | 'downloading' | 'uploading' | 'done'
  message: string
  bytesLoaded: number
  bytesTotal: number
  /** 0-100 */
  percent: number
  /** Current shard being downloaded (if any) */
  currentShard?: string
  /** Whether current shard was a cache hit */
  cacheHit?: boolean
}

// ============================================================
// OPFS — per-origin persistent storage. Faster than Cache API for
// multi-hundred-MB blobs. Graceful no-op on Safari (no createWritable).
// ============================================================

type OPFSDir = FileSystemDirectoryHandle | null

async function openOPFS(): Promise<OPFSDir> {
  try {
    if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) return null
    const root = await navigator.storage.getDirectory()
    return await root.getDirectoryHandle(OPFS_DIR, { create: true })
  } catch {
    return null
  }
}

function opfsKey(dataPath: string): string {
  // OPFS filenames can't contain '/'. Flatten to a safe ASCII key.
  return dataPath.replace(/[^A-Za-z0-9._-]/g, '_')
}

async function opfsRead(dir: OPFSDir, dataPath: string): Promise<ArrayBuffer | null> {
  if (!dir) return null
  try {
    const fh = await dir.getFileHandle(opfsKey(dataPath))
    const file = await fh.getFile()
    return await file.arrayBuffer()
  } catch {
    return null
  }
}

async function opfsWrite(dir: OPFSDir, dataPath: string, data: ArrayBuffer): Promise<void> {
  if (!dir) return
  try {
    const fh = await dir.getFileHandle(opfsKey(dataPath), { create: true })
    // createWritable is widely supported; Safari falls through to the catch.
    const writable = await (fh as unknown as { createWritable: () => Promise<{ write(d: ArrayBuffer): Promise<void>; close(): Promise<void> }> }).createWritable()
    await writable.write(data)
    await writable.close()
  } catch {
    // best-effort — failures just mean next visit pays network again
  }
}

// ============================================================
// Storage inspection + cleanup — surfaced in the UI so users can see
// what's cached and free disk space without dev-tools.
// ============================================================

/** A weight shard URL across any storage tier we know about. */
function isWeightUrl(url: string): boolean {
  return (
    url.includes('Phi-3-mini-4k-instruct-q4f16_1-MLC') ||
    url.includes('/local-weights/') ||
    url.includes('/hf/mlc-ai/')
  )
}

export interface StoredWeightStats {
  /** Bytes stored in any Cache API bucket whose entries match a weight URL. */
  cacheBytes: number
  /** Bytes stored in our OPFS directory. */
  opfsBytes: number
  /** Sum of cache + OPFS. */
  totalBytes: number
  /** Number of shard responses found across all caches. */
  shardCount: number
  /** Number of OPFS files found. */
  opfsFileCount: number
}

export async function getStoredWeightStats(): Promise<StoredWeightStats> {
  let cacheBytes = 0
  let shardCount = 0
  let opfsBytes = 0
  let opfsFileCount = 0

  try {
    if (typeof caches !== 'undefined') {
      const names = await caches.keys()
      for (const name of names) {
        const store = await caches.open(name)
        const requests = await store.keys()
        for (const req of requests) {
          if (!isWeightUrl(req.url)) continue
          const resp = await store.match(req)
          if (!resp) continue
          const cl = resp.headers.get('content-length')
          if (cl) {
            cacheBytes += parseInt(cl, 10) || 0
          } else {
            const blob = await resp.blob()
            cacheBytes += blob.size
          }
          shardCount++
        }
      }
    }
  } catch {
    /* no cache api */
  }

  try {
    if (typeof navigator !== 'undefined' && navigator.storage?.getDirectory) {
      const root = await navigator.storage.getDirectory()
      try {
        const dir = await root.getDirectoryHandle(OPFS_DIR)
        // entries() exists at runtime but is missing from current TS lib types.
        const entries = (dir as unknown as {
          entries(): AsyncIterable<[string, FileSystemHandle]>
        }).entries()
        for await (const [, handle] of entries) {
          if (handle.kind === 'file') {
            const file = await (handle as FileSystemFileHandle).getFile()
            opfsBytes += file.size
            opfsFileCount++
          }
        }
      } catch {
        /* directory doesn't exist yet */
      }
    }
  } catch {
    /* no OPFS */
  }

  return {
    cacheBytes,
    opfsBytes,
    totalBytes: cacheBytes + opfsBytes,
    shardCount,
    opfsFileCount,
  }
}

/** Wipe every cached weight blob from Cache API caches + OPFS. The next
 *  visit will re-download from HuggingFace. */
export async function clearStoredWeights(): Promise<void> {
  try {
    if (typeof caches !== 'undefined') {
      const names = await caches.keys()
      for (const name of names) {
        const store = await caches.open(name)
        const requests = await store.keys()
        for (const req of requests) {
          if (isWeightUrl(req.url)) await store.delete(req)
        }
      }
    }
  } catch {
    /* no cache api */
  }
  try {
    if (typeof navigator !== 'undefined' && navigator.storage?.getDirectory) {
      const root = await navigator.storage.getDirectory()
      try {
        await (root as unknown as {
          removeEntry(name: string, options?: { recursive?: boolean }): Promise<void>
        }).removeEntry(OPFS_DIR, { recursive: true })
      } catch {
        /* not present — nothing to clear */
      }
    }
  } catch {
    /* no OPFS */
  }
}

// ============================================================
// Tiered shard fetch
// ============================================================

/** Scan *every* Cache API bucket for a URL. Picks up WebLLM and old
 *  `neural-pulse-phi3-weights` caches transparently. */
async function anyCacheMatch(url: string): Promise<Response | null> {
  try {
    const names = await caches.keys()
    for (const name of names) {
      const store = await caches.open(name)
      const resp = await store.match(url)
      if (resp) return resp
    }
  } catch {
    /* no Cache API */
  }
  return null
}

async function putInOurCache(url: string, buf: ArrayBuffer): Promise<void> {
  try {
    const store = await caches.open(CACHE_NAME)
    await store.put(
      url,
      new Response(buf.slice(0), {
        headers: {
          'Content-Type': 'application/octet-stream',
          'Content-Length': String(buf.byteLength),
        },
      }),
    )
  } catch {
    /* storage quota or cookie mode — ok */
  }
}

interface FetchResult {
  buf: ArrayBuffer
  /** true if served from any cache tier (OPFS / Cache API / local mirror) */
  fromCache: boolean
}

/** Network fetch with byte-level streaming progress + 5-attempt retry. */
async function streamFetch(
  url: string,
  onBytes: (loaded: number) => void,
): Promise<ArrayBuffer> {
  let resp: Response | null = null
  let lastErr: unknown
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      resp = await fetch(url)
      if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`)
      break
    } catch (e) {
      lastErr = e
      resp = null
      await new Promise((r) => setTimeout(r, 500 * (attempt + 1)))
    }
  }
  if (!resp) throw lastErr instanceof Error ? lastErr : new Error(`fetch failed: ${url}`)

  const reader = resp.body!.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.byteLength
    onBytes(loaded)
  }

  const buf = new Uint8Array(loaded)
  let offset = 0
  for (const chunk of chunks) {
    buf.set(chunk, offset)
    offset += chunk.byteLength
  }
  return buf.buffer as ArrayBuffer
}

/** SHA-256 of a buffer as lowercase hex. */
async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', buf)
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
}

/** Verify a freshly-downloaded shard against its pinned SHA-256. No-op for
 *  paths we have no hash for (e.g. the ndarray-cache manifest). Throws on
 *  mismatch so the caller can fall through to another tier or fail loudly —
 *  a tampered/corrupt shard is never written to OPFS or the browser cache. */
async function verifyShard(dataPath: string, buf: ArrayBuffer): Promise<void> {
  const expected = SHARD_SHA256[dataPath]
  if (!expected) return
  const actual = await sha256Hex(buf)
  if (actual !== expected) {
    throw new Error(
      `Integrity check failed for ${dataPath}: expected ${expected.slice(0, 12)}…, got ${actual.slice(0, 12)}…`,
    )
  }
}

async function fetchShard(
  url: string,
  dataPath: string,
  opfs: OPFSDir,
  onBytes: (loaded: number) => void,
): Promise<FetchResult> {
  // Tier 0: dev-only local Vite mirror
  if (LOCAL_MIRROR_BASE) {
    try {
      const resp = await fetch(LOCAL_MIRROR_BASE + dataPath)
      if (resp.ok) {
        const buf = await resp.arrayBuffer()
        onBytes(buf.byteLength)
        void opfsWrite(opfs, dataPath, buf)
        return { buf, fromCache: true }
      }
    } catch {
      /* mirror not primed — fall through */
    }
  }

  // Tier 1: OPFS
  const fromOPFS = await opfsRead(opfs, dataPath)
  if (fromOPFS) {
    onBytes(fromOPFS.byteLength)
    return { buf: fromOPFS, fromCache: true }
  }

  // Tier 2: any Cache API bucket (migration + WebLLM reuse)
  const cached = await anyCacheMatch(url)
  if (cached) {
    const buf = await cached.arrayBuffer()
    onBytes(buf.byteLength)
    // Promote into OPFS + our named cache for faster future loads
    void opfsWrite(opfs, dataPath, buf)
    void putInOurCache(url, buf)
    return { buf, fromCache: true }
  }

  // Tier 3: CF Pages Function edge-cache proxy (prod only). Stream-fetched
  // so the progress UI still updates per-byte. If the function is down or
  // not deployed yet, we fall through to direct HF.
  if (CF_PROXY_BASE) {
    try {
      const buf = await streamFetch(CF_PROXY_BASE + dataPath, onBytes)
      await verifyShard(dataPath, buf) // reject tampered edge bytes before caching
      void opfsWrite(opfs, dataPath, buf)
      void putInOurCache(url, buf)
      return { buf, fromCache: false }
    } catch {
      /* proxy down or integrity mismatch — fall through to direct HF */
    }
  }

  // Tier 4: direct HuggingFace network (streaming + retry)
  const buf = await streamFetch(url, onBytes)
  await verifyShard(dataPath, buf) // hard-fail if even HF served unexpected bytes
  void opfsWrite(opfs, dataPath, buf)
  void putInOurCache(url, buf)
  return { buf, fromCache: false }
}

// ============================================================
// Record helpers
// ============================================================

function flattenRecords(cache: NDArrayCache): FlatRecord[] {
  const out: FlatRecord[] = []
  for (const rec of cache.records) {
    if ('records' in rec && Array.isArray((rec as ShardGroup).records)) {
      for (const r of (rec as ShardGroup).records) {
        out.push({ ...r, dataPath: r.dataPath ?? (rec as ShardGroup).dataPath })
      }
    } else {
      out.push(rec as FlatRecord)
    }
  }
  return out
}

const USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

function uploadRecord(device: GPUDevice, shard: ArrayBuffer, rec: FlatRecord): GPUBuffer {
  const gpuBuf = device.createBuffer({
    size: Math.max(rec.nbytes, 4),
    usage: USAGE,
    label: rec.name,
  })
  // Uint8Array view avoids an ArrayBuffer.slice() copy before writeBuffer.
  device.queue.writeBuffer(gpuBuf, 0, new Uint8Array(shard, rec.byteOffset, rec.nbytes))
  return gpuBuf
}

function find(index: Map<string, GPUBuffer>, ...candidates: string[]): GPUBuffer {
  for (const c of candidates) {
    const b = index.get(c)
    if (b) return b
  }
  throw new Error(
    `Weight not found. Tried: ${candidates.join(', ')}\n` +
      `Available sample: ${[...index.keys()].slice(0, 10).join(', ')}…`,
  )
}

// ============================================================
// Main loader — parallel shards, streaming GPU upload
// ============================================================

export async function loadWeights(
  device: GPUDevice,
  onProgress?: (p: LoadProgress) => void,
): Promise<LoadedWeights> {
  const baseUrl = PHI3_MODEL_BASE
  const opfs = await openOPFS()

  const report = (patch: Partial<LoadProgress> & { message: string }): void => {
    onProgress?.({
      phase: 'downloading',
      bytesLoaded: 0,
      bytesTotal: 0,
      percent: 0,
      ...patch,
    })
  }

  // ── 1. Manifest ──
  report({ phase: 'manifest', message: 'Reading ndarray-cache.json…' })
  const { buf: manifestBuf } = await fetchShard(
    baseUrl + 'ndarray-cache.json',
    'ndarray-cache.json',
    opfs,
    () => {},
  )
  const manifest: NDArrayCache = JSON.parse(new TextDecoder().decode(manifestBuf))
  const allRecords = flattenRecords(manifest)

  // ── 2. Group records by shard ──
  const byShard = new Map<string, FlatRecord[]>()
  for (const r of allRecords) {
    const existing = byShard.get(r.dataPath)
    if (existing) existing.push(r)
    else byShard.set(r.dataPath, [r])
  }
  const shardList = [...byShard.keys()]

  // ── 3. Estimate total bytes ──
  const shardSizes = new Map<string, number>()
  for (const r of allRecords) {
    const existing = shardSizes.get(r.dataPath) ?? 0
    shardSizes.set(r.dataPath, Math.max(existing, r.byteOffset + r.nbytes))
  }
  let totalBytes = 0
  for (const s of shardSizes.values()) totalBytes += s

  report({
    phase: 'downloading',
    message: `${allRecords.length} params across ${shardList.length} shards (${(totalBytes / 1e9).toFixed(2)} GB)`,
    bytesTotal: totalBytes,
  })

  // ── 4. Fetch all shards IN PARALLEL, GPU-upload as each arrives ──
  const gpuBuffers = new Map<string, GPUBuffer>()
  const perShardBytes = new Map<string, number>()
  let totalBytesLoaded = 0
  let shardsCompleted = 0
  const shardHitFromCache = new Map<string, boolean>()

  await Promise.all(
    shardList.map(async (dataPath) => {
      const url = baseUrl + dataPath
      const { buf, fromCache } = await fetchShard(url, dataPath, opfs, (loaded) => {
        const delta = loaded - (perShardBytes.get(dataPath) ?? 0)
        perShardBytes.set(dataPath, loaded)
        totalBytesLoaded += delta
        const pct = totalBytes > 0 ? Math.min(100, (totalBytesLoaded / totalBytes) * 100) : 0
        onProgress?.({
          phase: 'downloading',
          message: fromCacheHint(shardHitFromCache, dataPath)
            ? `Loading from cache · ${(totalBytesLoaded / 1e6).toFixed(0)} / ${(totalBytes / 1e6).toFixed(0)} MB`
            : `Downloading · ${(totalBytesLoaded / 1e6).toFixed(0)} / ${(totalBytes / 1e6).toFixed(0)} MB`,
          bytesLoaded: totalBytesLoaded,
          bytesTotal: totalBytes,
          percent: pct,
          currentShard: dataPath,
          cacheHit: fromCacheHint(shardHitFromCache, dataPath),
        })
      })
      shardHitFromCache.set(dataPath, fromCache)

      // Upload every parameter in this shard to its own GPUBuffer as soon as
      // the shard arrives — overlaps with ongoing parallel downloads.
      const records = byShard.get(dataPath)!
      for (const rec of records) {
        gpuBuffers.set(rec.name, uploadRecord(device, buf, rec))
      }

      shardsCompleted++
      onProgress?.({
        phase: 'downloading',
        message: `[${shardsCompleted}/${shardList.length}] ${dataPath} uploaded to GPU`,
        bytesLoaded: totalBytesLoaded,
        bytesTotal: totalBytes,
        percent: totalBytes > 0 ? Math.min(100, (totalBytesLoaded / totalBytes) * 100) : 0,
        currentShard: dataPath,
        cacheHit: fromCache,
      })
    }),
  )

  // ── 5. Resolve named parameters ──
  const embdWeights = find(gpuBuffers, 'transformer.embd.q_weight', 'embed_tokens.q_weight', 'model.embed_tokens.q_weight')
  const embdScales = find(gpuBuffers, 'transformer.embd.q_scale', 'embed_tokens.q_scale', 'model.embed_tokens.q_scale')
  const initNormGamma = find(gpuBuffers, 'transformer.h.0.ln.weight', 'model.layers.0.input_layernorm.weight')
  const lmHeadWeights = find(gpuBuffers, 'lm_head.q_weight', 'model.lm_head.q_weight')
  const lmHeadScales = find(gpuBuffers, 'lm_head.q_scale', 'model.lm_head.q_scale')
  const finalNormGamma = find(gpuBuffers, 'transformer.norm.weight', 'model.norm.weight', 'norm.weight')

  const LAYERS = 32
  const layers: LoadedWeights['layers'] = []
  for (let L = 0; L < LAYERS; L++) {
    const h = `transformer.h.${L}`
    const p = `model.layers.${L}`
    layers.push({
      qkvWeights: find(gpuBuffers, `${h}.mixer.qkv_proj.q_weight`, `${p}.self_attn.qkv_proj.q_weight`),
      qkvScales: find(gpuBuffers, `${h}.mixer.qkv_proj.q_scale`, `${p}.self_attn.qkv_proj.q_scale`),
      oProjWeights: find(gpuBuffers, `${h}.mixer.out_proj.q_weight`, `${p}.self_attn.o_proj.q_weight`),
      oProjScales: find(gpuBuffers, `${h}.mixer.out_proj.q_scale`, `${p}.self_attn.o_proj.q_scale`),
      normGamma1: find(gpuBuffers, `${h}.ln.weight`, `${p}.input_layernorm.weight`),
      normGamma2: find(gpuBuffers, `${h}.post_attention_layernorm.weight`, `${p}.post_attention_layernorm.weight`),
      ffnWeights: find(gpuBuffers, `${h}.mlp.gate_up_proj.q_weight`, `${p}.mlp.gate_up_proj.q_weight`),
      ffnScales: find(gpuBuffers, `${h}.mlp.gate_up_proj.q_scale`, `${p}.mlp.gate_up_proj.q_scale`),
      ffnDownWeights: find(gpuBuffers, `${h}.mlp.down_proj.q_weight`, `${p}.mlp.down_proj.q_weight`),
      ffnDownScales: find(gpuBuffers, `${h}.mlp.down_proj.q_scale`, `${p}.mlp.down_proj.q_scale`),
    })
  }

  onProgress?.({
    phase: 'done',
    message: `Ready · ${(totalBytes / 1e6).toFixed(0)} MB · ${opfs ? 'OPFS active' : 'Cache API fallback'}`,
    bytesLoaded: totalBytesLoaded,
    bytesTotal: totalBytes,
    percent: 100,
  })

  return {
    device,
    embdWeights,
    embdScales,
    lmHeadWeights,
    lmHeadScales,
    initNormGamma,
    finalNormGamma,
    layers,
  }
}

function fromCacheHint(m: Map<string, boolean>, k: string): boolean {
  return m.get(k) ?? false
}

// ============================================================
// Public type (identical shape to zero-tvm's — inference engine agnostic)
// ============================================================

export interface LoadedWeights {
  device: GPUDevice
  embdWeights: GPUBuffer
  embdScales: GPUBuffer
  lmHeadWeights: GPUBuffer
  lmHeadScales: GPUBuffer
  initNormGamma: GPUBuffer
  finalNormGamma: GPUBuffer
  layers: Array<{
    qkvWeights: GPUBuffer
    qkvScales: GPUBuffer
    oProjWeights: GPUBuffer
    oProjScales: GPUBuffer
    normGamma1: GPUBuffer
    normGamma2: GPUBuffer
    ffnWeights: GPUBuffer
    ffnScales: GPUBuffer
    ffnDownWeights: GPUBuffer
    ffnDownScales: GPUBuffer
  }>
}

/**
 * WEIGHT LOADER — Load Phi-3 MLC weights with Cache API + download progress.
 *
 * Flow:
 *   1. Fetch ndarray-cache.json (manifest of all weight shards)
 *   2. Calculate total download size
 *   3. For each shard: check Cache API → if miss, stream-download with
 *      byte-level progress → store in Cache API for instant reload
 *   4. Slice individual parameters from shards → GPUBuffers
 *
 * Second visit = instant load from cache (no network).
 */

// ============================================================
// Model URL
// ============================================================

export const PHI3_MODEL_BASE =
  'https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC/resolve/main/'

const CACHE_NAME = 'neural-pulse-phi3-weights'

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
// Progress callback
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
// Fetch with streaming progress + Cache API
// ============================================================

async function fetchWithCache(
  url: string,
  onBytes: (loaded: number) => void,
): Promise<ArrayBuffer> {
  // 1. Check our dedicated cache
  try {
    const store = await caches.open(CACHE_NAME)
    const cached = await store.match(url)
    if (cached) {
      const buf = await cached.arrayBuffer()
      onBytes(buf.byteLength) // report full size instantly
      return buf
    }
  } catch { /* Cache API unavailable */ }

  // 2. Also check any tvmjs/webllm caches from prior sessions
  try {
    const cacheNames = await caches.keys()
    for (const name of cacheNames) {
      if (name === CACHE_NAME) continue
      const store = await caches.open(name)
      const resp = await store.match(url)
      if (resp) {
        const buf = await resp.arrayBuffer()
        onBytes(buf.byteLength)
        // Copy into our cache for next time
        try {
          const myStore = await caches.open(CACHE_NAME)
          await myStore.put(url, new Response(buf.slice(0)))
        } catch { /* ok */ }
        return buf
      }
    }
  } catch { /* no Cache API */ }

  // 3. Stream-download with byte progress
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`)

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

  // Combine chunks
  const buf = new Uint8Array(loaded)
  let offset = 0
  for (const chunk of chunks) {
    buf.set(chunk, offset)
    offset += chunk.byteLength
  }
  const arrayBuf = buf.buffer as ArrayBuffer

  // 4. Store in Cache API for next visit
  try {
    const store = await caches.open(CACHE_NAME)
    await store.put(url, new Response(arrayBuf.slice(0), {
      headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(loaded) }
    }))
  } catch { /* ok if cache storage fails */ }

  return arrayBuf
}

// ============================================================
// Check if a URL is already cached (for pre-scan)
// ============================================================

async function isCached(url: string): Promise<boolean> {
  try {
    const cacheNames = await caches.keys()
    for (const name of cacheNames) {
      const store = await caches.open(name)
      const resp = await store.match(url)
      if (resp) return true
    }
  } catch { /* no Cache API */ }
  return false
}

// ============================================================
// Flatten all records from ndarray-cache.json
// ============================================================

function flattenRecords(cache: NDArrayCache): FlatRecord[] {
  const out: FlatRecord[] = []
  for (const rec of cache.records) {
    if ('records' in rec && Array.isArray(rec.records)) {
      for (const r of rec.records) {
        out.push({ ...r, dataPath: r.dataPath ?? rec.dataPath })
      }
    } else {
      out.push(rec as FlatRecord)
    }
  }
  return out
}

// ============================================================
// Build GPUBuffer from a flat record
// ============================================================

function recordToGPUBuffer(
  device: GPUDevice,
  rec: FlatRecord,
  shardData: ArrayBuffer,
): GPUBuffer {
  const slice = shardData.slice(rec.byteOffset, rec.byteOffset + rec.nbytes)
  const buf = device.createBuffer({
    size: Math.max(rec.nbytes, 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: rec.name,
  })
  device.queue.writeBuffer(buf, 0, slice)
  return buf
}

// ============================================================
// Parameter name lookup
// ============================================================

function find(index: Map<string, FlatRecord>, ...candidates: string[]): FlatRecord {
  for (const c of candidates) {
    const r = index.get(c)
    if (r) return r
  }
  throw new Error(`Weight not found. Tried: ${candidates.join(', ')}`)
}

// ============================================================
// Main loader
// ============================================================

export async function loadWeights(
  device: GPUDevice,
  onProgress?: (p: LoadProgress) => void,
): Promise<LoadedWeights> {
  const baseUrl = PHI3_MODEL_BASE

  const report = (p: Partial<LoadProgress> & { message: string }) =>
    onProgress?.({ phase: 'downloading', bytesLoaded: 0, bytesTotal: 0, percent: 0, ...p })

  // 1. Fetch manifest
  report({ phase: 'manifest', message: 'Fetching model manifest...' })
  const manifestUrl = baseUrl + 'ndarray-cache.json'
  let manifestBuf: ArrayBuffer
  try {
    const store = await caches.open(CACHE_NAME)
    const cached = await store.match(manifestUrl)
    if (cached) {
      manifestBuf = await cached.arrayBuffer()
    } else {
      const resp = await fetch(manifestUrl)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      manifestBuf = await resp.arrayBuffer()
      await store.put(manifestUrl, new Response(manifestBuf.slice(0)))
    }
  } catch {
    const resp = await fetch(manifestUrl)
    if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching manifest`)
    manifestBuf = await resp.arrayBuffer()
  }

  const cacheJson: NDArrayCache = JSON.parse(new TextDecoder().decode(manifestBuf))
  const allRecords = flattenRecords(cacheJson)
  const index = new Map<string, FlatRecord>()
  for (const r of allRecords) index.set(r.name, r)

  // 2. Collect unique shards and total size
  const shardSizes = new Map<string, number>()
  for (const r of allRecords) {
    const existing = shardSizes.get(r.dataPath) ?? 0
    shardSizes.set(r.dataPath, Math.max(existing, r.byteOffset + r.nbytes))
  }

  // Compute total bytes (sum of shard file sizes)
  let totalBytes = 0
  const shardList = [...shardSizes.keys()]
  // Check which shards are already cached
  const shardCachedStatus = new Map<string, boolean>()
  for (const shard of shardList) {
    const cached = await isCached(baseUrl + shard)
    shardCachedStatus.set(shard, cached)
  }

  // For total: we need actual shard file sizes. Best estimate = max(byteOffset + nbytes) per shard
  for (const size of shardSizes.values()) totalBytes += size

  report({
    message: `Found ${index.size} parameters in ${shardList.length} shards (${(totalBytes / 1e9).toFixed(2)} GB)`,
    bytesTotal: totalBytes,
  })

  // 3. Download shards with progress
  const shardCache = new Map<string, ArrayBuffer>()
  let globalBytesLoaded = 0
  // Track per-shard loaded bytes (for incremental progress)
  const shardBytesLoaded = new Map<string, number>()

  async function ensureShard(dataPath: string): Promise<ArrayBuffer> {
    if (shardCache.has(dataPath)) return shardCache.get(dataPath)!

    const url = baseUrl + dataPath
    const wasCached = shardCachedStatus.get(dataPath) ?? false

    const prevBytes = shardBytesLoaded.get(dataPath) ?? 0

    const buf = await fetchWithCache(url, (loaded) => {
      const delta = loaded - (shardBytesLoaded.get(dataPath) ?? 0)
      shardBytesLoaded.set(dataPath, loaded)
      globalBytesLoaded += delta
      const pct = totalBytes > 0 ? Math.min(100, (globalBytesLoaded / totalBytes) * 100) : 0

      onProgress?.({
        phase: 'downloading',
        message: wasCached
          ? `Loading from cache: ${dataPath}`
          : `Downloading: ${dataPath} — ${(globalBytesLoaded / 1e6).toFixed(0)} / ${(totalBytes / 1e6).toFixed(0)} MB`,
        bytesLoaded: globalBytesLoaded,
        bytesTotal: totalBytes,
        percent: pct,
        currentShard: dataPath,
        cacheHit: wasCached,
      })
    })

    shardCache.set(dataPath, buf)
    return buf
  }

  // Helper: load a named parameter
  async function load(name: string, ...alts: string[]): Promise<GPUBuffer> {
    const rec = find(index, name, ...alts)
    const shard = await ensureShard(rec.dataPath)
    return recordToGPUBuffer(device, rec, shard)
  }

  // 4. Load global weights
  const embdWeights = await load('transformer.embd.q_weight', 'embed_tokens.q_weight', 'model.embed_tokens.q_weight')
  const embdScales = await load('transformer.embd.q_scale', 'embed_tokens.q_scale', 'model.embed_tokens.q_scale')
  const initNormGamma = await load('transformer.h.0.ln.weight', 'model.layers.0.input_layernorm.weight')
  const lmHeadWeights = await load('lm_head.q_weight', 'model.lm_head.q_weight')
  const lmHeadScales = await load('lm_head.q_scale', 'model.lm_head.q_scale')

  // 5. Per-layer weights
  const LAYERS = 32
  const layers: LoadedWeights['layers'] = []

  for (let L = 0; L < LAYERS; L++) {
    const h = `transformer.h.${L}`
    const p = `model.layers.${L}`

    const qkvWeights = await load(`${h}.mixer.qkv_proj.q_weight`, `${p}.self_attn.qkv_proj.q_weight`)
    const qkvScales = await load(`${h}.mixer.qkv_proj.q_scale`, `${p}.self_attn.qkv_proj.q_scale`)
    const oProjWeights = await load(`${h}.mixer.out_proj.q_weight`, `${p}.self_attn.o_proj.q_weight`)
    const oProjScales = await load(`${h}.mixer.out_proj.q_scale`, `${p}.self_attn.o_proj.q_scale`)
    const normGamma1 = await load(`${h}.ln.weight`, `${p}.input_layernorm.weight`)
    const normGamma2 = await load(`${h}.post_attention_layernorm.weight`, `${p}.post_attention_layernorm.weight`)
    const ffnWeights = await load(`${h}.mlp.gate_up_proj.q_weight`, `${p}.mlp.gate_up_proj.q_weight`)
    const ffnScales = await load(`${h}.mlp.gate_up_proj.q_scale`, `${p}.mlp.gate_up_proj.q_scale`)
    const ffnDownWeights = await load(`${h}.mlp.down_proj.q_weight`, `${p}.mlp.down_proj.q_weight`)
    const ffnDownScales = await load(`${h}.mlp.down_proj.q_scale`, `${p}.mlp.down_proj.q_scale`)

    layers.push({
      qkvWeights, qkvScales,
      oProjWeights, oProjScales,
      normGamma1, normGamma2,
      ffnWeights, ffnScales,
      ffnDownWeights, ffnDownScales,
    })
  }

  const finalNormGamma = await load('transformer.norm.weight', 'model.norm.weight', 'norm.weight')

  // 6. Upload to GPU complete
  report({
    phase: 'done',
    message: 'All weights loaded!',
    bytesLoaded: totalBytes,
    bytesTotal: totalBytes,
    percent: 100,
  })

  return {
    device,
    embdWeights, embdScales,
    lmHeadWeights, lmHeadScales,
    initNormGamma,
    finalNormGamma,
    layers,
  }
}

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

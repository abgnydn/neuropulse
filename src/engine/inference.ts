/**
 * INFERENCE ENGINE — Phi-3 inference with per-layer visualization callbacks.
 *
 * Adapts chat.ts's buildDecodeEngine to submit GPU work per-layer instead of
 * all-at-once, yielding between layers so the visualizer can animate each step.
 */

import { loadWeights, LoadedWeights, LoadProgress } from './weight-loader'
import { loadTokenizer, buildChatPrompt, Tokenizer } from './tokenizer'
import { compile, PHI3 } from './compiler'
import { f16ToF32 } from './activation-reducer'

// ============================================================
// GPU helpers (same as chat.ts)
// ============================================================

function makeBuf(device: GPUDevice, size: number, label: string): GPUBuffer {
  return device.createBuffer({
    size: Math.max(size, 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label,
  })
}

function uniformBuf(device: GPUDevice, data: ArrayBuffer[]): GPUBuffer {
  const size = data.reduce((s, p) => s + p.byteLength, 0)
  const padded = Math.ceil(size / 16) * 16
  const buf = device.createBuffer({ size: Math.max(padded, 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const arr = new Uint8Array(padded)
  let off = 0
  for (const p of data) { arr.set(new Uint8Array(p), off); off += p.byteLength }
  device.queue.writeBuffer(buf, 0, arr)
  return buf
}

function u32(v: number): ArrayBuffer { const a = new ArrayBuffer(4); new DataView(a).setUint32(0, v, true); return a }
function i32(v: number): ArrayBuffer { const a = new ArrayBuffer(4); new DataView(a).setInt32(0, v, true); return a }
function f32(v: number): ArrayBuffer { const a = new ArrayBuffer(4); new DataView(a).setFloat32(0, v, true); return a }

function bg(device: GPUDevice, pipeline: GPUComputePipeline, bufs: GPUBuffer[]): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
  })
}

function dispatch(
  enc: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  wgX: number, wgY = 1, wgZ = 1
): void {
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(wgX, wgY, wgZ)
  pass.end()
}

// ============================================================
// KV cache
// ============================================================

function allocKVPages(device: GPUDevice): GPUBuffer[] {
  const bytesPerPage = 98304 * 2
  const pages = PHI3.MAX_PAGES * bytesPerPage
  return Array.from({ length: PHI3.LAYERS }, (_, i) =>
    makeBuf(device, pages, `kvPages_${i}`)
  )
}

// ============================================================
// Callbacks
// ============================================================

export interface InferenceCallbacks {
  /** Called for each dispatch step with optional real activation data (f32, variable length).
   *  Return a Promise to pace the visualization. */
  onLayer?: (layer: number, step: number, stepName: string, activations?: Float32Array) => void | Promise<void>
  /** Called for each generated token */
  onToken?: (token: string, id: number, index: number) => void
  /** Called during weight loading */
  onProgress?: (msg: string) => void
  /** Called when prefill starts/ends */
  onPrefill?: (phase: 'start' | 'end', promptLength: number) => void
}

const STEP_NAMES = ['QKV Matmul', 'RoPE', 'KV Append', 'Attention', 'O Project', 'Add+Norm', 'FFN Gate+Up', 'FFN Down', 'Add+Norm']

// ============================================================
// Engine
// ============================================================

export type { LoadProgress }

export interface InferenceEngine {
  generate(prompt: string, maxTokens: number, callbacks: InferenceCallbacks): Promise<string>
  tokenizer: Tokenizer
  ready: boolean
}

export async function createInferenceEngine(
  onProgress?: (p: LoadProgress) => void
): Promise<InferenceEngine> {
  if (!navigator.gpu) throw new Error('WebGPU not supported')

  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('No GPU adapter found')

  const report = (message: string, extra?: Partial<LoadProgress>) =>
    onProgress?.({ phase: 'manifest', message, bytesLoaded: 0, bytesTotal: 0, percent: 0, ...extra })

  report('Requesting GPU device...')
  const device = await adapter.requestDevice({
    requiredFeatures: ['shader-f16' as GPUFeatureName],
  })

  report('Loading tokenizer...')
  const tokenizer = await loadTokenizer((msg) => report(msg))

  report('Loading model weights...')
  const weights = await loadWeights(device, onProgress)

  report('Allocating KV cache...', { phase: 'uploading', percent: 98 })
  const kvPages = allocKVPages(device)

  onProgress?.('Compiling shaders...')
  const { pipelines: P } = compile(device)

  // Activation buffers
  const B = {
    residual:   makeBuf(device, PHI3.D * 2, 'residual'),
    residual2:  makeBuf(device, PHI3.D * 2, 'residual2'),
    hidden1:    makeBuf(device, PHI3.D * 2, 'hidden1'),
    hidden2:    makeBuf(device, PHI3.D * 2, 'hidden2'),
    qkvOut:     makeBuf(device, 9216 * 2, 'qkvOut'),
    qOut:       makeBuf(device, PHI3.D * 2, 'qOut'),
    kOut:       makeBuf(device, PHI3.D * 2, 'kOut'),
    vOut:       makeBuf(device, PHI3.D * 2, 'vOut'),
    attnOut:    makeBuf(device, PHI3.D * 2, 'attnOut'),
    ffnOut:     makeBuf(device, PHI3.FFN * 2, 'ffnOut'),
    logits:     makeBuf(device, PHI3.VOCAB * 4, 'logits'),
    tokenOut:   makeBuf(device, 4, 'tokenOut'),
    inputIds:   makeBuf(device, 4, 'inputIds'),
    posMap:     makeBuf(device, 4, 'posMap'),
    pageIndptr: makeBuf(device, 8, 'pageIndptr'),
    pageValues: makeBuf(device, PHI3.MAX_PAGES * 4, 'pageValues'),
    lengthInfo: makeBuf(device, 12, 'lengthInfo'),
  }

  // Static uniforms
  const qkvU    = uniformBuf(device, [u32(384), u32(96), u32(9216)])
  const oProjU  = uniformBuf(device, [u32(384), u32(96), u32(3072)])
  const ffnDnU  = uniformBuf(device, [u32(1024), u32(256), u32(3072)])
  const lmHdU   = uniformBuf(device, [u32(384), u32(96), u32(PHI3.VOCAB)])
  const embU    = uniformBuf(device, [u32(1), u32(12)])
  const normU   = uniformBuf(device, [u32(1)])
  const ffnU    = uniformBuf(device, [u32(PHI3.FFN)])
  const argmaxU = uniformBuf(device, [u32(PHI3.VOCAB)])

  // Page table identity
  const pageVals = new Int32Array(PHI3.MAX_PAGES)
  for (let i = 0; i < PHI3.MAX_PAGES; i++) pageVals[i] = i
  device.queue.writeBuffer(B.pageValues, 0, pageVals)

  const SM_SCALE = 1.0 / Math.sqrt(PHI3.HEAD_DIM)

  // Readback staging buffer (sized for largest buffer: qkvOut = 9216 * 2 = 18432 bytes)
  const readbackBuf = device.createBuffer({
    size: 9216 * 2,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    label: 'readback_staging',
  })

  /** Read f16 activation values from a GPU buffer, convert to f32 */
  async function readActivations(srcBuf: GPUBuffer, f16Count: number): Promise<Float32Array> {
    const byteSize = f16Count * 2
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(srcBuf, 0, readbackBuf, 0, byteSize)
    device.queue.submit([enc.finish()])
    await readbackBuf.mapAsync(GPUMapMode.READ)
    const f16 = new Uint16Array(readbackBuf.getMappedRange(0, byteSize).slice(0))
    readbackBuf.unmap()
    return f16ToF32(f16)
  }

  // Which steps get readback and from which buffer + size
  // Steps without readback get undefined activations (visualizer uses fallback)
  const READBACK_STEPS: Record<number, { buf: () => GPUBuffer, count: number }> = {
    0: { buf: () => B.qkvOut, count: 9216 },     // QKV Matmul → full QKV projection
    3: { buf: () => B.attnOut, count: 3072 },     // Attention → per-head output
    5: { buf: () => B.hidden1, count: 3072 },     // Add+Norm (attn) → normed hidden
    6: { buf: () => B.ffnOut, count: 8192 },      // FFN Gate+Up → most variance
    8: { buf: () => B.hidden1, count: 3072 },     // Add+Norm (FFN) → final layer output
  }

  /**
   * Decode one token with per-step callbacks.
   * When onLayer is provided, submits GPU work per-step and awaits the
   * callback (which can return a Promise with a delay for speed control).
   * When onLayer is null, batches everything into one submit (fast prefill).
   */
  async function decodeToken(
    tokenId: number,
    position: number,
    onLayer?: (layer: number, step: number, stepName: string, activations?: Float32Array) => void | Promise<void>,
    perStepYield = true
  ): Promise<number> {
    const nnzPages = Math.floor(position / PHI3.PAGE_SIZE) + 1

    device.queue.writeBuffer(B.inputIds, 0, new Int32Array([tokenId]))
    device.queue.writeBuffer(B.posMap, 0, new Int32Array([position]))
    device.queue.writeBuffer(B.pageIndptr, 0, new Int32Array([0, nnzPages]))
    device.queue.writeBuffer(B.lengthInfo, 0, new Int32Array([position + 1, 0, 0]))

    // --- EMBEDDING ---
    const embEnc = device.createCommandEncoder()
    dispatch(embEnc, P.embedding, bg(device, P.embedding, [
      B.residual, B.inputIds, weights.embdScales, weights.embdWeights, embU,
    ]), 12)
    dispatch(embEnc, P.rmsNorm, bg(device, P.rmsNorm, [
      B.hidden1, B.residual, weights.layers[0].normGamma1, normU,
    ]), 1)
    device.queue.submit([embEnc.finish()])

    // Helper: submit GPU work, optionally read back activations, call onLayer
    async function step(
      enc: GPUCommandEncoder, L: number, s: number
    ) {
      device.queue.submit([enc.finish()])
      if (perStepYield && onLayer) {
        // Read back activations for key steps
        const rb = READBACK_STEPS[s]
        let activations: Float32Array | undefined
        if (rb) {
          activations = await readActivations(rb.buf(), rb.count)
        }
        await onLayer(L, s, STEP_NAMES[s], activations)
      }
    }

    // --- 32 TRANSFORMER LAYERS ---
    let resIn = B.residual
    let resOut = B.residual2

    for (let L = 0; L < PHI3.LAYERS; L++) {
      const lw = weights.layers[L]

      if (!perStepYield) {
        // Fast mode: batch entire layer into one encoder, no callbacks
        const enc = device.createCommandEncoder()

        dispatch(enc, P.qkvMatmul, bg(device, P.qkvMatmul, [
          B.qkvOut, B.hidden1, lw.qkvScales, lw.qkvWeights, qkvU,
        ]), 9216)

        const ropeU = uniformBuf(device, [i32(1), i32(0), i32(1), u32(36)])
        dispatch(enc, P.rope, bg(device, P.rope, [
          B.qOut, B.kOut, B.vOut, B.qkvOut, B.posMap, ropeU,
        ]), 36)

        const kvAppU = uniformBuf(device, [i32(1), i32(PHI3.MAX_PAGES), i32(0), i32(0), u32(12)])
        dispatch(enc, P.kvAppend, bg(device, P.kvAppend, [
          B.kOut, B.vOut, kvPages[L], B.posMap, kvAppU,
        ]), 12)

        const attnU = uniformBuf(device, [
          i32(1), i32(PHI3.MAX_PAGES), i32(nnzPages), i32(0),
          i32(0), i32(0), i32(0), f32(SM_SCALE), u32(1),
        ])
        dispatch(enc, P.attention, bg(device, P.attention, [
          B.qOut, B.pageIndptr, B.pageValues, kvPages[L],
          B.lengthInfo, B.attnOut, attnU,
        ]), 1, PHI3.HEADS)

        dispatch(enc, P.oProjMatmul, bg(device, P.oProjMatmul, [
          B.hidden2, B.attnOut, lw.oProjScales, lw.oProjWeights, oProjU,
        ]), 3072)

        dispatch(enc, P.addNorm, bg(device, P.addNorm, [
          B.hidden2, resIn, lw.normGamma2, B.hidden1, resOut, normU,
        ]), 1)
        ;[resIn, resOut] = [resOut, resIn]

        dispatch(enc, P.fusedFfn, bg(device, P.fusedFfn, [
          B.ffnOut, B.hidden1, lw.ffnScales, lw.ffnWeights, ffnU,
        ]), PHI3.FFN)

        dispatch(enc, P.ffnDownMatmul, bg(device, P.ffnDownMatmul, [
          B.hidden2, B.ffnOut, lw.ffnDownScales, lw.ffnDownWeights, ffnDnU,
        ]), 3072)

        const nextGamma = L < PHI3.LAYERS - 1
          ? weights.layers[L + 1].normGamma1
          : weights.finalNormGamma
        dispatch(enc, P.addNorm, bg(device, P.addNorm, [
          B.hidden2, resIn, nextGamma, B.hidden1, resOut, normU,
        ]), 1)
        ;[resIn, resOut] = [resOut, resIn]

        device.queue.submit([enc.finish()])
        continue
      }

      // --- Slow/visualized mode: one submit per step, await callback ---

      // [0] QKV matmul
      let enc = device.createCommandEncoder()
      dispatch(enc, P.qkvMatmul, bg(device, P.qkvMatmul, [
        B.qkvOut, B.hidden1, lw.qkvScales, lw.qkvWeights, qkvU,
      ]), 9216)
      await step(enc, L, 0)

      // [1] RoPE
      enc = device.createCommandEncoder()
      const ropeU = uniformBuf(device, [i32(1), i32(0), i32(1), u32(36)])
      dispatch(enc, P.rope, bg(device, P.rope, [
        B.qOut, B.kOut, B.vOut, B.qkvOut, B.posMap, ropeU,
      ]), 36)
      await step(enc, L, 1)

      // [2] KV append
      enc = device.createCommandEncoder()
      const kvAppU = uniformBuf(device, [i32(1), i32(PHI3.MAX_PAGES), i32(0), i32(0), u32(12)])
      dispatch(enc, P.kvAppend, bg(device, P.kvAppend, [
        B.kOut, B.vOut, kvPages[L], B.posMap, kvAppU,
      ]), 12)
      await step(enc, L, 2)

      // [3] Attention
      enc = device.createCommandEncoder()
      const attnU = uniformBuf(device, [
        i32(1), i32(PHI3.MAX_PAGES), i32(nnzPages), i32(0),
        i32(0), i32(0), i32(0), f32(SM_SCALE), u32(1),
      ])
      dispatch(enc, P.attention, bg(device, P.attention, [
        B.qOut, B.pageIndptr, B.pageValues, kvPages[L],
        B.lengthInfo, B.attnOut, attnU,
      ]), 1, PHI3.HEADS)
      await step(enc, L, 3)

      // [4] O projection
      enc = device.createCommandEncoder()
      dispatch(enc, P.oProjMatmul, bg(device, P.oProjMatmul, [
        B.hidden2, B.attnOut, lw.oProjScales, lw.oProjWeights, oProjU,
      ]), 3072)
      await step(enc, L, 4)

      // [5] AddNorm (attention)
      enc = device.createCommandEncoder()
      dispatch(enc, P.addNorm, bg(device, P.addNorm, [
        B.hidden2, resIn, lw.normGamma2, B.hidden1, resOut, normU,
      ]), 1)
      ;[resIn, resOut] = [resOut, resIn]
      await step(enc, L, 5)

      // [6] Fused FFN gate+up+SiLU
      enc = device.createCommandEncoder()
      dispatch(enc, P.fusedFfn, bg(device, P.fusedFfn, [
        B.ffnOut, B.hidden1, lw.ffnScales, lw.ffnWeights, ffnU,
      ]), PHI3.FFN)
      await step(enc, L, 6)

      // [7] FFN down
      enc = device.createCommandEncoder()
      dispatch(enc, P.ffnDownMatmul, bg(device, P.ffnDownMatmul, [
        B.hidden2, B.ffnOut, lw.ffnDownScales, lw.ffnDownWeights, ffnDnU,
      ]), 3072)
      await step(enc, L, 7)

      // [8] AddNorm (FFN)
      enc = device.createCommandEncoder()
      const nextGamma = L < PHI3.LAYERS - 1
        ? weights.layers[L + 1].normGamma1
        : weights.finalNormGamma
      dispatch(enc, P.addNorm, bg(device, P.addNorm, [
        B.hidden2, resIn, nextGamma, B.hidden1, resOut, normU,
      ]), 1)
      ;[resIn, resOut] = [resOut, resIn]
      await step(enc, L, 8)
    }

    // --- LM HEAD + ARGMAX ---
    const headEnc = device.createCommandEncoder()
    dispatch(headEnc, P.lmHead, bg(device, P.lmHead, [
      B.logits, B.hidden1, weights.lmHeadScales, weights.lmHeadWeights, lmHdU,
    ]), PHI3.VOCAB)
    dispatch(headEnc, P.argmax, bg(device, P.argmax, [
      B.logits, B.tokenOut, argmaxU,
    ]), 1)
    device.queue.submit([headEnc.finish()])

    // Read result
    const readBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const readEnc = device.createCommandEncoder()
    readEnc.copyBufferToBuffer(B.tokenOut, 0, readBuf, 0, 4)
    device.queue.submit([readEnc.finish()])

    await readBuf.mapAsync(GPUMapMode.READ)
    const result = new DataView(readBuf.getMappedRange()).getInt32(0, true)
    readBuf.unmap()
    readBuf.destroy()

    return result
  }

  const STOP = new Set([2, 32000, 32007])

  async function generate(
    prompt: string,
    maxTokens: number,
    callbacks: InferenceCallbacks
  ): Promise<string> {
    const messages = [
      { role: 'system' as const, content: 'You are a helpful assistant.' },
      { role: 'user' as const, content: prompt },
    ]
    const promptIds = buildChatPrompt(messages, tokenizer)

    callbacks.onPrefill?.('start', promptIds.length)

    // Prefill: no per-layer yield (fast)
    for (let i = 0; i < promptIds.length; i++) {
      await decodeToken(promptIds[i], i, undefined, false)
    }

    callbacks.onPrefill?.('end', promptIds.length)

    // Read first generated token
    const readBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const readEnc = device.createCommandEncoder()
    readEnc.copyBufferToBuffer(B.tokenOut, 0, readBuf, 0, 4)
    device.queue.submit([readEnc.finish()])
    await readBuf.mapAsync(GPUMapMode.READ)
    let tokenId = new DataView(readBuf.getMappedRange()).getInt32(0, true)
    readBuf.unmap()
    readBuf.destroy()

    // Decode loop
    const allIds: number[] = []
    let pos = promptIds.length

    for (let i = 0; i < maxTokens; i++) {
      if (tokenId < 0 || tokenId >= PHI3.VOCAB || STOP.has(tokenId)) break

      allIds.push(tokenId)
      const fullText = tokenizer.decode(allIds)
      const prevText = allIds.length > 1 ? tokenizer.decode(allIds.slice(0, -1)) : ''
      const delta = fullText.slice(prevText.length)
      callbacks.onToken?.(delta, tokenId, i)

      pos++
      tokenId = await decodeToken(tokenId, pos, callbacks.onLayer, true)
    }

    return tokenizer.decode(allIds)
  }

  report('Engine ready!', { phase: 'done', percent: 100 })

  return { generate, tokenizer, ready: true }
}

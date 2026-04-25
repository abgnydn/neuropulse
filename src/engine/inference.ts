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

// Layers whose post-FFN residual is snapshotted for HF cross-validation.
// Must match LAYERS_TO_CAPTURE in tools/dump_phi3_reference.py.
const VALIDATE_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31] as const

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

export interface TopKEntry {
  token: string
  id: number
  prob: number
}

/** Per-generation attention-head ablation. If `head` is omitted the entire
 *  layer's attention output is zeroed (all 32 heads). Semantics: attention
 *  still runs normally, but the head's 96-dim slice of attnOut is cleared
 *  before O-proj mixes heads — so the head contributes zero to the residual
 *  stream for every generated token. KV cache is untouched. */
export interface Ablation {
  layer: number
  head?: number
}

export interface InferenceCallbacks {
  /** Called for each dispatch step with optional real activation data (f32, variable length).
   *  Return a Promise to pace the visualization. */
  onLayer?: (layer: number, step: number, stepName: string, activations?: Float32Array) => void | Promise<void>
  /** Called for each generated token, includes top-k candidates and full logits (32064 f32) */
  onToken?: (token: string, id: number, index: number, topK?: TopKEntry[], logits?: Float32Array) => void
  /** Called during weight loading */
  onProgress?: (msg: string) => void
  /** Called when prefill starts/ends */
  onPrefill?: (phase: 'start' | 'end', promptLength: number) => void
  /** Called during prefill for each prompt token (for animation) */
  onPrefillToken?: (tokenIndex: number, totalTokens: number, tokenText: string) => void | Promise<void>
  /** Called with KV cache info after each token */
  onKVCache?: (position: number, totalPages: number, usedPages: number) => void
  /** Called once with the embedded vector for the current input token (3072 f32) */
  onEmbedding?: (tokenId: number, embedding: Float32Array) => void
  /** Called once per generated token with post-softmax attention scores
   *  for ALL 32 layers. scores has shape
   *  [LAYERS=32 × HEADS=32 × MAX_SCORE_SLOTS=256], all f32. Each layer's
   *  block is at offset (layer * 32 * 256). Slots beyond kv_len are 0. */
  onAllAttentionScores?: (scores: Float32Array, kvLen: number) => void
  /** Logit lens: called at the end of each sampled layer after add+norm.
   *  Projects the raw post-layer residual through finalRMSNorm + lm_head +
   *  argmax and reports what the model "would say next" if it stopped at
   *  this layer. Only fires for layers in LENS_LAYERS during per-step
   *  visualized decode (not during prefill or the validation suite). */
  onLayerLogitLens?: (layer: number, tokenId: number, token: string) => void
}

/** Layers at which the logit lens runs during visualized decode. Each lens
 *  dispatch costs an extra rmsNorm + full lm_head + argmax + 4-byte readback
 *  (~5ms on Apple M-series). Keep this sparse so per-token overhead stays
 *  under ~50ms. L=31 is omitted because the real final lm_head produces the
 *  same result for free. */
const LENS_LAYERS: ReadonlySet<number> = new Set([0, 4, 8, 12, 16, 20, 24, 28])

const ATTN_SCORE_LAYERS = 32
const ATTN_SCORE_HEADS = 32
const ATTN_SCORE_MAX_SLOTS = 256
const ATTN_SCORE_LAYER_WORDS = ATTN_SCORE_HEADS * ATTN_SCORE_MAX_SLOTS // 8192
const ATTN_SCORE_TOTAL_WORDS = ATTN_SCORE_LAYERS * ATTN_SCORE_LAYER_WORDS // 262144
const ATTN_SCORE_TOTAL_BYTES = ATTN_SCORE_TOTAL_WORDS * 4 // 1 MiB

const STEP_NAMES = ['QKV Matmul', 'RoPE', 'KV Append', 'Attention', 'O Project', 'Add+Norm', 'FFN Gate+Up', 'FFN Down', 'Add+Norm']

// ============================================================
// Engine
// ============================================================

export type { LoadProgress }

export interface ValidationResult {
  /** Layer the validation was run on (currently always LAYERS-1). */
  layer: number
  /** kv_len at the moment of validation. */
  kvLen: number
  /** L2 norm of (reconstructed_from_scores - actual_attnOut). */
  l2Error: number
  /** L2 norm of actual_attnOut (denominator for relative error). */
  attnNorm: number
  /** Maximum absolute element-wise error. */
  maxError: number
  /** l2Error / attnNorm — relative error (target: < 1e-2 for f16). */
  relError: number
  /** Whether the test passed (relError < 1e-2). */
  passed: boolean
}

/** One per-layer hidden-state diff vs HF reference. */
export interface LayerHiddenDiff {
  /** Layer index (or -1 for the embedding output / "input to layer 0"). */
  layer: number
  /** L2 norm of (gpu - hf) over the captured leading dims. */
  l2Error: number
  /** L2 norm of hf (denominator for relative error). */
  hfNorm: number
  /** L2 norm of gpu (for debugging scale mismatches). */
  gpuNorm: number
  /** Cosine similarity in [-1, 1]. */
  cosine: number
  /** Relative L2 error = l2Error / hfNorm. */
  relError: number
  /** First 8 GPU values, for debugging. */
  gpuHead8: number[]
  /** First 8 HF values, for debugging. */
  hfHead8: number[]
}

/** Result of one decode step diff vs HF reference. */
export interface TokenDiff {
  step: number
  /** GPU greedy token id. */
  gpuId: number
  /** HF greedy token id. */
  hfId: number
  /** True iff gpuId === hfId. */
  match: boolean
  /** Jensen-Shannon divergence between top-K probability distributions. */
  jsd: number
  /** Number of GPU top-5 ids that also appear in HF top-5. */
  top5Overlap: number
}

/** Result of checking one prompt: tokenizer + teacher-forced logit diff. */
export interface PromptCheck {
  /** The prompt text. */
  prompt: string
  /** Whether GPU tokenizer matches HF byte-for-byte. */
  tokenizerAgrees: boolean
  /** GPU input ids after chat template. */
  gpuInputIds: number[]
  /** HF input ids from reference.json. */
  hfInputIds: number[]
  /** Per-step teacher-forced logit diff. */
  tokenDiffs: TokenDiff[]
  /** Number of top-1 matches across all steps. */
  topMatches: number
  /** Mean JSD across all steps. */
  meanJsd: number
}

/** Sampling self-test: empirical distribution vs theoretical softmax. */
export interface SamplingSelfTest {
  /** Number of Monte Carlo samples drawn. */
  numSamples: number
  /** Temperature used for softmax. */
  temperature: number
  /** Number of distinct ids sampled. */
  uniqueIds: number
  /** JSD between empirical frequencies and theoretical softmax (top-20). */
  empiricalJsd: number
  /** Max |empirical_freq - theoretical_prob| over the top-20 ids. */
  maxL1Error: number
  /** Passes if empiricalJsd < 1e-2 and maxL1Error < 0.02. */
  passed: boolean
}

/** Full validation report — printed to console at boot. */
export interface ValidationReport {
  /** Whether reference.json was loaded. False = no validation possible. */
  hasReference: boolean
  /** Main prompt: tokenizer + hidden states + teacher-forced logits. */
  main: {
    prompt: string
    tokenizerAgrees: boolean
    gpuInputIds: number[]
    hfInputIds: number[]
    layerDiffs: LayerHiddenDiff[]
    tokenDiffs: TokenDiff[]
    topMatches: number
    meanJsd: number
  }
  /** Sweep prompts — 15 short prompts covering ASCII/Unicode/emoji/JSON. */
  sweep: PromptCheck[]
  /** Long-context prompt — ~290 tokens, exercises paged KV beyond one page. */
  longContext: (PromptCheck & { kvLen: number })
  /** Attention shader equivalence on layer 31 (online ≡ explicit). */
  attentionEquivalence: ValidationResult
  /** Sampling self-test (empirical vs theoretical softmax distribution). */
  samplingSelfTest: SamplingSelfTest
  /** Single-line summary string. */
  summary: string
}

export interface InferenceEngine {
  generate(prompt: string, maxTokens: number, callbacks: InferenceCallbacks, ablations?: Ablation[]): Promise<string>
  /** Request the in-flight `generate()` to stop at the next token boundary.
   *  Returns truthfully whether a run was active when called. The promise
   *  returned by the original generate() resolves cleanly with whatever was
   *  decoded so far — it does NOT reject. */
  interrupt(): boolean
  /** Dispatch only the embedding kernel for one token id and read back the
   *  first 32 dims. Purely a debugging hook for comparing the GPU's q4
   *  embedding against an HF fp16 reference. */
  debugEmbedToken?(tokenId: number): Promise<Float32Array>
  /** Run a numerical equivalence check between attention.wgsl and attention_scores.wgsl
   *  using the live state from the most recent token. Verifies that
   *    attnOut[h,d] ≈ sum_s scores[h,s] * V[s,h,d]
   *  which proves the two shaders use the same softmax weights. */
  validateLastAttention(): Promise<ValidationResult>
  /** Run the full HF cross-validation suite (tokenizer + per-layer hidden
   *  states + 20-step greedy decode). Resolves with a complete report. */
  runValidationSuite(): Promise<ValidationReport>
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
    attnScores: makeBuf(device, ATTN_SCORE_TOTAL_BYTES, 'attnScores'),
    // Validation snapshots: copied from attnOut/attnScores/lengthInfo/page
    // table immediately after layer 31's attention dispatch each token, so
    // validateLastAttention sees a fully coherent state from the same token.
    attnOutSnap:    makeBuf(device, PHI3.D * 2, 'attnOutSnap'),
    attnScoresSnap: makeBuf(device, ATTN_SCORE_TOTAL_BYTES, 'attnScoresSnap'),
    lenInfoSnap:    makeBuf(device, 12, 'lenInfoSnap'),
    pageIndptrSnap: makeBuf(device, 8, 'pageIndptrSnap'),
    pageValuesSnap: makeBuf(device, PHI3.MAX_PAGES * 4, 'pageValuesSnap'),
    // Per-layer hidden-state snapshots for HF cross-validation. We snapshot
    // the residual stream right after each layer's FFN AddNorm (= what HF
    // calls hidden_states[L]). Indexed by VALIDATE_LAYERS position.
    hiddenSnaps:    VALIDATE_LAYERS.map((L) =>
      makeBuf(device, PHI3.D * 2, `hiddenSnap_L${L}`)),
    // Snapshot of the embedding output (residual stream BEFORE layer 0)
    embeddingSnap:  makeBuf(device, PHI3.D * 2, 'embeddingSnap'),
    // Logit lens scratch: rmsNorm(residual, finalGamma) is written here so
    // the main pipeline's B.hidden1 (which holds the next layer's
    // differently-scaled normed input) is not clobbered.
    lensHidden:     makeBuf(device, PHI3.D * 2, 'lensHidden'),
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

  // Dedicated f32 readback for attention scores (32 layers × 32 heads × 256 slots × 4 bytes = 1 MiB)
  const attnScoresReadBuf = device.createBuffer({
    size: ATTN_SCORE_TOTAL_BYTES,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    label: 'attn_scores_readback',
  })

  // Tiny persistent readback for the logit-lens argmax id (one i32 per call).
  const lensTokenReadBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    label: 'lens_token_readback',
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

  /** Read full per-token attention scores buffer (all 32 layers). */
  async function readAttentionScores(): Promise<Float32Array> {
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(B.attnScores, 0, attnScoresReadBuf, 0, ATTN_SCORE_TOTAL_BYTES)
    device.queue.submit([enc.finish()])
    await attnScoresReadBuf.mapAsync(GPUMapMode.READ)
    const out = new Float32Array(attnScoresReadBuf.getMappedRange().slice(0))
    attnScoresReadBuf.unmap()
    return out
  }

  /**
   * Decode one token with per-step callbacks.
   * When onLayer is provided, submits GPU work per-step and awaits the
   * callback (which can return a Promise with a delay for speed control).
   * When onLayer is null, batches everything into one submit (fast prefill).
   */
  /**
   * Logit lens: project a raw residual-stream buffer through
   * rmsNorm(finalGamma) → lm_head → argmax, read back the argmax id, and
   * invoke the callback with a decoded token. Reuses B.logits / B.tokenOut
   * as scratch — safe because during a decode step those are only read at
   * the very end (by generate) AFTER the real final lm_head repopulates
   * them. B.lensHidden is a dedicated buffer so the next layer's normed
   * input in B.hidden1 is NOT disturbed.
   */
  async function logitLensAt(
    srcResidual: GPUBuffer,
    layer: number,
    onLayerLogitLens: (layer: number, tokenId: number, token: string) => void,
  ): Promise<void> {
    const enc = device.createCommandEncoder()
    dispatch(enc, P.rmsNorm, bg(device, P.rmsNorm, [
      B.lensHidden, srcResidual, weights.finalNormGamma, normU,
    ]), 1)
    dispatch(enc, P.lmHead, bg(device, P.lmHead, [
      B.logits, B.lensHidden, weights.lmHeadScales, weights.lmHeadWeights, lmHdU,
    ]), PHI3.VOCAB)
    dispatch(enc, P.argmax, bg(device, P.argmax, [
      B.logits, B.tokenOut, argmaxU,
    ]), 1)
    enc.copyBufferToBuffer(B.tokenOut, 0, lensTokenReadBuf, 0, 4)
    device.queue.submit([enc.finish()])
    await lensTokenReadBuf.mapAsync(GPUMapMode.READ)
    const topId = new DataView(lensTokenReadBuf.getMappedRange().slice(0)).getInt32(0, true)
    lensTokenReadBuf.unmap()
    onLayerLogitLens(layer, topId, tokenizer.decode([topId]))
  }

  async function decodeToken(
    tokenId: number,
    position: number,
    onLayer?: (layer: number, step: number, stepName: string, activations?: Float32Array) => void | Promise<void>,
    perStepYield = true,
    /** If true, capture post-softmax attention scores for ALL 32 layers. */
    captureAllScores = false,
    onAllAttentionScores?: (scores: Float32Array, kvLen: number) => void,
    onEmbedding?: (tokenId: number, embedding: Float32Array) => void,
    /** If true, snapshot the residual stream after each layer in
     *  VALIDATE_LAYERS into B.hiddenSnaps for later HF cross-validation. */
    captureHiddenStates = false,
    /** Logit lens callback — fires at each layer in LENS_LAYERS. */
    onLayerLogitLens?: (layer: number, tokenId: number, token: string) => void,
    /** Per-layer attention-head ablation. Key = layer index.
     *  Value = 'all' → zero the whole attnOut; Set<head> → zero only those
     *  heads. Applied on the encoder that dispatches oProjMatmul, so any
     *  attention-step callbacks still see the real pre-ablation attnOut. */
    ablationByLayer?: Map<number, Set<number> | 'all'>,
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
    if (captureHiddenStates) {
      // Snapshot the embedding output (= residual stream before layer 0).
      // HF calls this hidden_states[0] (embedding output).
      embEnc.copyBufferToBuffer(B.residual, 0, B.embeddingSnap, 0, PHI3.D * 2)
    }
    dispatch(embEnc, P.rmsNorm, bg(device, P.rmsNorm, [
      B.hidden1, B.residual, weights.layers[0].normGamma1, normU,
    ]), 1)
    device.queue.submit([embEnc.finish()])

    // Optional readback of token embedding (for visualization)
    if (perStepYield && onEmbedding) {
      const emb = await readActivations(B.residual, PHI3.D)
      onEmbedding(tokenId, emb)
    }

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

    // Zero B.attnOut in-place on the given encoder for each ablated head at
    // layer L. f16, so each head occupies HEAD_DIM*2 = 192 bytes at offset
    // head*192. 'all' → one 6144-byte clear covering all 32 heads.
    const applyAblation = (enc: GPUCommandEncoder, L: number) => {
      const abl = ablationByLayer?.get(L)
      if (!abl) return
      if (abl === 'all') {
        enc.clearBuffer(B.attnOut, 0, PHI3.D * 2)
        return
      }
      for (const h of abl) {
        enc.clearBuffer(B.attnOut, h * PHI3.HEAD_DIM * 2, PHI3.HEAD_DIM * 2)
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

        applyAblation(enc, L)

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

        // Hidden-state snapshot: HF's hidden_states[L+1] convention.
        // For L ∈ [0, 30]: copy resOut (= raw residual = layer L output).
        // For L = 31: HF's hidden_states[32] is norm(layer 31 output), which
        // is what P.addNorm above wrote to B.hidden1 (since nextGamma was
        // weights.finalNormGamma for the last layer). Snap hidden1 instead.
        if (captureHiddenStates) {
          const snapIdx = VALIDATE_LAYERS.indexOf(L as typeof VALIDATE_LAYERS[number])
          if (snapIdx >= 0) {
            const src = (L === PHI3.LAYERS - 1) ? B.hidden1 : resOut
            enc.copyBufferToBuffer(src, 0, B.hiddenSnaps[snapIdx], 0, PHI3.D * 2)
          }
        }
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
      // Capture post-softmax attention scores for THIS layer into the
      // mega-buffer at offset L * 32 * 256 words. The dispatch only depends
      // on captureAllScores — the callback is just for streaming the result
      // out, but the GPU write must always happen so that the layer-31
      // snapshot below has fresh data for validateLastAttention.
      if (captureAllScores) {
        const scoresU = uniformBuf(device, [
          i32(1), i32(PHI3.MAX_PAGES), i32(nnzPages), i32(0),
          i32(0), i32(0), i32(0), f32(SM_SCALE), u32(1),
          i32(L * ATTN_SCORE_LAYER_WORDS),
        ])
        dispatch(enc, P.attentionScores, bg(device, P.attentionScores, [
          B.qOut, B.pageIndptr, B.pageValues, kvPages[L],
          B.lengthInfo, B.attnScores, scoresU,
        ]), 1, PHI3.HEADS)
      }
      // Validation snapshot: after layer 31's attention dispatch (and its
      // scores companion if enabled), copy attnOut and the full per-layer
      // scores buffer into private snapshot buffers. This guarantees
      // validateLastAttention always sees a coherent (out, scores) pair from
      // layer 31 of the *same* decoded token, regardless of when it's called.
      if (L === PHI3.LAYERS - 1 && captureAllScores) {
        enc.copyBufferToBuffer(B.attnOut,    0, B.attnOutSnap,    0, PHI3.D * 2)
        enc.copyBufferToBuffer(B.attnScores, 0, B.attnScoresSnap, 0, ATTN_SCORE_TOTAL_BYTES)
        enc.copyBufferToBuffer(B.lengthInfo, 0, B.lenInfoSnap,    0, 12)
        enc.copyBufferToBuffer(B.pageIndptr, 0, B.pageIndptrSnap, 0, 8)
        enc.copyBufferToBuffer(B.pageValues, 0, B.pageValuesSnap, 0, PHI3.MAX_PAGES * 4)
      }
      await step(enc, L, 3)

      // [4] O projection
      enc = device.createCommandEncoder()
      applyAblation(enc, L)
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
      if (captureHiddenStates) {
        const snapIdx = VALIDATE_LAYERS.indexOf(L as typeof VALIDATE_LAYERS[number])
        if (snapIdx >= 0) {
          const src = (L === PHI3.LAYERS - 1) ? B.hidden1 : resOut
          enc.copyBufferToBuffer(src, 0, B.hiddenSnaps[snapIdx], 0, PHI3.D * 2)
        }
      }
      ;[resIn, resOut] = [resOut, resIn]
      await step(enc, L, 8)

      // Logit lens (sparse — only on LENS_LAYERS during visualized decode).
      // After the swap above, resIn is the buffer that holds the freshly
      // computed layer-L output residual (the "hidden_states[L+1]" in HF
      // terms, pre-normalization). Project it through the final rmsNorm +
      // lm_head to see what the model would predict if it stopped here.
      if (perStepYield && onLayerLogitLens && LENS_LAYERS.has(L)) {
        await logitLensAt(resIn, L, onLayerLogitLens)
      }
    }

    // After all 32 layers' attention_scores have been written, read the
    // mega-buffer ONCE and emit a single callback with the full tensor.
    if (perStepYield && captureAllScores && onAllAttentionScores) {
      try {
        const scores = await readAttentionScores()
        onAllAttentionScores(scores, position + 1)
      } catch { /* non-critical */ }
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

    // Read result token
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

  // Logits readback buffer (32064 * 4 bytes = ~125KB)
  const logitsReadBuf = device.createBuffer({
    size: PHI3.VOCAB * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    label: 'logits_readback',
  })

  /** Read logits and compute top-k with softmax probabilities. Also returns the raw logits. */
  async function readTopK(k: number): Promise<{ entries: TopKEntry[], logits: Float32Array }> {
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(B.logits, 0, logitsReadBuf, 0, PHI3.VOCAB * 4)
    device.queue.submit([enc.finish()])
    await logitsReadBuf.mapAsync(GPUMapMode.READ)
    const logits = new Float32Array(logitsReadBuf.getMappedRange().slice(0))
    logitsReadBuf.unmap()

    // Find top-k by scanning
    const topIndices: number[] = []
    const topValues: number[] = []
    for (let i = 0; i < logits.length; i++) {
      const v = logits[i]
      if (topIndices.length < k) {
        topIndices.push(i)
        topValues.push(v)
      } else {
        let minIdx = 0
        for (let j = 1; j < k; j++) {
          if (topValues[j] < topValues[minIdx]) minIdx = j
        }
        if (v > topValues[minIdx]) {
          topIndices[minIdx] = i
          topValues[minIdx] = v
        }
      }
    }

    // Full-vocab softmax — use the max over ALL logits for numerical
    // stability, and sum exp(logit - max) across the entire vocabulary so
    // that entries[i].prob is the true probability p_i = exp(l_i) / Σ exp(l_j).
    // (An older version softmaxed only over the top-k, which inflated head
    // probabilities because the tail mass was dropped.)
    let maxLogit = -Infinity
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i]
    }
    let sumExp = 0
    for (let i = 0; i < logits.length; i++) {
      sumExp += Math.exp(logits[i] - maxLogit)
    }

    const entries: TopKEntry[] = topIndices.map((id, i) => ({
      token: tokenizer.decode([id]),
      id,
      prob: Math.exp(topValues[i] - maxLogit) / sumExp,
    }))
    entries.sort((a, b) => b.prob - a.prob)
    return { entries, logits }
  }

  const STOP = new Set([2, 32000, 32007])

  // Interrupt mechanism — engine.interrupt() flips this to true. The decode
  // loop checks it after each token and exits cleanly with the partial
  // string. Reset to false on every generate() entry.
  let interruptRequested = false
  let generationActive = false

  function interrupt(): boolean {
    const wasActive = generationActive
    if (wasActive) interruptRequested = true
    return wasActive
  }

  async function generate(
    prompt: string,
    maxTokens: number,
    callbacks: InferenceCallbacks,
    ablations?: Ablation[]
  ): Promise<string> {
    // Reset the interrupt latch and mark a run as active so engine.interrupt()
    // can request a clean stop at the next token boundary.
    interruptRequested = false
    generationActive = true

    // Build a per-layer lookup once: layer -> Set<head> (empty set = ablate all heads)
    const ablationByLayer = new Map<number, Set<number> | 'all'>()
    for (const a of ablations ?? []) {
      const cur = ablationByLayer.get(a.layer)
      if (a.head === undefined) {
        ablationByLayer.set(a.layer, 'all')
      } else if (cur !== 'all') {
        const s = cur instanceof Set ? cur : new Set<number>()
        s.add(a.head)
        ablationByLayer.set(a.layer, s)
      }
    }
    const messages = [
      { role: 'system' as const, content: 'You are a helpful assistant.' },
      { role: 'user' as const, content: prompt },
    ]
    const promptIds = buildChatPrompt(messages, tokenizer)

    callbacks.onPrefill?.('start', promptIds.length)

    // Prefill: no per-layer yield (fast), but notify per token for animation
    for (let i = 0; i < promptIds.length; i++) {
      await decodeToken(
        promptIds[i], i, undefined, false,
        false, undefined, undefined, false, undefined,
        ablationByLayer,
      )
      const tokenText = tokenizer.decode([promptIds[i]])
      await callbacks.onPrefillToken?.(i, promptIds.length, tokenText)
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
      // External interrupt — caller wants to stop NOW (e.g. user typed a new
      // prompt while we were generating). Bail with whatever we've decoded.
      if (interruptRequested) break

      allIds.push(tokenId)
      const fullText = tokenizer.decode(allIds)
      const prevText = allIds.length > 1 ? tokenizer.decode(allIds.slice(0, -1)) : ''
      const delta = fullText.slice(prevText.length)

      // Read top-k probabilities from logits + full logits for LM-head viz
      let topK: TopKEntry[] | undefined
      let logits: Float32Array | undefined
      try {
        const r = await readTopK(5)
        topK = r.entries
        logits = r.logits
      } catch { /* readback may fail on some GPUs — non-critical */ }

      callbacks.onToken?.(delta, tokenId, i, topK, logits)

      // KV cache info
      const nnzPages = Math.floor(pos / PHI3.PAGE_SIZE) + 1
      callbacks.onKVCache?.(pos, PHI3.MAX_PAGES, nnzPages)

      pos++
      // Capture post-softmax attention scores for ALL 32 layers every token.
      tokenId = await decodeToken(
        tokenId, pos, callbacks.onLayer, true,
        true, callbacks.onAllAttentionScores, callbacks.onEmbedding,
        false, callbacks.onLayerLogitLens,
        ablationByLayer,
      )
    }

    generationActive = false
    interruptRequested = false
    return tokenizer.decode(allIds)
  }

  /**
   * Numerical validation: reconstruct attention.wgsl's output using
   * attention_scores.wgsl's softmax weights and the V slice of the KV cache.
   * If the two shaders agree (online softmax ≡ explicit softmax), the
   * relative error should be at f16 precision (~1e-2 or better).
   */
  async function validateLastAttention(): Promise<ValidationResult> {
    const layer = PHI3.LAYERS - 1

    // Use dedicated staging buffers so we don't collide with the per-token
    // readback paths (readbackBuf / attnScoresReadBuf are single-mapped).

    // Read from the snapshot buffers populated each token after layer 31's
    // attention dispatch — guarantees a coherent (out, scores) pair.

    // (1) attention.wgsl's output for layer 31 (3072 f16 → f32)
    const attnOutBytes = PHI3.D * 2
    const attnOutStaging = device.createBuffer({
      size: attnOutBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    let enc0 = device.createCommandEncoder()
    enc0.copyBufferToBuffer(B.attnOutSnap, 0, attnOutStaging, 0, attnOutBytes)
    device.queue.submit([enc0.finish()])
    await attnOutStaging.mapAsync(GPUMapMode.READ)
    const attnOut = f16ToF32(new Uint16Array(attnOutStaging.getMappedRange().slice(0)))
    attnOutStaging.unmap(); attnOutStaging.destroy()

    // (2) Full attention_scores buffer snapshot (32 layers × 32 heads × 256 slots f32)
    const scoresStaging = device.createBuffer({
      size: ATTN_SCORE_TOTAL_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    let enc1 = device.createCommandEncoder()
    enc1.copyBufferToBuffer(B.attnScoresSnap, 0, scoresStaging, 0, ATTN_SCORE_TOTAL_BYTES)
    device.queue.submit([enc1.finish()])
    await scoresStaging.mapAsync(GPUMapMode.READ)
    const allScores = new Float32Array(scoresStaging.getMappedRange().slice(0))
    scoresStaging.unmap(); scoresStaging.destroy()

    // (3) Read kv_len from lengthInfo snapshot (3 i32 — first one is kv_len)
    const lenStaging = device.createBuffer({
      size: 12,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    let enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(B.lenInfoSnap, 0, lenStaging, 0, 12)
    device.queue.submit([enc.finish()])
    await lenStaging.mapAsync(GPUMapMode.READ)
    const kvLen = new Int32Array(lenStaging.getMappedRange().slice(0))[0]
    lenStaging.unmap()
    lenStaging.destroy()

    if (kvLen <= 0) {
      return { layer, kvLen, l2Error: NaN, attnNorm: NaN, maxError: NaN, relError: NaN, passed: false }
    }

    // (4) Read page table for batch 0 to map logical→physical page numbers.
    // The paged allocator may give non-contiguous physical pages.
    const indptrStaging = device.createBuffer({
      size: 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(B.pageIndptrSnap, 0, indptrStaging, 0, 8)
    device.queue.submit([enc.finish()])
    await indptrStaging.mapAsync(GPUMapMode.READ)
    const indptr = new Int32Array(indptrStaging.getMappedRange().slice(0))
    indptrStaging.unmap(); indptrStaging.destroy()
    const numPages = indptr[1] - indptr[0]

    const valuesBytes = numPages * 4
    const valuesStaging = device.createBuffer({
      size: valuesBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(B.pageValuesSnap, indptr[0] * 4, valuesStaging, 0, valuesBytes)
    device.queue.submit([enc.finish()])
    await valuesStaging.mapAsync(GPUMapMode.READ)
    const physicalPages = new Int32Array(valuesStaging.getMappedRange().slice(0))
    valuesStaging.unmap(); valuesStaging.destroy()

    // (5) Read the entire kvPages[31] buffer (we don't know which physical
    // pages are used, so just grab everything in use). Use max physical page
    // index + 1 as the upper bound.
    let maxPage = 0
    for (let i = 0; i < numPages; i++) {
      if (physicalPages[i] > maxPage) maxPage = physicalPages[i]
    }
    const pagesNeeded = maxPage + 1
    const pageBytes = 98304 * 2 // bytes per page (f16)
    const totalBytes = pagesNeeded * pageBytes
    const vStaging = device.createBuffer({
      size: totalBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(kvPages[layer], 0, vStaging, 0, totalBytes)
    device.queue.submit([enc.finish()])
    await vStaging.mapAsync(GPUMapMode.READ)
    const pagesU16 = new Uint16Array(vStaging.getMappedRange().slice(0))
    vStaging.unmap()
    vStaging.destroy()
    const pagesF32 = f16ToF32(pagesU16)

    // (6) CPU reconstruct: out[h, d] = sum_s scores[h, s] * V[h, s, d]
    // Mirrors attention.wgsl exactly: iterate logical pages, then 16 slots per
    // page, and look up the physical page via the page table.
    const recon = new Float32Array(PHI3.D)
    const layerOff = layer * ATTN_SCORE_LAYER_WORDS

    for (let h = 0; h < PHI3.HEADS; h++) {
      let slotGlobal = 0
      for (let p = 0; p < numPages; p++) {
        const physPage = physicalPages[p]
        const pageStart = p * PHI3.PAGE_SIZE
        const slotsInPage = Math.min(PHI3.PAGE_SIZE, kvLen - pageStart)
        for (let inPageSlot = 0; inPageSlot < slotsInPage; inPageSlot++) {
          const w = allScores[layerOff + h * ATTN_SCORE_MAX_SLOTS + slotGlobal]
          if (w !== 0) {
            const vBase = physPage * 98304 + 49152 + h * 1536 + inPageSlot * 96
            const outBase = h * PHI3.HEAD_DIM
            for (let d = 0; d < PHI3.HEAD_DIM; d++) {
              recon[outBase + d] += w * pagesF32[vBase + d]
            }
          }
          slotGlobal++
        }
      }
    }

    // (6) Compare reconstructed against actual attnOut
    let maxErr = 0
    let sumSqErr = 0
    let sumSqOut = 0
    for (let i = 0; i < PHI3.D; i++) {
      const e = Math.abs(recon[i] - attnOut[i])
      if (e > maxErr) maxErr = e
      sumSqErr += e * e
      sumSqOut += attnOut[i] * attnOut[i]
    }
    const l2Error = Math.sqrt(sumSqErr)
    const attnNorm = Math.sqrt(sumSqOut)
    const relError = attnNorm > 1e-12 ? l2Error / attnNorm : NaN
    // f16 attention output gives a relative error floor around 1e-3 to 5e-3.
    const passed = isFinite(relError) && relError < 1e-2

    return { layer, kvLen, l2Error, attnNorm, maxError: maxErr, relError, passed }
  }

  // ============================================================
  // HF cross-validation suite
  // ============================================================

  /** Read a hidden-state snapshot buffer (3072 f16 → 3072 f32) and return
   *  the full vector. HF reference now stores the full 3072 dims so the
   *  comparison isn't biased by whichever dims happen to land first. */
  async function readHiddenSnap(srcBuf: GPUBuffer): Promise<Float32Array> {
    const bytes = PHI3.D * 2
    const staging = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(srcBuf, 0, staging, 0, bytes)
    device.queue.submit([enc.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const all = f16ToF32(new Uint16Array(staging.getMappedRange().slice(0)))
    staging.unmap()
    staging.destroy()
    return all
  }

  function diffVectors(gpu: Float32Array | number[], hf: Float32Array | number[]): {
    l2Error: number, hfNorm: number, gpuNorm: number, cosine: number, relError: number,
    gpuHead8: number[], hfHead8: number[]
  } {
    const n = Math.min(gpu.length, hf.length)
    let sumSqErr = 0, sumSqHf = 0, sumSqGpu = 0, dot = 0
    for (let i = 0; i < n; i++) {
      const g = gpu[i], h = hf[i], d = g - h
      sumSqErr += d * d
      sumSqHf  += h * h
      sumSqGpu += g * g
      dot      += g * h
    }
    const l2Error = Math.sqrt(sumSqErr)
    const hfNorm = Math.sqrt(sumSqHf)
    const gpuNorm = Math.sqrt(sumSqGpu)
    const cosine = (sumSqHf > 0 && sumSqGpu > 0)
      ? dot / (hfNorm * gpuNorm)
      : NaN
    const relError = hfNorm > 1e-12 ? l2Error / hfNorm : NaN
    const gpuHead8: number[] = []
    const hfHead8: number[] = []
    for (let i = 0; i < Math.min(8, n); i++) {
      gpuHead8.push(gpu[i])
      hfHead8.push(hf[i])
    }
    return { l2Error, hfNorm, gpuNorm, cosine, relError, gpuHead8, hfHead8 }
  }

  /** Jensen-Shannon divergence between two probability distributions over the
   *  same support, using natural log. Both arrays must be the same length and
   *  sum to ~1.0. JSD ∈ [0, ln 2 ≈ 0.693]. Smaller = more similar. */
  function jensenShannon(p: number[], q: number[]): number {
    const n = Math.min(p.length, q.length)
    let kl_pm = 0, kl_qm = 0
    for (let i = 0; i < n; i++) {
      const pi = p[i], qi = q[i]
      const mi = 0.5 * (pi + qi)
      if (mi <= 0) continue
      if (pi > 0) kl_pm += pi * Math.log(pi / mi)
      if (qi > 0) kl_qm += qi * Math.log(qi / mi)
    }
    return 0.5 * (kl_pm + kl_qm)
  }

  /** Build a top-K probability vector aligned to a fixed id list (union of
   *  GPU top-K and HF top-K). Missing ids get prob 0. */
  function alignProbs(
    refIds: number[],
    entries: { id: number, prob: number }[],
  ): number[] {
    const map = new Map<number, number>()
    for (const e of entries) map.set(e.id, e.prob)
    return refIds.map((id) => map.get(id) ?? 0)
  }

  /** Draw N samples from temperature-scaled softmax of the provided logits
   *  using a simple inverse-CDF on a PRNG. Returns a Map<id, count>. */
  function sampleFromLogits(
    logits: Float32Array, temperature: number, numSamples: number,
    seed = 0xC0FFEE,
  ): Map<number, number> {
    const T = Math.max(temperature, 1e-6)
    let maxLogit = -Infinity
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i]
    }
    let sumExp = 0
    const exps = new Float64Array(logits.length)
    for (let i = 0; i < logits.length; i++) {
      const e = Math.exp((logits[i] - maxLogit) / T)
      exps[i] = e
      sumExp += e
    }
    // Build cumulative CDF
    const cdf = new Float64Array(logits.length)
    let acc = 0
    for (let i = 0; i < logits.length; i++) {
      acc += exps[i] / sumExp
      cdf[i] = acc
    }
    // Deterministic xorshift32 PRNG so the test is reproducible
    let state = seed | 0
    const nextU = () => {
      state ^= state << 13; state ^= state >>> 17; state ^= state << 5
      return ((state >>> 0) / 0xFFFFFFFF)
    }
    const counts = new Map<number, number>()
    for (let s = 0; s < numSamples; s++) {
      const u = nextU()
      // Binary search for smallest i with cdf[i] >= u
      let lo = 0, hi = logits.length - 1
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (cdf[mid] >= u) hi = mid
        else lo = mid + 1
      }
      counts.set(lo, (counts.get(lo) ?? 0) + 1)
    }
    return counts
  }

  /** Run a prefill on the given input ids starting at position 0, then do a
   *  teacher-forced comparison against the HF reference decode steps. Returns
   *  per-step diffs plus aggregate stats. Does NOT capture hidden states. */
  async function checkPrompt(
    prompt: string,
    hfInputIds: number[],
    hfDecodeSteps: { argmax: number, top: { id: number, prob: number }[] }[],
    topKForDiff: number,
  ): Promise<PromptCheck & { lastPos: number, lastGpuTok: number }> {
    const messages = [
      { role: 'system' as const, content: 'You are a helpful assistant.' },
      { role: 'user' as const, content: prompt },
    ]
    const gpuInputIds = buildChatPrompt(messages, tokenizer)
    const tokenizerAgrees = gpuInputIds.length === hfInputIds.length &&
      gpuInputIds.every((id, i) => id === hfInputIds[i])

    // Prefill: start at position 0, which overwrites any prior KV slots.
    for (let i = 0; i < gpuInputIds.length - 1; i++) {
      await decodeToken(gpuInputIds[i], i, undefined, false)
    }
    const firstGpuId = await decodeToken(
      gpuInputIds[gpuInputIds.length - 1],
      gpuInputIds.length - 1,
      undefined, false,
    )

    const tokenDiffs: TokenDiff[] = []

    // Step 0: GPU argmax already in firstGpuId; read full top-K for diff.
    {
      const { entries: gpuTopK } = await readTopK(topKForDiff)
      const hfStep = hfDecodeSteps[0]
      const ids = Array.from(new Set([
        ...gpuTopK.map((e) => e.id),
        ...hfStep.top.map((e) => e.id),
      ]))
      const gpuP = alignProbs(ids, gpuTopK)
      const hfP = alignProbs(ids, hfStep.top)
      const top5G = new Set(gpuTopK.slice(0, 5).map((e) => e.id))
      const top5H = hfStep.top.slice(0, 5).map((e) => e.id)
      let overlap = 0
      for (const id of top5H) if (top5G.has(id)) overlap++
      tokenDiffs.push({
        step: 0,
        gpuId: firstGpuId,
        hfId: hfStep.argmax,
        match: firstGpuId === hfStep.argmax,
        jsd: jensenShannon(gpuP, hfP),
        top5Overlap: overlap,
      })
    }

    // Teacher-forced: feed HF's previous argmax, compare the GPU's next step.
    let pos = gpuInputIds.length
    let lastGpuTok = firstGpuId
    for (let step = 1; step < hfDecodeSteps.length; step++) {
      const feedTok = hfDecodeSteps[step - 1].argmax
      const gpuTok = await decodeToken(feedTok, pos, undefined, false)
      const { entries: gpuTopK } = await readTopK(topKForDiff)
      const hfStep = hfDecodeSteps[step]
      const ids = Array.from(new Set([
        ...gpuTopK.map((e) => e.id),
        ...hfStep.top.map((e) => e.id),
      ]))
      const gpuP = alignProbs(ids, gpuTopK)
      const hfP = alignProbs(ids, hfStep.top)
      const top5G = new Set(gpuTopK.slice(0, 5).map((e) => e.id))
      const top5H = hfStep.top.slice(0, 5).map((e) => e.id)
      let overlap = 0
      for (const id of top5H) if (top5G.has(id)) overlap++
      tokenDiffs.push({
        step,
        gpuId: gpuTok,
        hfId: hfStep.argmax,
        match: gpuTok === hfStep.argmax,
        jsd: jensenShannon(gpuP, hfP),
        top5Overlap: overlap,
      })
      lastGpuTok = gpuTok
      pos++
    }

    const topMatches = tokenDiffs.filter((t) => t.match).length
    const meanJsd = tokenDiffs.length > 0
      ? tokenDiffs.reduce((s, t) => s + t.jsd, 0) / tokenDiffs.length
      : NaN

    return {
      prompt,
      tokenizerAgrees,
      gpuInputIds,
      hfInputIds,
      tokenDiffs,
      topMatches,
      meanJsd,
      lastPos: pos,
      lastGpuTok,
    }
  }

  async function runValidationSuite(): Promise<ValidationReport> {
    const attnStub: ValidationResult = {
      layer: -1, kvLen: 0, l2Error: NaN, attnNorm: NaN,
      maxError: NaN, relError: NaN, passed: false,
    }
    const samplingStub: SamplingSelfTest = {
      numSamples: 0, temperature: 1, uniqueIds: 0,
      empiricalJsd: NaN, maxL1Error: NaN, passed: false,
    }
    const mainStub = {
      prompt: '', tokenizerAgrees: false,
      gpuInputIds: [] as number[], hfInputIds: [] as number[],
      layerDiffs: [] as LayerHiddenDiff[], tokenDiffs: [] as TokenDiff[],
      topMatches: 0, meanJsd: NaN,
    }
    const empty: ValidationReport = {
      hasReference: false,
      main: mainStub,
      sweep: [],
      longContext: { ...mainStub, tokenDiffs: [], kvLen: 0 },
      attentionEquivalence: attnStub,
      samplingSelfTest: samplingStub,
      summary: 'no reference.json',
    }

    // ---- Load reference -----------------------------------------------------
    interface RefPrompt {
      prompt: string
      promptTokens: number
      inputIds: number[]
      decodeSteps: { argmax: number, top: { id: number, token: string, prob: number }[] }[]
      hiddenStates?: Record<string, number[]>
      layersCaptured?: number[]
    }
    interface RefFile {
      main: RefPrompt
      sweep: RefPrompt[]
      longContext: RefPrompt
    }
    let ref: RefFile
    try {
      const r = await fetch('/reference.json')
      const ct = r.headers.get('content-type') ?? ''
      if (!r.ok || !ct.includes('json')) return empty
      ref = await r.json()
      if (!ref.main || !ref.sweep || !ref.longContext) return empty
    } catch {
      return empty
    }

    // ---- [MAIN] Main prompt with full hidden-state capture ------------------
    const mainMessages = [
      { role: 'system' as const, content: 'You are a helpful assistant.' },
      { role: 'user' as const, content: ref.main.prompt },
    ]
    const mainGpuInputIds = buildChatPrompt(mainMessages, tokenizer)
    const mainTokenizerAgrees =
      mainGpuInputIds.length === ref.main.inputIds.length &&
      mainGpuInputIds.every((id, i) => id === ref.main.inputIds[i])

    // Prefill at pos 0..N-1; capture hidden states on the LAST prompt token.
    for (let i = 0; i < mainGpuInputIds.length - 1; i++) {
      await decodeToken(mainGpuInputIds[i], i, undefined, false)
    }
    const mainFirstGpuId = await decodeToken(
      mainGpuInputIds[mainGpuInputIds.length - 1],
      mainGpuInputIds.length - 1,
      undefined, false, false, undefined, undefined, /*captureHidden=*/true,
    )

    // Per-layer hidden-state diffs (full 3072 dims).
    const layerDiffs: LayerHiddenDiff[] = []
    if (ref.main.hiddenStates?.['embedding']) {
      const gpu = await readHiddenSnap(B.embeddingSnap)
      const hf = ref.main.hiddenStates['embedding']
      layerDiffs.push({ layer: -1, ...diffVectors(gpu, hf) })
    }
    for (let i = 0; i < VALIDATE_LAYERS.length; i++) {
      const L = VALIDATE_LAYERS[i]
      const hf = ref.main.hiddenStates?.[String(L)]
      if (!hf) continue
      const gpu = await readHiddenSnap(B.hiddenSnaps[i])
      layerDiffs.push({ layer: L, ...diffVectors(gpu, hf) })
    }

    // Main step 0 logit diff against HF.
    const mainTokenDiffs: TokenDiff[] = []
    {
      const { entries: gpuTop20 } = await readTopK(20)
      const hfStep = ref.main.decodeSteps[0]
      const ids = Array.from(new Set([
        ...gpuTop20.map((e) => e.id),
        ...hfStep.top.map((e) => e.id),
      ]))
      const gpuP = alignProbs(ids, gpuTop20)
      const hfP = alignProbs(ids, hfStep.top)
      const top5G = new Set(gpuTop20.slice(0, 5).map((e) => e.id))
      const top5H = hfStep.top.slice(0, 5).map((e) => e.id)
      let overlap = 0
      for (const id of top5H) if (top5G.has(id)) overlap++
      mainTokenDiffs.push({
        step: 0,
        gpuId: mainFirstGpuId,
        hfId: hfStep.argmax,
        match: mainFirstGpuId === hfStep.argmax,
        jsd: jensenShannon(gpuP, hfP),
        top5Overlap: overlap,
      })
    }

    // Main teacher-forced decode (steps 1..19).
    let mainPos = mainGpuInputIds.length
    let mainLastGpuTok = mainFirstGpuId
    for (let step = 1; step < ref.main.decodeSteps.length; step++) {
      const feedTok = ref.main.decodeSteps[step - 1].argmax
      const gpuTok = await decodeToken(feedTok, mainPos, undefined, false)
      const { entries: gpuTopK } = await readTopK(20)
      const hfStep = ref.main.decodeSteps[step]
      const ids = Array.from(new Set([
        ...gpuTopK.map((e) => e.id),
        ...hfStep.top.map((e) => e.id),
      ]))
      const gpuP = alignProbs(ids, gpuTopK)
      const hfP = alignProbs(ids, hfStep.top)
      const top5G = new Set(gpuTopK.slice(0, 5).map((e) => e.id))
      const top5H = hfStep.top.slice(0, 5).map((e) => e.id)
      let overlap = 0
      for (const id of top5H) if (top5G.has(id)) overlap++
      mainTokenDiffs.push({
        step,
        gpuId: gpuTok,
        hfId: hfStep.argmax,
        match: gpuTok === hfStep.argmax,
        jsd: jensenShannon(gpuP, hfP),
        top5Overlap: overlap,
      })
      mainLastGpuTok = gpuTok
      mainPos++
    }

    const mainTopMatches = mainTokenDiffs.filter((t) => t.match).length
    const mainMeanJsd = mainTokenDiffs.reduce((s, t) => s + t.jsd, 0) / mainTokenDiffs.length

    // ---- [ATTENTION] Layer-31 online vs explicit softmax --------------------
    // Run one extra decode with captureAllScores=true so the snapshot buffers
    // are primed with live layer-31 state from THIS token; then compare.
    await decodeToken(
      mainLastGpuTok, mainPos, undefined, true, true, undefined, undefined,
    )
    const attentionEquivalence = await validateLastAttention()

    // ---- [SAMPLING] Empirical sampling distribution vs softmax --------------
    // Read the raw logits from the last decode we ran (attention snapshot pass)
    // and draw 5000 samples at temperature=1. Compare empirical frequency
    // distribution over the top-20 ids against the true softmax.
    const samplingSelfTest = await (async (): Promise<SamplingSelfTest> => {
      const { entries, logits } = await readTopK(20)
      const NUM_SAMPLES = 5000
      const T = 1.0
      const counts = sampleFromLogits(logits, T, NUM_SAMPLES)
      // Theoretical probs for the top-20 (already in entries).
      const ids = entries.map((e) => e.id)
      const theoretical = entries.map((e) => e.prob)
      const empirical = ids.map((id) => (counts.get(id) ?? 0) / NUM_SAMPLES)
      // Top-20 covers most mass; the rest is lumped as an "other" bin.
      const top20Theoretical = theoretical.reduce((s, p) => s + p, 0)
      const top20Empirical = empirical.reduce((s, p) => s + p, 0)
      const pVec = [...theoretical, 1 - top20Theoretical]
      const qVec = [...empirical, 1 - top20Empirical]
      const empiricalJsd = jensenShannon(pVec, qVec)
      let maxL1 = 0
      for (let i = 0; i < ids.length; i++) {
        const e = Math.abs(empirical[i] - theoretical[i])
        if (e > maxL1) maxL1 = e
      }
      const passed = empiricalJsd < 1e-2 && maxL1 < 0.02
      return {
        numSamples: NUM_SAMPLES,
        temperature: T,
        uniqueIds: counts.size,
        empiricalJsd,
        maxL1Error: maxL1,
        passed,
      }
    })()

    // ---- [SWEEP] 15 prompts — tokenizer + 5-step teacher-forced logits -----
    const sweep: PromptCheck[] = []
    for (const p of ref.sweep) {
      const c = await checkPrompt(p.prompt, p.inputIds, p.decodeSteps, 10)
      sweep.push({
        prompt: c.prompt,
        tokenizerAgrees: c.tokenizerAgrees,
        gpuInputIds: c.gpuInputIds,
        hfInputIds: c.hfInputIds,
        tokenDiffs: c.tokenDiffs,
        topMatches: c.topMatches,
        meanJsd: c.meanJsd,
      })
    }

    // ---- [LONG CONTEXT] ~290 tokens, 10 decode steps ------------------------
    const lc = await checkPrompt(
      ref.longContext.prompt,
      ref.longContext.inputIds,
      ref.longContext.decodeSteps,
      10,
    )
    const longContext = {
      prompt: lc.prompt,
      tokenizerAgrees: lc.tokenizerAgrees,
      gpuInputIds: lc.gpuInputIds,
      hfInputIds: lc.hfInputIds,
      tokenDiffs: lc.tokenDiffs,
      topMatches: lc.topMatches,
      meanJsd: lc.meanJsd,
      kvLen: lc.lastPos,
    }

    // ---- Summary ------------------------------------------------------------
    const sweepMatches = sweep.reduce((s, p) => s + p.topMatches, 0)
    const sweepTotal = sweep.reduce((s, p) => s + p.tokenDiffs.length, 0)
    const sweepTokenizerOk = sweep.every((p) => p.tokenizerAgrees)
    const sweepMeanJsd = sweep.length > 0
      ? sweep.reduce((s, p) => s + p.meanJsd, 0) / sweep.length
      : NaN
    const summary =
      `main: tok=${mainTokenizerAgrees ? 'OK' : 'BAD'} ${mainTopMatches}/${mainTokenDiffs.length} JSD=${mainMeanJsd.toExponential(2)}  ` +
      `sweep: tok=${sweepTokenizerOk ? 'OK' : 'BAD'} ${sweepMatches}/${sweepTotal} JSD=${sweepMeanJsd.toExponential(2)}  ` +
      `long: tok=${longContext.tokenizerAgrees ? 'OK' : 'BAD'} ${longContext.topMatches}/${longContext.tokenDiffs.length} kv=${longContext.kvLen}  ` +
      `attn=${(attentionEquivalence.relError * 100).toFixed(4)}%  ` +
      `sampling: JSD=${samplingSelfTest.empiricalJsd.toExponential(2)} ${samplingSelfTest.passed ? 'PASS' : 'FAIL'}`

    return {
      hasReference: true,
      main: {
        prompt: ref.main.prompt,
        tokenizerAgrees: mainTokenizerAgrees,
        gpuInputIds: mainGpuInputIds,
        hfInputIds: ref.main.inputIds,
        layerDiffs,
        tokenDiffs: mainTokenDiffs,
        topMatches: mainTopMatches,
        meanJsd: mainMeanJsd,
      },
      sweep,
      longContext,
      attentionEquivalence,
      samplingSelfTest,
      summary,
    }
  }

  /** Dispatch the embedding kernel for a single token id in isolation and
   *  read back the first 32 dims of the result via a private staging buffer
   *  (so it doesn't race with the shared readbackBuf). */
  async function debugEmbedToken(tokenId: number): Promise<Float32Array> {
    device.queue.writeBuffer(B.inputIds, 0, new Int32Array([tokenId]))
    const enc = device.createCommandEncoder()
    dispatch(enc, P.embedding, bg(device, P.embedding, [
      B.residual, B.inputIds, weights.embdScales, weights.embdWeights, embU,
    ]), 12)
    const bytes = PHI3.D * 2
    const staging = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    enc.copyBufferToBuffer(B.residual, 0, staging, 0, bytes)
    device.queue.submit([enc.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const f16 = new Uint16Array(staging.getMappedRange().slice(0))
    staging.unmap()
    staging.destroy()
    return f16ToF32(f16)  // full 3072 dims
  }

  report('Engine ready!', { phase: 'done', percent: 100 })

  return {
    generate, interrupt, validateLastAttention, runValidationSuite, tokenizer,
    debugEmbedToken,
    ready: true,
  }
}

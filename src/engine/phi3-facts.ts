/**
 * PHI3_FACTS — Single source of truth for user-facing claims about
 * Phi-3-mini and the neuropulse forward pass.
 *
 * Every prose claim ("32 layers", "292 dispatches", "11 kernels", ~2 GB)
 * MUST resolve to a value here. The verify-claims.mjs CI script greps
 * READMEs and HTML for those numbers and asserts they match these constants.
 *
 * If you change a number, run `npm run verify` to find every place that
 * needs updating. Do not edit copy without updating this file first.
 */

import { PHI3 } from './compiler'

/** Architecture re-export so docs scripts only need to import from one place. */
export const ARCH = {
  /** Transformer layers (decoder blocks). */
  layers: PHI3.LAYERS,
  /** Multi-head attention heads per layer. Phi-3-mini is non-GQA. */
  heads: PHI3.HEADS,
  /** Per-head dimension. heads · headDim = hiddenDim. */
  headDim: PHI3.HEAD_DIM,
  /** Residual / hidden dimension. */
  hiddenDim: PHI3.D,
  /** SwiGLU FFN inner dimension. */
  ffnDim: PHI3.FFN,
  /** Vocabulary size of the BPE tokenizer. */
  vocab: PHI3.VOCAB,
  /** Max KV cache pages × page size = 4112 ≈ 4096 token context. */
  maxContextTokens: PHI3.PAGE_SIZE * PHI3.MAX_PAGES,
} as const

/** GPU dispatch budget per generated token. The 9 step names below are
 *  in `STEP_NAMES` in inference.ts; counts here are derived to keep them
 *  in sync. */
const LAYER_STEPS_PER_DISPATCH = 9 // QKV·RoPE·KVAppend·Attn·O·AddNorm·FFNGateUp·FFNDown·AddNorm
const PROLOGUE_DISPATCHES = 1      // embedding lookup
const EPILOGUE_DISPATCHES = 3      // final rmsNorm + lm_head + argmax
const ATTENTION_SCORES_PER_LAYER = 1 // captureAllScores flag, one per layer in viz mode
const LENS_LAYERS_COUNT = 8          // [0,4,8,12,16,20,24,28] in inference.ts
const LENS_DISPATCHES_PER_LAYER = 3  // rmsNorm + lm_head + argmax

export const DISPATCHES = {
  /** Dispatch count for one decoded token in the headless / fast path
   *  (no per-step yield, no visualizer instrumentation). This is the
   *  number to quote in performance comparisons against PyTorch / HF. */
  fast:
    LAYER_STEPS_PER_DISPATCH * ARCH.layers + PROLOGUE_DISPATCHES + EPILOGUE_DISPATCHES,
  /** Dispatch count when the visualizer is active — adds per-layer
   *  attention-scores readback and logit-lens probes. This is what the
   *  user actually sees on screen. Quote this in UI copy. */
  visualized:
    LAYER_STEPS_PER_DISPATCH * ARCH.layers + PROLOGUE_DISPATCHES + EPILOGUE_DISPATCHES +
    ATTENTION_SCORES_PER_LAYER * ARCH.layers +
    LENS_LAYERS_COUNT * LENS_DISPATCHES_PER_LAYER,
  /** Distinct step names in the per-layer pipeline (matches STEP_NAMES). */
  layerSteps: LAYER_STEPS_PER_DISPATCH,
} as const

/** Hand-written WGSL kernels in src/engine/shaders/. Update this list
 *  when adding/removing a .wgsl file; verify-claims.mjs cross-checks
 *  the directory contents. */
export const KERNELS = [
  'add_norm',
  'argmax',
  'attention',
  'attention_scores',
  'embedding',
  'fused_ffn',
  'int4_matmul',
  'int4_matmul_f32',
  'kv_append',
  'rms_norm',
  'rope',
] as const

export const STORAGE = {
  /** Approximate quantized weight payload streamed on first visit. */
  weightSizeGB: 2.0,
  /** Approximate GPU memory footprint of allocated weight buffers. */
  gpuBytesGB: 2.0,
  /** Recommended VRAM headroom. */
  recommendedVramGB: 4.0,
  /** Cache API bucket name. */
  cacheName: 'neuropulse-phi3-weights',
  /** Legacy cache bucket from the pre-rename era — still scanned. */
  legacyCacheName: 'neural-pulse-phi3-weights',
  /** OPFS directory name. */
  opfsDir: 'neuropulse-weights',
} as const

export const MODEL = {
  /** Human-readable model name. */
  name: 'Phi-3-mini-4k-instruct',
  /** Quantization scheme of the weights we load. */
  quant: 'q4f16_1',
  /** Distribution format. NOT GGUF. */
  format: 'MLC ndarray-cache',
  /** HF repository slug. */
  hfRepo: 'mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC',
} as const

export const VALIDATION = {
  /** Layers checkpointed for HF cross-validation. Mirror this list in
   *  inference.ts VALIDATE_LAYERS — verify-claims runs that check too. */
  validateLayers: [0, 4, 8, 12, 16, 20, 24, 28, 31] as const,
  /** Per-token sampling sanity check. Compare an empirical histogram of
   *  N samples against the softmax distribution; assert JSD < tolerance. */
  samplerSamples: 5000,
  samplerJSDTolerance: 1e-2,
} as const

/** All facts in one bag for the verify-claims script. */
export const PHI3_FACTS = {
  ARCH,
  DISPATCHES,
  KERNELS,
  STORAGE,
  MODEL,
  VALIDATION,
} as const

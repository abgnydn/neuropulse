/**
 * RECORDING — schema + encode helpers for captured forward-pass runs.
 *
 * A recording is a compact, honest capture of a REAL run's visualization
 * stream: everything the panels and 3D scene actually render, nothing they
 * don't. Full logits, embeddings, raw activations, and non-L31 attention are
 * deliberately excluded (never rendered / expert-only), which is what keeps a
 * 30–40-token run around ~150 KB raw (~40 KB over the wire) instead of
 * gigabytes.
 *
 * Capture: open the app with `?record=1`, run a prompt on the LIVE engine,
 * then use the "Download recording JSON" strip that appears when the run
 * settles. Commit curated files under public/recordings/. CI validates every
 * committed recording via tools/verify-recordings.mjs (schema twin — keep in
 * sync, bump schemaVersion on breaking changes).
 *
 * Playback: src/playback.ts drives the UI from one of these files in demo
 * mode — real tensors, captured live earlier, replayed.
 */

import type { TopKEntry } from './engine/inference'

export const RECORDING_SCHEMA_VERSION = 1 as const

export interface RecordedToken {
  /** decoded text delta for this token */
  text: string
  id: number
  /** top-5 candidates {t: token text, id, p: softmax prob (4dp)} */
  topK: { t: string; id: number; p: number }[]
  /** L2 norm of the residual at each of the 32 layers (3 sig figs) */
  residualNorms: number[]
  /** ||residual(L) − residual(L−1)|| per layer (3 sig figs) */
  layerDeltas: number[]
  /** 32×32 head-activity grid, base64 u8, max-scaled by headActivityScale */
  headActivity: string
  headActivityScale: number
  /** logit-lens picks fired during this token: layers ⊆ {0,4,…,28} */
  lens: { L: number; t: string; id: number }[]
  /** layer-31 attention rows (32 heads × kvLen), base64 u8 row-max scaled */
  attnL31: string
  attnL31Scale: number
  kvLen: number
  kvUsedPages: number
}

export interface NpRecording {
  schemaVersion: typeof RECORDING_SCHEMA_VERSION
  capturedAt: string
  /** runtime fingerprint of the capturing machine — provenance, not vanity */
  engineFingerprint: string
  prompt: string
  mode: 'ask' | 'complete'
  /** prompt tokens in prefill order (drives the "Reading prompt…" phase) */
  prefillTokens: string[]
  kvTotalPages: number
  tokens: RecordedToken[]
}

// ─── u8 quantization helpers ───────────────────────────────────────────────

/** Quantize a float array to u8 against its own max; returns base64 + scale.
 *  Dequantize as: value[i] = u8[i] / 255 * scale. */
export function quantizeU8(values: ArrayLike<number>): { b64: string; scale: number } {
  let max = 0
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!
    if (Number.isFinite(v) && v > max) max = v
  }
  const bytes = new Uint8Array(values.length)
  if (max > 0) {
    for (let i = 0; i < values.length; i++) {
      const v = values[i]!
      bytes[i] = Number.isFinite(v) && v > 0 ? Math.min(255, Math.round((v / max) * 255)) : 0
    }
  }
  let bin = ''
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]!)
  return { b64: btoa(bin), scale: max }
}

/** Inverse of quantizeU8 → Float32Array of length `expectedLen` (throws on
 *  length mismatch so corrupt recordings fail loudly, not visually). */
export function dequantizeU8(b64: string, scale: number, expectedLen: number): Float32Array {
  const bin = atob(b64)
  if (bin.length !== expectedLen) {
    throw new Error(`recording payload length ${bin.length} ≠ expected ${expectedLen}`)
  }
  const out = new Float32Array(expectedLen)
  const s = scale / 255
  for (let i = 0; i < expectedLen; i++) out[i] = bin.charCodeAt(i) * s
  return out
}

// ─── capture-side rounding helpers ─────────────────────────────────────────

export const round3 = (v: number): number => (Number.isFinite(v) ? Number(v.toPrecision(3)) : 0)
export const round4dp = (v: number): number => (Number.isFinite(v) ? Math.round(v * 10000) / 10000 : 0)

export function compactTopK(topK: TopKEntry[] | undefined): { t: string; id: number; p: number }[] {
  return (topK ?? []).slice(0, 5).map((e) => ({ t: e.token, id: e.id, p: round4dp(e.prob) }))
}

// ─── Capture-side recorder ─────────────────────────────────────────────────

import type { InferenceCallbacks } from './engine/inference'

const LAYERS = 32
const HEADS = 32
const ATTN_MAX_SLOTS = 256
const L31_OFFSET = 31 * HEADS * ATTN_MAX_SLOTS

export interface RecorderSources {
  /** live module state read at each onToken (same arrays the panels render) */
  getResidualNorms(): ArrayLike<number>
  getLayerDeltas(): ArrayLike<number>
  getHeadHeatmap(): ArrayLike<number>[] // [32][32]
  getFingerprint(): string
}

export interface Recorder {
  /** merge into the generate() callbacks (after base + storyteller) */
  hooks(): Partial<InferenceCallbacks>
  /** finish and build the recording; null if nothing was captured */
  build(prompt: string, mode: 'ask' | 'complete'): NpRecording | null
  tokenCount(): number
  reset(): void
}

/** Taps the live callback stream and accumulates a compact NpRecording.
 *  Engine ordering: lens / attention / KV events for token N fire BEFORE
 *  onToken(N), so those are buffered and flushed when the token lands. */
export function createRecorder(src: RecorderSources): Recorder {
  let prefillTokens: string[] = []
  let tokens: RecordedToken[] = []
  let pendingLens: { L: number; t: string; id: number }[] = []
  let pendingAttn: { b64: string; scale: number; kvLen: number } | null = null
  let pendingKv = { pos: 0, total: 0, used: 0 }

  return {
    hooks(): Partial<InferenceCallbacks> {
      return {
        onPrefill(phase) {
          if (phase === 'start') { prefillTokens = []; tokens = []; pendingLens = []; pendingAttn = null }
        },
        async onPrefillToken(_i, _total, text) { prefillTokens.push(text) },
        onLayerLogitLens(layer, tokenId, token) {
          pendingLens.push({ L: layer, t: token, id: tokenId })
        },
        onAllAttentionScores(scores, kvLen) {
          // Slice layer 31 (the only layer ever rendered) into 32 rows × kvLen.
          const rows = new Float32Array(HEADS * kvLen)
          for (let h = 0; h < HEADS; h++) {
            for (let s = 0; s < kvLen && s < ATTN_MAX_SLOTS; s++) {
              rows[h * kvLen + s] = scores[L31_OFFSET + h * ATTN_MAX_SLOTS + s]!
            }
          }
          const q = quantizeU8(rows)
          pendingAttn = { b64: q.b64, scale: q.scale, kvLen }
        },
        onKVCache(position, totalPages, usedPages) {
          pendingKv = { pos: position, total: totalPages, used: usedPages }
        },
        onToken(text, id, _index, topK) {
          const heat = src.getHeadHeatmap()
          const flat = new Float32Array(LAYERS * HEADS)
          for (let L = 0; L < LAYERS && L < heat.length; L++) {
            const row = heat[L]!
            for (let h = 0; h < HEADS; h++) flat[L * HEADS + h] = Number(row[h]) || 0
          }
          const heatQ = quantizeU8(flat)
          tokens.push({
            text, id,
            topK: compactTopK(topK),
            residualNorms: Array.from(src.getResidualNorms(), round3),
            layerDeltas: Array.from(src.getLayerDeltas(), round3),
            headActivity: heatQ.b64,
            headActivityScale: round4dp(heatQ.scale),
            lens: pendingLens,
            attnL31: pendingAttn?.b64 ?? '',
            attnL31Scale: round4dp(pendingAttn?.scale ?? 0),
            kvLen: pendingAttn?.kvLen ?? pendingKv.pos,
            kvUsedPages: pendingKv.used,
          })
          pendingLens = []
          pendingAttn = null
        },
      }
    },
    build(prompt, mode): NpRecording | null {
      if (tokens.length === 0) return null
      return {
        schemaVersion: RECORDING_SCHEMA_VERSION,
        capturedAt: new Date().toISOString(),
        engineFingerprint: src.getFingerprint(),
        prompt, mode,
        prefillTokens: [...prefillTokens],
        kvTotalPages: pendingKv.total,
        tokens: [...tokens],
      }
    },
    tokenCount(): number { return tokens.length },
    reset(): void { prefillTokens = []; tokens = []; pendingLens = []; pendingAttn = null },
  }
}

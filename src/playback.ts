/**
 * PLAYBACK — drives the UI from a recorded run (src/recording.ts) in demo
 * mode. Real tensors, captured live earlier, replayed.
 *
 * Architecture: the driver knows the recording format + pacing; the SINK is
 * implemented by main.ts from its existing engine-free update functions
 * (updateTopK, updateHeatmapLayer, viz.* …). No fake engine callbacks, no
 * synthesized raw activations — the recording stores exactly what the panels
 * render, and the sink feeds exactly that.
 *
 * Pacing mirrors the live engine's speed-slider math (main.ts onLayer): the
 * per-token gap is 1000/speed ms at the last layer, slow speeds add per-layer
 * delays, and 20× runs at a small fixed per-layer cost roughly matching real
 * GPU decode rates.
 */

import type { NpRecording, RecordedToken } from './recording'
import { dequantizeU8 } from './recording'

const LAYERS = 32
const HEADS = 32

export interface PlaybackSink {
  onPrefillStart(total: number): void
  onPrefillToken(index: number, total: number, text: string): void
  onPrefillEnd(): void
  /** Per-layer sweep during a token: head activity row + residual norm. */
  onLayerPulse(layer: number, attnHeads: Float32Array, residualNorm: number): void
  onLens(layer: number, text: string, id: number): void
  onToken(tok: RecordedToken, index: number): void
  /** Layer-31 attention rows, 32 heads × kvLen (head-major). */
  onAttentionL31(rows: Float32Array, kvLen: number): void
  onKV(position: number, totalPages: number, usedPages: number): void
  onDone(interrupted: boolean): void
}

export interface PlaybackHandle {
  start(): Promise<void>
  stop(): void
  isPlaying(): boolean
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

export function createPlaybackDriver(
  rec: NpRecording,
  sink: PlaybackSink,
  getSpeed: () => number,
  opts?: { /** resume mid-run: first token index to play; skips prefill when > 0 */ startAt?: number },
): PlaybackHandle {
  let aborted = false
  let playing = false
  const startAt = Math.max(0, Math.min(opts?.startAt ?? 0, rec.tokens.length))

  async function start(): Promise<void> {
    if (playing) return
    playing = true
    aborted = false
    try {
      // ── Prefill phase: replay the prompt being read (fresh starts only —
      // a resume picks up mid-decode like a video scrubbed forward) ──
      if (startAt === 0) {
        const pf = rec.prefillTokens
        sink.onPrefillStart(pf.length)
        for (let i = 0; i < pf.length; i++) {
          if (aborted) return
          sink.onPrefillToken(i, pf.length, pf[i]!)
          // Brisk fixed pace — prefill is context, not the show.
          await sleep(Math.max(20, 260 / Math.max(1, getSpeed())))
        }
        sink.onPrefillEnd()
      }

      // ── Decode phase: per token, sweep the 32 layers then land the token ──
      for (let i = startAt; i < rec.tokens.length; i++) {
        if (aborted) return
        const tok = rec.tokens[i]!
        const heat = dequantizeU8(tok.headActivity, tok.headActivityScale, LAYERS * HEADS)
        // lens picks keyed by layer for interleaving at the right sweep point
        const lensAt = new Map<number, { t: string; id: number }>()
        for (const l of tok.lens) lensAt.set(l.L, { t: l.t, id: l.id })

        for (let L = 0; L < LAYERS; L++) {
          if (aborted) return
          sink.onLayerPulse(L, heat.subarray(L * HEADS, (L + 1) * HEADS), tok.residualNorms[L] ?? 0)
          const lens = lensAt.get(L)
          if (lens) sink.onLens(L, lens.t, lens.id)

          // Speed-slider pacing, mirroring the live engine's onLayer math.
          const speed = getSpeed()
          if (L === LAYERS - 1 && speed < 20) {
            await sleep(Math.round(1000 / speed))
          } else if (speed <= 3) {
            await sleep((4 - speed) * 4)
          } else {
            await sleep(2) // keeps 20× at realistic decode pace, not instant
          }
        }

        if (tok.attnL31) {
          sink.onAttentionL31(dequantizeU8(tok.attnL31, tok.attnL31Scale, HEADS * tok.kvLen), tok.kvLen)
        }
        sink.onKV(tok.kvLen, rec.kvTotalPages, tok.kvUsedPages)
        sink.onToken(tok, i)
      }
    } finally {
      const wasAborted = aborted
      playing = false
      sink.onDone(wasAborted)
    }
  }

  return {
    start,
    stop(): void { aborted = true },
    isPlaying(): boolean { return playing },
  }
}

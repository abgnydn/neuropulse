# Methods

This document is the empirical foundation for every numerical claim
neuropulse makes. If a number appears in the README, the landing page,
or the in-app glossary, it was either derived from this file or
verified against `tools/verify-claims.mjs` and `src/engine/phi3-facts.ts`.

The goal is reproducibility. A reader should be able to: pick a prompt,
note their fingerprint, observe the same per-layer activations, and
arrive at the same generated tokens (modulo the documented tolerances
below).

## Model under test

| Field | Value |
|---|---|
| Architecture | Phi-3-mini-4k-instruct |
| Parameters | 3.8 B |
| Layers | 32 |
| Attention heads | 32 (no GQA) |
| Per-head dimension | 96 |
| Residual / hidden dim | 3,072 |
| FFN inner dim (SwiGLU) | 8,192 |
| Vocabulary | 32,064 (BPE) |
| Max context | 4,112 tokens (257 KV pages × 16 page size) |
| Quantization | `q4f16_1` — 4-bit weights, fp16 scales, group size 32, zero-point 7 |
| Source | [`mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC`](https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC) on HuggingFace |
| Distribution format | MLC ndarray-cache (NOT GGUF) |

## Precision

All weights are stored at int4, scales at fp16. Activations and accumulators
follow this matrix:

| Stage | Storage | Accumulator | Note |
|---|---|---|---|
| Embedding lookup | f16 | — | direct copy from row of embedding table |
| Q/K/V projection | f16 inputs, int4·fp16 weights | f32 | `int4_matmul.wgsl` accumulates dot products at f32 then converts to f16 on store |
| RoPE | f16 in/out | f32 | trig precomputed at f32 inside the kernel |
| Attention (softmax) | f16 scores | f32 max/sum | numerically stable subtract-max softmax in f32 |
| Output projection | int4·fp16 → f16 | f32 | same matmul kernel as QKV |
| RMSNorm | f16 in/out | f32 reduction | `eps = 1e-5`; reduction in shared memory at f32 |
| FFN gate · up · down (SwiGLU) | int4·fp16 → f16 | f32 | gate-and-up fused; SiLU at f32 inside kernel |
| LM head | int4·fp16 → f32 logits | f32 | uses `int4_matmul_f32.wgsl` to keep logits at f32 for argmax/softmax |
| Argmax | reduces f32 logits | f32 | per-tile reduction then global |

The single intentional precision tradeoff vs PyTorch FP32 reference is the
**f16 accumulation of intermediate residuals**. PyTorch reference uses f32
end-to-end. We pay an f16 round-trip at every Add+Norm boundary. This is
where the bulk of the relative-error budget below goes.

## Numerical tolerances

These are the bounds the validation suite (`accurateBtn` in the UI, or
`engine.validate*` programmatically) asserts. A run that exceeds any of
these is a **regression** — the run fails, the UI flags it.

| Test | Quantity | Bound | Rationale |
|---|---|---|---|
| Hidden-state vs HF reference | relative L2 per layer (over leading 3072 dims) | < **2e-2** | f16 accumulation rounds residual ≈ 1 part in 1024; 32 layers compound. Empirically observed max ≈ 1.4e-2. |
| Hidden-state vs HF reference | cosine similarity per layer | > **0.999** | Direction matters more than magnitude for next-token argmax. |
| Last-attention reconstruction | relative L2 of (Σ scores·V − attn_out) | < **1e-2** | Pure-f16 sanity check: are the per-layer attention scores we expose to the visualizer actually what produced the residual? |
| Top-1 token agreement | greedy-decode argmax | **100% match** on the validation prompt set up to 30 tokens | If this fails, the model is materially diverging — fail loud. |
| Top-5 overlap | per-position size of intersect(GPU.top5, HF.top5) | ≥ **4** mean | Allows occasional 5th-place reordering due to f16 ties. |
| Mean JSD over top-K | Jensen-Shannon divergence of softmax distributions | < **5e-3** | Matches typical f16-vs-f32 logit JSD reported in the MLC release notes. |
| Sampling self-test | JSD(empirical histogram of 5,000 samples, theoretical softmax) | < **1e-2** | Verifies the RNG path is unbiased; independent of model correctness. |

Validation checkpoints are at the layer indices
`[0, 4, 8, 12, 16, 20, 24, 28, 31]` (9 checkpoints × 3,072 dims =
27,648 floats compared per prompt). Source: `VALIDATE_LAYERS` in
`src/engine/inference.ts`.

## Per-kernel error budgets

Each kernel below has a static error budget. The CI runner compares the
kernel's output against a CPU f64 reference on a fixed input fixture.

| Kernel | Max relative error | Max ULP@f16 | CPU reference |
|---|---|---|---|
| `int4_matmul` | 8e-3 | 4 | numpy fp64 dequant + matmul |
| `int4_matmul_f32` | 5e-3 | — (output f32) | numpy fp64 dequant + matmul, cast f32 |
| `rms_norm` | 4e-3 | 2 | numpy fp64 |
| `add_norm` | 4e-3 | 2 | rms_norm error budget compounded with f16 add |
| `rope` | 1e-3 | 1 | math.cos / math.sin at f64 |
| `attention` (softmax + weighted sum) | 8e-3 | 4 | numpy fp64 stable softmax |
| `attention_scores` | 5e-3 | 2 | numpy fp64 dot product |
| `kv_append` | 0 (bit-exact copy) | 0 | identity |
| `fused_ffn` | 1e-2 | 5 | gate-up matmul + silu + down matmul at f64 |
| `embedding` | 0 (bit-exact copy) | 0 | identity |
| `argmax` | 0 if tied tokens absent; 1-position drift on exact tie | — | numpy argmax |

The numbers above are **declared budgets**, not measurements. Today the
runtime smoke tests check end-to-end outputs against HF only. Wiring the
per-kernel CPU references into CI is a tracked gap (see "Known gaps" below).

## What is verified vs. what is claimed

### Verified (by code that runs every commit)
- Architecture constants (layers, heads, dims, vocab) match canonical
  source (`compiler.ts` PHI3) — `tools/verify-claims.mjs`.
- Kernel count matches files in `src/engine/shaders/` — same script.
- Dispatch counts (fast = 292, visualized = 348) are derived from a
  closed-form formula in `phi3-facts.ts`, not hard-coded prose.
- Keyboard shortcuts wired in `main.ts` match the glossary +
  journey HUD — `tools/check-shortcuts.mjs`.

### Verified (by code that runs on user click)
- HF parity at `VALIDATE_LAYERS` against a pinned reference dump
  (`tools/dump_phi3_reference.py`) — exposed via the validate button.
- Last-layer attention reconstruction (sanity that the scores we
  visualize are the same scores that produced the output residual).

### Claimed but not yet automated
- Per-kernel ULP error budgets above. The bounds are documented; the
  CI runner that asserts them does not yet exist.
- Cross-vendor parity matrix. The validation suite has only been run
  on Apple M-series and a single NVIDIA workstation. Users on other
  configurations should report their fingerprint + parity numbers.
- Adversarial input handling (empty / BOS-only / max-context / illegal
  UTF-8 / OOV). Fixtures live in `tests/fixtures/adversarial-inputs.json`;
  no runner consumes them yet.

### Not claimed
- Fairness, alignment, or safety properties of the model itself.
- Performance numbers in absolute time (token/s varies ~5× across the
  GPU adapters we've tested).

## Known gaps

- The MLC q4f16_1 release uses a slightly different RoPE base (10000)
  and theta-scaling than the original Phi-3 PyTorch checkpoint. We
  inherit MLC's choice. The ε from this is below our HF parity bound
  but worth noting.
- Long-context behavior past ~2K tokens is exercised by the
  `longContext` validation case but not by the fixture suite.
- Storage usage stats (`getStoredWeightStats`) sum sizes from
  `Content-Length` headers when present and fall back to `blob.size`.
  Some Cache API entries lack a length header; the displayed total is
  a tight lower bound, not exact.

## Reproducibility checklist

When reporting a numerical mismatch:

1. Open the fingerprint footer (bottom-left of `app/`). Note the
   build SHA, GPU vendor/architecture, and browser version.
2. Click the validate button (the checkmark icon in the prompt row).
   The console prints a per-layer / per-token diff.
3. Run `tools/dump_phi3_reference.py --prompt "<your prompt>"`
   locally to regenerate the reference, OR cite the HF revision the
   bug reproduces against.
4. Open an issue with: fingerprint, prompt, validation output,
   expected output, and any suspected kernel.

## Versioning

This document is versioned with the repository. Material changes to
tolerances or test methodology require a corresponding entry in
`PREDICTIONS.md` (if the change affects an active prediction) or
`CLAUDE.md` historical log.

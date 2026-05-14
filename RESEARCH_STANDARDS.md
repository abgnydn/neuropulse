# Research-grade engineering standards

**Canonical document. Mirrored across four sibling WebGPU/WGSL research
projects:**

- [`webgpu-q`](https://github.com/abgnydn/webgpu-q) — quantum chemistry
- [`webgpu-dna`](https://github.com/abgnydn/webgpu-dna) — radiation track-structure / radiobiology
- [`zero-tvm`](https://github.com/abgnydn/zero-tvm) — Phi-3 LLM inference (hand-written WGSL, head-to-head vs WebLLM)
- [`neuropulse`](https://github.com/abgnydn/neuropulse) — live 1:1 LLM forward-pass visualization (Phi-3, 3.8B params)

Edit any one and propagate. Project-specific examples in §§ 1, 6, 7, 8, 10
diverge per repo; sections 2–5, 9, 11–15 are universal.

This is the discipline that makes the work publishable in JOSS, citable
years later, and reproducible by reviewers on different hardware. The
patterns matured in different repos and back-port / forward-port between
them (research-grade artifact discipline first in `webgpu-dna`, the
"falsify before shipping" CPU pre-screen in `zero-tvm`, **automated
doc-vs-code drift detection first in `neuropulse`** via
`verify-claims.mjs` reading `phi3-facts.ts`, full porting framework in
`webgpu-q`). Future siblings inherit the union.

**Umbrella thesis**: every advanced physics simulation in the world
should ship as a URL. The browser/WebGPU layer is what's novel; the
chemistry/physics/model architecture is textbook. **Hand-write only the
novel layer; port everything with a peer-reviewed reference.**

For `neuropulse` specifically, the novel layer is the *visualization*:
rendering 3.8B parameters as live geometry, 1:1 with the underlying
compute. The Phi-3 architecture, MLC weight format, and BPE tokenizer
are textbook and ported.

---

## 1. Single source of truth for quantitative claims

All measured numbers for `neuropulse` live in **one** canonical place:

- `src/engine/phi3-facts.ts` — every architectural constant (32 layers,
  32 attention heads, 32 KV heads, 96 head dim, 3072 hidden, 8192 FFN,
  3.821B params, 11 WGSL kernels, 22 GPU buffers, 292 dispatches/token,
  etc.). Imported by both the rendering pipeline and the doc-checker.
- `README.md` § Validation — derived from `phi3-facts.ts` via
  `tools/verify-claims.mjs`, which **fails CI** if README claims drift
  from the source of truth.

Anywhere else (`METHODS.md`, `PREDICTIONS.md`, `index.html`, hero SVG,
chip badges, blog posts) may *summarize* numbers but never *introduce*
new ones. **If a number isn't in `phi3-facts.ts`, it isn't measured.**

The `verify-claims.mjs` runs as part of CI and pre-push. Drift between
the rendering pipeline and the user-facing claims is impossible by
construction — this is `neuropulse`'s back-port-worthy contribution to
the sibling discipline.

Before stating a measurement anywhere:

  measure on the live forward pass → write to `phi3-facts.ts` → regenerate README claim → verify-claims green

Not the other way around.

---

## 2. Falsifiable JSON artifacts back every claim

Path: `tests/results/YYYY-MM-DD/<id>.json`.

Shape (locked; don't add top-level keys without updating the harness):

```json
{
  "meta":     { "protocol": "...", "hypothesis": "...", "passBar": "...",
                "seed": "named-seed-id", "warmup": 5, "trials": 20 },
  "env":      { "gitSha": "...", "userAgent": "...", "adapter": {...},
                "limits": {...}, "timestamp": "2026-05-14T...",
                "shaderHashes": {"matmul_wgsl": "...", "attention_wgsl": "...",
                                 "qkv_fused_wgsl": "...", "ffn_wgsl": "..."} },
  "rows":     [ { /* per-cell measurements */ } ],
  "status":   "pass" | "fail" | "noisy" | "partial",
  "diagnosis": "first-failing-cell + smoking-gun explanation"
}
```

Re-runnable deterministically given fixed seed + identical GPU + same
shader hash. fp16 GEMM and fp16 reductions are NOT order-deterministic
across GPU vendors — same WGSL on different hardware (Apple Metal vs
Nvidia Vulkan vs Intel iGPU) yields statistically equivalent logits
(top-k mass within ε of HF reference) but not bit-exact;
`shaderHashes` lets reviewers group rows correctly.

---

## 3. Status labels are first-class

- **`pass`** — meets the protocol's pass bar (top-k match vs HF
  reference logits, throughput within band, etc.).
- **`fail`** — doesn't. Commit anyway with a `diagnosis` field naming
  the first failing cell and the smoking gun. **Never silently rerun
  until pass.**
- **`noisy`** — `std/median > 0.1` on any cell. Informational, not
  pass/fail.
- **`partial`** — some cells pass, others don't; explicit `N of M`
  count in the diagnosis.
- **`honest negative`** — failures that are evidence. `PREDICTIONS.md`
  (pre-registered hypotheses about visualization fidelity, throughput,
  perceptual claims) and the `## Known gaps` section of `CLAUDE.md`
  cite the artifact and the rejected hypothesis.

Honest negatives become the project's evidence base. They are not
bugs to fix; they are findings.

---

## 4. Reproducibility (no randomness left to chance)

- `Math.random()` is **banned** in any test/experiment path. Sampling
  uses argmax (deterministic) or seeded top-p with a named seed from
  `src/engine/seeds.ts`. WGSL random draws (none in the current
  forward pass) would use a uniform-routed seed channel.
- Every JSON artifact records: git SHA (when available), full
  `navigator.userAgent`, `adapter.info`, WebGPU `limits`, UTC ISO8601
  timestamp, **shader-file SHA-256 / git-rev-parse hashes** for each
  of the 11 WGSL kernels (`matmul`, `attention`, `qkv_fused`,
  `fused_ffn`, `add_norm`, etc.).
- 5 warmup samples are discarded; 20 trials retained.
- Report **median + p10/p90/p99 + std + IQR** for throughput
  measurements — never single-shot.
- If `std/median > 0.1` on any cell → label the artifact `"noisy"`.

---

## 5. GPU timing requires a forced sync

`performance.now()` deltas around `queue.submit` alone are fiction —
WebGPU is asynchronous. **Mandatory pattern**: a mapped readback of a
tiny buffer (a single `f32`) before AND after the work. The throughput
counter shown live in the chat UI uses this pattern; the offline
benchmark harness in `tests/perf/` likewise.

---

## 6. Multi-level correctness verification

Match against more than one reference frame. Listed in increasing
sophistication / decreasing strength:

1. **Closed-form invariants**: norm preservation across RMSNorm,
   softmax mass = 1 across attention, KV-cache append idempotence,
   tokenizer round-trip identity on UTF-8 corpora.
2. **Brute-force diagnostic on a single token**: deterministic
   forward pass on a fixed prompt, intermediate-tensor dumps after
   each of the 32 layers, hand-checked against an HF reference run
   on the same prompt.
3. **Peer-reviewed reference packages**:
   - HuggingFace `microsoft/Phi-3-mini-4k-instruct` in PyTorch fp16
     as the bit-comparable reference for *logits* (top-k mass match
     within ε of fp16 noise floor).
   - MLC q4f16_1 quantized weights as the bit-comparable reference
     for *weights*.
   - WebLLM (`mlc-ai/web-llm`) as a peer browser-runtime reference
     for *throughput* and *output equivalence*.
4. **Experiment**: human evaluation of the visualization fidelity
   claim ("every tensor rendered 1:1") — pre-registered in
   `PREDICTIONS.md` so that an outside reviewer can falsify it.

Multiple independent reference frames > one. Each artifact should
state which it's checking against in `meta.hypothesis`.

---

## 7. Port from references; hand-write only the novel layer

This is the architectural rule. The differentiator of `neuropulse` is
the **live visualization of the forward pass at 1:1 scale** — not the
Phi-3 architecture or the quantization scheme. So:

- **Hand-written and owned**:
  - All 11 WGSL compute kernels (`matmul.wgsl`, `qkv_fused.wgsl`,
    `attention.wgsl`, `fused_ffn.wgsl`, `add_norm.wgsl`, etc.).
  - WebGPU dispatch glue and the 292 dispatches/token schedule.
  - The 22-buffer GPU memory layout.
  - The rendering / visualization layer that maps tensors to live
    geometry.
  - `verify-claims.mjs` doc-vs-code drift detector.
  - The research-grade harness.
- **Ported from peer-reviewed source with attribution**:
  - **Phi-3 architecture spec** (32 layers / 32 attention heads /
    32 KV heads / 96 head dim / SwiGLU / RoPE / RMSNorm /
    grouped-query attention) from Microsoft's released
    `Phi-3-mini-4k-instruct` model card and `config.json`.
  - **MLC `ndarray-cache.json` weight format** from `mlc-ai/web-llm`
    and `mlc-ai/mlc-llm`, used directly so `neuropulse` and WebLLM
    can read the same on-disk weights.
  - **BPE tokenizer** patterns from `tokenizers` (HuggingFace) /
    `sentencepiece`.
  - **RoPE / GQA / SwiGLU** reference numerics from the original
    papers (Su et al. 2021 RoFormer, Ainslie et al. 2023 GQA,
    Shazeer 2020 GLU).

**Per-file header** for ported code:

```
// Ported from <upstream> (<upstream-url>), <license> license.
// Source: <relative-path> at commit <SHA>
// Original authors: <upstream/AUTHORS>
// Adaptations for neuropulse:
//   - <substantive change 1>
//   - ...
// See LICENSE-<UPSTREAM> at repo root for the <license> notice.
```

**Repo-level**: `LICENSE-MLC` and `LICENSE-PHI3` at root (verbatim
from upstream). Per-module status table belongs in a `MIGRATION.md`
table:

| module | reference | license | status |
|---|---|---|---|
| `tokenizer.ts` BPE | `huggingface/tokenizers` | Apache 2.0 | 🟢 |
| weight loader | MLC `ndarray-cache.json` format | Apache 2.0 | 🟢 |
| Phi-3 architecture | Microsoft Phi-3 model card / config | MIT | 🟢 |

License compatibility: MIT + Apache 2.0 work together — the ported
portion keeps its upstream license obligations (notice + state
changes); the rest of the repo stays MIT.

---

## 8. No fudge factors without a citation

Any tunable scalar in production code that isn't backed by a
peer-reviewed source is:

1. **Labeled empirical** in the code comment at point of use.
2. **Documented in `CLAUDE.md` § Known gaps** with the magnitude of
   the empirical correction and what observable it was tuned against.
3. **Queued for removal** once the structural fix lands.
4. **Tracked in `CHANGELOG.md` / commit messages** when added and
   when removed.

`neuropulse` aims to have **zero fudge factors in the forward pass** —
every numeric scale (head-dim scale, RMS epsilon, RoPE base, softmax
temperature) is sourced from Phi-3's released config and pinned in
`phi3-facts.ts`. Drift here is fatal: the whole pitch is *accurate*
1:1 visualization, so any unsourced constant is documented as an
honest gap.

Tested-and-rejected hypotheses (e.g., "fused QKV+RoPE+KV-append loses
fp16 precision" — falsified by top-k match against HF reference) go
into the same documents so future sessions don't re-test them.

---

## 9. Shader byte-hashing for reproducibility

Every artifact records the SHA-256 (or `git rev-parse <gitSha>:<path>`
short hash) of each of the 11 WGSL shader files the experiment
depended on. This lets reviewers group rows by shader version when a
kernel implementation changes (subgroup tile size, int4 dequant
strategy, fused-vs-unfused FFN, etc.).

The `env` block carries `shaderHashes: { matmul_wgsl: "...",
attention_wgsl: "...", qkv_fused_wgsl: "...", fused_ffn_wgsl: "...",
add_norm_wgsl: "...", ... }`.

---

## 10. Living open-gaps document

Two siblings document open gaps:

- `PREDICTIONS.md` at root — **pre-registered hypotheses** about
  performance, visualization fidelity, and perceptual claims. Each
  prediction is dated, falsifiable, and gets resolved (confirmed /
  refuted / partial) in the same commit that surfaces the evidence.
- `CLAUDE.md § Known gaps` — operational diagnoses with the same
  three-part structure as the canonical doc:

```
## N. The <observable> deficit vs <reference> (<artifact>, <date>)

Observed.  <quantitative gap with σ-significance>

Hypothesis A — <candidate root cause>
Hypothesis B — <alternative>

Falsification experiment: <what would distinguish them>
```

Entries are removed when the underlying gap closes; the artifact
references stay in `CHANGELOG.md`. Tested-and-rejected hypotheses
get a strikethrough entry with the refutation artifact link, so
the same hypothesis isn't tried twice.

---

## 11. Honest self-corrections

When a prior claim turns out wrong, revise it **in the same commit
that surfaces the data**, with the full arc preserved. The
`verify-claims.mjs` doc-vs-code checker makes silent drift
impossible — any narrative shift forces a corresponding source-of-
truth edit. Examples:

- "11 WGSL kernels" claim drifted from an earlier "8 kernels"
  number; `verify-claims.mjs` caught the drift in CI and forced the
  README + hero SVG + stats badge to update together with the
  source-of-truth fact in `phi3-facts.ts`.
- "292 dispatches/token" was the result of a measurement, not a
  prediction; an earlier prediction of "~250 dispatches" was
  archived as a refuted prediction in `PREDICTIONS.md` rather than
  retconned out of the narrative.

This is publication-grade transparency. **Wrong hypotheses become
part of the public scientific record, not an embarrassment to
hide.**

---

## 12. Citation infrastructure per release

Each minor release ships:

1. Git tag (`v0.X.Y`)
2. GitHub Release with notes drawn from `CHANGELOG.md`
3. **Zenodo DOI** minted via the GitHub-Zenodo integration
4. `CITATION.cff` `preferred-citation` block updated with the real
   DOI

Patch releases (doc-only, refactor, etc.) skip the Zenodo step.

---

## 13. WebGPU gotchas (carry forward across all projects)

- `initGPU()` MUST pass `requiredLimits` for
  `maxStorageBufferBindingSize` and `maxBufferSize`. The default
  128 MiB cap silently truncates large dispatches; Phi-3-mini's
  full weight set exceeds this without explicit limits.
- `atomicAdd` works only on `u32` — not f32. The forward pass
  avoids atomic reductions entirely (tree reductions in shared
  memory instead).
- No recursion in WGSL. All shaders are single-pass.
- Uniform buffers must be 16-byte aligned.
- No subgroup intrinsics in WebGPU 1.0 spec; subgroup-based GEMM
  tiles live behind feature flags and ship as A/B variants of the
  matmul kernel.

---

## 14. Test discipline (non-negotiable)

- TypeScript `strict` + `noUncheckedIndexedAccess`. No exceptions.
- ESLint clean — 0 errors. Warnings tracked, ideally 0.
- CI green. Every PR runs unit + Playwright e2e + typecheck + lint
  + `verify-claims.mjs`.
- Each kernel has paired test coverage by **intent**, not by metric:
  - **Closed-form invariant** (norm preservation, softmax mass = 1,
    tokenizer round-trip) where it exists.
  - **Peer-package** (HuggingFace Phi-3-mini fp16 logits, MLC
    quantized weights, WebLLM throughput) on a fixed prompt.
  - **Brute-force** single-token deterministic forward pass with
    intermediate-tensor dumps where feasible.
- Honest negatives (status: "fail" tests) live alongside passes; they
  don't break CI but they're surfaced in the suite output.

---

## 15. Release cadence

- **Minor releases** (`v0.X.0`) for substantive features or
  scientific findings. Tag + GitHub Release + Zenodo DOI.
- **Patch releases** (`v0.X.Y`) for doc-only, refactor, SVG refresh,
  narrative updates. Tag + GitHub Release, no DOI.
- **CHANGELOG** follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
  format: `### Added / Changed / Fixed / Documented / Honest negatives`.
- **CITATION.cff version** matches `package.json` version matches
  Git tag matches GitHub Release tag, all pinned per release.

---

## On adding a new sibling project

Inherit these 15 principles from day one. Copy this file verbatim into
the new repo. Replace project-specific references in sections 1, 6, 7,
8, 10 with the new project's analogs. Cross-link sibling projects in
the header.

The discipline is the product.

---

*Last revised: 2026-05-14. Canonical mirror of
[`webgpu-q/RESEARCH_STANDARDS.md`](https://github.com/abgnydn/webgpu-q/blob/main/RESEARCH_STANDARDS.md).
Edit either and propagate.*

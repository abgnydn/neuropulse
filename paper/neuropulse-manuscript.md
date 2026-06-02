# Watching a 3.8-billion-parameter transformer think: a browser-native, reference-validated, 1:1 real-time visualization of an LLM forward pass

**Ahmet Barış Günaydın**
*Independent researcher · github.com/abgnydn/neuropulse · neuropulse.live*

---

## Abstract

Existing visualizations of transformer language models fall into two disjoint
camps: pedagogical tools that animate *toy* models (a few thousand parameters, a
sort-the-alphabet task) in real 3D, and inference engines that run real
multi-billion-parameter models with *zero* internal visibility. Nothing connects
them — you can either watch a model that isn't real, or run a real model you
can't watch. **neuropulse** closes that gap: it renders a real forward pass of
Phi-3-mini-4k-instruct (3.8 B parameters, 32 layers) in a web browser, in real
time, with every rendered element bound 1:1 to a named tensor in the model's
compute graph and read back from the same WebGPU buffers the model just wrote —
no interpolation, no sampling, no schematic stand-in, no server. To our
knowledge this is the first visualization that is simultaneously (i) a real
multi-billion-parameter model, (ii) running in-browser, (iii) in 3D, (iv) over
*all* live tensors rather than attention alone, and (v) validated against a
reference implementation. The inference engine is hand-written: eleven WGSL
compute kernels over MLC's `q4f16_1` weights, 292 GPU dispatches per token on the
headless path and 348 with the visualizer reading back per-layer attention
scores and eight logit-lens probes. Correctness is not asserted but *tested*: a
built-in suite diffs the live WebGPU forward pass against a pinned HuggingFace
fp16 reference at nine layer checkpoints and holds per-layer relative L2 < 2×10⁻²,
cosine > 0.999, **100% top-1 token agreement**, and softmax JSD < 5×10⁻³ — the
user's own GPU is the test rig. The platform is built as an empirical lab with a
pre-registered prediction log; we demonstrate what that enables with experiment
E45, a falsification of a continuous-attention self-consistency hypothesis:
iterating Phi-3-mini's attention to its Picard fixed point destroys language
coherence at a sharp cliff (iteration 2, universal across 16/16 prompts) and
converges to a degenerate low-dimensional attractor, while a well-trained 4-layer
control model stays robust (91% top-1 retained) — locating the brittleness in
scale and training breadth, not in transformers per se.

---

## 1. Introduction

Most "AI visualizations" are decoration: particle systems and pulsing dots with
no model behind them. They show how a designer *imagines* an LLM works. A smaller
set of serious tools show something real, but each gives up a dimension that
matters. Bycroft's LLM-Viz renders a transformer in beautiful 3D with live
tensors — of a ~1,000-parameter toy that sorts letters. Transformer Explainer
runs a real GPT-2-small (124 M) in the browser, in 2D, with partial tensor
exposure. BertViz exposes real attention from real HuggingFace models, but only
attention, only in a notebook, only in 2D. WebLLM runs real multi-billion-
parameter models in the browser at full speed — and shows you nothing inside.

The gap is structural: visualization tools scale *down* to stay legible;
inference engines treat internals as opaque to stay fast. neuropulse takes the
position that the internals of a *full-scale* model, drawn faithfully, are worth
the engineering to expose — both as pedagogy (you can watch a real 3.8 B
transformer generate a token) and as an interpretability substrate (every tensor
is addressable, so experiments can be run *on the thing being shown*). The
contribution is not a new model or a new rendering technique; it is the
demonstration that a real, full-scale forward pass can be rendered 1:1 in a
browser tab, that the 1:1 claim can be made falsifiable against a reference, and
that the resulting apparatus is a usable lab. We show the last point with a
pre-registered falsification (§3.3).

## 2. Methods

### 2.1 Inference engine

The model is Phi-3-mini-4k-instruct (3.8 B parameters; 32 layers; 32 attention
heads, no GQA; head dimension 96; hidden dimension 3,072; SwiGLU FFN inner
dimension 8,192; vocabulary 32,064), loaded in MLC's `q4f16_1` ndarray-cache
format (4-bit weights, fp16 scales, group size 32) — *not* GGUF. The decode loop
is hand-written as eleven WGSL compute kernels (`embedding`, `int4_matmul`,
`int4_matmul_f32`, `rope`, `attention`, `attention_scores`, `kv_append`,
`rms_norm`, `add_norm`, `fused_ffn`, `argmax`) with no inference framework in the
path. One generated token is 292 GPU dispatches on the headless fast path; the
visualized path adds 56 (one attention-scores readback per layer plus eight
logit-lens probes of three dispatches each) for 348. Weights are streamed once
and cached in OPFS and the Cache API for instant reload.

### 2.2 1:1 rendering by shared-buffer readback

Inference and visualization **share the same GPU buffers**. The renderer does not
recompute, approximate, or re-simulate anything: it reads the values the model
already produced and maps each to a scene element bound to a named tensor — the
residual stream, the 32×32 attention-head grid, the gated MLP, the token strip,
the per-layer logit-lens. Spatial layout is derived from the model rather than
chosen by a designer: residual-stream positions come from a PCA of the model's
own layer-0 `qkv_proj` weight matrix, so dimensions read into attention together
cluster together. Ten draggable panels each surface one live tensor (top-k,
confidence, KV-cache occupancy, residual norm, per-layer delta, last-layer
attention, logit-lens, …).

### 2.3 Validation methodology

"Strict 1:1" is a strong claim, so it is made falsifiable. A built-in suite diffs
the live WebGPU forward pass against a pinned HuggingFace fp16 Phi-3-mini at nine
layer checkpoints (`VALIDATE_LAYERS = {0,4,8,12,16,20,24,28,31}`, 27,648 floats
compared per prompt). The single intentional precision tradeoff vs the FP32
reference is f16 accumulation of intermediate residuals at every Add+Norm
boundary; matmul, softmax, RMSNorm, RoPE, and SiLU all accumulate at f32 inside
the kernels. The suite runs on the user's own GPU in under a minute and prints to
their console; a runtime fingerprint (build SHA, GPU vendor/architecture, browser)
stamps every report. A second, commit-time gate (`verify-claims.mjs`,
`check-shortcuts.mjs`) cross-checks every numeric claim in the documentation
(layer count, kernel count, dispatch counts, keyboard shortcuts) against the
source, so prose cannot silently drift from code.

## 3. Results

### 3.1 The visualization system

neuropulse renders a real Phi-3-mini forward pass at interactive rates in a
browser tab: type a prompt, and 3.8 B parameters process it with every
intermediate tensor read back and drawn. Five view modes (Journey, Scene,
Attention, Logit-Lens, Cinematic) and four stacking overlays present the same
underlying activations from different angles; the camera, panels, and a
keyboard-driven cinematic flythrough cover the interaction surface. Because the
render path reads the inference buffers directly, what is on screen *is* the
model's state, not a depiction of it.

### 3.2 Correctness against a reference

The 1:1 claim holds against HuggingFace fp16 within the declared tolerances. The
bounds the suite asserts (a run exceeding any is flagged a regression):

| Test | Quantity | Bound |
|---|---|---|
| Hidden state vs HF | relative L2 per layer (3,072 dims) | < 2×10⁻² (observed max ≈ 1.4×10⁻²) |
| Hidden state vs HF | cosine similarity per layer | > 0.999 |
| Last-attention reconstruction | rel. L2 of (Σ scores·V − attn_out) | < 1×10⁻² |
| Top-1 token agreement | greedy argmax, ≤ 30 tokens | **100% match** |
| Top-5 overlap | mean number of shared top-5 tokens (GPU vs HF) | ≥ 4 |
| Softmax divergence | mean JSD over top-k | < 5×10⁻³ |
| Sampler self-test | JSD(5,000 samples, softmax) | < 1×10⁻² |

The deltas at hidden-state level are the cost of int4 quantization and f16
residual round-trips, not implementation drift; the bar that matters —
identical top-1 tokens vs the fp16 reference on the validation prompt set — is
met. Each kernel additionally carries a declared per-kernel error budget (e.g.
`int4_matmul` ≤ 4 ULP@f16, `rope` ≤ 1 ULP, `kv_append`/`embedding` bit-exact);
wiring the per-kernel CPU references into CI is a tracked gap (§4).

### 3.3 What the lab produces: experiment E45

The platform exposes every tensor and ships a pre-registered prediction log, so
hypotheses can be tested on the model being visualized. Prediction P-20260526-07
asked whether iterating Phi-3-mini's attention to a per-layer self-consistency
(Picard) fixed point — `Q ← softmax(QKᵀ/√d)V`, RoPE applied once, K/V held fixed
— yields hidden states close to the standard one-step operator. The fixed-point
kernel ships behind a flag (`?attn=fixedpoint`); a wiring gate confirmed
`max_iter = 1` reproduces the baseline byte-for-byte before any claim was made.

The hypothesis is **falsified at zero-shot inference**. One extra Picard step
destroys coherence at a sharp cliff: on "The capital of Japan is …", iteration 1
matches the standard kernel exactly ("…Tokyo. Tokyo is a major global"),
iteration 2 collapses ("lee\\n\\n\\nTop \\nTop"), and iterations 3–100 converge
byte-for-byte to a degenerate topic-projector attractor. A 16-prompt sweep
(English, code, math, Japanese, emoji, JSON) makes the cliff universal: **16/16
prompts that are coherent at iteration 1 collapse at iteration 2**, with a
low-dimensional attractor vocabulary (`"lee"`, `"ício"`, `"RESS"`, newlines) and
a second failure mode (immediate stop-token / empty output) on two prompts. The
Picard fixed point demonstrably *exists*, is numerically reachable
(‖Q_t − Q_{t-1}‖_∞ falls to f32 noise within 100 iterations), and costs only 1.41×
compute at `max_iter = 100` — so the cliff is the science, not an artifact of
under-iteration or budget.

A disentanglement control separates scale from the phenomenon: a well-trained
4-layer / 64-hidden toy transformer (same RMSNorm + SwiGLU + RoPE family) under
the identical protocol stays in the "match" bucket — 91% top-1 retained at
iteration 2, stable through iteration 100 — where Phi-3-mini retains 0% coherent
output. The publishable claim is therefore stronger than "Path A falsified":
**a trained discrete-attention layer's distance from its own Picard fixed point
grows with model scale and training breadth; brittleness to attention iteration
is a property of well-trained large LMs specifically, not of transformers in
general.** A per-layer single-layer-ablation (Phase 3) to adjudicate the two
pre-registered per-layer hypotheses is the natural follow-up and is not yet run.

## 4. Honest limitations

- **Per-kernel error budgets are declared, not yet CI-asserted.** The bounds in
  §3.2 are documented and the end-to-end HF parity is enforced on every user
  click, but the runner that checks each kernel against a CPU f64 reference does
  not yet exist.
- **Cross-vendor parity is under-sampled.** The validation suite has been run on
  Apple M-series and a single NVIDIA workstation; the cross-vendor matrix is open
  and users are asked to report fingerprint + parity numbers.
- **One model.** The engine is specialized to Phi-3-mini's architecture and the
  MLC `q4f16_1` format (whose RoPE base/theta-scaling we inherit). Generalizing
  to other architectures is engineering not yet done.
- **Not a training or fine-tuning tool**, and no claim is made about the model's
  fairness, alignment, or safety — only that what is rendered faithfully matches
  the reference forward pass.
- **E45 per-layer attribution is deferred.** The output cliff at iteration 2 is
  sharp enough that the two pre-registered per-layer hypotheses (deepest-middle
  vs high-entropy layers as the catastrophic ones) are both partially refuted at
  model-level granularity; the single-layer ablation that would pick a winner is
  Phase 3, unrun.

## 5. Related work

Prior interactive transformer visualizations trade scale for visibility or vice
versa: Bycroft's LLM-Viz (real 3D, live tensors, ~1 K-parameter toy), Transformer
Explainer (real GPT-2-small, 124 M, 2D, partial tensors), BertViz (real
attention only, 2D, notebook), and inference runtimes such as WebLLM (real
multi-billion-parameter models in-browser, no internal visibility). Across the
axes *real model · multi-billion scale · browser · 3D · all live tensors ·
reference-validated*, neuropulse is, to our knowledge, the first to satisfy all
six simultaneously. The interpretability result in §3.3 relates to fixed-point /
equilibrium views of attention (deep equilibrium models; Ramsauer et al.'s modern
Hopfield reading of attention); we test the *inference-time* self-consistency
point of a trained discrete-attention model and find it qualitatively unlike the
one-step operator, with a scale-dependent severity made visible by the control.

## 6. Software availability and reproducibility

Source: `github.com/abgnydn/neuropulse` (MIT); live at `neuropulse.live`. The
WGSL kernels are in `src/engine/shaders/`; the canonical architecture constants
and derived dispatch counts in `src/engine/compiler.ts` and
`src/engine/phi3-facts.ts`; precision matrix, tolerances, and per-kernel budgets
in `METHODS.md`; the pre-registered prediction log in `PREDICTIONS.md`. The HF
parity suite runs on any WebGPU-capable Chromium from the demo's validate button
and regenerates its reference via `tools/dump_phi3_reference.py`; the E45
artifacts are committed under `tests/results/`. Documentation numbers are gated
against source by `npm run verify`. The Butterfly context-compaction study that
also ran on this platform has its own repository (`github.com/abgnydn/butterfly`).

**Generative-AI disclosure.** Portions of the software, documentation, and this
manuscript were drafted with a large language model used as a coding and writing
aid; all output was author-reviewed, correctness was enforced by the HF parity
suite and the commit-time claim checks, and every quantitative claim is traceable
to `METHODS.md`, `phi3-facts.ts`, or a committed experiment artifact.

**Statements.** Sole author; no competing interests; no external funding. Model
weights are Microsoft's Phi-3-mini redistributed by MLC and are not part of this
work.

## 7. Conclusion

A full-scale transformer forward pass can be rendered 1:1, in real time, in a
browser tab — every element a real tensor read from the buffers the model just
wrote, every "1:1" claim falsifiable against a HuggingFace reference the user
runs on their own GPU. Closing the gap between toy-but-visible and real-but-opaque
yields both a pedagogical artifact (watch 3.8 B parameters produce a token) and
an interpretability substrate: experiment E45 used it to falsify a
continuous-attention self-consistency hypothesis and, via a trained small-model
control, to attribute the resulting brittleness to scale and training breadth
rather than to transformers in general. The platform's discipline — derived
constants, a pre-registered prediction log, and a validation suite that fails
loud — is the point as much as the rendering is.

---

## References (to be formatted)

- **Bycroft** — LLM-Viz, an interactive 3D visualization of a small GPT. bbycroft.net/llm
- **Cho et al.** — Transformer Explainer: interactive learning of GPT-2 in the browser. poloclub.github.io/transformer-explainer
- **Vig** — BertViz: a tool for visualizing attention in transformer models.
- **MLC team** — WebLLM / MLC-LLM: in-browser LLM inference via WebGPU. webllm.mlc.ai
- **Bai, Kolter, Koltun** — Deep Equilibrium Models, NeurIPS 2019.
- **Ramsauer et al.** — Hopfield Networks is All You Need, ICLR 2021.
- **Microsoft** — Phi-3 Technical Report (Phi-3-mini-4k-instruct).
- **W3C GPU for the Web Working Group** — WebGPU specification.

---

*Draft v0.1. Built from the committed engine, `METHODS.md` tolerances, and the
`PREDICTIONS.md` / `tests/results/` E45 artifacts; every documentation number is
gated against source by `npm run verify`. To be converted to LaTeX before
submission to a visualization (VIS/EuroVis), ML-systems, or interpretability
venue.*

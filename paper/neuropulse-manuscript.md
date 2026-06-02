# Watching a 3.8-billion-parameter transformer think: a browser-native, reference-validated, 1:1 real-time visualization of an LLM forward pass

**Ahmet Barış Günaydın**
*Independent researcher · github.com/abgnydn/neuropulse · neuropulse.live*

**Keywords:** WebGPU; WGSL; large language models; transformer; mechanistic interpretability; in-browser computation; visualization; reproducibility; pre-registration; attention fixed points

---

## Abstract

Interactive visualizations of transformer language models fall into two disjoint
regimes: pedagogical tools that animate *toy* models — on the order of $10^3$
parameters, a sort-the-alphabet task — in faithful 3D, and inference engines
that run real multi-billion-parameter models with *no* internal visibility. The
two have never met: one can watch a model that is not real, or run a real model
one cannot watch. **neuropulse** closes that gap. It renders a real forward pass
of Phi-3-mini-4k-instruct (3.8 B parameters, 32 layers) in a web browser, in
real time, with every rendered element bound one-to-one to a named tensor in the
model's compute graph and read back from the same WebGPU buffers the model just
wrote — no interpolation, no sampling, no schematic stand-in, and no server. To
our knowledge this is the first visualization that is simultaneously (i) a real
multi-billion-parameter model, (ii) in-browser, (iii) in 3D, (iv) over *all*
live tensors rather than attention alone, and (v) validated against a reference
implementation. The engine is hand-written: eleven WGSL compute kernels over
MLC's `q4f16_1` weights, 292 GPU dispatches per token headless and 348 with the
visualizer instrumented. The one-to-one claim is made falsifiable: a built-in
suite diffs the live WebGPU forward pass against a pinned HuggingFace fp16
reference at nine layer checkpoints and holds per-layer relative $L_2 < 2\times
10^{-2}$, cosine $> 0.999$, **100 % top-1 token agreement**, and softmax
Jensen–Shannon divergence $< 5\times10^{-3}$ — the reader's own GPU is the test
rig. We treat the platform as an empirical lab with a pre-registered prediction
log and demonstrate the payoff with experiment E45, a falsification of a
continuous-attention self-consistency hypothesis: iterating Phi-3-mini's
attention to its Picard fixed point destroys language coherence at a sharp cliff
(iteration 2, universal across 16/16 prompts) and converges to a degenerate
low-dimensional attractor, whereas a well-trained 4-layer control model retains
91 % of its behavior — locating the brittleness in model scale and training
breadth rather than in transformers as such.

---

## 1. Introduction

Most public "AI visualizations" are decoration: particle systems and pulsing
dots with no model behind them, showing how a designer *imagines* a language
model works. A smaller set of serious tools show something real, but each
surrenders a dimension that matters for understanding a *production-scale* model.
Bycroft's LLM-Viz renders a transformer in 3D with live tensors — of a
$\sim\!10^3$-parameter toy that sorts letters. Transformer Explainer [1] runs a
real GPT-2-small (124 M) live in the browser, in 2D, with partial tensor
exposure. BertViz [2] surfaces real attention from real HuggingFace models, but
attention only, in a notebook, in 2D. In-browser inference runtimes such as
WebLLM run real multi-billion-parameter models at full speed and expose nothing
inside.

The gap is structural. Visualization tools scale *down* to stay legible;
inference engines treat internals as opaque to stay fast. neuropulse takes the
position that the internals of a *full-scale* model, drawn faithfully, justify
the engineering required to expose them — as pedagogy (one can watch a real
3.8 B transformer generate a token) and as an interpretability substrate (every
tensor is addressable, so experiments run *on the artifact being shown*, not on
a proxy).

**Contributions.** This work contributes:

1. **A browser-native, server-free engine** that runs a real Phi-3-mini forward
   pass as eleven hand-written WGSL kernels over 4-bit weights, with a derived
   (not hard-coded) dispatch budget of 292 per token (§2.1).
2. **A one-to-one rendering architecture** in which the renderer reads the
   inference buffers directly rather than recomputing, so on-screen geometry
   *is* the model state; spatial layout is derived from the model's own weights
   (§2.2).
3. **A falsifiable validation methodology** — a per-layer diff against a pinned
   HuggingFace fp16 reference that the reader runs on their own GPU, plus a
   commit-time gate that fails the build when documentation drifts from source
   (§2.3, §3.2).
4. **An empirical-lab demonstration (experiment E45)** that uses the platform to
   pre-register and falsify a continuous-attention self-consistency hypothesis,
   and — via a trained small-model control — attributes the resulting
   brittleness to scale and training breadth (§3.3).

The contribution is neither a new model nor a new rendering primitive; it is the
demonstration that a real, full-scale forward pass can be rendered one-to-one in
a browser tab, that the claim can be made falsifiable against a reference, and
that the resulting apparatus is a usable interpretability lab.

## 2. Methods

### 2.1 Inference engine

The model is Phi-3-mini-4k-instruct [3]: 3.8 B parameters; 32 decoder layers; 32
attention heads (no grouped-query attention); head dimension 96; hidden
dimension 3,072; SwiGLU feed-forward inner dimension 8,192; vocabulary 32,064. It
is loaded in MLC's `q4f16_1` ndarray-cache format — 4-bit weights, fp16 scales,
group size 32 — *not* GGUF. The decode loop is hand-written as eleven WGSL
compute kernels (`embedding`, `int4_matmul`, `int4_matmul_f32`, `rope`,
`attention`, `attention_scores`, `kv_append`, `rms_norm`, `add_norm`,
`fused_ffn`, `argmax`) with no inference framework in the path. One generated
token costs

$$D_\text{fast} = 9L + 4 = 292, \qquad D_\text{viz} = D_\text{fast} + L + 8\cdot 3 = 348 \quad (L = 32),$$

where the headless cost is nine per-layer dispatches plus a four-dispatch
prologue/epilogue, and the visualized path adds one attention-scores readback
per layer and eight three-dispatch logit-lens probes. Counts are derived in
source from a closed form, not written as prose (§2.3). Weights stream once and
persist in OPFS and the Cache API for instant reload.

### 2.2 One-to-one rendering by shared-buffer readback

Inference and visualization **share the same GPU buffers** (Figure 1). The
renderer does not recompute, approximate, or re-simulate: it reads the values the
model already produced and maps each to a scene element bound to a named tensor —
the residual stream, the $32\times32$ attention-head grid, the gated MLP, the
token strip, the per-layer logit-lens. Spatial layout is derived from the model
rather than chosen by a designer: residual-stream positions are the
two-dimensional PCA projection of the model's own layer-0 `qkv_proj` weight
matrix, so dimensions read into attention together cluster together. Ten
draggable panels each surface one live tensor (top-$k$, confidence, KV-cache
occupancy, residual norm, per-layer delta, last-layer attention, logit-lens, and
so on), persisted to `localStorage`.

![System architecture. The auto-regressive inference path (prompt, tokenizer, embedding, 32 transformer blocks, LM head, sampler) and the render path read the *same* GPU buffers; the renderer surfaces the activations the model already produced rather than recomputing them.](fig-architecture.pdf){width=92%}

### 2.3 Validation methodology

"Strict one-to-one" is a strong claim, so it is made falsifiable. A built-in
suite diffs the live WebGPU forward pass against a pinned HuggingFace fp16
Phi-3-mini at nine checkpoints, $\mathcal{V} = \{0,4,8,12,16,20,24,28,31\}$
(27,648 floats compared per prompt). For a layer $\ell$ with live hidden state
$h^\text{GPU}_\ell$ and reference $h^\text{HF}_\ell$ over the leading 3,072 dims,
the suite asserts

$$\frac{\lVert h^\text{GPU}_\ell - h^\text{HF}_\ell\rVert_2}{\lVert h^\text{HF}_\ell\rVert_2} < 2\times10^{-2}, \qquad \cos\!\big(h^\text{GPU}_\ell, h^\text{HF}_\ell\big) > 0.999 .$$

The single intentional precision tradeoff versus the FP32 reference is f16
accumulation of intermediate residuals at every Add+Norm boundary; matmul,
softmax, RMSNorm ($\varepsilon = 10^{-5}$), RoPE, and SiLU each accumulate at f32
inside the kernels. A second, commit-time gate cross-checks every numeric claim
in the documentation (layer count, kernel count, dispatch counts, keyboard
shortcuts) against source, failing the build on drift, so prose cannot silently
diverge from code.

### 2.4 Experimental protocol

All measurements are on an Apple M2 Pro under Chromium with WebGPU. Decoding is
deterministic greedy (argmax) for every parity and E45 measurement; sampling is
exercised only by the RNG self-test. Each run is stamped with a runtime
fingerprint — build SHA, GPU vendor/architecture, and browser version — surfaced
in the demo footer and recorded with every reported artifact, so any number can
be regenerated under a known configuration. The reference dump is produced by
`tools/dump_phi3_reference.py` against a pinned HuggingFace revision; the parity
suite runs from the demo's validate button in under a minute on the reader's own
GPU. E45 (§3.3) uses the existing 15-prompt logit sweep plus a 16-prompt
multi-format sweep (English, code, mathematics, Japanese, emoji, JSON), 12 tokens
per prompt, run through a headed-Chromium Playwright harness; the small-model
control runs on CPU.

## 3. Results

### 3.1 The visualization system

neuropulse renders a real Phi-3-mini forward pass at interactive rates in a
browser tab: a prompt is typed, and 3.8 B parameters process it with every
intermediate tensor read back and drawn. Five view modes (Journey, Scene,
Attention, Logit-Lens, Cinematic) and four stacking overlays present the same
underlying activations from different angles; a keyboard-driven cinematic
flythrough, camera control, and the panel set cover the interaction surface.
Because the render path reads the inference buffers directly (§2.2), what is on
screen *is* the model's state rather than a depiction of it. Table 1 positions
the system against prior interactive transformer visualizations.

Table: Interactive transformer visualizations across six axes. neuropulse is, to our knowledge, the first to satisfy all six simultaneously.

| Tool | Real model | Scale | Browser | 3D | Live tensors | Ref-validated |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| LLM-Viz (Bycroft) | toy (sorts ABC) | $\sim\!10^3$ | yes | yes | yes | no |
| Transformer Explainer [1] | GPT-2-small | 124 M | yes | no | partial | no |
| BertViz [2] | HF models | any | no | no | attention only | no |
| WebLLM | yes | multi-B | yes | — | no | — |
| **neuropulse** | **Phi-3-mini** | **3.8 B** | **yes** | **yes** | **all** | **HF fp16** |

### 3.2 Correctness against a reference

The one-to-one claim holds against HuggingFace fp16 within the declared
tolerances (Table 2); a run exceeding any bound is flagged a regression. The
hidden-state deltas are the cost of int4 quantization and f16 residual
round-trips, not implementation drift; the decisive bar — identical top-1 tokens
versus the fp16 reference on the validation prompt set — is met. Each kernel
additionally carries a declared per-kernel error budget (e.g. `int4_matmul`
$\le 4$ ULP at f16, `rope` $\le 1$ ULP, `kv_append` and `embedding` bit-exact);
wiring the per-kernel CPU references into continuous integration is a tracked gap
(§4).

Table: Validation bounds asserted by the in-app suite against a pinned HuggingFace fp16 Phi-3-mini reference.

| Test | Quantity | Bound |
|:--|:--|:--|
| Hidden state vs HF | relative $L_2$ per layer (3,072 dims) | $< 2\times10^{-2}$ (obs. max $\approx 1.4\times10^{-2}$) |
| Hidden state vs HF | cosine similarity per layer | $> 0.999$ |
| Last-attention reconstruction | rel. $L_2$ of $(\sum \text{scores}\cdot V - \text{attn\_out})$ | $< 1\times10^{-2}$ |
| Top-1 token agreement | greedy argmax, $\le 30$ tokens | **100 % match** |
| Top-5 overlap | mean shared top-5 tokens (GPU vs HF) | $\ge 4$ |
| Softmax divergence | mean JSD over top-$k$ | $< 5\times10^{-3}$ |
| Sampler self-test | JSD(5,000 samples, softmax) | $< 1\times10^{-2}$ |

### 3.3 What the lab produces: experiment E45

Because the platform exposes every tensor and ships a pre-registered prediction
log, hypotheses can be tested on the model being visualized. Prediction
P-20260526-07 asked whether iterating Phi-3-mini's attention to a per-layer
self-consistency (Picard) fixed point of

$$Q \;\leftarrow\; \mathrm{softmax}\!\big(QK^{\top}/\sqrt{d}\big)\,V,$$

with RoPE applied once at iteration 0 and $K,V$ held fixed, yields hidden states
close to the standard one-step operator. The fixed-point kernel ships behind a
flag (`?attn=fixedpoint`); a wiring gate first confirmed that `max_iter = 1`
reproduces the baseline byte-for-byte, before any claim was made.

The hypothesis is **falsified at zero-shot inference**. One additional Picard
step destroys coherence at a sharp cliff (Figure 2). On "The capital of Japan
is …", iteration 1 matches the standard kernel exactly ("…Tokyo. Tokyo is a major
global"), iteration 2 collapses ("lee\\n\\n\\nTop \\nTop"), and iterations 3–100
converge byte-for-byte to a degenerate topic-projector attractor. A 16-prompt
multi-format sweep makes the cliff universal: **16/16 prompts coherent at
iteration 1 collapse at iteration 2**, with a low-dimensional attractor
vocabulary (`"lee"`, `"ício"`, `"RESS"`, newlines) and a second failure mode
(immediate stop-token / empty output) on two prompts. The fixed point
demonstrably *exists* and is numerically reachable — the max-norm change between
successive iterates falls to f32 noise within 100 iterations — at only $1.41\times$
the compute of one step, so the cliff is the phenomenon, not an artifact of
under-iteration or a compute budget.

![The Picard cliff. Iterating Phi-3-mini's attention past one step destroys language coherence at iteration 2 (all 16 prompts collapse), while a well-trained 4-layer control under the identical protocol retains 91 % top-1 agreement and is stable through 100 iterations. Iteration 1 equals the standard kernel by construction. The two series use different but comparable "behavior retained" metrics (see text); the qualitative gap is the result.](fig-e45-cliff.pdf){width=82%}

A disentanglement control separates scale from the phenomenon. A well-trained
4-layer / 64-hidden toy transformer (same RMSNorm + SwiGLU + RoPE family) under
the identical protocol stays in the "match" regime — 91 % top-1 retained at
iteration 2, stable through iteration 100 — where Phi-3-mini retains no coherent
output (Figure 2). The defensible claim is therefore stronger than "the
hypothesis is false": *a trained discrete-attention layer's distance from its own
Picard fixed point grows with model scale and training breadth; brittleness to
attention iteration is a property of well-trained large LMs specifically, not of
transformers in general.* This relates the inference-time behavior of trained
attention to the equilibrium view of attention as a Hopfield update [4] and to
deep equilibrium models [5], and shows the trained operator is qualitatively
unlike its own one-step fixed point. A per-layer single-layer-ablation (Phase 3)
to adjudicate the two pre-registered per-layer hypotheses is the natural
follow-up and is not yet run.

## 4. Limitations and threats to validity

- **Per-kernel error budgets are declared, not yet CI-asserted.** The bounds of
  §3.2 are documented and the end-to-end HF parity is enforced on every user
  click, but the runner that checks each kernel against a CPU f64 reference does
  not yet exist; the per-kernel ULP figures are therefore design budgets, not
  measurements.
- **Cross-vendor parity is under-sampled.** The suite has been exercised on Apple
  M-series and one NVIDIA workstation. WebGPU f16 behavior can differ across
  adapters; the cross-vendor matrix is open, and users are asked to report
  fingerprint plus parity numbers. This is the main external-validity threat.
- **One architecture.** The engine is specialized to Phi-3-mini and the MLC
  `q4f16_1` format, whose RoPE base/theta-scaling it inherits; generalization to
  other architectures is unimplemented.
- **E45 metric heterogeneity.** Figure 2 compares a coherent-output rate (Phi-3,
  16 prompts) against top-1 agreement (control); the two are not the same
  statistic. They are both "fraction of normal behavior retained," and the
  qualitative gap (0 % vs 91 %) is far larger than any reasonable metric
  reconciliation, but a single shared metric across scales is future work.
- **E45 per-layer attribution is deferred.** The output cliff at iteration 2 is
  sharp enough that the two pre-registered per-layer hypotheses (deepest-middle
  vs high-entropy layers as catastrophic) are both partially refuted at
  model-level granularity; the single-layer ablation that would pick a winner is
  Phase 3, unrun.
- **Not a training or safety tool.** No claim is made about the model's fairness,
  alignment, or safety — only that what is rendered matches the reference forward
  pass within the stated bounds.

## 5. Related work

Prior interactive transformer visualizations trade scale for visibility or vice
versa (Table 1): LLM-Viz (real 3D, live tensors, $\sim\!10^3$-parameter toy),
Transformer Explainer [1] (real GPT-2-small, 124 M, 2D, partial tensors), BertViz
[2] (real attention only, 2D, notebook), and in-browser runtimes such as WebLLM
(real multi-billion-parameter models, no internal visibility). Across the axes
*real model · multi-billion scale · browser · 3D · all live tensors ·
reference-validated*, neuropulse is, to our knowledge, the first to satisfy all
six simultaneously. The interpretability result of §3.3 connects to
equilibrium/fixed-point views of attention: modern Hopfield networks identify the
attention update with a Hopfield retrieval step whose fixed points are energy
minima [4], and deep equilibrium models train networks *defined* by a fixed-point
solve [5]. We instead probe the *inference-time* self-consistency point of an
already-trained discrete-attention model and find it qualitatively unlike the
one-step operator, with a severity that grows with scale.

## 6. Reproducibility and software availability

Source: `github.com/abgnydn/neuropulse` (MIT); live at `neuropulse.live`. The
WGSL kernels are in `src/engine/shaders/`; canonical architecture constants and
the derived dispatch formulas in `src/engine/compiler.ts` and
`src/engine/phi3-facts.ts`; the precision matrix, tolerances, and per-kernel
budgets in `METHODS.md`; the pre-registered prediction log in `PREDICTIONS.md`.
The HF parity suite runs from the demo's validate button on any WebGPU-capable
Chromium and regenerates its reference via `tools/dump_phi3_reference.py`; the
E45 artifacts (iteration sweep, multi-prompt sweep, small-model control) are
committed under `tests/results/` and `tools/small-model-control/`. Every
documentation number is gated against source by `npm run verify` on each commit.
The Butterfly context-compaction study that also ran on this platform has its own
repository (`github.com/abgnydn/butterfly`) and deposit.

**Data availability.** All experimental artifacts are committed to the
repository; the figures in this manuscript are reproducible from those artifacts.
Model weights are Microsoft's Phi-3-mini redistributed by MLC and are not part of
this deposit.

**Author contributions.** A.B.G. is the sole author and conducted all design,
implementation, experiments, and writing.

**Generative-AI disclosure.** Portions of the software, documentation, and this
manuscript were drafted with a large language model used as a coding and writing
aid; all output was author-reviewed, correctness was enforced by the HF parity
suite and the commit-time claim checks, and every quantitative claim is traceable
to `METHODS.md`, `phi3-facts.ts`, or a committed experiment artifact.

**Statements.** Sole author; no competing interests; no external funding.

## 7. Conclusion

A full-scale transformer forward pass can be rendered one-to-one, in real time,
in a browser tab — every element a real tensor read from the buffers the model
just wrote, every "one-to-one" claim falsifiable against a HuggingFace reference
the reader runs on their own GPU. Closing the gap between toy-but-visible and
real-but-opaque yields both a pedagogical artifact and an interpretability
substrate: experiment E45 used it to falsify a continuous-attention
self-consistency hypothesis and, via a trained small-model control, to attribute
the resulting brittleness to scale and training breadth rather than to
transformers in general. The platform's discipline — derived constants, a
pre-registered prediction log, and a validation suite that fails loud — is as
much the contribution as the rendering.

---

## References

[1] A. Cho, G. Kim, A. Karpekov, A. Helbling, Z. J. Wang, S. Lee, B. Hoover, and
D. H. Chau. "Transformer Explainer: Interactive Learning of Text-Generative
Models." *arXiv:2408.04619*, 2024. <https://arxiv.org/abs/2408.04619>

[2] J. Vig. "A Multiscale Visualization of Attention in the Transformer Model."
*Proc. 57th Annual Meeting of the ACL: System Demonstrations*, 2019, pp. 37–42.
*arXiv:1906.05714*. <https://aclanthology.org/P19-3007/>

[3] M. Abdin et al. (Microsoft). "Phi-3 Technical Report: A Highly Capable
Language Model Locally on Your Phone." *arXiv:2404.14219*, 2024.
<https://arxiv.org/abs/2404.14219>

[4] H. Ramsauer, B. Schäfl, J. Lehner, P. Seidl, M. Widrich, T. Adler, L. Gruber,
M. Holzleitner, M. Pavlović, G. K. Sandve, V. Greiff, D. Kreil, M. Kopp, G.
Klambauer, J. Brandstetter, and S. Hochreiter. "Hopfield Networks is All You
Need." *Int. Conf. on Learning Representations (ICLR)*, 2021. *arXiv:2008.02217*.
<https://arxiv.org/abs/2008.02217>

[5] S. Bai, J. Z. Kolter, and V. Koltun. "Deep Equilibrium Models." *Advances in
Neural Information Processing Systems (NeurIPS)*, 2019. *arXiv:1909.01377*.
<https://arxiv.org/abs/1909.01377>

[6] W3C GPU for the Web Working Group. "WebGPU." W3C Candidate Recommendation.
<https://www.w3.org/TR/webgpu/>

---

*Draft v0.2. Built from the committed engine, `METHODS.md` tolerances, and the
`PREDICTIONS.md` / `tests/results/` E45 artifacts; every documentation number is
gated against source by `npm run verify`. To be converted to a venue LaTeX
template before submission to a visualization (IEEE VIS / EuroVis), ML-systems, or
interpretability venue.*

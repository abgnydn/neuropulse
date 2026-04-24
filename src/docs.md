# Neuropulse reference

These are the runtime reference docs loaded into Phi-3's context when the
user clicks **Ask**. Keep each section short — the whole file budgets
~1,500 tokens so answers have room to breathe.

## What Neuropulse is

Neuropulse is a live 3D visualization of a real Phi-3-mini forward pass,
running entirely on the user's GPU in a browser tab via WebGPU. Every
glowing element in the 3D scene is a direct readout of a real GPU buffer —
no mockups, no approximations. When the model thinks about a prompt, the
user watches *it* think.

Published by Ahmet Barış Günaydın at https://neuropulse.live — companion
piece to the research project at https://zerotvm.com.

## The model being visualized

- **Phi-3-mini**, the q4f16_1 quantization (3.8 billion parameters,
  stored as 4-bit integers with fp16 scales; about 2 GB of weights).
- **32 transformer layers**, **3,072-dim residual stream**, **32 attention
  heads per layer** (96 dims each), **8,192-dim FFN** with SwiGLU gating,
  **paged KV cache** (16 tokens per page, up to 257 pages).
- Vocabulary: 32,064 tokens. Chat template: `<|system|>…<|end|>\n<|user|>…<|end|>\n<|assistant|>`.
- 292 WebGPU dispatches per output token. 13 pipelines. 22 GPU buffers.

## Key concepts

**Token** — A fragment of text (usually 3–4 characters). Phi-3's vocabulary
holds 32,064 possible tokens. Generation picks one token at a time.

**Residual stream** — The 3,072-dim vector flowing through the model. Each
layer *reads from* and *adds back to* it. Information accumulates rather
than being replaced. In the 3D scene this is the central axis.

**Attention head** — A 96-dim projection deciding which past tokens to
focus on. Phi-3 has 32 heads per layer × 32 layers = 1,024 heads total,
each specialized for a different kind of pattern.

**Feed-forward network (FFN / MLP)** — Expands residual to 8,192 dims,
applies SiLU gating, projects back to 3,072. Most compute happens here;
it's also where the model's *world knowledge* is thought to be stored.

**KV cache** — Stored keys and values from past tokens per layer, so
attention doesn't recompute them. Grows by one slot per token per layer.

**Softmax** — Turns raw logits into probabilities: exponentiate, then
normalize to sum to 1. Peaky softmax = confident. Flat = guessing.

**Quantization (q4f16_1)** — Weights stored as 4-bit integers with fp16
scales. Roughly 4× smaller than fp16, with near-zero quality loss on
Phi-3. The reason a 3.8B model fits in a browser tab.

**Dispatch** — One WebGPU kernel invocation. Phi-3 needs 292 dispatches
per output token; each one is visible as its tensor lights up.

## What the 3D scene shows

- **32 layer rings** — one transformer block each; brightness tracks the
  post-attention + post-FFN residual norm.
- **1,024 cyan attention-head neurons** — each lights up proportional to
  its head's live output magnitude.
- **Amber FFN slab** per layer — the 8,192-neuron expansion; pulses as
  the MLP activates.
- **3,072-point residual stream** — points placed by PCA of the model's
  own layer-0 qkv_proj weights, so functionally related dims sit near
  each other. Brightness = live residual value at that dim.
- **KV cache strips** — growing memory of past tokens per layer.
- **LM head** — final projection to 32,064 vocab logits.

## Layer narratives (rough interpretability priors)

- **Layers 0–3**: token identity, surface features (case, punctuation,
  bigrams). Attention heads act as position detectors.
- **Layers 4–13**: syntax and local structure. Induction heads, POS
  disambiguation, phrase-level features. Attention spans widen.
- **Layers 14–22**: semantic concepts. Long-range attention, coreference,
  fact retrieval in the FFN. Residual norm peaks here.
- **Layers 23–29**: task-specific circuits. Formatting, tone, style.
  Copy heads, output-shaping FFN.
- **Layers 30–31**: final block. Last RMSNorm → LM head → softmax.

These are broad priors from interpretability research, not tight claims
about Phi-3-mini specifically.

## App features

**Universe view** — one 3D universe with the model floating in it.
Glowing cyan **pips** orbit the model at anchor positions. Click a pip
to expand it into a full glass card; click × to collapse. Drag to orbit
the camera, wheel to zoom, right-drag to pan.

**Journey** — scroll-driven cinematic flythrough of all 32 layers. Space
to auto-play (~60s per full journey). Arrow keys to step. The camera
dollies behind the "signal position" as it moves through the model.

**Panels** (each has an `i` info button):
- Output, Top-K, Confidence, KV Cache (right cluster)
- Head Activity heatmap (above the model)
- Residual Norm, Layer Δ, Residual Strip, Raw GPU State (left cluster)
- Prompt input + token strip (floating below / above)

**Keyboard**: `?` = glossary · `space` = play/pause journey · arrows = step
layers · `P` or `Tab` = toggle all panels · `R` = reset camera · `Esc` =
close overlays · drag = orbit · wheel = zoom.

## Performance and honesty

- The HF fp16 cross-validation suite ships with the app and can be run
  from the wrench icon — it confirms identical top-1 tokens vs reference
  Phi-3.
- First load downloads ~2 GB of weights to the GPU. Subsequent visits
  are near-instant thanks to OPFS caching.
- Zero API calls. Zero server. Zero telemetry. Close the tab, inference
  stops; nothing leaves the machine.

## Limitations to be honest about

- Phi-3-mini is 3.8 B parameters — strong on transformer concepts and
  this app's features, weaker on very specific interpretability research
  citations. If you don't know something, say so.
- The 3D scene is a *faithful reduction* of high-dimensional tensors to
  visible elements. Brightness faithfully tracks magnitudes, but the
  particular XY layout (PCA of layer-0 qkv_proj) is an editorial choice,
  not a universal truth about the model.
- This app visualizes one forward pass — not training, not fine-tuning,
  not loss landscapes.

You are running inside this app right now. The user can literally see
your current layer's activations pulse as you generate each word of
your answer.

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="public/hero-dark.svg">
  <img alt="Neuropulse — real-time 1:1 visualization of a full-scale LLM forward pass" src="public/hero-light.svg" width="100%">
</picture>

<br>
<br>

### The first accurate real-time visualization of a full-scale LLM forward pass.

3.8 billion parameters. Your GPU. Your browser. Every tensor rendered 1:1.<br>
No server. No API key. No fakery.

<br>

[![License: MIT](https://img.shields.io/badge/license-MIT-1f2328?style=flat-square)](LICENSE)
[![Model: Phi-3-mini](https://img.shields.io/badge/model-Phi--3--mini-1f2328?style=flat-square)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
[![Runtime: WebGPU](https://img.shields.io/badge/runtime-WebGPU-1f2328?style=flat-square)](https://www.w3.org/TR/webgpu/)
[![Validated: HF reference](https://img.shields.io/badge/validated-HF%20reference-1f2328?style=flat-square)](#validation)
[![GitHub stars](https://img.shields.io/github/stars/abgnydn/neuropulse?style=flat-square&color=1f2328)](https://github.com/abgnydn/neuropulse/stargazers)

<br>

[**Launch Demo**](https://neuropulse.live/app/) &nbsp;·&nbsp; [**Read the Essay**](https://neuropulse.live/) &nbsp;·&nbsp; [**Methods**](METHODS.md) &nbsp;·&nbsp; [**Predictions**](PREDICTIONS.md)

<br>

</div>

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="public/stats-dark.svg">
  <img alt="3.8B parameters · 11 WGSL kernels · 22 GPU buffers · 292 dispatches per token · 32 transformer layers · 0 frameworks in inference" src="public/stats-light.svg" width="100%">
</picture>

</div>

<br>

> [!NOTE]
> **Strict 1:1.** Every pixel on screen is a function of a real GPU tensor. The brightness of each point **is** the activation value. The lines between attention heads **are** the real attention weights. The token probabilities rolling across the screen **are** the actual logits from the final layer.

<br>

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="public/preview-dark.svg">
  <img alt="Neuropulse running in the browser — residual stream cluster, attention rays, side panels for ablation and per-head attention, logit lens, and prompt input" src="public/preview-light.svg" width="100%">
</picture>

<sub>A stylized snapshot of the live in-browser visualizer. Open <a href="https://neuropulse.live/app/">neuropulse.live/app/</a> to drive your own.</sub>

</div>

<br>

## The problem

Every "AI visualization" you've seen online is **decoration**.

Animated dots pulsing to a fake rhythm. Particle systems that aren't connected to anything real. A beautiful metaphor with no model behind the curtain. You walk away thinking you saw how an LLM works. You didn't — you saw how a designer *imagines* it works.

Neuropulse is the opposite. Type a prompt. Watch 3.8 billion parameters process it. Nothing is interpolated. Nothing is smoothed. Nothing is made up.

<br>

## How it compares

Two separate worlds existed — visualization tools that run toy models, and inference engines with zero internal visibility. Nothing connected them.

<div align="center">

| | Real model | Scale | Browser | 3D | Live tensors | Validated |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| [Brendan Bycroft's LLM Viz](https://bbycroft.net/llm) | Toy (sorts ABC) | ~1K | Yes | Yes | Yes | No |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | GPT-2 small | 124M | Yes | No | Partial | No |
| [BertViz](https://github.com/jessevig/bertviz) | HF models | Any | No | No | Attn only | No |
| [WebLLM](https://webllm.mlc.ai/) | Yes | Multi-B | Yes | — | — | — |
| **Neuropulse** | **Phi-3-mini** | **3.8B** | **Yes** | **Yes** | **All** | **HF ref** |

</div>

<br>

## What you're actually watching

The 3D scene is not a metaphor. Each element maps to a named tensor in Phi-3-mini's compute graph.

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="public/scene-dark.svg">
  <img alt="The four elements of the Neuropulse scene: residual stream, attention heads, gated MLP, and token strip" src="public/scene-light.svg" width="100%">
</picture>

</div>

The layout isn't arbitrary. Residual-stream positions come from PCA of the model's own layer-0 `qkv_proj` weight matrix — dimensions that get read into attention together cluster together. The geometry is shaped by the model, not by a designer.

<br>

## Architecture

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="public/architecture-dark.svg">
  <img alt="Neuropulse architecture: prompt → tokenizer → embedding → 32× transformer block → LM head → sampler → token (auto-regressive loop), with a parallel render path reading activations from the same GPU buffers" src="public/architecture-light.svg" width="100%">
</picture>

</div>

Inference and visualization share the same GPU buffers. The renderer doesn't recompute anything — it reads the values the model already produced.

<br>

## Validation

"Strict 1:1" is a strong claim, so it has to be falsifiable. Neuropulse ships with a built-in test suite that diffs the WebGPU forward pass against a reference HuggingFace fp16 Phi-3-mini.

> [!TIP]
> Click the wrench icon in the demo. The numbers from **your** GPU print to **your** browser console in under a minute. No setup, no install — your machine is the test rig.

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="public/validation-dark.svg">
  <img alt="Validation suite — 6 tests against a HuggingFace fp16 reference, each with its bound and observed value" src="public/validation-light.svg" width="100%">
</picture>

</div>

Expected result: tiny deltas at hidden-state level (int4 quantization cost, not implementation drift) and identical top-1 tokens vs the fp16 reference. That last bit is the bar that matters.

A second layer of validation runs in CI: `npm run verify` cross-checks documented claims (layer count, kernel count, dispatch counts, keyboard shortcuts) against the actual source. If the README drifts from the code, the build fails.

<br>

## The stack

Four pieces. No frameworks in the inference path. No dependency soup.

<table>
<tr>
<td width="50%" valign="top">

**Inference**

```text
WebGPU compute shaders
11 WGSL kernels
22 GPU buffers
292 dispatches / token
q4f16_1 quantization
```

**Weights**

```text
Phi-3-mini (MLC, HuggingFace)
Browser Cache API (~2 GB)
Instant reload on return
```

</td>
<td width="50%" valign="top">

**Rendering**

```text
Three.js (WebGLRenderer)
Strict 1:1 tensor mapping
PCA-derived 3D layout
Audio sonification
```

**UI**

```text
Vanilla TypeScript
Zero frameworks
Vite (dev/build only)
```

</td>
</tr>
</table>

<details>
<summary><strong>Source tree</strong></summary>

```
src/
├── engine/
│   ├── shaders/               # WGSL compute kernels
│   │   ├── attention.wgsl         # multi-head attention
│   │   ├── attention_scores.wgsl  # QK^T / sqrt(d)
│   │   ├── rope.wgsl              # rotary position embeddings
│   │   ├── int4_matmul.wgsl       # quantized matrix multiply (f16 accum)
│   │   ├── int4_matmul_f32.wgsl   # quantized matrix multiply (f32 accum)
│   │   ├── fused_ffn.wgsl         # gated MLP (SiLU + gate + down)
│   │   ├── rms_norm.wgsl          # RMS layer normalization
│   │   ├── embedding.wgsl         # token embedding lookup
│   │   ├── kv_append.wgsl         # KV cache append
│   │   ├── add_norm.wgsl          # residual add + norm
│   │   └── argmax.wgsl            # greedy token selection
│   ├── compiler.ts            # pipeline compilation + buffer mgmt
│   ├── inference.ts           # forward pass orchestration
│   ├── tokenizer.ts           # BPE tokenizer (zero deps)
│   ├── weight-loader.ts       # Cache API streaming + progress
│   └── activation-reducer.ts  # tensor readback for visualization
├── visualizer.ts              # Three.js — strict 1:1 tensor rendering
├── audio.ts                   # sonification of live tensor data
└── main.ts                    # app shell + UI + interaction
```

</details>

<br>

## Run locally

<table>
<tr>
<td width="55%" valign="top">

```bash
git clone https://github.com/abgnydn/neuropulse.git
cd neuropulse
npm install
npm run dev
```

Open **http://localhost:5173/app/** in Chrome, Edge, or Safari Technology Preview.

First visit downloads ~2 GB of model weights into the browser cache. Every visit after that loads instantly.

</td>
<td width="45%" valign="top">

**Requirements**

|  | Minimum | Recommended |
|:---|:---|:---|
| Browser | Chrome 113+ | latest Chrome |
| GPU memory | ~2 GB free | 4+ GB |
| Disk | ~2 GB cache | — |

</td>
</tr>
</table>

> [!IMPORTANT]
> WebGPU is required. Firefox does not ship WebGPU on stable yet; use Chrome, Edge, or Safari Technology Preview.

<br>

## Acknowledgments

- **[Microsoft Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)** — the model under the glass.
- **[MLC](https://mlc.ai/)** — q4f16_1 weight format and the WebGPU inference patterns this project builds on.
- **[Three.js](https://threejs.org/)** — the renderer.
- **[Brendan Bycroft's LLM Viz](https://bbycroft.net/llm)** — proved a transformer could be *seen*. This project asks: can it be seen at scale?

<br>

## Cite

```bibtex
@software{gunaydin_neuropulse_2026,
  author  = {Günaydın, Ahmet Barış},
  title   = {Neuropulse: Real-Time 1:1 Visualization of a Full-Scale LLM
             Forward Pass in the Browser},
  year    = {2026},
  url     = {https://github.com/abgnydn/neuropulse}
}
```

<br>

## License

[MIT](LICENSE) — do whatever you want, just keep the copyright notice.

<br>

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="public/monogram-dark.svg">
  <img alt="neuropulse — see what your LLM is doing" src="public/monogram-light.svg" width="100%">
</picture>

<sub>Built by <a href="https://github.com/abgnydn">Ahmet Barış Günaydın</a></sub>

</div>

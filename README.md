# Neuropulse

**The first accurate real-time visualization of a full-scale LLM forward pass.**

Phi-3-mini (3.8B parameters) running entirely on your GPU, in your browser, with every neuron, attention head, and activation rendered 1:1 from live WebGPU tensors. No server. No API key. No fakery.

**[Launch the demo](https://neural-pulse-two.vercel.app/app/)** | **[Read the essay](https://neural-pulse-two.vercel.app/)**

---

## Why this is different

Every "AI visualization" you've seen online is decoration. Animated dots pulsing to a fake rhythm. Particle systems that aren't connected to anything. A metaphor with no model behind the curtain.

Neuropulse is the opposite. Type a prompt and watch a real 3.8-billion-parameter transformer think. The brightness of each point **is** the activation value. The connections between attention heads **are** the real attention weights. Nothing is interpolated, smoothed, or made up.

### The landscape

| | Real model? | Scale | Browser? | 3D? | Live tensors? | Validated? |
|---|---|---|---|---|---|---|
| [Brendan Bycroft's LLM Viz](https://bbycroft.net/llm) | Toy sorting model | ~1K params | Yes | Yes | Yes (toy) | No |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | GPT-2 small | 124M | Yes | No (2D) | Partial | No |
| [BertViz](https://github.com/jessevig/bertviz) | Real HF models | Any | No (Jupyter) | No | Attention only | No |
| [WebLLM](https://webllm.mlc.ai/) | Yes | Multi-B | Yes | No viz | N/A | N/A |
| **Neuropulse** | **Yes** | **3.8B** | **Yes** | **Yes** | **All tensors** | **Yes (HF ref)** |

Two separate worlds existed — visualization tools that run toy models, and WebGPU inference engines with zero internal visibility. Neuropulse connects them.

---

## What you're watching

The 3D scene is not a metaphor. Each element corresponds to a specific tensor in Phi-3-mini's compute graph:

- **Residual stream** — 3,072 points laid out by PCA of the model's own `qkv_proj` weights. Brightness = live value of that residual dimension.
- **Attention heads** — 32 heads per layer, 32 layers. Each head's output magnitude drives its visual intensity.
- **FFN layers** — gated MLP activations rendered from the actual intermediate buffer.
- **Token strip** — input and generated tokens shown in real time as they flow through the model.

The PCA layout means dimensions that get read into attention together end up near each other. The geometry is shaped by the model itself, not by a designer.

---

## Validation

"Strict 1:1" is a strong claim, so it has to be falsifiable. Neuropulse ships with a built-in test suite that diffs the WebGPU implementation against a reference HuggingFace fp16 Phi-3-mini:

| Check | What it verifies |
|---|---|
| Tokenizer | GPU input IDs match HF byte-for-byte on every prompt |
| Hidden states | Full 3,072-dim residual diffed at layers 0, 4, 8, 12, 16, 20, 24, 28, 31 |
| Attention | Online softmax cross-checked against explicit-softmax reference path |
| Logits | Top-k probabilities + Jensen-Shannon divergence vs HF on a 15-prompt sweep |
| Long context | 290-token prompt, 10 decode steps, top-1 matched against HF |
| Sampler | 5,000-sample empirical distribution vs softmax, JSD < 1e-2 |

Click the wrench icon inside the demo to run it yourself. The numbers from your GPU print to your browser console.

---

## The stack

Four pieces. No frameworks in the inference path.

| Component | Details |
|---|---|
| **WebGPU compute** | 11 WGSL shaders, 22 buffers, 292 dispatches per token. `q4f16_1` quantization. Hand-written attention and FFN kernels. |
| **Phi-3-mini weights** | Same `mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC` weights from HuggingFace. Cached in browser Cache API (~2 GB first load, instant after). |
| **Three.js scene** | Plain `WebGLRenderer`. No bloom, no particles, no decorative shaders. Every pixel pulls from a real tensor every frame. |
| **PCA layout** | Residual points placed by PCA of layer 0 `qkv_proj.weight` columns. FFN points by PCA of `down_proj.weight`. |

```
src/
├── engine/
│   ├── shaders/          # 11 WGSL compute shaders
│   │   ├── attention.wgsl
│   │   ├── attention_scores.wgsl
│   │   ├── int4_matmul.wgsl
│   │   ├── fused_ffn.wgsl
│   │   ├── rms_norm.wgsl
│   │   ├── rope.wgsl
│   │   ├── embedding.wgsl
│   │   ├── kv_append.wgsl
│   │   ├── add_norm.wgsl
│   │   ├── argmax.wgsl
│   │   └── int4_matmul_f32.wgsl
│   ├── compiler.ts       # Pipeline compilation + buffer management
│   ├── inference.ts       # Forward pass orchestration
│   ├── tokenizer.ts       # BPE tokenizer (no dependencies)
│   ├── weight-loader.ts   # Cache API weight loading with progress
│   └── activation-reducer.ts
├── visualizer.ts          # Three.js scene — strict 1:1 tensor rendering
├── audio.ts               # Sonification of live tensor data
└── main.ts                # App shell + UI
```

---

## Run locally

```bash
git clone https://github.com/abgnydn/neural-pulse.git
cd neural-pulse
npm install
npm run dev
```

Open `http://localhost:5173/app/` in Chrome, Edge, or Safari Technology Preview (WebGPU required).

First visit downloads ~2 GB of model weights into the browser cache. Subsequent loads are instant.

---

## Requirements

- A browser with WebGPU support (Chrome 113+, Edge 113+, Safari TP)
- A GPU with at least 4 GB VRAM
- ~2 GB free disk space for weight cache

---

## License

MIT

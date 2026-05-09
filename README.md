<div align="center">

<br>

# n e u r o p u l s e

### The first accurate real-time visualization of a full-scale LLM forward pass.

3.8 billion parameters. Your GPU. Your browser. Every tensor rendered 1:1.<br>
No server. No API key. No fakery.

<br>

[**Launch Demo**](https://neuropulse.live/app/) &nbsp;&nbsp;|&nbsp;&nbsp; [**Read the Essay**](https://neuropulse.live/)

<br>

<!-- Drop a screenshot here: ![Neuropulse](docs/screenshot.png) -->

<sub>Phi-3-mini running live in Chrome — every glow is a real activation value read back from WebGPU</sub>

<br>

---

</div>

<br>

## The problem

Every "AI visualization" you've seen online is **decoration**.

Animated dots pulsing to a fake rhythm. Particle systems that aren't connected to anything real. A beautiful metaphor with no model behind the curtain. You walk away thinking you saw how an LLM works. You didn't — you saw how a designer *imagines* it works.

Neuropulse is the opposite.

Type a prompt. Watch 3.8 billion parameters process it. The brightness of each point **is** the activation value. The lines between attention heads **are** the real attention weights. The token probabilities rolling across the screen **are** the actual logits from the final layer. Nothing is interpolated. Nothing is smoothed. Nothing is made up.

> **Strict 1:1.** Every pixel on screen is a function of a real GPU tensor.

<br>

## How it compares

Two separate worlds existed — visualization tools that run toy models, and inference engines with zero internal visibility. Nothing connected them.

| | Real model | Scale | Browser | 3D | Live tensors | Validated |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| [Brendan Bycroft's LLM Viz](https://bbycroft.net/llm) | Toy (sorts ABC) | ~1K | Yes | Yes | Yes | No |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | GPT-2 small | 124M | Yes | No | Partial | No |
| [BertViz](https://github.com/jessevig/bertviz) | HF models | Any | No | No | Attn only | No |
| [WebLLM](https://webllm.mlc.ai/) | Yes | Multi-B | Yes | — | — | — |
| **Neuropulse** | **Phi-3-mini** | **3.8B** | **Yes** | **Yes** | **All** | **HF ref** |

<br>

## What you're actually watching

The 3D scene is not a metaphor. Each element maps to a named tensor in Phi-3-mini's compute graph.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   RESIDUAL STREAM                                               │
│   3,072 points — PCA of qkv_proj weights                        │
│   brightness = live activation value per dimension               │
│                                                                 │
│   ATTENTION HEADS                                               │
│   32 heads x 32 layers = 1,024 elements                         │
│   intensity = output magnitude of each head                      │
│                                                                 │
│   FFN (GATED MLP)                                               │
│   intermediate activations from the actual buffer                │
│   gate values drive the visual pulse                             │
│                                                                 │
│   TOKEN STRIP                                                   │
│   input + generated tokens rendered as they flow                 │
│   through the forward pass in real time                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The 3D layout isn't arbitrary. Residual stream positions come from PCA of the model's own layer-0 `qkv_proj` weight matrix — dimensions that get read into attention together cluster together. The geometry is shaped by the model, not by a designer.

<br>

## Validation

"Strict 1:1" is a strong claim, so it has to be falsifiable.

Neuropulse ships with a built-in test suite that diffs the WebGPU forward pass against a reference HuggingFace fp16 Phi-3-mini. Click the wrench icon in the demo to run it — the numbers from **your** GPU print to **your** browser console.

```
═══ Validation Suite ═══════════════════════════════════════════════

 [1]  Tokenizer        GPU input IDs match HF byte-for-byte
 [2]  Hidden states    3,072-dim residual diffed at 9 layer checkpoints
 [3]  Attention        Online softmax vs explicit-softmax reference
 [4]  Logits           Top-k probs + JS divergence, 15-prompt sweep
 [5]  Long context     290 tokens in, 10 decode steps, top-1 match
 [6]  Sampler          5,000-sample distribution vs softmax, JSD < 1e-2

════════════════════════════════════════════════════════════════════
```

Expected result: tiny deltas at hidden-state level (int4 quantization cost, not implementation drift) and identical top-1 tokens vs the fp16 reference. That last bit is the bar that matters — and you can verify it yourself, on your own machine, in under a minute.

<br>

## The stack

Four pieces. No frameworks in the inference path. No dependency soup.

```
INFERENCE                          RENDERING
─────────────────────────          ─────────────────────────
WebGPU compute shaders             Three.js (WebGLRenderer)
11 WGSL kernels                    Strict 1:1 tensor mapping
22 GPU buffers                     PCA-derived 3D layout
292 dispatches / token             Audio sonification
q4f16_1 quantization               
                                   
WEIGHTS                            UI
─────────────────────────          ─────────────────────────
Phi-3-mini (MLC, HuggingFace)     Vanilla TypeScript
Browser Cache API (~2 GB)          Zero frameworks
Instant reload on return           Vite (dev/build only)
```

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

```bash
git clone https://github.com/abgnydn/neuropulse.git
cd neuropulse
npm install
npm run dev
```

Open **http://localhost:5173/app/** in Chrome, Edge, or Safari Technology Preview.

First visit downloads ~2 GB of model weights into the browser cache. Every visit after that loads instantly.

<br>

## Requirements

| | Minimum | Recommended |
|:---|:---|:---|
| **Browser** | Chrome 113+, Edge 113+, Safari Technology Preview | latest desktop Chrome |
| **GPU memory** | ~2 GB free (the weight payload) | 4+ GB for headroom |
| **Disk** | ~2 GB for weight cache (OPFS or Cache API) | — |

<br>

## License

MIT

<br>

---

<div align="center">
<sub>Built by <a href="https://github.com/abgnydn">Ahmet Baris Gunaydin</a></sub>
</div>

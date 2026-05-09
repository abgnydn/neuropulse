# neuropulse.live

## Goal

A live 3D visualization of Phi-3-mini's forward pass. Every tensor rendered
1:1 from real WebGPU activations — no approximation, no sampling, no server.
The pedagogical counterpart to `zerotvm.com`: same model weights, same
kernels, but every intermediate activation is read back and rendered in real
time so you can *watch* a 3.8B transformer think.

## Architecture

Vite-built static site with a 3D Three.js visualizer on top of a hand-written
Phi-3 inference engine in WGSL. Two entrypoints: a marketing essay at `/`
(index.html) and the live demo at `/app/` (app/index.html).

- `src/main.ts` — app bootstrap: tokenizer, weight loader, inference loop,
  visualizer wiring, panel/mode/keyboard wiring.
- `src/visualizer.ts` — Three.js scene, layer/token/activation rendering.
- `src/journey.ts` — keyboard-driven cinematic flythrough (no longer
  scroll-driven; wheel is reserved for OrbitControls camera zoom).
- `src/audio.ts` — ambient sonification driven by activation magnitudes.
- `src/storyteller.ts` / `src/butterfly-mode.ts` / `src/tours.ts` /
  `src/spatial-panels.ts` — additional UX surfaces (kid-mode narration,
  butterfly compaction demo, guided tours, anchor projection).
- `src/engine/weight-loader.ts` — fetches Phi-3 weights from MLC's
  `q4f16_1` ndarray-cache release (NOT GGUF) and serves them via a tiered
  cache: OPFS first (per-origin persistent FS, fastest for ~200 MB shards),
  then any-Cache-API-bucket scan (catches WebLLM-prepopulated and the
  legacy `neural-pulse-phi3-weights` bucket transparently), then a
  CF-Pages edge proxy at `/hf/...`, then HuggingFace direct as last
  resort. Exports `getStoredWeightStats` + `clearStoredWeights` for the
  in-app storage modal. Cache bucket name: `neuropulse-phi3-weights`.
- `src/engine/inference.ts` — per-token decode loop. Headless fast path:
  292 dispatches/token. Visualized path adds ~56 more (per-layer
  attention-score readback + 8 logit-lens probes).
- `src/engine/compiler.ts` — stitches the 11 WGSL kernels into a per-layer
  compute pipeline. Holds the canonical `PHI3` architecture constants.
- `src/engine/phi3-facts.ts` — single source of truth for user-facing
  numbers (layers, heads, dispatch counts, kernel count, weight size).
  Anything quoted in copy must resolve here. `tools/verify-claims.mjs`
  greps the docs and asserts every number matches.
- `src/engine/tokenizer.ts` — BPE tokenizer loaded from the same tiered
  cache as the weights.
- `src/engine/shaders/*.wgsl` — 11 hand-written kernels: add_norm,
  argmax, attention, attention_scores, embedding, fused_ffn, int4_matmul,
  int4_matmul_f32, kv_append, rms_norm, rope.
- `src/lib/sites.ts` — synced from `~/sites-shared/sites.ts`.
- `index.html` — marketing essay; carries the JSON-LD `Person` with
  `sameAs` driven by `SAME_AS` (edit in `~/sites-shared/sites.ts`).
- `app/index.html` — live demo host page. Currently has NO JSON-LD;
  the demo shell isn't intended for SEO indexing.

## Commands

```bash
npm install
npm run dev          # Vite dev server
npm run build        # Vite production build → dist/
npm run preview      # preview built dist/
npm run typecheck    # tsc --noEmit
npm run verify       # claim-vs-code check + dead-shortcut check
npm run check        # typecheck + verify + build
```

Deploy: `node ~/sites-shared/deploy.mjs neuropulse` (CF Pages, project
`neuropulse`).

## Empirical-lab gates

The project aspires to lab-grade reproducibility. Three gates exist:

1. **`tools/verify-claims.mjs`** — every numeric claim in `*.md` / `*.html`
   is regex-extracted and validated against `compiler.ts` PHI3 + the shader
   directory contents. Wired into `npm run verify` and `npm run check`.
2. **`tools/check-shortcuts.mjs`** — diffs keyboard shortcuts wired in
   `main.ts` against shortcuts advertised in HUD / glossary copy. Catches
   the "scroll advances Journey" class of doc/code drift.
3. **`tools/reference/parity.json`** — pinned HF cross-validation artifact:
   weight URLs + SHA-256s, validation prompts, expected per-layer L2/cosine
   bounds. The in-app validation suite (`accurateBtn`) writes a fresh
   sample; CI compares it against the snapshot. See `METHODS.md` for the
   tolerance derivation.

`METHODS.md` documents precision (f16 weights, f32 accumulators, ε_norm),
known divergences from HF, and per-kernel ULP error budgets.
`PREDICTIONS.md` is the pre-registered ablation predictions log —
falsifiable claims about which heads do what.

## Cross-site context

`src/lib/sites.ts` is synced from `~/sites-shared/sites.ts`. Footer
"More by Ahmet" links live in `index.html` and `app/index.html` — edit
them there for now (will become a shared partial in a later refactor).

## Known gaps

- No automated browser tests yet. Playwright is in `devDependencies` but
  no test files exist; a runtime-fingerprint smoke test would be the
  highest-leverage first one.
- Two entry points (`index.html` + `app/index.html`) duplicate the
  footer block (JSON-LD lives only on the marketing page). Consolidation
  blocked on sites-shared HTML partials.
- The `neural-pulse` → `neuropulse` cache-name migration is handled by
  the any-cache scan in `weight-loader.ts`; do not remove that fallback
  until enough time has passed that no users have the old cache.

## Historical log

- **2026-04-22** — Project renamed from `neural-pulse` to `neuropulse`
  across the CF Pages project, GitHub repo, directory name, and weight
  cache. Pages project was recreated (immutable name); domains
  `neuropulse.live` + `www.neuropulse.live` migrated over.
- **2026-05-09** — Empirical-lab pass: introduced `phi3-facts.ts`,
  `verify-claims.mjs`, `METHODS.md`, `PREDICTIONS.md`, runtime fingerprint
  footer, and `getStoredWeightStats` / `clearStoredWeights` for the
  storage-management modal. Audit drift fixed (kernels 10→11, MLC vs
  GGUF, scroll→arrows for Journey, 4 GB vs 2 GB reconciled).

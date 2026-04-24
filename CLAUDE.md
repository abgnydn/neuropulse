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
  visualizer wiring.
- `src/visualizer.ts` — Three.js scene, layer/token/activation rendering.
- `src/audio.ts` — ambient sonification driven by activation magnitudes.
- `src/engine/weight-loader.ts` — fetches GGUF-converted Phi-3 weights,
  caches them in the browser Cache API under `CACHE_NAME =
  'neuropulse-phi3-weights'`. Includes a step-2 fallback that scans *any*
  cache name — old `neural-pulse-phi3-weights` caches (from the pre-rename
  era) get re-homed transparently on first load, so users don't re-download
  the 2 GB.
- `src/engine/inference.ts` — per-token decode loop, calls the compiler
  + dispatcher.
- `src/engine/compiler.ts` — stitches WGSL shaders into a per-layer compute
  pipeline.
- `src/engine/tokenizer.ts` — BPE tokenizer loaded from the same cache as
  the weights.
- `src/engine/shaders/*.wgsl` — 10 hand-written kernels (int4 matmul,
  rms_norm, attention, rope, fused_ffn, etc.).
- `src/lib/sites.ts` — synced from `~/sites-shared/sites.ts`.
- `index.html` + `app/index.html` — marketing essay + demo host pages.
  Both have JSON-LD `Person` with `sameAs` driven by `SAME_AS` (edit in
  `~/sites-shared/sites.ts`).

## Commands

```bash
npm install
npm run dev          # Vite dev server
npm run build        # Vite production build → dist/
npm run preview      # preview built dist/
npm run typecheck    # tsc --noEmit
npm run check        # typecheck + build
```

Deploy: `node ~/sites-shared/deploy.mjs neuropulse` (CF Pages, project
`neuropulse`).

## Cross-site context

`src/lib/sites.ts` is synced from `~/sites-shared/sites.ts`. Footer
"More by Ahmet" links live in `index.html` and `app/index.html` — edit
them there for now (will become a shared partial in a later refactor).

## Known gaps

- No lint/test configured yet. `check` currently runs typecheck + build
  only.
- Two entry points (`index.html` + `app/index.html`) duplicate the JSON-LD
  and footer blocks. Consolidation blocked on sites-shared HTML partials.
- The `neural-pulse` → `neuropulse` cache-name migration is handled by
  the step-2 scan in `weight-loader.ts`; do not remove that fallback until
  enough time has passed that no users have the old cache.

## Historical log

- **2026-04-22** — Project renamed from `neural-pulse` to `neuropulse` across
  the CF Pages project, GitHub repo, directory name, and weight cache.
  Pages project was recreated (immutable name); domains `neuropulse.live`
  + `www.neuropulse.live` migrated over.

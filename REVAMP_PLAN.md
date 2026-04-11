# Neuropulse Demo Revamp

## Diagnosis
The current `/app/` is information-rich but the layout is a Bloomberg Terminal:
10+ visualizations (head heatmap, residual strip, attention pattern, logit lens,
raw f32 readout, residual norm, layer delta, etc.) all squeezed into a single
420px side panel of 4-12px-tall thumbnails. None of them is given the screen
space to be a hero. The 3D scene is the centerpiece visually but the *content*
lives in the side panel — a presentation/substance mismatch.

The data capture is excellent. The data presentation is not.

## Strategy: MODES

Add a top-of-canvas mode bar. Each mode reorganizes the layout to put ONE
thing front-and-center, while a slimmed-down right column shows the
persistent context (output, top-k, KV, raw stats).

## Modes

### 1. Scene (default — the existing 3D view)
- Three.js scene as the hero
- Right column: output, top-k bars, confidence, KV cache, raw GPU stats
- Removed from right column: every secondary chart (moved to its dedicated mode)

### 2. Attention (the iconic missing view)
- Full-screen 32×32 grid of mini per-head attention heatmaps (1024 cells)
- Click any cell → expands to a big detail view of that head's full attention
  pattern across all past tokens
- Token strip at the bottom shows prompt + generated tokens, with arcs drawn
  for the currently selected head
- Uses existing `onAllAttentionScores(scores, kvLen)` data path — all layers,
  all heads already captured every token

### 3. Logit Lens (watch the answer crystallize)
- Full-height vertical stack of all 32 layers
- Each row: layer number, top-5 predicted tokens with probability bars
- Updates live as the model decodes
- The "what would the model say if it stopped at layer N" view, finally given
  the space it deserves
- Requires extending the lens from 9 layers to all 32, and capturing top-5
  not just top-1

### 4. Cinematic (slow-motion forward pass)
- Three.js scene + a play/pause/step toolbar
- Auto-camera flythrough that tracks the active layer down the residual axis
- "First token slow-motion" — first token of every prompt runs at 1 dispatch/s,
  rest at full speed
- The 292-dispatch counter becomes a scrubbable timeline

## Persistent additions

- **Prompt presets**: row of 6-8 clickable chips above the input (math, code,
  philosophy, fact, japanese, json, instruction). Lowers the bar to "type
  something interesting".
- **Mode-aware screenshot button**: exports the active mode's hero view, not
  just the 3D canvas.
- **Recording mode**: 10s `MediaRecorder` capture of the active mode →
  downloadable webm. Pairs with the `?capture=1` RAF polyfill.
- **First-visit tutorial overlay**: arrow pointing at the mode bar saying
  "try clicking through the modes". Dismissed forever via localStorage.
- **Inspect mode polish**: when a neuron is locked, dim everything else and
  show its activation history across the last N tokens.

## Layout

```
┌───────────────────────────────────────────────────────────┐
│ Header (logo · stats · star)                              │
├───────────────────────────────────────────────────────────┤
│ Mode bar:  [Scene]  [Attention]  [Logit Lens]  [Cinema]   │
├──────────────────────────────────────┬────────────────────┤
│                                      │ Output             │
│                                      │ Top-k              │
│           Hero area                  │ Confidence         │
│           (mode-dependent)           │ KV Cache           │
│                                      │ Raw GPU stats      │
│                                      │ ─────              │
│                                      │ (mode extras)      │
├──────────────────────────────────────┴────────────────────┤
│ Token strip (prompt + generated, attention arcs)          │
├───────────────────────────────────────────────────────────┤
│ Prompt presets · input · speed · 🔊 📷 🎥 🔗 🧪 [Think]    │
└───────────────────────────────────────────────────────────┘
```

## Execution order

1. Mode infrastructure (CSS grid + state machine + mode bar)
2. Slim the right column (remove everything that's moving to a dedicated mode)
3. Attention mode (highest visual leverage)
4. Logit Lens mode
5. Cinematic mode (camera flythrough + step controls)
6. Inspect mode polish
7. Prompt presets + tutorial overlay
8. Recording mode + mode-aware screenshot
9. Verify in preview, commit, deploy

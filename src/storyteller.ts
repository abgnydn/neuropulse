// Phi the Storyteller — kid-friendly narration overlay.
//
// A floating "what is Phi doing right now" panel that hooks the existing
// engine callbacks (onPrefill, onLayer, onLayerLogitLens, onToken) and
// translates each technical event into one short sentence in plain
// language. Same data the analytical panels show; different vocabulary.
//
// Toggle: K key. Off by default. On boot: panel hidden, callbacks no-op.
//
// Design intent — cf. CHI 2026 Transformer Explainer paper:
//   - existing tools use toy models + technical labels;
//   - this is the first frontier-scale model with a kid layer.
//   - narration is generated from REAL telemetry, not pre-scripted.
//
// Vocabulary swaps:
//   token         → "word piece"
//   layer / step  → "thinking step" / "thought"
//   attention head→ "looker"
//   ablation      → "covered eyes" / "turned off"
//   logit lens    → "what would Phi say if he stopped here?"
//   residual      → "scratchpad"

import type { InferenceCallbacks, TopKEntry } from './engine/inference'

interface StorytellerHandle {
  setActive: (on: boolean) => void
  toggleActive: () => void
  isActive: () => boolean
  /** Merge into the InferenceCallbacks bag passed to engine.generate. */
  hooks: () => Partial<InferenceCallbacks>
  /** Manual narration for non-engine moments (ablation, sweep, errors). */
  say: (text: string, cls?: 'phi' | 'you' | 'think' | 'ablate' | 'good' | 'bad') => void
  /** Clear the feed (used between runs). */
  reset: () => void
}

export function initStoryteller(): StorytellerHandle {
  // Inject styles once.
  const style = document.createElement('style')
  style.textContent = `
    .storyteller-panel {
      display: none;
      position: fixed;
      top: 96px;
      left: 16px;
      width: 320px;
      max-height: 50vh;
      overflow-y: auto;
      padding: 14px 16px 12px;
      background: rgba(10, 8, 22, 0.92);
      border: 1px solid rgba(180, 130, 255, 0.4);
      border-radius: 12px;
      backdrop-filter: blur(16px) saturate(1.4);
      -webkit-backdrop-filter: blur(16px) saturate(1.4);
      box-shadow: 0 0 24px rgba(180, 130, 255, 0.18), 0 14px 50px rgba(0,0,0,0.55);
      z-index: 19;
      font-family: 'Inter', system-ui, sans-serif;
      font-size: 13px;
      color: #f4ecdf;
      line-height: 1.55;
    }
    body.kid-mode .storyteller-panel { display: block; }
    .storyteller-header {
      display: flex; align-items: baseline; gap: 8px; margin-bottom: 8px;
      padding-bottom: 6px;
      border-bottom: 1px solid rgba(180, 130, 255, 0.2);
    }
    .storyteller-name { color: #b483ff; font-weight: 700; font-size: 14px; letter-spacing: 0.02em; }
    .storyteller-mood { color: #8a8170; font-size: 11px; font-style: italic; flex: 1; text-align: right; }
    .storyteller-feed > div {
      margin-bottom: 4px;
      opacity: 0; transform: translateY(3px);
      animation: storyFade 0.28s cubic-bezier(0.2,0.7,0.2,1) forwards;
    }
    @keyframes storyFade { to { opacity: 1; transform: translateY(0); } }
    .storyteller-feed .you    { color: #ff8c42; font-weight: 500; }
    .storyteller-feed .phi    { color: #f4ecdf; }
    .storyteller-feed .think  { color: #5eead4; font-style: italic; }
    .storyteller-feed .ablate { color: #ff9a1f; }
    .storyteller-feed .good   { color: #5eead4; font-weight: 500; }
    .storyteller-feed .bad    { color: #ff7a85; font-weight: 500; }
    .storyteller-tip {
      margin-top: 8px; padding-top: 8px;
      border-top: 1px dashed rgba(180,130,255,0.25);
      font-size: 11px; color: #8a8170; font-style: italic;
    }
  `
  document.head.appendChild(style)

  // Build the panel DOM.
  const panel = document.createElement('div')
  panel.className = 'storyteller-panel'
  panel.innerHTML = `
    <div class="storyteller-header">
      <span class="storyteller-name">Phi</span>
      <span class="storyteller-mood" id="storytellerMood">ready</span>
    </div>
    <div class="storyteller-feed" id="storytellerFeed"></div>
    <div class="storyteller-tip">
      Tip: shift-click any little blue ball to cover Phi's eyes there.
      Tip: press K to turn me off.
    </div>
  `
  document.body.appendChild(panel)

  const feed = panel.querySelector<HTMLDivElement>('#storytellerFeed')!
  const moodEl = panel.querySelector<HTMLSpanElement>('#storytellerMood')!

  let active = false
  // Per-generation state — keeps narration sparse so it doesn't drown the kid.
  let layerCount = 0
  let lastLensToken = ''
  let tokenCount = 0
  let firstNonStopToken = true

  function say(text: string, cls: 'phi'|'you'|'think'|'ablate'|'good'|'bad' = 'phi') {
    if (!active) return
    const line = document.createElement('div')
    line.className = cls
    line.textContent = text
    feed.appendChild(line)
    feed.scrollTop = feed.scrollHeight
    // Cap the feed length so old lines fall off — kids won't scroll back.
    while (feed.childElementCount > 60) feed.firstElementChild?.remove()
  }

  function setMood(m: string) { if (active) moodEl.textContent = m }

  function reset() {
    feed.innerHTML = ''
    layerCount = 0
    lastLensToken = ''
    tokenCount = 0
    firstNonStopToken = true
  }

  const hooks = (): Partial<InferenceCallbacks> => ({
    onPrefill: (phase, length) => {
      if (!active) return
      if (phase === 'start') {
        reset()
        say(`Reading your question — ${length} word puzzle pieces.`, 'phi')
        setMood('reading')
      } else {
        say(`I read it all. Now I'll think 32 times to find the answer.`, 'think')
        setMood('thinking')
      }
    },
    onLayer: (layer, step) => {
      if (!active) return
      // Track layer transitions sparsely. Each token causes step 0..8 across
      // 32 layers — way too many for a feed line. Only narrate at layer
      // milestones during the FIRST decoded token's pass.
      if (tokenCount > 0) return
      if (step !== 0) return
      layerCount = layer
      if (layer === 8 || layer === 16 || layer === 24) {
        say(`...thinking step ${layer}/32 ...`, 'think')
      }
    },
    onLayerLogitLens: (layer, _id, token) => {
      if (!active) return
      // Phi's "early guess" — a real readout. Only narrate when the guess
      // CHANGES (signal, not noise) and only on the first decoded token's
      // pass so we don't repeat for every emitted word.
      if (tokenCount > 0) return
      const clean = token.trim()
      if (!clean) return
      if (clean === lastLensToken) return
      lastLensToken = clean
      say(`At step ${layer}, my guess would be: "${clean}"`, 'think')
    },
    onToken: (token, _id, _index, topK?: TopKEntry[]) => {
      if (!active) return
      tokenCount++
      const t = token.replace(/\s+/g, ' ').trim()
      // Show every meaningful token; collapse pure-whitespace ones.
      if (!t) return
      // First few tokens get extra confidence color from top-k spread.
      let cls: 'phi' | 'good' = 'phi'
      if (topK && topK.length > 0 && topK[0].prob > 0.7) cls = 'good'
      const sym = firstNonStopToken ? '' : ''
      firstNonStopToken = false
      say(`${sym}I pick: "${t}"`, cls)
      setMood(`saying word ${tokenCount}`)
    },
  })

  function setActive(on: boolean) {
    active = on
    document.body.classList.toggle('kid-mode', on)
    if (on) {
      reset()
      say('Hi! I\'m Phi. I\'m a robot brain made of 3.8 billion little numbers.', 'phi')
      say('Type a question up there and I\'ll show you how I think.', 'phi')
      setMood('ready')
    }
  }

  return {
    setActive,
    toggleActive: () => setActive(!active),
    isActive: () => active,
    hooks,
    say,
    reset,
  }
}

/** Merge two callback bags. Storyteller hooks run AFTER the original ones
 *  so they see the same data and don't override. Both are awaited if the
 *  original returns a Promise (onPrefillToken / onLayer can be async). */
export function mergeCallbacks(
  base: InferenceCallbacks,
  extra: Partial<InferenceCallbacks>,
): InferenceCallbacks {
  const out: InferenceCallbacks = { ...base }
  for (const key of Object.keys(extra) as (keyof InferenceCallbacks)[]) {
    const a = base[key] as ((...args: unknown[]) => unknown) | undefined
    const b = extra[key] as ((...args: unknown[]) => unknown) | undefined
    if (!b) continue
    if (!a) {
      ;(out as Record<string, unknown>)[key] = b
      continue
    }
    ;(out as Record<string, unknown>)[key] = async (...args: unknown[]) => {
      try { await a(...args) } catch { /* ignore */ }
      try { await b(...args) } catch { /* ignore */ }
    }
  }
  return out
}

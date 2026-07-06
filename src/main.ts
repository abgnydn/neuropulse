import { BrainVisualizer, LayerActivation } from './visualizer'
import { createInferenceEngine, InferenceEngine, InferenceCallbacks, LoadProgress, TopKEntry, Ablation } from './engine/inference'
import { getStoredWeightStats, clearStoredWeights } from './engine/weight-loader'
import { initStoryteller, mergeCallbacks } from './storyteller'
import { initButterflyPanel } from './butterfly-mode'
import { reduceQKVForAttnHeads, reduceForAttnHeads, reduceForFFNGroups, reduceForResidual, normalizeFull } from './engine/activation-reducer'
import { createJourney, JourneyHandle } from './journey'
import { SpatialPanels } from './spatial-panels'
import { TOURS, createTourRunner } from './tours'
import { LESSONS, type Lesson } from './lessons'
import { createRecorder, type Recorder, type NpRecording } from './recording'
import { createPlaybackDriver, type PlaybackHandle, type PlaybackSink } from './playback'
// Ask-mode reference docs — Vite imports markdown as a raw string. See
// src/docs.md. Budgeted to ~1500 tokens so Phi-3 has room for the reply.
import NEUROPULSE_DOCS from './docs.md?raw'

// ═══════════════════════════════════════════════════════════════
// Neuropulse — Main v4 (Real Phi-3 Inference)
// ═══════════════════════════════════════════════════════════════

const canvas = document.getElementById('brainCanvas') as HTMLCanvasElement

// Null-safe audio stub so viz.audio.* never throws when WebGL is unavailable
const nullAudio = { isMuted: () => true, toggleMute: () => true, resume: () => {}, startDrone: () => {}, stopDrone: () => {}, tokenChime: () => {}, setTokenConfidence: () => {} }

// Create a no-op proxy so every viz.method() silently returns undefined
// instead of crashing when WebGL init fails. The proxy also exposes
// a stubbed `audio` property so viz.audio.* calls are safe.
function createNullViz(): BrainVisualizer {
  return new Proxy({} as any, {
    get(_target, prop) {
      if (prop === 'audio') return nullAudio
      return () => {}
    },
  }) as BrainVisualizer
}

let viz: BrainVisualizer = createNullViz()

function initVisualizer() {
  viz = new BrainVisualizer(canvas)
  viz.start()
  wireJourney()
  initAblationPanel()

  // Test hook: current camera position. Lets Playwright assert that tours /
  // journey actually FLY the camera (canvas pixels are unreadable in headless
  // WebGL, which once hid a camera-stomp bug).
  ;(window as unknown as { __npCamPos?: () => number[] }).__npCamPos = () => {
    const cam = (viz as unknown as { getCamera?: () => { position: { toArray(): number[] } } }).getCamera?.()
    return cam ? cam.position.toArray().map((v: number) => Math.round(v * 100) / 100) : []
  }

  // Test hook: programmatic shift-click equivalent. Used by Playwright to
  // exercise the ablation UI without needing to raycast 3D screen coords.
  // No-op if the visualizer is the null stub.
  ;(window as unknown as {
    __testToggleAblation: (layer: number, head: number) => boolean
  }).__testToggleAblation = (layer, head) => {
    const maybeNeurons = (viz as unknown as { neurons?: { layer: number; role: string; subIndex: number }[] }).neurons
    if (!maybeNeurons) return false
    const n = maybeNeurons.find(m => m.layer === layer && m.role === 'attn' && m.subIndex === head)
    if (!n) return false
    ;(viz as unknown as { toggleAblation: (n: unknown) => void }).toggleAblation(n)
    viz.onAblationChange?.(viz.getAblations())
    return true
  }
}

// Shift-click an attention head in 3D to mark it ablated. The panel below
// appears, showing the current selection and a "Run ablated" button that
// generates once normally and once with those heads zeroed post-attention,
// displaying both outputs side by side.
function initAblationPanel() {
  const inputWrap = document.querySelector('.input-wrap')
  if (!inputWrap) return

  const style = document.createElement('style')
  style.textContent = `
    .ablate-panel {
      position: fixed; top: 64px; right: 20px;
      width: 360px; max-height: calc(100vh - 100px); overflow-y: auto;
      background: rgba(12, 14, 20, 0.92); backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 154, 31, 0.45);
      border-radius: 10px; padding: 12px 14px; z-index: 20;
      color: #f4ecdf; font-family: inherit; font-size: 12px;
      box-shadow: 0 0 24px rgba(255, 154, 31, 0.18);
      display: none;
      transition: width 0.25s ease, height 0.25s ease, padding 0.25s ease, border-radius 0.25s ease;
    }
    .ablate-panel.open { display: block; }
    /* Collapsed pip mode — same size + treatment as the bottom-rail orbs,
       amber-tinted to keep its identity. Click to re-expand. */
    .ablate-panel.collapsed {
      width: 18px; height: 18px;
      padding: 0;
      overflow: hidden;
      border-radius: 50%;
      border-width: 2px;
      background: radial-gradient(circle, rgba(255, 154, 31, 0.7), rgba(255, 154, 31, 0.12) 70%);
      box-shadow: 0 0 16px rgba(255, 154, 31, 0.55), inset 0 0 6px rgba(255, 154, 31, 0.6);
      cursor: pointer;
    }
    .ablate-panel.collapsed > * { display: none !important; }
    .ablate-panel.collapsed:hover::after {
      content: 'Ablation';
      position: absolute;
      top: calc(100% + 8px); left: 50%; transform: translateX(-50%);
      padding: 4px 10px;
      background: rgba(8, 6, 15, 0.92);
      border: 1px solid rgba(244, 236, 223, 0.18);
      border-radius: 4px; font-size: 11px; color: #f4ecdf;
      white-space: nowrap; pointer-events: none;
      letter-spacing: 0.02em;
    }
    /* Sync with the global panels toggle (P / Tab) — when other panels hide,
       this hides too. Pressing A re-toggles it independently. */
    body.panels-hidden .ablate-panel { display: none !important; }
    @media (max-width: 900px) {
      .ablate-panel { right: 10px; left: 10px; width: auto; }
    }
    .ablate-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
    .ablate-status { color: #ff9a1f; font-weight: 600; flex: 1 1 100%; min-width: 0; font-size: 11px; }
    .ablate-close {
      background: transparent; border: 1px solid rgba(244,236,223,0.18);
      color: #8a7f6c; width: 22px; height: 22px; border-radius: 50%;
      cursor: pointer; font-size: 11px; line-height: 1;
      display: inline-flex; align-items: center; justify-content: center;
      flex: 0 0 auto;
    }
    .ablate-close:hover { color: #f4ecdf; border-color: #f4ecdf; }
    .ablate-info {
      background: transparent; border: 1px solid rgba(255,154,31,0.5);
      color: #ffd28a; width: 22px; height: 22px; border-radius: 50%;
      cursor: pointer; font-family: 'Fraunces', Georgia, serif; font-style: italic;
      font-size: 13px; line-height: 1; display: inline-flex; align-items: center;
      justify-content: center; flex: 0 0 auto; padding: 0 0 1px 0;
    }
    .ablate-info:hover, .ablate-info.on { background: rgba(255,154,31,0.2); border-color: #ff9a1f; color: #fff; }
    .ablate-info-body {
      display: none; margin: 0 0 10px; padding: 10px 12px;
      background: rgba(255,154,31,0.06); border: 1px solid rgba(255,154,31,0.25);
      border-radius: 6px; font-size: 12px; line-height: 1.5; color: #d8cdb8;
    }
    .ablate-info-body.on { display: block; }
    .ablate-info-body b { color: #ffd28a; font-weight: 600; }
    .ablate-info-body p { margin: 0 0 7px; }
    .ablate-info-body p:last-child { margin-bottom: 0; }
    .ablate-learnmore {
      background: transparent; border: none; color: #00e5ff; cursor: pointer;
      font-family: inherit; font-size: 12px; padding: 2px 0; text-decoration: underline;
      text-underline-offset: 2px;
    }
    .ablate-learnmore:hover { color: #7df; }
    .ablate-hint { color: #8a7f6c; font-size: 11px; font-style: italic; }
    .ablate-btn {
      background: rgba(255, 154, 31, 0.18); color: #ffd28a;
      border: 1px solid rgba(255, 154, 31, 0.5); border-radius: 5px;
      padding: 5px 12px; cursor: pointer; font-size: 12px; font-family: inherit;
      transition: all 0.15s;
    }
    .ablate-btn:hover { background: rgba(255, 154, 31, 0.3); color: #fff; }
    .ablate-btn[disabled] { opacity: 0.4; cursor: wait; }
    .ablate-btn.clear { background: transparent; color: #8a7f6c; border-color: #3a3429; }
    .ablate-btn.clear:hover { color: #f4ecdf; border-color: #514a3e; }
    .ablate-btn.sweep { background: rgba(0, 229, 255, 0.12); color: #00e5ff; border-color: rgba(0, 229, 255, 0.5); }
    .ablate-btn.sweep:hover { background: rgba(0, 229, 255, 0.22); color: #fff; }
    /* Stacked vertically — top-right panel is narrow, side-by-side wraps badly */
    .ablate-outputs { display: flex; flex-direction: column; gap: 8px; }
    .ablate-output { background: rgba(0,0,0,0.35); border-radius: 6px; padding: 8px 10px; min-height: 36px; }
    .ablate-output-label { color: #8a7f6c; font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
    .ablate-output-label.abl { color: #ff9a1f; }
    .ablate-output-text { color: #f4ecdf; font-size: 12px; line-height: 1.45; white-space: pre-wrap; word-break: break-word; }
    .ablate-output-text.empty { color: #514a3e; font-style: italic; }

    .ablate-sweep-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; padding: 6px 8px; background: rgba(0, 229, 255, 0.05); border-radius: 6px; border: 1px solid rgba(0, 229, 255, 0.15); }
    .ablate-sweep-row label { color: #8a7f6c; font-size: 11px; }
    .ablate-sweep-row input[type=number] {
      width: 44px; background: rgba(0,0,0,0.4); color: #f4ecdf;
      border: 1px solid #3a3429; border-radius: 4px; padding: 3px 6px;
      font-family: inherit; font-size: 12px; text-align: center;
    }
    .ablate-sweep-status { color: #8a7f6c; font-size: 11px; flex: 1; }
    .ablate-sweep-status.running { color: #00e5ff; }

    /* 32 fixed-width head cells that scroll horizontally within the panel
       rather than stretching the panel wider than its container. */
    .ablate-strip { display: none; grid-template-columns: repeat(32, minmax(12px, 1fr)); gap: 2px; margin-bottom: 8px; max-width: 100%; overflow-x: auto; padding-bottom: 4px; }
    .ablate-strip.visible { display: grid; }
    .ablate-strip-cell {
      aspect-ratio: 1; background: rgba(80,80,80,0.2); border-radius: 2px;
      cursor: pointer; position: relative;
      display: flex; align-items: center; justify-content: center;
      font-size: 9px; color: rgba(255,255,255,0.3);
      transition: transform 0.1s, box-shadow 0.1s;
    }
    .ablate-strip-cell:hover { transform: scale(1.35); z-index: 2; box-shadow: 0 0 8px rgba(255,255,255,0.4); }
    .ablate-strip-cell.picked { outline: 1px solid #ff9a1f; outline-offset: 1px; }
    .ablate-strip-legend { display: flex; gap: 8px; font-size: 10px; color: #8a7f6c; margin-top: 2px; }
    .ablate-strip-legend span::before { content: ''; display: inline-block; width: 8px; height: 8px; border-radius: 1px; margin-right: 4px; vertical-align: middle; }
    .ablate-strip-legend .none::before { background: rgba(0,229,255,0.7); }
    .ablate-strip-legend .some::before { background: rgba(255,200,120,0.7); }
    .ablate-strip-legend .big::before  { background: rgba(255,80,80,0.8); }
  `
  document.head.appendChild(style)

  const panel = document.createElement('div')
  panel.className = 'ablate-panel open'
  panel.innerHTML = `
    <div class="ablate-header">
      <span class="ablate-status" id="ablateStatus">No heads ablated — shift-click attention spheres, or sweep a layer to see impact.</span>
      <button class="ablate-btn" id="ablateRunBtn" type="button" disabled>Run ablated</button>
      <button class="ablate-btn clear" id="ablateClearBtn" type="button">Clear</button>
      <button class="ablate-info" id="ablateInfoBtn" type="button" aria-label="What is ablation?" title="What is this?">i</button>
      <button class="ablate-close" id="ablateCloseBtn" type="button" aria-label="Hide ablation panel (A)" title="Hide · A">✕</button>
    </div>
    <div class="ablate-info-body" id="ablateInfoBody">
      <p><b>Ablation</b> switches off attention heads to see what they do. Shift-click heads in the 3D scene, then <b>Run ablated</b> — the model answers with and without them, side by side.</p>
      <p>Turning off 1–2 of 1,024 heads usually changes nothing (the model is very redundant) — that's expected, not a glitch. <b>Sweep</b> finds the heads that matter.</p>
      <p><button class="ablate-learnmore" id="ablateLearnMore" type="button">Full explanation →</button></p>
    </div>
    <div class="ablate-sweep-row">
      <label for="ablateSweepLayer">Sweep layer</label>
      <input type="number" id="ablateSweepLayer" min="0" max="31" value="31">
      <button class="ablate-btn sweep" id="ablateSweepBtn" type="button">Sweep 32 heads</button>
      <span class="ablate-sweep-status" id="ablateSweepStatus">(~60 s at 8 decoded tokens each)</span>
    </div>
    <div class="ablate-strip" id="ablateStrip"></div>
    <div class="ablate-outputs">
      <div class="ablate-output">
        <div class="ablate-output-label">Baseline</div>
        <div class="ablate-output-text empty" id="ablateBaseOut">— not yet run —</div>
      </div>
      <div class="ablate-output">
        <div class="ablate-output-label abl">Ablated</div>
        <div class="ablate-output-text empty" id="ablateAblOut">— not yet run —</div>
      </div>
    </div>
  `
  inputWrap.parentNode?.insertBefore(panel, inputWrap)
  // Drag-to-move with localStorage persistence (`neuropulse:panel-pos:ablate-panel`).
  makeDraggable(panel, 'ablate-panel')

  const statusEl = panel.querySelector<HTMLSpanElement>('#ablateStatus')!
  const runBtn = panel.querySelector<HTMLButtonElement>('#ablateRunBtn')!
  const clearBtn = panel.querySelector<HTMLButtonElement>('#ablateClearBtn')!
  const baseOut = panel.querySelector<HTMLDivElement>('#ablateBaseOut')!
  const ablOut = panel.querySelector<HTMLDivElement>('#ablateAblOut')!
  const sweepLayerInput = panel.querySelector<HTMLInputElement>('#ablateSweepLayer')!
  const sweepBtn = panel.querySelector<HTMLButtonElement>('#ablateSweepBtn')!
  const sweepStatus = panel.querySelector<HTMLSpanElement>('#ablateSweepStatus')!
  const stripEl = panel.querySelector<HTMLDivElement>('#ablateStrip')!
  const closeBtn = panel.querySelector<HTMLButtonElement>('#ablateCloseBtn')!
  const infoBtn = panel.querySelector<HTMLButtonElement>('#ablateInfoBtn')!
  const infoBody = panel.querySelector<HTMLDivElement>('#ablateInfoBody')!
  const learnMore = panel.querySelector<HTMLButtonElement>('#ablateLearnMore')!

  function setPanelOpen(open: boolean) {
    panel.classList.toggle('open', open)
  }
  function setCollapsed(collapsed: boolean) {
    panel.classList.toggle('collapsed', collapsed)
  }
  // Open by default. Shift-clicking heads will also force-open it.
  setPanelOpen(true)

  // × button collapses to a pip rather than hiding entirely. Click the pip
  // to expand. Drag still works on both states.
  closeBtn.addEventListener('click', (e) => {
    e.stopPropagation()
    setCollapsed(true)
  })
  // "i" toggles the plain-English teaser; "Full explanation →" opens the
  // glossary (the roomy home for detail) at the Ablation entry.
  infoBtn.addEventListener('click', (e) => {
    e.stopPropagation()
    infoBtn.classList.toggle('on')
    infoBody.classList.toggle('on')
  })
  learnMore.addEventListener('click', (e) => {
    e.stopPropagation()
    openGlossaryAt('gloss-ablation')
  })
  panel.addEventListener('click', (e) => {
    if (!panel.classList.contains('collapsed')) return
    if (panel.dataset.justDragged) return
    // Don't expand from a child button — but the close button doesn't exist
    // in collapsed mode (children hidden), so any click is a "wake up".
    setCollapsed(false)
    e.stopPropagation()
  })

  // Expose toggle for the global keymap (A key) — toggles collapsed state
  // when the panel is open, and toggles open visibility otherwise.
  ;(window as unknown as { __toggleAblatePanel: () => void }).__toggleAblatePanel = () => {
    if (!panel.classList.contains('open')) { setPanelOpen(true); setCollapsed(false); return }
    setCollapsed(!panel.classList.contains('collapsed'))
  }

  function setSelectionStatus(abls: { layer: number; head: number }[]) {
    if (abls.length === 0) {
      statusEl.textContent = 'No heads ablated — shift-click attention spheres, or sweep a layer to see impact.'
      runBtn.disabled = true
    } else {
      const layers = new Set(abls.map(a => a.layer))
      statusEl.textContent =
        `${abls.length} head${abls.length === 1 ? '' : 's'} ablated across ${layers.size} layer${layers.size === 1 ? '' : 's'}`
      runBtn.disabled = false
    }
  }

  viz.onAblationChange = (abls) => {
    setSelectionStatus(abls)
    // When the user shift-clicks a head, force the panel open so they
    // see what just happened. Closing it is on them (× button or A key).
    if (abls.length > 0) setPanelOpen(true)
    // Sync the strip's picked markers with the visualizer's truth.
    const picked = new Set(abls.map(a => `${a.layer}:${a.head}`))
    stripEl.querySelectorAll<HTMLDivElement>('.ablate-strip-cell').forEach(cell => {
      cell.classList.toggle('picked', picked.has(cell.dataset.key || ''))
    })
  }
  setSelectionStatus([])

  clearBtn.addEventListener('click', () => {
    viz.clearAblations()
    baseOut.textContent = '— not yet run —'
    ablOut.textContent = '— not yet run —'
    baseOut.classList.add('empty')
    ablOut.classList.add('empty')
  })

  runBtn.addEventListener('click', async () => {
    if (!engine) { alert('Engine not ready yet.'); return }
    if (isValidating) { alert('Validation is running — wait for it to finish.'); return }
    // Cancel any in-flight generation instead of refusing, then run.
    if (isRunning) {
      cancelInFlightInference()
      if (!(await waitForInferenceIdle())) { alert('Could not stop the current run — try again in a moment.'); return }
    }
    const abls = viz.getAblations()
    if (abls.length === 0) return
    const prompt = (promptInput.value.trim() || lastPrompt || 'What is consciousness?')
    runBtn.disabled = true; clearBtn.disabled = true; sweepBtn.disabled = true
    runBtn.textContent = 'Running…'
    baseOut.classList.remove('empty'); ablOut.classList.remove('empty')
    baseOut.textContent = 'generating…'
    ablOut.textContent = 'waiting…'
    isRunning = true
    try {
      const cb: InferenceCallbacks = {}
      const merged = mergeCallbacks(cb, storyteller.hooks())
      storyteller.say(`Let me try first without anything covered...`, 'phi')
      const base = await engine.generate(prompt, 40, merged)
      baseOut.textContent = base || '(empty)'
      ablOut.textContent = 'generating…'
      const layers = new Set(abls.map(a => a.layer))
      storyteller.say(
        `Now I'll cover up ${abls.length} of my lookers in step${layers.size === 1 ? '' : 's'} ${[...layers].join(', ')} and try again...`,
        'ablate',
      )
      const abl = await engine.generate(prompt, 40, merged, abls)
      ablOut.textContent = abl || '(empty)'
      // Lessons: report how much the ablation changed the output (0..1).
      window.dispatchEvent(new CustomEvent('neuropulse:lesson-signal', {
        detail: { type: 'ablation', impact: impactScore(base, abl) },
      }))
      if (base !== abl) {
        storyteller.say(`Look! My answer changed when I covered those lookers — they were doing important work.`, 'good')
      } else {
        storyteller.say(`Hmm, my answer is the same. Those lookers weren't needed for this question.`, 'phi')
      }
    } catch (err) {
      ablOut.textContent = `error: ${err}`
    } finally {
      isRunning = false
      runBtn.disabled = viz.getAblations().length === 0
      clearBtn.disabled = false; sweepBtn.disabled = false
      runBtn.textContent = 'Run ablated'
    }
  })

  // Prefix-match divergence: returns how many leading chars the ablated
  // output shares with the baseline. 0 = total divergence, baseline.length
  // = identical. Converted to a 0..1 "impact" score (1 = big impact).
  function impactScore(baseline: string, ablated: string): number {
    if (baseline.length === 0) return 0
    let i = 0
    const n = Math.min(baseline.length, ablated.length)
    while (i < n && baseline[i] === ablated[i]) i++
    // 0 shared chars → impact 1; full baseline shared AND same length → 0.
    // If ablated diverged but matches length, impact scales with prefix miss.
    const miss = 1 - i / baseline.length
    const lenDelta = Math.abs(baseline.length - ablated.length) / Math.max(1, baseline.length)
    return Math.min(1, 0.7 * miss + 0.3 * lenDelta)
  }

  function impactColor(impact: number): string {
    // Cyan (no impact) → amber → red (big impact).
    if (impact < 0.5) {
      // cyan → amber
      const t = impact / 0.5
      const r = Math.round(0   + (255 - 0)   * t)
      const g = Math.round(229 + (200 - 229) * t)
      const b = Math.round(255 + (120 - 255) * t)
      const a = 0.35 + 0.35 * impact
      return `rgba(${r},${g},${b},${a.toFixed(2)})`
    } else {
      const t = (impact - 0.5) / 0.5
      const r = Math.round(255 + (255 - 255) * t)
      const g = Math.round(200 + (80  - 200) * t)
      const b = Math.round(120 + (80  - 120) * t)
      const a = 0.55 + 0.35 * impact
      return `rgba(${r},${g},${b},${a.toFixed(2)})`
    }
  }

  // Sweep: run N single-head ablations at a fixed layer, render a 32-cell
  // impact strip. Clicking any cell shift-click-toggles that head, feeding
  // it into the full-generation path above.
  sweepBtn.addEventListener('click', async () => {
    if (!engine) { alert('Engine not ready yet.'); return }
    if (isValidating) { alert('Validation is running — wait for it to finish.'); return }
    // Cancel any in-flight generation instead of refusing, then sweep.
    if (isRunning) {
      cancelInFlightInference()
      if (!(await waitForInferenceIdle())) { alert('Could not stop the current run — try again in a moment.'); return }
    }
    const L = Math.max(0, Math.min(31, Number(sweepLayerInput.value) || 0))
    sweepLayerInput.value = String(L)

    // Heads-up before a multi-minute GPU load. The mobile guard already
    // blocks phones from reaching the panel at all (__NEUROPULSE_MOBILE_BLOCK__
    // / matchMedia ≤ 820px in app/index.html); this confirm protects desktops
    // with weak/integrated GPUs from inadvertently melting their fans.
    const ok = confirm(
      `Sweep layer ${L}: 33 short generations on your GPU (1 baseline + 32 ablated).\n\n` +
      `Roughly 60 seconds of sustained full GPU load. Laptop fans will spin up.\n\n` +
      `Continue?`
    )
    if (!ok) return

    const prompt = (promptInput.value.trim() || lastPrompt || 'What is consciousness?')
    const maxTokens = 8  // short: sweep is qualitative, not the final answer

    sweepBtn.disabled = true; runBtn.disabled = true; clearBtn.disabled = true
    sweepStatus.classList.add('running')
    stripEl.classList.add('visible')
    stripEl.innerHTML = ''
    isRunning = true

    try {
      sweepStatus.textContent = `sweeping L${L} — baseline…`
      const cb = {}
      const baseline = await engine.generate(prompt, maxTokens, cb)

      // Seed all 32 cells as "pending" (grey) up-front so the user sees the
      // strip widen immediately, then each lands a color as it completes.
      for (let h = 0; h < 32; h++) {
        const cell = document.createElement('div')
        cell.className = 'ablate-strip-cell'
        cell.dataset.key = `${L}:${h}`
        cell.title = `L${L} head ${h} — pending`
        cell.textContent = String(h)
        cell.addEventListener('click', () => {
          const maybeNeurons = (viz as unknown as {
            neurons?: { layer: number; role: string; subIndex: number }[]
          }).neurons
          const n = maybeNeurons?.find(m => m.layer === L && m.role === 'attn' && m.subIndex === h)
          if (n) {
            ;(viz as unknown as { toggleAblation: (n: unknown) => void }).toggleAblation(n)
            viz.onAblationChange?.(viz.getAblations())
          }
        })
        stripEl.appendChild(cell)
      }

      for (let h = 0; h < 32; h++) {
        sweepStatus.textContent = `sweeping L${L} — head ${h + 1}/32`
        const out = await engine.generate(prompt, maxTokens, cb, [{ layer: L, head: h }])
        const impact = impactScore(baseline, out)
        const cell = stripEl.children[h] as HTMLDivElement
        cell.style.background = impactColor(impact)
        cell.title = `L${L} head ${h} — impact ${(impact * 100).toFixed(0)}%\nbase: ${baseline}\nabl:  ${out}`
      }

      const legend = document.createElement('div')
      legend.className = 'ablate-strip-legend'
      legend.innerHTML = `<span class="none">no impact</span><span class="some">some impact</span><span class="big">big impact</span>`
      stripEl.appendChild(legend)

      sweepStatus.textContent = `L${L} swept — click a cell to toggle that head`
    } catch (err) {
      sweepStatus.textContent = `sweep failed: ${err}`
    } finally {
      isRunning = false
      sweepStatus.classList.remove('running')
      sweepBtn.disabled = false
      clearBtn.disabled = false
      runBtn.disabled = viz.getAblations().length === 0
    }
  })
}

const output = document.getElementById('output')!
const goBtn = document.getElementById('goBtn') as HTMLButtonElement
const promptInput = document.getElementById('promptInput') as HTMLInputElement

let isRunning = false
let isValidating = false
// Last prompt actually run — the prompt box is cleared after each generation,
// so ablation/sweep fall back to this (not a hardcoded string) to stay in sync.
let lastPrompt = ''
let totalTokens = 0
let engine: InferenceEngine | null = null

// Phi the Storyteller — kid-mode narration overlay (K key toggles).
// Off by default; merged into engine.generate callbacks when active.
const storyteller = initStoryteller()

// ─── Learner levels — presentation presets over one lesson set ─────────────
// Kid = storyteller narration on + simplified vocabulary; Explorer (default
// for new visitors) = expert chrome hidden (.xp-only); Expert = the full
// cockpit. Levels change PRESENTATION only — same lessons, same glossary.
type LearnerLevel = 'kid' | 'explorer' | 'expert'
const LEVEL_KEY = 'neuropulse:level'
const LEVELS: LearnerLevel[] = ['kid', 'explorer', 'expert']

function loadLevel(): LearnerLevel | null {
  try {
    const v = JSON.parse(localStorage.getItem(LEVEL_KEY) ?? 'null')
    return LEVELS.includes(v) ? v : null
  } catch { return null }
}

/** Anyone with prior neuropulse:* state has used the full UI — don't yank
 *  their chrome; suggest Expert. Fresh visitors get Explorer. */
function suggestedLevel(): LearnerLevel {
  try {
    for (let i = 0; i < localStorage.length; i++) {
      if (localStorage.key(i)?.startsWith('neuropulse:')) return 'expert'
    }
  } catch { /* storage disabled */ }
  return 'explorer'
}

let currentLevel: LearnerLevel = loadLevel() ?? suggestedLevel()

function applyLevel(level: LearnerLevel, persist = true): void {
  currentLevel = level
  for (const l of LEVELS) document.body.classList.toggle(`level-${l}`, l === level)
  // Kid mode drives the storyteller; leaving kid turns narration back off
  // (the K key stays a session-only override and never writes the level).
  storyteller.setActive(level === 'kid')
  const label = document.getElementById('level-toggle-label')
  if (label) label.textContent = level
  if (persist) { try { localStorage.setItem(LEVEL_KEY, JSON.stringify(level)) } catch { /* ok */ } }
}

/** First overlay after the boot fade (the slot the welcome overlay vacated).
 *  Shows once — the stored level doubles as the "seen" flag. */
function maybeShowLevelChooser(): void {
  if (loadLevel() !== null) return
  const overlay = document.getElementById('level-overlay')
  if (!overlay) return
  const suggest = suggestedLevel()
  overlay.querySelectorAll<HTMLButtonElement>('.level-option').forEach((btn) => {
    btn.classList.toggle('suggested', btn.dataset.level === suggest)
  })
  window.setTimeout(() => overlay.classList.add('visible'), 900)
}

;(function wireLevels() {
  // Apply the stored/suggested level immediately (no persist for the implicit
  // default — the chooser stays pending until an explicit pick).
  applyLevel(currentLevel, loadLevel() !== null)

  const overlay = document.getElementById('level-overlay')
  overlay?.querySelectorAll<HTMLButtonElement>('.level-option').forEach((btn) => {
    btn.addEventListener('click', () => {
      const lv = btn.dataset.level as LearnerLevel
      if (LEVELS.includes(lv)) applyLevel(lv)
      overlay.classList.remove('visible')
    })
  })
  // Backdrop click = accept the suggested default (recorded as chosen so the
  // chooser doesn't nag on every visit).
  overlay?.addEventListener('click', (e) => {
    if (e.target === overlay) { applyLevel(currentLevel); overlay.classList.remove('visible') }
  })

  // Top-right pill cycles kid → explorer → expert.
  document.getElementById('level-toggle')?.addEventListener('click', () => {
    const next = LEVELS[(LEVELS.indexOf(currentLevel) + 1) % LEVELS.length]!
    applyLevel(next)
  })
})()

// Speed control
const speedSlider = document.getElementById('speedSlider') as HTMLInputElement
const speedLabel = document.getElementById('speedLabel')!
speedSlider.addEventListener('input', () => {
  speedLabel.textContent = speedSlider.value + 'x'
})
// Single source of truth for pace: the Speed slider (1–20). Tours (src/tours.ts)
// and journey autoplay (src/journey.ts) read this so one control drives the
// forward-pass tokens, the tour camera, and the journey flythrough together.
function currentSpeed(): number { return parseInt(speedSlider.value, 10) || 5 }
;(window as unknown as { __npSpeed?: () => number }).__npSpeed = currentSpeed

// SVG icon helpers for tool buttons
const SVG_SOUND_ON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>'
const SVG_SOUND_OFF = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>'
const SVG_RECORD = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>'
const SVG_RECORD_ACTIVE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="5" fill="#ef4444"/></svg>'
const SVG_VALIDATE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>'
const SVG_VALIDATE_BUSY = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>'

// Sound toggle — viz.audio is always safe (nullAudio proxy before WebGL init)
const soundBtn = document.getElementById('soundBtn') as HTMLButtonElement
if (soundBtn) {
  soundBtn.innerHTML = viz.audio.isMuted() ? SVG_SOUND_OFF : SVG_SOUND_ON
  soundBtn.title = viz.audio.isMuted() ? 'Unmute' : 'Mute'
  soundBtn.addEventListener('click', () => {
    const muted = viz.audio.toggleMute()
    soundBtn.innerHTML = muted ? SVG_SOUND_OFF : SVG_SOUND_ON
    soundBtn.title = muted ? 'Unmute' : 'Mute'
  })
}

// Strict 1:1 mode is permanent — every pixel on screen is a function of
// a real GPU-side number. Apply the `strict` body class on boot so the
// raw-values readout panel is visible and decoration CSS stays off.
document.body.classList.add('strict')

// ─── Mode state machine ───
// Each mode picks ONE captured tensor stream and gives it the entire hero
// area. Same forward pass, totally different views. Body class drives the
// CSS that shows/hides per-mode hero overlays + scene-only side panels.
//
// 'journey' (default) is the scroll-driven cinematic flythrough — all classic
// chrome hidden, the 3D scene is the whole page. The classic modes remain
// accessible via a pill toggle and cover the side-panel + scene workflows.
type ViewMode = 'journey' | 'scene' | 'attention' | 'lens' | 'cinema'
let currentMode: ViewMode = 'journey'
document.body.classList.add('mode-journey')

// Journey handle — lazily created once the real visualizer exists
// (createNullViz proxies can't run camera paths). See wireJourney() below.
let journey: JourneyHandle | null = null
// SpatialPanels handle — Universe (scene) mode's floating 3D cards.
let spatial: SpatialPanels | null = null

// Cinema mode snaps the speed slider down so an in-flight pass becomes
// visibly cinematic without spawning a second run. We stash the prior value
// here and restore it when leaving cinema.
let cinemaPrevSpeed: string | null = null
const CINEMA_SPEED = '1'

function setMode(mode: ViewMode) {
  if (mode === currentMode) return
  if (currentMode === 'journey' && journey) journey.exit()
  if (currentMode === 'cinema' && cinemaPrevSpeed !== null) {
    speedSlider.value = cinemaPrevSpeed
    speedSlider.dispatchEvent(new Event('input'))
    cinemaPrevSpeed = null
  }
  document.body.classList.remove(`mode-${currentMode}`)
  document.body.classList.add(`mode-${mode}`)
  currentMode = mode
  for (const btn of document.querySelectorAll<HTMLButtonElement>('.mode-btn')) {
    btn.classList.toggle('active', btn.dataset.mode === mode)
  }
  if (mode === 'journey' && journey) journey.enter()
  // Spatial-panel projection is permanently disabled — panels dock via CSS
  // (game-style HUD). Left here as a no-op for symmetry; do not re-enable.
  if (spatial) spatial.enable(false)
  // Force a one-shot re-render of mode-specific views from cached state so
  // the hero isn't blank when switching mid-decode.
  if (mode === 'attention' && lastAttentionScores) {
    renderAttentionGrid(lastAttentionScores, lastAttentionKvLen)
  }
  if (mode === 'lens') refreshLensHero()
  if (mode === 'cinema') {
    if (parseInt(speedSlider.value) > parseInt(CINEMA_SPEED)) {
      cinemaPrevSpeed = speedSlider.value
      speedSlider.value = CINEMA_SPEED
      speedSlider.dispatchEvent(new Event('input'))
    }
    updateCinemaScrub()
  }
}

document.querySelectorAll<HTMLButtonElement>('.mode-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const m = btn.dataset.mode as ViewMode
    if (m) setMode(m)
  })
})

// ─── Toggle-panels button (repurposed from the old Classic-view pill) ───
// Clicking (or pressing P / Tab) shows/hides every pip + expanded card +
// the prompt + token-strip so the user gets a fully empty 3D universe.
// R resets the OrbitControls camera to its home position.
const togglePanelsBtn = document.getElementById('journey-exit')
function togglePanels(show?: boolean): void {
  const willHide = show === undefined ? !document.body.classList.contains('panels-hidden') : !show
  document.body.classList.toggle('panels-hidden', willHide)
  togglePanelsBtn?.classList.toggle('off', willHide)
}
togglePanelsBtn?.addEventListener('click', () => togglePanels())

// ─── Open-all / collapse-all-to-orbs toggle ───
// Distinct from the hide-everything button above: this expands every panel
// in the side rail at once, or collapses them all back to orbs. Output area
// and side-header stay as they are (always visible labels).
const expandAllBtn = document.getElementById('panels-expand-all')
function toggleAllPanels(expand?: boolean): void {
  const targets = document.querySelectorAll<HTMLElement>(
    '.side > [data-anchor]:not(#output):not(.side-header)'
  )
  const willExpand = expand === undefined
    ? Array.from(targets).some((p) => !p.classList.contains('expanded'))
    : expand
  targets.forEach((p) => p.classList.toggle('expanded', willExpand))
  expandAllBtn?.classList.toggle('on', willExpand)
  const label = expandAllBtn?.querySelector('.jx-label')
  if (label) label.textContent = willExpand ? 'collapse all' : 'expand all'
  if (willExpand) layoutExpandedPanels()
  else layoutBottomOrbRail()
}
expandAllBtn?.addEventListener('click', () => toggleAllPanels())

// Journey HUD has its own dismiss — orthogonal to the panels toggle.
// Clicking × on the HUD or pressing H hides just the bottom overlay.
function toggleJourneyHud(show?: boolean): void {
  const willHide = show === undefined
    ? !document.body.classList.contains('journey-hud-hidden')
    : !show
  document.body.classList.toggle('journey-hud-hidden', willHide)
}
document.getElementById('journey-hide')?.addEventListener('click', () => toggleJourneyHud())

window.addEventListener('keydown', (e) => {
  const target = e.target as HTMLElement | null
  if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return
  if (e.key === 'p' || e.key === 'P' || e.key === 'Tab') {
    e.preventDefault()
    togglePanels()
  } else if (e.key === 'o' || e.key === 'O') {
    // Expand all panels / collapse all panels back to orbs (does not hide them).
    toggleAllPanels()
  } else if (e.key === 'h' || e.key === 'H') {
    toggleJourneyHud()
  } else if (e.key === 'a' || e.key === 'A') {
    // Toggle just the ablation panel.
    ;(window as unknown as { __toggleAblatePanel?: () => void }).__toggleAblatePanel?.()
  } else if (e.key === 's' || e.key === 'S') {
    // Toggle soft Gaussian-sprite rendering — discrete spheres become
    // soft volumetric puffs. Picking still works (meshes stay raycastable).
    ;(viz as unknown as { toggleSoftMode?: () => void }).toggleSoftMode?.()
  } else if (e.key === 'k' || e.key === 'K') {
    // Toggle Phi the Storyteller — kid-mode narration overlay.
    storyteller.toggleActive()
  } else if (e.key === 'r' || e.key === 'R') {
    // Reset camera to home position via OrbitControls
    const controls = (viz as unknown as { controls?: { reset?: () => void } }).controls
    controls?.reset?.()
  } else if (e.key === 'l' || e.key === 'L') {
    // Toggle the Lessons learning-path overlay.
    ;(window as unknown as { __toggleLessons?: () => void }).__toggleLessons?.()
  }
})

/** Call after the real visualizer is created (post-WebGPU init). */
function wireJourney(): void {
  if (journey) return
  journey = createJourney(viz)
  // SpatialPanels (camera-space projection of [data-anchor] elements) is
  // intentionally NOT enabled here. Game-quality HUDs dock to viewport
  // corners; they do not orbit with the camera. CSS rules in app/index.html
  // handle per-panel docking by class. The instance is still created so
  // any code that calls into it remains a no-op.
  spatial = new SpatialPanels(viz)
  spatial.registerFromDOM()
  wirePanelToggles()
  journey.enter()
}

/** Make any element drag-to-move with viewport-clamped position persisted to
 *  localStorage. Click-vs-drag is settled by a 5px movement threshold so a
 *  small click on a pip still triggers the existing expand/collapse. While
 *  dragging the .dragging class is added (cursor: grabbing, z-index bump).
 *  Works on grid/flex children too: on first drag we snapshot the visible
 *  rect and switch the element to position:fixed so left/top take effect. */
function makeDraggable(el: HTMLElement, key: string) {
  const STORAGE_KEY = `neuropulse:panel-pos:${key}`
  const THRESH = 5

  const ensureFixed = () => {
    const cs = getComputedStyle(el)
    if (cs.position === 'fixed' || cs.position === 'absolute') return
    const rect = el.getBoundingClientRect()
    el.style.position = 'fixed'
    el.style.left = `${rect.left}px`
    el.style.top = `${rect.top}px`
    el.style.right = 'auto'
    el.style.bottom = 'auto'
    el.style.width = `${rect.width}px`
    el.style.zIndex = el.style.zIndex || '90'
  }

  // Restore saved position (if any). We override top/right via inline
  // styles so the per-class CSS dock is the *default*, not the cap.
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      const { x, y, fixed, w } = JSON.parse(saved)
      if (fixed) {
        el.style.position = 'fixed'
        if (w) el.style.width = `${w}px`
        el.style.zIndex = el.style.zIndex || '90'
      }
      el.style.left = `${x}px`
      el.style.top = `${y}px`
      el.style.right = 'auto'
      el.style.bottom = 'auto'
    }
  } catch { /* malformed JSON — fall back to CSS defaults */ }

  el.addEventListener('mousedown', (e) => {
    // Don't drag from interactive children — let buttons/inputs/sliders
    // receive their own clicks.
    const target = e.target as HTMLElement
    if (target.closest('button, input, select, textarea, a')) return
    // Only primary button.
    if (e.button !== 0) return

    const startX = e.clientX
    const startY = e.clientY
    const rect = el.getBoundingClientRect()
    const offsetX = startX - rect.left
    const offsetY = startY - rect.top
    let moved = false

    const onMove = (ev: MouseEvent) => {
      const dx = ev.clientX - startX
      const dy = ev.clientY - startY
      if (!moved && (Math.abs(dx) > THRESH || Math.abs(dy) > THRESH)) {
        moved = true
        ensureFixed()
        el.classList.add('dragging')
      }
      if (moved) {
        const x = Math.max(0, Math.min(window.innerWidth  - 24, ev.clientX - offsetX))
        const y = Math.max(0, Math.min(window.innerHeight - 24, ev.clientY - offsetY))
        el.style.left = `${x}px`
        el.style.top  = `${y}px`
        el.style.right = 'auto'
        el.style.bottom = 'auto'
      }
    }
    const onUp = (ev: MouseEvent) => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      if (moved) {
        // Mark the post-drag window so the click handler bails. Use a data
        // attribute (read by the click handler) and a delayed cleanup —
        // the click event fires AFTER mouseup so we have to be still set
        // when it lands.
        el.dataset.justDragged = '1'
        setTimeout(() => { delete el.dataset.justDragged }, 200)
        el.classList.remove('dragging')
        const r = el.getBoundingClientRect()
        const fixed = getComputedStyle(el).position === 'fixed'
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ x: r.left, y: r.top, fixed, w: r.width })) } catch { /* quota */ }
        // Swallow the click so it doesn't trigger expand/collapse on
        // panels whose own click handler runs after mouseup.
        ev.stopPropagation()
        ev.preventDefault()
      }
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  })
}

/** Click a pip panel → .expanded; click the injected × → collapse.
 * Also injects an "i" button that toggles the educational data-info caption.
 * Inputs/buttons/canvases inside an expanded panel don't collapse on click. */
function wirePanelToggles(): void {
  const panels = document.querySelectorAll<HTMLElement>('.side > [data-anchor]')
  panels.forEach((panel) => {
    // Inject the close button once per panel
    if (!panel.querySelector(':scope > .panel-close')) {
      const close = document.createElement('button')
      close.className = 'panel-close'
      close.innerHTML = '×'
      close.type = 'button'
      close.setAttribute('aria-label', 'Close')
      close.addEventListener('click', (e) => {
        e.stopPropagation()
        panel.classList.remove('expanded')
        layoutBottomOrbRail()
        layoutExpandedPanels()
      })
      panel.appendChild(close)
    }

    // Inject the educational "i" toggle + hidden caption body (if data-info exists)
    const info = panel.dataset.info
    if (info && !panel.querySelector(':scope > .panel-info')) {
      const btn = document.createElement('button')
      btn.className = 'panel-info'
      btn.textContent = 'i'
      btn.type = 'button'
      btn.setAttribute('aria-label', 'What does this panel show?')
      const body = document.createElement('div')
      body.className = 'panel-info-body'
      body.innerHTML = info  // data-info is trusted copy from our own HTML
      // "Learn more →" deep-links to the roomy glossary entry for this panel.
      const glossAnchor = panel.dataset.glossary
      if (glossAnchor) {
        const more = document.createElement('button')
        more.className = 'panel-info-more'
        more.type = 'button'
        more.textContent = 'Learn more →'
        more.addEventListener('click', (e) => { e.stopPropagation(); openGlossaryAt(glossAnchor) })
        body.appendChild(more)
      }
      btn.addEventListener('click', (e) => {
        e.stopPropagation()
        btn.classList.toggle('on')
        body.classList.toggle('on')
      })
      panel.appendChild(btn)
      panel.appendChild(body)
    }

    panel.addEventListener('click', (e) => {
      // Drag must NOT trigger expand. justDragged is set by makeDraggable
      // immediately after a real movement; cleared 200 ms later. The click
      // event fires after mouseup so it lands inside that window.
      if (panel.dataset.justDragged) return
      if (panel.classList.contains('dragging')) return
      if (!panel.classList.contains('expanded')) {
        panel.classList.add('expanded')
        layoutExpandedPanels()
        return
      }
      // Expanded: only collapse if clicking the panel *background* itself,
      // not interactive children (inputs, buttons, canvases, links).
      const t = e.target as HTMLElement
      if (t === panel) {
        panel.classList.remove('expanded')
        layoutBottomOrbRail()
        layoutExpandedPanels()
      }
    })

    // Drag-to-move with localStorage persistence. Key is class+id so each
    // panel has a stable identity across reloads.
    const key = (panel.id || '') + ':' + (panel.className || '').split(/\s+/)[0]
    makeDraggable(panel, key)
  })
  // (The ablation panel wires its own makeDraggable inside initAblationPanel
  // since it lives outside .side and is created later.)

  // Set the hover tooltip on every orb from data-title.
  document.querySelectorAll<HTMLElement>('.side > [data-anchor][data-title]').forEach(p => {
    if (!p.hasAttribute('title')) p.setAttribute('title', p.dataset.title || '')
  })

  layoutBottomOrbRail()
  window.addEventListener('resize', () => { layoutBottomOrbRail(); layoutExpandedPanels() })

  // Make all the other top-level UI containers draggable too. Grid/flex
  // children get position:fixed snapshotted on first drag (see makeDraggable
  // → ensureFixed) so they pop out cleanly without breaking layout until
  // the user actually grabs them.
  const extraDraggables: Array<[string, string]> = [
    ['.token-strip', 'token-strip'],
    ['.input-wrap', 'input-wrap'],
    ['.preset-row', 'preset-row'],
    ['.legend', 'legend'],
    ['.cinema-controls', 'cinema-controls'],
    ['#journey-hud', 'journey-hud'],
    ['#tutorialOverlay', 'tutorial-overlay'],
  ]
  for (const [sel, key] of extraDraggables) {
    const el = document.querySelector<HTMLElement>(sel)
    if (el) {
      el.style.cursor = el.style.cursor || 'grab'
      makeDraggable(el, key)
    }
  }
}

/** Lay out every [data-anchor] panel except #output (top-center answer
 *  card) and .side-header (hidden) as a centered horizontal pip rail
 *  along the bottom, just above the journey HUD. Saved positions in
 *  localStorage trump these defaults. */
function layoutBottomOrbRail() {
  const ORB_GAP = 32
  const ORB_SIZE = 16
  const RAIL_BOTTOM = 200

  const panels = Array.from(
    document.querySelectorAll<HTMLElement>('.side > [data-anchor]')
  ).filter(p => p.id !== 'output' && !p.classList.contains('side-header')
                 && !p.classList.contains('expanded')
                 && getComputedStyle(p).display !== 'none')

  const total = panels.length
  if (total === 0) return
  const totalWidth = total * ORB_SIZE + (total - 1) * ORB_GAP
  const startX = (window.innerWidth - totalWidth) / 2

  panels.forEach((p, i) => {
    // Clear any expanded-card sizing left over from layoutExpandedPanels so
    // the pip/orb styles apply cleanly again.
    p.style.transition = ''
    p.style.width = ''
    p.style.maxWidth = ''
    p.style.maxHeight = ''
    p.style.overflow = ''
    p.style.boxSizing = ''

    // Honor any user-dragged position saved in localStorage.
    const key = (p.id || '') + ':' + (p.className || '').split(/\s+/)[0]
    const saved = localStorage.getItem(`neuropulse:panel-pos:${key}`)
    if (saved) return

    p.style.left = `${startX + i * (ORB_SIZE + ORB_GAP)}px`
    p.style.bottom = `${RAIL_BOTTOM}px`
    p.style.top = 'auto'
    p.style.right = 'auto'
  })
}

/** Tile expanded panels into the empty space AROUND the centered answer card
 *  and 3D model — never on top of them. Cards fill the left and right gutters
 *  first (full height), then spill below the answer card only if needed, so
 *  opening several (or EXPAND ALL) stays tidy instead of piling over the
 *  scene. Panels the user has dragged (saved position) keep their spot. */
function layoutExpandedPanels() {
  const cards = Array.from(
    document.querySelectorAll<HTMLElement>('.side > [data-anchor].expanded')
  ).filter(p => {
    if (p.id === 'output' || p.classList.contains('side-header')) return false
    const key = (p.id || '') + ':' + (p.className || '').split(/\s+/)[0]
    return !localStorage.getItem(`neuropulse:panel-pos:${key}`)
  })
  if (cards.length === 0) return

  const GAP = 12, MARGIN = 20, TOP = 72, TARGET_W = 260, MIN_COL = 232, MAX_W = 380
  const vw = window.innerWidth, vh = window.innerHeight, bottom = vh - MARGIN

  // The centered answer card defines the "keep clear" center band.
  const out = document.getElementById('output')
  const oR = out && getComputedStyle(out).display !== 'none' ? out.getBoundingClientRect() : null
  const hasCenter = !!oR && oR.width > 40
  const cxL = hasCenter ? oR!.left - GAP : vw * 0.34
  const cxR = hasCenter ? oR!.right + GAP : vw * 0.66

  // Right edge clears the ablation panel when it's open (right-docked).
  let rightBound = vw - MARGIN
  const abl = document.querySelector<HTMLElement>('.ablate-panel.open:not(.collapsed)')
  if (abl) {
    const r = abl.getBoundingClientRect()
    if (r.left > vw * 0.4) rightBound = Math.min(rightBound, r.left - GAP)
  }

  // Three X-disjoint regions so columns can never collide: left gutter, right
  // gutter (both full height), and center-below the answer card (overflow).
  const regions: Array<{ x0: number; x1: number; top: number }> = []
  if (cxL - MARGIN >= MIN_COL) regions.push({ x0: MARGIN, x1: cxL, top: TOP })
  if (rightBound - cxR >= MIN_COL) regions.push({ x0: cxR, x1: rightBound, top: TOP })
  if (hasCenter && oR!.width >= MIN_COL) regions.push({ x0: oR!.left, x1: oR!.right, top: oR!.bottom + GAP })
  if (regions.length === 0) regions.push({ x0: MARGIN, x1: rightBound, top: TOP })

  const columns: Array<{ x: number; w: number; top: number }> = []
  for (const rg of regions) {
    const w = rg.x1 - rg.x0
    const n = Math.max(1, Math.floor((w + GAP) / (TARGET_W + GAP)))
    const cw = Math.min(MAX_W, Math.floor((w - (n - 1) * GAP) / n))
    for (let i = 0; i < n; i++) columns.push({ x: rg.x0 + i * (cw + GAP), w: cw, top: rg.top })
  }

  // Cap card height so a full column fits (tall cards scroll internally).
  const rowsPerCol = Math.ceil(cards.length / columns.length)
  const maxCardH = Math.max(96, Math.floor(((bottom - TOP) - (rowsPerCol - 1) * GAP) / rowsPerCol))

  // Masonry: each card drops into the currently-shortest column. Gutter columns
  // start higher (top=72) than the center-below column, so cards fill the
  // gutters first and only spill below the answer card when they must.
  const colY = columns.map(c => c.top)
  for (const p of cards) {
    p.style.transition = 'none'
    p.style.boxSizing = 'border-box'
    p.style.maxHeight = `${maxCardH}px`
    p.style.overflow = 'auto'
    let c = 0
    for (let k = 1; k < columns.length; k++) if (colY[k] < colY[c]) c = k
    p.style.width = `${columns[c]!.w}px`
    p.style.maxWidth = `${columns[c]!.w}px`
    const h = p.getBoundingClientRect().height
    p.style.left = `${columns[c]!.x}px`
    p.style.top = `${colY[c]}px`
    p.style.right = 'auto'
    p.style.bottom = 'auto'
    colY[c]! += h + GAP
  }
}

// ─── Replay scrubber — per-token snapshots of panel state + timeline UI ───
// Captured in the `onToken` callback. Click any chip in the token-strip to
// re-display the state that existed when that token was generated.
interface TokenSnapshot {
  index: number
  tokenText: string
  topK: TopKEntry[]
  residualNorms: number[]   // length 32
  layerDeltas: number[]     // length 32
  headActivity: number[][]  // 32 × 32 — flattened from the headHeatmap arrays
}
const replayBuffer: TokenSnapshot[] = []
let replayActiveIdx: number | null = null

function captureTokenSnapshot(tokenText: string, topK: TopKEntry[] | undefined): void {
  if (!topK) return
  replayBuffer.push({
    index: replayBuffer.length,
    tokenText,
    topK: topK.slice(0, 5).map((e) => ({ ...e })),
    residualNorms: Array.from(residualNorms),
    layerDeltas: Array.from(layerDeltas),
    headActivity: headHeatmap.map((row) => Array.from(row)),
  })
}

function clearReplayBuffer(): void {
  replayBuffer.length = 0
  replayActiveIdx = null
  // Remove any "replay-active" marker on token chips
  document.querySelectorAll('#tokenStripBody .ts-tok.replay-active').forEach((el) => {
    el.classList.remove('replay-active')
  })
}

function replaySnapshot(idx: number): void {
  const snap = replayBuffer[idx]
  if (!snap) return
  replayActiveIdx = idx

  // Restore Top-K + Confidence from cached state
  updateTopK(snap.topK)
  updateConfidence(snap.topK)

  // Restore chart buffers + force redraw. The chart update functions each
  // iterate the full Float32Array, so we overwrite the arrays first and
  // then invoke them once at layer 31 to trigger a full repaint.
  for (let L = 0; L < 32; L++) {
    residualNorms[L] = snap.residualNorms[L] ?? 0
    layerDeltas[L] = snap.layerDeltas[L] ?? 0
  }
  updateResidualChart(31, residualNorms[31] ?? 0)
  // updateDeltaChart recomputes delta from prevResidualNorm — bypass by
  // restoring layerDeltas directly and calling with the last known value.
  prevResidualNorm = residualNorms[30] ?? 0
  updateDeltaChart(31, residualNorms[31] ?? 0)
  // Heatmap is per-layer; loop
  for (let L = 0; L < 32; L++) {
    const row = snap.headActivity[L]
    if (!row) continue
    const heads = new Float32Array(row)
    updateHeatmapLayer(L, heads)
  }

  // Mark the chip in the token strip
  document.querySelectorAll('#tokenStripBody .ts-tok').forEach((el, i) => {
    el.classList.toggle('replay-active', i === idx)
  })
}

// Delegated click on the token strip — any chip becomes a scrubber button
document.getElementById('tokenStripBody')?.addEventListener('click', (e) => {
  const target = (e.target as HTMLElement)?.closest('.ts-tok') as HTMLElement | null
  if (!target) return
  // Find this chip's index among generated tokens (skip the prompt chips at the start)
  const chips = Array.from(document.querySelectorAll('#tokenStripBody .ts-tok'))
  const allIdx = chips.indexOf(target)
  if (allIdx < 0) return
  // The generated portion is the last replayBuffer.length chips
  const firstGenIdx = chips.length - replayBuffer.length
  const genIdx = allIdx - firstGenIdx
  if (genIdx < 0 || genIdx >= replayBuffer.length) return
  replaySnapshot(genIdx)
})

// ─── Guided tours (src/tours.ts) — populate list in glossary, wire play/stop ───
;(function wireTours() {
  const list = document.getElementById('tours-list')
  if (!list) return
  let runner: ReturnType<typeof createTourRunner> | null = null
  const badge = document.getElementById('tour-running-badge')

  // Seed the list from data
  list.innerHTML = TOURS.map(
    (t) => `
      <div class="tour-item" data-tour-id="${t.id}" role="button" tabindex="0">
        <div class="tour-item-body">
          <div class="tour-item-title">${t.title}</div>
          <div class="tour-item-summary">${t.summary}</div>
        </div>
        <button class="tour-item-play" type="button">▶ play</button>
      </div>
    `,
  ).join('')

  function updateCaption(caption: string, sub: string): void {
    const capEl = document.getElementById('journey-caption')
    const subEl = document.getElementById('journey-sub')
    if (capEl) {
      capEl.textContent = caption
      capEl.classList.remove('flash')
      void capEl.offsetWidth
      capEl.classList.add('flash')
    }
    if (subEl) subEl.innerHTML = sub
  }

  // ── Transport bar: pause/play · prev/next · step counter · stop ──
  // Buttons only (no new keyboard keys — check-shortcuts stays untouched).
  const transport = document.createElement('div')
  transport.id = 'tour-transport'
  transport.setAttribute('role', 'toolbar')
  transport.setAttribute('aria-label', 'Tour controls')
  transport.innerHTML = `
    <div id="tt-segments" aria-hidden="true"></div>
    <div id="tt-controls">
      <button id="tt-prev" type="button" aria-label="Previous step" title="Previous step">◀</button>
      <button id="tt-toggle" type="button" aria-label="Pause tour" title="Pause / resume">⏸</button>
      <button id="tt-next" type="button" aria-label="Next step" title="Next step">▶</button>
      <span id="tt-count" aria-live="polite">–/–</span>
      <span id="tt-paused-label">paused — you have the camera</span>
      <button id="tt-stop" type="button" aria-label="Stop tour" title="Stop tour">✕ stop</button>
    </div>
  `
  document.body.appendChild(transport)
  const ttToggle = transport.querySelector<HTMLButtonElement>('#tt-toggle')!
  const ttCount = transport.querySelector<HTMLSpanElement>('#tt-count')!
  const ttSegments = transport.querySelector<HTMLDivElement>('#tt-segments')!

  // Story-style step segments: past = full, current = fills over its hold,
  // future = empty. Click a segment to jump straight to that step.
  let segCurrent = 0
  function buildSegments(total: number): void {
    if (ttSegments.childElementCount === total) return
    ttSegments.innerHTML = ''
    for (let i = 0; i < total; i++) {
      const seg = document.createElement('button')
      seg.className = 'tt-seg'
      seg.type = 'button'
      seg.setAttribute('aria-label', `Go to step ${i + 1}`)
      seg.innerHTML = '<span class="tt-seg-fill"></span>'
      seg.addEventListener('click', () => runner?.goTo(i))
      ttSegments.appendChild(seg)
    }
  }
  function paintSegments(index: number): void {
    segCurrent = index
    ttSegments.querySelectorAll<HTMLElement>('.tt-seg-fill').forEach((f, i) => {
      if (i < index) f.style.width = '100%'
      else if (i > index) f.style.width = '0%'
      // current segment is driven by the rAF fill loop below
    })
  }
  // rAF loop fills the current segment from the runner's hold progress —
  // survives pause (frozen), resume, prev/next/goTo jumps, and speed changes.
  function segTick(): void {
    if (runner && runner.isPlaying()) {
      const fill = ttSegments.children[segCurrent]?.querySelector<HTMLElement>('.tt-seg-fill')
      if (fill) fill.style.width = `${(runner.holdProgress() * 100).toFixed(1)}%`
    }
    requestAnimationFrame(segTick)
  }
  requestAnimationFrame(segTick)

  function onStepChange(index: number, total: number, paused: boolean): void {
    buildSegments(total)
    paintSegments(index)
    ttCount.textContent = `${index + 1}/${total}`
    ttToggle.textContent = paused ? '▶' : '⏸'
    ttToggle.setAttribute('aria-label', paused ? 'Resume tour' : 'Pause tour')
    transport.classList.toggle('paused', paused)
  }

  function onTourEnd(): void {
    document.body.classList.remove('tour-running')
    transport.classList.remove('paused')
    ttSegments.querySelectorAll<HTMLElement>('.tt-seg-fill').forEach((f) => { f.style.width = '100%' })
  }

  function playTour(id: string): void {
    if (!runner) runner = createTourRunner(viz, updateCaption, onStepChange, onTourEnd)
    runner.play(id)
    document.body.classList.add('tour-running')
    // Close any covering overlays so the flythrough is actually visible —
    // the tour catalog now lives in the lessons overlay, which otherwise
    // stayed open on top of the running tour.
    document.getElementById('glossary-overlay')?.classList.remove('visible')
    document.getElementById('lessons-overlay')?.classList.remove('visible')
  }

  function stopTour(): void {
    runner?.stop() // runner fires onTourEnd for class cleanup
    document.body.classList.remove('tour-running')
  }

  transport.querySelector('#tt-prev')?.addEventListener('click', () => runner?.prev())
  transport.querySelector('#tt-next')?.addEventListener('click', () => runner?.next())
  transport.querySelector('#tt-stop')?.addEventListener('click', () => stopTour())
  ttToggle.addEventListener('click', () => {
    if (!runner) return
    if (runner.isPaused()) runner.resume()
    else runner.pause()
  })

  // Exposed so the Lessons flow can start a tour for a lesson.
  ;(window as unknown as { __playTour?: (id: string) => void }).__playTour = playTour

  list.querySelectorAll<HTMLElement>('.tour-item').forEach((item) => {
    const id = item.dataset.tourId
    if (!id) return
    item.addEventListener('click', () => playTour(id))
    item.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); playTour(id) }
    })
  })

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.body.classList.contains('tour-running')) {
      stopTour()
    }
  })

  // Manual interaction PAUSES the tour (camera yields to the user) rather than
  // silently killing it — resume/stop from the transport bar.
  window.addEventListener('pointerdown', (e) => {
    if (!document.body.classList.contains('tour-running')) return
    if (!runner || runner.isPaused()) return
    const t = e.target as HTMLElement | null
    if (t && t.closest('#journey-hud, .side, #journey-exit, #glossary-overlay, #tour-transport, #lessons-overlay, #lesson-check-card')) return
    runner.pause()
  })

  // Suppress unused warning for `badge` — its visibility is CSS-driven
  void badge
})()

// ─── "show in scene" — glossary terms pulse their 3D group ───
;(function wireGlossaryHighlights() {
  document.querySelectorAll<HTMLButtonElement>('.glossary-show').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation()
      const group = btn.dataset.highlight
      if (!group) return
      // Fade the glossary briefly so the user sees the scene pulse
      const overlay = document.getElementById('glossary-overlay')
      overlay?.classList.remove('visible')
      setTimeout(() => {
        const v = viz as unknown as { highlightGroup?: (g: string) => void }
        v.highlightGroup?.(group)
      }, 250)
    })
  })
})()

// ─── Glossary overlay (? key or HUD chip) ───
;(function wireGlossaryOverlay() {
  const overlay = document.getElementById('glossary-overlay')
  const close = document.getElementById('glossary-close')
  const chip = document.getElementById('journey-help')
  if (!overlay) return

  function toggle(show?: boolean): void {
    const willShow = show ?? !overlay!.classList.contains('visible')
    overlay!.classList.toggle('visible', willShow)
  }

  close?.addEventListener('click', () => toggle(false))
  chip?.addEventListener('click', (e) => {
    e.stopPropagation()
    toggle()
  })
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) toggle(false)
  })

  window.addEventListener('keydown', (e) => {
    const target = e.target as HTMLElement | null
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return
    if (e.key === '?' || (e.key === '/' && e.shiftKey)) {
      e.preventDefault()
      toggle()
    } else if (e.key === 'Escape' && overlay.classList.contains('visible')) {
      toggle(false)
    }
  })
})()

/** Open the glossary overlay and scroll to / flash a specific entry by id.
 *  Used by panel "Learn more →" links so detail lives in the roomy glossary
 *  rather than cramped floating cards. */
function openGlossaryAt(entryId?: string): void {
  const overlay = document.getElementById('glossary-overlay')
  if (!overlay) return
  overlay.classList.add('visible')
  if (!entryId) return
  const el = document.getElementById(entryId)
  if (!el) return
  requestAnimationFrame(() => {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    el.classList.add('gloss-flash')
    setTimeout(() => el.classList.remove('gloss-flash'), 1600)
  })
}

// ─── Lessons — sequenced learning path (📚 Learn overlay / L key) ───
;(function wireLessons() {
  const overlay = document.getElementById('lessons-overlay')
  const list = document.getElementById('lessons-list')
  if (!overlay || !list) return
  const bar = document.getElementById('lessons-progress-bar')
  const barLabel = document.getElementById('lessons-progress-label')
  const closeBtn = document.getElementById('lessons-close')
  const toggleBtn = document.getElementById('lessons-toggle')

  const KEY = 'neuropulse:lessons:progress'
  function load(): Record<string, boolean> {
    try { const s = localStorage.getItem(KEY); return s ? JSON.parse(s) : {} } catch { return {} }
  }
  function save(p: Record<string, boolean>): void {
    try { localStorage.setItem(KEY, JSON.stringify(p)) } catch { /* quota / disabled */ }
  }
  const progress = load()
  const isDone = (id: string) => progress[id] === true
  function markDone(id: string): void { progress[id] = true; save(progress) }

  // A signal-kind check waiting for a runtime event (generate / ablation / sweep).
  let armed: { lesson: Lesson; card: HTMLElement } | null = null

  const GLOSS_LABEL: Record<string, string> = {
    'gloss-token': 'Token', 'gloss-attention': 'Attention head', 'gloss-softmax': 'Softmax',
    'gloss-ablation': 'Ablation', 'gloss-sweep': 'Sweep', 'gloss-residual': 'Residual stream',
    'gloss-kv': 'KV cache', 'gloss-logitlens': 'Logit lens',
  }
  const label = (id: string) => GLOSS_LABEL[id] ?? id.replace(/^gloss-/, '')

  function render(): void {
    const total = LESSONS.length
    const done = LESSONS.filter((l) => isDone(l.id)).length
    if (bar) bar.style.width = `${Math.round((done / total) * 100)}%`
    if (barLabel) barLabel.textContent = done === total ? `all ${total} done ✓` : `${done} / ${total} done`
    const banner = done === total
      ? `<div class="lessons-done-banner">Path complete — you've seen the whole machine. Replay anything, or go free-range: sweep a layer, break a circuit, ask it something strange.</div>`
      : ''
    list!.innerHTML = banner + LESSONS.map((l, i) => `
      <div class="lesson-item ${isDone(l.id) ? 'done' : ''}" data-lesson-id="${l.id}" role="button" tabindex="0">
        <div class="lesson-status">${isDone(l.id) ? '✓' : i + 1}</div>
        <div class="lesson-body">
          <div class="lesson-title">${l.title} <span class="lesson-min">≈${l.minutes} min</span></div>
          <div class="lesson-blurb">${l.blurb}${l.requiresLive && demoMode ? ' <span class="lesson-live-tag">needs live model</span>' : ''}</div>
          <div class="lesson-objective">You'll be able to: ${l.objective}</div>
        </div>
        <button class="lesson-start" type="button">${isDone(l.id) ? 'replay' : 'start'}</button>
      </div>`).join('')
    list!.querySelectorAll<HTMLElement>('.lesson-item').forEach((el) => {
      const id = el.dataset.lessonId
      if (!id) return
      el.addEventListener('click', () => startLesson(id))
      el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); startLesson(id) }
      })
    })
  }

  function open(show?: boolean): void {
    const willShow = show ?? !overlay!.classList.contains('visible')
    if (willShow) render()
    overlay!.classList.toggle('visible', willShow)
  }

  function readingHtml(l: Lesson): string {
    if (!l.reading?.length) return ''
    return `<div class="lesson-reading">Read: ${l.reading
      .map((r) => `<button class="lesson-read-link" data-gloss="${r}" type="button">${label(r)}</button>`)
      .join(' · ')}</div>`
  }

  function removeCard(card: HTMLElement): void {
    if (armed?.card === card) armed = null
    card.remove()
    document.body.classList.remove('lesson-active')
  }

  /** The next not-yet-done lesson in path order (skipping live-only ones in
   *  demo mode), or null when the path is complete. */
  function nextLesson(afterId: string): Lesson | null {
    const rest = LESSONS.filter((l) => !isDone(l.id) && l.id !== afterId && !(l.requiresLive && demoMode))
    return rest[0] ?? null
  }

  function passLesson(l: Lesson, card: HTMLElement): void {
    markDone(l.id)
    armed = null
    card.classList.add('passed')
    const body = card.querySelector('.lcc-body') ?? card
    const next = nextLesson(l.id)
    body.innerHTML = `<div class="lcc-title">✓ Lesson complete</div>
      <div class="lcc-feedback">Nice — “${l.title}” done.</div>
      <div class="lcc-next-row">
        ${next ? `<button class="lcc-next" type="button">Next lesson: ${next.title} →</button>` : ''}
        <button class="lcc-done-close" type="button">${next ? 'later' : 'see the full path'}</button>
      </div>`
    body.querySelector('.lcc-next')?.addEventListener('click', () => {
      removeCard(card)
      if (next) startLesson(next.id)
    })
    body.querySelector('.lcc-done-close')?.addEventListener('click', () => {
      removeCard(card)
      if (!next) open(true) // path complete — show the banner
    })
  }

  function startLesson(id: string): void {
    const lesson = LESSONS.find((l) => l.id === id)
    if (!lesson) return
    open(false)
    if (lesson.tourId) {
      ;(window as unknown as { __playTour?: (t: string) => void }).__playTour?.(lesson.tourId)
    }
    presentCheck(lesson)
  }
  // Exposed so the boot fork can open the learning path directly.
  ;(window as unknown as { __openLesson?: (id: string) => void }).__openLesson = startLesson

  function presentCheck(lesson: Lesson): void {
    document.getElementById('lesson-check-card')?.remove()
    armed = null
    const card = document.createElement('div')
    card.id = 'lesson-check-card'
    card.className = 'lesson-check-card'

    if (lesson.check.kind === 'quiz') {
      const c = lesson.check
      card.innerHTML = `
        <button class="lcc-close" type="button" aria-label="Close">✕</button>
        <div class="lcc-body">
          <div class="lcc-title">${lesson.title}</div>
          ${lesson.intro ? `<div class="lcc-instruction">${lesson.intro}</div>` : ''}
          ${readingHtml(lesson)}
          <div class="lcc-q">${c.question}</div>
          <div class="lcc-opts"></div>
          <div class="lcc-feedback"></div>
        </div>`
      const optsEl = card.querySelector<HTMLElement>('.lcc-opts')!
      const fb = card.querySelector<HTMLElement>('.lcc-feedback')!
      let attempts = 0

      // Options render in shuffled order and RESHUFFLE after a wrong pick —
      // guessing costs a re-read and position memory won't help. Wrong
      // answers teach: each option carries its own `why`.
      function renderOptions(): void {
        const order = c.options.map((_, i) => i)
        for (let i = order.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1))
          ;[order[i], order[j]] = [order[j]!, order[i]!]
        }
        optsEl.innerHTML = order
          .map((oi) => `<button class="lcc-opt" data-i="${oi}" type="button">${c.options[oi]!.text}</button>`)
          .join('')
        optsEl.querySelectorAll<HTMLButtonElement>('.lcc-opt').forEach((btn) => {
          btn.addEventListener('click', () => {
            const i = Number(btn.dataset.i)
            const opt = c.options[i]!
            if (i === c.answer) {
              btn.classList.add('correct')
              fb.textContent = opt.why
              optsEl.querySelectorAll<HTMLButtonElement>('.lcc-opt').forEach((b) => { b.disabled = true })
              window.setTimeout(() => passLesson(lesson, card), 1600)
            } else {
              attempts++
              btn.classList.add('wrong')
              optsEl.querySelectorAll<HTMLButtonElement>('.lcc-opt').forEach((b) => { b.disabled = true })
              const retryCta = attempts >= 3 && lesson.tourId
                ? ` <button class="lesson-read-link lcc-retry-tour" type="button">Take the tour again →</button>`
                : ''
              fb.innerHTML = `${opt.why}${retryCta}`
              fb.querySelector('.lcc-retry-tour')?.addEventListener('click', () => {
                ;(window as unknown as { __playTour?: (t: string) => void }).__playTour?.(lesson.tourId!)
              })
              // Let the explanation land, then reshuffle for another go.
              // (Feedback clears; the tour CTA re-appears on the next miss
              // since `attempts` persists.)
              window.setTimeout(() => { fb.innerHTML = ''; renderOptions() }, 2600)
            }
          })
        })
      }
      renderOptions()
    } else if (lesson.requiresLive && demoMode) {
      // Signal check that needs the real engine — in demo mode, explain
      // rather than arm (the recorded run can't be ablated).
      card.innerHTML = `
        <button class="lcc-close" type="button" aria-label="Close">✕</button>
        <div class="lcc-body">
          <div class="lcc-title">${lesson.title}</div>
          ${lesson.intro ? `<div class="lcc-instruction">${lesson.intro}</div>` : ''}
          ${readingHtml(lesson)}
          <div class="lcc-instruction">${lesson.check.instruction}</div>
          <div class="lcc-waiting" style="color:#ffd28a">This check needs the live model — download &amp; run to complete it.</div>
        </div>`
    } else {
      const c = lesson.check
      card.innerHTML = `
        <button class="lcc-close" type="button" aria-label="Close">✕</button>
        <div class="lcc-body">
          <div class="lcc-title">${lesson.title}</div>
          ${lesson.intro ? `<div class="lcc-instruction">${lesson.intro}</div>` : ''}
          ${readingHtml(lesson)}
          <div class="lcc-instruction">${c.instruction}</div>
          <div class="lcc-waiting">Waiting for you to try it…</div>
        </div>`
      armed = { lesson, card }
    }

    card.querySelector('.lcc-close')?.addEventListener('click', () => removeCard(card))
    card.querySelectorAll<HTMLElement>('.lesson-read-link').forEach((lnk) => {
      lnk.addEventListener('click', () => {
        const g = lnk.dataset.gloss
        if (g) openGlossaryAt(g)
      })
    })
    document.body.appendChild(card)
    // Focus mode: dim non-relevant chrome while a lesson is active (CSS-only,
    // no pointer-events change — the ablation lesson needs its panel usable).
    document.body.classList.add('lesson-active')
  }

  // Consume runtime signals for the armed signal-check.
  window.addEventListener('neuropulse:lesson-signal', (e) => {
    if (!armed) return
    const check = armed.lesson.check
    if (check.kind !== 'signal') return
    const detail = (e as CustomEvent).detail as { type?: string; impact?: number }
    if (detail.type !== check.signal) return
    if (check.minImpact != null && !(typeof detail.impact === 'number' && detail.impact >= check.minImpact)) return
    passLesson(armed.lesson, armed.card)
  })

  toggleBtn?.addEventListener('click', () => open())
  closeBtn?.addEventListener('click', () => open(false))
  overlay.addEventListener('click', (e) => { if (e.target === overlay) open(false) })
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && overlay.classList.contains('visible')) open(false)
  })
  ;(window as unknown as { __toggleLessons?: () => void }).__toggleLessons = () => open()

  // "Continue where you left off" — a dismissible chip on return visits with
  // partial progress. Session-scoped dismissal, so it returns next visit.
  ;(function continueChip() {
    const done = LESSONS.filter((l) => isDone(l.id)).length
    if (done === 0 || done >= LESSONS.length) return
    try { if (sessionStorage.getItem('np:continue-dismissed') === '1') return } catch { /* ok */ }
    const next = LESSONS.find((l) => !isDone(l.id))
    if (!next) return
    const chip = document.createElement('div')
    chip.id = 'lessons-continue-chip'
    chip.innerHTML = `
      <button class="lcc-continue" type="button">Continue: ${next.title} →</button>
      <button class="lcc-continue-x" type="button" aria-label="Dismiss">✕</button>`
    document.body.appendChild(chip)
    const dismiss = () => {
      chip.remove()
      try { sessionStorage.setItem('np:continue-dismissed', '1') } catch { /* ok */ }
    }
    chip.querySelector('.lcc-continue')?.addEventListener('click', () => { dismiss(); startLesson(next.id) })
    chip.querySelector('.lcc-continue-x')?.addEventListener('click', dismiss)
  })()
})()

// ─── Preset-hint floating toast (educational annotation for each preset) ───
;(function wirePresetHints() {
  let toast: HTMLElement | null = null
  let hideTimer: number | null = null

  function showHint(html: string): void {
    if (!toast) {
      toast = document.createElement('div')
      toast.className = 'preset-hint-toast'
      document.body.appendChild(toast)
    }
    toast.innerHTML = html
    // Allow the next frame to paint before applying .visible so transition plays
    requestAnimationFrame(() => toast!.classList.add('visible'))
    if (hideTimer) window.clearTimeout(hideTimer)
    hideTimer = window.setTimeout(() => {
      toast!.classList.remove('visible')
    }, 6500)
  }

  document.querySelectorAll<HTMLButtonElement>('.preset-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const hint = chip.dataset.hint
      if (hint) showHint(hint)
    })
  })
})()

// ─── Welcome overlay (first visit) ───
// (The old first-visit "Four things to try" welcome overlay was folded into
// Lesson 0 "Get your bearings" — see src/lessons.ts. Its localStorage key
// np:welcome-dismissed is intentionally orphaned so returning users see no
// surprise overlays.)

// Repurpose the old 🔬 button as a manual "Run validation test" trigger.
// The HF cross-validation suite no longer runs automatically on boot; the
// user kicks it off with this button when they want the accuracy report.
const validateBtn = document.getElementById('accurateBtn') as HTMLButtonElement
if (validateBtn) {
  validateBtn.innerHTML = SVG_VALIDATE
  validateBtn.title = 'Run HF cross-validation suite (prints accuracy report to console)'
  validateBtn.addEventListener('click', async () => {
    if (isValidating || isRunning) return
    isValidating = true
    validateBtn.innerHTML = SVG_VALIDATE_BUSY
    validateBtn.disabled = true
    goBtn.disabled = true
    const prevGoText = goBtn.textContent
    goBtn.textContent = 'Validating...'
    try {
      await runValidationSilently()
    } catch (e) {
      console.warn('[validation] suite failed', e)
    } finally {
      isValidating = false
      validateBtn.disabled = false
      validateBtn.innerHTML = SVG_VALIDATE
      goBtn.disabled = false
      goBtn.textContent = prevGoText || 'Think'
    }
  })
}

// Screenshot button — mode-aware. Captures the active hero so the saved PNG
// matches what the user is actually looking at.
const screenshotBtn = document.getElementById('screenshotBtn') as HTMLButtonElement
if (screenshotBtn) {
  screenshotBtn.addEventListener('click', () => {
    let dataUrl: string
    const suffix = currentMode
    if (currentMode === 'attention') {
      const c = document.getElementById('attnGridCanvas') as HTMLCanvasElement | null
      dataUrl = c?.toDataURL('image/png') ?? viz.getScreenshot()
    } else {
      // Scene, cinema, lens — use brain canvas (lens hero is DOM, fall back)
      dataUrl = viz.getScreenshot()
    }
    const a = document.createElement('a')
    a.href = dataUrl
    a.download = `neuropulse-${suffix}-${Date.now()}.png`
    a.click()
  })
}

// HF cross-validation suite — runs once at boot, prints a single accuracy
// report to the console. No UI, no toggle. Stores the report on
// window.__validationReport so it can be inspected from devtools.
async function runValidationSilently() {
  if (!engine || !engine.runValidationSuite) return
  console.log('[validation] running HF cross-validation suite (~30s)...')
  const report = await engine.runValidationSuite()

  if (!report.hasReference) {
    console.warn('[validation] /reference.json missing — re-run tools/dump_phi3_reference.py')
    return
  }

  const lines: string[] = []
  const m = report.main
  lines.push('═══════════════ Neuropulse — Accuracy Report ═══════════════')
  lines.push('GPU: q4f16_1 Phi-3-mini   Reference: HF fp16 Phi-3-mini')
  lines.push('')
  lines.push('[1] Tokenizer (GPU buildChatPrompt == HF apply_chat_template):')
  lines.push(`    main: ${m.tokenizerAgrees ? 'OK — input ids match HF byte-for-byte' : 'MISMATCH — BPE divergence'}`)
  if (!m.tokenizerAgrees) {
    lines.push(`      GPU ids: [${m.gpuInputIds.join(', ')}]`)
    lines.push(`      HF  ids: [${m.hfInputIds.join(', ')}]`)
  }
  const sweepTokOk = report.sweep.every((s) => s.tokenizerAgrees)
  lines.push(`    sweep: ${sweepTokOk ? `OK — ${report.sweep.length}/${report.sweep.length} prompts match (ASCII, numbers, code, Japanese, emoji, JSON)` : 'MISMATCH'}`)
  if (!sweepTokOk) {
    for (const s of report.sweep) {
      if (!s.tokenizerAgrees) {
        lines.push(`      BAD: ${JSON.stringify(s.prompt).slice(0, 60)}`)
      }
    }
  }
  lines.push(`    longContext: ${report.longContext.tokenizerAgrees ? `OK — ${report.longContext.hfInputIds.length} tokens` : 'MISMATCH'}`)
  lines.push('')
  lines.push('[2] Attention shader equivalence (layer 31, online softmax ≡')
  lines.push('    explicit softmax reference in-shader, same q4 weights):')
  const ae = report.attentionEquivalence
  const attnTag = ae.passed ? 'PASS' : 'FAIL'
  lines.push(
    `    ${attnTag} — relErr=${(ae.relError * 100).toFixed(4)}% ` +
    `(target <1e-2%)  l2=${ae.l2Error.toExponential(2)}  kv_len=${ae.kvLen}`
  )
  lines.push('')
  lines.push('[3] Logit agreement vs HF (teacher-forced: feed HF tokens at')
  lines.push('    each step, compare full-vocab top-K distributions):')
  lines.push('    step  gpu_id  hf_id  match  JSD          top5∩')
  for (const t of m.tokenDiffs) {
    lines.push(
      `    ${String(t.step).padStart(4)}  ${String(t.gpuId).padStart(6)}  ${String(t.hfId).padStart(5)}  ${t.match ? ' OK' : ' XX'}    ${t.jsd.toExponential(2)}    ${t.top5Overlap}/5`
    )
  }
  const mainMeanJsd = m.tokenDiffs.length > 0
    ? m.tokenDiffs.reduce((s, t) => s + t.jsd, 0) / m.tokenDiffs.length
    : NaN
  lines.push(
    `    → main: ${m.topMatches}/${m.tokenDiffs.length} top-1 match, meanJSD=${mainMeanJsd.toExponential(2)}`
  )
  lines.push('')
  lines.push('[4] Multi-prompt sweep (15 prompts × 5 steps, top-10):')
  lines.push('    prompt                                       tok  matches  meanJSD')
  for (const s of report.sweep) {
    const label = (s.prompt.length > 42 ? s.prompt.slice(0, 39) + '...' : s.prompt).padEnd(42)
    const tok = s.tokenizerAgrees ? 'OK ' : 'BAD'
    const matchStr = `${s.topMatches}/${s.tokenDiffs.length}`.padStart(6)
    lines.push(`    ${label}  ${tok}  ${matchStr}   ${s.meanJsd.toExponential(2)}`)
  }
  const sweepAllMatches = report.sweep.reduce((a, s) => a + s.topMatches, 0)
  const sweepAllSteps = report.sweep.reduce((a, s) => a + s.tokenDiffs.length, 0)
  const sweepAllJsd = report.sweep.length > 0
    ? report.sweep.reduce((a, s) => a + s.meanJsd, 0) / report.sweep.length
    : NaN
  lines.push(`    → sweep: ${sweepAllMatches}/${sweepAllSteps} top-1 match, meanJSD=${sweepAllJsd.toExponential(2)}`)
  lines.push('')
  lines.push('[5] Long context (paged KV cache past one page boundary):')
  const lc = report.longContext
  lines.push(`    prompt: ${JSON.stringify(lc.prompt.slice(0, 60) + '...')}`)
  lines.push(`    tok=${lc.tokenizerAgrees ? 'OK' : 'BAD'}  tokens=${lc.hfInputIds.length}  kv_len=${lc.kvLen}  pages=${Math.ceil(lc.kvLen / 16)}`)
  lines.push(`    → ${lc.topMatches}/${lc.tokenDiffs.length} top-1 match, meanJSD=${lc.meanJsd.toExponential(2)}`)
  lines.push('')
  lines.push('[6] Sampling self-test (xorshift32 inverse-CDF from softmax):')
  const sst = report.samplingSelfTest
  lines.push(
    `    ${sst.passed ? 'PASS' : 'FAIL'} — ${sst.numSamples} samples @ T=${sst.temperature}  ` +
    `unique_ids=${sst.uniqueIds}  JSD=${sst.empiricalJsd.toExponential(2)}  maxL1=${sst.maxL1Error.toFixed(4)}`
  )
  lines.push('')
  lines.push('[7] Per-layer hidden state vs HF fp16 (full 3072 dims, last')
  lines.push('    prompt position). This is the q4 quantization floor —')
  lines.push('    individual layer residuals drift as q4 error compounds,')
  lines.push('    but the final RMSNorm before lm_head absorbs the scale')
  lines.push('    drift so the logits remain close (see [3]).')
  lines.push('    layer    relErr      cosine     gpu_norm    hf_norm')
  for (const d of m.layerDiffs) {
    const name = d.layer === -1 ? 'embed' : `L${d.layer}`
    lines.push(
      `    ${name.padStart(5)}  ${(d.relError * 100).toFixed(2).padStart(7)}%  ` +
      `${d.cosine.toFixed(4).padStart(8)}  ${d.gpuNorm.toFixed(2).padStart(9)}  ${d.hfNorm.toFixed(2).padStart(9)}`
    )
  }
  lines.push('')
  lines.push(`SUMMARY: ${report.summary}`)
  lines.push('═══════════════════════════════════════════════════════════════')
  console.log(lines.join('\n'))
  ;(window as Window & { __validationReport?: typeof report }).__validationReport = report

  // Debug: direct embedding readback for a known token to compare against HF
  // without the validator's multi-step snapshot path.
  if (engine?.debugEmbedToken) {
    ;(window as unknown as { __debugEmbedToken: (id: number) => Promise<Float32Array> })
      .__debugEmbedToken = engine.debugEmbedToken
  }
}

// PCA layout auto-load: fetched once at boot, applied as the permanent layout.
// No toggle — points are always arranged by functional similarity (Phi-3 L0
// qkv_proj/down_proj weight columns → PCA(2)).
async function loadPcaLayoutPermanent() {
  try {
    const res = await fetch('/pca-layout.json')
    const ct = res.headers.get('content-type') ?? ''
    if (!res.ok || !ct.includes('json')) {
      console.warn('[pca] /pca-layout.json missing — run tools/build_pca_layout.py')
      return
    }
    const data = await res.json() as { residual: number[][], ffn: number[][] }
    const r = new Float32Array(data.residual.length * 2)
    for (let i = 0; i < data.residual.length; i++) {
      r[i * 2] = data.residual[i][0]
      r[i * 2 + 1] = data.residual[i][1]
    }
    const f = new Float32Array(data.ffn.length * 2)
    for (let i = 0; i < data.ffn.length; i++) {
      f[i * 2] = data.ffn[i][0]
      f[i * 2 + 1] = data.ffn[i][1]
    }
    viz.setPcaLayout(r, f)
    // layout applied silently
  } catch (e) {
    console.warn('[pca] load failed', e)
  }
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))

// ─── Top-K display ───
const topkBars = document.getElementById('topkBars')!
const TOP_K_COLORS = ['#00e5ff', '#5eead4', '#06b6d4', '#ff8c42', '#8a8170']

function updateTopK(entries: TopKEntry[]) {
  if (!topkBars) return
  topkBars.innerHTML = entries.map((e, i) => {
    const pct = (e.prob * 100).toFixed(1)
    const width = Math.max(2, e.prob * 100)
    const token = e.token.replace(/</g, '&lt;').replace(/>/g, '&gt;')
    const display = token.trim() || JSON.stringify(e.token).slice(1, -1)
    return `<div class="topk-row">
      <span class="topk-token">"${display}"</span>
      <div class="topk-bar-bg"><div class="topk-bar-fill" style="width:${width}%;background:${TOP_K_COLORS[i]}"></div></div>
      <span class="topk-pct">${pct}%</span>
    </div>`
  }).join('')
}

// ─── KV Cache display ───
const kvBar = document.getElementById('kvBar') as HTMLDivElement
const kvInfo = document.getElementById('kvInfo')!

function updateKVCache(position: number, totalPages: number, usedPages: number) {
  const pct = (usedPages / totalPages * 100).toFixed(1)
  if (kvBar) kvBar.style.width = `${pct}%`
  if (kvInfo) kvInfo.textContent = `${usedPages} / ${totalPages} pages (pos ${position})`
}

// ─── Prefill overlay ───
let prefillOverlay: HTMLDivElement | null = null

function showPrefillToken(index: number, total: number, token: string) {
  if (!prefillOverlay) {
    prefillOverlay = document.createElement('div')
    prefillOverlay.className = 'prefill-overlay'
    document.querySelector('.brain-wrap')!.appendChild(prefillOverlay)
  }
  const pct = ((index + 1) / total * 100).toFixed(0)
  const display = token.replace(/</g, '&lt;').replace(/>/g, '&gt;')
  prefillOverlay.innerHTML = `
    Reading prompt: <span class="prefill-token">${display}</span> <span style="color:#8a8170">${index + 1}/${total}</span>
    <div class="prefill-bar-bg"><div class="prefill-bar-fill" style="width:${pct}%"></div></div>
  `
  prefillOverlay.style.display = 'block'
}

function hidePrefill() {
  if (prefillOverlay) {
    prefillOverlay.style.opacity = '0'
    prefillOverlay.style.transition = 'opacity 0.3s'
    setTimeout(() => {
      if (prefillOverlay) {
        prefillOverlay.style.display = 'none'
        prefillOverlay.style.opacity = '1'
      }
    }, 300)
  }
}

// ─── Confidence meter (entropy-based) ───
const confidenceBar = document.getElementById('confidenceBar') as HTMLDivElement
const confidenceVal = document.getElementById('confidenceVal')!

function updateConfidence(topK: TopKEntry[]) {
  // Shannon entropy: H = -Σ p·log2(p), max for 5 uniform = log2(5) ≈ 2.32
  let entropy = 0
  for (const e of topK) {
    if (e.prob > 1e-8) entropy -= e.prob * Math.log2(e.prob)
  }
  const maxEntropy = Math.log2(topK.length)
  const confidence = Math.max(0, 1 - entropy / maxEntropy)
  const pct = (confidence * 100).toFixed(0)

  if (confidenceBar) {
    confidenceBar.style.width = `${pct}%`
    if (confidence > 0.7) confidenceBar.style.background = '#5eead4'
    else if (confidence > 0.4) confidenceBar.style.background = '#06b6d4'
    else confidenceBar.style.background = '#ff8c42'
  }
  if (confidenceVal) {
    confidenceVal.textContent = `${pct}%`
    confidenceVal.style.color = confidence > 0.7 ? '#5eead4' : confidence > 0.4 ? '#06b6d4' : '#ff8c42'
  }
}

// ─── Head activity heatmap (32 layers × 32 heads) ───
const heatmapGrid = document.getElementById('heatmapGrid')!
const headHeatmap: Float32Array[] = []  // [layer][head] = activation 0-1
for (let L = 0; L < 32; L++) headHeatmap.push(new Float32Array(32))

function initHeatmap() {
  if (!heatmapGrid) return
  heatmapGrid.innerHTML = ''
  for (let L = 0; L < 32; L++) {
    const row = document.createElement('div')
    row.className = 'heatmap-row'
    row.id = `hm-row-${L}`
    for (let h = 0; h < 32; h++) {
      const cell = document.createElement('div')
      cell.className = 'heatmap-cell'
      cell.id = `hm-${L}-${h}`
      cell.title = `L${L} H${h}`
      row.appendChild(cell)
    }
    heatmapGrid.appendChild(row)
  }
}
initHeatmap()

function updateHeatmapLayer(layer: number, heads: Float32Array) {
  headHeatmap[layer] = heads
  for (let h = 0; h < 32; h++) {
    const cell = document.getElementById(`hm-${layer}-${h}`)
    if (!cell) continue
    const v = heads[h]
    if (v > 0.6) { cell.style.background = `rgba(94,234,212,${0.3 + v * 0.7})`; cell.style.boxShadow = `0 0 3px rgba(94,234,212,${v * 0.3})` }
    else if (v > 0.25) { cell.style.background = `rgba(255,140,66,${0.2 + v * 0.5})`; cell.style.boxShadow = 'none' }
    else { cell.style.background = `rgba(94,234,212,${0.02 + v * 0.06})`; cell.style.boxShadow = 'none' }
  }
}

// ─── Residual stream chart ───
const residualCanvas = document.getElementById('residualChart') as HTMLCanvasElement
const residualCtx = residualCanvas?.getContext('2d')
const residualNorms = new Float32Array(32)

function updateResidualChart(layer: number, value: number) {
  residualNorms[layer] = value
  if (!residualCtx) return

  const w = residualCanvas.width, h = residualCanvas.height
  residualCtx.clearRect(0, 0, w, h)

  // Find range for scaling
  let maxVal = 0
  for (let i = 0; i <= layer; i++) if (residualNorms[i] > maxVal) maxVal = residualNorms[i]
  if (maxVal < 0.01) maxVal = 1

  // Draw bars
  const barW = w / 32
  for (let i = 0; i < 32; i++) {
    const v = residualNorms[i] / maxVal
    const barH = v * h
    const t = i / 31
    // Gradient: cyan → violet
    const r = Math.round(0 + t * 124)
    const g = Math.round(229 - t * 171)
    const b = Math.round(255)
    residualCtx.fillStyle = i <= layer ? `rgba(${r},${g},${b},${0.3 + v * 0.7})` : 'rgba(244,236,223,0.04)'
    residualCtx.fillRect(i * barW, h - barH, barW - 1, barH)
  }
}

// ─── Layer contribution delta chart ───
const deltaCanvas = document.getElementById('deltaChart') as HTMLCanvasElement
const deltaCtx = deltaCanvas?.getContext('2d')
const layerDeltas = new Float32Array(32)
let prevResidualNorm = 0

// ─── Run recorder (?record=1) — capture a real run for demo-mode playback ───
// Dev/curation tool: taps the generate() callback stream via mergeCallbacks
// (same seam the storyteller uses) and offers the result as a JSON download.
// See src/recording.ts for the schema; committed files live in
// public/recordings/ and are CI-validated by tools/verify-recordings.mjs.
let recorderRunMeta: { prompt: string; mode: 'ask' | 'complete' } | null = null
const recorder: Recorder | null = new URLSearchParams(location.search).has('record')
  ? createRecorder({
      getResidualNorms: () => residualNorms,
      getLayerDeltas: () => layerDeltas,
      getHeadHeatmap: () => headHeatmap,
      getFingerprint: () => {
        const sha = typeof __BUILD_SHA__ !== 'undefined' ? __BUILD_SHA__ : 'dev'
        const gpu = document.getElementById('fp-gpu')?.textContent ?? 'gpu unknown'
        return `${sha} · ${gpu}`
      },
    })
  : null

if (recorder) {
  // Show the download strip whenever a run settles with captured tokens.
  window.addEventListener('neuropulse:lesson-signal', (e) => {
    const detail = (e as CustomEvent).detail as { type?: string }
    if (detail.type !== 'generate' || recorder.tokenCount() === 0) return
    let strip = document.getElementById('record-strip')
    if (!strip) {
      strip = document.createElement('div')
      strip.id = 'record-strip'
      strip.style.cssText =
        'position:fixed;bottom:64px;left:50%;transform:translateX(-50%);z-index:120;' +
        'background:rgba(12,14,20,.94);border:1px solid rgba(255,154,31,.5);border-radius:10px;' +
        'padding:10px 14px;font-family:JetBrains Mono,monospace;font-size:12px;color:#ffd28a;' +
        'display:flex;gap:10px;align-items:center;'
      document.body.appendChild(strip)
    }
    strip.innerHTML = `<span>● rec — ${recorder.tokenCount()} tokens captured</span>`
    const btn = document.createElement('button')
    btn.textContent = 'Download recording JSON'
    btn.style.cssText =
      'background:rgba(255,154,31,.18);color:#ffd28a;border:1px solid rgba(255,154,31,.5);' +
      'border-radius:5px;padding:5px 12px;cursor:pointer;font:inherit;'
    btn.addEventListener('click', () => {
      const rec = recorder.build(recorderRunMeta?.prompt ?? '', recorderRunMeta?.mode ?? 'complete')
      if (!rec) return
      const blob = new Blob([JSON.stringify(rec)], { type: 'application/json' })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = `np-recording-${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(a.href)
    })
    strip.appendChild(btn)
  })
}

function updateDeltaChart(layer: number, residualValue: number) {
  const delta = Math.abs(residualValue - prevResidualNorm)
  layerDeltas[layer] = delta
  prevResidualNorm = residualValue

  if (!deltaCtx) return
  const w = deltaCanvas.width, h = deltaCanvas.height
  deltaCtx.clearRect(0, 0, w, h)

  let maxDelta = 0
  for (let i = 0; i <= layer; i++) if (layerDeltas[i] > maxDelta) maxDelta = layerDeltas[i]
  if (maxDelta < 0.001) maxDelta = 1

  const barW = w / 32
  for (let i = 0; i < 32; i++) {
    const v = layerDeltas[i] / maxDelta
    const barH = v * h
    // High delta = warm color, low = cool
    if (i <= layer) {
      if (v > 0.6) deltaCtx.fillStyle = `rgba(255,184,120,${0.4 + v * 0.6})`
      else if (v > 0.3) deltaCtx.fillStyle = `rgba(94,234,212,${0.3 + v * 0.5})`
      else deltaCtx.fillStyle = `rgba(255,140,66,${0.2 + v * 0.4})`
    } else {
      deltaCtx.fillStyle = 'rgba(244,236,223,0.04)'
    }
    deltaCtx.fillRect(i * barW, h - barH, barW - 1, barH)
  }
}

// ─── Residual stream strip (3072 dims × 32 layers diverging heatmap) ───
// Each layer's post-FFN residual (3072 f32) is compressed to 768 bins by
// averaging every 4 dims, then drawn as one row of a 768×64 canvas (each
// row spans 2 pixels for visibility). Cyan = positive, magenta = negative,
// brightness = magnitude. Reset per token so each heatmap reflects the
// residual stream being built up layer-by-layer for the CURRENT token.
const resStripCanvas = document.getElementById('resStripCanvas') as HTMLCanvasElement
const resStripCtx = resStripCanvas?.getContext('2d')
const resStripInfo = document.getElementById('resStripInfo')!
const RES_STRIP_BINS = 768
const RES_STRIP_ROW_H = 2
let resStripImg: ImageData | null = null

function clearResStrip() {
  if (!resStripCtx) return
  resStripCtx.fillStyle = '#08060f'
  resStripCtx.fillRect(0, 0, resStripCanvas.width, resStripCanvas.height)
  resStripImg = resStripCtx.getImageData(0, 0, resStripCanvas.width, resStripCanvas.height)
}

function updateResStripLayer(layer: number, activations: Float32Array) {
  if (!resStripCtx) return
  if (!resStripImg) clearResStrip()
  const img = resStripImg!
  // Compress 3072 → 768 by averaging groups of 4. Also track magnitude
  // stats for contrast scaling.
  const bins = new Float32Array(RES_STRIP_BINS)
  let maxAbs = 0
  for (let b = 0; b < RES_STRIP_BINS; b++) {
    const base = b * 4
    const v = (activations[base] + activations[base + 1] + activations[base + 2] + activations[base + 3]) * 0.25
    bins[b] = v
    const a = Math.abs(v)
    if (a > maxAbs) maxAbs = a
  }
  if (maxAbs < 1e-6) maxAbs = 1
  // Draw into rows [layer*2, layer*2+1]
  const rowTop = layer * RES_STRIP_ROW_H
  for (let r = 0; r < RES_STRIP_ROW_H; r++) {
    const y = rowTop + r
    if (y >= resStripCanvas.height) break
    for (let b = 0; b < RES_STRIP_BINS; b++) {
      const v = bins[b] / maxAbs  // in [-1, 1]
      let red: number, grn: number, blu: number
      if (v >= 0) {
        // positive → cyan (0, 229, 255)
        red = Math.round(v * 0)
        grn = Math.round(v * 229)
        blu = Math.round(v * 255)
      } else {
        // negative → magenta (236, 72, 153)
        const a = -v
        red = Math.round(a * 236)
        grn = Math.round(a * 72)
        blu = Math.round(a * 153)
      }
      const idx = (y * resStripCanvas.width + b) * 4
      img.data[idx] = red
      img.data[idx + 1] = grn
      img.data[idx + 2] = blu
      img.data[idx + 3] = 255
    }
  }
  resStripCtx.putImageData(img, 0, 0)
  if (resStripInfo) resStripInfo.textContent = `L${layer} max=${maxAbs.toFixed(1)}`
}

// ─── Per-head attention pattern (Layer 31, 32 heads × kvLen) ───
// Reads the full attention-scores tensor [32 layers × 32 heads × 256 slots]
// from onAllAttentionScores and draws the last layer as a 32×kvLen heatmap.
// The last layer is the model's final "decision layer" — where each head is
// looking gives the most interpretable attention pattern. Cells are drawn
// in the attention-score color scale (black=0 → cyan=max per row).
const attnCanvas = document.getElementById('attnCanvas') as HTMLCanvasElement
const attnCtx = attnCanvas?.getContext('2d')
const attnInfo = document.getElementById('attnInfo')!
const ATTN_LAYER = 31
const ATTN_HEADS = 32
const ATTN_MAX_SLOTS = 256

function updateAttentionHeatmap(scores: Float32Array, kvLen: number) {
  // Cache for the new attention grid hero (used by Attention mode)
  lastAttentionScores = scores
  lastAttentionKvLen = kvLen
  if (currentMode === 'attention') renderAttentionGrid(scores, kvLen)
  // Slice layer 31 out of the full buffer for the legacy mini canvas.
  const cols = Math.min(kvLen, ATTN_MAX_SLOTS)
  if (cols === 0) return
  const layerOff = ATTN_LAYER * ATTN_HEADS * ATTN_MAX_SLOTS
  const rows = new Float32Array(ATTN_HEADS * cols)
  for (let h = 0; h < ATTN_HEADS; h++) {
    for (let s = 0; s < cols; s++) rows[h * cols + s] = scores[layerOff + h * ATTN_MAX_SLOTS + s]!
  }
  updateAttentionHeatmapL31(rows, kvLen)
}

/** Paint the layer-31 mini attention canvas from head-major rows
 *  (32 × min(kvLen, 256)). Engine-free — demo playback calls this directly
 *  with dequantized recorded rows. */
function updateAttentionHeatmapL31(rows: Float32Array, kvLen: number) {
  if (!attnCtx) return
  attnCtx.fillStyle = '#08060f'
  attnCtx.fillRect(0, 0, attnCanvas.width, attnCanvas.height)
  const cols = Math.min(kvLen, ATTN_MAX_SLOTS)
  if (cols === 0) return
  const cellW = attnCanvas.width / cols
  const cellH = attnCanvas.height / ATTN_HEADS
  for (let h = 0; h < ATTN_HEADS; h++) {
    const headBase = h * cols
    // Per-head max for contrast scaling (softmax sums to 1, but a head that
    // attends uniformly would have ~1/kvLen per slot, so we re-scale).
    let rowMax = 0
    for (let s = 0; s < cols; s++) {
      const v = rows[headBase + s]!
      if (v > rowMax) rowMax = v
    }
    if (rowMax < 1e-6) rowMax = 1
    for (let s = 0; s < cols; s++) {
      const v = rows[headBase + s]! / rowMax
      // cyan → violet gradient based on magnitude, dark for low
      const r = Math.round(v * 124)
      const g = Math.round(v * 229)
      const b = Math.round(60 + v * 195)
      const alpha = 0.08 + v * 0.92
      attnCtx.fillStyle = `rgba(${r},${g},${b},${alpha})`
      attnCtx.fillRect(s * cellW, h * cellH, Math.max(1, cellW), Math.max(1, cellH))
    }
  }
  if (attnInfo) attnInfo.textContent = `kv=${kvLen}`
}

// ─── Attention grid hero (Attention mode) ───
// Renders the full [32 layers × 32 heads × 256 slots] tensor that
// onAllAttentionScores already captures into a 1024-cell grid: each cell is
// one head's attention pattern as a tiny per-token bar. Click to inspect.
const attnGridCanvas = document.getElementById('attnGridCanvas') as HTMLCanvasElement
const attnGridCtx = attnGridCanvas?.getContext('2d')
const attnDetailCanvas = document.getElementById('attnDetailCanvas') as HTMLCanvasElement
const attnDetailCtx = attnDetailCanvas?.getContext('2d')
const attnDetailHead = document.getElementById('attnDetailHead')!
const attnDetailTokens = document.getElementById('attnDetailTokens')!
const ATTN_GRID_LAYERS = 32
const ATTN_GRID_HEADS = 32
const ATTN_GRID_SLOTS = 256
let lastAttentionScores: Float32Array | null = null
let lastAttentionKvLen = 0
let selectedHeadLayer = -1
let selectedHeadIdx = -1

function renderAttentionGrid(scores: Float32Array, kvLen: number) {
  if (!attnGridCtx) return
  const W = attnGridCanvas.width  // 1024
  const H = attnGridCanvas.height // 1024
  const cellW = W / ATTN_GRID_HEADS // 32 px
  const cellH = H / ATTN_GRID_LAYERS // 32 px
  const img = attnGridCtx.createImageData(W, H)
  const cols = Math.max(1, Math.min(kvLen, ATTN_GRID_SLOTS))

  for (let L = 0; L < ATTN_GRID_LAYERS; L++) {
    for (let h = 0; h < ATTN_GRID_HEADS; h++) {
      const headBase = (L * ATTN_GRID_HEADS + h) * ATTN_GRID_SLOTS
      // Per-head normalization (heads can be very peaky or very diffuse)
      let rowMax = 0
      for (let s = 0; s < cols; s++) {
        const v = scores[headBase + s]
        if (v > rowMax) rowMax = v
      }
      if (rowMax < 1e-6) rowMax = 1
      const cellX = h * cellW
      const cellY = L * cellH
      // Draw cell as a wide horizontal bar inside the cell — use most of the
      // cell vertically, leave a 2px margin on each side for grid lines.
      for (let py = 1; py < cellH - 1; py++) {
        for (let px = 1; px < cellW - 1; px++) {
          // Map px → token slot
          const slotF = (px - 1) / (cellW - 3) * (cols - 1)
          const slot = Math.min(cols - 1, Math.max(0, Math.round(slotF)))
          const v = scores[headBase + slot] / rowMax
          // Color: violet → cyan based on magnitude
          const r = Math.round(v * v * 60 + 4)
          const g = Math.round(v * 220 + 6)
          const b = Math.round(v * 255 + 16)
          const idx = ((cellY + py) * W + (cellX + px)) * 4
          img.data[idx] = r
          img.data[idx + 1] = g
          img.data[idx + 2] = b
          img.data[idx + 3] = 255
        }
      }
      // Cell border (faint)
      for (let px = 0; px < cellW; px++) {
        const top = (cellY * W + cellX + px) * 4
        const bot = ((cellY + cellH - 1) * W + cellX + px) * 4
        img.data[top + 3] = 80; img.data[top + 0] = 0; img.data[top + 1] = 30; img.data[top + 2] = 50
        img.data[bot + 3] = 80
      }
      for (let py = 0; py < cellH; py++) {
        const lt = ((cellY + py) * W + cellX) * 4
        img.data[lt + 3] = 80
      }
    }
  }
  attnGridCtx.putImageData(img, 0, 0)

  // Overlay selection ring on top of the ImageData (stroke after putImageData)
  if (selectedHeadLayer >= 0 && selectedHeadIdx >= 0) {
    attnGridCtx.strokeStyle = '#5eead4'
    attnGridCtx.lineWidth = 2
    attnGridCtx.strokeRect(
      selectedHeadIdx * cellW + 1,
      selectedHeadLayer * cellH + 1,
      cellW - 2,
      cellH - 2,
    )
  }

  // If a cell is selected, also refresh the detail panel for the new tensor
  if (selectedHeadLayer >= 0 && selectedHeadIdx >= 0) {
    renderAttentionDetail(scores, kvLen, selectedHeadLayer, selectedHeadIdx)
  }
}

function renderAttentionDetail(scores: Float32Array, kvLen: number, L: number, h: number) {
  if (!attnDetailCtx) return
  const W = attnDetailCanvas.width
  const H = attnDetailCanvas.height
  attnDetailCtx.fillStyle = '#08060f'
  attnDetailCtx.fillRect(0, 0, W, H)

  const cols = Math.max(1, Math.min(kvLen, ATTN_GRID_SLOTS))
  const headBase = (L * ATTN_GRID_HEADS + h) * ATTN_GRID_SLOTS

  // Find max for scaling
  let rowMax = 0
  for (let s = 0; s < cols; s++) {
    const v = scores[headBase + s]
    if (v > rowMax) rowMax = v
  }
  if (rowMax < 1e-6) rowMax = 1

  // Draw bars
  const barW = W / cols
  for (let s = 0; s < cols; s++) {
    const v = scores[headBase + s] / rowMax
    const barH = v * (H - 12)
    const x = s * barW
    const grad = attnDetailCtx.createLinearGradient(0, H - barH, 0, H)
    grad.addColorStop(0, `rgba(94,234,212,${0.3 + v * 0.7})`)
    grad.addColorStop(1, `rgba(255,140,66,${0.2 + v * 0.5})`)
    attnDetailCtx.fillStyle = grad
    attnDetailCtx.fillRect(x, H - barH, Math.max(1, barW - 1), barH)
  }

  // Header
  const peak = Array.from({ length: cols }, (_, s) => ({ s, v: scores[headBase + s] }))
    .sort((a, b) => b.v - a.v)[0]
  attnDetailHead.innerHTML = `<b>L${L} · H${h}</b> &nbsp; kv=${kvLen} &nbsp; peak=token #${peak.s} (${(peak.v * 100).toFixed(1)}%)`

  // Highlight token strip
  highlightTokensFromAttention(headBase, scores, cols)
}

function highlightTokensFromAttention(headBase: number, scores: Float32Array, cols: number) {
  // Map attention score → highlight on the token strip + the detail tokens box
  const toks = collectedTokens.slice(0, cols)
  // Find the max
  let mx = 0
  for (let s = 0; s < cols; s++) if (scores[headBase + s] > mx) mx = scores[headBase + s]
  if (mx < 1e-6) mx = 1
  // Render detail tokens
  attnDetailTokens.innerHTML = toks.map((t, i) => {
    const w = scores[headBase + i] / mx
    const cls = w > 0.45 ? 'tok tok-hot' : 'tok'
    const safe = (t || '·').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '↵')
    const op = (0.25 + w * 0.75).toFixed(2)
    return `<span class="${cls}" style="opacity:${op}">${safe}</span>`
  }).join(' ')
  // Also highlight the bottom token strip
  const stripChildren = document.querySelectorAll<HTMLSpanElement>('.token-strip .ts-tok')
  stripChildren.forEach((el, i) => {
    el.classList.toggle('hot', i < cols && scores[headBase + i] / mx > 0.45)
  })
}

if (attnGridCanvas) {
  attnGridCanvas.addEventListener('click', (e) => {
    const rect = attnGridCanvas.getBoundingClientRect()
    const cx = (e.clientX - rect.left) / rect.width * attnGridCanvas.width
    const cy = (e.clientY - rect.top) / rect.height * attnGridCanvas.height
    const cellW = attnGridCanvas.width / ATTN_GRID_HEADS
    const cellH = attnGridCanvas.height / ATTN_GRID_LAYERS
    const h = Math.min(ATTN_GRID_HEADS - 1, Math.max(0, Math.floor(cx / cellW)))
    const L = Math.min(ATTN_GRID_LAYERS - 1, Math.max(0, Math.floor(cy / cellH)))
    selectedHeadLayer = L
    selectedHeadIdx = h
    if (lastAttentionScores) {
      renderAttentionGrid(lastAttentionScores, lastAttentionKvLen)
    }
  })
}

// ─── Token strip (always-visible bottom row) ───
// Tracks prompt + generated tokens in display order so the attention hero
// can highlight which past tokens a head is looking at.
const tokenStripBody = document.getElementById('tokenStripBody')!
const collectedTokens: string[] = []

function tokenStripAppendGenerated(tok: string) {
  collectedTokens.push(tok)
  rerenderTokenStrip()
}

function rerenderTokenStrip() {
  // The first chunk are prompt tokens (purple), the rest are generated (cyan).
  // We track the boundary as the length when we last pushed a prompt — store
  // it on the function as a property.
  const out: string[] = []
  for (let i = 0; i < collectedTokens.length; i++) {
    const t = collectedTokens[i]
    const cls = i < tokenStripPromptLen ? 'ts-tok prompt' : 'ts-tok gen'
    const safe = (t || '·').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '↵').replace(/\t/g, '→')
    out.push(`<span class="${cls}">${safe}</span>`)
  }
  tokenStripBody.innerHTML = out.join('')
}

let tokenStripPromptLen = 0
function tokenStripStart(prompt: string) {
  collectedTokens.length = 0
  const parts = prompt.split(/(\s+)/).filter((s) => s.length > 0)
  for (const p of parts) collectedTokens.push(p)
  tokenStripPromptLen = collectedTokens.length
  rerenderTokenStrip()
}
// ─── Lens hero (Logit Lens mode) ───
// Renders all 32 layers as a vertical stack. Layers in LENS_LAYERS get live
// updates from onLayerLogitLens; layer 31 gets the actual sampled token.
// Other layers stay greyed out (the engine doesn't compute lens for them
// to keep decode fast).
const lensHeroBody = document.getElementById('lensHeroBody')!
const lensHeroRows: HTMLDivElement[] = []
const lensHeroState: { token: string; sampled: boolean }[] = []
const LENS_SAMPLED = new Set([0, 4, 8, 12, 16, 20, 24, 28, 31])

function initLensHero() {
  if (!lensHeroBody) return
  lensHeroBody.innerHTML = ''
  lensHeroRows.length = 0
  lensHeroState.length = 0
  for (let L = 0; L < 32; L++) {
    const row = document.createElement('div')
    row.className = 'lens-row'
    const sampled = LENS_SAMPLED.has(L)
    if (!sampled) row.classList.add('unsampled')
    if (L === 31) row.classList.add('final')
    row.innerHTML = `
      <span class="lens-row-layer">L${L}</span>
      <span class="lens-row-tok">—</span>
      <div class="lens-row-bar"><div style="width:${sampled ? 4 : 0}%"></div></div>
    `
    lensHeroBody.appendChild(row)
    lensHeroRows.push(row)
    lensHeroState.push({ token: '—', sampled })
  }
}
initLensHero()

function lensHeroSet(layer: number, token: string) {
  if (layer < 0 || layer >= 32) return
  const row = lensHeroRows[layer]
  if (!row) return
  const tokEl = row.querySelector('.lens-row-tok') as HTMLSpanElement
  const barEl = row.querySelector('.lens-row-bar > div') as HTMLDivElement
  const disp = (token || '·').trim() || JSON.stringify(token).slice(1, -1)
  tokEl.textContent = disp.length > 32 ? disp.slice(0, 31) + '…' : disp
  // Confidence proxy: how "certain" the layer was by depth (later layers more confident)
  const conf = Math.min(100, ((layer + 1) / 32) * 100)
  barEl.style.width = `${conf}%`
  row.classList.add('fresh')
  setTimeout(() => row.classList.remove('fresh'), 1100)
  lensHeroState[layer] = { token, sampled: true }
}

function refreshLensHero() {
  // Re-apply cached state (used when switching to lens mode mid-decode)
  for (let L = 0; L < 32; L++) {
    const s = lensHeroState[L]
    if (!s || s.token === '—') continue
    const row = lensHeroRows[L]
    if (!row) continue
    const tokEl = row.querySelector('.lens-row-tok') as HTMLSpanElement
    tokEl.textContent = s.token.length > 32 ? s.token.slice(0, 31) + '…' : s.token
  }
}

// ─── Cinema controls (Cinematic mode) ───
const cinemaPlayBtn = document.getElementById('cinemaPlayBtn') as HTMLButtonElement | null
const cinemaStepBtn = document.getElementById('cinemaStepBtn') as HTMLButtonElement | null
const cinemaCamBtn = document.getElementById('cinemaCamBtn') as HTMLButtonElement | null
const cinemaScrubFill = document.getElementById('cinemaScrubFill') as HTMLDivElement | null
const cinemaStepLabel = document.getElementById('cinemaStepLabel') as HTMLSpanElement | null
const cinemaDispLabel = document.getElementById('cinemaDispLabel') as HTMLSpanElement | null

let cinemaPaused = false
let cinemaStepRequested = false
let cinemaCameraFly = false
let cinemaLastDispatch = 0

function updateCinemaScrub() {
  if (!cinemaScrubFill) return
  const total = viz.totalDispatches || 292
  const cur = viz.dispatchCount || 0
  const pct = Math.min(100, (cur / total) * 100)
  cinemaScrubFill.style.width = `${pct}%`
  if (cinemaDispLabel) cinemaDispLabel.textContent = `${cur} / ${total}`
  if (cinemaStepLabel) {
    const stepNames = ['QKV', 'RoPE', 'KV', 'Attn', 'OProj', '+Norm', 'FFN-Up', 'FFN-Dn', '+Norm']
    const step = Math.max(0, Math.min(8, ((cur - 2) % 9)))
    const layer = Math.max(0, Math.floor((cur - 2) / 9))
    cinemaStepLabel.textContent = `L${layer} · ${stepNames[step]}`
  }
}

if (cinemaPlayBtn) {
  cinemaPlayBtn.addEventListener('click', () => {
    cinemaPaused = !cinemaPaused
    cinemaPlayBtn.textContent = cinemaPaused ? '▶' : '⏸'
  })
}
if (cinemaStepBtn) {
  cinemaStepBtn.addEventListener('click', () => {
    cinemaStepRequested = true
  })
}
if (cinemaCamBtn) {
  cinemaCamBtn.addEventListener('click', () => {
    cinemaCameraFly = !cinemaCameraFly
    cinemaCamBtn.style.background = cinemaCameraFly ? 'rgba(244,236,223,0.18)' : 'rgba(244,236,223,0.06)'
    cinemaCamBtn.style.color = cinemaCameraFly ? '#5eead4' : '#cbc1ad'
    viz.setCinematicCamera(cinemaCameraFly)
  })
}

// Pump cinema scrub on a slow interval — the dispatch counter updates from
// the engine callbacks, but we want the UI bar to follow even when no
// callback fires (between dispatches).
setInterval(() => {
  if (currentMode === 'cinema') {
    if (viz.dispatchCount !== cinemaLastDispatch) {
      cinemaLastDispatch = viz.dispatchCount
      updateCinemaScrub()
    }
  }
}, 60)

// ─── Prompt presets ───
document.querySelectorAll<HTMLButtonElement>('.preset-chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    const p = chip.dataset.prompt
    if (!p || isValidating) return
    if (isRunning) {
      // Fill the input so the user can see what will run, and pulse the chip
      promptInput.value = p
      chip.style.borderColor = '#f59e0b'
      chip.style.color = '#f59e0b'
      setTimeout(() => { chip.style.borderColor = ''; chip.style.color = '' }, 600)
      // Show a brief toast warning
      showPresetToast('Generation in progress — click again to start a new one', chip, p)
      return
    }
    promptInput.value = p
    startInference()
  })
})

/** Toast + second-click confirmation for preset chips during active generation */
let _presetPending: { chip: HTMLButtonElement; prompt: string; timer: number } | null = null
function showPresetToast(msg: string, chip: HTMLButtonElement, prompt: string) {
  // If user clicks the same chip twice, treat as confirmation
  if (_presetPending && _presetPending.chip === chip) {
    clearTimeout(_presetPending.timer)
    _presetPending = null
    removePresetToast()
    isRunning = false
    setTimeout(() => { promptInput.value = prompt; startInference() }, 100)
    return
  }
  // Clear any previous pending
  if (_presetPending) { clearTimeout(_presetPending.timer); _presetPending = null }
  removePresetToast()
  // Show toast
  const toast = document.createElement('div')
  toast.id = 'presetToast'
  toast.style.cssText = 'position:fixed;bottom:80px;left:50%;transform:translateX(-50%);background:rgba(30,25,20,0.95);border:1px solid #f59e0b;color:#f4ecdf;font-family:"JetBrains Mono",monospace;font-size:0.68rem;padding:8px 16px;border-radius:6px;z-index:9999;pointer-events:none;animation:fadeIn 0.2s ease'
  toast.textContent = msg
  document.body.appendChild(toast)
  const timer = window.setTimeout(() => { _presetPending = null; removePresetToast() }, 3000)
  _presetPending = { chip, prompt, timer }
}
function removePresetToast() {
  document.getElementById('presetToast')?.remove()
}

// ─── Tutorial overlay (first visit) ───
const TUTORIAL_KEY = 'np:tutorial-dismissed-v1'
function maybeShowTutorial() {
  try {
    if (localStorage.getItem(TUTORIAL_KEY)) return
  } catch { /* localStorage blocked — show anyway */ }
  const overlay = document.getElementById('tutorialOverlay')
  if (!overlay) return
  overlay.style.display = 'block'
  const dismiss = document.getElementById('tutorialDismiss')
  dismiss?.addEventListener('click', () => {
    overlay.style.display = 'none'
    try { localStorage.setItem(TUTORIAL_KEY, '1') } catch { /* ignore */ }
  }, { once: true })
}
// Defer until after the loading gate clears
setTimeout(maybeShowTutorial, 2200)

// ─── Recording mode ───
// 10s MediaRecorder of the active hero canvas (brain in scene/cinema, the
// attention grid in attention mode, full hero DOM via canvas in lens mode).
// Skips lens mode (no canvas to capture) — falls back to brain in that case.
const recordBtn = document.getElementById('recordBtn') as HTMLButtonElement | null
let isRecording = false
function pickRecordCanvas(): HTMLCanvasElement | null {
  if (currentMode === 'attention') return attnGridCanvas
  return canvas // brain canvas
}
if (recordBtn) {
  recordBtn.addEventListener('click', async () => {
    if (isRecording) return
    const target = pickRecordCanvas()
    if (!target) return
    const stream = (target as HTMLCanvasElement).captureStream(30)
    let mr: MediaRecorder
    try {
      mr = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9', videoBitsPerSecond: 8_000_000 })
    } catch {
      try {
        mr = new MediaRecorder(stream, { mimeType: 'video/webm' })
      } catch {
        console.warn('[record] MediaRecorder not supported')
        return
      }
    }
    const chunks: Blob[] = []
    mr.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data) }
    mr.onstop = () => {
      const blob = new Blob(chunks, { type: 'video/webm' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `neuropulse-${currentMode}-${Date.now()}.webm`
      a.click()
      URL.revokeObjectURL(url)
    }
    isRecording = true
    recordBtn.innerHTML = SVG_RECORD_ACTIVE
    recordBtn.style.color = '#ef4444'
    mr.start()
    setTimeout(() => {
      mr.stop()
      isRecording = false
      recordBtn.innerHTML = SVG_RECORD
      recordBtn.style.color = ''
    }, 10_000)
  })
}

// ─── Logit Lens grid (9 layers: 0,4,8,12,16,20,24,28,31) ───
// Shows what each sampled layer would predict as the next token if the
// model "stopped thinking" at that layer. Demonstrates how predictions
// sharpen through the depth of the network — early layers produce
// garbage, late layers converge on the actual output.
const LENS_LAYERS_UI = [0, 4, 8, 12, 16, 20, 24, 28, 31]
const lensGrid = document.getElementById('lensGrid')!
const lensInfo = document.getElementById('lensInfo')!
const lensCells: Record<number, HTMLDivElement> = {}

function initLensGrid() {
  if (!lensGrid) return
  lensGrid.innerHTML = ''
  for (const L of LENS_LAYERS_UI) {
    const cell = document.createElement('div')
    cell.className = 'lens-cell'
    cell.innerHTML = `<span class="lens-lay">L${L}</span><span class="lens-tok">—</span>`
    lensGrid.appendChild(cell)
    lensCells[L] = cell
  }
}
initLensGrid()

function displayLensToken(layer: number, token: string) {
  // Always feed the lens hero (active in lens mode)
  lensHeroSet(layer, token)
  // Legacy 9-cell grid (scene mode side panel)
  const cell = lensCells[layer]
  if (!cell) return
  const tokEl = cell.querySelector('.lens-tok') as HTMLSpanElement
  // Escape whitespace-only tokens so they're visible
  const disp = token.trim() || JSON.stringify(token).slice(1, -1)
  tokEl.textContent = disp.length > 10 ? disp.slice(0, 9) + '…' : disp
  cell.classList.add('fresh')
  setTimeout(() => cell.classList.remove('fresh'), 1200)
}

function clearLensGrid() {
  for (const L of LENS_LAYERS_UI) {
    const cell = lensCells[L]
    if (!cell) continue
    const tokEl = cell.querySelector('.lens-tok') as HTMLSpanElement
    tokEl.textContent = '—'
    cell.classList.remove('fresh')
  }
}

// ─── Raw GPU state readout (strict mode only) ───
// Displays the exact f32 values that back the visualization: first 16 dims
// of the most recent residual, the raw L2 norm, the top-1 logit id + value,
// the current kv_len, and the most recent lens layer + predicted token id.
// This is the "proof of accuracy" panel — nothing is smoothed or averaged.
const rawGrid = document.getElementById('rawGrid')!
const rawVec = document.getElementById('rawVec')!
interface RawState {
  lastLayer: number
  resNorm: number
  resMin: number
  resMax: number
  resMean: number
  topId: number
  topProb: number
  kvLen: number
  lensLayer: number
  lensId: number
  resHead16: number[]
}
const rawState: RawState = {
  lastLayer: -1, resNorm: 0, resMin: 0, resMax: 0, resMean: 0,
  topId: -1, topProb: 0, kvLen: 0, lensLayer: -1, lensId: -1,
  resHead16: [],
}

function renderRawReadout() {
  if (!rawGrid) return
  const rows: [string, string][] = [
    ['layer',      rawState.lastLayer.toString()],
    ['res_norm',   rawState.resNorm.toFixed(4)],
    ['res_min',    rawState.resMin.toFixed(4)],
    ['res_max',    rawState.resMax.toFixed(4)],
    ['res_mean',   rawState.resMean.toExponential(2)],
    ['top_id',     rawState.topId.toString()],
    ['top_prob',   rawState.topProb.toFixed(4)],
    ['kv_len',     rawState.kvLen.toString()],
    ['lens_layer', rawState.lensLayer.toString()],
    ['lens_id',    rawState.lensId.toString()],
  ]
  rawGrid.innerHTML = rows
    .map(([k, v]) => `<div class="raw-row"><span class="k">${k}</span><span class="v">${v}</span></div>`)
    .join('')
  const head = rawState.resHead16
  if (head.length > 0) {
    rawVec.innerHTML = `<b>residual[0..15]</b> = [${head.map((v) => v.toFixed(3)).join(', ')}]`
  }
}

function captureRawResidual(layer: number, activations: Float32Array) {
  let mn = Infinity, mx = -Infinity, sum = 0, sq = 0
  for (let i = 0; i < activations.length; i++) {
    const v = activations[i]
    if (v < mn) mn = v
    if (v > mx) mx = v
    sum += v
    sq += v * v
  }
  rawState.lastLayer = layer
  rawState.resNorm = Math.sqrt(sq)
  rawState.resMin = mn
  rawState.resMax = mx
  rawState.resMean = sum / activations.length
  rawState.resHead16 = Array.from(activations.slice(0, 16))
  renderRawReadout()
}

// ─── Shareable links ───
const shareBtn = document.getElementById('shareBtn') as HTMLButtonElement
if (shareBtn) {
  shareBtn.addEventListener('click', () => {
    const prompt = promptInput.value.trim() || 'What is consciousness?'
    const url = new URL(window.location.href)
    url.searchParams.set('q', prompt)
    navigator.clipboard.writeText(url.toString()).then(() => {
      shareBtn.textContent = '✓'
      setTimeout(() => { shareBtn.textContent = '🔗' }, 1500)
    }).catch(() => {
      // Fallback: select the URL in the input
      promptInput.value = url.toString()
      promptInput.select()
    })
  })
}

// Check URL for shared prompt on load
function getSharedPrompt(): string | null {
  const params = new URLSearchParams(window.location.search)
  return params.get('q')
}

// ─── Runtime fingerprint footer — what code, what GPU, what browser ───
// Empirical-lab basics: identify the exact build + hardware running this
// forward pass so a numerical mismatch report is reproducible.
async function populateFingerprint(): Promise<void> {
  const fp = document.getElementById('fingerprint') as HTMLDivElement | null
  if (!fp) return
  const detail = document.getElementById('fp-detail') as HTMLDivElement | null

  const set = (id: string, v: string) => {
    const el = document.getElementById(id)
    if (el) el.textContent = v
  }

  // Build identifiers (substituted by Vite at build time).
  const sha = typeof __BUILD_SHA__ !== 'undefined' ? __BUILD_SHA__ : 'dev'
  const branch = typeof __BUILD_BRANCH__ !== 'undefined' ? __BUILD_BRANCH__ : 'dev'
  const dirty = typeof __BUILD_DIRTY__ !== 'undefined' ? __BUILD_DIRTY__ : false
  const builtAt = typeof __BUILD_TIME__ !== 'undefined' ? __BUILD_TIME__ : '—'
  if (dirty) fp.classList.add('dirty')

  set('fp-sha', `${sha}${dirty ? '+dirty' : ''}`)
  set('fp-build', `${sha}${dirty ? ' (dirty tree)' : ''}`)
  set('fp-branch', branch)
  set('fp-time', builtAt)

  // Browser + platform.
  const ua = navigator.userAgent
  // Best-effort browser sniff for the summary; the full UA goes in detail.
  let browserName = 'unknown'
  const m = ua.match(/(Chrome|Edg|Firefox|Safari|OPR)\/(\d+)/)
  if (m) browserName = `${m[1].replace('Edg', 'Edge').replace('OPR', 'Opera')} ${m[2]}`
  set('fp-browser', `${browserName} · ${ua.slice(0, 100)}${ua.length > 100 ? '…' : ''}`)
  set('fp-platform', navigator.platform || '—')
  set('fp-screen', `${screen.width}×${screen.height} @${window.devicePixelRatio}x`)

  // GPU — request a fresh adapter (cheap; we discard the device).
  try {
    if (!navigator.gpu) {
      set('fp-gpu', 'no WebGPU')
      return
    }
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) {
      set('fp-gpu', 'no adapter')
      return
    }
    type AdapterInfo = { vendor?: string; architecture?: string; device?: string; description?: string }
    let info: AdapterInfo = {}
    try {
      const reqInfo = (adapter as unknown as { requestAdapterInfo?: () => Promise<AdapterInfo> }).requestAdapterInfo
      if (reqInfo) info = await reqInfo.call(adapter)
      else if ((adapter as unknown as { info?: AdapterInfo }).info) {
        info = (adapter as unknown as { info: AdapterInfo }).info
      }
    } catch { /* ignore */ }
    const vendor = info.vendor || 'unknown'
    const arch = info.architecture || ''
    const device = info.device || ''
    const summary = arch ? `${vendor} · ${arch}` : vendor
    set('fp-gpu', `gpu ${summary}`)
    set('fp-gpu-vendor', vendor)
    set('fp-gpu-arch', arch || '—')
    set('fp-gpu-device', device || info.description || '—')

    const limits = adapter.limits as unknown as { maxComputeWorkgroupStorageSize?: number }
    set('fp-wg', limits.maxComputeWorkgroupStorageSize ? `${limits.maxComputeWorkgroupStorageSize.toLocaleString()} B` : '—')
    set('fp-f16', adapter.features.has('shader-f16') ? 'yes' : 'no')
  } catch (err) {
    console.warn('[fingerprint] gpu probe failed:', err)
    set('fp-gpu', 'gpu probe failed')
  }

  // Click to expand/collapse the detail panel.
  fp.addEventListener('click', () => {
    if (detail) detail.hidden = !detail.hidden
  })
}
populateFingerprint()

// ─── Model storage modal — see + delete the cached Phi-3 weights ───
const storageBtn = document.getElementById('storageBtn') as HTMLButtonElement | null
const storageModal = document.getElementById('storageModal') as HTMLDivElement | null
const storageTotalEl = document.getElementById('storageTotal')
const storageCacheEl = document.getElementById('storageCache')
const storageOpfsEl = document.getElementById('storageOpfs')
const storageFilesEl = document.getElementById('storageFiles')
const storageCancelBtn = document.getElementById('storageCancel') as HTMLButtonElement | null
const storageDeleteBtn = document.getElementById('storageDelete') as HTMLButtonElement | null

function fmtBytes(b: number): string {
  if (b <= 0) return '0 B'
  if (b < 1024) return `${b} B`
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`
  if (b < 1024 ** 3) return `${(b / 1024 ** 2).toFixed(1)} MB`
  return `${(b / 1024 ** 3).toFixed(2)} GB`
}

let storageDeleteArmed = false
let storageDeleteResetTimer: number | null = null

async function refreshStorageStats(): Promise<void> {
  if (storageTotalEl) storageTotalEl.textContent = 'measuring…'
  if (storageCacheEl) storageCacheEl.textContent = '—'
  if (storageOpfsEl) storageOpfsEl.textContent = '—'
  if (storageFilesEl) storageFilesEl.textContent = '—'
  const s = await getStoredWeightStats()
  if (storageTotalEl) storageTotalEl.textContent = fmtBytes(s.totalBytes)
  if (storageCacheEl) storageCacheEl.textContent = `${fmtBytes(s.cacheBytes)} · ${s.shardCount} shard${s.shardCount === 1 ? '' : 's'}`
  if (storageOpfsEl) storageOpfsEl.textContent = `${fmtBytes(s.opfsBytes)} · ${s.opfsFileCount} file${s.opfsFileCount === 1 ? '' : 's'}`
  if (storageFilesEl) storageFilesEl.textContent = String(s.shardCount + s.opfsFileCount)
}

function openStorageModal(): void {
  if (!storageModal) return
  storageModal.hidden = false
  // Reset any armed/post-delete state from a previous open.
  storageDeleteArmed = false
  storageDeleteDone = false
  if (storageDeleteResetTimer) {
    clearTimeout(storageDeleteResetTimer)
    storageDeleteResetTimer = null
  }
  if (storageDeleteBtn) {
    storageDeleteBtn.classList.remove('confirm')
    storageDeleteBtn.disabled = false
    storageDeleteBtn.textContent = 'Delete cached model'
  }
  refreshStorageStats()
}

function closeStorageModal(): void {
  if (storageModal) storageModal.hidden = true
}

storageBtn?.addEventListener('click', openStorageModal)
storageCancelBtn?.addEventListener('click', closeStorageModal)
storageModal?.addEventListener('click', (e) => {
  // Click backdrop (the modal element itself, not the card) to close.
  if (e.target === storageModal) closeStorageModal()
})

let storageDeleteDone = false
storageDeleteBtn?.addEventListener('click', async () => {
  if (!storageDeleteBtn) return
  // After a successful delete, the button becomes a reload trigger.
  if (storageDeleteDone) {
    location.reload()
    return
  }
  // Two-tap confirmation: first click arms, second click commits.
  if (!storageDeleteArmed) {
    storageDeleteArmed = true
    storageDeleteBtn.classList.add('confirm')
    storageDeleteBtn.textContent = 'Tap again to confirm'
    if (storageDeleteResetTimer) clearTimeout(storageDeleteResetTimer)
    storageDeleteResetTimer = window.setTimeout(() => {
      storageDeleteArmed = false
      storageDeleteBtn.classList.remove('confirm')
      storageDeleteBtn.textContent = 'Delete cached model'
    }, 4000)
    return
  }
  if (storageDeleteResetTimer) {
    clearTimeout(storageDeleteResetTimer)
    storageDeleteResetTimer = null
  }
  storageDeleteBtn.disabled = true
  storageDeleteBtn.textContent = 'deleting…'
  try {
    await clearStoredWeights()
  } catch (err) {
    console.warn('[storage] delete failed:', err)
  }
  await refreshStorageStats()
  storageDeleteBtn.disabled = false
  storageDeleteBtn.classList.remove('confirm')
  storageDeleteBtn.textContent = 'Reload to re-download'
  storageDeleteArmed = false
  storageDeleteDone = true
})

// ─── Boot screen (HTML-based, no flash of raw UI) ───
function showBootLoading() {
  const gate = document.getElementById('bootGate')
  const loading = document.getElementById('bootLoading')
  if (gate) { gate.classList.remove('visible'); gate.classList.add('hidden') }
  if (loading) { loading.classList.remove('hidden'); loading.classList.add('visible') }
}

function updateLoading(p: LoadProgress) {
  showBootLoading()
  const bar = document.getElementById('bootBar')
  const pct = document.getElementById('bootPct')
  const size = document.getElementById('bootSize')
  const msg = document.getElementById('bootMsg')
  const cache = document.getElementById('bootCache')

  if (bar) bar.style.width = `${Math.min(100, p.percent).toFixed(1)}%`
  if (pct) pct.textContent = `${Math.min(100, p.percent).toFixed(1)}%`

  if (size && p.bytesTotal > 0) {
    const loaded = (p.bytesLoaded / 1e6).toFixed(0)
    const total = (p.bytesTotal / 1e6).toFixed(0)
    size.textContent = `${loaded} / ${total} MB`
  }

  if (msg) msg.textContent = p.message

  if (cache && p.cacheHit) {
    cache.textContent = 'Loading from browser cache (instant)'
    cache.style.opacity = '1'
  }
}

function hideLoading() {
  const boot = document.getElementById('bootScreen')
  const container = document.querySelector('.container')
  if (boot) {
    boot.classList.add('fade-out')
    setTimeout(() => boot.remove(), 600)
  }
  if (container) {
    container.classList.add('revealed')
  }
  // First-run learner-level chooser appears once the boot has settled
  // (no-ops when a level was already chosen).
  maybeShowLevelChooser()
}

// ─── Demo mode (no WebGPU fallback) ───
function tokenize(text: string): string[] {
  return text.split(/(\s+)/).filter(t => t.trim().length > 0)
}

function appendToken(text: string) {
  const cursor = output.querySelector('.cursor')
  const span = document.createElement('span')
  span.className = 'token'
  span.textContent = text
  if (cursor) output.insertBefore(span, cursor)
  else output.appendChild(span)
  output.scrollTop = output.scrollHeight
}

// ─── Ask mode: wrap user question with Phi-3 chat template + docs context ───
// The docs come from src/docs.md (imported as raw text). When the user
// clicks Ask instead of Think, we route their question through this template
// so Phi-3 answers as an informed explainer, not a raw completion.
const ASK_SYSTEM_PREAMBLE =
  'You are Phi-3-mini running live in the user\'s browser inside Neuropulse — ' +
  'an interpretability visualizer where the user can literally see every ' +
  'tensor you compute, rendered as a 3D scene around them as you generate. ' +
  'Answer their question about transformers, this app, or yourself. Be ' +
  'concise (2–5 sentences unless asked for depth). Use plain language. ' +
  'If you don\'t know something, say so — don\'t invent interpretability ' +
  'research. Reference the docs below when relevant.\n\n' +
  '=== NEUROPULSE REFERENCE DOCS ===\n'

function buildAskPrompt(question: string): string {
  const system = ASK_SYSTEM_PREAMBLE + NEUROPULSE_DOCS + '\n=== END DOCS ==='
  return `<|system|>\n${system}\n<|end|>\n<|user|>\n${question}\n<|end|>\n<|assistant|>\n`
}

// ─── Real Phi-3 inference ───
async function runRealInference(prompt: string, mode: 'think' | 'ask' = 'think') {
  if (!engine) return
  isRunning = true
  goBtn.disabled = true
  goBtn.textContent = mode === 'ask' ? 'Asking...' : 'Thinking...'
  viz.audio.resume()

  output.innerHTML = ''
  if (mode === 'ask') {
    // Q&A formatting: the user's original question (not the full templated
    // prompt) in a cyan italic pull-quote, then an "Answer" label.
    const q = document.createElement('div')
    q.className = 'qa-question'
    q.textContent = prompt
    output.appendChild(q)
    const aLabel = document.createElement('div')
    aLabel.className = 'qa-answer-label'
    aLabel.textContent = 'Answer'
    output.appendChild(aLabel)
  } else {
    const promptEcho = document.createElement('div')
    promptEcho.style.cssText = 'color:#ff8c42;margin-bottom:16px;font-size:0.8rem;font-style:italic;opacity:0.7'
    promptEcho.textContent = `> ${prompt}`
    output.appendChild(promptEcho)
  }
  const cursor = document.createElement('span')
  cursor.className = 'cursor'
  output.appendChild(cursor)

  // In Ask mode the REAL prompt sent to the engine is the chat-templated
  // wrap around the user's question + docs system prompt. The user's
  // original question is what we display + tokenize for the token strip.
  const realPrompt = mode === 'ask' ? buildAskPrompt(prompt) : prompt
  const inputTokens = tokenize(prompt)
  viz.setInputTokens(inputTokens)
  tokenStripStart(prompt)

  const t0 = performance.now()
  totalTokens = 0

  // Reset charts for new run
  residualNorms.fill(0)
  layerDeltas.fill(0)
  prevResidualNorm = 0
  for (let L = 0; L < 32; L++) headHeatmap[L].fill(0)
  clearResStrip()
  // Reset replay buffer — per-token snapshots captured during this run
  clearReplayBuffer()
  clearLensGrid()
  initLensHero()
  lastAttentionScores = null
  lastAttentionKvLen = 0
  selectedHeadLayer = -1
  selectedHeadIdx = -1

  try {
  const baseCallbacks: InferenceCallbacks = {
    async onLayer(layer, step, _stepName, activations) {
      // Build role-specific activation data from GPU readback
      let data: LayerActivation

      if (activations) {
        switch (step) {
          case 0: { // QKV Matmul: 9216 values → Q portion → 32 attn heads
            const heads = reduceQKVForAttnHeads(activations)
            data = {
              attnHeads: heads,
              ffnGroups: new Float32Array(16),
              residual: 0.1,
            }
            // Update heatmap with QKV head data
            updateHeatmapLayer(layer, heads)
            break
          }
          case 3: { // Attention output: 3072 → 32 heads
            const heads = reduceForAttnHeads(activations)
            data = {
              attnHeads: heads,
              ffnGroups: new Float32Array(16),
              residual: 0.2,
            }
            // Update per-head attention heatmap (the heatmap IS the full
            // attention visualization; we no longer draw decorative beams).
            updateHeatmapLayer(layer, heads)
            break
          }
          case 5: { // Add+Norm (attn): 3072 → residual + dense column
            const resVal = reduceForResidual(activations)
            const residualVec = normalizeFull(activations, false)
            data = {
              attnHeads: new Float32Array(32),
              ffnGroups: reduceForFFNGroups(new Float32Array(0)),
              residual: resVal,
              residualVec,
            }
            updateResidualChart(layer, resVal)
            updateDeltaChart(layer, resVal)
            break
          }
          case 6: { // FFN Gate+Up: 8192 → dense FFN slab + 16-group fallback
            const ffnVec = normalizeFull(activations, false)
            data = {
              attnHeads: new Float32Array(32),
              ffnGroups: reduceForFFNGroups(activations),
              residual: 0.15,
              ffnVec,
            }
            break
          }
          case 8: { // Add+Norm (FFN): 3072 → residual + dense column
            const resVal = reduceForResidual(activations)
            const residualVec = normalizeFull(activations, false)
            data = {
              attnHeads: new Float32Array(32),
              ffnGroups: reduceForFFNGroups(new Float32Array(0)),
              residual: resVal,
              residualVec,
            }
            updateResidualChart(layer, resVal)
            updateDeltaChart(layer, resVal)
            // Live 3072-wide residual stream strip — one row per layer,
            // same 3072 f32 readback as the 3D point cloud, but drawn as a
            // 2D heatmap so you can see the full stream build up.
            updateResStripLayer(layer, activations)
            // Raw readout: exact f32 stats of the current residual stream.
            captureRawResidual(layer, activations)
            break
          }
          default:
            data = { attnHeads: new Float32Array(32), ffnGroups: new Float32Array(16), residual: 0 }
        }
        viz.activateNeurons(layer, step, data)
      } else {
        // No readback (RoPE, KV Append, O Project, FFN Down) — uniform fallback
        viz.activateLayer(layer, (step + 1) / 9)
      }

      // Speed control — slider value ≈ target tokens/sec ceiling. Apply
      // the entire gap once per token (at the last step of layer 31) so
      // the streaming pace is visibly tied to the slider regardless of
      // GPU speed. 20x removes the cap entirely (GPU-bound).
      const speed = parseInt(speedSlider.value) || 5
      if (layer === 31 && step === 8 && speed < 20) {
        const gapMs = Math.round(1000 / speed)
        await sleep(gapMs)
      } else if (speed <= 3) {
        // At very slow speeds add a tiny per-step delay too, so each
        // layer's animation through the scene reads as slow-motion.
        await sleep((4 - speed) * 4)
      }
      // Cinema mode pause/step gate — block here until user presses play
      // or the step button. The engine's GPU work continues but the next
      // layer dispatch waits, giving you a slow-motion forward pass.
      while (cinemaPaused && currentMode === 'cinema' && !cinemaStepRequested) {
        await sleep(40)
      }
      cinemaStepRequested = false
      // Cinema mode also drives the camera fly toward the active layer
      if (currentMode === 'cinema' && cinemaCameraFly) {
        viz.focusCameraOnLayer(layer)
      }
      if (currentMode === 'cinema') updateCinemaScrub()
    },
    onToken(delta, _id, _index, topK, logits) {
      appendToken(delta)
      const topConfidence = topK?.[0]?.prob ?? 0
      viz.addOutputToken(delta, topConfidence)
      totalTokens++
      tokenStripAppendGenerated(delta)

      const elapsed = (performance.now() - t0) / 1000
      document.getElementById('speedStat')!.innerHTML =
        `Speed: <strong class="live">${(totalTokens / elapsed).toFixed(1)} tok/s</strong>`
      document.getElementById('tokenStat')!.innerHTML =
        `Tokens: <strong style="color:#f4ecdf">${totalTokens}</strong>`

      // Update top-k display + confidence meter (probs are full-vocab softmax)
      if (topK) {
        updateTopK(topK)
        updateConfidence(topK)
        rawState.topId = topK[0].id
        rawState.topProb = topK[0].prob
        renderRawReadout()
      }
      // Snapshot panel state for the replay scrubber
      captureTokenSnapshot(delta, topK)
      // L=31 in the logit lens grid is always the real final token: the
      // engine's final lm_head is mathematically equivalent to a lens at
      // the last layer, so skip the extra GPU dispatch and display the
      // actual generated token here.
      displayLensToken(31, delta)
      // Push real logits into the LM head strip
      if (logits) viz.setLogits(logits)
    },
    onEmbedding(tokenId, embedding) {
      viz.setEmbedding(tokenId, embedding)
    },
    onAllAttentionScores(scores, kvLen) {
      // Per-head attention canvas for layer 31 (final decision layer).
      // The 3D scene no longer draws beams — the DOM heatmap is the full
      // visualization of the attention tensor.
      updateAttentionHeatmap(scores, kvLen)
    },
    onLayerLogitLens(layer, tokenId, token) {
      // Early layers often predict garbage; late layers converge on the
      // actual output. Updates as each layer's lens fires during decode.
      displayLensToken(layer, token)
      if (lensInfo) lensInfo.textContent = `L${layer}: ${token.trim().slice(0, 12) || '·'}`
      rawState.lensLayer = layer
      rawState.lensId = tokenId
      renderRawReadout()
    },
    onPrefill(phase, length) {
      if (phase === 'start') {
        goBtn.textContent = `Prefill (${length} tok)...`
      }
      if (phase === 'end') {
        goBtn.textContent = 'Stop'
        // Re-enable so Stop is actually clickable mid-generation — the button
        // is disabled at run start and was shipping as a dead control.
        goBtn.disabled = false
        hidePrefill()
      }
    },
    async onPrefillToken(index, total, token) {
      showPrefillToken(index, total, token)
      // Quick pulse through visualizer for each prefill token
      const layer = Math.floor((index / total) * 32)
      viz.activateLayer(layer, (index % 4) / 4)
      // Minimal delay to let UI update
      if (index % 4 === 0) await sleep(1)
    },
    onKVCache(position, totalPages, usedPages) {
      updateKVCache(position, totalPages, usedPages)
      // Update KV strips for ALL layers (the cache grows synchronously across layers)
      const frac = usedPages / totalPages
      for (let L = 0; L < 32; L++) viz.setKvCacheStrip(L, frac)
      rawState.kvLen = position
      renderRawReadout()
    },
  }
  let genCallbacks = mergeCallbacks(baseCallbacks, storyteller.hooks())
  if (recorder) {
    genCallbacks = mergeCallbacks(genCallbacks, recorder.hooks())
    recorderRunMeta = { prompt, mode: mode === 'ask' ? 'ask' : 'complete' }
  }
  await engine.generate(
    realPrompt,
    mode === 'ask' ? 140 : 180,
    genCallbacks,
  )

  viz.setDone()
  const c2 = output.querySelector('.cursor')
  if (c2) c2.remove()
  } catch (err) {
    console.warn('[inference] GPU error:', err)
    const cur = output.querySelector('.cursor')
    if (cur) cur.remove()
    // Detect device-lost: AbortError on mapAsync with "Instance reference"
    // means the GPUDevice was destroyed mid-generation. Retrying with the same
    // engine will fail identically — we have to rebuild from scratch.
    const msg = err instanceof Error ? `${err.name}: ${err.message}` : String(err)
    const deviceLost = /Instance reference|device.*lost|destroyed|context lost/i.test(msg)
    if (deviceLost) {
      output.innerHTML += `<span style="color:#f44;opacity:.7"> [GPU device lost — reinitializing...]</span>`
      engine = null
      try {
        engine = await createInferenceEngine((p) => updateLoading(p))
        hideLoading()
        // E45: re-apply continuous-attention URL flags after device-lost reinit.
        {
          const params = new URLSearchParams(location.search)
          if (params.get('attn') === 'fixedpoint') {
            engine.e45Config.attentionKernel = 'fixedpoint'
            const maxIterRaw = params.get('max_iter')
            if (maxIterRaw !== null) {
              const n = parseInt(maxIterRaw, 10)
              if (Number.isFinite(n) && n > 0 && n <= 1000) {
                engine.e45Config.fixedPointMaxIter = n
              }
            }
            console.log(`[E45 fixedpoint] re-activated after device-lost reinit`)
          }
        }
      } catch (initErr) {
        console.error('[inference] reinit failed:', initErr)
        output.innerHTML += `<span style="color:#f44;opacity:.7"> [reinit failed — please reload the page]</span>`
        isRunning = false
        goBtn.disabled = false
        goBtn.textContent = 'Think'
        return
      }
    } else {
      output.innerHTML += `<span style="color:#f44;opacity:.7"> [GPU hiccup — retrying...]</span>`
      await sleep(1500)
    }
    // Retry once with same prompt
    isRunning = false
    try {
      await runRealInference(prompt)
    } catch (retryErr) {
      console.error('[inference] retry failed:', retryErr)
      output.innerHTML += `<span style="color:#f44;opacity:.7"> [retry failed]</span>`
    }
    return // runRealInference manages cleanup on success/fail
  } finally {
    viz.setDone()
    isRunning = false
    goBtn.disabled = false
    goBtn.textContent = 'Think'
    // Lessons: a normal forward pass completed.
    window.dispatchEvent(new CustomEvent('neuropulse:lesson-signal', { detail: { type: 'generate' } }))
  }
}

// ─── Demo mode — replay a recorded run (no download, no WebGPU) ───────────
// Real tensors, captured live earlier (public/recordings/, see src/recording
// .ts), replayed through the same panel-update functions the live engine
// drives. The RECORDED RUN badge stays visible the whole time — honesty is
// the product here.
let demoMode = false
let demoDriver: PlaybackHandle | null = null
let demoRec: NpRecording | null = null
let exitDemoAfterPlayback = false
// Video-player model: Stop keeps your place and Play resumes from it; a run
// that finishes naturally resets so the next Play starts from the top.
let demoResumeAt = 0

function enterDemoMode(): void {
  if (demoMode) return
  demoMode = true
  document.body.classList.add('demo-mode')
  goBtn.disabled = false
  goBtn.textContent = 'Play recording'
  promptInput.readOnly = true
  promptInput.title = 'Recorded run replays this prompt — download the model to ask your own.'
  document.getElementById('demo-badge-live')?.addEventListener('click', () => {
    // Back to the gate to pick the live download.
    location.href = location.pathname
  })
}

/** Called when the live engine finishes loading underneath a demo session
 *  (the "learn while you wait" path) — hand the controls back to live. */
function exitDemoMode(): void {
  if (!demoMode) return
  if (demoDriver?.isPlaying()) { exitDemoAfterPlayback = true; return }
  demoMode = false
  exitDemoAfterPlayback = false
  promptInput.readOnly = false
  promptInput.title = ''
  promptInput.value = ''
  goBtn.textContent = 'Think'
  const badge = document.getElementById('demo-badge')
  if (badge) {
    badge.classList.add('live-ready')
    badge.innerHTML = '<span class="db-dot">●</span> LIVE — model ready, prompts are yours now'
    setTimeout(() => document.body.classList.remove('demo-mode'), 6000)
  } else {
    document.body.classList.remove('demo-mode')
  }
}

async function loadDemoRecording(): Promise<NpRecording> {
  if (demoRec) return demoRec
  const resp = await fetch('/recordings/intro-01.json')
  if (!resp.ok) throw new Error(`recording fetch failed: HTTP ${resp.status}`)
  demoRec = (await resp.json()) as NpRecording
  return demoRec
}

function makeDemoSink(): PlaybackSink {
  return {
    onPrefillStart(total) {
      goBtn.textContent = `Prefill (${total} tok)...`
    },
    onPrefillToken(i, total, text) {
      showPrefillToken(i, total, text)
      viz.activateLayer(Math.floor((i / total) * 32), (i % 4) / 4)
    },
    onPrefillEnd() {
      goBtn.textContent = 'Stop'
      hidePrefill()
    },
    onLayerPulse(layer, attnHeads, residualNorm) {
      updateHeatmapLayer(layer, attnHeads instanceof Float32Array ? attnHeads : new Float32Array(attnHeads))
      updateResidualChart(layer, residualNorm)
      updateDeltaChart(layer, residualNorm)
      viz.activateLayer(layer, 0.75)
    },
    onLens(layer, text, id) {
      displayLensToken(layer, text)
      // Kid-mode narration rides the replay too.
      if (storyteller.isActive()) storyteller.hooks().onLayerLogitLens?.(layer, id, text)
    },
    onToken(tok, index) {
      appendToken(tok.text)
      viz.addOutputToken(tok.text, tok.topK[0]?.p ?? 0)
      tokenStripAppendGenerated(tok.text)
      const topK: TopKEntry[] = tok.topK.map((k) => ({ token: k.t, id: k.id, prob: k.p }))
      updateTopK(topK)
      updateConfidence(topK)
      captureTokenSnapshot(tok.text, topK) // token-strip scrubbing works post-replay
      displayLensToken(31, tok.text)
      if (storyteller.isActive()) storyteller.hooks().onToken?.(tok.text, tok.id, index, topK, undefined)
      demoResumeAt = index + 1 // resume point if the user stops here
    },
    onAttentionL31(rows, kvLen) {
      updateAttentionHeatmapL31(rows, kvLen)
    },
    onKV(position, totalPages, usedPages) {
      updateKVCache(position, totalPages, usedPages)
      const frac = usedPages / totalPages
      for (let L = 0; L < 32; L++) viz.setKvCacheStrip(L, frac)
    },
    onDone(interrupted) {
      viz.setDone()
      isRunning = false
      const finished = !interrupted || demoResumeAt >= (demoRec?.tokens.length ?? 0)
      if (finished) demoResumeAt = 0
      goBtn.textContent = demoResumeAt > 0 ? 'Resume recording' : 'Play recording'
      const c = output.querySelector('.cursor')
      if (c) c.remove()
      // A fully replayed forward pass counts for the "watch a generation"
      // lesson check — the tensors are real. A stopped-after-two-tokens run
      // doesn't.
      if (finished) {
        window.dispatchEvent(new CustomEvent('neuropulse:lesson-signal', { detail: { type: 'generate' } }))
      }
      if (exitDemoAfterPlayback) exitDemoMode()
    },
  }
}

async function playDemoRecording(): Promise<void> {
  if (demoDriver?.isPlaying()) return
  let rec: NpRecording
  try {
    rec = await loadDemoRecording()
  } catch (err) {
    console.warn('[demo] recording unavailable:', err)
    return
  }
  const resuming = demoResumeAt > 0 && demoResumeAt < rec.tokens.length
  isRunning = true
  if (!resuming) {
    // Fresh run: reset all run surfaces exactly like a live run.
    demoResumeAt = 0
    clearReplayBuffer()
    output.innerHTML = ''
    const promptEcho = document.createElement('div')
    promptEcho.style.cssText = 'color:#ff8c42;margin-bottom:16px;font-size:0.8rem;font-style:italic;opacity:0.7'
    promptEcho.textContent = `> ${rec.prompt}`
    output.appendChild(promptEcho)
    tokenStripStart(rec.prompt)
  }
  // (Re)attach the streaming cursor.
  if (!output.querySelector('.cursor')) {
    const cursor = document.createElement('span')
    cursor.className = 'cursor'
    output.appendChild(cursor)
  }
  promptInput.value = rec.prompt
  goBtn.textContent = 'Stop'
  goBtn.disabled = false

  demoDriver = createPlaybackDriver(rec, makeDemoSink(), () =>
    parseInt(speedSlider.value, 10) || 5, { startAt: resuming ? demoResumeAt : 0 })
  await demoDriver.start()
}

// ─── Cancellation ───
/** Ask the engine to stop the in-flight generation at the next token. Returns
 *  true if a run was actually interrupted. */
function cancelInFlightInference(): boolean {
  return (engine as unknown as { interrupt?: () => boolean }).interrupt?.() ?? false
}
/** Resolve once no generation is running (isRunning clears in generate()'s
 *  finally). Resolves false if it doesn't settle within ~10s so callers don't
 *  hang or double-run. */
function waitForInferenceIdle(maxFrames = 600): Promise<boolean> {
  return new Promise((resolve) => {
    let n = 0
    const check = () => {
      if (!isRunning) resolve(true)
      else if (++n > maxFrames) resolve(false)
      else requestAnimationFrame(check)
    }
    check()
  })
}

// ─── Dispatch ───
function startInference(mode: 'think' | 'ask' = 'think') {
  if (demoMode) { void playDemoRecording(); return } // demo replays the recording
  const prompt = promptInput.value.trim()
  if (!prompt) return
  if (isValidating) return  // never interrupt the validation suite
  if (!engine) return // no engine = no inference (error screen is already up)
  lastPrompt = prompt
  promptInput.value = ''

  // If a generation is already running, ask it to stop at the next token
  // boundary, then start the new one once the in-flight promise settles.
  // The user's intent is "this prompt is now stale, run the new one."
  if (isRunning) {
    const wasInFlight = (engine as unknown as { interrupt?: () => boolean })
      .interrupt?.() ?? false
    if (wasInFlight) {
      // Wait one frame so the await chain inside the running generate()
      // can unwind cleanly (isRunning gets reset in its finally block).
      const pollForIdle = () => {
        if (!isRunning) {
          runRealInference(prompt, mode)
        } else {
          requestAnimationFrame(pollForIdle)
        }
      }
      requestAnimationFrame(pollForIdle)
      return
    }
  }

  runRealInference(prompt, mode)
}

// While a generation is running the Think button reads "Stop" and cancels it.
goBtn.addEventListener('click', () => {
  if (demoMode) {
    // Play/Stop toggle for the recorded run — engine interrupt doesn't apply.
    if (demoDriver?.isPlaying()) demoDriver.stop()
    else void playDemoRecording()
    return
  }
  if (isRunning) { cancelInFlightInference(); return }
  startInference('think')
})
document.getElementById('askBtn')?.addEventListener('click', () => startInference('ask'))
promptInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    // Shift+Enter = Ask (docs-augmented Q&A); plain Enter = Think (completion)
    startInference(e.shiftKey ? 'ask' : 'think')
  }
})
// Escape cancels an in-flight generation (but not the validation suite, and
// not while a guided tour is running — that has its own Escape handling).
window.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return
  if (document.body.classList.contains('tour-running')) return
  if (demoMode && demoDriver?.isPlaying()) {
    e.preventDefault()
    demoDriver.stop()
    return
  }
  if (isRunning && !isValidating) {
    e.preventDefault()
    cancelInFlightInference()
  }
})

// ─── Boot-screen error state (replaces the loading phase) ───
function showEngineError(title: string, detail: string, tip?: string) {
  const boot = document.getElementById('bootScreen')
  if (!boot) return
  // Reveal boot screen if it was hidden, hide all other phases
  boot.style.display = 'flex'
  boot.classList.remove('hidden')
  const container = document.querySelector('.container')
  if (container) container.classList.remove('revealed')
  document.querySelectorAll('.boot-phase').forEach(p => {
    p.classList.remove('visible')
    p.classList.add('hidden')
  })
  // Reuse or create the error phase
  let err = document.getElementById('bootError')
  if (!err) {
    err = document.createElement('div')
    err.id = 'bootError'
    err.className = 'boot-phase boot-gate visible'
    const inner = boot.querySelector('.boot-inner')
    if (inner) inner.appendChild(err)
  } else {
    err.classList.remove('hidden')
    err.classList.add('visible')
  }
  err.innerHTML = `
    <h2 style="color:#f4ecdf">${title}</h2>
    <p style="color:#b8b4a8">${detail}</p>
    <p class="fine" style="color:#b8b4a8;opacity:0.7">${tip ?? 'Tip: close other tabs using the GPU (video calls, YouTube, 3D sites), then reload.'}</p>
    <div class="btn-row">
      <button class="btn-go" onclick="location.reload()">Reload</button>
      <a href="/" class="btn-back">Back to landing</a>
    </div>
  `
}

// ─── Init: load real engine; show error screen if unavailable ───
async function isBraveBrowser(): Promise<boolean> {
  try {
    const nav = navigator as any
    if (nav.brave && typeof nav.brave.isBrave === 'function') {
      return await nav.brave.isBrave()
    }
  } catch {}
  return false
}

async function initEngine() {
  const brave = await isBraveBrowser()

  // Check WebGPU
  if (!navigator.gpu) {
    console.log('[neuropulse] WebGPU unavailable')
    if (brave) {
      showEngineError(
        'Brave is blocking WebGPU.',
        `Brave disables WebGPU by default. To enable it:
        <ol style="text-align:left;margin:12px auto;max-width:520px;line-height:1.7">
          <li>Open a new tab → paste <code style="background:#1a1a1a;padding:2px 6px;border-radius:3px">brave://flags/#enable-unsafe-webgpu</code> → set to <strong>Enabled</strong></li>
          <li>Paste <code style="background:#1a1a1a;padding:2px 6px;border-radius:3px">brave://settings/shields</code> → set <strong>Fingerprinting</strong> to <strong>Standard</strong> (not Strict)</li>
          <li>Relaunch Brave, then reload this tab</li>
        </ol>`,
        'Easier: open this site in Chrome, Edge, or Safari TP — WebGPU works out of the box.'
      )
    } else {
      showEngineError(
        'WebGPU not supported in this browser.',
        'Neuropulse runs the real Phi-3-mini on your GPU. Open in <strong>desktop Chrome, Edge, or Safari TP</strong> to continue.'
      )
    }
    return
  }

  showBootLoading()

  try {
    engine = await createInferenceEngine((p) => {
      updateLoading(p)
    })

    hideLoading()
    // "Learn while you wait" handoff: the download finished under a demo
    // session — return the controls to live (waits for any in-flight replay).
    if (demoMode) exitDemoMode()

    // E45 / P-20260526-07: apply continuous-attention fixed-point probe flags
    // from URL. ?attn=fixedpoint activates the Picard-iterated kernel;
    // ?max_iter=N (1..1000) sets the iteration budget. Default 'standard'
    // preserves the validated baseline. See attention_fixedpoint.wgsl.
    {
      const params = new URLSearchParams(location.search)
      if (params.get('attn') === 'fixedpoint') {
        engine.e45Config.attentionKernel = 'fixedpoint'
        const maxIterRaw = params.get('max_iter')
        if (maxIterRaw !== null) {
          const n = parseInt(maxIterRaw, 10)
          if (Number.isFinite(n) && n > 0 && n <= 1000) {
            engine.e45Config.fixedPointMaxIter = n
          }
        }
        console.log(
          `[E45 fixedpoint] URL-activated: max_iter=${engine.e45Config.fixedPointMaxIter}. ` +
          `Sub-step probe; sees the same validation harness as discrete attention. ` +
          `Watch console for per-token telemetry.`,
        )
      }
    }

    // E45: dev-only handle for the iter-sweep harness (tools/e45-iter-sweep).
    // Lets a devtools session do `await window.__e45.sweep(...)` without
    // touching the prod UI. Gated on import.meta.env.DEV so this never ships
    // to neuropulse.live.
    if (import.meta.env.DEV) {
      ;(globalThis as { __e45?: unknown }).__e45 = {
        engine,
        /** Run a max_iter sweep on the same prompt, capture decoded token
         *  strings + per-token telemetry diff-max per iter setting. */
        async sweep(prompt: string, iters: number[], maxTokens = 12) {
          if (!engine) throw new Error('engine not ready')
          const results: Array<{
            iter: number
            kernel: 'standard' | 'fixedpoint'
            text: string
            tokens: { id: number; t: string }[]
          }> = []
          for (const iter of iters) {
            engine.e45Config.attentionKernel = iter === 0 ? 'standard' : 'fixedpoint'
            if (iter > 0) engine.e45Config.fixedPointMaxIter = iter
            const collected: { id: number; t: string }[] = []
            const text = await engine.generate(prompt, maxTokens, {
              onToken: (t, id) => collected.push({ id, t }),
            })
            results.push({
              iter,
              kernel: engine.e45Config.attentionKernel,
              text,
              tokens: collected,
            })
            console.log(`[E45 sweep] iter=${iter} kernel=${engine.e45Config.attentionKernel} text=${JSON.stringify(text)}`)
          }
          console.log('[E45 sweep] complete', results)
          return results
        },
      }
      console.log('[E45 dev] window.__e45 exposed (engine, sweep). Try: await __e45.sweep("The capital of Japan is", [0, 1, 2, 3, 5, 10, 20, 50, 100], 12)')
    }

    // Butterfly experiment: floating panel runs an in-browser transgenerational
    // compaction demo using the same Phi-3 instance. See src/butterfly-mode.ts.
    //
    // Initialized unconditionally as of v2.5 — discoverable via the
    // "Butterfly" button in the mode bar OR the legacy ?mode=butterfly URL
    // flag (the flag now just auto-opens the panel on first paint).
    initButterflyPanel({
      getEngine: () => engine,
      isBusy:    () => isRunning || isValidating,
      setBusy:   (b) => { isRunning = b },
      // v2: pass the visualizer so butterfly-mode paints tag importance
      // into the residual-stream slab during a run.
      viz:       viz as unknown as { updateResidualLayer(layer: number, vec: Float32Array): void },
      // v2.4: snapshot the ablation panel's selection at run-start. Heads
      // shift-clicked in the 3D scene are zeroed inside butterfly's tagger,
      // chrysalis, and answer arms (judge stays unablated as the meter).
      getAblations: () => viz.getAblations() as Ablation[],
    })
    const __toggleBfly = (window as unknown as { __toggleButterflyPanel?: () => void }).__toggleButterflyPanel
    // Wire the new mode-bar button. Behaves like a panel toggle, NOT a
    // setMode() target — Butterfly is an overlay, not a view mode.
    document.querySelectorAll<HTMLButtonElement>('button[data-bfly-toggle]').forEach(btn => {
      btn.addEventListener('click', () => __toggleBfly?.())
    })
    // Legacy ?mode=butterfly URL flag: auto-open the panel.
    if (new URLSearchParams(window.location.search).get('mode') === 'butterfly') {
      __toggleBfly?.()
    }
    // ?sweep=N URL flag: auto-run the pre-registered scaling sweep over
    // all 4 built-in transcripts with N runs each. Result JSON downloads
    // when the loop completes. Used by tools/auto-sweep.sh to drive the
    // sweep from outside the browser without a paste-in-console step.
    const sweepArg = new URLSearchParams(window.location.search).get('sweep')
    if (sweepArg) {
      const runsPer = Math.max(1, Math.min(50, parseInt(sweepArg, 10) || 20))
      __toggleBfly?.()
      // Defer so the panel + picker DOM is ready and the user's first
      // engine.generate() warm-up has settled.
      setTimeout(() => {
        const fn = (window as unknown as { butterflySweep?: (o: { runsPer: number }) => Promise<unknown> }).butterflySweep
        if (typeof fn === 'function') {
          console.log(`[neuropulse] auto-sweep starting with runsPer=${runsPer}`)
          fn({ runsPer }).then(() => console.log('[neuropulse] auto-sweep complete'))
            .catch((e) => console.error('[neuropulse] auto-sweep failed:', e))
        } else {
          console.error('[neuropulse] auto-sweep requested but butterflySweep() is not exposed')
        }
      }, 2000)
    }

    // Devtools smoke test: window.__ablate('prompt', [{layer: 15}])
    // Runs two short generations — baseline and ablated — and logs both.
    // Proves the engine path works end-to-end before wiring UI.
    ;(window as unknown as {
      __ablate: (prompt: string, ablations: Ablation[], maxTokens?: number) => Promise<void>
    }).__ablate = async (prompt, ablations, maxTokens = 30) => {
      if (!engine) { console.warn('engine not ready'); return }
      if (isRunning || isValidating) {
        console.warn('[ablate] app is busy (auto-run or user generation in flight). Wait for it to finish, then retry.')
        return
      }
      isRunning = true
      try {
        const cb = {}
        console.log('[ablate] baseline...')
        const t0 = performance.now()
        const base = await engine.generate(prompt, maxTokens, cb)
        console.log('[ablate] baseline:', JSON.stringify(base), `(${((performance.now()-t0)/1000).toFixed(1)}s)`)
        console.log('[ablate] ablated', ablations, '...')
        const t1 = performance.now()
        const abl = await engine.generate(prompt, maxTokens, cb, ablations)
        console.log('[ablate] ablated :', JSON.stringify(abl), `(${((performance.now()-t1)/1000).toFixed(1)}s)`)
        console.log('[ablate] differ?', base !== abl)
      } finally {
        isRunning = false
      }
    }

    // Update header to show real engine
    const dispatchStat = document.getElementById('dispatchStat')
    if (dispatchStat) {
      dispatchStat.innerHTML = `Engine: <strong style="color:#5eead4">ZeroTVM</strong>`
    }

    // Auto-run an opening prompt so the visualization lights up on arrival.
    // Skipped when ?noauto is in the URL — useful for __ablate smoke tests
    // and the ?sweep=N auto-sweep mode (which needs an idle engine to
    // start clicking the butterfly Run button).
    const _qs = new URLSearchParams(location.search)
    const skipAuto = _qs.has('noauto') || _qs.has('sweep')
    if (!skipAuto) {
      setTimeout(() => {
        if (!isRunning && !isValidating) {
          promptInput.value = getSharedPrompt() || 'What is consciousness?'
          startInference()
        }
      }, 1500)
    } else {
      console.log('[neuropulse] auto-run suppressed (?noauto). Engine idle, __ablate ready.')
    }

  } catch (e) {
    console.warn('[neuropulse] engine init failed:', e)
    if (brave) {
      showEngineError(
        'Brave is blocking the GPU.',
        `Brave blocks WebGL/WebGPU fingerprinting by default. To enable:
        <ol style="text-align:left;margin:12px auto;max-width:520px;line-height:1.7">
          <li>Open a new tab → paste <code style="background:#1a1a1a;padding:2px 6px;border-radius:3px">brave://flags/#enable-unsafe-webgpu</code> → set to <strong>Enabled</strong></li>
          <li>Paste <code style="background:#1a1a1a;padding:2px 6px;border-radius:3px">brave://settings/shields</code> → set <strong>Fingerprinting</strong> to <strong>Standard</strong> (not Strict)</li>
          <li>Relaunch Brave, then reload this tab</li>
        </ol>`,
        'Easier: open this site in Chrome, Edge, or Safari TP — WebGPU works out of the box.'
      )
    } else {
      showEngineError(
        'Couldn\'t start the GPU engine.',
        'Your browser reported the GPU as unavailable — this usually means too many other tabs are using it. Close some tabs and reload.'
      )
    }
  }
}

// PCA is the permanent, accurate layout — load independently of engine init
// so it applies even if engine initialization fails and we fall back to demo mode.
loadPcaLayoutPermanent()

// ─── Download gate ───
// ─── Download gate (HTML-based) ───
async function modelIsCached(): Promise<boolean> {
  if (typeof caches === 'undefined') return false
  try {
    const names = await caches.keys()
    for (const name of names) {
      const store = await caches.open(name)
      const keys = await store.keys()
      for (const req of keys) {
        if (req.url.includes('Phi-3-mini-4k-instruct-q4f16_1-MLC')) return true
      }
    }
  } catch { /* Cache API blocked */ }
  return false
}

function waitForGateClick(): Promise<'live' | 'demo'> {
  return new Promise((resolve) => {
    const live = document.getElementById('bootGoBtn')
    const learn = document.getElementById('bootLearnBtn')
    if (!live && !learn) { resolve('live'); return }
    live?.addEventListener('click', () => resolve('live'), { once: true })
    learn?.addEventListener('click', () => resolve('demo'), { once: true })
  })
}

/** Boot straight into demo mode: WebGL scene + recorded-run replay, no
 *  engine, no download, no WebGPU requirement. The single seam future mobile
 *  support will call too. */
function enterDemoBoot(openLessonId?: string): void {
  try { initVisualizer() } catch { /* WebGL missing — panels still work */ }
  enterDemoMode()
  hideLoading()
  if (openLessonId) {
    // Give the boot fade a beat, then open the learning path where the
    // welcome overlay used to appear.
    window.setTimeout(() => {
      ;(window as unknown as { __openLesson?: (id: string) => void }).__openLesson?.(openLessonId)
    }, 1000)
  }
}

;(async () => {
  // Mobile guard — the inline script in app/index.html sets this flag when
  // the user is on a phone. Skip engine init entirely; the mobile block
  // phase is already visible. (Future mobile demo support = call
  // enterDemoBoot() here instead of returning.)
  if ((window as any).__NEUROPULSE_MOBILE_BLOCK__) return

  const params = new URLSearchParams(location.search)

  // Test bypass: ?bypass=1 skips both the cache check and the download gate,
  // boots the visualizer immediately, and (crucially) skips engine init so
  // Playwright UI tests don't trigger a 2 GB weight download per test.
  if (params.has('bypass')) {
    try { initVisualizer() } catch {}
    return
  }

  // ?demo=1 — shareable demo-mode link. Overrides the cached-model skip so
  // the link behaves identically for everyone.
  if (params.has('demo')) {
    enterDemoBoot('bearings')
    return
  }

  // "Learn while you wait" — visible in the loading phase during the 2 GB
  // download; starts the demo immediately, download continues underneath,
  // and initEngine hands control back to live when it finishes.
  document.getElementById('bootDemoWait')?.addEventListener('click', () => {
    enterDemoBoot('bearings')
  }, { once: true })

  if (await modelIsCached()) {
    // Cached — skip gate, go straight to loading phase
    showBootLoading()
    try { initVisualizer() } catch {}
    initEngine()
  } else {
    // First visit — the education fork: learn now (recorded run) or
    // download and run live.
    const choice = await waitForGateClick()
    if (choice === 'demo') {
      enterDemoBoot('bearings')
      return
    }
    try { initVisualizer() } catch {}
    initEngine()
  }
})()

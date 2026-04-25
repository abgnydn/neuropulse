import { BrainVisualizer, LayerActivation } from './visualizer'
import { createInferenceEngine, InferenceEngine, LoadProgress, TopKEntry, Ablation } from './engine/inference'
import { initButterflyPanel } from './butterfly-mode'
import { reduceQKVForAttnHeads, reduceForAttnHeads, reduceForFFNGroups, reduceForResidual, normalizeFull } from './engine/activation-reducer'
import { createJourney, JourneyHandle } from './journey'
import { SpatialPanels } from './spatial-panels'
import { TOURS, createTourRunner } from './tours'
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
    }
    .ablate-panel.open { display: block; }
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

    .ablate-strip { display: none; grid-template-columns: repeat(32, 1fr); gap: 2px; margin-bottom: 8px; }
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
      <button class="ablate-close" id="ablateCloseBtn" type="button" aria-label="Hide ablation panel (A)" title="Hide · A">✕</button>
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

  function setPanelOpen(open: boolean) {
    panel.classList.toggle('open', open)
  }
  // Open by default. Shift-clicking heads will also force-open it.
  setPanelOpen(true)
  closeBtn.addEventListener('click', () => setPanelOpen(false))

  // Expose toggle for the global keymap (A key).
  ;(window as unknown as { __toggleAblatePanel: () => void }).__toggleAblatePanel = () => {
    setPanelOpen(!panel.classList.contains('open'))
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
    if (isRunning || isValidating) { alert('Inference already in flight.'); return }
    const abls = viz.getAblations()
    if (abls.length === 0) return
    const prompt = (promptInput.value.trim() || 'Paris is the capital of')
    runBtn.disabled = true; clearBtn.disabled = true; sweepBtn.disabled = true
    runBtn.textContent = 'Running…'
    baseOut.classList.remove('empty'); ablOut.classList.remove('empty')
    baseOut.textContent = 'generating…'
    ablOut.textContent = 'waiting…'
    isRunning = true
    try {
      const cb = {}
      const base = await engine.generate(prompt, 40, cb)
      baseOut.textContent = base || '(empty)'
      ablOut.textContent = 'generating…'
      const abl = await engine.generate(prompt, 40, cb, abls)
      ablOut.textContent = abl || '(empty)'
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
    if (isRunning || isValidating) { alert('Inference already in flight.'); return }
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

    const prompt = (promptInput.value.trim() || 'Paris is the capital of')
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
let totalTokens = 0
let engine: InferenceEngine | null = null

// Speed control
const speedSlider = document.getElementById('speedSlider') as HTMLInputElement
const speedLabel = document.getElementById('speedLabel')!
speedSlider.addEventListener('input', () => {
  speedLabel.textContent = speedSlider.value + 'x'
})

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

function setMode(mode: ViewMode) {
  if (mode === currentMode) return
  if (currentMode === 'journey' && journey) journey.exit()
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
  if (mode === 'cinema') updateCinemaScrub()
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
  } else if (e.key === 'h' || e.key === 'H') {
    toggleJourneyHud()
  } else if (e.key === 'a' || e.key === 'A') {
    // Toggle just the ablation panel.
    ;(window as unknown as { __toggleAblatePanel?: () => void }).__toggleAblatePanel?.()
  } else if (e.key === 's' || e.key === 'S') {
    // Toggle soft Gaussian-sprite rendering — discrete spheres become
    // soft volumetric puffs. Picking still works (meshes stay raycastable).
    ;(viz as unknown as { toggleSoftMode?: () => void }).toggleSoftMode?.()
  } else if (e.key === 'r' || e.key === 'R') {
    // Reset camera to home position via OrbitControls
    const controls = (viz as unknown as { controls?: { reset?: () => void } }).controls
    controls?.reset?.()
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
      btn.addEventListener('click', (e) => {
        e.stopPropagation()
        btn.classList.toggle('on')
        body.classList.toggle('on')
      })
      panel.appendChild(btn)
      panel.appendChild(body)
    }

    panel.addEventListener('click', (e) => {
      if (!panel.classList.contains('expanded')) {
        panel.classList.add('expanded')
        return
      }
      // Expanded: only collapse if clicking the panel *background* itself,
      // not interactive children (inputs, buttons, canvases, links).
      const t = e.target as HTMLElement
      if (t === panel) panel.classList.remove('expanded')
    })
  })
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

  function playTour(id: string): void {
    if (!runner) runner = createTourRunner(viz, updateCaption)
    runner.play(id)
    document.body.classList.add('tour-running')
    // Close the glossary so the tour is visible
    document.getElementById('glossary-overlay')?.classList.remove('visible')
  }

  function stopTour(): void {
    runner?.stop()
    document.body.classList.remove('tour-running')
  }

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

  // Any manual journey interaction also stops the tour
  window.addEventListener('pointerdown', (e) => {
    if (!document.body.classList.contains('tour-running')) return
    const t = e.target as HTMLElement | null
    if (t && t.closest('#journey-hud, .side, #journey-exit, #glossary-overlay')) return
    stopTour()
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
;(function wireWelcomeOverlay() {
  const STORAGE_KEY = 'np:welcome-dismissed'
  const overlay = document.getElementById('welcome-overlay')
  const dismiss = document.getElementById('welcome-dismiss') as HTMLButtonElement | null
  if (!overlay || !dismiss) return
  let seen = false
  try { seen = localStorage.getItem(STORAGE_KEY) === '1' } catch { /* storage disabled */ }
  // Show after a short delay so the boot screen fade has settled
  if (!seen) {
    setTimeout(() => {
      overlay.classList.add('visible')
    }, 900)
  }
  dismiss.addEventListener('click', () => {
    overlay.classList.remove('visible')
    try { localStorage.setItem(STORAGE_KEY, '1') } catch { /* ok */ }
  })
  // Click outside card also dismisses
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      overlay.classList.remove('visible')
      try { localStorage.setItem(STORAGE_KEY, '1') } catch { /* ok */ }
    }
  })
  // Esc dismisses
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && overlay.classList.contains('visible')) {
      overlay.classList.remove('visible')
      try { localStorage.setItem(STORAGE_KEY, '1') } catch { /* ok */ }
    }
  })
})()

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
    Prefill: <span class="prefill-token">${display}</span> <span style="color:#8a8170">${index + 1}/${total}</span>
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
  // Fall through to the legacy mini canvas (still used in scene mode)
  if (!attnCtx) return
  attnCtx.fillStyle = '#08060f'
  attnCtx.fillRect(0, 0, attnCanvas.width, attnCanvas.height)
  const cols = Math.min(kvLen, ATTN_MAX_SLOTS)
  if (cols === 0) return
  const cellW = attnCanvas.width / cols
  const cellH = attnCanvas.height / ATTN_HEADS
  const layerOff = ATTN_LAYER * ATTN_HEADS * ATTN_MAX_SLOTS
  for (let h = 0; h < ATTN_HEADS; h++) {
    const headBase = layerOff + h * ATTN_MAX_SLOTS
    // Per-head max for contrast scaling (softmax sums to 1, but a head that
    // attends uniformly would have ~1/kvLen per slot, so we re-scale).
    let rowMax = 0
    for (let s = 0; s < cols; s++) {
      const v = scores[headBase + s]
      if (v > rowMax) rowMax = v
    }
    if (rowMax < 1e-6) rowMax = 1
    for (let s = 0; s < cols; s++) {
      const v = scores[headBase + s] / rowMax
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
  await engine.generate(realPrompt, mode === 'ask' ? 300 : 500, {
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

      // Speed-controlled delay
      const speed = parseInt(speedSlider.value)
      const delayMs = speed >= 20 ? 0
        : speed >= 10 ? Math.max(1, Math.round(12 - (speed - 10) * 1.2))
        : speed >= 4 ? Math.round(30 - (speed - 3) * 3)
        : Math.round(100 - speed * 20)

      if (delayMs > 0) {
        await sleep(delayMs)
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
        goBtn.textContent = 'Generating...'
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
  })

  viz.setDone()
  const c2 = output.querySelector('.cursor')
  if (c2) c2.remove()
  } catch (err) {
    console.warn('[inference] GPU error:', err)
    const cur = output.querySelector('.cursor')
    if (cur) cur.remove()
    // Show brief error indicator, then auto-retry after GPU settles
    output.innerHTML += `<span style="color:#f44;opacity:.7"> [GPU hiccup — retrying...]</span>`
    await sleep(1500)
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
  }
}

// ─── Dispatch ───
function startInference(mode: 'think' | 'ask' = 'think') {
  const prompt = promptInput.value.trim()
  if (!prompt) return
  if (isValidating) return  // never interrupt the validation suite
  if (!engine) return // no engine = no inference (error screen is already up)
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

goBtn.addEventListener('click', () => startInference('think'))
document.getElementById('askBtn')?.addEventListener('click', () => startInference('ask'))
promptInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    // Shift+Enter = Ask (docs-augmented Q&A); plain Enter = Think (completion)
    startInference(e.shiftKey ? 'ask' : 'think')
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

    // Butterfly mode (?mode=butterfly): floating panel runs an in-browser
    // transgenerational compaction demo using the same Phi-3 instance.
    // See src/butterfly-mode.ts.
    if (new URLSearchParams(window.location.search).get('mode') === 'butterfly') {
      initButterflyPanel({
        getEngine: () => engine,
        isBusy:    () => isRunning || isValidating,
        setBusy:   (b) => { isRunning = b },
      })
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
    // Skipped when ?noauto is in the URL — useful for __ablate smoke tests.
    const skipAuto = new URLSearchParams(location.search).has('noauto')
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

function waitForGateClick(): Promise<void> {
  return new Promise((resolve) => {
    const btn = document.getElementById('bootGoBtn')
    if (!btn) { resolve(); return }
    btn.addEventListener('click', () => resolve(), { once: true })
  })
}

;(async () => {
  // Mobile guard — the inline script in app/index.html sets this flag when
  // the user is on a phone. Skip engine init entirely; the mobile block
  // phase is already visible.
  if ((window as any).__NEUROPULSE_MOBILE_BLOCK__) return

  // Test bypass: ?bypass=1 skips both the cache check and the download gate,
  // boots the visualizer immediately, and (crucially) skips engine init so
  // Playwright UI tests don't trigger a 2 GB weight download per test.
  const bypass = new URLSearchParams(location.search).has('bypass')
  if (bypass) {
    try { initVisualizer() } catch {}
    return
  }
  if (await modelIsCached()) {
    // Cached — skip gate, go straight to loading phase
    showBootLoading()
    try { initVisualizer() } catch {}
    initEngine()
  } else {
    // First visit — show gate, wait for click
    await waitForGateClick()
    try { initVisualizer() } catch {}
    initEngine()
  }
})()

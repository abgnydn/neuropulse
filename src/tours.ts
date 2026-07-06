/**
 * GUIDED TOURS — short curated narratives that drive the journey camera
 * through a specific lesson, with captions timed to each step.
 *
 * A tour is a list of steps. At each step: set a camera pose (via the
 * existing journey infrastructure), show a caption in the journey HUD,
 * optionally pre-fill the prompt, then wait `hold` ms before advancing.
 *
 * Authorship is intentionally data-only so adding a tour = add an object
 * to this array. No code changes needed.
 */

import * as THREE from 'three'
import type { BrainVisualizer } from './visualizer'

export interface TourStep {
  /** milliseconds to hold on this frame before advancing */
  hold: number
  /** camera world position (x, y, z) */
  pos: [number, number, number]
  /** camera look-at target (x, y, z) */
  lookAt: [number, number, number]
  /** caption shown in the journey HUD for this step (HTML <b>/<i> allowed) */
  caption: string
  /** sub-caption (explanatory line under the caption) */
  sub?: string
  /** optional 3D-group pulse: 'attention' | 'ffn' | 'residual' | 'all' */
  highlight?: string
  /** optional prompt to pre-fill (for prompts that demo the tour's point) */
  prompt?: string
}

export interface Tour {
  id: string
  title: string
  summary: string
  steps: TourStep[]
}

export const TOURS: Tour[] = [
  {
    id: 'orientation',
    title: 'What am I looking at?',
    summary: 'A 90-second tour of the 3D scene.',
    steps: [
      {
        hold: 4500,
        pos: [0, 4.5, 18],
        lookAt: [0, 0, 0],
        caption: 'Phi-3-mini',
        sub: '3.8 billion parameters, running live on your GPU. The model is laid out horizontally — 32 transformer layers left to right.',
      },
      {
        hold: 5500,
        pos: [-5, 1.5, 4],
        lookAt: [-3, 0, 0],
        caption: 'Embedding',
        sub: 'Your tokens enter here as <b>3,072 floats</b> each. Every color you see is a real tensor, read back from a GPU buffer.',
      },
      {
        hold: 5500,
        pos: [0, 1.2, 2.8],
        lookAt: [0, 0, 0],
        caption: 'Residual stream',
        sub: 'The central axis. Every layer reads from it and adds back to it — information <i>accumulates</i> as you move right.',
        highlight: 'residual',
      },
      {
        hold: 5500,
        pos: [0, 2, 2],
        lookAt: [0, 0, 0],
        caption: '1,024 attention heads',
        sub: 'Cyan neurons on the outer ring. 32 layers × 32 heads. Each decides which past tokens to look at.',
        highlight: 'attention',
      },
      {
        hold: 5500,
        pos: [0, -1.4, 2.5],
        lookAt: [0, 0, 0],
        caption: 'FFN slabs',
        sub: 'Amber layers — each expands residual to <b>8,192</b> dims, applies SiLU gating, projects back. This is where most compute lives.',
        highlight: 'ffn',
      },
      {
        hold: 6000,
        pos: [5.5, 1.5, 3.2],
        lookAt: [3, 0, 0],
        caption: 'LM head',
        sub: 'Final projection to <b>32,064 vocabulary logits</b>. The highest one becomes the next token. Then it loops.',
      },
    ],
  },

  {
    id: 'factual-recall',
    title: 'How the model remembers facts',
    summary: 'Watch "The capital of Japan is" → "Tokyo" fire through the model.',
    steps: [
      {
        hold: 3500,
        pos: [0, 4, 16],
        lookAt: [0, 0, 0],
        caption: 'Factual recall',
        sub: 'Type <i>"The capital of Japan is"</i> and watch the model produce "Tokyo". Research suggests facts live in the FFN.',
        prompt: 'The capital of Japan is',
      },
      {
        hold: 4500,
        pos: [-1.2, 1, 2],
        lookAt: [0, 0, 0],
        caption: 'Mid-stream: layers 14–22',
        sub: 'This band is where <b>semantic concepts</b> assemble. If "Tokyo" is going to emerge, it is being staged here.',
      },
      {
        hold: 5500,
        pos: [0, -1.2, 2.2],
        lookAt: [0, 0, 0],
        caption: 'FFN as memory',
        sub: 'Each amber slab is a <b>3,072 × 8,192</b> lookup table learned during pre-training. Facts are encoded as sparse patterns across these 8,192 neurons.',
        highlight: 'ffn',
      },
      {
        hold: 5500,
        pos: [4.5, 1.3, 2.8],
        lookAt: [3, 0, 0],
        caption: 'The winner',
        sub: 'By layer 31 the LM head softmaxes into a peaky distribution. <b>"Tokyo" &gt; 0.9</b> probability. Open the Top-K panel to see it live.',
      },
    ],
  },

  {
    id: 'attention-story',
    title: 'Attention — what it actually does',
    summary: 'Looking at heads and how they decide what to focus on.',
    steps: [
      {
        hold: 4000,
        pos: [0, 3.5, 8],
        lookAt: [0, 0, 0],
        caption: 'Attention',
        sub: 'For every token, each head decides <i>which past tokens to look at</i>. 32 heads × 32 layers = 1,024 decisions happening in parallel.',
        highlight: 'attention',
      },
      {
        hold: 5000,
        pos: [-2.5, 1.2, 2],
        lookAt: [-1, 0, 0],
        caption: 'Early heads (0–3)',
        sub: 'These mostly act as <b>position detectors</b>: "the previous token," "the first token," "adjacent tokens." Surface-level patterns.',
      },
      {
        hold: 5000,
        pos: [0.5, 1.2, 2],
        lookAt: [0.5, 0, 0],
        caption: 'Induction heads (4–13)',
        sub: 'The interpretability famous ones. They find patterns like <i>[A][B] … [A]</i> and predict <i>[B]</i> next. This is how in-context learning starts.',
      },
      {
        hold: 5500,
        pos: [2.5, 1.2, 2],
        lookAt: [2, 0, 0],
        caption: 'Late heads (14+)',
        sub: 'Handle <b>long-range attention</b>, coreference, semantic binding. A head here might attend to a name mentioned 100 tokens ago to resolve a pronoun.',
      },
      {
        hold: 5000,
        pos: [0, 2.5, 3],
        lookAt: [0, 0, 0],
        caption: 'Open the Heatmap',
        sub: 'Click the <b>Heatmap</b> pip above the model. Each cell = one head\'s live magnitude. Bright cells are heads doing heavy work <i>right now</i>.',
      },
    ],
  },

  {
    id: 'confidence',
    title: 'When the model is guessing',
    summary: 'Open-ended prompts → flat softmax → low confidence.',
    steps: [
      {
        hold: 4000,
        pos: [0, 4, 15],
        lookAt: [0, 0, 0],
        caption: 'Confidence',
        sub: 'The <b>Confidence</b> panel shows <code>1 − entropy(softmax)</code>. Peaky = sure. Flat = guessing.',
        prompt: 'What is consciousness?',
      },
      {
        hold: 5500,
        pos: [4.3, 0, 3],
        lookAt: [2, 0, 0],
        caption: 'Open Top-K',
        sub: 'Run a factual prompt (<i>"The capital of Japan is"</i>) vs an open-ended one (<i>"What is consciousness?"</i>). Watch top-5 go from one big spike to five similar-sized bars.',
      },
      {
        hold: 5000,
        pos: [4.3, -0.9, 3],
        lookAt: [3, 0, 0],
        caption: 'Two signals of uncertainty',
        sub: '<b>Flat top-K</b> and <b>amber confidence bar</b> tell you the model is hedging. Temperature sampling matters most here.',
      },
    ],
  },
]

export interface TourRunnerState {
  tourId: string | null
  step: number
  total: number
  paused: boolean
}

interface TourRunnerHandle {
  play(id: string): void
  stop(): void
  pause(): void
  resume(): void
  next(): void
  prev(): void
  /** Jump straight to step i (clamped) — segment clicks in the transport. */
  goTo(i: number): void
  isPlaying(): boolean
  isPaused(): boolean
  state(): TourRunnerState
  /** 0..1 fill of the CURRENT step's hold — drives the story-style segment
   *  bars. Frozen while paused. */
  holdProgress(): number
}

export function createTourRunner(
  vis: BrainVisualizer,
  onCaption?: (caption: string, sub: string) => void,
  /** Fired on every step entry AND on pause/resume — drives the transport UI. */
  onStepChange?: (index: number, total: number, paused: boolean) => void,
  /** Fired when a tour finishes naturally or is stopped. */
  onEnd?: () => void,
): TourRunnerHandle {
  let stepTimer: number | null = null
  let tour: Tour | null = null
  let index = 0
  let paused = false
  // Hold bookkeeping so pause/resume can freeze and restore the countdown.
  let armedAt = 0
  let holdMs = 0
  let remainingMs = 0

  function clearTimer(): void {
    if (stepTimer !== null) {
      clearTimeout(stepTimer)
      stepTimer = null
    }
  }

  function holdFor(step: TourStep): number {
    // Pace follows the Speed slider (default 5×), but a hard 1.8s floor keeps
    // captions readable even at 20× — one control, still legible.
    const speed = (window as unknown as { __npSpeed?: () => number }).__npSpeed?.() ?? 5
    const mult = Math.max(0.3, Math.min(1.1, 2.5 / speed))
    return Math.max(1800, step.hold * mult)
  }

  function armHold(ms: number): void {
    clearTimer()
    armedAt = performance.now()
    holdMs = ms
    stepTimer = window.setTimeout(() => advance(1, true), ms)
  }

  /** Apply step `i`'s camera pose, caption, highlight, and prompt — a pure
   *  function of the step, so transport prev/next can jump anywhere. */
  function applyStep(i: number): void {
    if (!tour) return
    const step = tour.steps[i]!
    vis.setJourneyCamera(new THREE.Vector3(...step.pos), new THREE.Vector3(...step.lookAt))
    onCaption?.(step.caption, step.sub ?? '')
    if (step.highlight) vis.highlightGroup(step.highlight)
    if (step.prompt) {
      const input = document.getElementById('promptInput') as HTMLInputElement | null
      if (input) input.value = step.prompt
    }
    onStepChange?.(i, tour.steps.length, paused)
  }

  function advance(delta: number, fromTimer = false): void {
    if (!tour) return
    const nextIdx = index + delta
    if (nextIdx >= tour.steps.length) {
      // Natural completion (timer) or Next past the last step — finish.
      stopInternal()
      return
    }
    index = Math.max(0, nextIdx)
    applyStep(index)
    if (paused && !fromTimer) {
      // Jumping while paused shows the step but stays paused.
      remainingMs = holdFor(tour.steps[index]!)
      return
    }
    armHold(holdFor(tour.steps[index]!))
  }

  function stopInternal(): void {
    clearTimer()
    tour = null
    index = 0
    paused = false
    vis.setJourneyDriving(false)
    onEnd?.()
  }

  return {
    play(id: string): void {
      const t = TOURS.find((x) => x.id === id)
      if (!t) return
      clearTimer()
      tour = t
      index = 0
      paused = false
      // Journey needs to own the camera during a tour
      vis.setJourneyDriving(true)
      applyStep(0)
      armHold(holdFor(t.steps[0]!))
    },
    stop(): void {
      if (tour) stopInternal()
    },
    pause(): void {
      if (!tour || paused) return
      paused = true
      remainingMs = Math.max(300, holdMs - (performance.now() - armedAt))
      clearTimer()
      // Camera yields to the user while paused so they can orbit freely.
      vis.setJourneyDriving(false)
      onStepChange?.(index, tour.steps.length, true)
    },
    resume(): void {
      if (!tour || !paused) return
      paused = false
      vis.setJourneyDriving(true)
      // Re-apply the current step so the camera glides back to the tour pose.
      applyStep(index)
      armHold(remainingMs)
    },
    next(): void { if (tour) advance(1) },
    prev(): void {
      if (!tour) return
      if (index === 0) { applyStep(0); if (!paused) armHold(holdFor(tour.steps[0]!)); return }
      advance(-1)
    },
    goTo(i: number): void {
      if (!tour) return
      const target = Math.max(0, Math.min(i, tour.steps.length - 1))
      index = target
      applyStep(index)
      if (paused) { remainingMs = holdFor(tour.steps[index]!); return }
      armHold(holdFor(tour.steps[index]!))
    },
    isPlaying(): boolean { return tour !== null },
    isPaused(): boolean { return paused },
    state(): TourRunnerState {
      return { tourId: tour?.id ?? null, step: index, total: tour?.steps.length ?? 0, paused }
    },
    holdProgress(): number {
      if (!tour || holdMs <= 0) return 0
      if (paused) return Math.max(0, Math.min(1, 1 - remainingMs / holdMs))
      return Math.max(0, Math.min(1, (performance.now() - armedAt) / holdMs))
    },
  }
}

/**
 * JOURNEY MODE — dolly-follow through the live forward pass.
 *
 * The camera rides just behind the "signal position" on the layer axis and
 * looks forward in the direction of the forward pass. As the user scrolls
 * (or presses space to auto-play), the signal sweeps from the embedding
 * on the left (x ≈ -3) to the LM head on the right (x ≈ +3), and the
 * camera tracks it with a continuous smooth path. No waypoint snaps —
 * one spline-ish curve driven by math.
 *
 * Controls:
 *   scroll / ↑↓ / ←→    : step progress (slow)
 *   space                : play/pause auto-advance (~55s full journey)
 *   drag (while paused)  : orbit the camera around the current focus
 *   Home / End           : jump to start / end
 *   S / Escape           : exit to Scene mode (handled in main.ts)
 *
 * Preserves strict 1:1: every tensor mesh still renders from real GPU
 * buffers each frame. Only the camera and an overlay HUD are scripted.
 */

import * as THREE from 'three'
import type { BrainVisualizer } from './visualizer'

const LAYERS = 32
const TOTAL_WIDTH = 6.0  // must match visualizer's TOTAL_WIDTH
const X_START = -TOTAL_WIDTH / 2 - 0.5    // -3.5 (slightly past embedding)
const X_END   =  TOTAL_WIDTH / 2 + 0.5    //  3.5 (slightly past LM head)

// Phase boundaries in progress space
const OPEN_END = 0.05   // 0 → 0.05: establishing shot → dolly-in
const FLY_END  = 0.92   // 0.05 → 0.92: dolly-follow across all 32 layers
//                       0.92 → 1.00: pull back to LM head overview

// Tuning knobs (all user-facing speed parameters in one place)
const WHEEL_SCALE   = 0.00022   // wheel: ~40 ticks to cross the whole journey
const KEY_STEP      = 0.010     // single arrow-key press
const TOUCH_SCALE   = 0.0012
const AUTO_RATE     = 0.00028   // per frame; 0.00028 * 60fps ≈ 1.7%/s → ~60s total
const LERP_TO_PROG  = 0.08      // progress inertia — smoother than raw scroll

interface Pose { pos: THREE.Vector3; lookAt: THREE.Vector3 }

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x))
}
function smoothstep(a: number, b: number, x: number): number {
  const t = clamp01((x - a) / (b - a))
  return t * t * (3 - 2 * t)
}
function mix(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** Signal X position for progress. `null` outside the fly phase. Linear —
 *  constant layer-per-second during auto-play, so the camera doesn't appear
 *  to accelerate in the middle. */
function sigXFor(p: number): number | null {
  if (p < OPEN_END || p > FLY_END) return null
  const flyT = (p - OPEN_END) / (FLY_END - OPEN_END)
  return mix(X_START, X_END, flyT)
}

/** Offset between the camera's looked-at point and sigX.
 *  The camera looks 0.35 units forward (`sigX + LOOK_AHEAD`), so focus-layer
 *  math must also apply this offset — otherwise HUD label drifts ahead of view. */
const LOOK_AHEAD = 0.35

/** Camera pose for dolly-follow at a given signal X position. */
function dollyPose(sigX: number): Pose {
  return {
    // Behind (lower x), slightly above, forward in z — like a chase cam.
    pos: new THREE.Vector3(sigX - 1.1, 0.55, 1.55),
    // Looking slightly ahead of current — into the direction of flow.
    lookAt: new THREE.Vector3(sigX + LOOK_AHEAD, 0.0, 0.0),
  }
}

/** Establishing shot — far out, high up, looking at the whole model. */
function openingPose(): Pose {
  return {
    pos: new THREE.Vector3(0, 6.5, 22),
    lookAt: new THREE.Vector3(0, 0, 0),
  }
}

/** Closing shot — pulled back behind the LM head, showing the output side. */
function closingPose(): Pose {
  return {
    pos: new THREE.Vector3(4.2, 2.2, 8.5),
    lookAt: new THREE.Vector3(2.4, 0, 0),
  }
}

/** Continuous camera pose as a function of progress [0, 1]. */
function poseFor(p: number): Pose {
  p = clamp01(p)

  if (p < OPEN_END) {
    // Opening: establishing → swoop into first-layer dolly (eased smooth)
    const t = smoothstep(0, 1, p / OPEN_END)
    const open = openingPose()
    const start = dollyPose(X_START)
    return {
      pos: new THREE.Vector3().lerpVectors(open.pos, start.pos, t),
      lookAt: new THREE.Vector3().lerpVectors(open.lookAt, start.lookAt, t),
    }
  }

  if (p < FLY_END) {
    // Dolly-follow across the model at constant rate (no easing here).
    // Constant dsigX/dp = constant layer rate during auto-play.
    const sigX = sigXFor(p)!
    return dollyPose(sigX)
  }

  // Closing: dolly end → pulled-back LM view (eased smooth)
  const t = smoothstep(0, 1, (p - FLY_END) / (1 - FLY_END))
  const endDolly = dollyPose(X_END)
  const close = closingPose()
  return {
    pos: new THREE.Vector3().lerpVectors(endDolly.pos, close.pos, t),
    lookAt: new THREE.Vector3().lerpVectors(endDolly.lookAt, close.lookAt, t),
  }
}

/** Current layer index 0..31 derived from the camera's looked-at x position,
 *  so HUD label stays in perfect sync with the visible viewpoint.
 *  Returns -1 during approach/closing phases. */
function focusLayerFor(p: number): number {
  const sigX = sigXFor(p)
  if (sigX === null) return -1
  // Layers placed at lx = (L/31 - 0.5) * TOTAL_WIDTH, so L = (lx/W + 0.5) * 31.
  const lookedAt = sigX + LOOK_AHEAD
  const raw = (lookedAt / TOTAL_WIDTH + 0.5) * (LAYERS - 1)
  return Math.max(0, Math.min(LAYERS - 1, Math.round(raw)))
}

/** Current layer label for HUD — maps progress to a station name. */
function stationFor(p: number): string {
  if (p < 0.02) return 'Approach'
  if (p < OPEN_END) return 'Descent'
  if (p > 0.97) return 'Loop'
  if (p > FLY_END) return 'LM head'
  const L = focusLayerFor(p)
  return `L${String(L).padStart(2, '0')}`
}

function captionFor(p: number): { caption: string; sub: string } {
  if (p < 0.02) {
    return {
      caption: '3.8 billion parameters',
      sub: 'Right now, on your GPU. 292 dispatches per token, 32 layers, 1,024 attention heads.',
    }
  }
  if (p < OPEN_END) {
    return {
      caption: 'Embedding',
      sub: 'Your tokens arrive as 3,072 floats each. Starting the descent.',
    }
  }
  if (p > 0.97) {
    return {
      caption: 'Next token',
      sub: 'Argmax sampled. Residual resets. 292 dispatches, again.',
    }
  }
  if (p > FLY_END) {
    return {
      caption: 'LM head',
      sub: '32,064 vocabulary projections. The winning token lights up.',
    }
  }

  const L = focusLayerFor(p)

  // Educational captions grouped by layer range. These reflect broad
  // interpretability findings — early layers do surface-level work,
  // middle layers build syntax/local structure, later-middle layers
  // build semantic concepts, and late layers shape task-specific output.
  let caption: string
  let sub: string

  if (L <= 3) {
    // Very early: tokenization, surface features
    caption = `Layer ${L} · reading the tokens`
    const earlySubs = [
      'These layers handle <b>token identity</b> and local bigram patterns. Attention mostly looks at adjacent positions.',
      '<b>Surface features</b>: letter case, punctuation, word boundaries, byte-level artifacts from BPE.',
      'The <b>embedding</b> just projected into 3,072 dims. Layer 0-3 is where "what token am I" is still dominant.',
      'Attention heads here are usually <b>position-detectors</b>: "previous token," "token at position 0," etc.',
    ]
    sub = earlySubs[L % earlySubs.length]!
  } else if (L <= 13) {
    // Syntax + local relationships
    caption = `Layer ${L} · building syntax`
    const synSubs = [
      '<b>Syntactic structure</b> forms here. Heads begin to attend across short spans — subject → verb, noun → article.',
      'Classic "<b>induction heads</b>" often appear in this range: they find <i>[A][B] … [A]</i> and predict <i>[B]</i> next.',
      'Part-of-speech disambiguation. The model decides whether <i>"bank"</i> is a noun or verb using surrounding tokens.',
      'Local phrase-level features. Attention spans widen from ~2 tokens to ~10.',
      'Factual <i>retrieval</i> begins in the FFN — early facts about specific token patterns get written to the residual.',
    ]
    sub = synSubs[L % synSubs.length]!
  } else if (L <= 22) {
    // Semantic / long-range
    caption = `Layer ${L} · building meaning`
    const semSubs = [
      '<b>Semantic concepts</b>. Abstract features like "this is a question," "this is about food," "this is formal."',
      'Long-range attention. Some heads look 50+ tokens back — coreference resolution lives in this range.',
      'The FFN slab stores most of the <b>world knowledge</b> at this depth. Facts are retrieved here.',
      'Cross-token <b>binding</b>: relating an entity mentioned earlier to a pronoun mentioned now.',
      'Attention sharpens — heads often become near-one-hot on a single past token they care about.',
      'The residual norm peaks in this band. Later layers will start pulling information out rather than adding.',
    ]
    sub = semSubs[L % semSubs.length]!
  } else if (L <= 29) {
    // Task-specific circuits + output shaping
    caption = `Layer ${L} · shaping output`
    const lateSubs = [
      'Task-specific circuits. Formatting, tone, style, and chat-template behavior live here.',
      'The model is now writing <i>toward</i> the next token. FFN outputs bias the residual toward specific vocab clusters.',
      'Attention narrows to recent context. The horizon shrinks — last few tokens dominate again, but semantically now.',
      '<b>Copy heads</b> often fire here — "if a rare name appeared earlier, reproduce it verbatim."',
    ]
    sub = lateSubs[L % lateSubs.length]!
  } else {
    // Final block(s)
    caption = L === LAYERS - 1 ? `Layer ${L} · final block` : `Layer ${L} · almost out`
    const finalSubs = [
      'Final residual norm applied. Everything here ends up in the LM head.',
      'The last transformer block. Next stop: LM head → 32,064 logits → softmax → next token.',
      'If a token is "decided," it was decided somewhere before layer 28. This layer mostly fine-tunes.',
    ]
    sub = finalSubs[L % finalSubs.length]!
  }

  return { caption, sub }
}

export interface JourneyHandle {
  enter(): void
  exit(): void
  isActive(): boolean
  setProgress(p: number): void
}

export function createJourney(vis: BrainVisualizer): JourneyHandle {
  let active = false
  let progTarget = 0            // scroll/key/auto-advance writes here
  let progCurrent = 0           // smoothed — what drives the camera
  let autoPlay = false
  let userAdvanced = false      // becomes true after first scroll/key — lets us fade out "scroll hint"

  // Orbit offsets applied around the current focus point
  let orbitYaw = 0
  let orbitPitch = 0
  let isDragging = false
  let lastPointerX = 0
  let lastPointerY = 0

  const hudEl = document.getElementById('journey-hud')
  const capEl = document.getElementById('journey-caption')
  const subEl = document.getElementById('journey-sub')
  const barEl = document.getElementById('journey-progress')
  const layerEl = document.getElementById('journey-layer')
  const playEl = document.getElementById('journey-play')
  const hintEl = document.getElementById('journey-hint')
  const stripEl = document.getElementById('journey-strip')

  // Build the 32-dot layer strip once
  const stripDots: HTMLElement[] = []
  if (stripEl && stripEl.childElementCount === 0) {
    for (let L = 0; L < LAYERS; L++) {
      const dot = document.createElement('div')
      dot.className = 'js-dot future'
      stripEl.appendChild(dot)
      stripDots.push(dot)
    }
  }

  let lastStation = ''
  let lastStripLayer = -2

  function updateHUD(p: number): void {
    const station = stationFor(p)
    if (layerEl && station !== lastStation) {
      layerEl.textContent = station
      lastStation = station
      const { caption, sub } = captionFor(p)
      if (capEl) {
        capEl.textContent = caption
        capEl.classList.remove('flash')
        void capEl.offsetWidth
        capEl.classList.add('flash')
      }
      // sub can contain trusted <b>/<i> markup for educational emphasis
      if (subEl) subEl.innerHTML = sub
    }
    if (barEl) barEl.style.width = (p * 100).toFixed(1) + '%'
    if (playEl) {
      playEl.textContent = autoPlay ? '▌▌ pause' : '▶ play'
      playEl.classList.toggle('on', autoPlay)
    }
    if (hintEl && userAdvanced) {
      hintEl.classList.add('dim')
    }

    // Update 32-dot strip — only when the current layer actually changes.
    const fl = focusLayerFor(p)
    if (fl !== lastStripLayer && stripDots.length === LAYERS) {
      lastStripLayer = fl
      for (let L = 0; L < LAYERS; L++) {
        const d = stripDots[L]!
        d.classList.remove('past', 'current', 'future')
        if (fl < 0) {
          // approach/closing — show approach/pull-back progress as a
          // symmetric "riser" across all dots
          d.classList.add(p < 0.5 ? 'future' : 'past')
        } else if (L < fl) d.classList.add('past')
        else if (L === fl) d.classList.add('current')
        else d.classList.add('future')
      }
    }
  }

  function applyOrbit(basePose: Pose): Pose {
    if (orbitYaw === 0 && orbitPitch === 0) return basePose
    const offset = basePose.pos.clone().sub(basePose.lookAt)
    // Yaw around world-up
    offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), orbitYaw)
    // Pitch around local-right (perpendicular to offset and world-up)
    const right = new THREE.Vector3().crossVectors(offset, new THREE.Vector3(0, 1, 0)).normalize()
    offset.applyAxisAngle(right, orbitPitch)
    return {
      pos: basePose.lookAt.clone().add(offset),
      lookAt: basePose.lookAt.clone(),
    }
  }

  // ─── main animation loop ───
  function tick(): void {
    if (active) {
      if (autoPlay) progTarget = clamp01(progTarget + AUTO_RATE)
      // Smooth the progress we actually render — prevents jittery wheel
      progCurrent += (progTarget - progCurrent) * LERP_TO_PROG
      // Snap when close enough to avoid endless tiny-delta churn
      if (Math.abs(progTarget - progCurrent) < 1e-4) progCurrent = progTarget

      const basePose = poseFor(progCurrent)
      const finalPose = applyOrbit(basePose)
      vis.setJourneyCamera(finalPose.pos, finalPose.lookAt)
      vis.setJourneyFocusLayer(focusLayerFor(progCurrent))
      updateHUD(progCurrent)
    }
    requestAnimationFrame(tick)
  }
  tick()

  // ─── wheel input ───
  function onWheel(e: WheelEvent): void {
    if (!active) return
    e.preventDefault()
    if (autoPlay) autoPlay = false  // scrolling interrupts auto-play
    progTarget = clamp01(progTarget + e.deltaY * WHEEL_SCALE)
    userAdvanced = true
    // Scrolling also decays orbit back to zero — user wants the journey view
    orbitYaw *= 0.85
    orbitPitch *= 0.85
  }

  // ─── keyboard input ───
  function onKey(e: KeyboardEvent): void {
    if (!active) return
    const target = e.target as HTMLElement | null
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return

    if (e.key === ' ') {
      e.preventDefault()
      autoPlay = !autoPlay
      userAdvanced = true
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
      e.preventDefault()
      autoPlay = false
      progTarget = clamp01(progTarget + KEY_STEP)
      userAdvanced = true
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
      e.preventDefault()
      autoPlay = false
      progTarget = clamp01(progTarget - KEY_STEP)
      userAdvanced = true
    } else if (e.key === 'Home') {
      autoPlay = false
      progTarget = 0
      userAdvanced = true
    } else if (e.key === 'End') {
      autoPlay = false
      progTarget = 1
      userAdvanced = true
    } else if (e.key === 'r' || e.key === 'R') {
      // Reset orbit view
      orbitYaw = 0
      orbitPitch = 0
    }
  }

  // ─── drag-to-orbit ───
  function onPointerDown(e: PointerEvent): void {
    if (!active) return
    // Only left button; ignore clicks on HUD (let UI receive them)
    if (e.button !== 0) return
    const target = e.target as HTMLElement | null
    if (target && target.closest('#journey-hud, #journey-exit, .mode-bar, .boot-screen')) return
    isDragging = true
    lastPointerX = e.clientX
    lastPointerY = e.clientY
  }
  function onPointerMove(e: PointerEvent): void {
    if (!isDragging) return
    const dx = e.clientX - lastPointerX
    const dy = e.clientY - lastPointerY
    lastPointerX = e.clientX
    lastPointerY = e.clientY
    orbitYaw -= dx * 0.005
    orbitPitch -= dy * 0.005
    orbitPitch = Math.max(-0.9, Math.min(0.9, orbitPitch))
  }
  function onPointerUp(): void {
    isDragging = false
  }

  // ─── touch pan ───
  let touchY = 0
  function onTouchStart(e: TouchEvent): void {
    if (!active) return
    touchY = e.touches[0].clientY
  }
  function onTouchMove(e: TouchEvent): void {
    if (!active) return
    e.preventDefault()
    const y = e.touches[0].clientY
    const dy = y - touchY
    touchY = y
    progTarget = clamp01(progTarget - dy * TOUCH_SCALE)
    userAdvanced = true
  }

  window.addEventListener('wheel', onWheel, { passive: false })
  window.addEventListener('keydown', onKey)
  window.addEventListener('pointerdown', onPointerDown)
  window.addEventListener('pointermove', onPointerMove)
  window.addEventListener('pointerup', onPointerUp)
  window.addEventListener('pointercancel', onPointerUp)
  window.addEventListener('touchstart', onTouchStart, { passive: false })
  window.addEventListener('touchmove', onTouchMove, { passive: false })

  // HUD-local click handler for the play button (delegated on enter())
  playEl?.addEventListener('click', () => {
    if (!active) return
    autoPlay = !autoPlay
    userAdvanced = true
  })

  return {
    enter() {
      if (active) return
      active = true
      progTarget = 0
      progCurrent = 0
      autoPlay = false
      orbitYaw = 0
      orbitPitch = 0
      userAdvanced = false
      lastStation = ''
      lastStripLayer = -2
      if (hudEl) {
        hudEl.style.display = 'block'
        hudEl.style.opacity = '0'
        requestAnimationFrame(() => { if (hudEl) hudEl.style.opacity = '1' })
      }
      if (hintEl) hintEl.classList.remove('dim')
      vis.enableJourneyMode(true)
      updateHUD(0)
    },
    exit() {
      if (!active) return
      active = false
      autoPlay = false
      isDragging = false
      vis.enableJourneyMode(false)
      if (hudEl) {
        hudEl.style.opacity = '0'
        setTimeout(() => { if (hudEl && !active) hudEl.style.display = 'none' }, 400)
      }
    },
    isActive() {
      return active
    },
    setProgress(p: number) {
      progTarget = clamp01(p)
      userAdvanced = true
    },
  }
}

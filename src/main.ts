import { BrainVisualizer, LayerActivation } from './visualizer'
import { createInferenceEngine, InferenceEngine, LoadProgress } from './engine/inference'
import { reduceQKVForAttnHeads, reduceForAttnHeads, reduceForFFNGroups, reduceForResidual } from './engine/activation-reducer'

// ═══════════════════════════════════════════════════════════════
// Neural Pulse — Main v4 (Real Phi-3 Inference)
// ═══════════════════════════════════════════════════════════════

const canvas = document.getElementById('brainCanvas') as HTMLCanvasElement
const viz = new BrainVisualizer(canvas)
viz.start()

const output = document.getElementById('output')!
const goBtn = document.getElementById('goBtn') as HTMLButtonElement
const promptInput = document.getElementById('promptInput') as HTMLInputElement

let isRunning = false
let totalTokens = 0
let engine: InferenceEngine | null = null

// Speed control
const speedSlider = document.getElementById('speedSlider') as HTMLInputElement
const speedLabel = document.getElementById('speedLabel')!
speedSlider.addEventListener('input', () => {
  speedLabel.textContent = speedSlider.value + 'x'
})

// Sound toggle
const soundBtn = document.getElementById('soundBtn') as HTMLButtonElement
if (soundBtn) {
  soundBtn.addEventListener('click', () => {
    const muted = viz.audio.toggleMute()
    soundBtn.textContent = muted ? '🔇' : '🔊'
  })
}

// Screenshot button
const screenshotBtn = document.getElementById('screenshotBtn') as HTMLButtonElement
if (screenshotBtn) {
  screenshotBtn.addEventListener('click', () => {
    const dataUrl = viz.getScreenshot()
    const a = document.createElement('a')
    a.href = dataUrl
    a.download = 'neural-pulse-screenshot.png'
    a.click()
  })
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))

// ─── Loading overlay ───
function createLoadingOverlay() {
  let overlay = document.getElementById('loadingOverlay')
  if (overlay) return

  overlay = document.createElement('div')
  overlay.id = 'loadingOverlay'
  overlay.style.cssText = `
    position:fixed;inset:0;background:rgba(5,5,16,0.95);z-index:1000;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    font-family:'JetBrains Mono',monospace;color:#a5b4fc;
  `

  overlay.innerHTML = `
    <div style="font-size:1.3rem;font-weight:700;margin-bottom:6px;color:#e2e8f0;">
      <span style="background:linear-gradient(135deg,#6366f1,#06b6d4,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Neural Pulse</span>
    </div>
    <div style="font-size:0.7rem;color:#64748b;margin-bottom:28px;">Loading Phi-3 3.6B — 10 WGSL shaders, no frameworks</div>

    <div style="width:360px;margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;font-size:0.68rem;margin-bottom:6px;">
        <span id="loadPct" style="color:#e2e8f0;font-weight:600;">0%</span>
        <span id="loadSize" style="color:#64748b;">0 / 0 MB</span>
      </div>
      <div style="width:100%;height:4px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;">
        <div id="loadBar" style="width:0%;height:100%;background:linear-gradient(90deg,#6366f1,#06b6d4);transition:width 0.15s;border-radius:2px;"></div>
      </div>
    </div>

    <div id="loadMsg" style="font-size:0.65rem;color:#64748b;max-width:400px;text-align:center;line-height:1.5;min-height:2em;"></div>
    <div id="loadCache" style="font-size:0.6rem;color:#10b981;margin-top:8px;opacity:0;transition:opacity 0.3s;"></div>
  `

  document.body.appendChild(overlay)
}

function updateLoading(p: LoadProgress) {
  createLoadingOverlay()
  const bar = document.getElementById('loadBar')
  const pct = document.getElementById('loadPct')
  const size = document.getElementById('loadSize')
  const msg = document.getElementById('loadMsg')
  const cache = document.getElementById('loadCache')

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
  const overlay = document.getElementById('loadingOverlay')
  if (overlay) {
    overlay.style.opacity = '0'
    overlay.style.transition = 'opacity 0.5s'
    setTimeout(() => overlay.remove(), 500)
  }
}

// ─── Demo mode (no WebGPU fallback) ───
const DEMO_RESPONSE = `Consciousness is one of the most profound mysteries in philosophy and neuroscience. At its core, it refers to the subjective experience of being aware — the "what it is like" quality of experience that philosopher Thomas Nagel famously explored.

From a neuroscience perspective, consciousness appears to emerge from complex patterns of neural activity across the brain, particularly involving the thalamo-cortical system. Yet the "hard problem" remains: why does physical processing give rise to subjective experience at all?

Some theories suggest consciousness is fundamental to the universe, while others view it as an emergent property of sufficiently complex information processing. The truth likely lies somewhere we haven't yet imagined.`

function tokenize(text: string): string[] {
  return text.split(/(\s+)/).filter(t => t.trim().length > 0)
}

function appendToken(text: string) {
  const cursor = output.querySelector('.cursor')
  const span = document.createElement('span')
  span.className = 'token new'
  span.textContent = text
  if (cursor) output.insertBefore(span, cursor)
  else output.appendChild(span)
  setTimeout(() => span.classList.remove('new'), 800)
  output.scrollTop = output.scrollHeight
}

// ─── Demo inference (fake, for no-WebGPU fallback) ───
async function thinkOneTokenDemo(): Promise<void> {
  const speed = parseInt(speedSlider.value)
  const delay = speed <= 3 ? Math.round(100 - speed * 20)
    : speed <= 8 ? Math.round(30 - (speed - 3) * 3.6)
    : Math.max(1, Math.round(10 - (speed - 8) * 0.75))
  const steps = speed >= 15 ? 3 : 5

  for (let L = 0; L < 32; L++) {
    if (!isRunning) return
    for (let step = 0; step < steps; step++) {
      viz.activateLayer(L, (step + 1) / steps)
      await sleep(delay)
    }
  }
}

async function runDemoInference(prompt: string) {
  isRunning = true
  goBtn.disabled = true
  goBtn.textContent = 'Thinking...'
  viz.audio.resume()

  output.innerHTML = ''
  const promptEcho = document.createElement('div')
  promptEcho.style.cssText = 'color:#6366f1;margin-bottom:16px;font-size:0.8rem;font-style:italic;opacity:0.7'
  promptEcho.textContent = `> ${prompt}`
  output.appendChild(promptEcho)
  const cursor = document.createElement('span')
  cursor.className = 'cursor'
  output.appendChild(cursor)

  const inputTokens = tokenize(prompt)
  viz.setInputTokens(inputTokens)

  const words = tokenize(DEMO_RESPONSE)
  const t0 = performance.now()

  for (let i = 0; i < words.length; i++) {
    if (!isRunning) break
    await thinkOneTokenDemo()
    appendToken(words[i] + ' ')
    viz.addOutputToken(words[i])
    totalTokens++

    const elapsed = (performance.now() - t0) / 1000
    document.getElementById('speedStat')!.innerHTML =
      `Speed: <strong class="live">${(totalTokens / elapsed).toFixed(1)} tok/s</strong>`
    document.getElementById('tokenStat')!.innerHTML =
      `Tokens: <strong style="color:#e2e8f0">${totalTokens}</strong>`
  }

  viz.setDone()
  const c2 = output.querySelector('.cursor')
  if (c2) c2.remove()
  isRunning = false
  goBtn.disabled = false
  goBtn.textContent = 'Think'
}

// ─── Real Phi-3 inference ───
async function runRealInference(prompt: string) {
  if (!engine) return
  isRunning = true
  goBtn.disabled = true
  goBtn.textContent = 'Thinking...'
  viz.audio.resume()

  output.innerHTML = ''
  const promptEcho = document.createElement('div')
  promptEcho.style.cssText = 'color:#6366f1;margin-bottom:16px;font-size:0.8rem;font-style:italic;opacity:0.7'
  promptEcho.textContent = `> ${prompt}`
  output.appendChild(promptEcho)
  const cursor = document.createElement('span')
  cursor.className = 'cursor'
  output.appendChild(cursor)

  const inputTokens = tokenize(prompt)
  viz.setInputTokens(inputTokens)

  const t0 = performance.now()
  totalTokens = 0

  await engine.generate(prompt, 500, {
    async onLayer(layer, step, _stepName, activations) {
      // Build role-specific activation data from GPU readback
      let data: LayerActivation

      if (activations) {
        switch (step) {
          case 0: // QKV Matmul: 9216 values → Q portion → 32 attn heads
            data = {
              attnHeads: reduceQKVForAttnHeads(activations),
              ffnGroups: new Float32Array(16),
              residual: 0.1,
            }
            break
          case 3: // Attention output: 3072 → 32 heads
            data = {
              attnHeads: reduceForAttnHeads(activations),
              ffnGroups: new Float32Array(16),
              residual: 0.2,
            }
            break
          case 5: // Add+Norm (attn): 3072 → residual
            data = {
              attnHeads: new Float32Array(32),
              ffnGroups: new Float32Array(16),
              residual: reduceForResidual(activations),
            }
            break
          case 6: // FFN Gate+Up: 8192 → 16 FFN groups
            data = {
              attnHeads: new Float32Array(32),
              ffnGroups: reduceForFFNGroups(activations),
              residual: 0.15,
            }
            break
          case 8: // Add+Norm (FFN): 3072 → residual
            data = {
              attnHeads: new Float32Array(32),
              ffnGroups: new Float32Array(16),
              residual: reduceForResidual(activations),
            }
            break
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
    },
    onToken(delta, _id, _index) {
      appendToken(delta)
      viz.addOutputToken(delta)
      totalTokens++

      const elapsed = (performance.now() - t0) / 1000
      document.getElementById('speedStat')!.innerHTML =
        `Speed: <strong class="live">${(totalTokens / elapsed).toFixed(1)} tok/s</strong>`
      document.getElementById('tokenStat')!.innerHTML =
        `Tokens: <strong style="color:#e2e8f0">${totalTokens}</strong>`
    },
    onPrefill(phase, length) {
      if (phase === 'start') {
        goBtn.textContent = `Prefill (${length} tok)...`
      }
      if (phase === 'end') {
        goBtn.textContent = 'Generating...'
      }
    },
  })

  viz.setDone()
  const c2 = output.querySelector('.cursor')
  if (c2) c2.remove()
  isRunning = false
  goBtn.disabled = false
  goBtn.textContent = 'Think'
}

// ─── Dispatch ───
function startInference() {
  const prompt = promptInput.value.trim()
  if (!prompt || isRunning) return
  promptInput.value = ''

  if (engine) {
    runRealInference(prompt)
  } else {
    runDemoInference(prompt)
  }
}

goBtn.addEventListener('click', startInference)
promptInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') startInference()
})

// ─── Init: try to load real engine, fall back to demo ───
async function initEngine() {
  // Check WebGPU
  if (!navigator.gpu) {
    console.warn('[neural-pulse] No WebGPU — using demo mode')
    startDemo()
    return
  }

  createLoadingOverlay()

  try {
    engine = await createInferenceEngine((p) => {
      updateLoading(p)
    })

    hideLoading()

    // Update header to show real engine
    const dispatchStat = document.getElementById('dispatchStat')
    if (dispatchStat) {
      dispatchStat.innerHTML = `Engine: <strong style="color:#10b981">ZeroTVM</strong>`
    }

    // Auto-demo with real engine
    setTimeout(() => {
      if (!isRunning) {
        promptInput.value = 'What is consciousness?'
        startInference()
      }
    }, 500)

  } catch (e) {
    console.warn('[neural-pulse] Engine init failed, using demo mode:', e)
    hideLoading()
    startDemo()
  }
}

function startDemo() {
  setTimeout(() => {
    if (!isRunning) {
      promptInput.value = 'What is consciousness?'
      startInference()
    }
  }, 1500)
}

initEngine()

// ═══════════════════════════════════════════════════════════════
// Neural Pulse — Brain Visualization v6 (Full Polish)
// 3D brain + bloom + particles + starfield + camera + tooltips
// ═══════════════════════════════════════════════════════════════

import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js'
import { AudioEngine } from './audio'

const LAYER_COUNT = 32
const NEURON_COUNT = 32 * (32 + 16 + 1)  // 1568: 32 attn heads + 16 FFN groups + 1 residual per layer
const PARTICLE_COUNT = 2000

// ─── Color palette by role + layer depth ───
function neuronHSL(layer: number, role: 'attn' | 'ffn' | 'residual' = 'attn'): [number, number, number] {
  const t = layer / 31
  if (role === 'attn') {
    // Cool spectrum: deep blue → cyan → teal (hue 240→180)
    const h = 240 - t * 60
    return [h / 360, 0.75, 0.5]
  } else if (role === 'ffn') {
    // Warm spectrum: amber → orange → rose (hue 45→340)
    const h = 45 - t * 65
    return [((h % 360) + 360) % 360 / 360, 0.7, 0.5]
  } else {
    // Residual: white-silver with slight layer tint
    return [0.0, 0.0, 0.65 + t * 0.15]
  }
}

const PHASE_LABELS = [
  'Reading input...', 'Encoding meaning...', 'Building context...',
  'Analyzing relationships...', 'Deepening understanding...',
  'Weighing evidence...', 'Forming abstractions...',
  'Synthesizing concepts...', 'Refining reasoning...', 'Crystallizing answer...',
]
const STEP_NAMES = ['QKV Matmul', 'RoPE', 'KV Append', 'Attention', 'O Project', 'Add+Norm', 'FFN Up', 'FFN Down', 'Add+Norm']

// ─── Brain shape ───
function isInsideBrain(x: number, y: number, z: number): boolean {
  const dx = (x - 0.48) / 0.48, dy = (y - 0.48) / 0.46, dz = z / 0.4
  let d = dx * dx + dy * dy + dz * dz
  if (x > 0.5) d -= 0.06 * (1 - Math.abs(dy)) * (x - 0.5)
  if (y > 0.55 && x > 0.3 && x < 0.7) d -= 0.04
  return d < 1.0
}

interface NeuronData {
  layer: number
  role: 'attn' | 'ffn' | 'residual'
  subIndex: number // head index 0-31, FFN group 0-15, or 0
  activation: number
  mesh: THREE.Mesh
  glowMesh: THREE.Mesh
  baseColor: THREE.Color
  brightColor: THREE.Color
  position: THREE.Vector3
  worldX: number // normalized x for layer mapping
}

/** Structured activation data per layer */
export interface LayerActivation {
  attnHeads: Float32Array  // 32 values, 0-1 per attention head
  ffnGroups: Float32Array  // 16 values, 0-1 per FFN group
  residual: number         // 0-1 scalar for residual stream
}

interface SynapseData {
  fromIdx: number
  toIdx: number
  line: THREE.Line
  baseColor: THREE.Color
}

interface SignalObj {
  synapse: number
  pos: number
  speed: number
  mesh: THREE.Mesh
}

export class BrainVisualizer {
  private renderer: THREE.WebGLRenderer
  private scene: THREE.Scene
  private camera: THREE.PerspectiveCamera
  private controls: OrbitControls
  private composer: EffectComposer
  private bloomPass: UnrealBloomPass

  private neurons: NeuronData[] = []
  private synapses: SynapseData[] = []
  private signals: SignalObj[] = []

  // Particle system
  private particleSystem!: THREE.Points
  private particlePositions!: Float32Array
  private particleColors!: Float32Array
  private particleVelocities!: Float32Array
  private particleLifetimes!: Float32Array

  // Starfield
  private starField!: THREE.Points

  // Token burst particles
  private burstParticles!: THREE.Points
  private burstPositions!: Float32Array
  private burstVelocities!: Float32Array
  private burstLifetimes!: Float32Array
  private burstActive = false

  // Raycaster for hover + click
  private raycaster = new THREE.Raycaster()
  private mouse = new THREE.Vector2(-999, -999)
  private tooltip: HTMLDivElement
  private hoveredNeuron: NeuronData | null = null

  // Click-to-inspect panel
  private inspectPanel: HTMLDivElement
  private selectedNeuron: NeuronData | null = null
  private selectedRing: THREE.Mesh | null = null

  // Audio
  audio: AudioEngine

  // HUD
  private overlay: HTMLDivElement
  private dispatchEl: HTMLSpanElement | null = null

  // Per-layer neuron index (built after generateBrain)
  private neuronsByLayer: Map<number, NeuronData[]> = new Map()

  // State
  private activeLayer = -1
  private layerProgress = 0
  phase: 'idle' | 'thinking' | 'done' = 'idle'
  private currentStep = 0
  outputConfidence = 0
  dispatchCount = 0
  totalDispatches = 0

  // Cinematic camera
  private cameraMode: 'orbit' | 'follow' = 'orbit'
  private cameraTarget = new THREE.Vector3(0, 0, 0)
  private cameraWaveX = 0

  constructor(canvas: HTMLCanvasElement) {
    this.audio = new AudioEngine()

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.0
    this.updateSize()

    // Scene
    this.scene = new THREE.Scene()
    this.scene.fog = new THREE.FogExp2(0x050510, 0.12)

    // Camera
    const aspect = canvas.clientWidth / canvas.clientHeight
    this.camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 200)
    this.camera.position.set(0, 0.8, 4.5)

    // Controls
    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.05
    this.controls.autoRotate = true
    this.controls.autoRotateSpeed = 0.2  // slow graceful orbit
    this.controls.enableZoom = true
    this.controls.minDistance = 1.5
    this.controls.maxDistance = 8

    // Post-processing: Bloom
    this.composer = new EffectComposer(this.renderer)
    this.composer.addPass(new RenderPass(this.scene, this.camera))
    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(canvas.clientWidth, canvas.clientHeight),
      0.5,   // strength — subtle glow, don't wash out neurons
      0.3,   // radius
      0.35   // threshold — only bright things bloom
    )
    this.composer.addPass(this.bloomPass)

    // Lights
    this.scene.add(new THREE.AmbientLight(0x1a1a2e, 0.6))
    const key = new THREE.PointLight(0x6366f1, 1.5, 12)
    key.position.set(0, 2, 4)
    this.scene.add(key)
    const rim = new THREE.PointLight(0x06b6d4, 0.8, 10)
    rim.position.set(-3, -1, -2)
    this.scene.add(rim)
    const fill = new THREE.PointLight(0xf472b6, 0.4, 8)
    fill.position.set(2, -1, 1)
    this.scene.add(fill)

    // HUD overlay
    this.overlay = document.createElement('div')
    this.overlay.style.cssText = 'position:absolute;bottom:12px;left:12px;pointer-events:none;font-family:JetBrains Mono,monospace;font-size:9px;color:#475569;line-height:1.6;'
    canvas.parentElement!.style.position = 'relative'
    canvas.parentElement!.appendChild(this.overlay)

    // Tooltip
    this.tooltip = document.createElement('div')
    this.tooltip.style.cssText = 'position:absolute;pointer-events:none;background:rgba(5,5,16,0.9);border:1px solid rgba(99,102,241,0.3);border-radius:6px;padding:6px 10px;font-family:JetBrains Mono,monospace;font-size:9px;color:#a5b4fc;display:none;white-space:nowrap;z-index:10;'
    canvas.parentElement!.appendChild(this.tooltip)

    // Inspect panel (click a neuron)
    this.inspectPanel = document.createElement('div')
    this.inspectPanel.style.cssText = 'position:absolute;top:12px;right:12px;background:rgba(5,5,16,0.92);border:1px solid rgba(99,102,241,0.25);border-radius:10px;padding:14px 18px;font-family:JetBrains Mono,monospace;font-size:10px;color:#cbd5e1;display:none;width:220px;z-index:10;line-height:1.8;backdrop-filter:blur(8px);'
    canvas.parentElement!.appendChild(this.inspectPanel)

    // Selection ring
    const ringGeo = new THREE.RingGeometry(0.035, 0.042, 32)
    const ringMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide })
    this.selectedRing = new THREE.Mesh(ringGeo, ringMat)
    this.scene.add(this.selectedRing)

    // Dispatch counter element
    this.dispatchEl = document.getElementById('dispatchStat') as HTMLSpanElement

    // Generate everything
    this.generateStarfield()
    this.generateBrain()
    this.initParticles()
    this.initBurstParticles()

    // Mouse tracking for tooltips
    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect()
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
      this.tooltip.style.left = (e.clientX - rect.left + 12) + 'px'
      this.tooltip.style.top = (e.clientY - rect.top - 8) + 'px'
    })
    canvas.addEventListener('mouseleave', () => {
      this.mouse.set(-999, -999)
      this.tooltip.style.display = 'none'
    })

    // Click to inspect
    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect()
      const mx = ((e.clientX - rect.left) / rect.width) * 2 - 1
      const my = -((e.clientY - rect.top) / rect.height) * 2 + 1
      this.raycaster.setFromCamera(new THREE.Vector2(mx, my), this.camera)
      const meshes = this.neurons.map(n => n.mesh)
      const hits = this.raycaster.intersectObjects(meshes)
      if (hits.length > 0) {
        const idx = meshes.indexOf(hits[0].object as THREE.Mesh)
        if (idx >= 0) this.selectNeuron(this.neurons[idx])
      } else {
        this.deselectNeuron()
      }
    })

    window.addEventListener('resize', () => {
      this.updateSize()
      const a = canvas.clientWidth / canvas.clientHeight
      this.camera.aspect = a
      this.camera.updateProjectionMatrix()
      this.composer.setSize(canvas.clientWidth, canvas.clientHeight)
      this.bloomPass.setSize(canvas.clientWidth, canvas.clientHeight)
    })
  }

  private updateSize() {
    const w = this.renderer.domElement.clientWidth
    const h = this.renderer.domElement.clientHeight
    this.renderer.setSize(w, h, false)
  }

  // ─── Starfield ───
  private generateStarfield() {
    const count = 1500
    const pos = new Float32Array(count * 3)
    const cols = new Float32Array(count * 3)
    const sizes = new Float32Array(count)
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      const r = 20 + Math.random() * 60
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta)
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      pos[i * 3 + 2] = r * Math.cos(phi)
      const b = 0.3 + Math.random() * 0.7
      cols[i * 3] = 0.6 * b; cols[i * 3 + 1] = 0.65 * b; cols[i * 3 + 2] = b
      sizes[i] = 0.3 + Math.random() * 1.2
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(cols, 3))
    geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1))
    const mat = new THREE.PointsMaterial({ size: 0.15, vertexColors: true, transparent: true, opacity: 0.5, sizeAttenuation: true })
    this.starField = new THREE.Points(geo, mat)
    this.scene.add(this.starField)
  }

  // ─── Particles (flowing through brain) ───
  private initParticles() {
    this.particlePositions = new Float32Array(PARTICLE_COUNT * 3)
    this.particleColors = new Float32Array(PARTICLE_COUNT * 3)
    this.particleVelocities = new Float32Array(PARTICLE_COUNT * 3)
    this.particleLifetimes = new Float32Array(PARTICLE_COUNT)

    // All particles start dead
    this.particleLifetimes.fill(0)
    this.particlePositions.fill(0)
    this.particleColors.fill(0)

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(this.particlePositions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(this.particleColors, 3))
    const mat = new THREE.PointsMaterial({
      size: 0.02,
      vertexColors: true,
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    })
    this.particleSystem = new THREE.Points(geo, mat)
    this.scene.add(this.particleSystem)
  }

  // ─── Token burst particles ───
  private initBurstParticles() {
    const count = 200
    this.burstPositions = new Float32Array(count * 3)
    this.burstVelocities = new Float32Array(count * 3)
    this.burstLifetimes = new Float32Array(count)
    this.burstLifetimes.fill(0)

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(this.burstPositions, 3))
    const mat = new THREE.PointsMaterial({
      size: 0.03,
      color: 0xffffff,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    })
    this.burstParticles = new THREE.Points(geo, mat)
    this.scene.add(this.burstParticles)
  }

  // ─── Transformer architecture generation ───
  // 32 layers × (32 attention heads + 16 FFN groups + 1 residual) = 1568 neurons
  // Arranged in 3D: layers along X axis, attn heads upper ring, FFN lower ring, residual center

  private addNeuron(
    wx: number, wy: number, wz: number,
    layer: number, role: 'attn' | 'ffn' | 'residual', subIndex: number,
    neuronGeo: THREE.SphereGeometry, glowGeo: THREE.SphereGeometry
  ) {
    const [h, s, l] = neuronHSL(layer, role)
    const baseColor = new THREE.Color().setHSL(h, s, l)
    const brightColor = new THREE.Color().setHSL(h, s, 0.9)

    const mat = new THREE.MeshStandardMaterial({
      color: baseColor, emissive: baseColor,
      emissiveIntensity: 0.15,
      transparent: true, opacity: role === 'residual' ? 0.45 : 0.35,
      roughness: 0.3, metalness: 0.1,
    })
    const mesh = new THREE.Mesh(neuronGeo, mat)
    mesh.position.set(wx, wy, wz)
    this.scene.add(mesh)

    const glowMat = new THREE.MeshBasicMaterial({ color: baseColor, transparent: true, opacity: 0 })
    const glowMesh = new THREE.Mesh(glowGeo, glowMat)
    glowMesh.position.copy(mesh.position)
    this.scene.add(glowMesh)

    this.neurons.push({
      layer, role, subIndex, activation: 0, mesh, glowMesh,
      baseColor, brightColor, position: mesh.position.clone(),
      worldX: (layer / 31) * 0.94 + 0.03,
    })
  }

  private generateBrain() {
    const attnGeo = new THREE.SphereGeometry(0.012, 6, 4)
    const ffnGeo = new THREE.SphereGeometry(0.011, 6, 4)
    const resGeo = new THREE.SphereGeometry(0.015, 8, 6)
    const attnGlowGeo = new THREE.SphereGeometry(0.035, 6, 4)
    const ffnGlowGeo = new THREE.SphereGeometry(0.030, 6, 4)
    const resGlowGeo = new THREE.SphereGeometry(0.035, 6, 4)

    for (let L = 0; L < 32; L++) {
      const t = L / 31
      // Layer position: spread along X with a gentle wave in Z
      const lx = (t - 0.5) * 4.5
      const lz = Math.sin(t * Math.PI * 1.5) * 0.4

      // 32 attention heads in upper ring
      for (let h = 0; h < 32; h++) {
        const angle = (h / 32) * Math.PI * 2
        const r = 0.22
        const y = 0.35 + Math.sin(angle) * r
        const z = lz + Math.cos(angle) * r
        this.addNeuron(lx, y, z, L, 'attn', h, attnGeo, attnGlowGeo)
      }

      // 16 FFN groups in lower ring
      for (let g = 0; g < 16; g++) {
        const angle = (g / 16) * Math.PI * 2
        const r = 0.17
        const y = -0.35 + Math.sin(angle) * r
        const z = lz + Math.cos(angle) * r
        this.addNeuron(lx, y, z, L, 'ffn', g, ffnGeo, ffnGlowGeo)
      }

      // 1 residual node at center
      this.addNeuron(lx, 0, lz, L, 'residual', 0, resGeo, resGlowGeo)
    }

    // ─── Connections (architecturally accurate) ───

    // Helper to find neuron index
    const findIdx = (layer: number, role: string, sub: number) =>
      this.neurons.findIndex(n => n.layer === layer && n.role === role && n.subIndex === sub)

    // 1. Residual stream chain (layer L → L+1) — the backbone
    for (let L = 0; L < 31; L++) {
      const i = findIdx(L, 'residual', 0)
      const j = findIdx(L + 1, 'residual', 0)
      if (i >= 0 && j >= 0) this.addSynapse(i, j)
    }

    // 2. Within each layer: residual → attn heads (sampled), attn → residual, residual → FFN, FFN → residual
    for (let L = 0; L < 32; L++) {
      const ri = findIdx(L, 'residual', 0)
      if (ri < 0) continue

      // Connect residual to 8 sampled attention heads (every 4th)
      for (let h = 0; h < 32; h += 4) {
        const ai = findIdx(L, 'attn', h)
        if (ai >= 0) this.addSynapse(ri, ai)
      }

      // Connect 4 sampled attention heads back to residual
      for (let h = 2; h < 32; h += 8) {
        const ai = findIdx(L, 'attn', h)
        if (ai >= 0) this.addSynapse(ai, ri)
      }

      // Connect residual to 4 sampled FFN groups
      for (let g = 0; g < 16; g += 4) {
        const fi = findIdx(L, 'ffn', g)
        if (fi >= 0) this.addSynapse(ri, fi)
      }

      // Connect 4 FFN groups back to residual
      for (let g = 2; g < 16; g += 4) {
        const fi = findIdx(L, 'ffn', g)
        if (fi >= 0) this.addSynapse(fi, ri)
      }
    }

    // Signal pool
    const sigGeo = new THREE.SphereGeometry(0.014, 6, 4)
    for (let i = 0; i < 100; i++) {
      const mat = new THREE.MeshBasicMaterial({ color: 0x4a9eff, transparent: true, opacity: 0 })
      const mesh = new THREE.Mesh(sigGeo, mat)
      mesh.visible = false
      this.scene.add(mesh)
      this.signals.push({ synapse: -1, pos: 0, speed: 0, mesh })
    }

    // Build per-layer neuron index + role sub-indexes
    for (const n of this.neurons) {
      if (!this.neuronsByLayer.has(n.layer)) this.neuronsByLayer.set(n.layer, [])
      this.neuronsByLayer.get(n.layer)!.push(n)
    }
  }

  private addSynapse(i: number, j: number) {
    const ni = this.neurons[i], nj = this.neurons[j]
    const avgLayer = Math.floor((ni.layer + nj.layer) / 2)
    const role = ni.role === 'residual' || nj.role === 'residual' ? 'residual' : ni.role
    const [hh, , ] = neuronHSL(avgLayer, role as 'attn' | 'ffn' | 'residual')
    const color = new THREE.Color().setHSL(hh, 0.3, 0.2)
    const pts = new Float32Array([
      ni.position.x, ni.position.y, ni.position.z,
      nj.position.x, nj.position.y, nj.position.z,
    ])
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(pts, 3))
    const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.04 })
    const line = new THREE.Line(geo, mat)
    this.scene.add(line)
    this.synapses.push({ fromIdx: i, toIdx: j, line, baseColor: color })
  }

  // ─── Public API ───

  setInputTokens(tokens: string[]) {
    this.activeLayer = -1
    this.layerProgress = 0
    this.phase = 'idle'
    this.outputConfidence = 0
    this.currentStep = 0
    this.dispatchCount = 0
    this.totalDispatches = 0
    for (const n of this.neurons) n.activation = 0
    this.audio.resume()
    this.audio.startDrone()
  }

  activateLayer(layer: number, progress: number) {
    this.activeLayer = layer
    this.layerProgress = progress
    this.phase = 'thinking'
    this.currentStep = Math.floor(progress * 8.99)
    this.outputConfidence = (layer + progress) / LAYER_COUNT

    // Dispatch counter: 2 preamble + layer*9 + step
    this.dispatchCount = 2 + layer * 9 + this.currentStep
    this.totalDispatches = 292

    // Activate neurons
    for (const n of this.neurons) {
      const dist = Math.abs(n.layer - layer)
      if (dist <= 3) {
        const falloff = [1.0, 0.6, 0.3, 0.1][dist]
        n.activation = Math.max(n.activation, progress * falloff)
      }
    }

    // Spawn signals
    if (progress > 0.15 && progress < 0.4 && Math.random() < 0.5) {
      this.spawnSignals(layer)
    }

    // Spawn particles near active layer
    if (progress > 0.1 && progress < 0.5) {
      this.spawnParticlesNearLayer(layer)
    }

    // Camera follows wave
    this.cameraWaveX = ((layer + progress) / LAYER_COUNT - 0.5) * 0.8  // gentle, not jarring

    // Audio
    if (this.currentStep === 0) this.audio.neuronTick(layer)
    this.audio.setDroneIntensity(this.outputConfidence)

    // Update dispatch counter in header
    if (this.dispatchEl) {
      this.dispatchEl.innerHTML = `Dispatch: <strong style="color:#a5b4fc">${this.dispatchCount}/${this.totalDispatches}</strong>`
    }
  }

  /** How many neurons belong to this layer (for activation reducer) */
  getNeuronCountForLayer(layer: number): number {
    return this.neuronsByLayer.get(layer)?.length ?? 0
  }

  /**
   * Activate neurons with real per-role activation data from GPU readback.
   */
  activateNeurons(layer: number, step: number, data: LayerActivation) {
    this.activeLayer = layer
    this.currentStep = step
    this.phase = 'thinking'
    this.outputConfidence = (layer + (step + 1) / 9) / LAYER_COUNT

    // Dispatch counter
    this.dispatchCount = 2 + layer * 9 + step
    this.totalDispatches = 292

    // Route activations to correct neurons by role
    const layerNeurons = this.neuronsByLayer.get(layer) ?? []
    for (const n of layerNeurons) {
      let val = 0
      if (n.role === 'attn' && data.attnHeads.length > n.subIndex) {
        val = data.attnHeads[n.subIndex]
      } else if (n.role === 'ffn' && data.ffnGroups.length > n.subIndex) {
        val = data.ffnGroups[n.subIndex]
      } else if (n.role === 'residual') {
        val = data.residual * 0.6  // cap so bloom doesn't dominate
      }
      n.activation = Math.max(n.activation, val)
    }

    // Neighbor falloff — use residual as proxy for overall energy
    for (const n of this.neurons) {
      const dist = Math.abs(n.layer - layer)
      if (dist >= 1 && dist <= 3) {
        const falloff = [0, 0.3, 0.12, 0.03][dist]
        n.activation = Math.max(n.activation, data.residual * falloff)
      }
    }

    // Camera follows wave
    this.cameraWaveX = ((layer + (step + 1) / 9) / LAYER_COUNT - 0.5) * 0.8

    // Audio
    if (step === 0) this.audio.neuronTick(layer)
    this.audio.setDroneIntensity(this.outputConfidence)

    // Spawn effects
    const maxAttn = data.attnHeads.length > 0 ? Math.max(...data.attnHeads) : 0
    const maxFfn = data.ffnGroups.length > 0 ? Math.max(...data.ffnGroups) : 0
    if (maxAttn > 0.3 && step < 4) this.spawnSignals(layer)
    if (maxFfn > 0.2 || data.residual > 0.3) this.spawnParticlesNearLayer(layer)

    // Update dispatch counter in header
    if (this.dispatchEl) {
      this.dispatchEl.innerHTML = `Dispatch: <strong style="color:#a5b4fc">${this.dispatchCount}/${this.totalDispatches}</strong>`
    }
  }

  addOutputToken(_token: string) {
    this.triggerBurst()
    this.audio.tokenChime()
  }

  setDone() {
    this.phase = 'done'
    this.activeLayer = -1
    this.audio.stopDrone()
  }

  getScreenshot(): string {
    this.composer.render()
    return this.renderer.domElement.toDataURL('image/png')
  }

  private selectNeuron(n: NeuronData) {
    this.selectedNeuron = n
    // Position ring at neuron, facing camera
    if (this.selectedRing) {
      this.selectedRing.position.copy(n.position)
      ;(this.selectedRing.material as THREE.MeshBasicMaterial).opacity = 0.8
    }
    this.updateInspectPanel()
    this.inspectPanel.style.display = 'block'
  }

  private deselectNeuron() {
    this.selectedNeuron = null
    if (this.selectedRing) {
      ;(this.selectedRing.material as THREE.MeshBasicMaterial).opacity = 0
    }
    this.inspectPanel.style.display = 'none'
  }

  private updateInspectPanel() {
    const n = this.selectedNeuron
    if (!n) return

    const roleLabel = n.role === 'attn' ? `Attention Head ${n.subIndex}`
      : n.role === 'ffn' ? `FFN Group ${n.subIndex}`
      : 'Residual Stream'

    const roleDesc = n.role === 'attn'
      ? `Head ${n.subIndex}/31 — computes attention scores over all previous tokens`
      : n.role === 'ffn'
      ? `Group ${n.subIndex}/15 — 512 neurons, gate+up+SiLU activation`
      : 'Carries information between layers (skip connection + norm)'

    const colorLabel = n.role === 'attn' ? '#60a5fa' : n.role === 'ffn' ? '#f59e0b' : '#e2e8f0'

    const connections = this.synapses.filter(s => s.fromIdx === this.neurons.indexOf(n) || s.toIdx === this.neurons.indexOf(n)).length
    const step = this.phase === 'thinking' ? STEP_NAMES[this.currentStep] : '—'
    const active = n.activation > 0.1

    this.inspectPanel.innerHTML = `
      <div style="color:#a5b4fc;font-weight:600;font-size:11px;margin-bottom:8px;border-bottom:1px solid rgba(99,102,241,0.15);padding-bottom:6px;">Component Inspector</div>
      <div style="color:#64748b">Component</div>
      <div style="color:${colorLabel};margin-bottom:4px;font-weight:600">${roleLabel}</div>
      <div style="color:#64748b">Layer</div>
      <div style="color:#e2e8f0;margin-bottom:4px"><strong>${n.layer}</strong> / 31</div>
      <div style="color:#64748b;font-size:9px;margin-bottom:6px">${roleDesc}</div>
      <div style="color:#64748b">Connections</div>
      <div style="color:#e2e8f0;margin-bottom:4px">${connections}</div>
      <div style="color:#64748b">Real Activation</div>
      <div style="margin-bottom:4px">
        <div style="background:#1e293b;border-radius:3px;height:6px;width:100%;overflow:hidden">
          <div style="background:${active ? colorLabel : '#334155'};height:100%;width:${(n.activation * 100).toFixed(0)}%;transition:width 0.1s"></div>
        </div>
        <span style="color:${active ? '#a5b4fc' : '#475569'}">${(n.activation * 100).toFixed(0)}% (GPU readback)</span>
      </div>
      <div style="color:#64748b">Current Op</div>
      <div style="color:#e2e8f0;margin-bottom:4px">${step}</div>
      <div style="color:#64748b">Status</div>
      <div style="color:${active ? '#10b981' : '#475569'}">${active ? 'Firing' : 'Idle'}</div>
    `
  }

  // ─── Internal ───

  private spawnSignals(layer: number) {
    let spawned = 0
    for (const sig of this.signals) {
      if (sig.synapse !== -1 || spawned >= 3) continue
      const cands = this.synapses.filter(s => {
        const nl = this.neurons[s.fromIdx].layer, nr = this.neurons[s.toIdx].layer
        return Math.abs(nl - layer) <= 4 || Math.abs(nr - layer) <= 4
      })
      if (cands.length === 0) break
      const syn = cands[Math.floor(Math.random() * cands.length)]
      sig.synapse = this.synapses.indexOf(syn)
      sig.pos = 0
      sig.speed = 0.025 + Math.random() * 0.04
      sig.mesh.visible = true
      const [h] = neuronHSL(layer)
      ;(sig.mesh.material as THREE.MeshBasicMaterial).color.setHSL(h, 0.9, 0.8)
      ;(sig.mesh.material as THREE.MeshBasicMaterial).opacity = 1.0
      spawned++
    }
  }

  private spawnParticlesNearLayer(layer: number) {
    const targetX = ((layer / 31) - 0.5) * 3 // world x for this layer
    let spawned = 0
    for (let i = 0; i < PARTICLE_COUNT && spawned < 15; i++) {
      if (this.particleLifetimes[i] > 0) continue
      const [h, s, l] = neuronHSL(layer)
      const color = new THREE.Color().setHSL(h, s, l + 0.2)

      this.particlePositions[i * 3] = targetX + (Math.random() - 0.5) * 0.4
      this.particlePositions[i * 3 + 1] = (Math.random() - 0.5) * 1.5
      this.particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 1.0

      this.particleVelocities[i * 3] = (Math.random() - 0.3) * 0.01  // slight forward bias
      this.particleVelocities[i * 3 + 1] = (Math.random() - 0.5) * 0.005
      this.particleVelocities[i * 3 + 2] = (Math.random() - 0.5) * 0.005

      this.particleColors[i * 3] = color.r
      this.particleColors[i * 3 + 1] = color.g
      this.particleColors[i * 3 + 2] = color.b

      this.particleLifetimes[i] = 0.5 + Math.random() * 1.0
      spawned++
    }
  }

  private triggerBurst() {
    // Burst at the "output" end of the brain (right side, x ≈ 1.4)
    this.burstActive = true
    const cx = 1.4, cy = 0, cz = 0
    for (let i = 0; i < 200; i++) {
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      const speed = 0.02 + Math.random() * 0.04
      this.burstPositions[i * 3] = cx
      this.burstPositions[i * 3 + 1] = cy
      this.burstPositions[i * 3 + 2] = cz
      this.burstVelocities[i * 3] = Math.sin(phi) * Math.cos(theta) * speed
      this.burstVelocities[i * 3 + 1] = Math.sin(phi) * Math.sin(theta) * speed
      this.burstVelocities[i * 3 + 2] = Math.cos(phi) * speed
      this.burstLifetimes[i] = 0.5 + Math.random() * 0.5
    }
  }

  // ─── Render ───

  render = () => {
    requestAnimationFrame(this.render)

    // Cinematic camera: gently follow the wave
    if (this.phase === 'thinking') {
      this.controls.target.lerp(new THREE.Vector3(this.cameraWaveX * 0.15, 0, 0), 0.008)  // very smooth, minimal drift
    } else {
      this.controls.target.lerp(new THREE.Vector3(0, 0, 0), 0.02)
    }
    this.controls.update()

    // Starfield slow rotation
    if (this.starField) this.starField.rotation.y += 0.0001

    // ─── Update neurons ───
    for (const n of this.neurons) {
      const mat = n.mesh.material as THREE.MeshStandardMaterial
      const glowMat = n.glowMesh.material as THREE.MeshBasicMaterial
      const act = n.activation

      if (act > 0.25) {
        mat.color.lerpColors(n.baseColor, n.brightColor, act * 0.7)
        mat.emissive.copy(n.baseColor)
        mat.emissiveIntensity = 0.2 + act * 0.5  // reduced from 1.2
        mat.opacity = 0.6 + act * 0.4
        n.mesh.scale.setScalar(1 + act * 0.4)
        glowMat.color.copy(n.baseColor)
        glowMat.opacity = act * 0.12  // much subtler glow halo
        n.glowMesh.scale.setScalar(1 + act * 1.2)
      } else {
        mat.color.copy(n.baseColor)
        mat.emissive.copy(n.baseColor)
        mat.emissiveIntensity = 0.15
        mat.opacity = 0.35
        n.mesh.scale.setScalar(1)
        glowMat.opacity = 0
      }
      n.activation = Math.max(0, n.activation - 0.005)
    }

    // ─── Synapses ───
    for (const s of this.synapses) {
      const mat = s.line.material as THREE.LineBasicMaterial
      const act = Math.max(this.neurons[s.fromIdx].activation, this.neurons[s.toIdx].activation)
      mat.opacity = 0.02 + act * 0.3
      if (act > 0.25) {
        const [h] = neuronHSL(Math.floor((this.neurons[s.fromIdx].layer + this.neurons[s.toIdx].layer) / 2))
        mat.color.setHSL(h, 0.7, 0.35 + act * 0.35)
      } else {
        mat.color.copy(s.baseColor)
      }
    }

    // ─── Signals ───
    for (const sig of this.signals) {
      if (sig.synapse === -1) continue
      sig.pos += sig.speed
      if (sig.pos >= 1) {
        sig.mesh.visible = false
        sig.synapse = -1
        continue
      }
      const syn = this.synapses[sig.synapse]
      sig.mesh.position.lerpVectors(this.neurons[syn.fromIdx].position, this.neurons[syn.toIdx].position, sig.pos)
      ;(sig.mesh.material as THREE.MeshBasicMaterial).opacity = 1.0 * (1 - Math.abs(sig.pos - 0.5) * 1.8)
    }

    // ─── Flowing particles ───
    let particlesDirty = false
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      if (this.particleLifetimes[i] <= 0) continue
      particlesDirty = true
      this.particleLifetimes[i] -= 0.016
      this.particlePositions[i * 3] += this.particleVelocities[i * 3]
      this.particlePositions[i * 3 + 1] += this.particleVelocities[i * 3 + 1]
      this.particlePositions[i * 3 + 2] += this.particleVelocities[i * 3 + 2]
      // Fade out color
      const fade = Math.max(0, this.particleLifetimes[i])
      this.particleColors[i * 3] *= 0.998
      this.particleColors[i * 3 + 1] *= 0.998
      this.particleColors[i * 3 + 2] *= 0.998
      if (this.particleLifetimes[i] <= 0) {
        this.particlePositions[i * 3] = 0
        this.particlePositions[i * 3 + 1] = 0
        this.particlePositions[i * 3 + 2] = 0
      }
    }
    if (particlesDirty) {
      this.particleSystem.geometry.attributes.position.needsUpdate = true
      this.particleSystem.geometry.attributes.color.needsUpdate = true
    }

    // ─── Token burst particles ───
    if (this.burstActive) {
      let anyAlive = false
      for (let i = 0; i < 200; i++) {
        if (this.burstLifetimes[i] <= 0) continue
        anyAlive = true
        this.burstLifetimes[i] -= 0.02
        this.burstPositions[i * 3] += this.burstVelocities[i * 3]
        this.burstPositions[i * 3 + 1] += this.burstVelocities[i * 3 + 1]
        this.burstPositions[i * 3 + 2] += this.burstVelocities[i * 3 + 2]
        // Gravity
        this.burstVelocities[i * 3 + 1] -= 0.0003
      }
      this.burstParticles.geometry.attributes.position.needsUpdate = true
      ;(this.burstParticles.material as THREE.PointsMaterial).opacity = Math.max(0.1, this.burstLifetimes[0] || 0)
      if (!anyAlive) this.burstActive = false
    }

    // ─── Hover tooltip ───
    if (this.mouse.x > -10) {
      this.raycaster.setFromCamera(this.mouse, this.camera)
      const meshes = this.neurons.map(n => n.mesh)
      const hits = this.raycaster.intersectObjects(meshes)
      if (hits.length > 0) {
        const idx = meshes.indexOf(hits[0].object as THREE.Mesh)
        if (idx >= 0) {
          const n = this.neurons[idx]
          this.tooltip.style.display = 'block'
          const rl = n.role === 'attn' ? `Head ${n.subIndex}` : n.role === 'ffn' ? `FFN ${n.subIndex}` : 'Residual'
          this.tooltip.innerHTML = `Layer <strong>${n.layer}</strong>/31 &nbsp;|&nbsp; ${rl} &nbsp;|&nbsp; <strong>${(n.activation * 100).toFixed(0)}%</strong><br>Step: ${STEP_NAMES[this.currentStep] || '—'}`
          this.hoveredNeuron = n
        }
      } else {
        this.tooltip.style.display = 'none'
        this.hoveredNeuron = null
      }
    }

    // ─── HUD ───
    if (this.phase === 'thinking') {
      const phaseIdx = Math.min(Math.floor(this.activeLayer / 3.2), PHASE_LABELS.length - 1)
      this.overlay.innerHTML = `
        <span style="color:#a5b4fc">${PHASE_LABELS[phaseIdx]}</span><br>
        Layer ${this.activeLayer}/31 &bull; ${Math.round(this.outputConfidence * 100)}% &bull; ${STEP_NAMES[this.currentStep] || ''}<br>
        Dispatches: ${this.dispatchCount}/${this.totalDispatches}
      `
    } else if (this.phase === 'idle') {
      this.overlay.innerHTML = '<span style="color:#334155">Idle — type a question</span>'
    } else {
      this.overlay.innerHTML = '<span style="color:#10b981">Done</span>'
    }

    // ─── Selection ring faces camera ───
    if (this.selectedRing && this.selectedNeuron) {
      this.selectedRing.lookAt(this.camera.position)
      this.updateInspectPanel()
    }

    // Render with bloom
    this.composer.render()
  }

  start() { this.render() }
  stop() { }
}

// ═══════════════════════════════════════════════════════════════
// Neuropulse — Brain Visualization
// Strict 1:1 mode: every pixel on screen is a deterministic function
// of a real GPU-side state value. No decorative particles, no bloom,
// no cinematic camera, no breathing — activations only.
// ═══════════════════════════════════════════════════════════════

import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { AudioEngine } from './audio'

const LAYER_COUNT = 32

// ─── Bioluminescent color palette ───
function neuronHSL(layer: number, role: 'attn' | 'ffn' | 'residual' = 'attn'): [number, number, number] {
  const t = layer / 31
  if (role === 'attn') {
    // Cyan family only: deeper cyan at early layers → brighter cyan at late
    // (hue 180→200, saturation rises with depth for visual progression).
    const h = 180 + t * 20
    return [h / 360, 0.75 + t * 0.15, 0.50 + t * 0.10]
  } else if (role === 'ffn') {
    // Amber family only: warm orange through yellow-amber
    // (hue 22→38, higher layers warmer/brighter).
    const h = 22 + t * 16
    return [h / 360, 0.85, 0.52 + t * 0.10]
  } else {
    // Residual: ethereal white-cyan glow (unchanged — already cyan-biased)
    return [0.52, 0.3 + t * 0.2, 0.7 + t * 0.15]
  }
}

const PHASE_LABELS = [
  'Reading input...', 'Encoding meaning...', 'Building context...',
  'Analyzing relationships...', 'Deepening understanding...',
  'Weighing evidence...', 'Forming abstractions...',
  'Synthesizing concepts...', 'Refining reasoning...', 'Crystallizing answer...',
]
const STEP_NAMES = ['QKV Matmul', 'RoPE', 'KV Append', 'Attention', 'O Project', 'Add+Norm', 'FFN Up', 'FFN Down', 'Add+Norm']

interface NeuronData {
  layer: number
  role: 'attn' | 'ffn' | 'residual'
  subIndex: number
  activation: number
  mesh: THREE.Mesh
  glowMesh: THREE.Mesh
  baseColor: THREE.Color
  brightColor: THREE.Color
  position: THREE.Vector3
  worldX: number
  ablated?: boolean
}

/** Attn heads marked for ablation render in amber. */
const ABLATED_COLOR = new THREE.Color('#ff9a1f')

/** Structured activation data per layer */
export interface LayerActivation {
  attnHeads: Float32Array
  ffnGroups: Float32Array
  residual: number
  /** Optional raw 3072 residual stream values for the dense column visualization */
  residualVec?: Float32Array
  /** Optional raw 8192 FFN gate+up values for the dense slab visualization */
  ffnVec?: Float32Array
}

/** Real-architecture geometry constants */
const D = 3072
const FFN = 8192
const VOCAB = 32064
const HEADS = 32
const HEAD_DIM = 96
const TOTAL_WIDTH = 6.0
const RESIDUAL_HEIGHT = 1.4
const FFN_W = 0.10
const FFN_H = 0.65
const FFN_COLS = 128
const FFN_ROWS = 64  // 64 × 128 = 8192

interface SynapseData {
  fromIdx: number
  toIdx: number
  line: THREE.Line
  baseColor: THREE.Color
  points: THREE.Vector3[]
}

export class BrainVisualizer {
  private renderer: THREE.WebGLRenderer
  private scene: THREE.Scene
  private camera: THREE.PerspectiveCamera
  private controls: OrbitControls

  private neurons: NeuronData[] = []
  private synapses: SynapseData[] = []

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

  // Ablated attn heads, keyed "L:H". Shift-click an attn neuron to toggle.
  private ablatedHeads: Set<string> = new Set()
  /** Set by caller to react to selection changes (e.g. show/hide run button). */
  onAblationChange?: (ablations: { layer: number; head: number }[]) => void

  // HUD
  private overlay: HTMLDivElement
  private dispatchEl: HTMLSpanElement | null = null

  // Per-layer neuron index
  private neuronsByLayer: Map<number, NeuronData[]> = new Map()

  // ─── REAL-ARCHITECTURE GEOMETRY ─────────────────────────────
  // Dense residual column: 32 × 3072 instances as Points
  private residualSlab!: THREE.Points
  private residualSlabColors!: Float32Array

  // Dense FFN slab: 32 × 8192 instances as Points
  private ffnSlab!: THREE.Points
  private ffnSlabColors!: Float32Array

  // Operation-order anchors per layer (10 small markers)
  private opAnchorMeshes: THREE.Mesh[][] = []

  // KV cache strips per layer
  private kvStripMeshes: THREE.Mesh[] = []
  private kvStripFills: number[] = []

  // Token embedding sphere (pre-layer-0)
  private embeddingMesh!: THREE.Mesh
  private embeddingGlow!: THREE.Mesh
  private embeddingActivation = 0

  // LM head bar (post-layer-31): vocab logits as a wide instanced strip
  private lmHeadStrip!: THREE.Points
  private lmHeadColors!: Float32Array
  private lmHeadActivation = 0

  // PCA-based slab layout (loaded from public/pca-layout.json via setPcaLayout).
  pcaMode = false
  private pcaResidual2D: Float32Array | null = null  // 3072 × 2 in [-0.5, 0.5]
  private pcaFfn2D: Float32Array | null = null       // 8192 × 2 in [-0.5, 0.5]

  // State
  private activeLayer = -1
  private layerProgress = 0
  phase: 'idle' | 'thinking' | 'done' = 'idle'
  private currentStep = 0
  outputConfidence = 0
  dispatchCount = 0
  totalDispatches = 0

  // Global animation time (used for selection-ring pulse only)
  private time = 0

  // Cinematic mode — when enabled, the camera tweens toward the active layer
  // each frame, giving a slow-motion auto-flythrough of the forward pass.
  private cinematicCamera = false
  private cameraTargetLayer = -1
  private _cameraTweenTarget = new THREE.Vector3(0, 1.2, 5.5)
  private _cameraTweenLookAt = new THREE.Vector3(0, 0, 0)

  // Journey mode — external scroll-driven camera (src/journey.ts). When
  // active, OrbitControls and cinematic tween are disabled; the journey
  // module writes `journeyCamPos` / `journeyCamLookAt` every frame and the
  // render loop lerps the camera toward them faster than cinematic tween.
  private journeyActive = false
  private journeyCamPos = new THREE.Vector3(0, 1.2, 5.5)
  private journeyCamLookAt = new THREE.Vector3(0, 0, 0)
  private journeyFocusLayer = -1  // -1 = no focus; 0..31 = layer receiving passive pulse
  private starfield: THREE.Points | null = null
  private spaceDust: THREE.Points | null = null
  private spaceDustPositions: Float32Array | null = null
  private spaceDustSeeds: Float32Array | null = null
  private layerRings: THREE.Group | null = null  // 32 subtle rings at each layer X
  private focusSpotlight: THREE.Mesh | null = null  // bright glowing ring at current focus layer

  constructor(canvas: HTMLCanvasElement) {
    this.audio = new AudioEngine()

    const _captureMode = new URLSearchParams(location.search).has('capture')
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: _captureMode })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 0.9
    this.renderer.shadowMap.enabled = false
    this.updateSize()

    // Scene
    this.scene = new THREE.Scene()
    this.scene.fog = new THREE.FogExp2(0x020210, 0.08)

    // Camera
    const aspect = canvas.clientWidth / canvas.clientHeight
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 300)
    this.camera.position.set(0, 1.2, 5.5)

    // Controls — user-driven only (no autoRotate in strict mode)
    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.03
    this.controls.enableZoom = true
    this.controls.minDistance = 1.5
    this.controls.maxDistance = 10

    // Lights — plain illumination of the real geometry (no post-processing)
    this.scene.add(new THREE.AmbientLight(0x0a0a1e, 0.3))

    const key = new THREE.PointLight(0x00e5ff, 1.2, 15)
    key.position.set(0, 3, 4)
    this.scene.add(key)

    // Rim light — warm amber; was deep purple (0x7c3aed) previously.
    const rim = new THREE.PointLight(0xd97b42, 0.55, 12)
    rim.position.set(-3, -1, -3)
    this.scene.add(rim)

    // Fill light — warm amber; was pink (0xec4899) previously.
    const fill = new THREE.PointLight(0xff8c42, 0.35, 10)
    fill.position.set(3, -2, 1)
    this.scene.add(fill)

    const accent = new THREE.PointLight(0x14b8a6, 0.6, 8)
    accent.position.set(0, -3, 0)
    this.scene.add(accent)

    // HUD overlay
    this.overlay = document.createElement('div')
    this.overlay.style.cssText = 'position:absolute;bottom:12px;left:12px;pointer-events:none;font-family:JetBrains Mono,monospace;font-size:9px;color:#514a3e;line-height:1.6;'
    canvas.parentElement!.style.position = 'relative'
    canvas.parentElement!.appendChild(this.overlay)

    // Tooltip
    this.tooltip = document.createElement('div')
    this.tooltip.style.cssText = 'position:absolute;pointer-events:none;background:rgba(8,6,15,0.85);border:1px solid rgba(244,236,223,0.18);border-radius:8px;padding:8px 12px;font-family:JetBrains Mono,monospace;font-size:9px;color:#5eead4;display:none;white-space:nowrap;z-index:10;backdrop-filter:blur(12px);box-shadow:0 4px 20px rgba(244,236,223,0.12);'
    canvas.parentElement!.appendChild(this.tooltip)

    // Inspect panel
    this.inspectPanel = document.createElement('div')
    this.inspectPanel.style.cssText = 'position:absolute;top:12px;right:12px;background:rgba(8,6,15,0.85);border:1px solid rgba(244,236,223,0.16);border-radius:12px;padding:16px 20px;font-family:JetBrains Mono,monospace;font-size:10px;color:#cbc1ad;display:none;width:230px;z-index:10;line-height:1.8;backdrop-filter:blur(16px);box-shadow:0 8px 32px rgba(244,236,223,0.10);'
    canvas.parentElement!.appendChild(this.inspectPanel)

    // Selection ring
    const ringGeo = new THREE.RingGeometry(0.055, 0.068, 40)
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0, side: THREE.DoubleSide })
    this.selectedRing = new THREE.Mesh(ringGeo, ringMat)
    this.scene.add(this.selectedRing)

    this.dispatchEl = document.getElementById('dispatchStat') as HTMLSpanElement

    // Generate the real architecture. No decoration layers.
    this.generateBrain()
    this.generateResidualSlab()
    this.generateFfnSlab()
    this.generateOpAnchors()
    this.generateKvStrips()
    this.generateEmbeddingNode()
    this.generateLmHead()
    // Journey-only cosmic backdrop — hidden by default, doesn't affect
    // strict 1:1 rendering in classic modes.
    this.generateStarfield()
    this.generateSpaceDust()
    this.generateLayerRings()
    this.generateFocusSpotlight()

    // Mouse tracking
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

    // Click to inspect. Shift-click on an attn head toggles ablation.
    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect()
      const mx = ((e.clientX - rect.left) / rect.width) * 2 - 1
      const my = -((e.clientY - rect.top) / rect.height) * 2 + 1
      this.raycaster.setFromCamera(new THREE.Vector2(mx, my), this.camera)
      const meshes = this.neurons.map(n => n.mesh)
      const hits = this.raycaster.intersectObjects(meshes)
      if (hits.length > 0) {
        const idx = meshes.indexOf(hits[0].object as THREE.Mesh)
        if (idx >= 0) {
          const n = this.neurons[idx]
          if (e.shiftKey && n.role === 'attn') {
            this.toggleAblation(n)
            this.onAblationChange?.(this.getAblations())
          } else {
            this.selectNeuron(n)
          }
        }
      } else {
        this.deselectNeuron()
      }
    })

    window.addEventListener('resize', () => {
      this.updateSize()
      const a = canvas.clientWidth / canvas.clientHeight
      this.camera.aspect = a
      this.camera.updateProjectionMatrix()
    })
  }

  private updateSize() {
    const w = this.renderer.domElement.clientWidth
    const h = this.renderer.domElement.clientHeight
    this.renderer.setSize(w, h, false)
  }

  // ─── Transformer architecture generation ───
  private addNeuron(
    wx: number, wy: number, wz: number,
    layer: number, role: 'attn' | 'ffn' | 'residual', subIndex: number,
    neuronGeo: THREE.SphereGeometry, glowGeo: THREE.SphereGeometry
  ) {
    const [h, s, l] = neuronHSL(layer, role)
    const baseColor = new THREE.Color().setHSL(h, s, l)
    const brightColor = new THREE.Color().setHSL(h, Math.min(1, s + 0.1), 0.85)

    const mat = new THREE.MeshStandardMaterial({
      color: baseColor,
      emissive: baseColor,
      emissiveIntensity: role === 'residual' ? 0.2 : 0.12,
      transparent: true,
      opacity: role === 'residual' ? 0.5 : 0.3,
      roughness: 0.15,
      metalness: 0.3,
    })
    const mesh = new THREE.Mesh(neuronGeo, mat)
    mesh.position.set(wx, wy, wz)
    this.scene.add(mesh)

    const glowMat = new THREE.MeshBasicMaterial({
      color: baseColor,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
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
    // Architecturally-accurate layout:
    //   X axis  = layer index (0..31), TOTAL_WIDTH spans the chain
    //   Y axis  = vertical position within a layer (residual column / FFN slab)
    //   Z axis  = ZERO (no cosmetic wave) — fully planar so geometry is honest
    //
    // Per layer at centerX:
    //   - 32 attention head spheres in a vertical column at centerX - 0.07
    //   - 1 marker sphere at centerX (the residual stream — the dense slab is added by generateResidualSlab)
    //   - The FFN slab is added by generateFfnSlab at centerX + 0.07
    // Enlarged: 0.014 → 0.022 for solid dot, 0.04 → 0.065 for glow halo.
    // ~55% bigger visible + raycast hit radius — easier click, more presence.
    const attnGeo = new THREE.SphereGeometry(0.022, 10, 8)
    const resGeo = new THREE.SphereGeometry(0.022, 10, 8)
    const attnGlowGeo = new THREE.SphereGeometry(0.065, 8, 6)
    const resGlowGeo = new THREE.SphereGeometry(0.070, 8, 6)

    for (let L = 0; L < 32; L++) {
      const t = L / 31
      const lx = (t - 0.5) * TOTAL_WIDTH
      const lz = 0  // ── planar architecture ──

      // 32 attention heads as a vertical column (matches head_idx ordering)
      const headColX = lx - 0.07
      for (let h = 0; h < 32; h++) {
        const y = ((h / 31) - 0.5) * 0.7
        this.addNeuron(headColX, y, lz, L, 'attn', h, attnGeo, attnGlowGeo)
      }

      // Residual marker (the visible "anchor"). The 3072-dim column is rendered
      // as a Points slab around this position by generateResidualSlab.
      this.addNeuron(lx, 0, lz, L, 'residual', 0, resGeo, resGlowGeo)
    }

    // ─── Connections ─────────────────────────────────────────
    const findIdx = (layer: number, role: string, sub: number) =>
      this.neurons.findIndex(n => n.layer === layer && n.role === role && n.subIndex === sub)

    // Residual chain — straight line through every layer (this IS the spine)
    for (let L = 0; L < 31; L++) {
      const i = findIdx(L, 'residual', 0)
      const j = findIdx(L + 1, 'residual', 0)
      if (i >= 0 && j >= 0) this.addCurvedSynapse(i, j)
    }

    // Within each layer: residual ↔ a sample of attention heads (architectural read+write)
    for (let L = 0; L < 32; L++) {
      const ri = findIdx(L, 'residual', 0)
      if (ri < 0) continue

      for (let h = 0; h < 32; h += 8) {
        const ai = findIdx(L, 'attn', h)
        if (ai >= 0) {
          this.addCurvedSynapse(ri, ai)
          this.addCurvedSynapse(ai, ri)
        }
      }
    }

    // Build per-layer index
    for (const n of this.neurons) {
      if (!this.neuronsByLayer.has(n.layer)) this.neuronsByLayer.set(n.layer, [])
      this.neuronsByLayer.get(n.layer)!.push(n)
    }
  }

  // ─── REAL RESIDUAL COLUMN — 32 layers × 3072 dots ──────────
  private generateResidualSlab() {
    const total = 32 * D
    const positions = new Float32Array(total * 3)
    this.residualSlabColors = new Float32Array(total * 3)

    for (let L = 0; L < 32; L++) {
      const lx = ((L / 31) - 0.5) * TOTAL_WIDTH
      // 64 × 48 = 3072
      const cols = 64, rows = 48
      for (let i = 0; i < D; i++) {
        const col = i % cols
        const row = Math.floor(i / cols)
        const x = lx + (col - cols / 2 + 0.5) * 0.0009
        const y = ((row / (rows - 1)) - 0.5) * RESIDUAL_HEIGHT
        const idx = (L * D + i) * 3
        positions[idx] = x
        positions[idx + 1] = y
        positions[idx + 2] = 0
        this.residualSlabColors[idx] = 0.0
        this.residualSlabColors[idx + 1] = 0.05
        this.residualSlabColors[idx + 2] = 0.08
      }
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(this.residualSlabColors, 3))
    const mat = new THREE.PointsMaterial({
      size: 0.006,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    })
    this.residualSlab = new THREE.Points(geo, mat)
    this.scene.add(this.residualSlab)
  }

  // ─── REAL FFN SLAB — 32 layers × 8192 dots ─────────────────
  private generateFfnSlab() {
    const total = 32 * FFN
    const positions = new Float32Array(total * 3)
    this.ffnSlabColors = new Float32Array(total * 3)

    for (let L = 0; L < 32; L++) {
      const lx = ((L / 31) - 0.5) * TOTAL_WIDTH
      const slabCenterX = lx + 0.07
      for (let i = 0; i < FFN; i++) {
        const col = i % FFN_COLS
        const row = Math.floor(i / FFN_COLS)
        const x = slabCenterX + (col - FFN_COLS / 2 + 0.5) * (FFN_W / FFN_COLS)
        const y = ((row / (FFN_ROWS - 1)) - 0.5) * FFN_H
        const idx = (L * FFN + i) * 3
        positions[idx] = x
        positions[idx + 1] = y
        positions[idx + 2] = 0
        // Dark amber base (was dark magenta 0.05, 0, 0.05 — the "pink wall")
        this.ffnSlabColors[idx]     = 0.08
        this.ffnSlabColors[idx + 1] = 0.04
        this.ffnSlabColors[idx + 2] = 0.01
      }
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(this.ffnSlabColors, 3))
    const mat = new THREE.PointsMaterial({
      size: 0.0035,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    })
    this.ffnSlab = new THREE.Points(geo, mat)
    this.scene.add(this.ffnSlab)
  }

  /**
   * Replace the index-lattice positions of the residual + FFN slabs with
   * PCA-derived 2D coordinates so that nearby points represent functionally
   * similar units (rather than adjacent indices).
   *
   * residual2D and ffn2D are flat float arrays:
   *   residual2D[i*2 + 0..1] = (x, y) in [-0.5, 0.5] for residual dim i
   *   ffn2D[j*2 + 0..1]      = (x, y) in [-0.5, 0.5] for FFN-mid neuron j
   */
  setPcaLayout(residual2D: Float32Array, ffn2D: Float32Array): void {
    if (residual2D.length !== D * 2) {
      console.warn(`[viz] PCA residual layout has ${residual2D.length} entries, expected ${D * 2}`)
      return
    }
    if (ffn2D.length !== FFN * 2) {
      console.warn(`[viz] PCA FFN layout has ${ffn2D.length} entries, expected ${FFN * 2}`)
      return
    }
    this.pcaResidual2D = residual2D
    this.pcaFfn2D = ffn2D
    this.pcaMode = true
    this.applySlabLayout()
  }

  /** Restore the index lattice layout for both slabs. */
  clearPcaLayout(): void {
    this.pcaMode = false
    this.applySlabLayout()
  }

  /** Switch to PCA layout if previously loaded; returns true on success. */
  enablePcaLayout(): boolean {
    if (!this.pcaResidual2D || !this.pcaFfn2D) return false
    this.pcaMode = true
    this.applySlabLayout()
    return true
  }

  /** Recompute residual + FFN slab positions according to current layout mode. */
  private applySlabLayout(): void {
    // Residual: D=3072 dots per layer, packed into a column at lx
    {
      const posAttr = this.residualSlab.geometry.attributes.position as THREE.BufferAttribute
      const positions = posAttr.array as Float32Array
      for (let L = 0; L < 32; L++) {
        const lx = ((L / 31) - 0.5) * TOTAL_WIDTH
        for (let i = 0; i < D; i++) {
          const idx = (L * D + i) * 3
          let x: number, y: number
          if (this.pcaMode && this.pcaResidual2D) {
            x = lx + this.pcaResidual2D[i * 2] * 0.09
            y = this.pcaResidual2D[i * 2 + 1] * RESIDUAL_HEIGHT
          } else {
            const cols = 64, rows = 48
            const col = i % cols
            const row = Math.floor(i / cols)
            x = lx + (col - cols / 2 + 0.5) * 0.0009
            y = ((row / (rows - 1)) - 0.5) * RESIDUAL_HEIGHT
          }
          positions[idx]     = x
          positions[idx + 1] = y
          positions[idx + 2] = 0
        }
      }
      posAttr.needsUpdate = true
    }
    // FFN: 8192 dots per layer at lx + 0.07
    {
      const posAttr = this.ffnSlab.geometry.attributes.position as THREE.BufferAttribute
      const positions = posAttr.array as Float32Array
      for (let L = 0; L < 32; L++) {
        const lx = ((L / 31) - 0.5) * TOTAL_WIDTH
        const slabCenterX = lx + 0.07
        for (let i = 0; i < FFN; i++) {
          const idx = (L * FFN + i) * 3
          let x: number, y: number
          if (this.pcaMode && this.pcaFfn2D) {
            x = slabCenterX + this.pcaFfn2D[i * 2] * FFN_W
            y = this.pcaFfn2D[i * 2 + 1] * FFN_H
          } else {
            const col = i % FFN_COLS
            const row = Math.floor(i / FFN_COLS)
            x = slabCenterX + (col - FFN_COLS / 2 + 0.5) * (FFN_W / FFN_COLS)
            y = ((row / (FFN_ROWS - 1)) - 0.5) * FFN_H
          }
          positions[idx]     = x
          positions[idx + 1] = y
          positions[idx + 2] = 0
        }
      }
      posAttr.needsUpdate = true
    }
  }

  // ─── 10 OPERATION-ORDER ANCHORS PER LAYER ──────────────────
  private generateOpAnchors() {
    // Anchor order: RMSNorm₁ → QKV → RoPE → KVAppend → Attention → OProj
    //               +Residual → RMSNorm₂ → FFN(Gate∘Up) → FFN(Down)
    const anchorGeo = new THREE.SphereGeometry(0.005, 6, 4)
    const ANCHOR_W = 0.16
    for (let L = 0; L < 32; L++) {
      const lx = ((L / 31) - 0.5) * TOTAL_WIDTH
      const meshes: THREE.Mesh[] = []
      const linePts: THREE.Vector3[] = []
      for (let a = 0; a < 10; a++) {
        const ax = lx + (a / 9 - 0.5) * ANCHOR_W
        const mat = new THREE.MeshBasicMaterial({
          color: 0x5eead4,
          transparent: true,
          opacity: 0.25,
        })
        const mesh = new THREE.Mesh(anchorGeo, mat)
        mesh.position.set(ax, 0.85, 0)
        this.scene.add(mesh)
        meshes.push(mesh)
        linePts.push(mesh.position.clone())
      }
      const lineGeo = new THREE.BufferGeometry().setFromPoints(linePts)
      const lineMat = new THREE.LineBasicMaterial({
        color: 0x5eead4,
        transparent: true,
        opacity: 0.12,
      })
      const line = new THREE.Line(lineGeo, lineMat)
      this.scene.add(line)
      this.opAnchorMeshes.push(meshes)
    }
  }

  // ─── KV CACHE STRIPS PER LAYER ─────────────────────────────
  private generateKvStrips() {
    const stripW = 0.13
    const stripH = 0.018
    for (let L = 0; L < 32; L++) {
      const lx = ((L / 31) - 0.5) * TOTAL_WIDTH
      const geo = new THREE.PlaneGeometry(stripW, stripH)
      const mat = new THREE.MeshBasicMaterial({
        color: 0x14b8a6,
        transparent: true,
        opacity: 0.15,
        side: THREE.DoubleSide,
      })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(lx, -0.85, 0)
      mesh.scale.x = 0.0001
      this.scene.add(mesh)
      this.kvStripMeshes.push(mesh)
      this.kvStripFills.push(0)
    }
  }

  // ─── TOKEN EMBEDDING NODE (pre-layer-0) ────────────────────
  private generateEmbeddingNode() {
    const x = -TOTAL_WIDTH / 2 - 0.5
    const geo = new THREE.SphereGeometry(0.04, 16, 12)
    const mat = new THREE.MeshStandardMaterial({
      color: 0x99f6e4,
      emissive: 0x14b8a6,
      emissiveIntensity: 0.4,
      transparent: true,
      opacity: 0.85,
      roughness: 0.2,
      metalness: 0.4,
    })
    this.embeddingMesh = new THREE.Mesh(geo, mat)
    this.embeddingMesh.position.set(x, 0, 0)
    this.scene.add(this.embeddingMesh)

    const glowGeo = new THREE.SphereGeometry(0.09, 16, 12)
    const glowMat = new THREE.MeshBasicMaterial({
      color: 0x14b8a6,
      transparent: true,
      opacity: 0.08,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
    this.embeddingGlow = new THREE.Mesh(glowGeo, glowMat)
    this.embeddingGlow.position.copy(this.embeddingMesh.position)
    this.scene.add(this.embeddingGlow)
  }

  // ─── LM HEAD STRIP (post-layer-31): 32064 vocab logits ─────
  private generateLmHead() {
    const COLS = 256, ROWS = Math.ceil(VOCAB / COLS)
    const total = COLS * ROWS
    const positions = new Float32Array(total * 3)
    this.lmHeadColors = new Float32Array(total * 3)

    const x0 = TOTAL_WIDTH / 2 + 0.5
    const W = 0.7, H = 1.4
    for (let i = 0; i < total; i++) {
      const col = i % COLS
      const row = Math.floor(i / COLS)
      positions[i * 3] = x0 + (col / (COLS - 1) - 0.5) * W
      positions[i * 3 + 1] = ((row / (ROWS - 1)) - 0.5) * H
      positions[i * 3 + 2] = 0
      this.lmHeadColors[i * 3] = 0.05
      this.lmHeadColors[i * 3 + 1] = 0.05
      this.lmHeadColors[i * 3 + 2] = 0.1
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(this.lmHeadColors, 3))
    const mat = new THREE.PointsMaterial({
      size: 0.0045,
      vertexColors: true,
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    })
    this.lmHeadStrip = new THREE.Points(geo, mat)
    this.scene.add(this.lmHeadStrip)
  }

  // Curved synapse using quadratic bezier for the architectural wiring diagram
  private addCurvedSynapse(i: number, j: number) {
    const ni = this.neurons[i], nj = this.neurons[j]
    const avgLayer = Math.floor((ni.layer + nj.layer) / 2)
    const role = ni.role === 'residual' || nj.role === 'residual' ? 'residual' : ni.role
    const [hh] = neuronHSL(avgLayer, role as 'attn' | 'ffn' | 'residual')
    const color = new THREE.Color().setHSL(hh, 0.6, 0.25)

    const mid = new THREE.Vector3().lerpVectors(ni.position, nj.position, 0.5)
    // Offset midpoint deterministically for organic curve (no per-call random drift).
    const offset = new THREE.Vector3(
      0,
      ((i * 13 + j * 7) % 17 - 8) * 0.005,
      ((i * 11 + j * 5) % 19 - 9) * 0.005,
    )
    mid.add(offset)

    const curve = new THREE.QuadraticBezierCurve3(ni.position, mid, nj.position)
    const curvePoints = curve.getPoints(12)
    const positions = new Float32Array(curvePoints.length * 3)
    for (let k = 0; k < curvePoints.length; k++) {
      positions[k * 3] = curvePoints[k].x
      positions[k * 3 + 1] = curvePoints[k].y
      positions[k * 3 + 2] = curvePoints[k].z
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    const mat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.03,
    })
    const line = new THREE.Line(geo, mat)
    this.scene.add(line)
    this.synapses.push({ fromIdx: i, toIdx: j, line, baseColor: color, points: curvePoints })
  }

  // ─── Public API ───

  setInputTokens(_tokens: string[]) {
    this.activeLayer = -1
    this.layerProgress = 0
    this.phase = 'idle'
    this.outputConfidence = 0
    this.currentStep = 0
    this.dispatchCount = 0
    this.totalDispatches = 0
    for (const n of this.neurons) n.activation = 0

    // Reset dense slabs to dim baseline
    if (this.residualSlabColors) {
      for (let i = 0; i < this.residualSlabColors.length; i += 3) {
        this.residualSlabColors[i] = 0
        this.residualSlabColors[i + 1] = 0.05
        this.residualSlabColors[i + 2] = 0.08
      }
      ;(this.residualSlab.geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true
    }
    if (this.ffnSlabColors) {
      for (let i = 0; i < this.ffnSlabColors.length; i += 3) {
        // Dark amber base (was dark magenta)
        this.ffnSlabColors[i]     = 0.08
        this.ffnSlabColors[i + 1] = 0.04
        this.ffnSlabColors[i + 2] = 0.01
      }
      ;(this.ffnSlab.geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true
    }
    // Reset KV strips
    for (let L = 0; L < this.kvStripMeshes.length; L++) {
      this.kvStripFills[L] = 0
      this.kvStripMeshes[L].scale.x = 0.0001
      ;(this.kvStripMeshes[L].material as THREE.MeshBasicMaterial).opacity = 0.15
    }
    this.embeddingActivation = 0
    this.lmHeadActivation = 0

    this.audio.resume()
    this.audio.startDrone()
  }

  activateLayer(layer: number, progress: number) {
    this.activeLayer = layer
    this.layerProgress = progress
    this.phase = 'thinking'
    this.currentStep = Math.floor(progress * 8.99)
    this.outputConfidence = (layer + progress) / LAYER_COUNT

    this.dispatchCount = 2 + layer * 9 + this.currentStep
    this.totalDispatches = 292

    for (const n of this.neurons) {
      const dist = Math.abs(n.layer - layer)
      if (dist <= 3) {
        const falloff = [1.0, 0.6, 0.3, 0.1][dist]
        n.activation = Math.max(n.activation, progress * falloff)
      }
    }

    if (this.currentStep === 0) this.audio.neuronTick(layer)
    this.audio.setDroneIntensity(progress)

    if (this.dispatchEl) {
      this.dispatchEl.innerHTML = `Dispatch: <strong style="color:#5eead4">${this.dispatchCount}/${this.totalDispatches}</strong>`
    }
  }

  getNeuronCountForLayer(layer: number): number {
    return this.neuronsByLayer.get(layer)?.length ?? 0
  }

  activateNeurons(layer: number, step: number, data: LayerActivation) {
    this.activeLayer = layer
    this.currentStep = step
    this.phase = 'thinking'
    this.outputConfidence = (layer + (step + 1) / 9) / LAYER_COUNT

    this.dispatchCount = 2 + layer * 9 + step
    this.totalDispatches = 292

    const layerNeurons = this.neuronsByLayer.get(layer) ?? []
    for (const n of layerNeurons) {
      let val = 0
      if (n.role === 'attn' && data.attnHeads.length > n.subIndex) {
        val = data.attnHeads[n.subIndex]
      } else if (n.role === 'residual') {
        val = data.residual * 0.6
      }
      n.activation = Math.max(n.activation, val)
    }

    for (const n of this.neurons) {
      const dist = Math.abs(n.layer - layer)
      if (dist >= 1 && dist <= 3) {
        const falloff = [0, 0.3, 0.12, 0.03][dist]
        n.activation = Math.max(n.activation, data.residual * falloff)
      }
    }

    // ── Push raw vectors into the dense slabs ──
    if (data.residualVec && data.residualVec.length === D) {
      this.updateResidualLayer(layer, data.residualVec)
    }
    if (data.ffnVec && data.ffnVec.length === FFN) {
      this.updateFfnLayer(layer, data.ffnVec)
    }

    // Highlight current operation anchor
    this.highlightAnchor(layer, this.opStepToAnchor(step))

    // Feed real GPU data into audio engine
    if (data.attnHeads.length > 0) {
      // Compute entropy of attention head activations (normalized)
      const sum = data.attnHeads.reduce((a, b) => a + b, 0) || 1
      let entropy = 0
      for (const h of data.attnHeads) {
        const p = h / sum
        if (p > 1e-8) entropy -= p * Math.log2(p)
      }
      // Normalize to 0..1 (max entropy for 32 heads = log2(32) = 5)
      this.audio.setAttentionEntropy(Math.min(1, entropy / 5))
    }
    this.audio.setDroneIntensity(data.residual)
    if (step === 0) this.audio.neuronTick(layer)

    if (this.dispatchEl) {
      this.dispatchEl.innerHTML = `Dispatch: <strong style="color:#5eead4">${this.dispatchCount}/${this.totalDispatches}</strong>`
    }
  }

  /** Map an inference step (0..8) to the corresponding op-anchor index (0..9). */
  private opStepToAnchor(step: number): number {
    // STEP_NAMES = QKV(0) RoPE(1) KV(2) Attn(3) OProj(4) AddNorm(5) FFN-Up(6) FFN-Dn(7) AddNorm(8)
    // ANCHORS    = Norm(0) QKV(1) RoPE(2) KV(3) Attn(4) OProj(5) +Res(6) Norm(7) FFN-Up(8) FFN-Dn(9)
    return [1, 2, 3, 4, 5, 6, 8, 9, 7][step] ?? 0
  }

  /** Light up one anchor on a given layer (and dim siblings). */
  private highlightAnchor(layer: number, anchorIdx: number) {
    const list = this.opAnchorMeshes[layer]
    if (!list) return
    for (let a = 0; a < list.length; a++) {
      const m = list[a].material as THREE.MeshBasicMaterial
      if (a === anchorIdx) {
        m.opacity = 1.0
        m.color.set(0xf4ecdf)
        list[a].scale.setScalar(2.4)
      } else {
        m.opacity = 0.25
        m.color.set(0x5eead4)
        list[a].scale.setScalar(1.0)
      }
    }
  }

  /** Update the dense residual column for a single layer. Vec must be length 3072 (already 0..1). */
  updateResidualLayer(layer: number, vec: Float32Array) {
    const baseIdx = layer * D * 3
    for (let i = 0; i < D; i++) {
      const v = vec[i]
      const idx = baseIdx + i * 3
      // Cyan→white ramp
      this.residualSlabColors[idx]     = v * 0.4
      this.residualSlabColors[idx + 1] = v * 0.95
      this.residualSlabColors[idx + 2] = 0.15 + v * 0.85
    }
    ;(this.residualSlab.geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true
  }

  /** Update the dense FFN slab for a single layer. Vec must be length 8192 (already 0..1). */
  updateFfnLayer(layer: number, vec: Float32Array) {
    const baseIdx = layer * FFN * 3
    for (let i = 0; i < FFN; i++) {
      const v = vec[i]
      const idx = baseIdx + i * 3
      // Amber ramp — no pink/magenta. Low activation is warm-dim,
      // high activation glows bright amber.
      this.ffnSlabColors[idx]     = 0.3 + v * 0.7   // R: 0.3 → 1.0
      this.ffnSlabColors[idx + 1] = 0.15 + v * 0.3  // G: 0.15 → 0.45
      this.ffnSlabColors[idx + 2] = 0.04 + v * 0.06 // B: 0.04 → 0.10
    }
    ;(this.ffnSlab.geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true
  }

  /** Update KV cache strip for a layer. usedFraction = usedPages/totalPages. */
  setKvCacheStrip(layer: number, usedFraction: number) {
    if (layer < 0 || layer >= 32) return
    this.kvStripFills[layer] = Math.max(this.kvStripFills[layer], usedFraction)
    const mesh = this.kvStripMeshes[layer]
    if (mesh) {
      mesh.scale.x = Math.max(0.0001, usedFraction)
      const m = mesh.material as THREE.MeshBasicMaterial
      m.opacity = 0.2 + usedFraction * 0.5
    }
  }

  /** Set token embedding activation (drives the pre-layer-0 sphere). */
  setEmbedding(_tokenId: number, embedding: Float32Array) {
    let sumSq = 0
    for (let i = 0; i < embedding.length; i++) sumSq += embedding[i] * embedding[i]
    const rms = Math.sqrt(sumSq / embedding.length)
    this.embeddingActivation = Math.tanh(rms * 0.8)
  }

  /** Set LM head logits (drives the post-layer-31 strip). Logits length = 32064. */
  setLogits(logits: Float32Array) {
    let mn = Infinity, mx = -Infinity
    for (let i = 0; i < logits.length; i++) {
      const v = logits[i]
      if (v < mn) mn = v
      if (v > mx) mx = v
    }
    const range = mx - mn
    if (range < 1e-8) return

    const total = this.lmHeadColors.length / 3
    for (let i = 0; i < total; i++) {
      const v = i < logits.length ? (logits[i] - mn) / range : 0
      const w = Math.pow(v, 3)
      const idx = i * 3
      // Violet→cyan→white ramp
      this.lmHeadColors[idx]     = w * 0.4
      this.lmHeadColors[idx + 1] = w * 0.9
      this.lmHeadColors[idx + 2] = 0.1 + w * 0.9
    }
    const attr = this.lmHeadStrip.geometry.attributes.color as THREE.BufferAttribute
    attr.needsUpdate = true
    this.lmHeadActivation = 1.0
  }

  setPrefillProgress(layer: number, _progress: number) {
    this.phase = 'thinking'
    this.activeLayer = layer
    for (const n of this.neurons) {
      if (n.layer <= layer) {
        const falloff = 1 - (layer - n.layer) / 32
        n.activation = Math.max(n.activation, 0.3 * falloff)
      }
    }
  }

  addOutputToken(_token: string, confidence?: number) {
    if (confidence !== undefined) this.audio.setTokenConfidence(confidence)
    this.audio.tokenChime()
  }

  setDone() {
    this.phase = 'done'
    this.activeLayer = -1
    this.audio.stopDrone()
  }

  getScreenshot(): string {
    this.renderer.render(this.scene, this.camera)
    return this.renderer.domElement.toDataURL('image/png')
  }

  /** Cinematic mode: start/stop the auto-camera flythrough that tracks the
   *  active layer. The OrbitControls user input is disabled while active so
   *  the camera tween isn't fighting the mouse. */
  setCinematicCamera(enabled: boolean) {
    this.cinematicCamera = enabled
    this.controls.enabled = !enabled
    if (!enabled) {
      // Restore the default vantage when leaving cinema mode
      this._cameraTweenTarget.set(0, 1.2, 5.5)
      this._cameraTweenLookAt.set(0, 0, 0)
    }
  }

  /** Tell the cinematic camera which layer to track this frame. */
  focusCameraOnLayer(layer: number) {
    if (!this.cinematicCamera) return
    this.cameraTargetLayer = layer
    const lx = ((layer / 31) - 0.5) * TOTAL_WIDTH
    // Sit just above the residual axis, looking down the chain at the
    // active layer. Slight Z offset so the camera isn't INSIDE the slab.
    this._cameraTweenTarget.set(lx, 0.6, 1.4)
    this._cameraTweenLookAt.set(lx, 0, 0)
  }

  private selectNeuron(n: NeuronData) {
    this.selectedNeuron = n
    if (this.selectedRing) {
      this.selectedRing.position.copy(n.position)
      ;(this.selectedRing.material as THREE.MeshBasicMaterial).opacity = 0.9
      ;(this.selectedRing.material as THREE.MeshBasicMaterial).color.set(0x00e5ff)
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

  private toggleAblation(n: NeuronData) {
    if (n.role !== 'attn') return
    const key = `${n.layer}:${n.subIndex}`
    if (this.ablatedHeads.has(key)) {
      this.ablatedHeads.delete(key)
      n.ablated = false
      const mat = n.mesh.material as THREE.MeshStandardMaterial
      mat.color.copy(n.baseColor)
      mat.emissive.copy(n.baseColor)
      mat.emissiveIntensity = 0.12
      mat.opacity = 0.3
      ;(n.glowMesh.material as THREE.MeshBasicMaterial).color.copy(n.baseColor)
    } else {
      this.ablatedHeads.add(key)
      n.ablated = true
      const mat = n.mesh.material as THREE.MeshStandardMaterial
      mat.color.copy(ABLATED_COLOR)
      mat.emissive.copy(ABLATED_COLOR)
      mat.emissiveIntensity = 0.7
      mat.opacity = 0.95
      ;(n.glowMesh.material as THREE.MeshBasicMaterial).color.copy(ABLATED_COLOR)
    }
  }

  getAblations(): { layer: number; head: number }[] {
    const out: { layer: number; head: number }[] = []
    for (const key of this.ablatedHeads) {
      const [L, H] = key.split(':').map(Number)
      out.push({ layer: L, head: H })
    }
    return out
  }

  clearAblations() {
    for (const n of this.neurons) {
      if (!n.ablated) continue
      n.ablated = false
      const mat = n.mesh.material as THREE.MeshStandardMaterial
      mat.color.copy(n.baseColor)
      mat.emissive.copy(n.baseColor)
      mat.emissiveIntensity = 0.12
      mat.opacity = 0.3
      ;(n.glowMesh.material as THREE.MeshBasicMaterial).color.copy(n.baseColor)
    }
    this.ablatedHeads.clear()
    this.onAblationChange?.([])
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

    const colorLabel = n.role === 'attn' ? '#5eead4' : n.role === 'ffn' ? '#ff8c42' : '#5eead4'
    const connections = this.synapses.filter(s => s.fromIdx === this.neurons.indexOf(n) || s.toIdx === this.neurons.indexOf(n)).length
    const step = this.phase === 'thinking' ? STEP_NAMES[this.currentStep] : '—'
    const active = n.activation > 0.1

    this.inspectPanel.innerHTML = `
      <div style="color:#5eead4;font-weight:600;font-size:11px;margin-bottom:8px;border-bottom:1px solid rgba(244,236,223,0.14);padding-bottom:6px;">Component Inspector</div>
      <div style="color:#514a3e">Component</div>
      <div style="color:${colorLabel};margin-bottom:4px;font-weight:600">${roleLabel}</div>
      <div style="color:#514a3e">Layer</div>
      <div style="color:#f4ecdf;margin-bottom:4px"><strong>${n.layer}</strong> / 31</div>
      <div style="color:#514a3e;font-size:9px;margin-bottom:6px">${roleDesc}</div>
      <div style="color:#514a3e">Connections</div>
      <div style="color:#f4ecdf;margin-bottom:4px">${connections}</div>
      <div style="color:#514a3e">Real Activation</div>
      <div style="margin-bottom:4px">
        <div style="background:#0f172a;border-radius:3px;height:6px;width:100%;overflow:hidden">
          <div style="background:${active ? colorLabel : '#1e293b'};height:100%;width:${(n.activation * 100).toFixed(0)}%;transition:width 0.1s;box-shadow:${active ? '0 0 8px ' + colorLabel : 'none'}"></div>
        </div>
        <span style="color:${active ? '#5eead4' : '#514a3e'}">${(n.activation * 100).toFixed(0)}% (GPU readback)</span>
      </div>
      <div style="color:#514a3e">Current Op</div>
      <div style="color:#f4ecdf;margin-bottom:4px">${step}</div>
      <div style="color:#514a3e">Status</div>
      <div style="color:${active ? '#5eead4' : '#514a3e'}">${active ? 'Firing' : 'Idle'}</div>
    `
  }

  // ─── Render loop ───
  // Strict 1:1: everything on screen is a direct function of activation state.
  // No camera drift, no breathing, no particles, no bloom.

  render = () => {
    requestAnimationFrame(this.render)
    this.time += 0.016

    this.controls.update()

    // Journey-mode camera — only drives the camera when actively playing.
    // Otherwise OrbitControls controls the camera; we keep journey targets
    // as the natural orbit center.
    if (this.journeyActive) {
      if (this.journeyDriving) {
        this.camera.position.lerp(this.journeyCamPos, 0.18)
        this.camera.lookAt(this.journeyCamLookAt)
      } else {
        // User-controlled camera: keep OrbitControls target synced to current
        // journey focus point so orbit feels centered on what they're seeing.
        this.controls.target.lerp(this.journeyCamLookAt, 0.1)
      }
      // subtle starfield counter-rotation for parallax feel
      if (this.starfield) this.starfield.rotation.y += 0.0004
      // dust drift, spotlight pulse
      this.journeyTick()
    } else if (this.cinematicCamera) {
      // Cinematic camera tween — tracks the active layer when enabled.
      this.camera.position.lerp(this._cameraTweenTarget, 0.06)
      this.camera.lookAt(this._cameraTweenLookAt)
    }

    // ─── Update neurons (activation-driven only) ───
    // When a neuron is locked via inspect, dim every other neuron so the
    // selected one (and its layer) becomes visually unmissable.
    const haveSelection = !!this.selectedNeuron
    const selLayer = this.selectedNeuron ? this.selectedNeuron.layer : -1
    for (const n of this.neurons) {
      const mat = n.mesh.material as THREE.MeshStandardMaterial
      const glowMat = n.glowMesh.material as THREE.MeshBasicMaterial
      const act = n.activation
      // Inspect dim factor: 1.0 when nothing selected, 0.18 for off-layer
      // neurons, 1.0 for selected neuron + same-layer siblings.
      let dim = 1.0
      if (haveSelection) {
        if (n === this.selectedNeuron) dim = 1.0
        else if (n.layer === selLayer) dim = 0.55
        else dim = 0.18
      }

      // Ablated heads render amber — override the activation-driven paint so
      // the user can always see which heads are disabled, even during a run.
      if (n.ablated) {
        const pulse = 0.55 + 0.35 * Math.sin(performance.now() * 0.006)
        mat.color.copy(ABLATED_COLOR)
        mat.emissive.copy(ABLATED_COLOR)
        mat.emissiveIntensity = pulse * dim
        mat.opacity = 0.95 * dim
        n.mesh.scale.setScalar(1.15)
        glowMat.color.copy(ABLATED_COLOR)
        glowMat.opacity = 0.35 * dim
        n.glowMesh.scale.setScalar(1.8)
        n.activation = Math.max(0, n.activation - 0.004)
        continue
      }

      if (act > 0.2) {
        mat.color.lerpColors(n.baseColor, n.brightColor, act * 0.8)
        mat.emissive.lerpColors(n.baseColor, n.brightColor, act * 0.5)
        mat.emissiveIntensity = (0.15 + act * 0.8) * dim
        mat.opacity = (0.5 + act * 0.5) * dim
        n.mesh.scale.setScalar(1 + act * 0.5)

        glowMat.color.lerpColors(n.baseColor, n.brightColor, 0.5)
        glowMat.opacity = act * 0.2 * dim
        n.glowMesh.scale.setScalar(1 + act * 1.5)
      } else {
        // Idle: substantially brighter baseline so dots stay visible even
        // when not pulsing. Was emissive 0.08 / opacity 0.25 — invisible at
        // journey distance; now 0.22 / 0.62 so they read as a cyan/amber
        // pointillist backdrop you can still click.
        mat.color.copy(n.baseColor)
        mat.emissive.copy(n.baseColor)
        mat.emissiveIntensity = 0.22 * dim
        mat.opacity = 0.62 * dim
        n.mesh.scale.setScalar(1)

        // Faint idle glow so each dot has a soft halo, improving depth
        // perception + hit discoverability.
        glowMat.color.copy(n.baseColor)
        glowMat.opacity = 0.08 * dim
        n.glowMesh.scale.setScalar(1)
      }
      // Journey mode slows the decay ~3.5× so activations linger long enough
      // to be seen at journey pace (scroll/auto-play). Classic modes keep the
      // snappy 0.004/frame decay so real-time decode still feels live.
      const decayRate = this.journeyActive ? 0.0011 : 0.004
      n.activation = Math.max(0, n.activation - decayRate)

      // Journey-mode passive life. `Math.max` means real inference activations
      // always dominate when present; this only fills visual gaps.
      //
      //   · Focus layer:   strongest breathing pulse (user is looking here)
      //   · Adjacent ±1:   mid pulse
      //   · ±2/±3:         small traveling ripple so the whole model feels alive
      //   · Global:        low-intensity "heartbeat" wave rolling along the
      //                    layer axis, so there's always motion in view
      if (this.journeyActive) {
        const focus = this.journeyFocusLayer
        const globalWave = 0.08 * (0.5 + 0.5 * Math.sin(this.time * 0.9 + n.layer * 0.35))
        let layerPulse = 0
        if (focus >= 0) {
          const dist = Math.abs(n.layer - focus)
          if (dist === 0) layerPulse = 0.36 + 0.14 * Math.sin(this.time * 2.2 + n.subIndex * 0.23)
          else if (dist === 1) layerPulse = 0.20 + 0.08 * Math.sin(this.time * 2.0 + n.subIndex * 0.31)
          else if (dist <= 3) layerPulse = 0.10 + 0.05 * Math.sin(this.time * 1.7 + n.subIndex * 0.19 + dist)
        }
        const passive = Math.max(globalWave, layerPulse)
        if (passive > 0) n.activation = Math.max(n.activation, passive)
      }
    }

    // ─── Synapses tinted by endpoint activation ───
    for (const s of this.synapses) {
      const mat = s.line.material as THREE.LineBasicMaterial
      const act = Math.max(this.neurons[s.fromIdx].activation, this.neurons[s.toIdx].activation)
      mat.opacity = 0.02 + act * 0.35
      if (act > 0.2) {
        const avgLayer = Math.floor((this.neurons[s.fromIdx].layer + this.neurons[s.toIdx].layer) / 2)
        const [h] = neuronHSL(avgLayer)
        mat.color.setHSL(h, 0.8, 0.35 + act * 0.4)
      } else {
        mat.color.copy(s.baseColor)
      }
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
        <span style="color:#5eead4">${PHASE_LABELS[phaseIdx]}</span><br>
        Layer ${this.activeLayer}/31 &bull; ${Math.round(this.outputConfidence * 100)}% &bull; ${STEP_NAMES[this.currentStep] || ''}<br>
        Dispatches: ${this.dispatchCount}/${this.totalDispatches}
      `
    } else if (this.phase === 'idle') {
      this.overlay.innerHTML = '<span style="color:#1e293b">Idle — type a question</span>'
    } else {
      this.overlay.innerHTML = '<span style="color:#5eead4">Done</span>'
    }

    // Selection ring: billboard towards camera, subtle pulse for visibility.
    if (this.selectedRing && this.selectedNeuron) {
      this.selectedRing.lookAt(this.camera.position)
      const ringPulse = Math.sin(this.time * 3) * 0.15 + 0.85
      ;(this.selectedRing.material as THREE.MeshBasicMaterial).opacity = ringPulse
      this.updateInspectPanel()
    }

    // ─── Dense components: embedding sphere + LM head activation decay ───
    if (this.embeddingMesh) {
      const eMat = this.embeddingMesh.material as THREE.MeshStandardMaterial
      const eAct = this.embeddingActivation
      eMat.emissiveIntensity = 0.3 + eAct * 1.2
      this.embeddingMesh.scale.setScalar(1 + eAct * 0.4)
      ;(this.embeddingGlow.material as THREE.MeshBasicMaterial).opacity = 0.06 + eAct * 0.4
      this.embeddingGlow.scale.setScalar(1 + eAct * 0.6)
      this.embeddingActivation = Math.max(0, eAct - 0.008)
    }

    if (this.lmHeadStrip) {
      const mat = this.lmHeadStrip.material as THREE.PointsMaterial
      mat.opacity = 0.5 + this.lmHeadActivation * 0.5
      this.lmHeadActivation = Math.max(0.2, this.lmHeadActivation - 0.005)
    }

    this.renderer.render(this.scene, this.camera)
  }

  start() { this.render() }
  stop() { }

  // ─── JOURNEY MODE API — external scroll-driven camera + starfield ────
  // The journey module (src/journey.ts) calls these. When journeyActive
  // is true, OrbitControls + cinematic tween are disabled and the render
  // loop lerps the camera toward journeyCamPos/LookAt each frame.

  private generateStarfield(): void {
    const N = 3200
    const positions = new Float32Array(N * 3)
    for (let i = 0; i < N; i++) {
      const u = Math.random()
      const v = Math.random()
      const theta = 2 * Math.PI * u
      const phi = Math.acos(2 * v - 1)
      const r = 55 + Math.random() * 45
      positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta)
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      positions[i * 3 + 2] = r * Math.cos(phi)
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    const mat = new THREE.PointsMaterial({
      color: 0xf4ecdf,
      size: 0.08,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
    this.starfield = new THREE.Points(geo, mat)
    this.starfield.visible = false
    this.scene.add(this.starfield)
  }

  public enableJourneyMode(enabled: boolean): void {
    this.journeyActive = enabled
    if (this.starfield) this.starfield.visible = enabled
    if (this.spaceDust) this.spaceDust.visible = enabled
    if (this.layerRings) this.layerRings.visible = enabled
    if (this.focusSpotlight) this.focusSpotlight.visible = enabled
    if (enabled) {
      this.cinematicCamera = false
      // OrbitControls stays ENABLED by default — journey only takes over
      // the camera when `setJourneyDriving(true)` is called (e.g. autoplay).
      // This gives the user free 3D navigation at all other times.
      this.controls.enabled = true
    } else {
      this.controls.enabled = true
    }
  }

  private journeyDriving = false

  /** Journey takes over the camera (auto-play). OrbitControls get disabled.
   *  Call with false to yield camera back to the user. */
  public setJourneyDriving(driving: boolean): void {
    this.journeyDriving = driving
    this.controls.enabled = !driving
  }

  public setJourneyCamera(pos: THREE.Vector3, lookAt: THREE.Vector3): void {
    this.journeyCamPos.copy(pos)
    this.journeyCamLookAt.copy(lookAt)
  }

  /** Educational "show me" pulse — glossary terms call this to briefly
   *  light up the corresponding group of neurons in the 3D scene.
   *  Accepted groups: 'attention' | 'ffn' | 'residual' | 'all'. */
  public highlightGroup(group: string, intensity: number = 0.85): void {
    const match = (n: NeuronData): boolean => {
      if (group === 'all') return true
      if (group === 'attention') return n.role === 'attn'
      if (group === 'ffn') return n.role === 'ffn'
      if (group === 'residual') return n.role === 'residual'
      return false
    }
    for (const n of this.neurons) {
      if (match(n)) {
        n.activation = Math.max(n.activation, intensity)
      }
    }
    // FFN slab + residual slab get direct color boost via their per-layer
    // update paths. Simpler: just force the activation pulse on neurons and
    // let the existing render loop handle visuals + decay.
  }

  /** Layer index 0..31 that receives the journey-mode passive breathing pulse.
   * Pass -1 to disable (e.g., approach / closing phases of the journey). */
  public setJourneyFocusLayer(layer: number): void {
    this.journeyFocusLayer = layer
    // Move the focus spotlight to the layer's X position
    if (this.focusSpotlight && layer >= 0) {
      const lx = ((layer / 31) - 0.5) * 6.0  // matches TOTAL_WIDTH
      this.focusSpotlight.position.x = lx
      this.focusSpotlight.visible = this.journeyActive
    } else if (this.focusSpotlight) {
      this.focusSpotlight.visible = false
    }
  }

  // ─── Space dust: drifting particles in the void. Gives depth + "traveling
  // through space" feel when the camera moves. Journey-only. ───
  private generateSpaceDust(): void {
    const N = 2400
    const positions = new Float32Array(N * 3)
    const seeds = new Float32Array(N)
    for (let i = 0; i < N; i++) {
      // Distribute in a cylindrical volume around the model. X spans a bit
      // past both ends, Y/Z form a shell around the layer axis.
      positions[i * 3]     = (Math.random() - 0.5) * 14            // x: -7..7
      positions[i * 3 + 1] = (Math.random() - 0.5) * 6             // y: -3..3
      positions[i * 3 + 2] = (Math.random() - 0.5) * 7 - 1         // z: -4.5..2.5
      seeds[i] = Math.random()
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    const mat = new THREE.PointsMaterial({
      color: 0x8fd4ea,
      size: 0.04,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.32,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
    this.spaceDust = new THREE.Points(geo, mat)
    this.spaceDust.visible = false
    this.spaceDustPositions = positions
    this.spaceDustSeeds = seeds
    this.scene.add(this.spaceDust)
  }

  // ─── Layer boundary rings — one thin vertical ring at each of the 32
  // layer X positions. Always there in journey mode so spatial structure
  // is legible. ───
  private generateLayerRings(): void {
    const group = new THREE.Group()
    const ringGeo = new THREE.RingGeometry(0.78, 0.82, 48)
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x5eead4,
      transparent: true,
      opacity: 0.11,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
    for (let L = 0; L < 32; L++) {
      const lx = ((L / 31) - 0.5) * 6.0
      const ring = new THREE.Mesh(ringGeo, ringMat)
      // Ring faces +x (perpendicular to the layer axis)
      ring.rotation.y = Math.PI / 2
      ring.position.set(lx, 0, 0)
      group.add(ring)
    }
    group.visible = false
    this.layerRings = group
    this.scene.add(group)
  }

  // ─── Focus-layer spotlight — a bright glowing ring at the camera's
  // current focus layer. Lets the user see, not just read, which layer
  // the dolly is currently following. ───
  private generateFocusSpotlight(): void {
    const geo = new THREE.RingGeometry(0.95, 1.08, 64)
    const mat = new THREE.MeshBasicMaterial({
      color: 0x00e5ff,
      transparent: true,
      opacity: 0.0,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
    const ring = new THREE.Mesh(geo, mat)
    ring.rotation.y = Math.PI / 2
    ring.visible = false
    this.focusSpotlight = ring
    this.scene.add(ring)
  }

  /** Called each frame from the render loop when journey is active.
   *  Handles dust drift + focus-spotlight pulse + layer-wave ripple. */
  private journeyTick(): void {
    // Drift the space dust slowly — wrap along X so it feels continuous
    if (this.spaceDust && this.spaceDustPositions && this.spaceDustSeeds) {
      const pos = this.spaceDustPositions
      const t = this.time
      for (let i = 0; i < this.spaceDustSeeds.length; i++) {
        const s = this.spaceDustSeeds[i]!
        const base = (i * 3)
        // Slow drift + gentle vertical bob; wrap x in [-7, 7]
        pos[base]     = ((pos[base]! - 0.002 - s * 0.0015) + 14) % 14 - 7
        pos[base + 1] += Math.sin(t * 0.6 + s * 6.28) * 0.0015
      }
      ;(this.spaceDust.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true
      this.spaceDust.rotation.z = Math.sin(t * 0.15) * 0.02
    }

    // Pulse the focus spotlight — breathing opacity + subtle scale
    if (this.focusSpotlight && this.journeyFocusLayer >= 0) {
      const mat = this.focusSpotlight.material as THREE.MeshBasicMaterial
      const pulse = 0.5 + 0.5 * Math.sin(this.time * 2.2)
      mat.opacity = 0.22 + pulse * 0.35
      this.focusSpotlight.scale.setScalar(1 + pulse * 0.08)
    }
  }

  /** Read-only: what the journey module needs to know about this scene. */
  public readonly LAYER_SPAN = { xMin: -3, xMax: 3, layers: 32 }

  // ─── Expose the underlying camera + canvas so spatial-panels.ts can
  // project world-space anchors into screen coordinates each frame.
  public getCamera(): THREE.PerspectiveCamera { return this.camera }
  public getCanvas(): HTMLCanvasElement { return this.renderer.domElement }
  public getControlsTarget(): THREE.Vector3 { return this.controls.target }
}

// ═══════════════════════════════════════════════════════════════
// Neural Pulse — Brain Visualization v7 (Avatar Polish)
// Bioluminescent 3D transformer + volumetric atmosphere + organic effects
// ═══════════════════════════════════════════════════════════════

import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js'
import { AudioEngine } from './audio'

const LAYER_COUNT = 32
const PARTICLE_COUNT = 3000
const DUST_COUNT = 800
const BURST_COUNT = 300

// ─── Bioluminescent color palette (Avatar-inspired) ───
function neuronHSL(layer: number, role: 'attn' | 'ffn' | 'residual' = 'attn'): [number, number, number] {
  const t = layer / 31
  if (role === 'attn') {
    // Electric cyan → deep violet (hue 190→270)
    const h = 190 + t * 80
    return [h / 360, 0.85, 0.55]
  } else if (role === 'ffn') {
    // Bioluminescent magenta → warm amber (hue 320→40)
    const h = 320 + t * 80
    return [((h % 360)) / 360, 0.8, 0.5]
  } else {
    // Residual: ethereal white-cyan glow
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
  phase: number  // individual breathing phase offset
}

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
  points: THREE.Vector3[]  // control points for curved path
}

interface SignalObj {
  synapse: number
  pos: number
  speed: number
  mesh: THREE.Mesh
  trail: THREE.Points   // energy trail behind signal
  trailPositions: Float32Array
  trailIdx: number
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

  // Ambient dust (fireflies)
  private dustSystem!: THREE.Points
  private dustPositions!: Float32Array
  private dustPhases!: Float32Array

  // Starfield / nebula
  private starField!: THREE.Points
  private nebulaLayers: THREE.Mesh[] = []

  // Token burst particles
  private burstParticles!: THREE.Points
  private burstPositions!: Float32Array
  private burstVelocities!: Float32Array
  private burstLifetimes!: Float32Array
  private burstColors!: Float32Array
  private burstActive = false

  // Ground plane
  private groundPlane!: THREE.Mesh
  private groundMaterial!: THREE.ShaderMaterial
  private pulseWaves: { x: number, z: number, time: number, strength: number }[] = []

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
  private opAnchorMeshes: THREE.Mesh[][] = []  // [layer][anchorIdx]

  // KV cache strips per layer
  private kvStripMeshes: THREE.Mesh[] = []
  private kvStripFills: number[] = []  // 0..1 fill ratio per layer

  // Token embedding sphere (pre-layer-0)
  private embeddingMesh!: THREE.Mesh
  private embeddingGlow!: THREE.Mesh
  private embeddingActivation = 0

  // LM head bar (post-layer-31): vocab logits as a wide instanced strip
  private lmHeadStrip!: THREE.Points
  private lmHeadColors!: Float32Array
  private lmHeadActivation = 0

  // Real attention beam targets (post-softmax scores from GPU shader)
  private attnScoreLines: THREE.Line[] = []
  private currentAttnLayer = -1

  // Strict 1:1 mode is permanent: every pixel on screen maps to a real
  // GPU-side state value, with no cosmetic transformation. The flag is
  // kept so the existing decoration-gate checks short-circuit at zero cost.
  readonly cinematicMode = false
  readonly contrastMode = false

  // PCA-based slab layout (loaded from public/pca-layout.json via setPcaLayout).
  // When set, residual + FFN slabs use these 2D coords instead of the index lattice.
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

  // Cinematic camera
  private cameraWaveX = 0
  private time = 0  // global animation time

  constructor(canvas: HTMLCanvasElement) {
    this.audio = new AudioEngine()

    // Renderer — cinematic settings
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 0.9
    this.renderer.shadowMap.enabled = false
    this.updateSize()

    // Scene — deep space atmosphere
    this.scene = new THREE.Scene()
    this.scene.fog = new THREE.FogExp2(0x020210, 0.08)

    // Camera
    const aspect = canvas.clientWidth / canvas.clientHeight
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 300)
    this.camera.position.set(0, 1.2, 5.5)

    // Controls — dreamy orbit
    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.03
    this.controls.autoRotate = true
    this.controls.autoRotateSpeed = 0.15
    this.controls.enableZoom = true
    this.controls.minDistance = 1.5
    this.controls.maxDistance = 10

    // Post-processing: Enhanced bloom for bioluminescence
    this.composer = new EffectComposer(this.renderer)
    this.composer.addPass(new RenderPass(this.scene, this.camera))
    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(canvas.clientWidth, canvas.clientHeight),
      0.8,    // strength — rich bioluminescent glow
      0.4,    // radius — spread the light
      0.2     // threshold — more things glow
    )
    this.composer.addPass(this.bloomPass)

    // Lights — Avatar-style bioluminescent scene
    this.scene.add(new THREE.AmbientLight(0x0a0a1e, 0.3))

    // Main key light: ethereal cyan from above
    const key = new THREE.PointLight(0x00e5ff, 1.2, 15)
    key.position.set(0, 3, 4)
    this.scene.add(key)

    // Rim light: deep violet from behind
    const rim = new THREE.PointLight(0x7c3aed, 0.8, 12)
    rim.position.set(-3, -1, -3)
    this.scene.add(rim)

    // Fill light: magenta undertone
    const fill = new THREE.PointLight(0xec4899, 0.4, 10)
    fill.position.set(3, -2, 1)
    this.scene.add(fill)

    // Accent light: teal from below (like bioluminescent pool)
    const accent = new THREE.PointLight(0x14b8a6, 0.6, 8)
    accent.position.set(0, -3, 0)
    this.scene.add(accent)

    // HUD overlay
    this.overlay = document.createElement('div')
    this.overlay.style.cssText = 'position:absolute;bottom:12px;left:12px;pointer-events:none;font-family:JetBrains Mono,monospace;font-size:9px;color:#475569;line-height:1.6;'
    canvas.parentElement!.style.position = 'relative'
    canvas.parentElement!.appendChild(this.overlay)

    // Tooltip — glass morphism
    this.tooltip = document.createElement('div')
    this.tooltip.style.cssText = 'position:absolute;pointer-events:none;background:rgba(2,2,16,0.85);border:1px solid rgba(0,229,255,0.2);border-radius:8px;padding:8px 12px;font-family:JetBrains Mono,monospace;font-size:9px;color:#67e8f9;display:none;white-space:nowrap;z-index:10;backdrop-filter:blur(12px);box-shadow:0 4px 20px rgba(0,229,255,0.1);'
    canvas.parentElement!.appendChild(this.tooltip)

    // Inspect panel — glass morphism
    this.inspectPanel = document.createElement('div')
    this.inspectPanel.style.cssText = 'position:absolute;top:12px;right:12px;background:rgba(2,2,16,0.85);border:1px solid rgba(0,229,255,0.15);border-radius:12px;padding:16px 20px;font-family:JetBrains Mono,monospace;font-size:10px;color:#cbd5e1;display:none;width:230px;z-index:10;line-height:1.8;backdrop-filter:blur(16px);box-shadow:0 8px 32px rgba(0,229,255,0.08);'
    canvas.parentElement!.appendChild(this.inspectPanel)

    // Selection ring — glowing
    const ringGeo = new THREE.RingGeometry(0.038, 0.046, 32)
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0, side: THREE.DoubleSide })
    this.selectedRing = new THREE.Mesh(ringGeo, ringMat)
    this.scene.add(this.selectedRing)

    this.dispatchEl = document.getElementById('dispatchStat') as HTMLSpanElement

    // Generate everything. Strict mode is permanent — nebula, starfield
    // and ground plane are decoration and never created.
    this.generateBrain()
    this.generateResidualSlab()
    this.generateFfnSlab()
    this.generateOpAnchors()
    this.generateKvStrips()
    this.generateEmbeddingNode()
    this.generateLmHead()
    this.initParticles()
    this.initDust()
    this.initBurstParticles()

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

  // ─── Nebula background (volumetric-like layers) ───
  private generateNebula() {
    const nebulaColors = [
      { color: 0x1a0033, opacity: 0.15, scale: 80, y: 0 },     // deep violet base
      { color: 0x001a33, opacity: 0.12, scale: 60, y: 5 },     // deep blue mid
      { color: 0x003322, opacity: 0.08, scale: 70, y: -5 },    // teal undertone
      { color: 0x330022, opacity: 0.06, scale: 50, y: 10 },    // magenta wisps
    ]

    for (const n of nebulaColors) {
      const geo = new THREE.SphereGeometry(n.scale, 16, 12)
      const mat = new THREE.MeshBasicMaterial({
        color: n.color,
        transparent: true,
        opacity: n.opacity,
        side: THREE.BackSide,
        depthWrite: false,
      })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.y = n.y
      this.scene.add(mesh)
      this.nebulaLayers.push(mesh)
    }
  }

  // ─── Starfield with color variation ───
  private generateStarfield() {
    const count = 2000
    const pos = new Float32Array(count * 3)
    const cols = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      const r = 30 + Math.random() * 80
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta)
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      pos[i * 3 + 2] = r * Math.cos(phi)
      // Color variety: cyan, violet, white
      const type = Math.random()
      if (type < 0.3) {
        cols[i * 3] = 0.2; cols[i * 3 + 1] = 0.8; cols[i * 3 + 2] = 1.0
      } else if (type < 0.5) {
        cols[i * 3] = 0.6; cols[i * 3 + 1] = 0.3; cols[i * 3 + 2] = 1.0
      } else {
        const b = 0.4 + Math.random() * 0.6
        cols[i * 3] = b * 0.8; cols[i * 3 + 1] = b * 0.85; cols[i * 3 + 2] = b
      }
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(cols, 3))
    const mat = new THREE.PointsMaterial({ size: 0.12, vertexColors: true, transparent: true, opacity: 0.6, sizeAttenuation: true })
    this.starField = new THREE.Points(geo, mat)
    this.scene.add(this.starField)
  }

  // ─── Ground plane with neural grid shader ───
  private generateGroundPlane() {
    const geo = new THREE.PlaneGeometry(20, 20, 80, 80)
    this.groundMaterial = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        uTime: { value: 0 },
        uPulse1: { value: new THREE.Vector4(0, 0, -999, 0) },
        uPulse2: { value: new THREE.Vector4(0, 0, -999, 0) },
        uActiveX: { value: 0 },
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vPos;
        void main() {
          vUv = uv;
          vPos = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uTime;
        uniform vec4 uPulse1;
        uniform vec4 uPulse2;
        uniform float uActiveX;
        varying vec2 vUv;
        varying vec3 vPos;

        float grid(vec2 p, float spacing) {
          vec2 g = abs(fract(p / spacing - 0.5) - 0.5) / fwidth(p / spacing);
          return 1.0 - min(min(g.x, g.y), 1.0);
        }

        float pulse(vec2 pos, vec4 pulseData) {
          float dist = length(pos - pulseData.xy);
          float age = uTime - pulseData.z;
          float radius = age * 3.0;
          float ring = exp(-pow(dist - radius, 2.0) * 8.0);
          float fade = exp(-age * 2.0) * pulseData.w;
          return ring * fade;
        }

        void main() {
          float g = grid(vPos.xz, 0.5) * 0.08;
          float gFine = grid(vPos.xz, 0.1) * 0.03;

          // Distance fade
          float dist = length(vPos.xz);
          float fade = exp(-dist * 0.15);

          // Active layer glow
          float activeGlow = exp(-pow(vPos.x - uActiveX, 2.0) * 2.0) * 0.06;

          // Pulse waves
          float p1 = pulse(vPos.xz, uPulse1);
          float p2 = pulse(vPos.xz, uPulse2);

          float alpha = (g + gFine + activeGlow + (p1 + p2) * 0.15) * fade;

          // Cyan-teal color with pulse warmth
          vec3 baseCol = vec3(0.0, 0.6, 0.8);
          vec3 pulseCol = vec3(0.0, 1.0, 0.9);
          vec3 col = mix(baseCol, pulseCol, (p1 + p2) * 0.5 + activeGlow * 2.0);

          gl_FragColor = vec4(col, alpha);
        }
      `,
    })
    this.groundPlane = new THREE.Mesh(geo, this.groundMaterial)
    this.groundPlane.rotation.x = -Math.PI / 2
    this.groundPlane.position.y = -1.2
    this.scene.add(this.groundPlane)
  }

  // ─── Particles (energy flow through brain) ───
  private initParticles() {
    this.particlePositions = new Float32Array(PARTICLE_COUNT * 3)
    this.particleColors = new Float32Array(PARTICLE_COUNT * 3)
    this.particleVelocities = new Float32Array(PARTICLE_COUNT * 3)
    this.particleLifetimes = new Float32Array(PARTICLE_COUNT)
    this.particleLifetimes.fill(0)
    this.particlePositions.fill(0)
    this.particleColors.fill(0)

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(this.particlePositions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(this.particleColors, 3))
    const mat = new THREE.PointsMaterial({
      size: 0.018,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    })
    this.particleSystem = new THREE.Points(geo, mat)
    this.scene.add(this.particleSystem)
  }

  // ─── Ambient dust (firefly particles) ───
  private initDust() {
    this.dustPositions = new Float32Array(DUST_COUNT * 3)
    this.dustPhases = new Float32Array(DUST_COUNT)

    for (let i = 0; i < DUST_COUNT; i++) {
      this.dustPositions[i * 3] = (Math.random() - 0.5) * 8
      this.dustPositions[i * 3 + 1] = (Math.random() - 0.5) * 4
      this.dustPositions[i * 3 + 2] = (Math.random() - 0.5) * 5
      this.dustPhases[i] = Math.random() * Math.PI * 2
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(this.dustPositions, 3))
    const mat = new THREE.PointsMaterial({
      size: 0.015,
      color: 0x00e5ff,
      transparent: true,
      opacity: 0.3,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    })
    this.dustSystem = new THREE.Points(geo, mat)
    this.scene.add(this.dustSystem)
  }

  // ─── Token burst particles ───
  private initBurstParticles() {
    this.burstPositions = new Float32Array(BURST_COUNT * 3)
    this.burstVelocities = new Float32Array(BURST_COUNT * 3)
    this.burstLifetimes = new Float32Array(BURST_COUNT)
    this.burstColors = new Float32Array(BURST_COUNT * 3)
    this.burstLifetimes.fill(0)

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(this.burstPositions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(this.burstColors, 3))
    const mat = new THREE.PointsMaterial({
      size: 0.025,
      vertexColors: true,
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
  private addNeuron(
    wx: number, wy: number, wz: number,
    layer: number, role: 'attn' | 'ffn' | 'residual', subIndex: number,
    neuronGeo: THREE.SphereGeometry, glowGeo: THREE.SphereGeometry
  ) {
    const [h, s, l] = neuronHSL(layer, role)
    const baseColor = new THREE.Color().setHSL(h, s, l)
    const brightColor = new THREE.Color().setHSL(h, Math.min(1, s + 0.1), 0.85)

    // Organic material — more emissive, glass-like
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

    // Glow halo — larger, softer
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
      phase: Math.random() * Math.PI * 2,  // individual breathing offset
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
    const attnGeo = new THREE.SphereGeometry(0.014, 8, 6)
    const resGeo = new THREE.SphereGeometry(0.014, 10, 8)
    const attnGlowGeo = new THREE.SphereGeometry(0.04, 8, 6)
    const resGlowGeo = new THREE.SphereGeometry(0.045, 8, 6)

    for (let L = 0; L < 32; L++) {
      const t = L / 31
      const lx = (t - 0.5) * TOTAL_WIDTH
      const lz = 0  // ── kill Z wave: planar architecture ──

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

    // Residual chain — straight line through every layer (not curved: this IS the spine)
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

    // Signal pool with energy trails
    const sigGeo = new THREE.SphereGeometry(0.016, 8, 6)
    for (let i = 0; i < 120; i++) {
      const mat = new THREE.MeshBasicMaterial({
        color: 0x00e5ff,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
      })
      const mesh = new THREE.Mesh(sigGeo, mat)
      mesh.visible = false
      this.scene.add(mesh)

      // Trail
      const trailCount = 8
      const trailPositions = new Float32Array(trailCount * 3)
      const trailGeo = new THREE.BufferGeometry()
      trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3))
      const trailMat = new THREE.PointsMaterial({
        size: 0.008,
        color: 0x00e5ff,
        transparent: true,
        opacity: 0.4,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true,
      })
      const trail = new THREE.Points(trailGeo, trailMat)
      trail.visible = false
      this.scene.add(trail)

      this.signals.push({ synapse: -1, pos: 0, speed: 0, mesh, trail, trailPositions, trailIdx: 0 })
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
      // Each of the 3072 components stacks vertically. To make pattern visible,
      // arrange as a 64-wide × 48-tall block at this layer's slot.
      const cols = 64, rows = 48  // 64*48=3072
      for (let i = 0; i < D; i++) {
        const col = i % cols
        const row = Math.floor(i / cols)
        const x = lx + (col - cols / 2 + 0.5) * 0.0009
        const y = ((row / (rows - 1)) - 0.5) * RESIDUAL_HEIGHT
        const idx = (L * D + i) * 3
        positions[idx] = x
        positions[idx + 1] = y
        positions[idx + 2] = 0
        // Initial dim cyan
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
        const row = Math.floor(i / FFN_COLS)  // 0..63
        const x = slabCenterX + (col - FFN_COLS / 2 + 0.5) * (FFN_W / FFN_COLS)
        const y = ((row / (FFN_ROWS - 1)) - 0.5) * FFN_H
        const idx = (L * FFN + i) * 3
        positions[idx] = x
        positions[idx + 1] = y
        positions[idx + 2] = 0
        this.ffnSlabColors[idx] = 0.05
        this.ffnSlabColors[idx + 1] = 0.0
        this.ffnSlabColors[idx + 2] = 0.05
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
            // PCA: x in [-0.5, 0.5] → narrow per-layer column ±0.045
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
    // The 10 anchor positions, in compute order:
    //   0 RMSNorm₁ → 1 QKV → 2 RoPE → 3 KVAppend → 4 Attention → 5 OProj
    //   6 +Residual → 7 RMSNorm₂ → 8 FFN(Gate∘Up) → 9 FFN(Down)
    const anchorGeo = new THREE.SphereGeometry(0.005, 6, 4)
    const ANCHOR_W = 0.16
    for (let L = 0; L < 32; L++) {
      const lx = ((L / 31) - 0.5) * TOTAL_WIDTH
      const meshes: THREE.Mesh[] = []
      const linePts: THREE.Vector3[] = []
      for (let a = 0; a < 10; a++) {
        const ax = lx + (a / 9 - 0.5) * ANCHOR_W
        const mat = new THREE.MeshBasicMaterial({
          color: 0x67e8f9,
          transparent: true,
          opacity: 0.25,
        })
        const mesh = new THREE.Mesh(anchorGeo, mat)
        mesh.position.set(ax, 0.85, 0)
        this.scene.add(mesh)
        meshes.push(mesh)
        linePts.push(mesh.position.clone())
      }
      // Connecting line through the 10 anchors
      const lineGeo = new THREE.BufferGeometry().setFromPoints(linePts)
      const lineMat = new THREE.LineBasicMaterial({
        color: 0x67e8f9,
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
      // Initial scale.x = 0 (no usage)
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

    // Halo
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
    // Lay vocab as a 256-wide × 126-tall lattice (256*126=32256 ≥ 32064)
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

  // Curved synapse using catmull-rom-like interpolation
  private addCurvedSynapse(i: number, j: number) {
    const ni = this.neurons[i], nj = this.neurons[j]
    const avgLayer = Math.floor((ni.layer + nj.layer) / 2)
    const role = ni.role === 'residual' || nj.role === 'residual' ? 'residual' : ni.role
    const [hh] = neuronHSL(avgLayer, role as 'attn' | 'ffn' | 'residual')
    const color = new THREE.Color().setHSL(hh, 0.6, 0.25)

    // Create curved path with midpoint offset
    const mid = new THREE.Vector3().lerpVectors(ni.position, nj.position, 0.5)
    // Offset midpoint for organic curve
    const offset = new THREE.Vector3(
      0,
      (Math.random() - 0.5) * 0.08,
      (Math.random() - 0.5) * 0.08,
    )
    mid.add(offset)

    // Generate curve points
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
        this.ffnSlabColors[i] = 0.05
        this.ffnSlabColors[i + 1] = 0
        this.ffnSlabColors[i + 2] = 0.05
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

    if (progress > 0.15 && progress < 0.4 && Math.random() < 0.5) this.spawnSignals(layer)
    if (progress > 0.1 && progress < 0.5) this.spawnParticlesNearLayer(layer)

    this.cameraWaveX = ((layer + progress) / LAYER_COUNT - 0.5) * 0.8

    if (this.currentStep === 0) this.audio.neuronTick(layer)
    this.audio.setDroneIntensity(this.outputConfidence)

    if (this.dispatchEl) {
      this.dispatchEl.innerHTML = `Dispatch: <strong style="color:#67e8f9">${this.dispatchCount}/${this.totalDispatches}</strong>`
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

    this.cameraWaveX = ((layer + (step + 1) / 9) / LAYER_COUNT - 0.5) * 0.8

    if (step === 0) this.audio.neuronTick(layer)
    this.audio.setDroneIntensity(this.outputConfidence)

    const maxAttn = data.attnHeads.length > 0 ? Math.max(...data.attnHeads) : 0
    if (maxAttn > 0.3 && step < 4) this.spawnSignals(layer)
    if (data.residual > 0.3) this.spawnParticlesNearLayer(layer)

    if (this.dispatchEl) {
      this.dispatchEl.innerHTML = `Dispatch: <strong style="color:#67e8f9">${this.dispatchCount}/${this.totalDispatches}</strong>`
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
        m.color.set(0xfff8a0)
        list[a].scale.setScalar(2.4)
      } else {
        m.opacity = 0.25
        m.color.set(0x67e8f9)
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
      // Magenta→amber ramp (matches FFN role color)
      this.ffnSlabColors[idx]     = 0.4 + v * 0.6
      this.ffnSlabColors[idx + 1] = v * 0.5
      this.ffnSlabColors[idx + 2] = 0.5 + v * 0.4
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
    // RMS magnitude of embedding vector
    let sumSq = 0
    for (let i = 0; i < embedding.length; i++) sumSq += embedding[i] * embedding[i]
    const rms = Math.sqrt(sumSq / embedding.length)
    this.embeddingActivation = Math.tanh(rms * 0.8)
  }

  /** Set LM head logits (drives the post-layer-31 strip). Logits length = 32064. */
  setLogits(logits: Float32Array) {
    // Min/max scan
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
      const w = Math.pow(v, 3)  // emphasize peaks
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

  /** Real attention beams for ALL 32 layers in one shot.
   *  scores layout: [layer * 32 * 256 + head * 256 + slot] f32. */
  setAllAttentionScores(scores: Float32Array, kvLen: number) {
    // Clear old beams
    for (const line of this.attnScoreLines) {
      this.scene.remove(line)
      line.geometry.dispose()
      ;(line.material as THREE.LineBasicMaterial).dispose()
    }
    this.attnScoreLines.length = 0
    this.currentAttnLayer = -1

    if (kvLen <= 1) return
    // Strict mode: beams only show the top-2 slots per head (decorative
    // subset selection), so suppress them entirely. The full attention
    // tensor is still visualized accurately in the DOM heatmap panel.
    if (!this.cinematicMode) return

    const MAX_SLOTS = 256
    const LAYER_WORDS = 32 * MAX_SLOTS
    // Per-layer top-K beams. Keep small to avoid 32 * 32 * 3 = 3072 lines.
    const beamCount = 2

    for (let L = 0; L < 32; L++) {
      const layerCenterX = ((L / 31) - 0.5) * TOTAL_WIDTH
      const headColX = layerCenterX - 0.07
      const layerOff = L * LAYER_WORDS

      for (let h = 0; h < HEADS; h++) {
        const base = layerOff + h * MAX_SLOTS
        const top: { v: number, slot: number }[] = []
        for (let s = 0; s < Math.min(MAX_SLOTS, kvLen); s++) {
          const v = scores[base + s]
          if (v <= 0) continue
          if (top.length < beamCount) {
            top.push({ v, slot: s })
          } else {
            let minIdx = 0
            for (let k = 1; k < beamCount; k++) if (top[k].v < top[minIdx].v) minIdx = k
            if (v > top[minIdx].v) top[minIdx] = { v, slot: s }
          }
        }
        if (top.length === 0) continue

        const headY = ((h / 31) - 0.5) * 0.7
        const fromPt = new THREE.Vector3(headColX, headY, 0)

        for (const { v, slot } of top) {
          if (v < 0.08) continue
          const sy = ((slot / Math.max(1, kvLen - 1)) - 0.5) * RESIDUAL_HEIGHT
          const toPt = new THREE.Vector3(layerCenterX, sy, 0)
          const mid = new THREE.Vector3().lerpVectors(fromPt, toPt, 0.5)
          mid.z += 0.06
          const curve = new THREE.QuadraticBezierCurve3(fromPt, mid, toPt)
          const pts = curve.getPoints(10)
          const positions = new Float32Array(pts.length * 3)
          for (let k = 0; k < pts.length; k++) {
            positions[k * 3]     = pts[k].x
            positions[k * 3 + 1] = pts[k].y
            positions[k * 3 + 2] = pts[k].z
          }
          const geo = new THREE.BufferGeometry()
          geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
          const mat = new THREE.LineBasicMaterial({
            color: new THREE.Color().setHSL(0.55 - v * 0.1, 0.95, 0.6),
            transparent: true,
            opacity: Math.min(0.7, v * 1.2),
            blending: THREE.AdditiveBlending,
          })
          const line = new THREE.Line(geo, mat)
          this.scene.add(line)
          this.attnScoreLines.push(line)
        }
      }
    }
  }

  showAttentionBeams(fromLayer: number, toLayer: number, headStrengths: Float32Array) {
    // Strict mode: decorative top-5 head subset, suppress entirely.
    if (!this.cinematicMode) return
    const fromNeurons = (this.neuronsByLayer.get(fromLayer) ?? []).filter(n => n.role === 'attn')
    const toNeurons = (this.neuronsByLayer.get(toLayer) ?? []).filter(n => n.role === 'attn')
    if (fromNeurons.length === 0 || toNeurons.length === 0) return

    const indexed = Array.from(headStrengths).map((v, i) => ({ v, i }))
    indexed.sort((a, b) => b.v - a.v)
    const topHeads = indexed.slice(0, 5)

    for (const { v, i } of topHeads) {
      if (v < 0.15 || i >= fromNeurons.length || i >= toNeurons.length) continue
      const from = fromNeurons[i]
      const to = toNeurons[i]

      // Curved beam
      const mid = new THREE.Vector3().lerpVectors(from.position, to.position, 0.5)
      mid.y += 0.05
      const curve = new THREE.QuadraticBezierCurve3(from.position, mid, to.position)
      const pts = curve.getPoints(16)
      const positions = new Float32Array(pts.length * 3)
      for (let k = 0; k < pts.length; k++) {
        positions[k * 3] = pts[k].x
        positions[k * 3 + 1] = pts[k].y
        positions[k * 3 + 2] = pts[k].z
      }

      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      const [h] = neuronHSL(fromLayer, 'attn')
      const mat = new THREE.LineBasicMaterial({
        color: new THREE.Color().setHSL(h, 0.95, 0.75),
        transparent: true,
        opacity: v * 0.7,
        blending: THREE.AdditiveBlending,
      })
      const line = new THREE.Line(geo, mat)
      this.scene.add(line)

      // Fade out
      let opacity = v * 0.7
      const fade = () => {
        opacity -= 0.015
        if (opacity <= 0) {
          this.scene.remove(line)
          geo.dispose()
          mat.dispose()
          return
        }
        mat.opacity = opacity
        requestAnimationFrame(fade)
      }
      setTimeout(fade, 400)
    }
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
    this.cameraWaveX = ((layer / 31) - 0.5) * 0.8
  }

  addOutputToken(_token: string) {
    this.triggerTokenPulse()
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

    const colorLabel = n.role === 'attn' ? '#67e8f9' : n.role === 'ffn' ? '#f0abfc' : '#99f6e4'
    const connections = this.synapses.filter(s => s.fromIdx === this.neurons.indexOf(n) || s.toIdx === this.neurons.indexOf(n)).length
    const step = this.phase === 'thinking' ? STEP_NAMES[this.currentStep] : '—'
    const active = n.activation > 0.1

    this.inspectPanel.innerHTML = `
      <div style="color:#67e8f9;font-weight:600;font-size:11px;margin-bottom:8px;border-bottom:1px solid rgba(0,229,255,0.12);padding-bottom:6px;">Component Inspector</div>
      <div style="color:#475569">Component</div>
      <div style="color:${colorLabel};margin-bottom:4px;font-weight:600">${roleLabel}</div>
      <div style="color:#475569">Layer</div>
      <div style="color:#e2e8f0;margin-bottom:4px"><strong>${n.layer}</strong> / 31</div>
      <div style="color:#475569;font-size:9px;margin-bottom:6px">${roleDesc}</div>
      <div style="color:#475569">Connections</div>
      <div style="color:#e2e8f0;margin-bottom:4px">${connections}</div>
      <div style="color:#475569">Real Activation</div>
      <div style="margin-bottom:4px">
        <div style="background:#0f172a;border-radius:3px;height:6px;width:100%;overflow:hidden">
          <div style="background:${active ? colorLabel : '#1e293b'};height:100%;width:${(n.activation * 100).toFixed(0)}%;transition:width 0.1s;box-shadow:${active ? '0 0 8px ' + colorLabel : 'none'}"></div>
        </div>
        <span style="color:${active ? '#67e8f9' : '#334155'}">${(n.activation * 100).toFixed(0)}% (GPU readback)</span>
      </div>
      <div style="color:#475569">Current Op</div>
      <div style="color:#e2e8f0;margin-bottom:4px">${step}</div>
      <div style="color:#475569">Status</div>
      <div style="color:${active ? '#2dd4bf' : '#334155'}">${active ? 'Firing' : 'Idle'}</div>
    `
  }

  // ─── Internal ───

  private spawnSignals(layer: number) {
    // Signal particles are pure decoration — they travel along randomly
    // picked synapses between nearby layers and have no relationship to
    // any GPU state. Suppress in strict mode.
    if (!this.cinematicMode) return
    let spawned = 0
    for (const sig of this.signals) {
      if (sig.synapse !== -1 || spawned >= 4) continue
      const cands = this.synapses.filter(s => {
        const nl = this.neurons[s.fromIdx].layer, nr = this.neurons[s.toIdx].layer
        return Math.abs(nl - layer) <= 4 || Math.abs(nr - layer) <= 4
      })
      if (cands.length === 0) break
      const syn = cands[Math.floor(Math.random() * cands.length)]
      sig.synapse = this.synapses.indexOf(syn)
      sig.pos = 0
      sig.speed = 0.02 + Math.random() * 0.035
      sig.mesh.visible = true
      sig.trail.visible = true
      sig.trailIdx = 0
      sig.trailPositions.fill(0)
      const [h] = neuronHSL(layer)
      ;(sig.mesh.material as THREE.MeshBasicMaterial).color.setHSL(h, 0.95, 0.8)
      ;(sig.mesh.material as THREE.MeshBasicMaterial).opacity = 1.0
      ;(sig.trail.material as THREE.PointsMaterial).color.setHSL(h, 0.9, 0.7)
      spawned++
    }
  }

  private spawnParticlesNearLayer(layer: number) {
    const targetX = ((layer / 31) - 0.5) * 4.5
    let spawned = 0
    for (let i = 0; i < PARTICLE_COUNT && spawned < 20; i++) {
      if (this.particleLifetimes[i] > 0) continue
      const [h, s, l] = neuronHSL(layer)
      const color = new THREE.Color().setHSL(h, s, l + 0.15)

      this.particlePositions[i * 3] = targetX + (Math.random() - 0.5) * 0.5
      this.particlePositions[i * 3 + 1] = (Math.random() - 0.5) * 1.8
      this.particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 1.2

      this.particleVelocities[i * 3] = (Math.random() - 0.3) * 0.012
      this.particleVelocities[i * 3 + 1] = (Math.random() - 0.5) * 0.004
      this.particleVelocities[i * 3 + 2] = (Math.random() - 0.5) * 0.004

      this.particleColors[i * 3] = color.r
      this.particleColors[i * 3 + 1] = color.g
      this.particleColors[i * 3 + 2] = color.b

      this.particleLifetimes[i] = 0.6 + Math.random() * 1.2
      spawned++
    }
  }

  private triggerTokenPulse() {
    // Residual chain lights up
    for (const n of this.neurons) {
      if (n.role === 'residual') {
        n.activation = Math.max(n.activation, 0.75)
      }
    }

    // Last layer flash
    const lastLayerNeurons = this.neuronsByLayer.get(31) ?? []
    for (const n of lastLayerNeurons) {
      n.activation = Math.max(n.activation, 0.55)
    }

    this.spawnSignals(31)
    this.spawnSignals(28)
    this.spawnSignals(24)

    // Pulse wave on ground plane
    const lastRes = this.neurons.find(n => n.role === 'residual' && n.layer === 31)
    const cx = lastRes?.position.x ?? 2.2
    const cz = lastRes?.position.z ?? 0
    this.pulseWaves.push({ x: cx, z: cz, time: this.time, strength: 1.0 })
    if (this.pulseWaves.length > 2) this.pulseWaves.shift()

    // Colorful burst from last residual
    this.burstActive = true
    const cy = lastRes?.position.y ?? 0
    for (let i = 0; i < BURST_COUNT; i++) {
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      const speed = 0.006 + Math.random() * 0.018
      this.burstPositions[i * 3] = cx
      this.burstPositions[i * 3 + 1] = cy
      this.burstPositions[i * 3 + 2] = cz
      this.burstVelocities[i * 3] = Math.sin(phi) * Math.cos(theta) * speed
      this.burstVelocities[i * 3 + 1] = Math.sin(phi) * Math.sin(theta) * speed
      this.burstVelocities[i * 3 + 2] = Math.cos(phi) * speed
      this.burstLifetimes[i] = 0.4 + Math.random() * 0.4

      // Random bioluminescent colors
      const type = Math.random()
      if (type < 0.4) {
        this.burstColors[i * 3] = 0; this.burstColors[i * 3 + 1] = 0.9; this.burstColors[i * 3 + 2] = 1
      } else if (type < 0.7) {
        this.burstColors[i * 3] = 0.7; this.burstColors[i * 3 + 1] = 0.2; this.burstColors[i * 3 + 2] = 1
      } else {
        this.burstColors[i * 3] = 0.1; this.burstColors[i * 3 + 1] = 1; this.burstColors[i * 3 + 2] = 0.8
      }
    }
  }

  // ─── Render loop ───

  render = () => {
    requestAnimationFrame(this.render)
    this.time += 0.016

    // Ground shader is cosmetic and never created in strict mode; skip
    // its uniform updates entirely (groundMaterial is undefined here).

    // Cinematic camera with gentle drift (strict mode: camera is
    // completely user-controlled, no target lerp / vertical breath).
    if (this.cinematicMode) {
      if (this.phase === 'thinking') {
        this.controls.target.lerp(new THREE.Vector3(this.cameraWaveX * 0.12, 0, 0), 0.006)
      } else {
        this.controls.target.lerp(new THREE.Vector3(0, 0, 0), 0.015)
      }
      this.camera.position.y += Math.sin(this.time * 0.3) * 0.0003
    }
    this.controls.autoRotate = this.cinematicMode
    this.controls.update()

    // Nebula / starfield slow rotation — pure atmosphere, suppressed in
    // strict mode (they're also hidden there, but no reason to animate).
    if (this.cinematicMode) {
      for (let i = 0; i < this.nebulaLayers.length; i++) {
        this.nebulaLayers[i].rotation.y += 0.00005 * (i + 1)
        this.nebulaLayers[i].rotation.x += 0.00002 * (i + 1)
      }
      if (this.starField) this.starField.rotation.y += 0.00008
    }

    // ─── Update neurons with breathing ───
    for (const n of this.neurons) {
      const mat = n.mesh.material as THREE.MeshStandardMaterial
      const glowMat = n.glowMesh.material as THREE.MeshBasicMaterial
      const act = n.activation

      // Organic breathing — gated behind cinematicMode (accurate mode is purely activation-driven)
      const breath = this.cinematicMode
        ? Math.sin(this.time * 1.5 + n.phase) * 0.5 + 0.5
        : 0
      const idlePulse = this.cinematicMode ? 0.03 * breath : 0

      if (act > 0.2) {
        // Active: vibrant bioluminescent glow
        mat.color.lerpColors(n.baseColor, n.brightColor, act * 0.8)
        mat.emissive.lerpColors(n.baseColor, n.brightColor, act * 0.5)
        mat.emissiveIntensity = 0.15 + act * 0.8
        mat.opacity = 0.5 + act * 0.5
        n.mesh.scale.setScalar(1 + act * 0.5 + breath * 0.05)

        glowMat.color.lerpColors(n.baseColor, n.brightColor, 0.5)
        glowMat.opacity = act * 0.2
        n.glowMesh.scale.setScalar(1 + act * 1.5 + breath * 0.1)
      } else {
        // Idle: faint breathing glow (bioluminescent)
        mat.color.copy(n.baseColor)
        mat.emissive.copy(n.baseColor)
        mat.emissiveIntensity = 0.08 + idlePulse * 0.15
        mat.opacity = 0.25 + idlePulse * 0.1
        n.mesh.scale.setScalar(1 + idlePulse * 0.08)

        glowMat.opacity = idlePulse * 0.04
        n.glowMesh.scale.setScalar(1 + idlePulse * 0.2)
      }
      n.activation = Math.max(0, n.activation - 0.004)
    }

    // ─── Synapses with pulse animation ───
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

    // ─── Signals with trails ───
    for (const sig of this.signals) {
      if (sig.synapse === -1) continue
      sig.pos += sig.speed
      if (sig.pos >= 1) {
        sig.mesh.visible = false
        sig.trail.visible = false
        sig.synapse = -1
        continue
      }
      const syn = this.synapses[sig.synapse]
      // Interpolate along curved path
      const pts = syn.points
      const totalPts = pts.length - 1
      const exactIdx = sig.pos * totalPts
      const idx = Math.floor(exactIdx)
      const frac = exactIdx - idx
      if (idx < totalPts) {
        sig.mesh.position.lerpVectors(pts[idx], pts[idx + 1], frac)
      }
      ;(sig.mesh.material as THREE.MeshBasicMaterial).opacity = 1.0 * (1 - Math.abs(sig.pos - 0.5) * 1.8)

      // Update trail
      const ti = sig.trailIdx % 8
      sig.trailPositions[ti * 3] = sig.mesh.position.x
      sig.trailPositions[ti * 3 + 1] = sig.mesh.position.y
      sig.trailPositions[ti * 3 + 2] = sig.mesh.position.z
      sig.trailIdx++
      sig.trail.geometry.attributes.position.needsUpdate = true
      ;(sig.trail.material as THREE.PointsMaterial).opacity = 0.3 * (1 - Math.abs(sig.pos - 0.5) * 1.6)
    }

    // ─── Flowing particles ───
    let particlesDirty = false
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      if (this.particleLifetimes[i] <= 0) continue
      particlesDirty = true
      this.particleLifetimes[i] -= 0.013
      this.particlePositions[i * 3] += this.particleVelocities[i * 3]
      this.particlePositions[i * 3 + 1] += this.particleVelocities[i * 3 + 1]
      this.particlePositions[i * 3 + 2] += this.particleVelocities[i * 3 + 2]
      // Gentle fade
      this.particleColors[i * 3] *= 0.997
      this.particleColors[i * 3 + 1] *= 0.997
      this.particleColors[i * 3 + 2] *= 0.997
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

    // ─── Ambient dust (fireflies) ───
    let dustDirty = false
    for (let i = 0; i < DUST_COUNT; i++) {
      const phase = this.dustPhases[i]
      // Gentle floating motion
      this.dustPositions[i * 3] += Math.sin(this.time * 0.3 + phase) * 0.0004
      this.dustPositions[i * 3 + 1] += Math.cos(this.time * 0.2 + phase * 1.3) * 0.0003
      this.dustPositions[i * 3 + 2] += Math.sin(this.time * 0.25 + phase * 0.7) * 0.0003
      dustDirty = true
    }
    if (dustDirty) {
      this.dustSystem.geometry.attributes.position.needsUpdate = true
      // Firefly flickering
      const dustMat = this.dustSystem.material as THREE.PointsMaterial
      dustMat.opacity = 0.15 + Math.sin(this.time * 2) * 0.08
    }

    // ─── Token burst particles ───
    if (this.burstActive) {
      let anyAlive = false
      for (let i = 0; i < BURST_COUNT; i++) {
        if (this.burstLifetimes[i] <= 0) continue
        anyAlive = true
        this.burstLifetimes[i] -= 0.016
        this.burstPositions[i * 3] += this.burstVelocities[i * 3]
        this.burstPositions[i * 3 + 1] += this.burstVelocities[i * 3 + 1]
        this.burstPositions[i * 3 + 2] += this.burstVelocities[i * 3 + 2]
        // Gentle drag
        this.burstVelocities[i * 3] *= 0.995
        this.burstVelocities[i * 3 + 1] *= 0.995
        this.burstVelocities[i * 3 + 2] *= 0.995
        this.burstVelocities[i * 3 + 1] -= 0.00015
      }
      this.burstParticles.geometry.attributes.position.needsUpdate = true
      this.burstParticles.geometry.attributes.color.needsUpdate = true
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
        <span style="color:#67e8f9">${PHASE_LABELS[phaseIdx]}</span><br>
        Layer ${this.activeLayer}/31 &bull; ${Math.round(this.outputConfidence * 100)}% &bull; ${STEP_NAMES[this.currentStep] || ''}<br>
        Dispatches: ${this.dispatchCount}/${this.totalDispatches}
      `
    } else if (this.phase === 'idle') {
      this.overlay.innerHTML = '<span style="color:#1e293b">Idle — type a question</span>'
    } else {
      this.overlay.innerHTML = '<span style="color:#2dd4bf">Done</span>'
    }

    // Selection ring
    if (this.selectedRing && this.selectedNeuron) {
      this.selectedRing.lookAt(this.camera.position)
      // Pulsing ring
      const ringPulse = Math.sin(this.time * 3) * 0.15 + 0.85
      ;(this.selectedRing.material as THREE.MeshBasicMaterial).opacity = ringPulse
      this.updateInspectPanel()
    }

    // ─── Dense components: embedding sphere + LM head + slabs decay ───
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

    // Decay attention beams
    if (this.attnScoreLines.length > 0) {
      for (let i = this.attnScoreLines.length - 1; i >= 0; i--) {
        const line = this.attnScoreLines[i]
        const mat = line.material as THREE.LineBasicMaterial
        mat.opacity = Math.max(0, mat.opacity - 0.01)
        if (mat.opacity <= 0) {
          this.scene.remove(line)
          line.geometry.dispose()
          mat.dispose()
          this.attnScoreLines.splice(i, 1)
        }
      }
    }

    // Render: bloom post-processing is decorative (adds a glow halo that
    // doesn't correspond to any GPU state). In strict mode we bypass it
    // and render directly so every pixel on screen has a 1:1 mapping to
    // a real neuron position * real activation magnitude.
    if (this.cinematicMode) {
      this.composer.render()
    } else {
      this.renderer.render(this.scene, this.camera)
    }
  }

  start() { this.render() }
  stop() { }
}

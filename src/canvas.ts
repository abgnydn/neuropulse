/**
 * PAGE-WIDE THREE.JS BACKDROP — one fixed canvas, scroll-driven morph.
 *
 * Replaces the per-section hero.ts + numbers-bg.ts with a single persistent
 * scene whose uniforms are driven by `window.scrollY / scrollMax`:
 *
 *   prog = 0.00 (hero)           → volumetric cylindrical cloud,
 *                                   vertical 32-band structure, cursor parallax
 *   prog ≈ 0.50 (mid-essay)       → flattening, leftward flow kicks in,
 *                                   master alpha dims for readability
 *   prog = 1.00 (numbers → CTA)   → strong horizontal scan-line flow,
 *                                   alpha rises again for the payoff
 *
 * One shader, three knobs — no scene graph swapping, no per-section canvas.
 *
 * A11y: respects prefers-reduced-motion (skips entirely).
 * Perf: DPR capped, particle count halved on mobile, additive-blended points
 * (no post-processing composer), IntersectionObserver pauses when tab hidden.
 */

import * as THREE from 'three'

(function initPageCanvas(): void {
  const canvas = document.getElementById('page-canvas') as HTMLCanvasElement | null
  if (!canvas) return
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

  const isMobile =
    window.innerWidth < 820 ||
    /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent)

  const N = isMobile ? 9_000 : 28_000
  const DPR_CAP = isMobile ? 1.5 : 2

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: !isMobile,
    alpha: true,
    powerPreference: 'high-performance',
  })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, DPR_CAP))

  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 300)
  camera.position.set(0, 0, 16)

  // ─── geometry: 32 soft horizontal bands, cylindrical distribution ───
  const pos = new Float32Array(N * 3)
  const seed = new Float32Array(N * 3)
  for (let i = 0; i < N; i++) {
    const layer = Math.floor(Math.random() * 32)
    const bandY = (layer / 31 - 0.5) * 18
    const r = Math.pow(Math.random(), 0.7) * 9 + 0.5
    const theta = Math.random() * Math.PI * 2
    pos[i * 3]     = r * Math.cos(theta)
    pos[i * 3 + 1] = bandY + (Math.random() - 0.5) * 0.4
    pos[i * 3 + 2] = r * Math.sin(theta) - 4
    seed[i * 3]     = Math.random()
    seed[i * 3 + 1] = Math.random()
    seed[i * 3 + 2] = Math.random()
  }
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  geom.setAttribute('seed',     new THREE.BufferAttribute(seed, 3))

  const uniforms = {
    time:    { value: 0 },
    prog:    { value: 0 }, // 0 at top of page → 1 at bottom
    fade:    { value: 1 }, // master alpha (readability + hero-entrance)
    cyan:    { value: new THREE.Color(0x5eead4) },
    cyanHot: { value: new THREE.Color(0x00e5ff) },
    violet:  { value: new THREE.Color(0xc084fc) },
  }

  const mat = new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      attribute vec3 seed;
      uniform float time;
      uniform float prog;
      uniform float fade;
      uniform vec3 cyan;
      uniform vec3 cyanHot;
      uniform vec3 violet;
      varying vec3 vColor;
      varying float vAlpha;

      void main() {
        vec3 p = position;
        float t = time * 0.12;

        // Volumetric drift (dominant in hero) fades as prog rises
        float ph  = seed.x * 6.2831;
        float ph2 = seed.y * 6.2831;
        float amp = 0.5 + seed.z * 1.3;
        float volWeight = mix(1.0, 0.35, prog);

        p.x += sin(t + ph)                        * amp * volWeight;
        p.y += cos(t * 0.8 + ph2)                 * amp * 0.35 * volWeight;
        p.z += sin(t * 0.6 + ph + seed.y * 3.0)   * amp * 0.7  * volWeight;

        // Leftward flow that ramps up as prog rises (mid-scroll onward)
        float flowAmt = smoothstep(0.15, 0.85, prog);
        float flowSpeed = (0.6 + seed.z * 1.4) * flowAmt;
        float wrap = 36.0;
        p.x = mod(p.x + wrap * 0.5 - time * flowSpeed, wrap) - wrap * 0.5;

        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;

        // Slightly smaller points as flow tightens
        float depth = -mv.z;
        float baseSize = 1.0 + seed.z * 3.4;
        gl_PointSize = baseSize * mix(1.0, 0.7, prog) * (90.0 / depth);

        // Palette: mostly cyan, rare violet, rarest hot-cyan activity
        float hot       = step(0.965, seed.x);
        float violetMix = smoothstep(0.55, 0.95, seed.y);
        vec3 base       = mix(cyan, violet, violetMix);
        vColor          = mix(base, cyanHot, hot);

        // Per-point breathing + depth fade + master alpha
        float pulse = 0.55 + 0.45 * sin(t * 2.4 + seed.x * 18.0 + seed.y * 12.0);
        vAlpha = pulse * smoothstep(40.0, 5.0, depth) * fade;
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        vec2 uv = gl_PointCoord - 0.5;
        float d = length(uv);
        if (d > 0.5) discard;
        float fall = smoothstep(0.5, 0.0, d);
        fall = pow(fall, 1.4);
        gl_FragColor = vec4(vColor * (1.0 + fall * 0.6), fall * vAlpha);
      }
    `,
  })

  const points = new THREE.Points(geom, mat)
  scene.add(points)

  // ─── sizing ───
  function resize(): void {
    // Canvas is fixed full-viewport, so use window dimensions.
    const w = window.innerWidth
    const h = window.innerHeight
    renderer.setSize(w, h, false)
    camera.aspect = w / h
    camera.updateProjectionMatrix()
  }
  resize()
  window.addEventListener('resize', resize)

  // ─── cursor parallax (only noticeable in hero — it attenuates with prog) ───
  let mx = 0
  let my = 0
  window.addEventListener(
    'pointermove',
    (e: PointerEvent) => {
      mx = (e.clientX / window.innerWidth) * 2 - 1
      my = (e.clientY / window.innerHeight) * 2 - 1
    },
    { passive: true },
  )

  // ─── scroll → page-progress (0→1) + readability-aware master fade ───
  let prog = 0
  let readability = 1

  function smoothstep(a: number, b: number, x: number): number {
    const t = Math.max(0, Math.min(1, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)
  }

  function updateScroll(): void {
    const scrollY = window.scrollY
    const maxY = Math.max(1, document.documentElement.scrollHeight - window.innerHeight)
    prog = Math.max(0, Math.min(1, scrollY / maxY))

    // Readability master fade curve:
    //   hero (0.00–0.08): full                       → 1.0
    //   essay body (0.15–0.70): dimmer for reading   → 0.35
    //   numbers (0.78–0.90): rise for the payoff     → 0.85
    //   CTA / end (0.95–1.00): full                  → 1.0
    const heroFull    = 1 - smoothstep(0.04, 0.15, prog)              // fade out
    const essayDim    = smoothstep(0.15, 0.25, prog) * (1 - smoothstep(0.70, 0.78, prog))
    const numbersRise = smoothstep(0.78, 0.88, prog) * (1 - smoothstep(0.93, 0.98, prog))
    const ctaFull     = smoothstep(0.93, 1.00, prog)

    // Compose: start at 1 (hero) → 0.35 (essay) → 0.85 (numbers) → 1.0 (cta).
    readability =
      Math.max(
        heroFull,                 // 1 at top
        0.35 + (1 - essayDim) * 0.0, // always floor 0.35 when essayDim=1
        numbersRise * 0.85,
        ctaFull,
      )
    // essayDim=1 means we're in the essay body; we want readability ~0.35.
    // Above Math.max gives heroFull≈0 + 0.35 + numbersRise≈0 + ctaFull≈0 → 0.35. Good.
  }
  updateScroll()
  window.addEventListener('scroll', updateScroll, { passive: true })

  // ─── visibility: pause when tab hidden ───
  let running = true
  document.addEventListener('visibilitychange', () => {
    running = document.visibilityState === 'visible'
  })

  // ─── entrance: fade in over 1s to cover initial layout thrash ───
  canvas.style.opacity = '0'
  canvas.style.transition = 'opacity 1.1s cubic-bezier(0.16, 1, 0.3, 1)'
  requestAnimationFrame(() => {
    canvas.style.opacity = '1'
  })

  // ─── main loop ───
  let t = 0
  function tick(): void {
    if (running) {
      t += 1 / 60
      uniforms.time.value = t
      uniforms.prog.value = prog
      uniforms.fade.value = readability

      // Cursor parallax — loud in hero, silent as we scroll away
      const parallaxWeight = 1 - Math.min(1, prog * 4)
      camera.position.x += (mx * 0.7 * parallaxWeight - camera.position.x) * 0.03
      camera.position.y += (-my * 0.5 * parallaxWeight - camera.position.y) * 0.03
      camera.lookAt(0, 0, 0)

      // Slow world spin (hero drama) — fades out as you scroll
      points.rotation.y += 0.0006 * (1 - prog * 0.7)

      renderer.render(scene, camera)
    }
    requestAnimationFrame(tick)
  }
  tick()
})()

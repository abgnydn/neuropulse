/**
 * SPATIAL PANELS — world-anchored floating UI cards around the 3D scene.
 *
 * In Universe mode (scene mode), each side-panel section becomes a glass
 * card floating at a specific world-space anchor around the model. The
 * cards orbit naturally when the user drags the OrbitControls (same camera)
 * but always billboard to face the viewer so the content stays legible.
 *
 * Implementation: DOM elements with `data-anchor="x,y,z"` attributes. Each
 * frame we project each anchor through the live WebGL camera, map NDC to
 * pixel coords, and apply a `translate + scale` transform to the DOM.
 *
 * Depth scaling: closer = larger (readable), farther = smaller (spatial).
 * Panels behind the camera (NDC z > 1 or < -1) fade to 0 opacity.
 */

import * as THREE from 'three'
import type { BrainVisualizer } from './visualizer'

interface Anchor {
  el: HTMLElement
  pos: THREE.Vector3
}

export class SpatialPanels {
  private anchors: Anchor[] = []
  private camera: THREE.PerspectiveCamera
  private canvas: HTMLCanvasElement
  private active = false
  private tmp = new THREE.Vector3()
  private rafId = 0

  constructor(vis: BrainVisualizer) {
    this.camera = vis.getCamera()
    this.canvas = vis.getCanvas()
  }

  /** Discover panels marked with `data-anchor="x,y,z"` and register them. */
  registerFromDOM(selector = '[data-anchor]'): void {
    const els = document.querySelectorAll<HTMLElement>(selector)
    els.forEach((el) => {
      const data = el.dataset.anchor
      if (!data) return
      const parts = data.split(',').map((s) => parseFloat(s.trim()))
      if (parts.length !== 3 || parts.some((n) => Number.isNaN(n))) return
      this.anchors.push({ el, pos: new THREE.Vector3(parts[0]!, parts[1]!, parts[2]!) })
    })
  }

  enable(enabled: boolean): void {
    if (enabled === this.active) return
    this.active = enabled
    if (enabled) {
      // Seed an initial pass so panels don't pop in at (0, 0)
      this.updateOnce()
      this.loop()
    } else {
      cancelAnimationFrame(this.rafId)
      // Reset styles so classic side panel reverts cleanly when switching away
      for (const a of this.anchors) {
        a.el.style.transform = ''
        a.el.style.opacity = ''
      }
    }
  }

  private loop = (): void => {
    if (!this.active) return
    this.updateOnce()
    this.rafId = requestAnimationFrame(this.loop)
  }

  private updateOnce(): void {
    const w = this.canvas.clientWidth
    const h = this.canvas.clientHeight
    if (w === 0 || h === 0) return

    for (const a of this.anchors) {
      // Skip panels the mode system has hidden via display:none (e.g., attn-panel
      // and lens-panel that sit dormant in Scene mode waiting for their mode).
      if (a.el.offsetParent === null && getComputedStyle(a.el).display === 'none') {
        continue
      }

      this.tmp.copy(a.pos).project(this.camera)
      const behind = this.tmp.z > 1 || this.tmp.z < -1

      const x = (this.tmp.x * 0.5 + 0.5) * w
      const y = (1 - (this.tmp.y * 0.5 + 0.5)) * h

      // Depth-based scale: closer camera → bigger card; clamped so text stays
      // readable at extremes of the OrbitControls zoom range (1.5..10).
      const dist = this.camera.position.distanceTo(a.pos)
      const rawScale = 4.8 / Math.max(0.5, dist)
      const scale = Math.max(0.55, Math.min(1.08, rawScale))

      a.el.style.transform =
        `translate(-50%, -50%) translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) scale(${scale.toFixed(3)})`
      a.el.style.opacity = behind ? '0' : '1'
    }
  }
}

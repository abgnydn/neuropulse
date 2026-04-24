/**
 * SPATIAL PANELS — HUD-docked floating UI cards.
 *
 * Panels are registered with `data-anchor="x,y,z"` but those coords are
 * interpreted as **camera-space offsets from the orbit target**, not
 * world-space positions. Each frame we:
 *
 *   1. read the camera's right / up / forward unit vectors
 *   2. build a world point = target + right*x + up*y − forward*z
 *   3. project that through the camera to get pixel coords
 *   4. apply a translate + scale to the DOM element
 *
 * Because the anchor travels WITH the camera, the panel stays at a fixed
 * screen position no matter how the user orbits or zooms. Panels also
 * billboard naturally (DOM is screen-aligned).
 *
 * A mild depth-scale is kept so zooming in/out gently grows/shrinks the
 * cards — maintains spatial feel without losing legibility.
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
  private vis: BrainVisualizer
  private active = false
  private tmp = new THREE.Vector3()
  private right = new THREE.Vector3()
  private up = new THREE.Vector3()
  private forward = new THREE.Vector3()
  private worldPos = new THREE.Vector3()
  private rafId = 0

  constructor(vis: BrainVisualizer) {
    this.vis = vis
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
      this.updateOnce()
      this.loop()
    } else {
      cancelAnimationFrame(this.rafId)
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

    // Camera basis in world space. matrixWorld columns 0/1/2 are right/up/back,
    // so forward (the direction camera is LOOKING) is -column 2.
    this.camera.matrixWorld.extractBasis(this.right, this.up, this.forward)
    this.forward.negate()

    // Pivot around the orbit target so panels hover around whatever the
    // camera is focused on. Zoom distance drives a mild depth scale.
    const target = this.vis.getControlsTarget()
    const camDist = this.camera.position.distanceTo(target)

    for (const a of this.anchors) {
      if (a.el.offsetParent === null && getComputedStyle(a.el).display === 'none') {
        continue
      }

      // Build a world point that sits at the (x, y, z) offset *in the
      // camera's frame*, anchored at the orbit target. As the user orbits,
      // this world point moves with the camera, so screen coords stay
      // roughly constant — that's the whole point of HUD docking.
      //
      //   +x = right in view,  +y = up in view,  +z = toward the camera
      this.worldPos.copy(target)
        .addScaledVector(this.right,   a.pos.x)
        .addScaledVector(this.up,      a.pos.y)
        .addScaledVector(this.forward, -a.pos.z)

      this.tmp.copy(this.worldPos).project(this.camera)
      const behind = this.tmp.z > 1 || this.tmp.z < -1

      const x = (this.tmp.x * 0.5 + 0.5) * w
      const y = (1 - (this.tmp.y * 0.5 + 0.5)) * h

      // Gentler scale curve than the old world-anchored math: near-constant
      // across the OrbitControls zoom range so panels stay equally legible
      // whether the camera is close or far.
      const rawScale = 5.5 / Math.max(2.0, camDist)
      const scale = Math.max(1.0, Math.min(1.35, rawScale))

      a.el.style.transform =
        `translate(-50%, -50%) translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) scale(${scale.toFixed(3)})`
      a.el.style.opacity = behind ? '0' : '1'
    }
  }
}

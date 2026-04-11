// ═══════════════════════════════════════════════════════════════
// Neuropulse — Audio Engine
// Warm ambient pad, soft pentatonic ticks, crystalline FM chimes.
// Lazy init on first user gesture; mute state persists in localStorage.
// ═══════════════════════════════════════════════════════════════

const STORAGE_KEY = 'neuropulse:muted'
// Legacy key from the pre-rename "Neural Pulse" era. On first load after
// the rename, copy it forward so returning users don't lose their mute
// preference. The old key is never written to again.
const LEGACY_STORAGE_KEY = 'neural-pulse:muted'
const MASTER_LEVEL = 0.32
const RAMP_FAST = 0.15
const RAMP_SLOW = 0.6

export class AudioEngine {
  private ctx: AudioContext | null = null
  private masterGain: GainNode | null = null
  private reverbSend: ConvolverNode | null = null
  private reverbGain: GainNode | null = null
  private dryGain: GainNode | null = null

  // Drone
  private padGain: GainNode | null = null
  private padFilter: BiquadFilterNode | null = null
  private padOscs: OscillatorNode[] = []
  private noiseNode: AudioBufferSourceNode | null = null
  private noiseGain: GainNode | null = null

  private started = false
  private muted = false
  private lastTickTime = 0
  private tickCount = 0
  private lastChimeTime = 0

  constructor() {
    // Load persisted mute preference before any user interaction so the
    // sound button reflects the user's last choice on page load.
    try {
      let v = localStorage.getItem(STORAGE_KEY)
      // One-time migration from the legacy key.
      if (v === null) {
        const legacy = localStorage.getItem(LEGACY_STORAGE_KEY)
        if (legacy !== null) {
          localStorage.setItem(STORAGE_KEY, legacy)
          localStorage.removeItem(LEGACY_STORAGE_KEY)
          v = legacy
        }
      }
      this.muted = v === '1'
    } catch { /* localStorage unavailable — ignore */ }
  }

  /** Whether mute is currently active (sync-readable from UI). */
  isMuted(): boolean {
    return this.muted
  }

  init() {
    if (this.ctx) return
    this.ctx = new AudioContext({ sampleRate: 44100 })
    this.masterGain = this.ctx.createGain()
    this.masterGain.gain.value = this.muted ? 0 : MASTER_LEVEL
    this.masterGain.connect(this.ctx.destination)

    // Dry path
    this.dryGain = this.ctx.createGain()
    this.dryGain.gain.value = 0.6
    this.dryGain.connect(this.masterGain)

    // Reverb (algorithmic via feedback delay network)
    this.reverbGain = this.ctx.createGain()
    this.reverbGain.gain.value = 0.4
    this.reverbSend = this.createReverb()
    this.reverbSend.connect(this.reverbGain)
    this.reverbGain.connect(this.masterGain)
  }

  // Algorithmic reverb using impulse response
  private createReverb(): ConvolverNode {
    const conv = this.ctx!.createConvolver()
    const rate = this.ctx!.sampleRate
    const length = rate * 2.5  // 2.5 second tail
    const impulse = this.ctx!.createBuffer(2, length, rate)

    for (let ch = 0; ch < 2; ch++) {
      const data = impulse.getChannelData(ch)
      for (let i = 0; i < length; i++) {
        const t = i / rate
        const decay = Math.exp(-t * 2.5)
        const early = t < 0.08 ? Math.random() * 0.5 : 0
        const late = Math.random() * decay
        data[i] = (early + late) * 0.3
      }
    }
    conv.buffer = impulse
    return conv
  }

  async resume() {
    if (!this.ctx) this.init()
    if (this.ctx!.state === 'suspended') await this.ctx!.resume()
  }

  setMuted(m: boolean) {
    this.muted = m
    try { localStorage.setItem(STORAGE_KEY, m ? '1' : '0') } catch { /* ignore */ }
    if (this.masterGain && this.ctx) {
      const now = this.ctx.currentTime
      this.masterGain.gain.cancelScheduledValues(now)
      this.masterGain.gain.linearRampToValueAtTime(m ? 0 : MASTER_LEVEL, now + 0.2)
    }
  }
  toggleMute() { this.setMuted(!this.muted); return this.muted }

  // ─── Warm ambient pad ───
  // Detuned saws through a lowpass, plus a sub sine and bandpassed noise bed.
  startDrone() {
    if (!this.ctx || !this.dryGain || this.started) return
    this.started = true

    this.padGain = this.ctx.createGain()
    this.padGain.gain.value = 0

    this.padFilter = this.ctx.createBiquadFilter()
    this.padFilter.type = 'lowpass'
    this.padFilter.frequency.value = 110
    this.padFilter.Q.value = 0.7

    // Warm pad: 4 detuned oscillators for chorus effect
    const base = 55 // A1
    const detunes = [-8, -3, 3, 7] // cents
    for (const det of detunes) {
      const osc = this.ctx.createOscillator()
      osc.type = 'sawtooth'
      osc.frequency.value = base
      osc.detune.value = det
      const oscGain = this.ctx.createGain()
      oscGain.gain.value = 0.07
      osc.connect(oscGain)
      oscGain.connect(this.padFilter)
      osc.start()
      this.padOscs.push(osc)
    }

    // Sub bass (pure sine)
    const sub = this.ctx.createOscillator()
    sub.type = 'sine'
    sub.frequency.value = 27.5 // A0
    const subGain = this.ctx.createGain()
    subGain.gain.value = 0.10
    sub.connect(subGain)
    subGain.connect(this.padFilter)
    sub.start()
    this.padOscs.push(sub)

    // Filtered noise bed
    this.noiseGain = this.ctx.createGain()
    this.noiseGain.gain.value = 0.006
    const noiseFilter = this.ctx.createBiquadFilter()
    noiseFilter.type = 'bandpass'
    noiseFilter.frequency.value = 220
    noiseFilter.Q.value = 0.5

    const noiseLen = this.ctx.sampleRate * 4
    const noiseBuf = this.ctx.createBuffer(1, noiseLen, this.ctx.sampleRate)
    const noiseData = noiseBuf.getChannelData(0)
    for (let i = 0; i < noiseLen; i++) noiseData[i] = Math.random() * 2 - 1
    this.noiseNode = this.ctx.createBufferSource()
    this.noiseNode.buffer = noiseBuf
    this.noiseNode.loop = true
    this.noiseNode.connect(noiseFilter)
    noiseFilter.connect(this.noiseGain)
    this.noiseGain.connect(this.padGain)
    this.noiseNode.start()

    this.padFilter.connect(this.padGain)
    this.padGain.connect(this.dryGain!)
    this.padGain.connect(this.reverbSend!)

    // Gentle fade in (2.5s) so entering the drone is not abrupt.
    this.padGain.gain.linearRampToValueAtTime(0.38, this.ctx.currentTime + 2.5)
  }

  setDroneIntensity(v: number) {
    if (!this.padGain || !this.padFilter || !this.noiseGain || !this.ctx) return
    const t = this.ctx.currentTime + RAMP_FAST
    // Softer floor + narrower sweep than before — feels like ambience, not
    // a car alarm ramping as confidence rises.
    this.padGain.gain.linearRampToValueAtTime(0.22 + v * 0.24, t)
    // Filter opens slightly — reveals more harmonics as thinking deepens.
    this.padFilter.frequency.linearRampToValueAtTime(110 + v * 280, t)
    // Noise bed tracks intensity.
    this.noiseGain.gain.linearRampToValueAtTime(0.006 + v * 0.012, t)
    // Pad pitch rises a few Hz at peak.
    if (this.padOscs[0]) {
      this.padOscs[0].frequency.linearRampToValueAtTime(55 + v * 6, t)
    }
  }

  // ─── Neural tick: soft water-drop pluck ───
  neuronTick(layer: number) {
    if (!this.ctx || !this.dryGain || !this.reverbSend || this.muted) return
    const now = this.ctx.currentTime
    if (now - this.lastTickTime < 0.08) return
    this.lastTickTime = now
    this.tickCount++

    // Only tick every 4th call to avoid density.
    if (this.tickCount % 4 !== 0) return

    // Major pentatonic across two octaves, spread by layer so deeper layers
    // hit higher notes — gives the thinking process a rising arc.
    const penta = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25, 783.99, 880.00]
    const freq = penta[layer % penta.length] * (layer >= 16 ? 1.5 : 1.0)

    // White-noise burst through a high-Q bandpass = tonal "plink".
    const bufLen = Math.floor(this.ctx.sampleRate * 0.015)
    const buf = this.ctx.createBuffer(1, bufLen, this.ctx.sampleRate)
    const data = buf.getChannelData(0)
    for (let i = 0; i < bufLen; i++) {
      data[i] = (Math.random() * 2 - 1) * (1 - i / bufLen)
    }

    const src = this.ctx.createBufferSource()
    src.buffer = buf

    const bp = this.ctx.createBiquadFilter()
    bp.type = 'bandpass'
    bp.frequency.value = freq
    bp.Q.value = 32

    const gain = this.ctx.createGain()
    gain.gain.setValueAtTime(0.045, now)
    gain.gain.exponentialRampToValueAtTime(0.0005, now + 0.18)

    src.connect(bp)
    bp.connect(gain)
    gain.connect(this.dryGain)
    gain.connect(this.reverbSend)
    src.start(now)
    src.stop(now + 0.2)
  }

  // ─── Token chime: crystalline FM bell ───
  tokenChime() {
    if (!this.ctx || !this.dryGain || !this.reverbSend || this.muted) return
    const now = this.ctx.currentTime
    // Rate-limit: never more than 6 chimes/sec even if tokens stream faster.
    if (now - this.lastChimeTime < 0.17) return
    this.lastChimeTime = now

    // Pitch varies per chime but within a gentle major-6th window, not wild.
    const choices = [660, 740, 784, 880, 988, 1046]
    const baseFreq = choices[Math.floor(Math.random() * choices.length)]

    // Carrier
    const carrier = this.ctx.createOscillator()
    carrier.type = 'sine'
    carrier.frequency.value = baseFreq

    // Modulator (FM). Inharmonic ratio → bell timbre.
    const mod = this.ctx.createOscillator()
    mod.type = 'sine'
    mod.frequency.value = baseFreq * 1.414
    const modGain = this.ctx.createGain()
    modGain.gain.setValueAtTime(baseFreq * 0.45, now)
    modGain.gain.exponentialRampToValueAtTime(1, now + 0.7)
    mod.connect(modGain)
    modGain.connect(carrier.frequency)

    // Envelope — softer peak, longer tail than before.
    const env = this.ctx.createGain()
    env.gain.setValueAtTime(0, now)
    env.gain.linearRampToValueAtTime(0.032, now + 0.006)
    env.gain.exponentialRampToValueAtTime(0.0005, now + RAMP_SLOW + 0.2)

    // High-shelf adds shimmer without harshness.
    const shelf = this.ctx.createBiquadFilter()
    shelf.type = 'highshelf'
    shelf.frequency.value = 2200
    shelf.gain.value = 2.5

    carrier.connect(shelf)
    shelf.connect(env)
    env.connect(this.dryGain)
    env.connect(this.reverbSend)

    carrier.start(now)
    mod.start(now)
    carrier.stop(now + 1)
    mod.stop(now + 1)

    // Inharmonic partial: adds bell sparkle without stepping on the root.
    const p2 = this.ctx.createOscillator()
    p2.type = 'sine'
    p2.frequency.value = baseFreq * 3.01
    const p2Gain = this.ctx.createGain()
    p2Gain.gain.setValueAtTime(0, now)
    p2Gain.gain.linearRampToValueAtTime(0.008, now + 0.004)
    p2Gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.45)
    p2.connect(p2Gain)
    p2Gain.connect(this.dryGain)
    p2Gain.connect(this.reverbSend)
    p2.start(now)
    p2.stop(now + 0.5)
  }

  stopDrone() {
    if (!this.padGain || !this.ctx) return
    this.started = false
    const t = this.ctx.currentTime
    this.padGain.gain.cancelScheduledValues(t)
    this.padGain.gain.linearRampToValueAtTime(0, t + 2)

    setTimeout(() => {
      for (const osc of this.padOscs) {
        try { osc.stop() } catch { /* ok */ }
      }
      this.padOscs = []
      try { this.noiseNode?.stop() } catch { /* ok */ }
      this.noiseNode = null
    }, 2500)
  }
}

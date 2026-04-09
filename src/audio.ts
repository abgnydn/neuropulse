// ═══════════════════════════════════════════════════════════════
// Neural Pulse — Audio Engine v3 (Professional Sound Design)
// Warm ambient pad, soft droplet ticks, crystalline chimes
// ═══════════════════════════════════════════════════════════════

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

  init() {
    if (this.ctx) return
    this.ctx = new AudioContext({ sampleRate: 44100 })
    this.masterGain = this.ctx.createGain()
    this.masterGain.gain.value = 0.35
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
        // Exponential decay with diffusion
        const t = i / rate
        const decay = Math.exp(-t * 2.5)
        // Early reflections + late diffuse tail
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
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.linearRampToValueAtTime(m ? 0 : 0.35, this.ctx.currentTime + 0.2)
    }
  }
  toggleMute() { this.setMuted(!this.muted); return this.muted }

  // ─── Warm ambient pad ───
  // Uses detuned saws through heavy low-pass + noise texture
  startDrone() {
    if (!this.ctx || !this.dryGain || this.started) return
    this.started = true

    this.padGain = this.ctx.createGain()
    this.padGain.gain.value = 0

    this.padFilter = this.ctx.createBiquadFilter()
    this.padFilter.type = 'lowpass'
    this.padFilter.frequency.value = 120
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
      oscGain.gain.value = 0.08
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
    subGain.gain.value = 0.12
    sub.connect(subGain)
    subGain.connect(this.padFilter)
    sub.start()
    this.padOscs.push(sub)

    // Filtered noise texture
    this.noiseGain = this.ctx.createGain()
    this.noiseGain.gain.value = 0.008
    const noiseFilter = this.ctx.createBiquadFilter()
    noiseFilter.type = 'bandpass'
    noiseFilter.frequency.value = 200
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

    // Fade in
    this.padGain.gain.linearRampToValueAtTime(0.5, this.ctx.currentTime + 2)
  }

  setDroneIntensity(v: number) {
    if (!this.padGain || !this.padFilter || !this.noiseGain || !this.ctx) return
    const t = this.ctx.currentTime + 0.15
    // Volume rises gently
    this.padGain.gain.linearRampToValueAtTime(0.3 + v * 0.3, t)
    // Filter opens — reveals more harmonics as thinking deepens
    this.padFilter.frequency.linearRampToValueAtTime(120 + v * 350, t)
    // Noise texture increases
    this.noiseGain.gain.linearRampToValueAtTime(0.008 + v * 0.015, t)
    // Pad pitch rises by a few Hz at peak
    if (this.padOscs[0]) {
      this.padOscs[0].frequency.linearRampToValueAtTime(55 + v * 8, t)
    }
  }

  // ─── Neural tick: soft water-drop pluck ───
  // Resonant filter impulse = organic "plink" sound
  neuronTick(layer: number) {
    if (!this.ctx || !this.dryGain || !this.reverbSend) return
    const now = this.ctx.currentTime
    if (now - this.lastTickTime < 0.06) return
    this.lastTickTime = now
    this.tickCount++

    // Only tick every few layers to avoid noise floor
    if (this.tickCount % 3 !== 0) return

    // Pentatonic frequencies for pleasant randomness
    const penta = [261, 293, 329, 392, 440, 523, 587, 659, 784, 880]
    const freq = penta[layer % penta.length]

    // White noise burst → resonant bandpass = water drop
    const bufLen = Math.floor(this.ctx.sampleRate * 0.015) // 15ms noise burst
    const buf = this.ctx.createBuffer(1, bufLen, this.ctx.sampleRate)
    const data = buf.getChannelData(0)
    for (let i = 0; i < bufLen; i++) {
      data[i] = (Math.random() * 2 - 1) * (1 - i / bufLen) // decaying noise
    }

    const src = this.ctx.createBufferSource()
    src.buffer = buf

    // Resonant filter gives the "plink" character
    const bp = this.ctx.createBiquadFilter()
    bp.type = 'bandpass'
    bp.frequency.value = freq
    bp.Q.value = 30 // high resonance = tonal ping

    const gain = this.ctx.createGain()
    gain.gain.setValueAtTime(0.06, now)
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15)

    src.connect(bp)
    bp.connect(gain)
    gain.connect(this.dryGain)
    gain.connect(this.reverbSend) // send to reverb for space
    src.start(now)
    src.stop(now + 0.15)
  }

  // ─── Token chime: crystalline bell ───
  // FM synthesis + inharmonic partials = bell timbre
  tokenChime() {
    if (!this.ctx || !this.dryGain || !this.reverbSend) return
    const now = this.ctx.currentTime

    // Bell via FM synthesis: carrier + modulator
    const baseFreq = 880 + Math.random() * 200 // slight variation each token

    // Carrier
    const carrier = this.ctx.createOscillator()
    carrier.type = 'sine'
    carrier.frequency.value = baseFreq

    // Modulator (FM)
    const mod = this.ctx.createOscillator()
    mod.type = 'sine'
    mod.frequency.value = baseFreq * 1.414 // inharmonic ratio = bell-like
    const modGain = this.ctx.createGain()
    modGain.gain.setValueAtTime(baseFreq * 0.5, now)
    modGain.gain.exponentialRampToValueAtTime(1, now + 0.8)
    mod.connect(modGain)
    modGain.connect(carrier.frequency) // FM connection

    // Envelope
    const env = this.ctx.createGain()
    env.gain.setValueAtTime(0, now)
    env.gain.linearRampToValueAtTime(0.04, now + 0.005) // fast attack
    env.gain.exponentialRampToValueAtTime(0.001, now + 0.8)

    // High-shelf to add shimmer
    const shelf = this.ctx.createBiquadFilter()
    shelf.type = 'highshelf'
    shelf.frequency.value = 2000
    shelf.gain.value = 3

    carrier.connect(shelf)
    shelf.connect(env)
    env.connect(this.dryGain)
    env.connect(this.reverbSend) // heavy reverb on chime

    carrier.start(now)
    mod.start(now)
    carrier.stop(now + 1)
    mod.stop(now + 1)

    // Soft second partial (octave + fifth above)
    const p2 = this.ctx.createOscillator()
    p2.type = 'sine'
    p2.frequency.value = baseFreq * 3.01 // slightly detuned 3rd harmonic
    const p2Gain = this.ctx.createGain()
    p2Gain.gain.setValueAtTime(0, now)
    p2Gain.gain.linearRampToValueAtTime(0.01, now + 0.003)
    p2Gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.5)
    p2.connect(p2Gain)
    p2Gain.connect(this.dryGain)
    p2Gain.connect(this.reverbSend)
    p2.start(now)
    p2.stop(now + 0.6)
  }

  stopDrone() {
    if (!this.padGain || !this.ctx) return
    this.started = false
    const t = this.ctx.currentTime
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

// ═══════════════════════════════════════════════════════════════
// Neuropulse — Audio Engine
// Data-driven sonification: every sound maps to real GPU state.
//
// Drone pad pitch & filter = residual stream norm (how "excited"
//   the representation is). Low norm → low rumble; high → bright.
// Neuron ticks = attention entropy. High-entropy (confused) layers
//   produce dissonant intervals; low-entropy (certain) → consonant.
// Token chime pitch = top-1 probability. High confidence → bright
//   high bell; low confidence → low muted tone.
// ═══════════════════════════════════════════════════════════════

const STORAGE_KEY = 'neuropulse:muted'
const LEGACY_STORAGE_KEY = 'neural-pulse:muted'
const MASTER_LEVEL = 0.30
const RAMP_FAST = 0.12
const RAMP_SLOW = 0.55

// Harmonic series rooted on A2 (110 Hz). Consonant intervals only.
// Tick pitch is chosen from this based on attention entropy.
const HARMONIC_FREQS = [
  110.00,  // A2  — unison (high entropy / confused)
  130.81,  // C3  — minor third
  164.81,  // E3  — fifth
  220.00,  // A3  — octave
  261.63,  // C4  — tenth
  329.63,  // E4  — twelfth
  440.00,  // A4  — double octave (low entropy / certain)
]

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

  // Data state — fed from the real forward pass
  private residualNorm = 0      // 0..1, how large the residual stream is
  private lastEntropy = 0.5     // 0..1, attention entropy of the last layer processed
  private lastConfidence = 0    // 0..1, top-1 softmax probability

  constructor() {
    try {
      let v = localStorage.getItem(STORAGE_KEY)
      if (v === null) {
        const legacy = localStorage.getItem(LEGACY_STORAGE_KEY)
        if (legacy !== null) {
          localStorage.setItem(STORAGE_KEY, legacy)
          localStorage.removeItem(LEGACY_STORAGE_KEY)
          v = legacy
        }
      }
      this.muted = v === '1'
    } catch { /* localStorage unavailable */ }
  }

  isMuted(): boolean { return this.muted }

  init() {
    if (this.ctx) return
    this.ctx = new AudioContext({ sampleRate: 44100 })
    this.masterGain = this.ctx.createGain()
    this.masterGain.gain.value = this.muted ? 0 : MASTER_LEVEL
    this.masterGain.connect(this.ctx.destination)

    this.dryGain = this.ctx.createGain()
    this.dryGain.gain.value = 0.55
    this.dryGain.connect(this.masterGain)

    this.reverbGain = this.ctx.createGain()
    this.reverbGain.gain.value = 0.45
    this.reverbSend = this.createReverb()
    this.reverbSend.connect(this.reverbGain)
    this.reverbGain.connect(this.masterGain)
  }

  private createReverb(): ConvolverNode {
    const conv = this.ctx!.createConvolver()
    const rate = this.ctx!.sampleRate
    const length = rate * 3  // 3 second tail — cathedral-ish
    const impulse = this.ctx!.createBuffer(2, length, rate)

    for (let ch = 0; ch < 2; ch++) {
      const data = impulse.getChannelData(ch)
      for (let i = 0; i < length; i++) {
        const t = i / rate
        // Multi-stage decay: fast early reflections, slow diffuse tail
        const early = t < 0.06 ? (Math.random() - 0.5) * 0.6 * (1 - t / 0.06) : 0
        const late = (Math.random() - 0.5) * Math.exp(-t * 2.2) * 0.25
        data[i] = early + late
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

  // ─── Ambient drone ───
  // Detuned saws + sub sine + noise bed. The filter cutoff and pad pitch
  // are driven by residual stream norm — when the representation is large
  // (model is "activated"), the pad brightens and rises.
  startDrone() {
    if (!this.ctx || !this.dryGain || this.started) return
    this.started = true

    this.padGain = this.ctx.createGain()
    this.padGain.gain.value = 0

    this.padFilter = this.ctx.createBiquadFilter()
    this.padFilter.type = 'lowpass'
    this.padFilter.frequency.value = 80
    this.padFilter.Q.value = 0.6

    // 4 detuned oscillators — chorus warmth
    const base = 55 // A1
    const detunes = [-7, -2, 3, 8]
    for (const det of detunes) {
      const osc = this.ctx.createOscillator()
      osc.type = 'sawtooth'
      osc.frequency.value = base
      osc.detune.value = det
      const oscGain = this.ctx.createGain()
      oscGain.gain.value = 0.065
      osc.connect(oscGain)
      oscGain.connect(this.padFilter)
      osc.start()
      this.padOscs.push(osc)
    }

    // Sub bass
    const sub = this.ctx.createOscillator()
    sub.type = 'sine'
    sub.frequency.value = 27.5 // A0
    const subGain = this.ctx.createGain()
    subGain.gain.value = 0.09
    sub.connect(subGain)
    subGain.connect(this.padFilter)
    sub.start()
    this.padOscs.push(sub)

    // Filtered noise bed — adds texture
    this.noiseGain = this.ctx.createGain()
    this.noiseGain.gain.value = 0.005
    const noiseFilter = this.ctx.createBiquadFilter()
    noiseFilter.type = 'bandpass'
    noiseFilter.frequency.value = 200
    noiseFilter.Q.value = 0.4

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

    // Gentle 3s fade in
    this.padGain.gain.linearRampToValueAtTime(0.32, this.ctx.currentTime + 3)
  }

  // ─── Drone intensity driven by residual norm ───
  // v = 0..1 from the actual residual stream magnitude. Higher norm means
  // the model's internal representation is more "excited" — the pad opens
  // up, pitch rises slightly, noise bed thickens.
  setDroneIntensity(v: number) {
    if (!this.padGain || !this.padFilter || !this.noiseGain || !this.ctx) return
    this.residualNorm = v
    const t = this.ctx.currentTime + RAMP_FAST

    // Pad volume: quiet floor that swells with activation
    this.padGain.gain.linearRampToValueAtTime(0.18 + v * 0.28, t)
    // Filter: 80 Hz floor → up to 440 Hz when fully lit
    this.padFilter.frequency.linearRampToValueAtTime(80 + v * 360, t)
    // Noise bed tracks
    this.noiseGain.gain.linearRampToValueAtTime(0.005 + v * 0.015, t)
    // Pad pitch rises ~8 Hz at peak — subtle but perceptible shift
    if (this.padOscs[0]) {
      this.padOscs[0].frequency.linearRampToValueAtTime(55 + v * 8, t)
    }
  }

  // ─── Feed real attention entropy for the current layer ───
  // entropy: 0 = one token has all attention (certain), 1 = uniform (confused).
  // This controls the pitch of the next neuronTick.
  setAttentionEntropy(entropy: number) {
    this.lastEntropy = Math.max(0, Math.min(1, entropy))
  }

  // ─── Neural tick: tonal pluck whose pitch = attention certainty ───
  // High entropy (confused) → low dissonant tone.
  // Low entropy (attending sharply) → high consonant harmonic.
  neuronTick(layer: number) {
    if (!this.ctx || !this.dryGain || !this.reverbSend || this.muted) return
    const now = this.ctx.currentTime
    if (now - this.lastTickTime < 0.09) return
    this.lastTickTime = now
    this.tickCount++
    if (this.tickCount % 3 !== 0) return

    // Certainty = 1 - entropy. Map to harmonic series index.
    const certainty = 1 - this.lastEntropy
    const idx = Math.min(HARMONIC_FREQS.length - 1, Math.floor(certainty * HARMONIC_FREQS.length))
    const freq = HARMONIC_FREQS[idx]

    // Layer depth modulates octave — deeper layers are slightly higher,
    // giving the forward pass a rising contour.
    const octaveShift = layer >= 24 ? 2.0 : layer >= 16 ? 1.5 : 1.0
    const finalFreq = freq * octaveShift

    // Filtered noise burst = tonal "plink"
    const bufLen = Math.floor(this.ctx.sampleRate * 0.012)
    const buf = this.ctx.createBuffer(1, bufLen, this.ctx.sampleRate)
    const data = buf.getChannelData(0)
    for (let i = 0; i < bufLen; i++) {
      data[i] = (Math.random() * 2 - 1) * (1 - i / bufLen)
    }

    const src = this.ctx.createBufferSource()
    src.buffer = buf

    const bp = this.ctx.createBiquadFilter()
    bp.type = 'bandpass'
    bp.frequency.value = finalFreq
    bp.Q.value = 28 + certainty * 20 // sharper resonance when certain

    // Volume scales with certainty — uncertain ticks are quieter
    const vol = 0.025 + certainty * 0.035
    const gain = this.ctx.createGain()
    gain.gain.setValueAtTime(vol, now)
    gain.gain.exponentialRampToValueAtTime(0.0003, now + 0.16)

    src.connect(bp)
    bp.connect(gain)
    gain.connect(this.dryGain)
    gain.connect(this.reverbSend)
    src.start(now)
    src.stop(now + 0.18)
  }

  // ─── Feed top-1 confidence for the next token chime ───
  setTokenConfidence(confidence: number) {
    this.lastConfidence = Math.max(0, Math.min(1, confidence))
  }

  // ─── Token tick: short filtered click, pitch = confidence ───
  // Soft percussive tick per token. No bell, no sustain, no reverb ring.
  tokenChime() {
    if (!this.ctx || !this.dryGain || this.muted) return
    const now = this.ctx.currentTime
    if (now - this.lastChimeTime < 0.12) return
    this.lastChimeTime = now

    const c = this.lastConfidence

    // Short noise burst filtered to a pitch region
    const bufferSize = Math.floor(this.ctx.sampleRate * 0.03) // 30ms
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate)
    const data = buffer.getChannelData(0)
    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufferSize * 0.15))
    }

    const source = this.ctx.createBufferSource()
    source.buffer = buffer

    // Bandpass filter — confidence shifts the center frequency
    const bp = this.ctx.createBiquadFilter()
    bp.type = 'bandpass'
    bp.frequency.value = 800 + c * 1600 // 800–2400 Hz
    bp.Q.value = 2 + c * 4

    // Very short envelope
    const env = this.ctx.createGain()
    env.gain.setValueAtTime(0.015 + c * 0.012, now)
    env.gain.exponentialRampToValueAtTime(0.0001, now + 0.04 + c * 0.02)

    source.connect(bp)
    bp.connect(env)
    env.connect(this.dryGain)

    source.start(now)
    source.stop(now + 0.08)
  }

  stopDrone() {
    if (!this.padGain || !this.ctx) return
    this.started = false
    const t = this.ctx.currentTime

    // Fade out everything — pad drone + dry (chime) + reverb tail
    this.padGain.gain.cancelScheduledValues(t)
    this.padGain.gain.linearRampToValueAtTime(0, t + 1.5)
    this.dryGain?.gain.cancelScheduledValues(t)
    this.dryGain?.gain.linearRampToValueAtTime(0, t + 0.4)
    this.reverbGain?.gain.cancelScheduledValues(t)
    this.reverbGain?.gain.linearRampToValueAtTime(0, t + 0.8)

    setTimeout(() => {
      for (const osc of this.padOscs) {
        try { osc.stop() } catch { /* ok */ }
      }
      this.padOscs = []
      try { this.noiseNode?.stop() } catch { /* ok */ }
      this.noiseNode = null
      // Restore gains for next run
      if (this.dryGain) this.dryGain.gain.value = 1
      if (this.reverbGain) this.reverbGain.gain.value = 0.35
    }, 2000)
  }
}

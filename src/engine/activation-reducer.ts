/**
 * ACTIVATION REDUCER — f16→f32 conversion + role-specific dimensionality reduction
 *
 * Maps GPU activation buffers to per-neuron brightness values for
 * architecturally accurate visualization.
 */

/** IEEE 754 binary16 → float32 */
export function f16ToF32(f16arr: Uint16Array): Float32Array {
  const out = new Float32Array(f16arr.length)
  for (let i = 0; i < f16arr.length; i++) {
    const h = f16arr[i]
    const sign = (h >> 15) & 1
    const exp = (h >> 10) & 0x1f
    const frac = h & 0x3ff
    if (exp === 0) {
      out[i] = (sign ? -1 : 1) * (2 ** -14) * (frac / 1024)
    } else if (exp === 31) {
      out[i] = frac ? NaN : (sign ? -Infinity : Infinity)
    } else {
      out[i] = (sign ? -1 : 1) * (2 ** (exp - 15)) * (1 + frac / 1024)
    }
  }
  return out
}

/** Normalize + contrast enhance an array of RMS values to 0-1 */
function normalizeRMS(values: Float32Array): Float32Array {
  let max = 0, min = Infinity
  for (let i = 0; i < values.length; i++) {
    if (values[i] > max) max = values[i]
    if (values[i] < min) min = values[i]
  }
  const range = max - min
  const out = new Float32Array(values.length)
  if (range > 1e-8) {
    for (let i = 0; i < values.length; i++) {
      out[i] = Math.pow((values[i] - min) / range, 0.55)
    }
  } else {
    out.fill(0.5)
  }
  return out
}

/**
 * Reduce attention output to per-head activations.
 * Input: 3072 f32 values = 32 heads × 96 dims
 * Output: 32 values, normalized 0-1
 */
export function reduceForAttnHeads(raw: Float32Array): Float32Array {
  const HEADS = 32, HEAD_DIM = 96
  const rms = new Float32Array(HEADS)
  for (let h = 0; h < HEADS; h++) {
    const off = h * HEAD_DIM
    let sumSq = 0
    for (let j = 0; j < HEAD_DIM; j++) {
      const v = raw[off + j]
      sumSq += v * v
    }
    rms[h] = Math.sqrt(sumSq / HEAD_DIM)
  }
  return normalizeRMS(rms)
}

/**
 * Reduce QKV output to per-head activations (using Q portion only).
 * Input: 9216 f32 values = Q(3072) + K(3072) + V(3072)
 * Output: 32 values from Q, normalized 0-1
 */
export function reduceQKVForAttnHeads(raw: Float32Array): Float32Array {
  // Q is the first 3072 values
  return reduceForAttnHeads(raw.subarray(0, 3072))
}

/**
 * Reduce FFN intermediate to per-group activations.
 * Input: 8192 f32 values
 * Output: 16 values (groups of 512), normalized 0-1
 */
export function reduceForFFNGroups(raw: Float32Array): Float32Array {
  const GROUPS = 16, GROUP_SIZE = 512
  const rms = new Float32Array(GROUPS)
  for (let g = 0; g < GROUPS; g++) {
    const off = g * GROUP_SIZE
    let sumSq = 0
    for (let j = 0; j < GROUP_SIZE; j++) {
      const v = raw[off + j]
      sumSq += v * v
    }
    rms[g] = Math.sqrt(sumSq / GROUP_SIZE)
  }
  return normalizeRMS(rms)
}

/**
 * Reduce hidden state to residual stream scalar.
 * Input: 3072 f32 values
 * Output: single 0-1 value (normalized RMS)
 */
export function reduceForResidual(raw: Float32Array): number {
  let sumSq = 0
  for (let i = 0; i < raw.length; i++) {
    sumSq += raw[i] * raw[i]
  }
  // Return raw RMS — caller normalizes over time
  // Use tanh to soft-cap into 0-1 range
  const rms = Math.sqrt(sumSq / raw.length)
  return Math.tanh(rms * 0.5)
}

/** Generic fallback reducer */
export function reduceActivations(raw: Float32Array, neuronCount: number): Float32Array {
  const chunkSize = Math.floor(raw.length / neuronCount)
  if (chunkSize < 1) return new Float32Array(neuronCount).fill(0.5)
  const rms = new Float32Array(neuronCount)
  for (let i = 0; i < neuronCount; i++) {
    const off = i * chunkSize
    let sumSq = 0
    for (let j = 0; j < chunkSize; j++) {
      const v = raw[off + j]
      sumSq += v * v
    }
    rms[i] = Math.sqrt(sumSq / chunkSize)
  }
  return normalizeRMS(rms)
}

#!/usr/bin/env node
// verify-claims.mjs — empirical-lab gate.
//
// Greps every user-facing surface (README, landing essay, demo HTML, docs)
// for numeric claims about Phi-3 architecture, dispatch counts, kernel counts,
// and storage footprint, then asserts each match the canonical values in
// src/engine/compiler.ts (PHI3) and src/engine/phi3-facts.ts.
//
// Run via `npm run verify`. Exits non-zero on drift. Wired into `npm run check`.

import { readFile, readdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, join, relative } from 'node:path'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const rel = (p) => relative(ROOT, p)

// ──────────────────────────────────────────────────────────────────────
// Parse the canonical constants from source
// ──────────────────────────────────────────────────────────────────────

async function parseCompilerConstants() {
  const src = await readFile(join(ROOT, 'src/engine/compiler.ts'), 'utf8')
  const block = src.match(/export const PHI3 = \{([\s\S]*?)\} as const/)
  if (!block) throw new Error('PHI3 block not found in compiler.ts')
  const out = {}
  for (const m of block[1].matchAll(/(\w+):\s*(\d+)/g)) out[m[1]] = Number(m[2])
  return out
}

async function countShaders() {
  const dir = join(ROOT, 'src/engine/shaders')
  const files = await readdir(dir)
  return files.filter((f) => f.endsWith('.wgsl')).sort()
}

function deriveFacts(phi3, kernels) {
  const layerSteps = 9 // STEP_NAMES in inference.ts
  const prologue = 1   // embedding
  const epilogue = 3   // final rmsNorm + lm_head + argmax
  const lensLayers = 8 // LENS_LAYERS = [0,4,8,12,16,20,24,28]
  const lensSteps = 3  // rmsNorm + lm_head + argmax per lens
  const fastDispatches = layerSteps * phi3.LAYERS + prologue + epilogue
  const visualizedDispatches =
    fastDispatches + phi3.LAYERS /* attention_scores */ + lensLayers * lensSteps
  return {
    layers: phi3.LAYERS,
    heads: phi3.HEADS,
    headDim: phi3.HEAD_DIM,
    hiddenDim: phi3.D,
    ffnDim: phi3.FFN,
    vocab: phi3.VOCAB,
    qkvDim: phi3.QKV_DIM,
    maxContext: phi3.PAGE_SIZE * phi3.MAX_PAGES,
    kernels: kernels.length,
    fastDispatches,
    visualizedDispatches,
  }
}

// ──────────────────────────────────────────────────────────────────────
// Claim patterns — each pattern points at a fact and accepts a set of
// acceptable values. Some claims (dispatches) accept either fast or vis.
// ──────────────────────────────────────────────────────────────────────

// "32" or "32,064" or "1,024" — captures a number with optional thousand separators.
const N = '(\\d{1,3}(?:,\\d{3})+|\\d+)'

function buildChecks(F) {
  const num = (s) => Number(String(s).replace(/,/g, ''))
  return [
    {
      name: 'layers',
      // "32 layers" / "32 transformer layers". Excludes "N layer checkpoints"
      // (validation count) and "this layer", "next layer", etc.
      pattern: new RegExp(`${N}\\s+(?:transformer\\s+)?layers?\\b(?!\\s+checkpoint)`, 'gi'),
      accept: (v) => num(v) === F.layers,
      expected: `${F.layers}`,
    },
    {
      name: 'attention heads',
      // "32 attention heads" but NOT "1,024 attention heads" (that's heads × layers).
      pattern: new RegExp(`(?<!,)\\b(\\d{1,3})\\s+attention\\s+heads?\\b`, 'gi'),
      accept: (v) => num(v) === F.heads,
      expected: `${F.heads}`,
    },
    {
      name: 'heads × layers',
      // "1,024 attention heads" or "32 × 32 = 1,024" — the product.
      pattern: new RegExp(`${N}\\s+attention\\s+heads?\\b|=\\s*${N}\\s+(?:attention\\s+)?heads?\\b`, 'gi'),
      accept: (v) => num(v) === F.heads * F.layers || num(v) === F.heads,
      expected: `${F.heads * F.layers}`,
    },
    {
      name: 'vocab',
      pattern: /\b(32,?064)\b/g,
      accept: (v) => num(v) === F.vocab,
      expected: `${F.vocab.toLocaleString()}`,
    },
    {
      name: 'hidden dim',
      // Strictly: "<n>-dim residual/hidden". The residual/hidden anchor is
      // mandatory so we don't false-match "8,192 dims" (FFN inner dim).
      pattern: new RegExp(`${N}[\\- ]dim(?:ensional)?\\s+(?:residual|hidden\\s+state)`, 'gi'),
      accept: (v) => num(v) === F.hiddenDim,
      expected: `${F.hiddenDim}`,
    },
    {
      name: 'ffn dim',
      // "8,192-dim FFN" / "expands to 8,192".
      pattern: new RegExp(`${N}[\\- ]dim(?:ensional)?\\s+(?:FFN|MLP|inner)|expands?\\s+(?:residual\\s+)?to\\s+${N}\\s+dims?`, 'gi'),
      accept: (v) => num(v) === F.ffnDim,
      expected: `${F.ffnDim}`,
    },
    {
      name: 'kernels',
      pattern: new RegExp(`${N}\\s+(?:hand-written\\s+)?(?:WGSL\\s+)?(?:GPU\\s+)?kernels?\\b`, 'gi'),
      accept: (v) => num(v) === F.kernels,
      expected: `${F.kernels}`,
    },
    {
      name: 'dispatches',
      // "292 dispatches per token" — accept either fast or visualized.
      pattern: new RegExp(`\\b${N}\\s+(?:GPU\\s+)?dispatches?(?:\\s+per\\s+token)?\\b`, 'gi'),
      accept: (v) => num(v) === F.fastDispatches || num(v) === F.visualizedDispatches,
      expected: `${F.fastDispatches} (fast) or ${F.visualizedDispatches} (visualized)`,
    },
    {
      name: 'parameters (3.8B)',
      pattern: /(\d+(?:\.\d+)?)\s*billion\s+(?:parameters|params)\b/gi,
      accept: (v) => Math.abs(Number(v) - 3.8) < 0.05,
      expected: '3.8 billion',
    },
    {
      name: 'weight size GB',
      // Only flag when "GB" is paired with download/weights/model; ignore
      // generic "X GB VRAM" style mentions which are headroom claims.
      pattern: /[~≈]?\s*(\d+(?:\.\d+)?)\s*GB\s+(?:download|weights?|model)\b/gi,
      accept: (v) => {
        const n = Number(v)
        return n >= 1.8 && n <= 2.2
      },
      expected: '~2 GB',
    },
  ]
}

// ──────────────────────────────────────────────────────────────────────
// Surfaces to scan
// ──────────────────────────────────────────────────────────────────────

const SURFACES = [
  'README.md',
  'CLAUDE.md',
  'METHODS.md',
  'PREDICTIONS.md',
  'index.html',
  'app/index.html',
  'src/docs.md',
]

async function readSurface(p) {
  try {
    return { path: p, src: await readFile(join(ROOT, p), 'utf8') }
  } catch {
    return null // surface optional
  }
}

function* iterClaims(src, pattern) {
  pattern.lastIndex = 0
  let m
  while ((m = pattern.exec(src))) {
    // Find the first numeric capture group (skip undefined).
    const value = m.slice(1).find((g) => g !== undefined)
    if (value === undefined) continue
    const lineStart = src.lastIndexOf('\n', m.index) + 1
    const line = src.slice(lineStart, src.indexOf('\n', m.index))
    const lineNo = src.slice(0, m.index).split('\n').length
    yield { value, match: m[0], line: line.trim(), lineNo }
  }
}

// ──────────────────────────────────────────────────────────────────────
// Run
// ──────────────────────────────────────────────────────────────────────

async function main() {
  const phi3 = await parseCompilerConstants()
  const kernels = await countShaders()
  const F = deriveFacts(phi3, kernels)

  console.log('— neuropulse claim verification —')
  console.log(`  layers=${F.layers} heads=${F.heads} D=${F.hiddenDim} ffn=${F.ffnDim} vocab=${F.vocab}`)
  console.log(`  kernels=${F.kernels} (files: ${kernels.join(', ')})`)
  console.log(`  dispatches: fast=${F.fastDispatches}, visualized=${F.visualizedDispatches}`)
  console.log('')

  const checks = buildChecks(F)
  const failures = []

  for (const surface of SURFACES) {
    const got = await readSurface(surface)
    if (!got) continue
    for (const check of checks) {
      for (const hit of iterClaims(got.src, check.pattern)) {
        if (!check.accept(hit.value)) {
          failures.push({
            surface: rel(join(ROOT, surface)),
            line: hit.lineNo,
            check: check.name,
            found: hit.value,
            expected: check.expected,
            context: hit.line.slice(0, 140),
          })
        }
      }
    }
  }

  if (failures.length === 0) {
    console.log('✓ all claims match canonical facts')
    process.exit(0)
  }

  console.log(`✗ ${failures.length} drift${failures.length === 1 ? '' : 's'}:\n`)
  for (const f of failures) {
    console.log(`  ${f.surface}:${f.line}  [${f.check}] found ${JSON.stringify(f.found)}, expected ${f.expected}`)
    console.log(`    > ${f.context}`)
  }
  process.exit(1)
}

main().catch((e) => {
  console.error('verify-claims crashed:', e)
  process.exit(2)
})

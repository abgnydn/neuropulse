#!/usr/bin/env node
// Butterfly v3.9 — train a tagger on LongMemEval's has_answer labels.
//
// Previous trained classifiers (v3.5 hand-features, v3.6 embed) were
// trained on 100 messages from 4 hand-written transcripts and failed
// to generalize to LongMemEval (4.7% turn rate, vs regex's 29.0%).
// The criticism: we never trained on a representative distribution.
//
// This script fixes that. LongMemEval oracle's has_answer:true flags
// are direct supervision — "this turn is evidence for the question."
// 500 examples × ~22 turns = ~11K labeled turn-level examples.
//
// Training: binary softmax (keep vs not-keep) on the 14-feature vector.
// has_answer:true → keep. Everything else → not-keep.
// Class imbalance: ~900 keep vs ~10K not-keep (1:11). Handled by
// class-balanced loss weighting.
//
// At inference (in butterfly-longmemeval.mjs as STRATEGY=longmem-trained):
//   - Top tier: this classifier picks keep or not-keep
//   - For not-keep turns, regex's existing rules decide summarize vs melt
//   - Chrysalis assembles as before
//
// Test set: longmemeval_s (different examples than training). Clean
// held-out.

import { readFileSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const DATA_PATH = join(ROOT, 'test-results', 'longmemeval', 'longmemeval_oracle.json')
const WEIGHTS_PATH = join(ROOT, 'tools', 'butterfly-longmem-weights.json')
const DEBUG = process.env.DEBUG === '1'

const FEATURES = [
  { name: 'file_path',     f: t => /\b\w+\/[\w\-./]+\.(ts|js|py|md|sql|yaml|toml|json|tsx|jsx|go|rs|html|css)\b/.test(t) ? 1 : 0 },
  { name: 'ticket',        f: t => /TICKET-\d+/i.test(t) ? 1 : 0 },
  { name: 'channel',       f: t => /#[\w-]+/.test(t) ? 1 : 0 },
  { name: 'pkg_mention',   f: t => /@[\w-]+\/[\w-]+/.test(t) ? 1 : 0 },
  { name: 'line_range',    f: t => /\blines?\s+\d+(\s*[-–]\s*\d+)?\b/i.test(t) ? 1 : 0 },
  { name: 'decision_kw',   f: t => /\b(Decision|Root cause|Confirmed|Found it):/i.test(t) ? 1 : 0 },
  { name: 'code_call',     f: t => /\b\w+\.\w+\(\)/.test(t) ? 1 : 0 },
  { name: 'proper_name',   f: t => /\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/.test(t) ? 1 : 0 },
  { name: 'quantity',      f: t => /\b\d+\s*(req\/min|ms|s|min|gb|mb|kb|tokens?|seconds?|minutes?)\b/i.test(t) ? 1 : 0 },
  { name: 'inline_code',   f: t => /`[^`\n]{2,}`/.test(t) ? 1 : 0 },
  { name: 'bare_ack',      f: t => /^\s*(ok|lgtm|sure|thx|noted|got it|will do|cool|sweet|alright|yep|yes|no|nope)[\s,.!]*$/i.test(t) ? 1 : 0 },
  { name: 'topic_shift',   f: t => /(while you're (here|at it)|btw unrelated|side q|while we're here|unrelated|aside|off-topic)/i.test(t) ? 1 : 0 },
  { name: 'short_question',f: t => (t.endsWith('?') && t.length < 50 && !/\w+\/[\w.]+/.test(t)) ? 1 : 0 },
  { name: 'log_length',    f: t => Math.min(1.0, Math.log(t.length + 1) / 6.5) },
]
const F_DIM = FEATURES.length

function encodeFeatures(text) {
  return FEATURES.map(({ f }) => f(text))
}

// ─── build labeled dataset from oracle ────────────────────────────
function buildDataset() {
  const data = JSON.parse(readFileSync(DATA_PATH, 'utf8'))
  const examples = []
  for (const ex of data) {
    for (const session of ex.haystack_sessions) {
      for (const turn of session) {
        examples.push({
          text: turn.content,
          role: turn.role,
          label: turn.has_answer === true ? 1 : 0,  // 1 = keep, 0 = not-keep
          features: encodeFeatures(turn.content),
        })
      }
    }
  }
  return examples
}

// ─── balanced-loss binary softmax via gradient descent ────────────
function sigmoid(z) { return 1 / (1 + Math.exp(-z)) }

function trainBinary(dataset, { epochs = 2000, lr = 0.3, l2 = 1e-4 } = {}) {
  // Class-balanced weighting. The keep class is rare (~8%) so we
  // upweight it 10x so the model doesn't trivially predict 0 on
  // everything (which would give 92% accuracy and be useless).
  const nKeep = dataset.filter(d => d.label === 1).length
  const nNot  = dataset.length - nKeep
  const wKeep = nNot / nKeep
  const wNot  = 1
  console.log(`[train] dataset: ${dataset.length} turns · ${nKeep} keep (${(100*nKeep/dataset.length).toFixed(1)}%) · ${nNot} not-keep`)
  console.log(`[train] class weights: keep=${wKeep.toFixed(2)}, not-keep=${wNot.toFixed(2)}`)

  const W = Array(F_DIM).fill(0)
  let b = 0
  const N = dataset.length
  for (let epoch = 0; epoch < epochs; epoch++) {
    const gW = Array(F_DIM).fill(0)
    let gb = 0
    let loss = 0
    for (const { features: x, label: y } of dataset) {
      const z = W.reduce((s, w, j) => s + w * x[j], b)
      const p = sigmoid(z)
      const w = y === 1 ? wKeep : wNot
      // weighted binary cross-entropy
      loss += w * (-y * Math.log(p + 1e-12) - (1 - y) * Math.log(1 - p + 1e-12))
      const err = w * (p - y)
      gb += err
      for (let j = 0; j < F_DIM; j++) gW[j] += err * x[j]
    }
    for (let j = 0; j < F_DIM; j++) gW[j] += l2 * W[j]
    b -= (lr / N) * gb
    for (let j = 0; j < F_DIM; j++) W[j] -= (lr / N) * gW[j]
    if (DEBUG && (epoch < 5 || epoch % 200 === 0)) console.log(`  epoch ${epoch.toString().padStart(4)}  loss=${(loss / N).toFixed(4)}`)
  }
  return { W, b }
}

// ─── eval on training data — precision/recall for keep class ─────
function evalBinary(model, dataset, threshold = 0.5) {
  let tp = 0, fp = 0, tn = 0, fn = 0
  for (const { features: x, label: y } of dataset) {
    const z = model.W.reduce((s, w, j) => s + w * x[j], model.b)
    const p = sigmoid(z)
    const pred = p >= threshold ? 1 : 0
    if (pred === 1 && y === 1) tp++
    else if (pred === 1 && y === 0) fp++
    else if (pred === 0 && y === 0) tn++
    else fn++
  }
  const prec = tp / (tp + fp + 1e-12)
  const rec  = tp / (tp + fn + 1e-12)
  const f1   = 2 * prec * rec / (prec + rec + 1e-12)
  return { tp, fp, tn, fn, prec, rec, f1 }
}

function main() {
  console.log('[train] building dataset from longmemeval_oracle…')
  const dataset = buildDataset()

  console.log('[train] training binary classifier (14 features → keep / not-keep)…')
  const model = trainBinary(dataset)

  console.log('\n[train] eval at threshold=0.5 on training set:')
  const r5 = evalBinary(model, dataset, 0.5)
  console.log(`  tp=${r5.tp}  fp=${r5.fp}  tn=${r5.tn}  fn=${r5.fn}`)
  console.log(`  precision=${(r5.prec*100).toFixed(1)}%  recall=${(r5.rec*100).toFixed(1)}%  f1=${(r5.f1*100).toFixed(1)}%`)

  console.log('\n[train] eval at multiple thresholds:')
  for (const t of [0.3, 0.5, 0.7, 0.8, 0.9]) {
    const r = evalBinary(model, dataset, t)
    console.log(`  t=${t}: prec=${(r.prec*100).toFixed(1)}%  rec=${(r.rec*100).toFixed(1)}%  f1=${(r.f1*100).toFixed(1)}%  (kept=${r.tp+r.fp})`)
  }

  console.log('\n[train] learned weights:')
  for (let j = 0; j < F_DIM; j++) {
    console.log(`  ${FEATURES[j].name.padEnd(16)} ${model.W[j].toFixed(3).padStart(8)}`)
  }
  console.log(`  ${'(bias)'.padEnd(16)} ${model.b.toFixed(3).padStart(8)}`)

  const out = {
    version: 'v3.9-longmem-trained',
    feature_names: FEATURES.map(f => f.name),
    W: model.W,
    b: model.b,
    threshold: 0.5,
    training: { n: dataset.length, ...r5 },
  }
  writeFileSync(WEIGHTS_PATH, JSON.stringify(out, null, 2))
  console.log(`\nwrote: ${WEIGHTS_PATH}`)
}

main()

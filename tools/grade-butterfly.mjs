#!/usr/bin/env node
// Grade a butterfly-sweep result file against pre-registered thresholds
// (PREDICTIONS.md P-20260512-05). Usage:
//   node tools/grade-butterfly.mjs test-results/butterfly-sweep/<file>.json
//   node tools/grade-butterfly.mjs               (picks the newest file)

import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, dirname, basename } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const RESULTS_DIR = join(__dirname, '..', 'test-results', 'butterfly-sweep')

function pickInput(arg) {
  if (arg) return arg
  let files = []
  try {
    files = readdirSync(RESULTS_DIR)
      .filter(f => f.endsWith('.json'))
      .map(f => ({ f, m: statSync(join(RESULTS_DIR, f)).mtimeMs }))
      .sort((a, b) => b.m - a.m)
  } catch (_e) {
    throw new Error(`No result dir yet at ${RESULTS_DIR}. Run a sweep first: npm run sweep:butterfly  OR  paste tools/console-sweep.js into a neuropulse.live tab.`)
  }
  if (files.length === 0) throw new Error(`No result files in ${RESULTS_DIR}. Run a sweep first.`)
  return join(RESULTS_DIR, files[0].f)
}

const input = pickInput(process.argv[2])
const raw = JSON.parse(readFileSync(input, 'utf8'))
console.log(`\n— grading ${basename(input)} —`)
console.log(`runs: ${raw.runs.length}  ·  started: ${raw.started_at}  ·  finished: ${raw.finished_at}`)
if (raw.fingerprint?.sha) console.log(`build sha: ${raw.fingerprint.sha}`)
if (raw.fingerprint?.gpu) console.log(`gpu: ${raw.fingerprint.gpu}`)
console.log()

// ─── partition by transcript ──────────────────────────────────────
const byTranscript = new Map()
for (const r of raw.runs) {
  if (!byTranscript.has(r.transcript)) byTranscript.set(r.transcript, [])
  byTranscript.get(r.transcript).push(r)
}

const scoreOf = (v) => v === 'hit' ? 2 : v === 'partial' ? 1 : 0
const hits = (rows, arm) => rows.filter(r => scoreOf(r[arm]) >= 1).length
const fullHits = (rows, arm) => rows.filter(r => scoreOf(r[arm]) === 2).length

const table = []
for (const [tid, rows] of byTranscript) {
  const N = rows.length
  const bH = hits(rows, 'bfly')
  const lH = hits(rows, 'lastn')
  const bF = fullHits(rows, 'bfly')
  const lF = fullHits(rows, 'lastn')
  table.push({
    transcript: tid, N,
    bfly_hits: bH, lastn_hits: lH,
    bfly_full: bF, lastn_full: lF,
    bfly_rate: bH / N, lastn_rate: lH / N,
    delta: (bH - lH) / N,
  })
}

// ─── print per-transcript table ───────────────────────────────────
const pad = (s, w) => String(s).padEnd(w)
const fmt = (n) => (n * 100).toFixed(1) + '%'
console.log(pad('transcript', 28), pad('N', 4), pad('bfly  (full)', 14), pad('lastN (full)', 14), pad('Δ rate', 10))
console.log('─'.repeat(78))
for (const r of table) {
  console.log(
    pad(r.transcript, 28),
    pad(r.N, 4),
    pad(`${r.bfly_hits}/${r.N} (${r.bfly_full})`, 14),
    pad(`${r.lastn_hits}/${r.N} (${r.lastn_full})`, 14),
    pad((r.delta * 100).toFixed(1) + '%', 10),
  )
}
console.log()

// ─── apply P-20260512-05 thresholds ───────────────────────────────
// Confirm if for all 4 transcripts, bfly_hits/N ≥ lastn_hits/N + 0.15
//   AND on jwt-clock-race specifically bfly_hits ≥ 2 × lastn_hits.
// Refute if on any 2 of 4 transcripts, lastn_hits ≥ bfly_hits.
// Inconclusive otherwise.

const allBeatByMargin = table.every(r => r.delta >= 0.15)
const jwt = table.find(r => r.transcript === 'jwt-clock-race')
const jwtDoubles = jwt ? jwt.bfly_hits >= 2 * jwt.lastn_hits : false
const lastnWins = table.filter(r => r.lastn_hits >= r.bfly_hits).length

let verdict, reasoning
if (allBeatByMargin && jwtDoubles) {
  verdict = 'CONFIRMED'
  reasoning = `All transcripts cleared the +15pp margin; jwt-clock-race bfly_hits (${jwt.bfly_hits}) ≥ 2 × lastn_hits (${jwt.lastn_hits}).`
} else if (lastnWins >= 2) {
  verdict = 'REFUTED'
  reasoning = `LastN reached or exceeded butterfly on ${lastnWins}/${table.length} transcripts. Compaction failed to beat naive truncation.`
} else {
  verdict = 'INCONCLUSIVE'
  const fails = []
  if (!allBeatByMargin) fails.push(`not all transcripts cleared +15pp margin (${table.filter(r => r.delta < 0.15).map(r => r.transcript).join(', ')})`)
  if (!jwtDoubles && jwt) fails.push(`jwt-clock-race ratio ${jwt.bfly_hits}:${jwt.lastn_hits} (need ≥ 2:1)`)
  reasoning = fails.join(' · ')
}

console.log(`PRE-REGISTERED VERDICT: ${verdict}`)
console.log(`reasoning: ${reasoning}`)
console.log()

// ─── secondary signals (not pre-registered, exploratory) ──────────
console.log('— secondary signals (exploratory, NOT pre-registered) —')
const overall = {
  bfly_hits: table.reduce((s, r) => s + r.bfly_hits, 0),
  lastn_hits: table.reduce((s, r) => s + r.lastn_hits, 0),
  N: table.reduce((s, r) => s + r.N, 0),
}
console.log(`overall: bfly ${overall.bfly_hits}/${overall.N} (${fmt(overall.bfly_hits/overall.N)}) · lastN ${overall.lastn_hits}/${overall.N} (${fmt(overall.lastn_hits/overall.N)})`)
const meanT = raw.runs.reduce((s, r) => s + (r.seconds || 0), 0) / raw.runs.length
console.log(`mean per-run time: ${meanT.toFixed(1)}s`)
if (raw.page_errors?.length) console.log(`page errors during sweep: ${raw.page_errors.length}`)

// machine-readable summary at the end for piping into PREDICTIONS.md update
console.log('\n— machine-readable summary —')
console.log(JSON.stringify({ input: basename(input), verdict, table, overall, page_errors: raw.page_errors?.length || 0 }, null, 2))

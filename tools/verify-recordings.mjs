#!/usr/bin/env node
// verify-recordings.mjs — validates every committed demo-mode recording.
//
// Schema twin of src/recording.ts (NpRecording v1). Tools are plain node, so
// the shape is duplicated here on purpose — bump RECORDING_SCHEMA_VERSION in
// both files together on breaking changes.
//
// Wired into `npm run verify`. Exits non-zero on any invalid recording so a
// corrupt or oversized file can never ship to demo mode silently.

import { readFile, readdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const DIR = join(ROOT, 'public/recordings')

const SCHEMA_VERSION = 1
const LAYERS = 32
const HEADS = 32
const LENS_LAYERS = new Set([0, 4, 8, 12, 16, 20, 24, 28])
const MAX_BYTES = 300 * 1024
const MIN_TOKENS = 10

const fail = (file, msg) => ({ file, msg })

function b64Len(s) {
  // decoded byte length of a base64 string
  if (typeof s !== 'string' || s.length === 0) return 0
  const pad = s.endsWith('==') ? 2 : s.endsWith('=') ? 1 : 0
  return (s.length * 3) / 4 - pad
}

function isFiniteArray(a, len) {
  return Array.isArray(a) && a.length === len && a.every((v) => Number.isFinite(v))
}

async function validate(file, buf) {
  const errs = []
  if (buf.byteLength > MAX_BYTES) {
    errs.push(fail(file, `file is ${(buf.byteLength / 1024).toFixed(0)} KB > ${MAX_BYTES / 1024} KB cap`))
  }
  let rec
  try {
    rec = JSON.parse(buf.toString('utf8'))
  } catch (e) {
    return [fail(file, `not valid JSON: ${e.message}`)]
  }

  if (rec.schemaVersion !== SCHEMA_VERSION) errs.push(fail(file, `schemaVersion ${rec.schemaVersion} ≠ ${SCHEMA_VERSION}`))
  if (typeof rec.prompt !== 'string' || rec.prompt.trim() === '') errs.push(fail(file, 'empty prompt'))
  if (rec.mode !== 'ask' && rec.mode !== 'complete') errs.push(fail(file, `bad mode ${rec.mode}`))
  if (!Array.isArray(rec.prefillTokens) || rec.prefillTokens.length === 0) errs.push(fail(file, 'empty prefillTokens'))
  if (!Number.isInteger(rec.kvTotalPages) || rec.kvTotalPages <= 0) errs.push(fail(file, `bad kvTotalPages ${rec.kvTotalPages}`))
  if (typeof rec.engineFingerprint !== 'string' || !rec.engineFingerprint) errs.push(fail(file, 'missing engineFingerprint'))
  if (!Array.isArray(rec.tokens) || rec.tokens.length < MIN_TOKENS) {
    errs.push(fail(file, `only ${rec.tokens?.length ?? 0} tokens (< ${MIN_TOKENS})`))
    return errs
  }

  let prevKvLen = 0
  rec.tokens.forEach((t, i) => {
    const at = `tokens[${i}]`
    if (typeof t.text !== 'string') errs.push(fail(file, `${at}.text missing`))
    if (!Number.isInteger(t.id)) errs.push(fail(file, `${at}.id missing`))
    if (!Array.isArray(t.topK) || t.topK.length !== 5) errs.push(fail(file, `${at}.topK length ${t.topK?.length} ≠ 5`))
    else {
      const probs = t.topK.map((k) => k.p)
      if (!probs.every((p) => Number.isFinite(p) && p >= 0 && p <= 1)) errs.push(fail(file, `${at}.topK probs out of [0,1]`))
      for (let j = 1; j < probs.length; j++) {
        if (probs[j] > probs[j - 1] + 1e-9) { errs.push(fail(file, `${at}.topK probs not descending`)); break }
      }
    }
    if (!isFiniteArray(t.residualNorms, LAYERS)) errs.push(fail(file, `${at}.residualNorms not ${LAYERS} finite numbers`))
    if (!isFiniteArray(t.layerDeltas, LAYERS)) errs.push(fail(file, `${at}.layerDeltas not ${LAYERS} finite numbers`))
    if (b64Len(t.headActivity) !== LAYERS * HEADS) errs.push(fail(file, `${at}.headActivity decodes to ${b64Len(t.headActivity)} ≠ ${LAYERS * HEADS} bytes`))
    if (!Number.isFinite(t.headActivityScale) || t.headActivityScale < 0) errs.push(fail(file, `${at}.headActivityScale bad`))
    if (!Array.isArray(t.lens)) errs.push(fail(file, `${at}.lens missing`))
    else for (const l of t.lens) {
      if (!LENS_LAYERS.has(l.L)) errs.push(fail(file, `${at}.lens layer ${l.L} ∉ LENS_LAYERS`))
    }
    if (!Number.isInteger(t.kvLen) || t.kvLen <= 0) errs.push(fail(file, `${at}.kvLen bad`))
    else {
      if (t.kvLen < prevKvLen) errs.push(fail(file, `${at}.kvLen ${t.kvLen} < previous ${prevKvLen} (not monotonic)`))
      prevKvLen = t.kvLen
      if (t.attnL31 !== '' && b64Len(t.attnL31) !== HEADS * t.kvLen) {
        errs.push(fail(file, `${at}.attnL31 decodes to ${b64Len(t.attnL31)} ≠ ${HEADS * t.kvLen} bytes (32 × kvLen)`))
      }
    }
    if (!Number.isInteger(t.kvUsedPages) || t.kvUsedPages < 0) errs.push(fail(file, `${at}.kvUsedPages bad`))
  })
  return errs
}

async function main() {
  let files = []
  try {
    files = (await readdir(DIR)).filter((f) => f.endsWith('.json')).sort()
  } catch {
    console.log('— recordings: public/recordings/ absent — nothing to validate')
    process.exit(0)
  }
  if (files.length === 0) {
    console.log('— recordings: none committed — nothing to validate')
    process.exit(0)
  }

  console.log('— recording validation —')
  const allErrs = []
  for (const f of files) {
    const buf = await readFile(join(DIR, f))
    const rec = JSON.parse(buf.toString('utf8'))
    const errs = await validate(f, buf)
    allErrs.push(...errs)
    if (errs.length === 0) {
      console.log(`  ✓ ${f} — ${rec.tokens.length} tokens, ${(buf.byteLength / 1024).toFixed(0)} KB, prompt "${String(rec.prompt).slice(0, 40)}"`)
    }
  }
  if (allErrs.length) {
    console.log(`✗ ${allErrs.length} problem${allErrs.length === 1 ? '' : 's'}:`)
    for (const e of allErrs) console.log(`  ${e.file}: ${e.msg}`)
    process.exit(1)
  }
  console.log('✓ all recordings valid')
}

main().catch((e) => {
  console.error('verify-recordings crashed:', e)
  process.exit(2)
})

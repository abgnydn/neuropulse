#!/usr/bin/env node
// check-shortcuts.mjs — empirical-lab gate.
//
// Parses the keys actually wired to the keydown handlers in src/main.ts and
// src/journey.ts, then parses the keys advertised in the Journey HUD hint
// strip + glossary controls grid in app/index.html. Reports:
//
//   * "documented but not wired" — copy says the key works; code disagrees.
//   * "wired but not documented" — code listens for the key; copy says nothing.
//
// Skips numeric / question-mark / Enter / arrow keys to avoid false flags
// on input-field handlers vs. global mode shortcuts.

import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))

// ──────────────────────────────────────────────────────────────────────
// Wired keys — extract from keydown handler bodies
// ──────────────────────────────────────────────────────────────────────

async function extractWiredKeys() {
  const wired = new Map() // key → set of source files
  const sources = ['src/main.ts', 'src/journey.ts']
  for (const file of sources) {
    const src = await readFile(join(ROOT, file), 'utf8')
    // Match `e.key === 'X'` and `e.key === "X"` in addEventListener('keydown', …) blocks.
    // We don't try to scope to keydown blocks specifically — every e.key check
    // is a key the app cares about.
    for (const m of src.matchAll(/e\.key\s*===\s*['"]([^'"]+)['"]/g)) {
      const k = normalizeKey(m[1])
      if (!k) continue
      if (!wired.has(k)) wired.set(k, new Set())
      wired.get(k).add(file)
    }
  }
  return wired
}

function normalizeKey(k) {
  if (!k) return null
  if (k === ' ') return 'space'
  if (k.toLowerCase() === 'esc') return 'Escape'
  if (k.length === 1) return k.toUpperCase()
  // Map ArrowUp/Down/Left/Right → "arrow"
  if (/^Arrow/.test(k)) return 'arrow'
  return k
}

// ──────────────────────────────────────────────────────────────────────
// Documented keys — extract from app/index.html
// ──────────────────────────────────────────────────────────────────────

async function extractDocumentedKeys() {
  const html = await readFile(join(ROOT, 'app/index.html'), 'utf8')
  const documented = new Map() // key → array of locations
  const add = (key, where) => {
    const k = normalizeKey(key) ?? key
    if (!documented.has(k)) documented.set(k, [])
    documented.get(k).push(where)
  }

  // 1. Glossary controls grid: <kbd>X</kbd>
  for (const m of html.matchAll(/<kbd>([^<]+)<\/kbd>/g)) {
    const inner = m[1].trim()
    // Possibly compound like "P" / "Tab" or "↑ ↓ ← →"
    if (/^[↑↓←→\s]+$/.test(inner)) {
      add('arrow', 'glossary-controls')
      continue
    }
    if (inner === 'space') { add('space', 'glossary-controls'); continue }
    if (inner === 'drag' || inner === 'wheel' || inner === 'right-drag') continue
    // Split on " / " or "+"
    for (const part of inner.split(/\s*\/\s*|\s*\+\s*/)) {
      const p = part.trim()
      if (!p) continue
      add(p, 'glossary-controls')
    }
  }

  // 2. Journey HUD hint strip: <span>X label</span>. Scope strictly to the
  // hint strip so we don't false-match copy like "<span>KV Cache panel</span>".
  const hintBlock = html.match(/<div\s+class="journey-hint"[\s\S]*?<\/div>/)
  if (hintBlock) {
    for (const m of hintBlock[0].matchAll(/<span>([^<]+)<\/span>/g)) {
      const inner = m[1].trim()
      if (inner === '·') continue
      // Each hint is "<token> <label>" — first token is the shortcut.
      const first = inner.split(/\s+/)[0]
      if (/^[↑↓←→]+$/.test(first)) { add('arrow', 'journey-hint'); continue }
      if (first === 'space') { add('space', 'journey-hint'); continue }
      // Only treat single-letter tokens as keys (rejects "drag", "wheel",
      // "Tour" etc. that have lowercase multi-char first words).
      if (/^[A-Z]$/.test(first)) add(first, 'journey-hint')
    }
  }

  return documented
}

// ──────────────────────────────────────────────────────────────────────
// Run
// ──────────────────────────────────────────────────────────────────────

// Keys we never expect to be wired/documented at the global level (input
// fields, tour navigation, dialog dismiss). Matches our normalizeKey output.
// '/' is the Shift-less form of '?' (the glossary trigger) — same handler.
const ALLOWLIST_WIRED_NOT_DOC = new Set(['Enter', 'Escape', '/'])
// Keys mentioned in copy that are intentionally context-scoped (we surface
// them as "Journey only", which is fine).
const ALLOWLIST_DOC_NOT_WIRED = new Set([])

async function main() {
  const wired = await extractWiredKeys()
  const documented = await extractDocumentedKeys()

  console.log('— shortcut consistency check —')
  console.log(`  wired:      ${[...wired.keys()].sort().join(', ') || '(none)'}`)
  console.log(`  documented: ${[...documented.keys()].sort().join(', ') || '(none)'}`)
  console.log('')

  const failures = []

  for (const [k, locs] of documented) {
    if (!wired.has(k) && !ALLOWLIST_DOC_NOT_WIRED.has(k)) {
      failures.push(`✗ "${k}" documented in [${locs.join(', ')}] but NOT wired in any keydown handler`)
    }
  }
  for (const [k, files] of wired) {
    if (!documented.has(k) && !ALLOWLIST_WIRED_NOT_DOC.has(k)) {
      failures.push(`⚠ "${k}" wired in [${[...files].join(', ')}] but NOT documented in glossary or HUD`)
    }
  }

  if (failures.length === 0) {
    console.log('✓ keyboard shortcuts: code and copy agree')
    process.exit(0)
  }

  for (const f of failures) console.log('  ' + f)
  // Treat orphan-wired (no docs) as warning, not failure. Treat doc-without-
  // wiring as a hard failure — that's the "scroll advances Journey" class.
  const hardFailures = failures.filter((s) => s.startsWith('✗'))
  process.exit(hardFailures.length > 0 ? 1 : 0)
}

main().catch((e) => {
  console.error('check-shortcuts crashed:', e)
  process.exit(2)
})

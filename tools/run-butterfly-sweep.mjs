#!/usr/bin/env node
// One-command Butterfly v2.5 sweep driver.
//   npm run sweep:butterfly             → 4 transcripts × 20 runs (~60 min)
//   RUNS_PER=1 npm run sweep:butterfly  → smoke (~3-5 min)
//
// Starts vite on :4000, runs the Playwright sweep against headed Chrome
// (programmatic-launch + headless Chrome do not expose navigator.gpu on
// macOS — a real window must appear), tears the server down, then grades
// the newest result file against the pre-registered thresholds in
// PREDICTIONS.md P-20260512-05.
//
// Output: test-results/butterfly-sweep/butterfly-sweep-<ts>.json
//         + a grader summary printed to stdout

import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const VITE_PORT = 4000

function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: 'inherit', cwd: ROOT, ...opts })
    child.on('exit', (code) => code === 0 ? resolve() : reject(new Error(`${cmd} exited with code ${code}`)))
    child.on('error', reject)
  })
}

function startVite() {
  return new Promise((resolve, reject) => {
    const child = spawn('npx', ['vite', '--port', String(VITE_PORT), '--host', '127.0.0.1', '--strictPort'], {
      cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'],
    })
    let buf = ''
    const onData = (chunk) => {
      const s = chunk.toString()
      buf += s
      process.stdout.write(`[vite] ${s}`)
      if (s.includes('Local:') || s.includes(`localhost:${VITE_PORT}`)) {
        resolve(child)
      }
    }
    child.stdout.on('data', onData)
    child.stderr.on('data', onData)
    child.on('exit', (code) => {
      if (code !== 0) reject(new Error(`vite exited early: ${buf.slice(-400)}`))
    })
    setTimeout(() => reject(new Error('vite did not become ready in 30s')), 30_000)
  })
}

async function main() {
  console.log('\n[sweep] starting vite on 127.0.0.1:' + VITE_PORT + '…')
  const vite = await startVite()
  console.log('[sweep] vite ready')

  try {
    console.log('[sweep] running playwright sweep (headed Chrome will appear)…')
    await run('npx', ['playwright', 'test', '--config=playwright.butterfly.config.mjs', 'butterfly-sweep'])
    console.log('\n[sweep] playwright completed — grading newest result file…\n')
    await run('node', ['tools/grade-butterfly.mjs'])
  } finally {
    console.log('\n[sweep] stopping vite…')
    vite.kill()
  }
}

main().catch((err) => {
  console.error('\n[sweep] FAILED:', err.message)
  process.exit(1)
})

#!/usr/bin/env node
// Butterfly v3.10 — cross-domain test.
//
// The longmem-trained classifier learned that LongMemEval evidence is
// SHORT user statements ("I graduated with Business Administration").
// Its learned weights are negative on identifier features (file_path,
// decision_kw, proper_name) — the OPPOSITE of regex's hand-tuned prior.
//
// Question: does the longmem-trained classifier still work on our
// original 4 hand-written engineering transcripts (where the needles
// ARE identifier-shaped: lib/jwt.ts, #auth-platform, etc.)?
//
// If yes: the architecture is more flexible than the weights suggest.
// If no: the trained classifier is domain-locked, which means a real
//   production deployment would need ONE classifier per domain (or a
//   bigger / multi-domain training set).

import { readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const TARGET_TOKENS = 100
const N_GENERATIONS = 3
const tokens = (s) => Math.ceil(s.length / 4)

const FEAT = [
  t => /\b\w+\/[\w\-./]+\.(ts|js|py|md|sql|yaml|toml|json|tsx|jsx|go|rs|html|css)\b/.test(t) ? 1 : 0,
  t => /TICKET-\d+/i.test(t) ? 1 : 0,
  t => /#[\w-]+/.test(t) ? 1 : 0,
  t => /@[\w-]+\/[\w-]+/.test(t) ? 1 : 0,
  t => /\blines?\s+\d+(\s*[-–]\s*\d+)?\b/i.test(t) ? 1 : 0,
  t => /\b(Decision|Root cause|Confirmed|Found it):/i.test(t) ? 1 : 0,
  t => /\b\w+\.\w+\(\)/.test(t) ? 1 : 0,
  t => /\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/.test(t) ? 1 : 0,
  t => /\b\d+\s*(req\/min|ms|s|min|gb|mb|kb|tokens?|seconds?|minutes?)\b/i.test(t) ? 1 : 0,
  t => /`[^`\n]{2,}`/.test(t) ? 1 : 0,
  t => /^\s*(ok|lgtm|sure|thx|noted|got it|will do|cool|sweet|alright|yep|yes|no|nope)[\s,.!]*$/i.test(t) ? 1 : 0,
  t => /(while you're (here|at it)|btw unrelated|side q|while we're here|unrelated|aside|off-topic)/i.test(t) ? 1 : 0,
  t => (t.endsWith('?') && t.length < 50 && !/\w+\/[\w.]+/.test(t)) ? 1 : 0,
  t => Math.min(1.0, Math.log(t.length + 1) / 6.5),
]
const sigmoid = (z) => 1 / (1 + Math.exp(-z))

// Load both trained classifiers for the comparison
const inDomainW = JSON.parse(readFileSync(join(ROOT, 'tools', 'butterfly-classifier-weights.json'), 'utf8'))
const longmemW = JSON.parse(readFileSync(join(ROOT, 'tools', 'butterfly-longmem-weights.json'), 'utf8'))

function softmax(z) { const m = Math.max(...z); const e = z.map(v => Math.exp(v-m)); const s = e.reduce((a,b)=>a+b,0); return e.map(v=>v/s) }

// In-domain trained: 3-class softmax (keep/summarize/melt)
function inDomainTag(text) {
  const x = FEAT.map(f => f(text))
  const z = inDomainW.W.map((wk, k) => wk.reduce((s, wv, j) => s + wv * x[j], inDomainW.b[k]))
  const p = softmax(z)
  let argmax = 0
  for (let k = 1; k < p.length; k++) if (p[k] > p[argmax]) argmax = k
  return inDomainW.classes[argmax]
}

// Regex tagger
function regexTag(text) {
  let score = 0
  if (/\b\w+\/[\w\-./]+\.(ts|js|py|md|sql|yaml|toml|json|tsx|jsx|go|rs|html|css)\b/.test(text)) score += 3
  if (/TICKET-\d+/i.test(text)) score += 3
  if (/#[\w-]+/.test(text)) score += 3
  if (/@[\w-]+\/[\w-]+/.test(text)) score += 3
  if (/\blines?\s+\d+(\s*[-–]\s*\d+)?\b/i.test(text)) score += 3
  if (/\b(Decision|Root cause|Confirmed|Found it):/i.test(text)) score += 3
  if (/\b\w+\.\w+\(\)/.test(text)) score += 2
  if (/\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/.test(text)) score += 2
  if (/\b\d+\s*(req\/min|ms|s|min|gb|mb|kb|tokens?|seconds?|minutes?)\b/i.test(text)) score += 2
  if (/`[^`\n]{2,}`/.test(text)) score += 1
  if (/^\s*(ok|lgtm|sure|thx|noted|got it|will do|cool|sweet|alright|yep|yes|no|nope)[\s,.!]*$/i.test(text)) score -= 4
  if (/^\s*(ok|lgtm|sure|thx|noted|got it)\b.{0,30}$/i.test(text) && text.length < 50) score -= 2
  if (/(while you're (here|at it)|btw unrelated|side q|while we're here|unrelated|aside|off-topic)/i.test(text)) score -= 1
  if (text.endsWith('?') && text.length < 50 && !/\w+\/[\w.]+/.test(text)) score -= 1
  if (text.length < 40) score -= 1
  return score >= 3 ? 'keep' : score >= 1 ? 'summarize' : 'melt'
}

// LongMem-trained: binary, demote regex's keep to summarize for not-keeps
function longmemTag(text) {
  const x = FEAT.map(f => f(text))
  const z = longmemW.W.reduce((s, wv, j) => s + wv * x[j], longmemW.b)
  if (sigmoid(z) >= longmemW.threshold) return 'keep'
  const rt = regexTag(text)
  return rt === 'keep' ? 'summarize' : rt
}

// ─── transcripts (mirror of butterfly-purecode-hard.mjs) ─────────
const CORES = [
  {
    id: 'jwt-clock-race',
    needle_keywords: ['issuetoken', 'lib/jwt.ts', 'date.now()', 'exp', 'two reads'],
    core: [
      { role: 'user',      content: "Our auth/session.test.ts is flaky in CI but passes locally. Help?" },
      { role: 'assistant', content: "Share the test output." },
      { role: 'user',      content: "It asserts decoded.exp equals Math.floor(Date.now()/1000) + 3600. CI says expected 1735689600 received 1735689599." },
      { role: 'assistant', content: "Off-by-one second. Timing race rather than logic bug." },
      { role: 'assistant', content: "Confirmed in lib/jwt.ts: issueToken reads Date.now() once for `exp`, the test reads Date.now() again in the assertion. On slow CI those two reads land on different seconds. Root cause: clock race." },
      { role: 'user',      content: "ok so the fix?" },
      { role: 'assistant', content: "Capture Date.now() once before calling issueToken, or have issueToken accept an optional `now` parameter so tests inject a fixed clock." },
      { role: 'user',      content: "let's do the proper one." },
    ],
  },
  {
    id: 'auth-owner-pto',
    needle_keywords: ['marcus lee', '#auth-platform', 'sarah chen', '@company/auth-service'],
    core: [
      { role: 'user',      content: "We're shipping the new checkout API Friday. Need someone on auth-service to bless the JWT format change." },
      { role: 'assistant', content: "Let me pull their owner." },
      { role: 'user',      content: "Slack channel works too if you have it." },
      { role: 'assistant', content: "Sarah Chen owns @company/auth-service — Slack #auth-platform. She's out through Tuesday, backup reviewer is Marcus Lee." },
      { role: 'user',      content: "noted. who do we ping for billing-service?" },
      { role: 'assistant', content: "Don, channel #billing-eng." },
      { role: 'user',      content: "and payment-svc — needs a review?" },
      { role: 'assistant', content: "No, payment-svc only proxies. No review needed." },
    ],
  },
  {
    id: 'rate-limit-decision',
    needle_keywords: ['token bucket', '100 req/min', 'redis', 'fail-open', 'fixed-window', 'ticket-4421'],
    core: [
      { role: 'user',      content: "We need rate limiting on /api/v1. What should we use?" },
      { role: 'assistant', content: "Token bucket or fixed window are the usual options." },
      { role: 'user',      content: "Which one — we have spiky traffic." },
      { role: 'assistant', content: "Token bucket handles bursts. Suggest 100 req/min per IP, refill every 600ms." },
      { role: 'assistant', content: "Decision: token bucket at 100 req/min per IP via Redis; if Redis is unreachable for >5s, fail-open with a local in-memory fixed-window of 50 req/min. Tracked under TICKET-4421." },
      { role: 'user',      content: "lgtm. you'll draft the RFC?" },
      { role: 'assistant', content: "Yes, by Wednesday." },
    ],
  },
  {
    id: 'cache-race-fileline',
    needle_keywords: ['lib/cache.ts', '142-148', 'cache.set', 'cache.get', 'version'],
    core: [
      { role: 'user',      content: "Cache occasionally returns stale data right after a write. Where is it?" },
      { role: 'assistant', content: "Do reads go through the same lock as writes?" },
      { role: 'user',      content: "no, reads bypass the lock. only writes hold it." },
      { role: 'assistant', content: "Found it: lib/cache.ts lines 142-148. cache.set updates the value, but the version-counter increment happens 2 statements later. A concurrent cache.get between sees the new value but the old version, downstream treats as still-cached." },
      { role: 'user',      content: "so version increment needs to be atomic with set?" },
      { role: 'assistant', content: "Yes. Move the version++ into the same critical section, or CAS on version." },
    ],
  },
]

const PADDING_POOL = [
  { role: 'user',      content: "hey morning. got time for a quick one today?" },
  { role: 'assistant', content: "sure thing — fire away when you're ready." },
  { role: 'user',      content: "first off — is the staging cluster still on the old node version?" },
  { role: 'assistant', content: "i think so, qa pushed back on the upgrade last week" },
  { role: 'user',      content: "lol. also did finance approve the q3 tooling budget?" },
  { role: 'assistant', content: "i'll ask ramesh tomorrow when he's back from leave" },
  { role: 'user',      content: "thx. one more — anyone owning the docs migration?" },
  { role: 'assistant', content: "rasmus i think, but he's on parental leave til month-end" },
  { role: 'user',      content: "ah right. ok and re: lunch — ramen place near the office?" },
  { role: 'assistant', content: "yes please. heard they raised prices, fyi." },
  { role: 'user',      content: "everyone has. anyway, conf cfp deadline is friday — you submitting?" },
  { role: 'assistant', content: "probably. still picking between two talk angles." },
  { role: 'user',      content: "cool. meeting with platform team got moved to thursday 2pm" },
  { role: 'assistant', content: "noted, on the calendar." },
  { role: 'user',      content: "before i forget — the new sprint cadence retro is next week" },
]
const NOISE_PER_GEN = [
  [
    { role: 'user',      content: "side q: should we use prettier or biome for formatting?" },
    { role: 'assistant', content: "biome is faster, prettier has wider plugins. Either works." },
    { role: 'user',      content: "punt on it. also CI build time crept up to 4 min." },
    { role: 'assistant', content: "Probably the snapshot suite. Split next sprint." },
  ],
  [
    { role: 'user',      content: "did anyone actually try the new prod metrics dashboard?" },
    { role: 'assistant', content: "i poked it briefly. ui's nice, query latency is rough." },
    { role: 'user',      content: "ok will file feedback. unrelated — oncall handoff doc?" },
    { role: 'assistant', content: "not yet. draft after the sprint review." },
  ],
  [
    { role: 'user',      content: "design system v3 dropped, anyone migrating yet?" },
    { role: 'assistant', content: "marketing site so far. infra apps lagging." },
    { role: 'user',      content: "fair. legal sign off on the new privacy policy?" },
    { role: 'assistant', content: "still in review. maybe end of next week." },
  ],
]

function firstSentence(s) { const m = s.match(/^[^.!?]*[.!?]/); return (m ? m[0] : s).trim() }
function chrysalis(turns, tags, budget) {
  const parts = []
  for (let i = 0; i < turns.length; i++) {
    const t = tags[i], m = turns[i]
    if (t === 'keep') parts.push(`[${m.role}] ${m.content}`)
    else if (t === 'summarize') parts.push(`[${m.role}] ${firstSentence(m.content)}`)
  }
  let text = parts.join('\n')
  const maxChars = budget * 4
  if (text.length > maxChars) text = text.slice(0, maxChars).replace(/\s\S*$/, '') + '…'
  return text
}
function pickLastN(turns, budget) {
  const out = []; let acc = 0
  for (let i = turns.length - 1; i >= 0; i--) {
    const t = tokens(turns[i].content)
    if (acc + t > budget && out.length > 0) break
    out.unshift(turns[i]); acc += t
  }
  return out.map(m => `[${m.role}] ${m.content}`).join('\n')
}
function scoreKW(memory, keywords) {
  const lower = memory.toLowerCase()
  return keywords.filter(k => lower.includes(k.toLowerCase())).length / keywords.length
}

// Multi-gen butterfly (same as v3.1 hard regime)
function runOne(core, tagFn) {
  let bflyMessages = [...PADDING_POOL, ...core.core, ...PADDING_POOL.slice().reverse()].slice()
  let lastnMessages = bflyMessages.slice()
  let bflyFinal = ''
  let lastNFinal = ''

  for (let g = 1; g <= N_GENERATIONS; g++) {
    const tags = bflyMessages.map(m => tagFn(m.content))
    bflyFinal = chrysalis(bflyMessages, tags, TARGET_TOKENS)
    lastNFinal = pickLastN(lastnMessages, TARGET_TOKENS)
    if (g < N_GENERATIONS) {
      const noise = NOISE_PER_GEN[(g - 1) % NOISE_PER_GEN.length]
      bflyMessages = [{ role: 'assistant', content: `[REBUILT FROM GEN ${g}]\n${bflyFinal}` }, ...noise]
      // approximate lastN: keep last few + noise
      const truncated = []
      let acc = 0
      for (let i = lastnMessages.length - 1; i >= 0; i--) {
        const t = tokens(lastnMessages[i].content)
        if (acc + t > TARGET_TOKENS && truncated.length > 0) break
        truncated.unshift(lastnMessages[i]); acc += t
      }
      lastnMessages = [...truncated, ...noise]
    }
  }
  return {
    bfly: scoreKW(bflyFinal, core.needle_keywords),
    lastn: scoreKW(lastNFinal, core.needle_keywords),
  }
}

// ─── run all 3 taggers on all 4 transcripts ──────────────────────
console.log('[crossdomain] hard regime: budget=100, gens=3, ~38 msgs base transcript')
console.log('              testing whether longmem-trained classifier (with flipped weights)')
console.log('              still preserves identifier-shaped needles on engineering chat.')
console.log()

const taggers = {
  regex:           regexTag,
  'in-domain':     inDomainTag,
  'longmem-trained': longmemTag,
}

console.log('transcript                  tagger             bfly%   lastN%   Δ')
console.log('─'.repeat(74))
for (const core of CORES) {
  for (const [name, tagFn] of Object.entries(taggers)) {
    const r = runOne(core, tagFn)
    console.log(
      `${core.id.padEnd(28)}${name.padEnd(20)}${(r.bfly*100).toFixed(0).padStart(4)}%${(r.lastn*100).toFixed(0).padStart(9)}%${((r.bfly-r.lastn)*100 >= 0 ? '+' : '') + ((r.bfly-r.lastn)*100).toFixed(0).padStart(5) + 'pp'}`
    )
  }
  console.log()
}

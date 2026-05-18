#!/usr/bin/env node
// Butterfly v3.3 — LLM-tagger variant. Same protocol as v3.1
// (butterfly-purecode-hard.mjs), but the regex tagger is swapped for
// a batched LLM call. Tests whether the mechanism's win in the
// CONFIRMED regime depends on the regex coincidentally fitting our
// 4 transcripts.
//
// What's the same:
//   - same 4 cores + padding + noise
//   - same chrysalis (concat keep verbatim, summarize first sentence, drop melt)
//   - same keyword-coverage scoring
//   - same multi-gen noise compounding
//
// What's different:
//   - tagAll() calls LM Studio with a batched prompt asking for a JSON
//     array of {idx, label} for every message in one shot.
//   - Falls back to regex labels per-message if LLM parsing fails.
//
// We smoke on TWO configs by default:
//   1. CONFIRMED point: len=38, budget=100, gens=3
//   2. REFUTED point:   len=12, budget=400, gens=1
// If the regex-tagger phase diagram persists with an LLM tagger, the
// mechanism is tagger-independent. If it doesn't, the regex was doing
// the work.
//
// Usage:
//   node tools/butterfly-llm-tagger.mjs                                # 2-config smoke
//   MODEL=qwen3-14b-mlx node tools/butterfly-llm-tagger.mjs            # explicit model
//   DEBUG=1 node tools/butterfly-llm-tagger.mjs                        # dump LLM tags + memories
//   CONFIGS=len38-bud100-gens3 node tools/butterfly-llm-tagger.mjs     # one config only

import { writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const RESULTS_DIR = join(ROOT, 'test-results', 'butterfly-sweep')
const LMS_BASE = process.env.LMS_BASE || 'http://localhost:1234/v1'
const MODEL    = process.env.MODEL    || 'qwen3-14b-mlx'
const DEBUG    = process.env.DEBUG === '1'
// STRATEGY:
//   'json'       (default) — original batched-JSON prompt. Tested 2026-05-18.
//   'onechar'    — one-char-per-message output (`k`/`s`/`m`). Hard cap on
//                  keeps. Identifier-first prompt. Designed to fix the
//                  over-tagging + parse-failure modes the JSON strategy hit.
const STRATEGY = process.env.STRATEGY || 'json'
const MAX_KEEPS = parseInt(process.env.MAX_KEEPS || '5', 10)
const tokens   = (s) => Math.ceil(s.length / 4)

// ─── transcripts (4 cores + their needle keywords) ──────────────
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
  { role: 'assistant', content: "i'll prepare some notes." },
  { role: 'user',      content: "design team wants a 30-min sync on the dashboard rework" },
  { role: 'assistant', content: "any morning next week works. send a poll." },
  { role: 'user',      content: "junior is starting monday — we doing the buddy thing?" },
  { role: 'assistant', content: "happy to pair them with me for the first week." },
  { role: 'user',      content: "any onboarding template doc in confluence?" },
  { role: 'assistant', content: "i think so, will dig it up." },
  { role: 'user',      content: "all-hands deck for tuesday — you got a slot?" },
  { role: 'assistant', content: "i've got platform updates. you take the metrics slide?" },
  { role: 'user',      content: "sure. and the q3 okr review is the week after" },
  { role: 'assistant', content: "noted. anything else for today?" },
  { role: 'user',      content: "the prod monitoring dashboard is slow lately" },
  { role: 'assistant', content: "saw that too. will file a ticket." },
  { role: 'user',      content: "feedback round for the staff role — you want input?" },
  { role: 'assistant', content: "yeah, send me the form when ready." },
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

function buildTranscript(core, totalMsgs) {
  const padBudget = Math.max(0, totalMsgs - core.length)
  const half = Math.floor(padBudget / 2)
  const pre  = PADDING_POOL.slice(0, Math.min(half, PADDING_POOL.length))
  const post = PADDING_POOL.slice(PADDING_POOL.length - Math.min(padBudget - pre.length, PADDING_POOL.length))
  return [...pre, ...core, ...post]
}

// ─── regex fallback (used per-message if LLM fails to label) ────
function regexTag(msg) {
  const t = msg.content
  let score = 0
  if (/\b\w+\/[\w\-./]+\.(ts|js|py|md|sql|yaml|toml|json|tsx|jsx|go|rs|html|css)\b/.test(t)) score += 3
  if (/TICKET-\d+/i.test(t)) score += 3
  if (/#[\w-]+/.test(t)) score += 3
  if (/@[\w-]+\/[\w-]+/.test(t)) score += 3
  if (/\blines?\s+\d+(\s*[-–]\s*\d+)?\b/i.test(t)) score += 3
  if (/\b(Decision|Root cause|Confirmed|Found it):/i.test(t)) score += 3
  if (/\b\w+\.\w+\(\)/.test(t)) score += 2
  if (/\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/.test(t)) score += 2
  if (/\b\d+\s*(req\/min|ms|s|min|gb|mb|kb|tokens?|seconds?|minutes?)\b/i.test(t)) score += 2
  if (/`[^`\n]{2,}`/.test(t)) score += 1
  if (/^\s*(ok|lgtm|sure|thx|noted|got it|will do|cool|sweet|alright|yep|yes|no|nope)[\s,.!]*$/i.test(t)) score -= 4
  if (/^\s*(ok|lgtm|sure|thx|noted|got it)\b.{0,30}$/i.test(t) && t.length < 50) score -= 2
  if (/(while you're (here|at it)|btw unrelated|side q|while we're here|unrelated|aside|off-topic)/i.test(t)) score -= 1
  if (t.endsWith('?') && t.length < 50 && !/\w+\/[\w.]+/.test(t)) score -= 1
  if (t.length < 40) score -= 1
  return score >= 3 ? 'keep' : score >= 1 ? 'summarize' : 'melt'
}

// ─── LLM tagger: batched JSON call ───────────────────────────────
const THINKING_MODELS = /(qwen3|deepseek-r1|r1-)/i
function applyNoThink(user) {
  return THINKING_MODELS.test(MODEL) ? `/no_think ${user}` : user
}
function stripThink(s) {
  return (s || '').replace(/<think>[\s\S]*?<\/think>/g, '').trim()
}

async function lmsChat(system, user, max_tokens, temperature = 0.0) {
  const res = await fetch(`${LMS_BASE}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: 'system', content: system }, { role: 'user', content: applyNoThink(user) }],
      max_tokens, temperature,
    }),
  })
  if (!res.ok) throw new Error(`LM Studio ${res.status}: ${await res.text()}`)
  const j = await res.json()
  const msg = j.choices?.[0]?.message
  const text = stripThink(msg?.content) || ''
  return text
}

async function llmTagJSON(messages) {
  const list = messages.map((m, i) => `[#${i} ${m.role}] ${m.content}`).join('\n')
  const sys = `You classify each message in a developer chat for context compaction.

For each message, choose exactly ONE label:
  keep      = irreplaceable atom — root cause named, owner+channel, file:line, decision, code snippet, specific identifier, line number
  summarize = substantive but a one-line gist suffices (multi-step explanation, verbose tool dump with one fact)
  melt      = greetings, acks, "lgtm/ok/sure", restatements, dead-end tangents, polite framing, generic questions

Distribution prior: MOST messages are MELT in a real conversation. Use KEEP only when the message contains something irreplaceable that an engineer would need to act on.

Return ONLY a JSON array, one object per message, in order:
[{"idx": 0, "label": "keep"}, {"idx": 1, "label": "melt"}, ...]

No preamble, no markdown fence, no reasoning. Just the JSON array. The LAST CHARACTER of your reply MUST be ].`

  let raw = ''
  try {
    raw = await lmsChat(sys, list, Math.min(2000, messages.length * 30 + 200), 0.0)
  } catch (e) {
    console.error(`[llm-tag] LMS error, falling back to regex: ${e.message}`)
    return { tags: messages.map(regexTag), source: 'regex-fallback', raw: '' }
  }

  raw = raw.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '').trim()
  const start = raw.indexOf('[')
  const end = raw.lastIndexOf(']')
  if (start === -1 || end === -1) return { tags: messages.map(regexTag), source: 'regex-fallback', raw }

  let parsed
  try { parsed = JSON.parse(raw.slice(start, end + 1)) } catch { return { tags: messages.map(regexTag), source: 'regex-fallback', raw } }

  const labels = new Array(messages.length)
  for (const entry of parsed) {
    if (typeof entry?.idx === 'number' && entry.idx >= 0 && entry.idx < messages.length) {
      const lab = String(entry.label || '').toLowerCase().trim()
      labels[entry.idx] = (lab === 'keep' || lab === 'summarize' || lab === 'melt') ? lab : null
    }
  }
  const tags = labels.map((l, i) => l || regexTag(messages[i]))
  return { tags, source: 'llm', raw }
}

// ─── one-char strategy: stricter selectivity + minimal output ────
// Goals (informed by what JSON strategy broke on):
//   - Hard cap on keep count via prompt.
//   - Identifier-first definition: only "keep" if the message contains
//     a UNIQUE concrete identifier (file path, line range, ticket ID,
//     person+channel, code call, decision marker).
//   - One character per message ('k', 's', 'm'). Eliminates JSON parse
//     failures; eliminates per-entry metadata overhead.
//   - Position-encoded output ordering — N input messages → exactly
//     N output characters, in order. Length check rejects malformed.
async function llmTagOneChar(messages) {
  const list = messages.map((m, i) => `[${i.toString().padStart(2, '0')} ${m.role}] ${m.content}`).join('\n')
  const sys = `You label conversation messages for context compaction at a tight token budget.

For each message, output ONE character:
  k = keep      — contains a UNIQUE identifier an engineer must act on:
                    a file path (lib/x.ts), line range (lines 142-148),
                    ticket ID (TICKET-4421), code call (Date.now()),
                    decision marker ("Decision:", "Root cause:", "Confirmed:"),
                    specific named owner + channel (#auth-platform).
                  No identifier → not keep. Generic context is not keep.
  s = summarize — multi-point reasoning or substantive non-identifier
                  content worth a one-line gist.
  m = melt      — everything else: greetings, acks, "ok/lgtm/sure",
                  clarifying questions without specifics, off-topic.

HARD CAP: at most ${MAX_KEEPS} 'k' tags total. The rest must be 's' or 'm'.
When in doubt: lean MELT. Most messages are melt in real conversations.

OUTPUT FORMAT: ${messages.length} characters total, no spaces, no newlines.
One character per input message, in the same order.
Example for 5 messages: mkmsm

Output ONLY the label string. No preamble. No commentary. No newlines before or after.`

  let raw = ''
  try {
    raw = await lmsChat(sys, list, 200, 0.0)
  } catch (e) {
    console.error(`[llm-tag] LMS error, falling back to regex: ${e.message}`)
    return { tags: messages.map(regexTag), source: 'regex-fallback', raw: '' }
  }

  // Extract just the k/s/m characters. The model may emit whitespace.
  const cleaned = raw.replace(/[^ksm]/gi, '').toLowerCase()
  let tags
  let source
  if (cleaned.length < messages.length) {
    // Mix LLM tags for the first positions with regex-fallback after.
    tags = messages.map((m, i) => {
      const c = cleaned[i]
      if (c === 'k') return 'keep'
      if (c === 's') return 'summarize'
      if (c === 'm') return 'melt'
      return regexTag(m)
    })
    source = cleaned.length === 0 ? 'regex-fallback' : 'llm-partial'
  } else {
    const head = cleaned.slice(0, messages.length)
    tags = head.split('').map(c => c === 'k' ? 'keep' : c === 's' ? 'summarize' : 'melt')
    source = 'llm'
  }

  // Enforce hard cap on keeps regardless of path. If we exceeded, demote
  // the LATEST keeps to summarize until cap is met (keeps the earlier
  // "more confident" messages; drops later "filler" ones).
  let keepCount = tags.filter(t => t === 'keep').length
  if (keepCount > MAX_KEEPS) {
    let excess = keepCount - MAX_KEEPS
    for (let i = tags.length - 1; i >= 0 && excess > 0; i--) {
      if (tags[i] === 'keep') { tags[i] = 'summarize'; excess-- }
    }
  }

  return { tags, source, raw, label_string: cleaned.slice(0, messages.length) }
}

async function llmTagAll(messages) {
  if (STRATEGY === 'onechar') return llmTagOneChar(messages)
  return llmTagJSON(messages)
}

// ─── chrysalis / lastN / scoring (identical to v3.1) ─────────────
function firstSentence(s) {
  const m = s.match(/^[^.!?]*[.!?]/)
  return (m ? m[0] : s).trim()
}
function chrysalis(messages, tags, budget) {
  const parts = []
  for (let i = 0; i < messages.length; i++) {
    const t = tags[i], m = messages[i]
    if (t === 'keep') parts.push(`[${m.role}] ${m.content}`)
    else if (t === 'summarize') parts.push(`[${m.role}] ${firstSentence(m.content)}`)
  }
  let text = parts.join('\n')
  const maxChars = budget * 4
  if (text.length > maxChars) text = text.slice(0, maxChars).replace(/\s\S*$/, '') + '…'
  return text
}
function pickLastN(messages, budget) {
  const out = []
  let acc = 0
  for (let i = messages.length - 1; i >= 0; i--) {
    const t = tokens(messages[i].content)
    if (acc + t > budget && out.length > 0) break
    out.unshift(messages[i])
    acc += t
  }
  return out
}
function scoreKW(memory, keywords) {
  const lower = memory.toLowerCase()
  const present = keywords.filter(k => lower.includes(k.toLowerCase())).length
  return { frac: present / keywords.length, present, total: keywords.length, missed: keywords.filter(k => !lower.includes(k.toLowerCase())) }
}

// ─── one config-run ─────────────────────────────────────────────
async function runConfig(core, totalMsgs, budget, gens) {
  let bflyMessages = buildTranscript(core.core, totalMsgs).slice()
  let lastnMessages = buildTranscript(core.core, totalMsgs).slice()
  let bflyFinal = ''
  let lastNFinal = ''
  const traces = []

  for (let g = 1; g <= gens; g++) {
    const t0 = Date.now()
    const tagResult = await llmTagAll(bflyMessages)
    bflyFinal = chrysalis(bflyMessages, tagResult.tags, budget)
    const truncated = pickLastN(lastnMessages, budget)
    lastNFinal = truncated.map(m => `[${m.role}] ${m.content}`).join('\n')

    const dist = tagResult.tags.reduce((acc, x) => ({ ...acc, [x]: (acc[x] || 0) + 1 }), {})
    traces.push({ gen: g, n_msgs: bflyMessages.length, source: tagResult.source, dist, label_string: tagResult.label_string || null, llm_ms: Date.now() - t0, rebuilt_tokens: tokens(bflyFinal), lastn_tokens: tokens(lastNFinal) })

    if (g < gens) {
      const noise = NOISE_PER_GEN[(g - 1) % NOISE_PER_GEN.length]
      bflyMessages = [{ role: 'assistant', content: `[REBUILT FROM GEN ${g}]\n${bflyFinal}` }, ...noise]
      lastnMessages = [...truncated, ...noise]
    }
  }

  const bflyScore = scoreKW(bflyFinal, core.needle_keywords)
  const lastNScore = scoreKW(lastNFinal, core.needle_keywords)

  if (DEBUG) {
    console.log(`\n=== ${core.id}  len=${totalMsgs} bud=${budget} gens=${gens} ===`)
    for (const tr of traces) {
      console.log(`  gen ${tr.gen}: ${tr.n_msgs} msgs · tagger=${tr.source} · ${JSON.stringify(tr.dist)} · llm=${tr.llm_ms}ms`)
    }
    console.log(`  butterfly: ${(bflyScore.frac*100).toFixed(0)}% (${bflyScore.present}/${bflyScore.total})  missed: ${bflyScore.missed.join(', ') || '∅'}`)
    console.log(`  lastN:     ${(lastNScore.frac*100).toFixed(0)}% (${lastNScore.present}/${lastNScore.total})  missed: ${lastNScore.missed.join(', ') || '∅'}`)
  }

  return { bfly: bflyScore, lastn: lastNScore, traces }
}

// ─── smoke driver ────────────────────────────────────────────────
const CONFIGS = {
  'len38-bud100-gens3': { length: 38, budget: 100, gens: 3, label: 'CONFIRMED point (P-20260515-06)' },
  'len12-bud400-gens1': { length: 12, budget: 400, gens: 1, label: 'REFUTED point (P-20260512-05)' },
  // Boundary probe: should be partial-win per the regex phase diagram
  'len30-bud150-gens2': { length: 30, budget: 150, gens: 2, label: 'boundary probe (regex showed ██)' },
}

async function main() {
  console.log(`[bfly-llm] model=${MODEL}  endpoint=${LMS_BASE}  strategy=${STRATEGY}  max_keeps=${MAX_KEEPS}`)

  const selected = process.env.CONFIGS
    ? process.env.CONFIGS.split(',').map(k => CONFIGS[k]).filter(Boolean)
    : [CONFIGS['len38-bud100-gens3'], CONFIGS['len12-bud400-gens1']]
  if (selected.length === 0) { console.error('no matching configs'); process.exit(1) }

  const t0 = Date.now()
  const results = {
    started_at: new Date().toISOString(),
    runner: 'butterfly-llm-tagger-v3.3',
    model: MODEL, endpoint: LMS_BASE,
    configs: selected.map(c => ({ length: c.length, budget: c.budget, gens: c.gens, label: c.label })),
    cells: [],
  }

  for (const cfg of selected) {
    console.log(`\n[bfly-llm] ── ${cfg.label} (len=${cfg.length}, bud=${cfg.budget}, gens=${cfg.gens}) ──`)
    for (const core of CORES) {
      const r = await runConfig(core, cfg.length, cfg.budget, cfg.gens)
      const cell = {
        transcript: core.id,
        length: cfg.length, budget: cfg.budget, gens: cfg.gens,
        bfly_frac: +r.bfly.frac.toFixed(3),
        lastn_frac: +r.lastn.frac.toFixed(3),
        delta: +(r.bfly.frac - r.lastn.frac).toFixed(3),
        traces: r.traces,
      }
      results.cells.push(cell)
      console.log(`[bfly-llm]   ${core.id.padEnd(22)} bfly=${(cell.bfly_frac*100).toFixed(0)}%  lastN=${(cell.lastn_frac*100).toFixed(0)}%  Δ=${(cell.delta*100).toFixed(0)}pp`)
    }
  }

  results.finished_at = new Date().toISOString()
  results.runtime_ms = Date.now() - t0

  // Compare to expected regex-tagger result
  console.log(`\n── summary vs regex baseline ──`)
  for (const cfg of selected) {
    const subset = results.cells.filter(c => c.length === cfg.length && c.budget === cfg.budget && c.gens === cfg.gens)
    const meanDelta = subset.reduce((s, c) => s + c.delta, 0) / subset.length
    const bflyTagged = subset.reduce((s, c) => s + c.traces.filter(t => t.source === 'llm').length, 0)
    const totalTaggers = subset.reduce((s, c) => s + c.traces.length, 0)
    console.log(`  ${cfg.label}`)
    console.log(`    mean Δ = ${(meanDelta * 100).toFixed(0)}pp · tagger=llm in ${bflyTagged}/${totalTaggers} calls`)
  }

  mkdirSync(RESULTS_DIR, { recursive: true })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const out = join(RESULTS_DIR, `butterfly-llmtagger-${stamp}.json`)
  writeFileSync(out, JSON.stringify(results, null, 2))
  console.log(`\nwrote: ${out}  (${results.runtime_ms}ms total)`)
}

main().catch(e => { console.error('FATAL:', e); process.exit(1) })

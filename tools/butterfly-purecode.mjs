#!/usr/bin/env node
// Butterfly v3 — pure-code memory test. No LLM, no browser, no GPU.
// Tests the narrowest version of the pre-registered claim: does
// tag-and-rebuild preserve the planted needle in 400 tokens, where
// naive lastN truncation does not?
//
// Determinism: every operation is rule-based. Output is identical
// across runs. `runs_per_transcript=1` is sufficient; we keep the
// loop structure so the JSON schema matches the browser+cloud
// sweeps and the existing grader works unchanged.
//
// What this REPLACES:
//   - LLM tagger        → regex+keyword-score heuristic
//   - LLM chrysalis     → mechanical concat (keep verbatim, summarize → first
//                         sentence, drop melt; truncate at TARGET_TOKENS)
//   - LLM judge         → keyword-coverage scoring against the expected fact's
//                         load-bearing identifiers
//   - LLM answer arms   → SKIPPED. We don't test "can an LLM extract the
//                         needle from the compaction" — we test "did the
//                         compaction preserve the needle at all." If the
//                         needle isn't in the compaction string, no
//                         downstream LLM can recover it.
//
// Usage:
//   node tools/butterfly-purecode.mjs                            # all 4 transcripts
//   TRANSCRIPTS=jwt-clock-race node tools/butterfly-purecode.mjs # just one
//   DEBUG=1 node tools/butterfly-purecode.mjs                    # print tags + memories

import { writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const RESULTS_DIR = join(ROOT, 'test-results', 'butterfly-sweep')

const TARGET_TOKENS = 400
const tokens = (s) => Math.ceil(s.length / 4)
const DEBUG = process.env.DEBUG === '1'

// ─── transcripts + planted-needle keywords ────────────────────────
//
// `needle_keywords` is the list of load-bearing identifiers in
// expected_fact that MUST survive compaction for an engineer to act.
// Picked manually per transcript, lower-cased. Each transcript has
// 5-7 keywords; scoring is fraction present.
const BUILT_INS = [
  {
    id: 'jwt-clock-race',
    name: 'JWT clock race · root cause',
    needle_keywords: ['issuetoken', 'lib/jwt.ts', 'date.now()', 'exp', 'test', 'twice'],
    transcript: [
      { role: 'user',      content: "Our auth/session.test.ts is flaky in CI but passes locally. Help?" },
      { role: 'assistant', content: "Sure — share the test and the failure output." },
      { role: 'user',      content: "It asserts decoded.exp equals Math.floor(Date.now()/1000) + 3600. CI says expected 1735689600 received 1735689599." },
      { role: 'assistant', content: "Off-by-one second. Smells like a timing race rather than a logic bug." },
      { role: 'assistant', content: "Confirmed in lib/jwt.ts: issueToken reads Date.now() once for `exp`, the test reads Date.now() again in the assertion. On slow CI those two reads can land on different seconds. Root cause: clock race, not a code bug." },
      { role: 'user',      content: "ok so the fix?" },
      { role: 'assistant', content: "Two options. Cheap: capture Date.now() once before calling issueToken. Proper: have issueToken accept an optional `now` parameter so tests inject a fixed clock." },
      { role: 'user',      content: "let's do the proper one. lgtm, pushing." },
    ],
    question: "What is the exact code mistake in lib/jwt.ts that causes the off-by-one second?",
    fact: "issueToken in lib/jwt.ts reads Date.now() once to compute the `exp` field. The test assertion ALSO reads Date.now() (a second time). The mistake is the second Date.now() read in the test.",
  },
  {
    id: 'auth-owner-pto',
    name: 'Auth-service owner · ownership info',
    needle_keywords: ['marcus lee', '#auth-platform', 'sarah chen', 'pto', 'tuesday', '@company/auth-service'],
    transcript: [
      { role: 'user',      content: "We're shipping the new checkout API Friday. Need someone on auth-service to bless the JWT format change." },
      { role: 'assistant', content: "auth-service team — let me pull their owner." },
      { role: 'user',      content: "Slack channel works too if you have it." },
      { role: 'assistant', content: "Sarah Chen owns @company/auth-service — Slack is #auth-platform. She's PTO through Tuesday, backup reviewer is Marcus Lee." },
      { role: 'user',      content: "noted, thx. while you're at it, who do we ping for billing-service?" },
      { role: 'assistant', content: "Don is the maintainer; channel I know of is #billing-eng." },
      { role: 'user',      content: "ok will tag #billing-eng then. one more — does payment-svc need a review for this change?" },
      { role: 'assistant', content: "No, payment-svc only proxies through. No review needed there." },
    ],
    question: "Who is the backup reviewer for auth-service and what channel?",
    fact: "Marcus Lee is the backup reviewer for @company/auth-service while Sarah Chen is on PTO through Tuesday. Slack channel is #auth-platform.",
  },
  {
    id: 'rate-limit-decision',
    name: 'Rate-limit algorithm · decision',
    needle_keywords: ['token bucket', '100 req/min', 'redis', '600ms', 'fail-open', 'fixed-window', '50 req/min', 'ticket-4421'],
    transcript: [
      { role: 'user',      content: "We need rate limiting on /api/v1. What should we use?" },
      { role: 'assistant', content: "Token bucket or fixed window are the usual options." },
      { role: 'user',      content: "which one for us? we have spiky traffic patterns." },
      { role: 'assistant', content: "Token bucket handles bursts better. Suggest token bucket at 100 req/min per IP, refill every 600ms." },
      { role: 'assistant', content: "Decision: token-bucket at 100 req/min per IP via Redis; if Redis is unreachable for >5s, fail-open with a local in-memory fixed-window of 50 req/min. We accept brief over-limit during Redis outages over hard-failing user requests. Tracked under TICKET-4421." },
      { role: 'user',      content: "lgtm. you'll draft the RFC?" },
      { role: 'assistant', content: "Yes, will have an RFC by Wednesday." },
      { role: 'user',      content: "sweet, while you're here — should we move to grpc later this year?" },
      { role: 'assistant', content: "Probably not before Q4 — too many internal clients on REST." },
    ],
    question: "What is the rate-limit decision and Redis fallback?",
    fact: "Token bucket at 100 req/min per IP via Redis (refill every 600ms). When Redis unreachable >5s, fail-open to local fixed-window at 50 req/min. TICKET-4421.",
  },
  {
    id: 'cache-race-fileline',
    name: 'Cache race · file:line location',
    needle_keywords: ['lib/cache.ts', '142-148', 'cache.set', 'cache.get', 'version', 'critical section'],
    transcript: [
      { role: 'user',      content: "Cache occasionally returns stale data right after a write. Where is it?" },
      { role: 'assistant', content: "How are reads structured? do they go through the same lock as writes?" },
      { role: 'user',      content: "no, reads bypass the lock. only writes hold it." },
      { role: 'assistant', content: "Found it: lib/cache.ts lines 142-148. cache.set updates the value, but the version-counter increment happens 2 statements later. A concurrent cache.get landing between those two statements sees the new value but the old version, and the downstream consumer treats that as 'still cached, use it' because version didn't change." },
      { role: 'user',      content: "ouch. so version increment needs to be atomic with set?" },
      { role: 'assistant', content: "Yes. Move the version++ inside the same critical section, or use a CAS on the version field." },
      { role: 'user',      content: "btw unrelated, the OPS dashboard times out a lot lately." },
      { role: 'assistant', content: "Saw that too — Grafana proxy timeout was lowered to 5s last week, should bump back to 15s." },
      { role: 'user',      content: "will file a ticket. thx for the cache catch." },
    ],
    question: "Where is the read-after-write race and what causes it?",
    fact: "Race is in lib/cache.ts lines 142-148. cache.set updates value, version-counter increment 2 statements later. cache.get between sees new value, old version. Fix: move version++ into critical section or CAS.",
  },
]

const NOISE = [
  { role: 'user',      content: "while we're here — should we use prettier or biome for formatting?" },
  { role: 'assistant', content: "biome is faster but prettier has wider plugin ecosystem. Either works." },
  { role: 'user',      content: "ok I'll punt on it. also CI build time crept up to 4 min, want to look later." },
  { role: 'assistant', content: "Probably the snapshot suite growing. We can split it next sprint." },
]

// ─── regex-heuristic tagger ───────────────────────────────────────
//
// Score per message: start at 0, add for "load-bearing" signals,
// subtract for "ack-like" patterns. Map to {keep, summarize, melt}.
function tagMessage(msg) {
  const t = msg.content
  const lower = t.toLowerCase()
  let score = 0
  const reasons = []

  // KEEP signals
  if (/\b\w+\/[\w\-./]+\.(ts|js|py|md|sql|yaml|toml|json|tsx|jsx|go|rs|html|css)\b/.test(t))
    { score += 3; reasons.push('file-path') }
  if (/TICKET-\d+/i.test(t))                  { score += 3; reasons.push('ticket') }
  if (/#[\w-]+/.test(t))                      { score += 3; reasons.push('slack-channel') }
  if (/@[\w-]+\/[\w-]+/.test(t))              { score += 3; reasons.push('pkg-mention') }
  if (/\blines?\s+\d+(\s*[-–]\s*\d+)?\b/i.test(t))  { score += 3; reasons.push('line-range') }
  if (/\b(Decision|Root cause|Confirmed|Found it):/i.test(t))
    { score += 3; reasons.push('decision-marker') }
  if (/\b\w+\.\w+\(\)/.test(t))               { score += 2; reasons.push('code-call') }
  if (/\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/.test(t)) { score += 2; reasons.push('proper-name') }
  if (/\b\d+\s*(req\/min|ms|s|min|gb|mb|kb|tokens?|seconds?|minutes?)\b/i.test(t))
    { score += 2; reasons.push('quantity') }
  if (/`[^`\n]{2,}`/.test(t))                 { score += 1; reasons.push('inline-code') }

  // MELT signals — bare acks
  if (/^\s*(ok|lgtm|sure|thx|noted|got it|will do|cool|sweet|alright|yep|yes|no|nope)[\s,.!]*$/i.test(t))
    { score -= 4; reasons.push('bare-ack') }
  if (/^\s*(ok|lgtm|sure|thx|noted|got it)\b.{0,30}$/i.test(t) && t.length < 50)
    { score -= 2; reasons.push('short-ack') }
  if (/(while you're (here|at it)|btw unrelated|while we're here)/i.test(t))
    { score -= 1; reasons.push('topic-shift') }
  if (lower.endsWith('?') && t.length < 50 && !/\w+\/[\w.]+/.test(t))
    { score -= 1; reasons.push('short-question') }
  if (t.length < 40) { score -= 1; reasons.push('very-short') }

  const label = score >= 3 ? 'keep' : score >= 1 ? 'summarize' : 'melt'
  return { label, score, reasons }
}

function tagAll(messages) {
  return messages.map((m, idx) => ({ idx, ...tagMessage(m), preview: m.content.slice(0, 70) }))
}

// ─── mechanical chrysalis: concat keep verbatim + first-sentence summarize, drop melt, truncate
function firstSentence(s) {
  const m = s.match(/^[^.!?]*[.!?]/)
  return (m ? m[0] : s).trim()
}

function chrysalis(messages, tags, budget) {
  const parts = []
  for (let i = 0; i < messages.length; i++) {
    const t = tags[i].label
    const m = messages[i]
    if (t === 'keep') parts.push(`[${m.role}] ${m.content}`)
    else if (t === 'summarize') parts.push(`[${m.role}] ${firstSentence(m.content)}`)
    // melt → drop
  }
  let text = parts.join('\n\n')
  // truncate to budget if needed
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
  return out.map(m => `[${m.role}] ${m.content}`).join('\n\n')
}

// ─── keyword-coverage scoring ─────────────────────────────────────
//
// For each needle keyword, check substring presence in the compacted
// memory (case-insensitive). Score = fraction present, mapped to
// hit / partial / miss.
function score(memory, keywords) {
  const lower = memory.toLowerCase()
  const hits = keywords.map(k => ({ k, present: lower.includes(k.toLowerCase()) }))
  const present = hits.filter(h => h.present).length
  const frac = present / keywords.length
  const verdict = frac >= 0.75 ? 'hit' : frac >= 0.33 ? 'partial' : 'miss'
  return { verdict, present, total: keywords.length, frac: +frac.toFixed(2), missed: hits.filter(h => !h.present).map(h => h.k) }
}

// ─── one transcript run ──────────────────────────────────────────
function runOne(t) {
  const t0 = Date.now()
  const messages = [...t.transcript, ...NOISE]

  const tags = tagAll(messages)
  const bflyMemory = chrysalis(messages, tags, TARGET_TOKENS)
  const lastNMemory = pickLastN(messages, TARGET_TOKENS)

  const bflyScore = score(bflyMemory, t.needle_keywords)
  const lastNScore = score(lastNMemory, t.needle_keywords)

  if (DEBUG) {
    console.log(`\n--- ${t.id} tags ---`)
    for (const tag of tags) {
      console.log(`  ${tag.idx} [${tag.label.padEnd(9)}] score=${tag.score.toString().padStart(2)} ${tag.preview}`)
    }
    console.log(`--- butterfly memory (${tokens(bflyMemory)} tok) ---\n${bflyMemory}`)
    console.log(`--- lastN memory (${tokens(lastNMemory)} tok) ---\n${lastNMemory}`)
    console.log(`--- score ---`)
    console.log(`  butterfly: ${bflyScore.verdict} (${bflyScore.present}/${bflyScore.total})  missed: ${bflyScore.missed.join(', ') || '∅'}`)
    console.log(`  lastN:     ${lastNScore.verdict} (${lastNScore.present}/${lastNScore.total})  missed: ${lastNScore.missed.join(', ') || '∅'}`)
  }

  return {
    transcript: t.id,
    bfly: bflyScore.verdict,
    lastn: lastNScore.verdict,
    seconds: +((Date.now() - t0) / 1000).toFixed(3),
    bfly_detail: bflyScore,
    lastn_detail: lastNScore,
    tag_distribution: tags.reduce((acc, x) => ({ ...acc, [x.label]: (acc[x.label] || 0) + 1 }), {}),
    rebuilt_tokens: tokens(bflyMemory),
    lastn_tokens: tokens(lastNMemory),
  }
}

// ─── sweep + grade ────────────────────────────────────────────────
function main() {
  const sel = process.env.TRANSCRIPTS
    ? process.env.TRANSCRIPTS.split(',').map(id => BUILT_INS.find(t => t.id === id)).filter(Boolean)
    : BUILT_INS

  const results = {
    started_at: new Date().toISOString(),
    runs_per_transcript: 1,
    transcripts: sel.map(t => t.id),
    fingerprint: { runner: 'pure-code-v3', method: 'regex-tagger + concat-chrysalis + keyword-coverage' },
    runs: [],
  }

  console.log(`[bfly-pure] runner=pure-code  transcripts=${sel.length}`)
  for (const t of sel) {
    const r = runOne(t)
    results.runs.push({ ...r, run: 1 })
    const bd = r.bfly_detail, ld = r.lastn_detail
    console.log(`[bfly-pure] ${t.id.padEnd(22)} bfly=${r.bfly.padEnd(7)} (${bd.present}/${bd.total})  lastN=${r.lastn.padEnd(7)} (${ld.present}/${ld.total})  tags=${JSON.stringify(r.tag_distribution)}  ${r.seconds * 1000 | 0}ms`)
  }
  results.finished_at = new Date().toISOString()

  // ─── grade inline ────────────────────────────────────────────────
  const table = sel.map(t => {
    const row = results.runs.find(r => r.transcript === t.id)
    return {
      transcript: t.id,
      bfly: row.bfly,
      lastn: row.lastn,
      bfly_frac: row.bfly_detail.frac,
      lastn_frac: row.lastn_detail.frac,
      delta: +(row.bfly_detail.frac - row.lastn_detail.frac).toFixed(2),
    }
  })

  console.log(`\n— verdict summary —`)
  console.log('transcript                  bfly      lastN     bfly-frac  lastN-frac  Δ')
  console.log('─'.repeat(78))
  for (const r of table) {
    console.log(
      `${r.transcript.padEnd(28)}${r.bfly.padEnd(10)}${r.lastn.padEnd(10)}${(r.bfly_frac * 100).toFixed(0).padStart(8)}%${(r.lastn_frac * 100).toFixed(0).padStart(11)}%${(r.delta >= 0 ? '+' : '') + (r.delta * 100).toFixed(0).padStart(5) + '%'}`
    )
  }

  // Pre-registered threshold logic (P-20260512-05): butterfly hits/N
  // ≥ lastN hits/N + 0.15 for ALL transcripts. For pure-code, "hits"
  // means verdict ≥ partial.
  const isHit = (v) => v === 'hit' || v === 'partial'
  const allBeatByMargin = table.every(r => r.bfly_frac - r.lastn_frac >= 0.15)
  const jwt = table.find(r => r.transcript === 'jwt-clock-race')
  const jwtDoubles = jwt ? (isHit(jwt.bfly) && !isHit(jwt.lastn)) : true
  const lastnWins = table.filter(r => r.lastn_frac >= r.bfly_frac).length

  let verdict, reasoning
  if (allBeatByMargin && jwtDoubles) {
    verdict = 'CONFIRMED'
    reasoning = `All transcripts: bfly-frac ≥ lastN-frac + 0.15. jwt: bfly preserves needle, lastN does not.`
  } else if (lastnWins >= 2) {
    verdict = 'REFUTED'
    reasoning = `LastN reached or exceeded butterfly on ${lastnWins}/${table.length} transcripts.`
  } else {
    verdict = 'INCONCLUSIVE'
    const fails = []
    if (!allBeatByMargin) fails.push(`not all transcripts cleared +15pp margin (${table.filter(r => r.bfly_frac - r.lastn_frac < 0.15).map(r => r.transcript).join(', ')})`)
    if (!jwtDoubles) fails.push(`jwt-clock-race did not asymmetrically favor butterfly`)
    reasoning = fails.join(' · ')
  }

  console.log(`\nPRE-REGISTERED VERDICT (P-20260512-05 keyword-coverage variant): ${verdict}`)
  console.log(`reasoning: ${reasoning}`)

  // ─── persist ──────────────────────────────────────────────────────
  results.verdict = { result: verdict, reasoning, table }
  mkdirSync(RESULTS_DIR, { recursive: true })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const out = join(RESULTS_DIR, `butterfly-purecode-${stamp}.json`)
  writeFileSync(out, JSON.stringify(results, null, 2))
  console.log(`\nwrote: ${out}`)
}

main()

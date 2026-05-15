#!/usr/bin/env node
// Butterfly v3.1 — harder regime: longer transcripts, tight budget,
// multi-generation noise compounding. Tests the TRANSGENERATIONAL
// survival claim that v3 (1-gen, 400-token, 12-msg) couldn't see.
//
// CHANGES vs butterfly-purecode.mjs:
//   - Each transcript padded with 15 pre-needle + 15 post-needle
//     melt-able messages → ~38 messages base. Plus 12 noise injected
//     across 3 generations → ~50 effective messages in the run.
//   - TARGET_TOKENS: 400 → 100. Real compression pressure: lastN
//     cannot hold the full original transcript.
//   - N_GENERATIONS: 1 → 3. Each generation tags + chrysalis-rebuilds
//     + injects fresh noise. Memory gets reborn each cocoon.
//   - LastN's parallel: each gen truncates to TARGET_TOKENS, then
//     appends 4 new noise messages. Noise compounds; original drifts
//     out of the window.
//
// PRE-REGISTRATION (P-20260515-06): filed in PREDICTIONS.md before
// this script was first run. Confirm/refute thresholds documented
// inline below.
//
// Usage:
//   node tools/butterfly-purecode-hard.mjs
//   DEBUG=1 node tools/butterfly-purecode-hard.mjs     # full tag + memory dumps

import { writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const RESULTS_DIR = join(ROOT, 'test-results', 'butterfly-sweep')
const TARGET_TOKENS = 100
const N_GENERATIONS = 3
const DEBUG = process.env.DEBUG === '1'

const tokens = (s) => Math.ceil(s.length / 4)

// ─── padding: AVOIDS keywords from any of the 4 transcripts ──────
// Topics: org logistics, lunch, deployment chatter, vacation, conf
// schedules. Carefully written to not include any needle keywords:
//   - no "Marcus", "Sarah", "Chen", "Lee", "PTO"
//   - no "auth-platform", "billing-eng", "@company/*"
//   - no "jwt", "Date.now", "issueToken", "exp"
//   - no "cache.ts", "cache.set/get", "version"
//   - no "token bucket", "Redis", "TICKET-4421", "fail-open"
//   - no "fixed-window", "req/min", "600ms", "5s"
const PADDING_PRE = [
  { role: 'user',      content: "hey morning. got time for a quick one today?" },
  { role: 'assistant', content: "sure thing — fire away when you're ready." },
  { role: 'user',      content: "first off — is the staging cluster still on the old node version btw? someone asked yesterday." },
  { role: 'assistant', content: "i think so, qa pushed back on the upgrade last week, said the smoke suite started flaking" },
  { role: 'user',      content: "lol of course. also did finance approve the q3 tooling budget?" },
  { role: 'assistant', content: "i'll ask ramesh tomorrow when he's back from leave" },
  { role: 'user',      content: "thx. one more — anyone owning the docs migration?" },
  { role: 'assistant', content: "rasmus i think, but he's on parental leave til the end of the month" },
  { role: 'user',      content: "ah right. ok and re: lunch — was thinking we do that ramen place near the office?" },
  { role: 'assistant', content: "yes please. though i heard they raised prices, fyi." },
  { role: 'user',      content: "everyone has lol. anyway, the conf cfp deadline is friday — you submitting?" },
  { role: 'assistant', content: "probably. still picking between two talk angles, will decide tonight" },
  { role: 'user',      content: "cool. before i forget — meeting with platform team got moved to thursday 2pm" },
  { role: 'assistant', content: "noted, on the calendar. ok so anyway —" },
  { role: 'user',      content: "right, so the actual thing i wanted to ask about..." },
]

const PADDING_POST = [
  { role: 'user',      content: "ok cool, lgtm." },
  { role: 'assistant', content: "great. i'll write it up in the rfc later this week." },
  { role: 'user',      content: "while you're here — any thoughts on the new sprint cadence?" },
  { role: 'assistant', content: "honestly mixed. two weeks felt too short for the platform-y work" },
  { role: 'user',      content: "yeah, i've heard that from a few folks. we should bring it up in retro" },
  { role: 'assistant', content: "agreed. also — design team wants a 30-min sync on the dashboard rework, when's good?" },
  { role: 'user',      content: "any morning next week. send a poll." },
  { role: 'assistant', content: "will do. one more thing — the new junior is starting monday, are we doing a buddy thing?" },
  { role: 'user',      content: "yes. i'll pair them with you for the first week if that's ok" },
  { role: 'assistant', content: "sure, happy to. quick aside — do we have a template doc for first-week onboarding?" },
  { role: 'user',      content: "there's one in confluence somewhere. i'll dig it up and slack it to you" },
  { role: 'assistant', content: "thanks. last thing — the all-hands deck for tuesday, you have a slot or shall i fill?" },
  { role: 'user',      content: "i've got platform updates. you can take the metrics slide if you want" },
  { role: 'assistant', content: "i'll prep it. anything else?" },
  { role: 'user',      content: "nope, all good. catch you later." },
]

// ─── transcripts: the original 4, padded ─────────────────────────
const BUILT_INS = [
  {
    id: 'jwt-clock-race',
    name: 'JWT clock race · root cause',
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
    question: "What is the exact code mistake in lib/jwt.ts that causes the off-by-one second?",
    fact: "issueToken in lib/jwt.ts reads Date.now() once for `exp`, the test reads Date.now() again. Two reads can land on different seconds.",
  },
  {
    id: 'auth-owner-pto',
    name: 'Auth-service owner · ownership info',
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
    question: "Who is the backup reviewer for auth-service and what channel?",
    fact: "Marcus Lee is backup reviewer for @company/auth-service. Slack channel #auth-platform.",
  },
  {
    id: 'rate-limit-decision',
    name: 'Rate-limit algorithm · decision',
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
    question: "What is the rate-limit decision and Redis fallback?",
    fact: "Token bucket 100 req/min per IP via Redis. Redis unreachable >5s → fail-open fixed-window at 50 req/min. TICKET-4421.",
  },
  {
    id: 'cache-race-fileline',
    name: 'Cache race · file:line location',
    needle_keywords: ['lib/cache.ts', '142-148', 'cache.set', 'cache.get', 'version'],
    core: [
      { role: 'user',      content: "Cache occasionally returns stale data right after a write. Where is it?" },
      { role: 'assistant', content: "Do reads go through the same lock as writes?" },
      { role: 'user',      content: "no, reads bypass the lock. only writes hold it." },
      { role: 'assistant', content: "Found it: lib/cache.ts lines 142-148. cache.set updates the value, but the version-counter increment happens 2 statements later. A concurrent cache.get between sees the new value but the old version, downstream treats as still-cached." },
      { role: 'user',      content: "so version increment needs to be atomic with set?" },
      { role: 'assistant', content: "Yes. Move the version++ into the same critical section, or CAS on version." },
    ],
    question: "Where is the read-after-write race and what causes it?",
    fact: "Race in lib/cache.ts lines 142-148. cache.set updates value, version increment 2 statements later. cache.get between sees new value, old version.",
  },
]

// Assemble each transcript: pre-padding + core + post-padding.
for (const t of BUILT_INS) {
  t.transcript = [...PADDING_PRE, ...t.core, ...PADDING_POST]
}

// Per-generation noise — 4 fresh off-topic messages each round.
const NOISE_PER_GEN = [
  [
    { role: 'user',      content: "side q: should we use prettier or biome for formatting?" },
    { role: 'assistant', content: "biome is faster, prettier has wider plugins. Either works." },
    { role: 'user',      content: "punt on it. also CI build time crept up to 4 min, want to look later." },
    { role: 'assistant', content: "Probably the snapshot suite. We can split it next sprint." },
  ],
  [
    { role: 'user',      content: "did anyone actually try the new prod metrics dashboard?" },
    { role: 'assistant', content: "i poked it briefly. ui's nice, query latency is rough." },
    { role: 'user',      content: "ok will file feedback. unrelated — have we picked an oncall handoff doc?" },
    { role: 'assistant', content: "not yet. will draft after the sprint review." },
  ],
  [
    { role: 'user',      content: "also — design system v3 dropped, anyone migrating yet?" },
    { role: 'assistant', content: "only the marketing site so far. infra apps lagging as usual." },
    { role: 'user',      content: "fair. one more: did legal sign off on the new privacy policy?" },
    { role: 'assistant', content: "still in review. heard maybe end of next week." },
  ],
]

// ─── regex-heuristic tagger (same as v3) ──────────────────────────
function tagMessage(msg) {
  const t = msg.content
  let score = 0
  const reasons = []

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

  if (/^\s*(ok|lgtm|sure|thx|noted|got it|will do|cool|sweet|alright|yep|yes|no|nope)[\s,.!]*$/i.test(t))
    { score -= 4; reasons.push('bare-ack') }
  if (/^\s*(ok|lgtm|sure|thx|noted|got it)\b.{0,30}$/i.test(t) && t.length < 50)
    { score -= 2; reasons.push('short-ack') }
  if (/(while you're (here|at it)|btw unrelated|side q|while we're here|unrelated|aside)/i.test(t))
    { score -= 1; reasons.push('topic-shift') }
  if (t.endsWith('?') && t.length < 50 && !/\w+\/[\w.]+/.test(t))
    { score -= 1; reasons.push('short-question') }
  if (t.length < 40) { score -= 1; reasons.push('very-short') }

  const label = score >= 3 ? 'keep' : score >= 1 ? 'summarize' : 'melt'
  return { label, score, reasons }
}

function tagAll(messages) {
  return messages.map((m, idx) => ({ idx, ...tagMessage(m), preview: m.content.slice(0, 70) }))
}

// ─── compaction ───────────────────────────────────────────────────
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

// ─── score: keyword coverage in compacted memory ─────────────────
function score(memory, keywords) {
  const lower = memory.toLowerCase()
  const hits = keywords.map(k => ({ k, present: lower.includes(k.toLowerCase()) }))
  const present = hits.filter(h => h.present).length
  const frac = present / keywords.length
  const verdict = frac >= 0.75 ? 'hit' : frac >= 0.33 ? 'partial' : 'miss'
  return { verdict, present, total: keywords.length, frac: +frac.toFixed(2), missed: hits.filter(h => !h.present).map(h => h.k) }
}

// ─── multi-generation run ────────────────────────────────────────
function runOne(t) {
  const t0 = Date.now()

  // Generation loop: butterfly = compact each round; lastN truncates each round.
  let bflyMessages = t.transcript.slice()
  let lastnMessages = t.transcript.slice()
  let bflyFinal = ''

  const generations = []

  for (let gen = 1; gen <= N_GENERATIONS; gen++) {
    // Butterfly arm
    const tags = tagAll(bflyMessages)
    const rebuilt = chrysalis(bflyMessages, tags, TARGET_TOKENS)
    bflyFinal = rebuilt
    const bflyState = { gen, n_msgs: bflyMessages.length, tags_dist: tags.reduce((acc, x) => ({ ...acc, [x.label]: (acc[x.label] || 0) + 1 }), {}), rebuilt, rebuilt_tokens: tokens(rebuilt) }

    // LastN arm — truncate, then append fresh noise
    const truncated = pickLastN(lastnMessages, TARGET_TOKENS)
    const lastnSnapshot = truncated.map(m => `[${m.role}] ${m.content}`).join('\n')
    const lastnState = { gen, n_msgs: lastnMessages.length, kept: truncated.length, snapshot: lastnSnapshot, tokens: tokens(lastnSnapshot) }

    generations.push({ butterfly: bflyState, lastn: lastnState })

    // Inject next generation's noise (except after the last gen).
    if (gen < N_GENERATIONS) {
      const noise = NOISE_PER_GEN[gen - 1] || []
      bflyMessages = [
        { role: 'assistant', content: `[REBUILT FROM GEN ${gen}]\n${rebuilt}` },
        ...noise,
      ]
      lastnMessages = [...truncated, ...noise]
    }
  }

  // Final lastN memory = the snapshot from the final generation
  const lastNFinal = generations[generations.length - 1].lastn.snapshot

  const bflyScore = score(bflyFinal, t.needle_keywords)
  const lastNScore = score(lastNFinal, t.needle_keywords)

  if (DEBUG) {
    console.log(`\n=== ${t.id} ===`)
    for (const g of generations) {
      console.log(`\n--- gen ${g.butterfly.gen} ---`)
      console.log(`  butterfly: ${g.butterfly.n_msgs} msgs → tags=${JSON.stringify(g.butterfly.tags_dist)} → rebuilt (${g.butterfly.rebuilt_tokens} tok):`)
      console.log(g.butterfly.rebuilt.split('\n').map(l => '    ' + l).join('\n'))
      console.log(`  lastN:     ${g.lastn.n_msgs} msgs → kept ${g.lastn.kept} (${g.lastn.tokens} tok):`)
      console.log(g.lastn.snapshot.split('\n').map(l => '    ' + l).join('\n'))
    }
    console.log(`\n--- final scores ---`)
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
    generations: generations.map(g => ({
      gen: g.butterfly.gen,
      bfly_msgs: g.butterfly.n_msgs,
      bfly_tags: g.butterfly.tags_dist,
      bfly_tokens: g.butterfly.rebuilt_tokens,
      lastn_msgs: g.lastn.n_msgs,
      lastn_kept: g.lastn.kept,
      lastn_tokens: g.lastn.tokens,
    })),
  }
}

// ─── sweep + grade (pre-registered thresholds) ───────────────────
function main() {
  const sel = process.env.TRANSCRIPTS
    ? process.env.TRANSCRIPTS.split(',').map(id => BUILT_INS.find(t => t.id === id)).filter(Boolean)
    : BUILT_INS

  const results = {
    started_at: new Date().toISOString(),
    transcripts: sel.map(t => t.id),
    config: { target_tokens: TARGET_TOKENS, n_generations: N_GENERATIONS, padding_pre: PADDING_PRE.length, padding_post: PADDING_POST.length, noise_per_gen: NOISE_PER_GEN[0].length },
    fingerprint: { runner: 'pure-code-hard-v3.1', method: 'regex-tagger + concat-chrysalis + keyword-coverage + multi-gen noise compounding' },
    pre_registration: 'PREDICTIONS.md P-20260515-06',
    runs: [],
  }

  console.log(`[bfly-hard] config: budget=${TARGET_TOKENS} tok, gens=${N_GENERATIONS}, transcripts=${sel.length}, msgs/transcript=${PADDING_PRE.length + 8 + PADDING_POST.length}`)
  for (const t of sel) {
    const r = runOne(t)
    results.runs.push({ ...r, run: 1 })
    const bd = r.bfly_detail, ld = r.lastn_detail
    console.log(`[bfly-hard] ${t.id.padEnd(22)} bfly=${r.bfly.padEnd(7)} (${bd.present}/${bd.total} = ${(bd.frac*100).toFixed(0)}%)  lastN=${r.lastn.padEnd(7)} (${ld.present}/${ld.total} = ${(ld.frac*100).toFixed(0)}%)  Δ=${((bd.frac-ld.frac)*100).toFixed(0)}pp  ${(r.seconds*1000)|0}ms`)
  }
  results.finished_at = new Date().toISOString()

  // ─── PRE-REGISTERED VERDICT (P-20260515-06) ──────────────────────
  // CONFIRM   if all 4 transcripts have bfly_frac > lastn_frac AND
  //              mean(bfly_frac - lastn_frac) >= 0.20.
  // REFUTE    if ≥2 of 4 transcripts have lastn_frac >= bfly_frac.
  // INCONCL.  otherwise.
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

  const allButterflyWins = table.every(r => r.delta > 0)
  const meanDelta = +(table.reduce((s, r) => s + r.delta, 0) / table.length).toFixed(3)
  const lastnWins = table.filter(r => r.lastn_frac >= r.bfly_frac).length

  let verdict, reasoning
  if (allButterflyWins && meanDelta >= 0.20) {
    verdict = 'CONFIRMED'
    reasoning = `All 4 transcripts: bfly > lastN. Mean Δ = ${(meanDelta * 100).toFixed(0)}pp (≥ 20pp threshold).`
  } else if (lastnWins >= 2) {
    verdict = 'REFUTED'
    reasoning = `LastN reached or exceeded butterfly on ${lastnWins}/${table.length} transcripts.`
  } else {
    verdict = 'INCONCLUSIVE'
    const fails = []
    if (!allButterflyWins) fails.push(`butterfly didn't win on all 4 (${table.filter(r => r.delta <= 0).map(r => r.transcript).join(', ')})`)
    if (meanDelta < 0.20) fails.push(`mean Δ = ${(meanDelta * 100).toFixed(0)}pp (< 20pp threshold)`)
    reasoning = fails.join(' · ')
  }

  console.log(`\n— per-transcript table —`)
  console.log('transcript                  bfly      lastN     bfly-frac  lastN-frac  Δ')
  console.log('─'.repeat(78))
  for (const r of table) {
    console.log(
      `${r.transcript.padEnd(28)}${r.bfly.padEnd(10)}${r.lastn.padEnd(10)}${(r.bfly_frac * 100).toFixed(0).padStart(8)}%${(r.lastn_frac * 100).toFixed(0).padStart(11)}%${(r.delta >= 0 ? '+' : '') + (r.delta * 100).toFixed(0).padStart(5) + 'pp'}`
    )
  }

  console.log(`\nPRE-REGISTERED VERDICT (P-20260515-06): ${verdict}`)
  console.log(`reasoning: ${reasoning}`)

  results.verdict = { result: verdict, reasoning, table, mean_delta: meanDelta }
  mkdirSync(RESULTS_DIR, { recursive: true })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const out = join(RESULTS_DIR, `butterfly-purecode-hard-${stamp}.json`)
  writeFileSync(out, JSON.stringify(results, null, 2))
  console.log(`\nwrote: ${out}`)
}

main()

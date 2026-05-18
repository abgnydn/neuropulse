#!/usr/bin/env node
// Butterfly v3.2 — phase-diagram sweep.
//
// We have two binary data points from earlier:
//   - P-20260512-05 (1-gen / 400-tok / 12-msg)  →  REFUTED  (lastN ties)
//   - P-20260515-06 (3-gen / 100-tok / 38-msg)  →  CONFIRMED (lastN: 0%, butterfly: 100%)
//
// Two points don't tell us WHERE the mechanism starts mattering. This
// script sweeps the (budget, transcript_length, generations) cube on
// the same 4 transcripts, outputs a JSON of all results, and prints
// an ASCII heatmap showing delta(bfly_frac − lastn_frac) per cell.
//
// Determinism + speed: pure code, no LLM. Full sweep (8 × 6 × 5 ×
// 4 = 960 runs) finishes in < 1 second.
//
// Usage:
//   node tools/butterfly-sweep-phasediagram.mjs
//   GENS=3 node tools/butterfly-sweep-phasediagram.mjs            # 2-D slice at fixed gens
//   BUDGETS=50,100,200 LENGTHS=20,40,80 node tools/...           # custom grid

import { writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const RESULTS_DIR = join(ROOT, 'test-results', 'butterfly-sweep')
const tokens = (s) => Math.ceil(s.length / 4)

// ─── parameter grid ──────────────────────────────────────────────
const BUDGETS  = (process.env.BUDGETS  || '50,75,100,150,200,300,400,600').split(',').map(Number)
const LENGTHS  = (process.env.LENGTHS  || '12,20,30,50,80,120').split(',').map(Number)
const GENS     = process.env.GENS ? [parseInt(process.env.GENS, 10)] : [1, 2, 3, 4, 5]
const TRANSCRIPTS = (process.env.TRANSCRIPTS || 'jwt-clock-race,auth-owner-pto,rate-limit-decision,cache-race-fileline').split(',')

// ─── padding pool: enough to scale to length=120 ─────────────────
// Same shape as butterfly-purecode-hard.mjs but with a bigger
// melt-only pool we slice into based on target length.
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
  { role: 'user',      content: "marketing wants a launch story for next quarter" },
  { role: 'assistant', content: "i'll draft something light." },
  { role: 'user',      content: "tax form deadlines are coming up — heads up to the team?" },
  { role: 'assistant', content: "i'll post in #ops." },
  { role: 'user',      content: "alright that's everything from my side" },
  { role: 'assistant', content: "cool. catch up later." },
  { role: 'user',      content: "have we picked the new oncall rotation tool yet?" },
  { role: 'assistant', content: "still evaluating two options. decision by friday." },
  { role: 'user',      content: "any progress on the localization push?" },
  { role: 'assistant', content: "translators came back with q3 estimates. on track." },
  { role: 'user',      content: "the customer feedback portal needs a refresh" },
  { role: 'assistant', content: "agreed. will scope after the current sprint." },
  { role: 'user',      content: "and procurement still wants vendor docs by next week" },
  { role: 'assistant', content: "i'll chase the open ones." },
  { role: 'user',      content: "anything else?" },
  { role: 'assistant', content: "nope, that's everything." },
  { role: 'user',      content: "great, talk later." },
  { role: 'assistant', content: "thanks!" },
  { role: 'user',      content: "actually — one more re: the conf travel budget" },
  { role: 'assistant', content: "haven't heard back from finance yet. will follow up." },
  { role: 'user',      content: "ok no rush, but worth poking" },
  { role: 'assistant', content: "noted." },
  { role: 'user',      content: "and the team retro feedback is in google docs?" },
  { role: 'assistant', content: "yep, link is in #team-syncs." },
  { role: 'user',      content: "perfect. ok bye for real this time" },
  { role: 'assistant', content: "bye!" },
]

// ─── core transcripts (same as v3.1) ─────────────────────────────
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
  [
    { role: 'user',      content: "off-topic — are we doing summer interns this year?" },
    { role: 'assistant', content: "yes, two in eng so far. recruiter still scoping." },
    { role: 'user',      content: "cool. also: the security audit report dropped." },
    { role: 'assistant', content: "saw it. mostly low-severity stuff. will triage." },
  ],
  [
    { role: 'user',      content: "did the new node version finally land in staging?" },
    { role: 'assistant', content: "yesterday. smoke suite green. promoting to prod fri." },
    { role: 'user',      content: "nice. also the slack-to-pagerduty bridge broke again" },
    { role: 'assistant', content: "i'll file with their support. third time this month." },
  ],
]

// Build a transcript of target message-count by slicing PADDING_POOL
// around the core. For length=12 the core (8) gets ~2 pre and ~2 post.
// For length=120 the core gets ~56 pre and ~56 post (capped by pool size).
function buildTranscript(coreObj, totalMsgs) {
  const padBudget = Math.max(0, totalMsgs - coreObj.core.length)
  const half = Math.floor(padBudget / 2)
  const pre  = PADDING_POOL.slice(0, Math.min(half, PADDING_POOL.length))
  const post = PADDING_POOL.slice(PADDING_POOL.length - Math.min(padBudget - pre.length, PADDING_POOL.length))
  return [...pre, ...coreObj.core, ...post]
}

// ─── tagger / chrysalis / lastN / score (identical to v3.1) ──────
function tagMessage(msg) {
  const t = msg.content
  let score = 0
  if (/\b\w+\/[\w\-./]+\.(ts|js|py|md|sql|yaml|toml|json|tsx|jsx|go|rs|html|css)\b/.test(t)) score += 3
  if (/TICKET-\d+/i.test(t))                  score += 3
  if (/#[\w-]+/.test(t))                      score += 3
  if (/@[\w-]+\/[\w-]+/.test(t))              score += 3
  if (/\blines?\s+\d+(\s*[-–]\s*\d+)?\b/i.test(t))  score += 3
  if (/\b(Decision|Root cause|Confirmed|Found it):/i.test(t)) score += 3
  if (/\b\w+\.\w+\(\)/.test(t))               score += 2
  if (/\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/.test(t)) score += 2
  if (/\b\d+\s*(req\/min|ms|s|min|gb|mb|kb|tokens?|seconds?|minutes?)\b/i.test(t)) score += 2
  if (/`[^`\n]{2,}`/.test(t))                 score += 1
  if (/^\s*(ok|lgtm|sure|thx|noted|got it|will do|cool|sweet|alright|yep|yes|no|nope)[\s,.!]*$/i.test(t)) score -= 4
  if (/^\s*(ok|lgtm|sure|thx|noted|got it)\b.{0,30}$/i.test(t) && t.length < 50) score -= 2
  if (/(while you're (here|at it)|btw unrelated|side q|while we're here|unrelated|aside|off-topic)/i.test(t)) score -= 1
  if (t.endsWith('?') && t.length < 50 && !/\w+\/[\w.]+/.test(t)) score -= 1
  if (t.length < 40) score -= 1
  return score >= 3 ? 'keep' : score >= 1 ? 'summarize' : 'melt'
}
const tagAll = (msgs) => msgs.map(tagMessage)

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
  return present / keywords.length
}

// ─── one config-run: returns {bfly_frac, lastn_frac, delta} ─────
function runConfig(coreObj, totalMsgs, budget, gens) {
  let bflyMessages = buildTranscript(coreObj, totalMsgs).slice()
  let lastnMessages = buildTranscript(coreObj, totalMsgs).slice()
  let bflyFinal = ''
  let lastNFinal = ''

  for (let g = 1; g <= gens; g++) {
    const tags = tagAll(bflyMessages)
    bflyFinal = chrysalis(bflyMessages, tags, budget)
    const truncated = pickLastN(lastnMessages, budget)
    lastNFinal = truncated.map(m => `[${m.role}] ${m.content}`).join('\n')

    if (g < gens) {
      const noise = NOISE_PER_GEN[(g - 1) % NOISE_PER_GEN.length]
      bflyMessages = [{ role: 'assistant', content: `[REBUILT FROM GEN ${g}]\n${bflyFinal}` }, ...noise]
      lastnMessages = [...truncated, ...noise]
    }
  }
  return {
    bfly_frac: scoreKW(bflyFinal, coreObj.needle_keywords),
    lastn_frac: scoreKW(lastNFinal, coreObj.needle_keywords),
  }
}

// ─── sweep all (transcript × length × budget × gens) ────────────
function main() {
  const cores = TRANSCRIPTS.map(id => CORES.find(c => c.id === id)).filter(Boolean)
  if (cores.length === 0) { console.error('no matching transcripts'); process.exit(1) }

  const t0 = Date.now()
  const cells = []   // flat: {transcript, length, budget, gens, bfly_frac, lastn_frac, delta}

  for (const core of cores) {
    for (const len of LENGTHS) {
      for (const budget of BUDGETS) {
        for (const gens of GENS) {
          const r = runConfig(core, len, budget, gens)
          cells.push({
            transcript: core.id,
            length: len, budget, gens,
            bfly_frac: +r.bfly_frac.toFixed(3),
            lastn_frac: +r.lastn_frac.toFixed(3),
            delta: +(r.bfly_frac - r.lastn_frac).toFixed(3),
          })
        }
      }
    }
  }

  const ms = Date.now() - t0
  console.log(`[bfly-phase] swept ${cells.length} cells (${cores.length} transcripts × ${LENGTHS.length} lengths × ${BUDGETS.length} budgets × ${GENS.length} gens) in ${ms}ms`)
  console.log()

  // Per-(length, budget, gens) means across the 4 transcripts: that's the phase diagram.
  // For each gens, print a 2-D heatmap of mean delta over (length rows × budget cols).
  function meanDeltaFor(len, budget, gens) {
    const subset = cells.filter(c => c.length === len && c.budget === budget && c.gens === gens)
    return subset.reduce((s, c) => s + c.delta, 0) / subset.length
  }

  // ASCII heatmap glyphs by delta magnitude
  function glyph(d) {
    if (d <= -0.30) return '◀◀'
    if (d <= -0.10) return ' ◀'
    if (d <  0.10)  return ' ·'
    if (d <  0.30)  return ' ▸'
    if (d <  0.60)  return '▸▸'
    return '██'
  }

  for (const g of GENS) {
    console.log(`╔═ gens = ${g} ═══════════════════════════════════════════════╗`)
    const hdr = 'len ↓ \\ budget →   ' + BUDGETS.map(b => String(b).padStart(4)).join(' ')
    console.log(hdr)
    for (const len of LENGTHS) {
      const row = String(len).padStart(4) + '              ' +
        BUDGETS.map(b => {
          const d = meanDeltaFor(len, b, g)
          return glyph(d).padStart(4)
        }).join(' ')
      console.log(row)
    }
    console.log()
  }

  console.log('legend: ██ Δ≥0.60   ▸▸ 0.30..0.60   ▸ 0.10..0.30   · -0.10..0.10   ◀ -0.30..-0.10   ◀◀ ≤-0.30')

  // ─── distill: where does butterfly start beating lastN? ─────────
  // For each gens, find the (length, budget) pairs where mean Δ ≥ 0.30
  // ("clear win" — meaningfully above the symmetric noise band).
  console.log()
  console.log('── clear-win region per generation (mean Δ ≥ 0.30 across 4 transcripts) ──')
  for (const g of GENS) {
    const wins = []
    for (const len of LENGTHS) {
      for (const budget of BUDGETS) {
        if (meanDeltaFor(len, budget, g) >= 0.30) wins.push(`len=${len},budget=${budget}`)
      }
    }
    console.log(`  gens=${g}: ${wins.length === 0 ? '(none)' : wins.length + ' cells — ' + wins.slice(0, 6).join(' · ') + (wins.length > 6 ? ` … +${wins.length - 6}` : '')}`)
  }

  // Persist
  mkdirSync(RESULTS_DIR, { recursive: true })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const out = join(RESULTS_DIR, `butterfly-phasediagram-${stamp}.json`)
  writeFileSync(out, JSON.stringify({
    started_at: new Date(t0).toISOString(),
    finished_at: new Date().toISOString(),
    runtime_ms: ms,
    config: { budgets: BUDGETS, lengths: LENGTHS, gens: GENS, transcripts: TRANSCRIPTS },
    cells,
  }, null, 2))
  console.log(`\nwrote: ${out}`)
}

main()

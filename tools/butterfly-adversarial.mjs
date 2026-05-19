#!/usr/bin/env node
// Butterfly v3.7 — adversarial transcripts.
//
// The first 4 transcripts had needles that took specific SHAPES: file
// paths (lib/jwt.ts), Slack channels (#auth-platform), package mentions
// (@company/auth-service), ticket IDs (TICKET-4421), line ranges
// (lines 142-148). The regex tagger's selectivity bias toward those
// exact patterns is what carried the v3.1 confirmation.
//
// To stress that bias: 4 new transcripts where the needle is real
// load-bearing content but does NOT take any of those shapes. The
// hypothesis: the regex (and any tagger trained on regex labels)
// will UNDERPERFORM on these. If true, it sharpens "the mechanism's
// win depends on tagger-vs-needle SHAPE-match" to a falsifiable
// claim with positive AND negative cases.
//
// Adversarial needle types:
//   - numeric-threshold-in-prose:  "we agreed on a hard cap of 47
//                                   concurrent connections per pod"
//   - implicit-deadline:            "cooper said end of next week,
//                                    so by friday the 24th"
//   - preference-statement:         "i don't want to go with postgres
//                                    for this — too heavy for the
//                                    write pattern we have"
//   - buried-causation:             "the rollback brought back the
//                                    version-pinned dependencies that
//                                    were the actual blocker"
//
// Usage:
//   node tools/butterfly-adversarial.mjs                     # all 5 taggers
//   STRATEGIES=regex,trained,embed node tools/butterfly-adversarial.mjs
//   DEBUG=1 node tools/butterfly-adversarial.mjs             # show selected msgs

import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const RESULTS_DIR = join(ROOT, 'test-results', 'butterfly-sweep')
const LMS_BASE = process.env.LMS_BASE || 'http://localhost:1234/v1'
const DEBUG = process.env.DEBUG === '1'
const TARGET_TOKENS = 100
const N_GENERATIONS = 3
const tokens = (s) => Math.ceil(s.length / 4)

// ─── adversarial transcripts ─────────────────────────────────────
// Needle keywords are PHRASES (not identifier shapes) — substring-
// matched case-insensitive. Each transcript's `core` puts the needle
// in message ~3-5 (mid-conversation), surrounded by genuine context.
const ADV_CORES = [
  {
    id: 'numeric-threshold',
    name: 'Numeric threshold in prose · no file paths',
    needle_keywords: ['47', 'concurrent connections', 'per pod', 'hard cap'],
    core: [
      { role: 'user',      content: "the new ingestion worker keeps falling over under load. think we need to cap concurrency somewhere" },
      { role: 'assistant', content: "what's the failure look like — OOM, or timeouts on the downstream call?" },
      { role: 'user',      content: "neither really, the worker just starts dropping events silently. saw it in the metrics yesterday." },
      { role: 'assistant', content: "ok so we need backpressure not just throttling. after digging into the worker config — we agreed on a hard cap of 47 concurrent connections per pod, anything above that goes onto an internal queue with a 30 second wait. matches what the downstream can actually absorb without buffering its own queue out of memory." },
      { role: 'user',      content: "wait was 47 the figure? not 50 or something round?" },
      { role: 'assistant', content: "yeah 47 specifically. came from the downstream team's own load tests — they hit a knee in latency at 50 per upstream caller and wanted us to stay one step below that." },
      { role: 'user',      content: "got it. you'll wire that into the worker config?" },
      { role: 'assistant', content: "already drafted the PR, just needs sign-off." },
    ],
  },
  {
    id: 'implicit-deadline',
    name: 'Implicit deadline · names, no dates',
    needle_keywords: ['cooper', 'end of next week', 'friday the 24th'],
    core: [
      { role: 'user',      content: "do we have a real ship date for the migration? marketing keeps asking and i'm tired of hedging." },
      { role: 'assistant', content: "the comms team wanted something by EOM but i don't think that's actually pinned." },
      { role: 'user',      content: "right so what IS pinned then" },
      { role: 'assistant', content: "talked to cooper after the leads meeting yesterday — he said end of next week, so by friday the 24th, full cutover. that's the only hard date that came out of that conversation. anything else is aspirational." },
      { role: 'user',      content: "ok so i can tell marketing the 24th" },
      { role: 'assistant', content: "yes, with the caveat that it's a soft cutover — old endpoints stay alive for two weeks after for rollback." },
      { role: 'user',      content: "good enough. they just need a date to put on a banner." },
      { role: 'assistant', content: "noted. i'll send cooper a confirmation message so the date is on record somewhere too." },
    ],
  },
  {
    id: 'preference-statement',
    name: 'Stated preference · architectural choice, no decision marker',
    needle_keywords: ['postgres', 'too heavy', 'write pattern', "don't want"],
    core: [
      { role: 'user',      content: "for the event log store — we leaning postgres? everyone keeps suggesting it" },
      { role: 'assistant', content: "it's the default everyone reaches for but i don't think it's actually right here" },
      { role: 'user',      content: "what's your concern" },
      { role: 'assistant', content: "honestly i don't want to go with postgres for this — too heavy for the write pattern we have. we're talking burst writes of 8-10k events/sec with no reads for hours at a time, then a giant batch read. postgres can do it but you end up tuning autovacuum and wal segments to within an inch of their life. there are stores designed for exactly this shape." },
      { role: 'user',      content: "like what" },
      { role: 'assistant', content: "clickhouse is the obvious one. scylla if we wanted to be exotic. even a tiered s3+parquet setup if we want to skip the database entirely." },
      { role: 'user',      content: "ok worth a real evaluation then. can you write up the comparison?" },
      { role: 'assistant', content: "yeah will have a doc by end of week." },
    ],
  },
  {
    id: 'buried-causation',
    name: 'Buried causation · root cause without "Root cause:" marker',
    needle_keywords: ['rollback', 'version-pinned', 'dependencies', 'actual blocker'],
    core: [
      { role: 'user',      content: "ci broke after the restore last night. tests pass locally but fail in pipeline." },
      { role: 'assistant', content: "which suite, and what error?" },
      { role: 'user',      content: "the integration suite. the auth tests can't reach the user service" },
      { role: 'assistant', content: "looked at the container image — the rollback brought back the version-pinned dependencies that were the actual blocker. specifically the old grpc-tools version mismatches what our generated client code expects, so the service starts but rejects requests with a protocol error. nothing wrong with the tests themselves." },
      { role: 'user',      content: "so what's the fix — re-pin or re-generate?" },
      { role: 'assistant', content: "re-pin to the version we had before the rollback. the codegen is downstream of that pin so it'll be consistent again once we do." },
      { role: 'user',      content: "want me to do it or you got it" },
      { role: 'assistant', content: "i'll handle it, it's faster than explaining the context." },
    ],
  },
]

// ─── padding + noise (reused, sanitized to avoid leaking any
// adversarial-needle keywords as substrings) ────────────────────────
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
    { role: 'user',      content: "side thing — should we use prettier or biome for formatting?" },
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

function buildTranscript(core) {
  // 15 pre + core + 15 post — same length as v3.1 (~38 messages).
  return [...PADDING_POOL.slice(0, 15), ...core, ...PADDING_POOL.slice(15)]
}

// ─── 4 taggers (regex, trained, embed, llm-onechar) ──────────────
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
  if (t.content && t.endsWith('?') && t.length < 50 && !/\w+\/[\w.]+/.test(t)) score -= 1
  if (t.length < 40) score -= 1
  return score >= 3 ? 'keep' : score >= 1 ? 'summarize' : 'melt'
}

const TRAINED_FEATURES = [
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
function softmax(z) { const m = Math.max(...z); const e = z.map(v => Math.exp(v-m)); const s = e.reduce((a,b)=>a+b,0); return e.map(v=>v/s) }
function trainedTag(msg, w) {
  const x = TRAINED_FEATURES.map(f => f(msg.content))
  const z = w.W.map((wk, k) => wk.reduce((s, wv, j) => s + wv * x[j], w.b[k]))
  const p = softmax(z)
  let argmax = 0
  for (let k = 1; k < p.length; k++) if (p[k] > p[argmax]) argmax = k
  return w.classes[argmax]
}

let embedCache = null
function loadEmbedCache() {
  if (embedCache) return embedCache
  const cpath = join(ROOT, 'tools', 'butterfly-embed-cache.json')
  embedCache = existsSync(cpath) ? JSON.parse(readFileSync(cpath, 'utf8')) : {}
  return embedCache
}
async function fetchEmbed(text, model) {
  const cache = loadEmbedCache()
  if (cache[text]) return cache[text]
  const res = await fetch(`${LMS_BASE}/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, input: text }),
  })
  if (!res.ok) throw new Error(`embed ${res.status}`)
  const j = await res.json()
  const v = j.data[0].embedding
  cache[text] = v
  // persist incrementally
  try { writeFileSync(join(ROOT, 'tools', 'butterfly-embed-cache.json'), JSON.stringify(cache)) } catch {}
  return v
}
async function embedTag(msg, w) {
  const x = await fetchEmbed(msg.content, w.embed_model)
  const z = w.W.map((wk, k) => wk.reduce((s, wv, j) => s + wv * x[j], w.b[k]))
  const p = softmax(z)
  let argmax = 0
  for (let k = 1; k < p.length; k++) if (p[k] > p[argmax]) argmax = k
  return w.classes[argmax]
}

async function llmTagOneChar(messages, model = 'qwen3-14b-mlx', max_keeps = 5) {
  const list = messages.map((m, i) => `[${i.toString().padStart(2, '0')} ${m.role}] ${m.content}`).join('\n')
  const sys = `You label conversation messages for context compaction at a tight token budget.
For each message output ONE character:
  k = keep      — irreplaceable atom an engineer must act on (named fact, decision, specific concrete content)
  s = summarize — substantive but a one-line gist suffices
  m = melt      — greetings, acks, restatements, off-topic. Most messages are melt.

HARD CAP: at most ${max_keeps} 'k' tags total. Rest must be 's' or 'm'.
OUTPUT FORMAT: ${messages.length} characters total, one per message, in order, no spaces, no newlines.`
  const res = await fetch(`${LMS_BASE}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [{ role: 'system', content: sys }, { role: 'user', content: `/no_think ${list}` }],
      max_tokens: 200, temperature: 0.0,
    }),
  })
  if (!res.ok) return messages.map(regexTag)  // fallback
  const j = await res.json()
  const raw = (j.choices?.[0]?.message?.content || '').replace(/<think>[\s\S]*?<\/think>/g, '')
  const cleaned = raw.replace(/[^ksm]/gi, '').toLowerCase()
  const tags = messages.map((m, i) => {
    const c = cleaned[i]
    return c === 'k' ? 'keep' : c === 's' ? 'summarize' : c === 'm' ? 'melt' : regexTag(m)
  })
  // enforce cap
  let keepCount = tags.filter(t => t === 'keep').length
  if (keepCount > max_keeps) {
    let excess = keepCount - max_keeps
    for (let i = tags.length - 1; i >= 0 && excess > 0; i--) if (tags[i] === 'keep') { tags[i] = 'summarize'; excess-- }
  }
  return tags
}

// ─── pipeline (regex, trained, embed run sync; llm-onechar async) ─
function firstSentence(s) { const m = s.match(/^[^.!?]*[.!?]/); return (m ? m[0] : s).trim() }
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
  const out = []; let acc = 0
  for (let i = messages.length - 1; i >= 0; i--) {
    const t = tokens(messages[i].content)
    if (acc + t > budget && out.length > 0) break
    out.unshift(messages[i]); acc += t
  }
  return out
}
function scoreKW(memory, keywords) {
  const lower = memory.toLowerCase()
  const present = keywords.filter(k => lower.includes(k.toLowerCase())).length
  return present / keywords.length
}

async function runOne(core, taggerName, taggerFn) {
  let bflyMessages = buildTranscript(core.core).slice()
  let lastnMessages = buildTranscript(core.core).slice()
  let bflyFinal = ''
  let lastNFinal = ''
  const tag_traces = []

  for (let g = 1; g <= N_GENERATIONS; g++) {
    const tags = await taggerFn(bflyMessages)
    const dist = tags.reduce((acc, t) => ({ ...acc, [t]: (acc[t] || 0) + 1 }), {})
    tag_traces.push({ gen: g, dist, n: bflyMessages.length })
    bflyFinal = chrysalis(bflyMessages, tags, TARGET_TOKENS)
    const truncated = pickLastN(lastnMessages, TARGET_TOKENS)
    lastNFinal = truncated.map(m => `[${m.role}] ${m.content}`).join('\n')
    if (g < N_GENERATIONS) {
      const noise = NOISE_PER_GEN[(g - 1) % NOISE_PER_GEN.length]
      bflyMessages = [{ role: 'assistant', content: `[REBUILT FROM GEN ${g}]\n${bflyFinal}` }, ...noise]
      lastnMessages = [...truncated, ...noise]
    }
  }
  const bfly = scoreKW(bflyFinal, core.needle_keywords)
  const lastn = scoreKW(lastNFinal, core.needle_keywords)
  return { transcript: core.id, tagger: taggerName, bfly, lastn, delta: bfly - lastn, tag_traces, bfly_memory: bflyFinal }
}

// ─── main ────────────────────────────────────────────────────────
async function main() {
  const STRATEGIES = (process.env.STRATEGIES || 'regex,trained,embed,llm-onechar').split(',')

  // Pre-load trained + embed weights so failures surface up front
  let trainedW = null, embedW = null
  if (STRATEGIES.includes('trained')) {
    trainedW = JSON.parse(readFileSync(join(ROOT, 'tools', 'butterfly-classifier-weights.json'), 'utf8'))
  }
  if (STRATEGIES.includes('embed')) {
    embedW = JSON.parse(readFileSync(join(ROOT, 'tools', 'butterfly-embed-weights.json'), 'utf8'))
  }

  const taggers = {
    regex: (msgs) => msgs.map(regexTag),
    trained: (msgs) => msgs.map(m => trainedTag(m, trainedW)),
    embed: async (msgs) => { const out = []; for (const m of msgs) out.push(await embedTag(m, embedW)); return out },
    'llm-onechar': (msgs) => llmTagOneChar(msgs, 'qwen3-14b-mlx', 5),
  }

  const t0 = Date.now()
  const rows = []
  console.log(`[adversarial] transcripts=${ADV_CORES.length} taggers=${STRATEGIES.join(',')} budget=${TARGET_TOKENS} gens=${N_GENERATIONS}`)
  for (const core of ADV_CORES) {
    console.log(`\n── ${core.id} (${core.name}) ──`)
    console.log(`   needle: [${core.needle_keywords.join(' · ')}]`)
    for (const sname of STRATEGIES) {
      const tagger = taggers[sname]
      if (!tagger) { console.log(`   (unknown tagger: ${sname})`); continue }
      const t1 = Date.now()
      const r = await runOne(core, sname, tagger)
      const dt = ((Date.now() - t1) / 1000).toFixed(1)
      rows.push(r)
      console.log(`   ${sname.padEnd(12)} bfly=${(r.bfly*100).toFixed(0).padStart(3)}%  lastN=${(r.lastn*100).toFixed(0).padStart(3)}%  Δ=${(r.delta*100).toFixed(0).padStart(4)}pp  ${dt}s  gen-tags=${JSON.stringify(r.tag_traces[0].dist)}`)
      if (DEBUG) console.log(`     memory: ${r.bfly_memory.slice(0, 240)}`)
    }
  }
  const ms = Date.now() - t0
  console.log(`\n[adversarial] ${rows.length} runs in ${(ms/1000).toFixed(1)}s`)

  // Per-tagger mean delta
  console.log(`\n── mean Δ per tagger across 4 adversarial transcripts ──`)
  for (const s of STRATEGIES) {
    const subset = rows.filter(r => r.tagger === s)
    if (subset.length === 0) continue
    const mean = subset.reduce((sum, r) => sum + r.delta, 0) / subset.length
    const beats = subset.filter(r => r.bfly > r.lastn).length
    console.log(`  ${s.padEnd(12)} mean Δ = ${(mean*100).toFixed(0).padStart(4)}pp  ·  bfly > lastN on ${beats}/${subset.length}`)
  }

  mkdirSync(RESULTS_DIR, { recursive: true })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const out = join(RESULTS_DIR, `butterfly-adversarial-${stamp}.json`)
  writeFileSync(out, JSON.stringify({
    started_at: new Date(t0).toISOString(),
    finished_at: new Date().toISOString(),
    runtime_ms: ms,
    config: { target_tokens: TARGET_TOKENS, n_generations: N_GENERATIONS, strategies: STRATEGIES },
    runs: rows,
  }, null, 2))
  console.log(`\nwrote: ${out}`)
}

main().catch(e => { console.error('FATAL:', e); process.exit(1) })

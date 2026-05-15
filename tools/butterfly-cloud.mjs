#!/usr/bin/env node
// Butterfly v2.6 — local-LM-Studio runner. No browser, no WebGPU, no
// Mac-sleep problem. Same 4 transcripts + same protocol from
// src/butterfly-mode.ts (well, a simplified 1-generation version of it),
// but routes all model calls through LM Studio's OpenAI-compatible
// endpoint on :1234. Defaults to gemma-4-e4b (small open model, same
// spirit as Phi-3-mini) so the experiment is still in the "can a tiny
// model do principled context compaction" lane.
//
// Simplified vs the browser butterfly (which followed the
// 3-generation pre-registered protocol):
//   - N_GENERATIONS = 1 (vs 3) — first answer the simpler question
//     "does butterfly beat lastN at one compaction?" before scaling.
//   - Tagger: ONE batched call returning a JSON array of {idx,label,reason}
//     for all messages, instead of one call per message.
//     This reduces 8 tagger calls/generation to 1.
//   - Everything else (chrysalis, 2 answers, 2 judges) is identical.
//   - Output JSON schema matches the browser sweep so the existing
//     tools/grade-butterfly.mjs scoring works unchanged.
//
// Usage:
//   node tools/butterfly-cloud.mjs                       # full sweep, runsPer=5
//   RUNS_PER=10 node tools/butterfly-cloud.mjs           # 10 per transcript
//   MODEL=qwen3-14b-mlx node tools/butterfly-cloud.mjs   # different model
//   TRANSCRIPTS=jwt-clock-race node tools/butterfly-cloud.mjs  # just one

import { writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const RESULTS_DIR = join(ROOT, 'test-results', 'butterfly-sweep')

// ─── config ───────────────────────────────────────────────────────
const LMS_BASE  = process.env.LMS_BASE  || 'http://localhost:1234/v1'
const MODEL     = process.env.MODEL     || 'google/gemma-4-e4b'
const RUNS_PER  = parseInt(process.env.RUNS_PER || '5', 10)
const TARGET_TOKENS = 400
const CHRYSALIS_MAX = 600
const ANSWER_MAX = 200
const JUDGE_MAX = 80

// ─── transcripts (mirror of BUILT_IN_TRANSCRIPTS in butterfly-mode.ts) ─
const BUILT_INS = [
  {
    id: 'jwt-clock-race',
    name: 'JWT clock race · root cause',
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
    question: "Looking at lib/jwt.ts specifically — what is the exact code mistake that causes the off-by-one second in CI? Name the function and what it does wrong.",
    fact: "issueToken in lib/jwt.ts reads Date.now() once to compute the `exp` field. The test assertion ALSO reads Date.now() (a second time). On slow CI those two reads can land on different seconds, causing the off-by-one. The mistake is the second Date.now() read in the test — there should be one captured `now` shared between the call and the assertion.",
  },
  {
    id: 'auth-owner-pto',
    name: 'Auth-service owner · ownership info',
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
    question: "We need to coordinate the auth-service JWT format change before Friday but we can't ping Sarah. Who is the backup reviewer for auth-service, and what is the right Slack channel to reach them on?",
    fact: "Marcus Lee is the backup reviewer for @company/auth-service while Sarah Chen is on PTO through Tuesday. The Slack channel is #auth-platform.",
  },
  {
    id: 'rate-limit-decision',
    name: 'Rate-limit algorithm · decision',
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
    question: "For the /api/v1 rate-limit decision in TICKET-4421: what is the primary algorithm and its parameters, and what specifically happens when Redis is unreachable?",
    fact: "Token bucket at 100 req/min per IP via Redis (refill every 600ms). When Redis is unreachable for more than 5 seconds, the system fails open to a local in-memory fixed-window limiter at 50 req/min, accepting brief over-limit rather than hard-failing user requests.",
  },
  {
    id: 'cache-race-fileline',
    name: 'Cache race · file:line location',
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
    question: "Where exactly in the codebase is the read-after-write race, and what is the precise sequence of operations that causes it?",
    fact: "The race is in lib/cache.ts at lines 142-148. cache.set updates the value, then the version-counter increment happens two statements later. A concurrent cache.get landing between those two statements sees the new value but the old version number, so the downstream consumer treats it as still-cached. The fix is to move the version increment into the same critical section, or use a CAS on the version field.",
  },
]

const NOISE = [
  { role: 'user',      content: "while we're here — should we use prettier or biome for formatting?" },
  { role: 'assistant', content: "biome is faster but prettier has wider plugin ecosystem. Either works." },
  { role: 'user',      content: "ok I'll punt on it. also CI build time crept up to 4 min, want to look later." },
  { role: 'assistant', content: "Probably the snapshot suite growing. We can split it next sprint." },
]

// ─── helpers ──────────────────────────────────────────────────────
const tokens = (s) => Math.ceil(s.length / 4)
const asText = (msgs) => msgs.map(m => `[${m.role}]\n${m.content}`).join('\n\n')

function pickLastN(msgs, budget) {
  const out = []
  let acc = 0
  for (let i = msgs.length - 1; i >= 0; i--) {
    const t = tokens(msgs[i].content)
    if (acc + t > budget && out.length > 0) break
    out.unshift(msgs[i])
    acc += t
  }
  return out
}

// Detect thinking-mode models (Qwen3, DeepSeek-R1, etc.) and disable
// chain-of-thought so the token budget goes to the actual answer.
//   - Qwen3 supports `/no_think` as a user-message control token.
//   - We also strip `<think>...</think>` blocks from any response that
//     leaks them, and fall back to `reasoning_content` if `content` is
//     empty but reasoning_content has the answer at the end.
const THINKING_MODELS = /(qwen3|deepseek-r1|r1-)/i

function applyNoThink(messages) {
  if (!THINKING_MODELS.test(MODEL)) return messages
  return messages.map((m, i, arr) => {
    if (m.role !== 'user' || i !== arr.length - 1) return m
    return { ...m, content: `/no_think ${m.content}` }
  })
}

function extractAnswer(msg) {
  const content = (msg?.content ?? '').trim()
  if (content) {
    // Strip any leftover <think>...</think> blocks the model may have emitted.
    return content.replace(/<think>[\s\S]*?<\/think>/g, '').trim()
  }
  // Thinking-mode models with content="" sometimes put the answer at the
  // end of reasoning_content as a final paragraph. Best-effort recovery.
  const reasoning = (msg?.reasoning_content ?? '').trim()
  if (reasoning) {
    const lines = reasoning.split('\n').filter(Boolean)
    return lines.slice(-3).join('\n').trim()
  }
  return ''
}

async function lmsChat(messages, max_tokens, temperature = 0.0) {
  const t0 = Date.now()
  const res = await fetch(`${LMS_BASE}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: MODEL, messages: applyNoThink(messages), max_tokens, temperature }),
  })
  if (!res.ok) throw new Error(`LM Studio ${res.status}: ${await res.text()}`)
  const j = await res.json()
  const msg = j.choices?.[0]?.message
  return { text: extractAnswer(msg), ms: Date.now() - t0, usage: j.usage }
}

// Batched tagger — one prompt classifies ALL messages in the transcript.
async function tagAll(transcript) {
  const list = transcript.map((m, i) => `[#${i} ${m.role}] ${m.content}`).join('\n\n')
  const sys = `You classify each message in a conversation for context compaction.

For each message, choose exactly one label:
  keep      = irreplaceable atom — root cause named, owner+channel, file:line, decision, code snippet
  summarize = substantive but a one-line gist suffices (multi-step explanation, verbose tool dump with one fact)
  melt      = greetings, acks, "lgtm/ok/sure", restatements, dead-end tangents, polite framing

Distribution prior: most messages are MELT. Use KEEP only when the message contains something irreplaceable.

Return a JSON array, one object per message, in order:
[{"idx": 0, "label": "keep", "reason": "4-7 word reason"}, ...]

Output ONLY the JSON array. No preamble, no markdown fence.`
  const r = await lmsChat([
    { role: 'system', content: sys },
    { role: 'user', content: list },
  ], 800, 0.0)
  // Strip code fences if model added them
  let raw = r.text.trim()
  raw = raw.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '').trim()
  // Find the first '[' and last ']' to extract the JSON array
  const start = raw.indexOf('[')
  const end = raw.lastIndexOf(']')
  if (start === -1 || end === -1) return { tags: transcript.map((_, i) => ({ idx: i, label: 'keep', reason: '(parse fail)' })), ms: r.ms, raw }
  try {
    const parsed = JSON.parse(raw.slice(start, end + 1))
    return { tags: parsed, ms: r.ms, raw }
  } catch {
    return { tags: transcript.map((_, i) => ({ idx: i, label: 'keep', reason: '(parse fail)' })), ms: r.ms, raw }
  }
}

async function chrysalis(transcript, tags, target) {
  const tagged = transcript.map((m, i) => `[#${i} ${m.role} action=${tags[i]?.label ?? 'keep'}]\n${m.content}`).join('\n\n')
  const sys = `You rebuild a tagged conversation transcript into a small coherent context the agent will resume from.

HARD CONSTRAINT: ~${target} tokens (~${target * 4} chars).
KEEP messages: preserve every load-bearing fact, name, file:line, decision.
SUMMARIZE messages: collapse to one phrase.
MELT messages: drop entirely.

Output ONLY the rebuilt context, no preamble.`
  return lmsChat([
    { role: 'system', content: sys },
    { role: 'user', content: `TAGGED TRANSCRIPT:\n\n${tagged}` },
  ], CHRYSALIS_MAX, 0.0)
}

async function answer(context, question) {
  const sys = `You continue a prior conversation. Answer the follow-up using ONLY the prior context. If a fact isn't there, say so plainly — do not invent. Be concise.`
  return lmsChat([
    { role: 'system', content: sys },
    { role: 'user', content: `PRIOR CONTEXT:\n\n${context}\n\nFOLLOW-UP: ${question}` },
  ], ANSWER_MAX, 0.0)
}

async function judge(question, expectedFact, answerText) {
  const sys = `You grade whether an answer accurately conveys an expected fact for context-compaction evaluation.

Return STRICTLY a single digit at the END of your response:
  2 = HIT     — answer accurately conveys the full expected fact (paraphrasing fine)
  1 = PARTIAL — answer preserves the load-bearing identification (file, function, mechanism — what an engineer needs to act) but mis-states or omits a detail
  0 = MISS    — fact missing, vague, or invented

The LAST CHARACTER of your reply MUST be 0, 1, or 2. Nothing after.`
  const r = await lmsChat([
    { role: 'system', content: sys },
    { role: 'user', content: `QUESTION: ${question}\nEXPECTED FACT: ${expectedFact}\nANSWER:\n${answerText}\n\nGrade. End with 0, 1, or 2.` },
  ], JUDGE_MAX, 0.0)
  // Find LAST occurrence of 0/1/2 in the response
  const m = r.text.match(/[012](?!.*[012])/s)
  const score = m ? Number(m[0]) : 0
  return { verdict: score === 2 ? 'hit' : score === 1 ? 'partial' : 'miss', ms: r.ms, raw: r.text }
}

// ─── one butterfly run ─────────────────────────────────────────────
async function runOne(t) {
  const t0 = Date.now()
  // Simplified: 1 generation. Tag the transcript + noise, rebuild,
  // then ask both arms at the same TARGET_TOKENS budget.
  const transcriptWithNoise = [...t.transcript, ...NOISE]

  const { tags } = await tagAll(transcriptWithNoise)
  const { text: rebuilt } = await chrysalis(transcriptWithNoise, tags, TARGET_TOKENS)

  const lastN = pickLastN(transcriptWithNoise, TARGET_TOKENS)
  const ctxBfly = `[REBUILT CONTEXT]\n${rebuilt}`
  const ctxLast = asText(lastN)

  const [{ text: ansBfly }, { text: ansLast }] = await Promise.all([
    answer(ctxBfly, t.question),
    answer(ctxLast, t.question),
  ])

  const [vBfly, vLast] = await Promise.all([
    judge(t.question, t.fact, ansBfly),
    judge(t.question, t.fact, ansLast),
  ])

  return {
    transcript: t.id,
    bfly: vBfly.verdict,
    lastn: vLast.verdict,
    seconds: +((Date.now() - t0) / 1000).toFixed(1),
    tags_summary: tags.reduce((acc, x) => ({ ...acc, [x.label]: (acc[x.label] || 0) + 1 }), {}),
    rebuilt_tokens: tokens(rebuilt),
    lastn_tokens: tokens(ctxLast),
    ans_bfly_excerpt: ansBfly.slice(0, 180),
    ans_lastn_excerpt: ansLast.slice(0, 180),
  }
}

// ─── sweep ────────────────────────────────────────────────────────
async function main() {
  console.log(`[bfly-cloud] model=${MODEL}  endpoint=${LMS_BASE}  runsPer=${RUNS_PER}`)
  const transcripts = process.env.TRANSCRIPTS
    ? process.env.TRANSCRIPTS.split(',').map(id => BUILT_INS.find(t => t.id === id)).filter(Boolean)
    : BUILT_INS
  if (transcripts.length === 0) { console.error('no matching transcripts'); process.exit(1) }

  const results = {
    started_at: new Date().toISOString(),
    runs_per_transcript: RUNS_PER,
    transcripts: transcripts.map(t => t.id),
    fingerprint: { ua: 'node:butterfly-cloud', model: MODEL, endpoint: LMS_BASE },
    runs: [],
  }

  for (const t of transcripts) {
    console.log(`\n[bfly-cloud] ── ${t.id} ──`)
    for (let i = 1; i <= RUNS_PER; i++) {
      try {
        const r = await runOne(t)
        results.runs.push({ ...r, run: i })
        console.log(`[bfly-cloud]   ${i}/${RUNS_PER}  bfly=${r.bfly}  lastN=${r.lastn}  ${r.seconds}s  tags=${JSON.stringify(r.tags_summary)}`)
      } catch (e) {
        console.error(`[bfly-cloud]   ${i}/${RUNS_PER}  ERROR: ${e.message}`)
        results.runs.push({ transcript: t.id, run: i, bfly: 'miss', lastn: 'miss', seconds: -1, status: e.message.slice(0, 200) })
      }
    }
  }

  results.finished_at = new Date().toISOString()
  mkdirSync(RESULTS_DIR, { recursive: true })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const out = join(RESULTS_DIR, `butterfly-cloud-${stamp}.json`)
  writeFileSync(out, JSON.stringify(results, null, 2))
  console.log(`\n[bfly-cloud] wrote ${out}`)
  console.log(`[bfly-cloud] ${results.runs.length} runs · ${((new Date(results.finished_at) - new Date(results.started_at)) / 1000).toFixed(1)}s total`)
}

main().catch((e) => { console.error('FATAL:', e); process.exit(1) })

#!/usr/bin/env node
// Butterfly v3.6 — train a 768-dim embedding + linear head classifier.
// Tests whether embedding-based similarity captures the same selectivity
// as the regex's hand-engineered features. If the linear head trained on
// regex labels matches regex performance, embeddings "see" identifier shape.
// If it underperforms, the regex's structural awareness isn't in the
// embedding space.
//
// Architecture: nomic-embed-text-v1.5 (768-dim) → 768×3 softmax
//   - Total parameters: 768×3 + 3 = 2,307
//   - Serialized weights: ~10 KB JSON
//   - Inference per message: ~1 ms embedding call + ~1 µs matmul
//
// Embeddings are cached to disk to avoid re-embedding the same 100
// training messages every run.
//
// Usage:
//   node tools/butterfly-train-embed.mjs
//   DEBUG=1 node tools/butterfly-train-embed.mjs

import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const WEIGHTS_PATH = join(ROOT, 'tools', 'butterfly-embed-weights.json')
const CACHE_PATH = join(ROOT, 'tools', 'butterfly-embed-cache.json')
const LMS_BASE = process.env.LMS_BASE || 'http://localhost:1234/v1'
const EMBED_MODEL = process.env.EMBED_MODEL || 'text-embedding-nomic-embed-text-v1.5'
const DEBUG = process.env.DEBUG === '1'

// ─── ground-truth labeler (same regex as v3.5) ───────────────────
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

const N_CLASSES = 3
const LABEL_IDX = { keep: 0, summarize: 1, melt: 2 }
const IDX_LABEL = ['keep', 'summarize', 'melt']

// ─── training texts (same set as v3.5; embeddings expensive — cache hard) ─
const CORES = [
  "Our auth/session.test.ts is flaky in CI but passes locally. Help?",
  "Sure — share the test and the failure output.",
  "Share the test output.",
  "It asserts decoded.exp equals Math.floor(Date.now()/1000) + 3600. CI says expected 1735689600 received 1735689599.",
  "Off-by-one second. Smells like a timing race rather than a logic bug.",
  "Off-by-one second. Timing race rather than logic bug.",
  "Confirmed in lib/jwt.ts: issueToken reads Date.now() once for `exp`, the test reads Date.now() again in the assertion. On slow CI those two reads can land on different seconds. Root cause: clock race, not a code bug.",
  "Confirmed in lib/jwt.ts: issueToken reads Date.now() once for `exp`, the test reads Date.now() again in the assertion. On slow CI those two reads land on different seconds. Root cause: clock race.",
  "ok so the fix?",
  "Two options. Cheap: capture Date.now() once before calling issueToken. Proper: have issueToken accept an optional `now` parameter so tests inject a fixed clock.",
  "Capture Date.now() once before calling issueToken, or have issueToken accept an optional `now` parameter so tests inject a fixed clock.",
  "let's do the proper one. lgtm, pushing.",
  "let's do the proper one.",
  "We're shipping the new checkout API Friday. Need someone on auth-service to bless the JWT format change.",
  "auth-service team — let me pull their owner.",
  "Let me pull their owner.",
  "Slack channel works too if you have it.",
  "Sarah Chen owns @company/auth-service — Slack is #auth-platform. She's PTO through Tuesday, backup reviewer is Marcus Lee.",
  "Sarah Chen owns @company/auth-service — Slack #auth-platform. She's out through Tuesday, backup reviewer is Marcus Lee.",
  "noted, thx. while you're at it, who do we ping for billing-service?",
  "noted. who do we ping for billing-service?",
  "Don is the maintainer; channel I know of is #billing-eng.",
  "Don, channel #billing-eng.",
  "ok will tag #billing-eng then. one more — does payment-svc need a review for this change?",
  "and payment-svc — needs a review?",
  "No, payment-svc only proxies through. No review needed there.",
  "No, payment-svc only proxies. No review needed.",
  "We need rate limiting on /api/v1. What should we use?",
  "Token bucket or fixed window are the usual options.",
  "which one for us? we have spiky traffic patterns.",
  "Which one — we have spiky traffic.",
  "Token bucket handles bursts better. Suggest token bucket at 100 req/min per IP, refill every 600ms.",
  "Token bucket handles bursts. Suggest 100 req/min per IP, refill every 600ms.",
  "Decision: token-bucket at 100 req/min per IP via Redis; if Redis is unreachable for >5s, fail-open with a local in-memory fixed-window of 50 req/min. We accept brief over-limit during Redis outages over hard-failing user requests. Tracked under TICKET-4421.",
  "Decision: token bucket at 100 req/min per IP via Redis; if Redis is unreachable for >5s, fail-open with a local in-memory fixed-window of 50 req/min. Tracked under TICKET-4421.",
  "lgtm. you'll draft the RFC?",
  "Yes, will have an RFC by Wednesday.",
  "Yes, by Wednesday.",
  "sweet, while you're here — should we move to grpc later this year?",
  "Probably not before Q4 — too many internal clients on REST.",
  "Cache occasionally returns stale data right after a write. Where?",
  "Cache occasionally returns stale data right after a write. Where is it?",
  "How are reads structured? do they go through the same lock as writes?",
  "Do reads go through the same lock as writes?",
  "no, reads bypass the lock. only writes hold it.",
  "That's it — there's a read-after-write race. Found it: lib/cache.ts lines 142-148. cache.set updates the value but the version-counter increment happens 2 statements later. A concurrent cache.get between those statements sees the new value but the old version, and the consumer downstream treats that as \"still cached, use it\" because version didn't change.",
  "Found it: lib/cache.ts lines 142-148. cache.set updates the value, but the version-counter increment happens 2 statements later. A concurrent cache.get landing between those two statements sees the new value but the old version, and the downstream consumer treats that as 'still cached, use it' because version didn't change.",
  "ouch. so version increment needs to be atomic with set?",
  "so version increment needs to be atomic with set?",
  "Yes. Move the version++ inside the same critical section, or use a CAS.",
  "Yes. Move the version++ into the same critical section, or CAS on version.",
  "btw unrelated, the OPS dashboard times out a lot lately.",
  "Saw that too — looks like the Grafana proxy timeout was lowered to 5s last week, should bump back to 15s.",
  "will file a ticket. thx for the cache catch.",
]
const PADDING = [
  "hey morning. got time for a quick one today?",
  "sure thing — fire away when you're ready.",
  "first off — is the staging cluster still on the old node version btw?",
  "i think so, qa pushed back on the upgrade last week, said the smoke suite started flaking",
  "lol of course. also did finance approve the q3 tooling budget?",
  "i'll ask ramesh tomorrow when he's back from leave",
  "thx. one more — anyone owning the docs migration?",
  "rasmus i think, but he's on parental leave til the end of the month",
  "ah right. ok and re: lunch — was thinking we do that ramen place near the office?",
  "yes please. though i heard they raised prices, fyi.",
  "everyone has lol. anyway, the conf cfp deadline is friday — you submitting?",
  "probably. still picking between two talk angles, will decide tonight",
  "cool. before i forget — meeting with platform team got moved to thursday 2pm",
  "noted, on the calendar. ok so anyway —",
  "right, so the actual thing i wanted to ask about...",
  "ok cool, lgtm.",
  "great. i'll write it up in the rfc later this week.",
  "while you're here — any thoughts on the new sprint cadence?",
  "honestly mixed. two weeks felt too short for the platform-y work",
  "yeah, i've heard that from a few folks. we should bring it up in retro",
  "agreed. also — design team wants a 30-min sync on the dashboard rework, when's good?",
  "any morning next week. send a poll.",
  "will do. one more thing — the new junior is starting monday, are we doing a buddy thing?",
  "yes. i'll pair them with you for the first week if that's ok",
  "sure, happy to. quick aside — do we have a template doc for first-week onboarding?",
  "there's one in confluence somewhere. i'll dig it up and slack it to you",
  "thanks. last thing — the all-hands deck for tuesday, you have a slot or shall i fill?",
  "i've got platform updates. you can take the metrics slide if you want",
  "i'll prep it. anything else?",
  "nope, all good. catch you later.",
]
const NOISE = [
  "while we're here — should we use prettier or biome for formatting?",
  "biome is faster but prettier has wider plugin ecosystem. Either works.",
  "ok I'll punt on it. also CI build time crept up to 4 min, want to look later.",
  "Probably the snapshot suite growing. We can split it next sprint.",
  "side q: should we use prettier or biome for formatting?",
  "biome is faster, prettier has wider plugins. Either works.",
  "punt on it. also CI build time crept up to 4 min.",
  "Probably the snapshot suite. Split next sprint.",
  "did anyone actually try the new prod metrics dashboard?",
  "i poked it briefly. ui's nice, query latency is rough.",
  "ok will file feedback. unrelated — oncall handoff doc?",
  "not yet. draft after the sprint review.",
  "design system v3 dropped, anyone migrating yet?",
  "marketing site so far. infra apps lagging.",
  "fair. legal sign off on the new privacy policy?",
  "still in review. maybe end of next week.",
]

// ─── embeddings via LM Studio ────────────────────────────────────
async function embed(text) {
  const res = await fetch(`${LMS_BASE}/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: EMBED_MODEL, input: text }),
  })
  if (!res.ok) throw new Error(`LM Studio embed error ${res.status}: ${await res.text()}`)
  const j = await res.json()
  return j.data[0].embedding
}

async function embedAll(texts, cache) {
  const results = []
  for (let i = 0; i < texts.length; i++) {
    const t = texts[i]
    if (cache[t]) { results.push(cache[t]); continue }
    const v = await embed(t)
    cache[t] = v
    results.push(v)
    if (i % 10 === 0) console.log(`  embedded ${i}/${texts.length}`)
  }
  return results
}

// ─── softmax LR (same as v3.5, just bigger D) ────────────────────
function softmax(z) {
  const maxZ = Math.max(...z)
  const exps = z.map(v => Math.exp(v - maxZ))
  const sum = exps.reduce((s, e) => s + e, 0)
  return exps.map(e => e / sum)
}

function trainHead(X, Y, D, { epochs = 1500, lr = 1.0, l2 = 1e-3 } = {}) {
  const W = Array.from({ length: N_CLASSES }, () => Array(D).fill(0))
  const b = Array(N_CLASSES).fill(0)
  const N = X.length
  for (let epoch = 0; epoch < epochs; epoch++) {
    const gW = Array.from({ length: N_CLASSES }, () => Array(D).fill(0))
    const gb = Array(N_CLASSES).fill(0)
    let loss = 0
    for (let n = 0; n < N; n++) {
      const x = X[n], y = Y[n]
      const z = W.map((wk, k) => wk.reduce((s, w, j) => s + w * x[j], b[k]))
      const p = softmax(z)
      loss += -Math.log(p[y] + 1e-12)
      for (let k = 0; k < N_CLASSES; k++) {
        const err = p[k] - (k === y ? 1 : 0)
        gb[k] += err
        for (let j = 0; j < D; j++) gW[k][j] += err * x[j]
      }
    }
    for (let k = 0; k < N_CLASSES; k++) for (let j = 0; j < D; j++) gW[k][j] += l2 * W[k][j]
    for (let k = 0; k < N_CLASSES; k++) {
      b[k] -= (lr / N) * gb[k]
      for (let j = 0; j < D; j++) W[k][j] -= (lr / N) * gW[k][j]
    }
    if (DEBUG && (epoch < 5 || epoch % 200 === 0)) console.log(`  epoch ${epoch.toString().padStart(4)}  loss=${(loss / N).toFixed(4)}`)
  }
  return { W, b }
}

function predict(model, x) {
  const z = model.W.map((wk, k) => wk.reduce((s, w, j) => s + w * x[j], model.b[k]))
  const p = softmax(z)
  let argmax = 0
  for (let k = 1; k < p.length; k++) if (p[k] > p[argmax]) argmax = k
  return IDX_LABEL[argmax]
}

function evalSet(model, X, Y, Yraw) {
  let correct = 0
  const confusion = Array.from({ length: N_CLASSES }, () => Array(N_CLASSES).fill(0))
  for (let i = 0; i < X.length; i++) {
    const pred = predict(model, X[i])
    if (LABEL_IDX[pred] === Y[i]) correct++
    confusion[Y[i]][LABEL_IDX[pred]]++
  }
  return { acc: correct / X.length, confusion }
}

// ─── main ────────────────────────────────────────────────────────
async function main() {
  const texts = [...new Set([...CORES, ...PADDING, ...NOISE])]
  const labels = texts.map(regexTag)
  const labelCounts = labels.reduce((acc, l) => ({ ...acc, [l]: (acc[l] || 0) + 1 }), {})
  console.log(`[train-embed] dataset: ${texts.length} unique messages · regex labels: ${JSON.stringify(labelCounts)}`)

  let cache = {}
  if (existsSync(CACHE_PATH)) {
    cache = JSON.parse(readFileSync(CACHE_PATH, 'utf8'))
    console.log(`[train-embed] embedding cache: ${Object.keys(cache).length} entries`)
  }

  console.log(`[train-embed] embedding ${texts.length} texts via ${EMBED_MODEL}…`)
  const t0 = Date.now()
  const X = await embedAll(texts, cache)
  writeFileSync(CACHE_PATH, JSON.stringify(cache))
  const D = X[0].length
  console.log(`[train-embed] done in ${((Date.now() - t0) / 1000).toFixed(1)}s · D=${D}`)

  const Y = labels.map(l => LABEL_IDX[l])
  console.log(`[train-embed] training 768→3 softmax head…`)
  const model = trainHead(X, Y, D)

  const { acc, confusion } = evalSet(model, X, Y, labels)
  console.log(`[train-embed] training-set accuracy: ${(acc * 100).toFixed(1)}%`)
  console.log(`[train-embed] confusion (rows = true, cols = pred — keep/summ/melt):`)
  for (let i = 0; i < N_CLASSES; i++) {
    console.log(`  ${IDX_LABEL[i].padEnd(9)} : ${confusion[i].map(n => n.toString().padStart(4)).join(' ')}`)
  }

  const out = {
    version: 'v3.6',
    embed_model: EMBED_MODEL,
    D,
    classes: IDX_LABEL,
    W: model.W,
    b: model.b,
    training: { n: texts.length, acc, label_counts: labelCounts },
  }
  writeFileSync(WEIGHTS_PATH, JSON.stringify(out))
  const bytes = Buffer.byteLength(JSON.stringify(out))
  console.log(`\nwrote: ${WEIGHTS_PATH}  (${(bytes / 1024).toFixed(1)} KB serialized JSON)`)
}

main().catch(e => { console.error('FATAL:', e); process.exit(1) })

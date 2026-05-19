#!/usr/bin/env node
// Butterfly v3.5 — train a 10 KB hand-engineered softmax classifier on
// regex labels, then plug it into the butterfly pipeline as an alternative
// tagger. Tests whether learned weights on the SAME features the regex uses
// can match the regex's selectivity — or whether the hand-tuned thresholds
// are doing extra work that optimization picks up differently.
//
// Output: tools/butterfly-classifier-weights.json (~1 KB)
//
// Architecture: 14 hand-engineered features → 3-class softmax (keep / summarize / melt)
//   - Total parameters: 14×3 + 3 = 45 (weights + biases)
//   - Serialized weights: ~1 KB JSON
//   - Inference: ~1 µs per message
//
// Usage:
//   node tools/butterfly-train-classifier.mjs        # train + report training metrics
//   DEBUG=1 node tools/butterfly-train-classifier.mjs  # verbose per-epoch loss
//
// To use the trained classifier:
//   MODEL=trained-classifier STRATEGY=trained CONFIGS=len38-bud100-gens3 \
//     node tools/butterfly-llm-tagger.mjs

import { writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const WEIGHTS_PATH = join(ROOT, 'tools', 'butterfly-classifier-weights.json')
const DEBUG = process.env.DEBUG === '1'

// ─── features (14-dimensional) ────────────────────────────────────
// Each function returns a NUMBER (0/1 binary, or a small continuous value).
// These are the SAME signals the regex tagger weighs by hand. The trained
// classifier picks its own coefficients on top.
const FEATURES = [
  { name: 'file_path',     f: t => /\b\w+\/[\w\-./]+\.(ts|js|py|md|sql|yaml|toml|json|tsx|jsx|go|rs|html|css)\b/.test(t) ? 1 : 0 },
  { name: 'ticket',        f: t => /TICKET-\d+/i.test(t) ? 1 : 0 },
  { name: 'channel',       f: t => /#[\w-]+/.test(t) ? 1 : 0 },
  { name: 'pkg_mention',   f: t => /@[\w-]+\/[\w-]+/.test(t) ? 1 : 0 },
  { name: 'line_range',    f: t => /\blines?\s+\d+(\s*[-–]\s*\d+)?\b/i.test(t) ? 1 : 0 },
  { name: 'decision_kw',   f: t => /\b(Decision|Root cause|Confirmed|Found it):/i.test(t) ? 1 : 0 },
  { name: 'code_call',     f: t => /\b\w+\.\w+\(\)/.test(t) ? 1 : 0 },
  { name: 'proper_name',   f: t => /\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/.test(t) ? 1 : 0 },
  { name: 'quantity',      f: t => /\b\d+\s*(req\/min|ms|s|min|gb|mb|kb|tokens?|seconds?|minutes?)\b/i.test(t) ? 1 : 0 },
  { name: 'inline_code',   f: t => /`[^`\n]{2,}`/.test(t) ? 1 : 0 },
  { name: 'bare_ack',      f: t => /^\s*(ok|lgtm|sure|thx|noted|got it|will do|cool|sweet|alright|yep|yes|no|nope)[\s,.!]*$/i.test(t) ? 1 : 0 },
  { name: 'topic_shift',   f: t => /(while you're (here|at it)|btw unrelated|side q|while we're here|unrelated|aside|off-topic)/i.test(t) ? 1 : 0 },
  { name: 'short_question',f: t => (t.endsWith('?') && t.length < 50 && !/\w+\/[\w.]+/.test(t)) ? 1 : 0 },
  { name: 'log_length',    f: t => Math.min(1.0, Math.log(t.length + 1) / 6.5) }, // log-scaled to [0,1]
]
const F_DIM = FEATURES.length
const N_CLASSES = 3
const LABEL_IDX = { keep: 0, summarize: 1, melt: 2 }
const IDX_LABEL = ['keep', 'summarize', 'melt']

function encodeFeatures(text) {
  return FEATURES.map(({ f }) => f(text))
}

// ─── regex tagger (ground-truth labeler) ──────────────────────────
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

// ─── dataset construction ────────────────────────────────────────
// Pulls every message-text we've used across the experiment: 4 transcript
// cores, padding pool (pre + post), per-gen noise injections. Each text
// gets labeled by the regex. That's the training set for "can a learned
// classifier reproduce hand-tuned regex thresholds."
const CORES = [
  // jwt-clock-race
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

  // auth-owner-pto
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

  // rate-limit-decision
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

  // cache-race-fileline
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

function buildDataset() {
  const texts = [...new Set([...CORES, ...PADDING, ...NOISE])]
  return texts.map(t => ({
    text: t,
    label: regexTag(t),
    features: encodeFeatures(t),
  }))
}

// ─── softmax + cross-entropy gradient descent ────────────────────
// Standard textbook multiclass LR. Weights = K × D matrix (K classes × D features),
// biases = K-vector. Forward: z = W·x + b ; p = softmax(z). Loss = −log p[y].
// Gradient: dL/dW = (p − onehot(y)) ⊗ x ; dL/db = p − onehot(y).
function softmax(z) {
  const maxZ = Math.max(...z)
  const exps = z.map(v => Math.exp(v - maxZ))
  const sum = exps.reduce((s, e) => s + e, 0)
  return exps.map(e => e / sum)
}

function trainClassifier(dataset, { epochs = 800, lr = 0.5, l2 = 1e-4 } = {}) {
  // Initialize W (K×D) and b (K) with zeros — small problem, zero init OK.
  const W = Array.from({ length: N_CLASSES }, () => Array(F_DIM).fill(0))
  const b = Array(N_CLASSES).fill(0)

  const N = dataset.length
  for (let epoch = 0; epoch < epochs; epoch++) {
    // Accumulate gradient over the whole batch (small N; full GD is fine).
    const gW = Array.from({ length: N_CLASSES }, () => Array(F_DIM).fill(0))
    const gb = Array(N_CLASSES).fill(0)
    let loss = 0
    for (const { features: x, label } of dataset) {
      const y = LABEL_IDX[label]
      const z = W.map((wk, k) => wk.reduce((s, w, j) => s + w * x[j], b[k]))
      const p = softmax(z)
      loss += -Math.log(p[y] + 1e-12)
      for (let k = 0; k < N_CLASSES; k++) {
        const err = p[k] - (k === y ? 1 : 0)
        gb[k] += err
        for (let j = 0; j < F_DIM; j++) gW[k][j] += err * x[j]
      }
    }
    // L2 regularization
    for (let k = 0; k < N_CLASSES; k++) for (let j = 0; j < F_DIM; j++) gW[k][j] += l2 * W[k][j]
    // Apply
    for (let k = 0; k < N_CLASSES; k++) {
      b[k] -= (lr / N) * gb[k]
      for (let j = 0; j < F_DIM; j++) W[k][j] -= (lr / N) * gW[k][j]
    }
    if (DEBUG && (epoch < 5 || epoch % 100 === 0)) {
      console.log(`  epoch ${epoch.toString().padStart(3)}  loss=${(loss / N).toFixed(4)}`)
    }
  }
  return { W, b }
}

// ─── eval ─────────────────────────────────────────────────────────
function predict(model, x) {
  const z = model.W.map((wk, k) => wk.reduce((s, w, j) => s + w * x[j], model.b[k]))
  const p = softmax(z)
  let argmax = 0
  for (let k = 1; k < N_CLASSES; k++) if (p[k] > p[argmax]) argmax = k
  return { label: IDX_LABEL[argmax], probs: p }
}

function evalDataset(model, dataset) {
  let correct = 0
  const confusion = Array.from({ length: N_CLASSES }, () => Array(N_CLASSES).fill(0))
  for (const ex of dataset) {
    const pred = predict(model, ex.features).label
    if (pred === ex.label) correct++
    confusion[LABEL_IDX[ex.label]][LABEL_IDX[pred]]++
  }
  return { acc: correct / dataset.length, confusion }
}

// ─── main ─────────────────────────────────────────────────────────
function main() {
  const dataset = buildDataset()
  const labelCounts = dataset.reduce((acc, ex) => ({ ...acc, [ex.label]: (acc[ex.label] || 0) + 1 }), {})
  console.log(`[train] dataset: ${dataset.length} unique messages · regex labels: ${JSON.stringify(labelCounts)}`)

  const model = trainClassifier(dataset)
  const { acc, confusion } = evalDataset(model, dataset)
  console.log(`[train] training-set accuracy: ${(acc * 100).toFixed(1)}%`)
  console.log(`[train] confusion (rows = true, cols = pred — keep/summ/melt):`)
  for (let i = 0; i < N_CLASSES; i++) {
    console.log(`  ${IDX_LABEL[i].padEnd(9)} : ${confusion[i].map(n => n.toString().padStart(4)).join(' ')}`)
  }

  // Print learned weights per feature (most-important first)
  console.log(`\n[train] learned weights (W[k, feature]):`)
  console.log(`  ${'feature'.padEnd(16)} ${'keep'.padStart(7)} ${'summ'.padStart(7)} ${'melt'.padStart(7)}`)
  for (let j = 0; j < F_DIM; j++) {
    const row = model.W.map(wk => wk[j].toFixed(2).padStart(7)).join(' ')
    console.log(`  ${FEATURES[j].name.padEnd(16)} ${row}`)
  }
  console.log(`  ${'(bias)'.padEnd(16)} ${model.b.map(b => b.toFixed(2).padStart(7)).join(' ')}`)

  const out = {
    version: 'v3.5',
    feature_names: FEATURES.map(f => f.name),
    classes: IDX_LABEL,
    W: model.W,
    b: model.b,
    training: { n: dataset.length, acc, label_counts: labelCounts },
  }
  writeFileSync(WEIGHTS_PATH, JSON.stringify(out, null, 2))
  const bytes = Buffer.byteLength(JSON.stringify(out))
  console.log(`\nwrote: ${WEIGHTS_PATH}  (${bytes} bytes serialized JSON)`)
}

main()

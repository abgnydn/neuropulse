// butterfly-mode — runs a butterfly transgenerational compaction demo
// in-browser using the same Phi-3-mini engine as the rest of neuropulse.
//
// Mirrors the ablation panel's UX: top-right floating panel, status line,
// run button, side-by-side outputs. Different accent (lavender) and a
// scripted multi-stage flow.
//
// Pipeline per run:
//   1. Walk a built-in 5-msg debugging transcript.
//   2. For each message → tagger call (Phi-3 emits keep/summarize/melt).
//   3. Chrysalis call → Phi-3 rebuilds the tagged transcript into ~400 tok.
//   4. Inject 4 hardcoded noise messages (off-topic, tag-as-melt).
//   5. Repeat (steps 2-4) for N_GENERATIONS.
//   6. Ask the needle question against (a) butterfly's final rebuild and
//      (b) a recency-truncated lastN baseline at the same token budget.
//   7. Show side-by-side, highlight which arm got the needle right.
//
// Pure single-engine sequential — neuropulse has no slotted-engine pattern
// (yet), so tagger / chrysalis / answer all share one Phi-3 instance and
// run one after the other. ~25-30 s per full run on M2-class hardware.

import type { InferenceEngine } from "./engine/inference"
import { normalizeFull } from "./engine/activation-reducer"

// ─── 3D viz integration (v2) ─────────────────────────────────────
// We hold an opaque viz handle. Only `updateResidualLayer(layer, vec)` is
// called — that recolors one of the 32 residual-stream layer slabs the
// neuropulse scene shows. By writing tag-modulated activations during the
// butterfly run, the viewer literally watches keep-tagged content stay
// bright across metamorphoses while melt-tagged content fades.
interface ResidualViz {
  updateResidualLayer(layer: number, vec: Float32Array): void
}

const RESIDUAL_DIM = 3072
const RESIDUAL_LAYERS = 32

// Modulation factors per tag: keep=bright, summarize=medium, melt=dim.
// Multiplied with the *normalized* activation magnitude (0..1).
const TAG_BRIGHTNESS = { keep: 1.0, summarize: 0.55, melt: 0.12 } as const

// Decay applied to the accumulated slab between messages within one
// generation, so older messages don't dominate forever. 0.7 = each new
// message starts the previous-state at 70%, then accumulates fresh.
const INTRA_GEN_DECAY = 0.7

// Decay between generations — emphasizes that surviving brightness is the
// "memory traces that made it through metamorphosis."
const INTER_GEN_DECAY = 0.4

// ─── Built-in demo content ───────────────────────────────────────

// Synthetic debugging session. Needle (root cause) at message index 4.
// Kept short so the whole demo runs in <30 s on a mid-range GPU.
const TRANSCRIPT: { role: "user" | "assistant"; content: string }[] = [
  { role: "user", content: "Our auth/session.test.ts is flaky in CI but passes locally. Help?" },
  { role: "assistant", content: "Sure — share the test and the failure output." },
  { role: "user", content: "It asserts decoded.exp equals Math.floor(Date.now()/1000) + 3600. CI says expected 1735689600 received 1735689599." },
  { role: "assistant", content: "Off-by-one second. Smells like a timing race rather than a logic bug." },
  { role: "assistant", content: "Confirmed in lib/jwt.ts: issueToken reads Date.now() once for `exp`, the test reads Date.now() again in the assertion. On slow CI those two reads can land on different seconds. Root cause: clock race, not a code bug." },
  { role: "user", content: "ok so the fix?" },
  { role: "assistant", content: "Two options. Cheap: capture Date.now() once before calling issueToken. Proper: have issueToken accept an optional `now` parameter so tests inject a fixed clock." },
  { role: "user", content: "let's do the proper one. lgtm, pushing." },
]

// Question whose correct answer requires the planted needle (msg 4).
// Phrased to require the SPECIFIC code-level mistake — pretraining alone
// shouldn't be enough to answer correctly without the actual context.
const NEEDLE_QUESTION = "Looking at lib/jwt.ts specifically — what is the exact code mistake that causes the off-by-one second in CI? Name the function and what it does wrong."
const NEEDLE_FACT = "issueToken in lib/jwt.ts reads Date.now() once to compute the `exp` field. The test assertion ALSO reads Date.now() (a second time). On slow CI those two reads can land on different seconds, causing the off-by-one. The mistake is the second Date.now() read in the test — there should be one captured `now` shared between the call and the assertion."

// Off-topic noise injected between metamorphoses. All clearly melt-able.
const NOISE_BATCH: { role: "user" | "assistant"; content: string }[] = [
  { role: "user", content: "while we're here — should we use prettier or biome for formatting?" },
  { role: "assistant", content: "biome is faster but prettier has wider plugin ecosystem. Either works." },
  { role: "user", content: "ok I'll punt on it. also CI build time crept up to 4 min, want to look later." },
  { role: "assistant", content: "Probably the snapshot suite growing. We can split it next sprint." },
]

// ─── Config ──────────────────────────────────────────────────────

const N_GENERATIONS = 3      // metamorphoses per run
const TARGET_TOKENS = 400    // chrysalis budget per gen
const TAG_MAX_TOKENS = 28    // label + brief reason ("keep | off-by-one diagnosis")
const CHRYSALIS_MAX = 500    // generous, model usually undershoots
const ANSWER_MAX = 200

// ─── Helpers ─────────────────────────────────────────────────────

const tokens = (s: string) => Math.ceil(s.length / 4)

function asText(msgs: typeof TRANSCRIPT): string {
  return msgs.map(m => `[${m.role}]\n${m.content}`).join("\n\n")
}

function pickLastN(msgs: typeof TRANSCRIPT, budget: number): typeof TRANSCRIPT {
  const out: typeof TRANSCRIPT = []
  let acc = 0
  for (let i = msgs.length - 1; i >= 0; i--) {
    const t = tokens(msgs[i].content)
    if (acc + t > budget && out.length > 0) break
    out.unshift(msgs[i])
    acc += t
  }
  return out
}

// Parse the tagger's output. New format requested from Phi-3:
//   "label | brief 4-7-word reason"
// We accept any prefix/suffix garbage and pick out (a) the first matching
// label keyword and (b) anything after a pipe (or after the label itself
// if no pipe) as the reason. Fallback "keep" with no reason if it's
// completely garbled — never lose information silently.
interface TagOutput { label: "keep" | "summarize" | "melt"; reason: string }
function parseTag(raw: string): TagOutput {
  const s = raw.toLowerCase()
  let label: TagOutput["label"] = "keep"
  if (/\bkeep\b/.test(s))                          label = "keep"
  else if (/\b(summarize|summary|summ)\b/.test(s)) label = "summarize"
  else if (/\b(melt|drop)\b/.test(s))              label = "melt"

  // Reason: prefer text after the first '|', else trim the label out.
  let reason = ""
  const pipeIdx = raw.indexOf("|")
  if (pipeIdx !== -1) {
    reason = raw.slice(pipeIdx + 1).trim()
  } else {
    reason = raw.replace(/keep|summarize|summary|melt|drop/gi, "").trim()
  }
  // Strip any markdown/quotes/end-of-turn artifacts; cap at ~50 chars
  reason = reason.replace(/[<>|`*"\n\r]+/g, " ").replace(/\s+/g, " ").trim().slice(0, 50)
  return { label, reason }
}

// LLM-as-judge: ask Phi-3 to grade the answer against the expected fact.
// Replaces the regex heuristic — it was too strict and ignored partial
// answers that an honest reader would credit. One extra engine call (~3s)
// per arm; cheap enough to swap in.
async function judgeAnswer(
  engine: InferenceEngine,
  question: string,
  expectedFact: string,
  answer: string,
): Promise<"hit" | "partial" | "miss"> {
  const prompt = `<|system|>\nYou grade whether an answer accurately conveys an expected fact for context-compaction evaluation.\n\nReturn STRICTLY a single digit at the END of your response:\n  2 = HIT     — answer accurately conveys the full expected fact (paraphrasing fine)\n  1 = PARTIAL — answer preserves the load-bearing identification (file, function, mechanism — what an engineer needs to act) but mis-states or omits a detail\n  0 = MISS    — fact missing, vague, or invented\n\nThe LAST CHARACTER of your reply MUST be 0, 1, or 2. Nothing after.<|end|>\n<|user|>\nQUESTION: ${question}\nEXPECTED FACT: ${expectedFact}\nANSWER:\n${answer}\n\nGrade. End with 0, 1, or 2.<|end|>\n<|assistant|>\n`
  const raw = await engine.generate(prompt, 80, {})
  const cleaned = raw.trim()
  // Find LAST occurrence of 0/1/2 in the response
  const m = cleaned.match(/[012](?!.*[012])/s)
  if (!m) return "miss"
  const score = Number(m[0])
  return score === 2 ? "hit" : score === 1 ? "partial" : "miss"
}

// ─── Public API ──────────────────────────────────────────────────

export interface ButterflyPanelOpts {
  getEngine: () => InferenceEngine | null
  isBusy: () => boolean
  setBusy: (busy: boolean) => void
  /** Optional residual-stream visualizer. When provided, butterfly-mode
   *  paints per-tag importance into the neuropulse 3D scene during a run. */
  viz?: ResidualViz
}

export function initButterflyPanel(opts: ButterflyPanelOpts): void {
  const inputWrap = document.querySelector(".input-wrap")
  if (!inputWrap) return

  const style = document.createElement("style")
  style.textContent = `
    .bfly-panel {
      position: fixed; top: 64px; right: 20px;
      width: 380px; max-height: calc(100vh - 100px); overflow-y: auto;
      background: rgba(12, 14, 20, 0.92); backdrop-filter: blur(12px);
      border: 1px solid rgba(183, 148, 246, 0.45);
      border-radius: 10px; padding: 12px 14px; z-index: 21;
      color: #f4ecdf; font-family: inherit; font-size: 12px;
      box-shadow: 0 0 24px rgba(183, 148, 246, 0.18);
      display: none;
    }
    .bfly-panel.open { display: block; }
    body.panels-hidden .bfly-panel { display: none !important; }
    @media (max-width: 900px) {
      .bfly-panel { right: 10px; left: 10px; width: auto; }
    }
    .bfly-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
    .bfly-title { color: #b794f6; font-weight: 600; flex: 1 1 100%; min-width: 0; font-size: 13px; letter-spacing: 0.04em; }
    .bfly-status { color: #c8b8e8; font-size: 11px; flex: 1 1 100%; min-width: 0; }
    .bfly-close {
      background: transparent; border: 1px solid rgba(244,236,223,0.18);
      color: #8a7f6c; width: 22px; height: 22px; border-radius: 50%;
      cursor: pointer; font-size: 11px; line-height: 1;
      display: inline-flex; align-items: center; justify-content: center;
      flex: 0 0 auto;
    }
    .bfly-close:hover { color: #f4ecdf; border-color: #f4ecdf; }
    .bfly-explain { color: #8a7f6c; font-size: 11px; line-height: 1.45; margin-bottom: 8px; }
    .bfly-btn {
      background: rgba(183, 148, 246, 0.18); color: #d8c4ff;
      border: 1px solid rgba(183, 148, 246, 0.5); border-radius: 5px;
      padding: 6px 14px; cursor: pointer; font-size: 12px; font-family: inherit;
      transition: all 0.15s;
    }
    .bfly-btn:hover { background: rgba(183, 148, 246, 0.3); color: #fff; }
    .bfly-btn[disabled] { opacity: 0.4; cursor: wait; }

    .bfly-progress {
      height: 4px; background: rgba(80, 60, 120, 0.25); border-radius: 2px;
      overflow: hidden; margin-bottom: 8px;
    }
    .bfly-progress-bar {
      height: 100%; background: linear-gradient(90deg, #b794f6, #ff8c42);
      width: 0%; transition: width 0.3s ease;
    }
    .bfly-tags-row {
      display: flex; gap: 4px; flex-wrap: wrap;
      margin: 6px 0 10px;
    }
    .bfly-tag {
      font-size: 10px; padding: 2px 7px; border-radius: 10px;
      border: 1px solid;
    }
    .bfly-tag.keep      { color: #ffd93d; border-color: rgba(255, 217, 61, 0.5); background: rgba(255, 217, 61, 0.1); }
    .bfly-tag.summarize { color: #5fd8d4; border-color: rgba(95, 216, 212, 0.5); background: rgba(95, 216, 212, 0.1); }
    .bfly-tag.melt      { color: #8a7f6c; border-color: rgba(138, 127, 108, 0.4); background: rgba(0,0,0,0.2); }

    .bfly-stage {
      background: rgba(0,0,0,0.35); border-radius: 6px;
      padding: 8px 10px; margin-bottom: 8px;
    }
    .bfly-stage-label {
      color: #b794f6; font-size: 10px; text-transform: uppercase;
      letter-spacing: 0.08em; margin-bottom: 4px;
      display: flex; justify-content: space-between;
    }
    .bfly-stage-text {
      color: #f4ecdf; font-size: 11px; line-height: 1.45;
      white-space: pre-wrap; word-break: break-word;
      max-height: 120px; overflow-y: auto;
    }
    .bfly-stage-text.empty { color: #514a3e; font-style: italic; }

    .bfly-throwaway {
      font-size: 10px; color: #8a7f6c; line-height: 1.5;
      margin-top: 4px;
    }
    .bfly-throwaway-row {
      padding: 2px 6px; background: rgba(0,0,0,0.2); border-radius: 3px;
      margin: 2px 0; display: inline-block; margin-right: 4px;
    }
    .bfly-throwaway-row .gen { color: #b794f6; font-weight: 600; }
    .bfly-throwaway-row .dropped { color: #ff6b6b; }

    /* Tier B: custom mode (editable transcript/question/needle) and
       step-by-step pause between metamorphoses. */
    .bfly-custom-toggle {
      display: flex; align-items: center; gap: 8px;
      font-size: 11px; color: #8a7f6c;
      margin-bottom: 6px; flex-wrap: wrap;
    }
    .bfly-custom-toggle label { cursor: pointer; user-select: none; }
    .bfly-custom-toggle input { cursor: pointer; }
    .bfly-custom-edit {
      background: rgba(0,0,0,0.35); border-radius: 6px;
      padding: 8px 10px; margin-bottom: 8px;
    }
    .bfly-custom-edit label {
      display: block; font-size: 10px; color: #b794f6;
      text-transform: uppercase; letter-spacing: 0.08em;
      margin: 6px 0 3px;
    }
    .bfly-custom-edit textarea, .bfly-custom-edit input[type="text"] {
      width: 100%; background: rgba(0,0,0,0.4); color: #f4ecdf;
      border: 1px solid #3a3429; border-radius: 4px;
      padding: 6px 8px; font-family: inherit; font-size: 11px;
      line-height: 1.4; resize: vertical; box-sizing: border-box;
    }
    .bfly-custom-edit textarea { min-height: 60px; max-height: 180px; }
    .bfly-custom-edit .bfly-custom-hint { color: #514a3e; font-size: 10px; font-style: italic; }
    .bfly-pause-banner {
      display: none; background: rgba(255, 217, 61, 0.12);
      border: 1px solid rgba(255, 217, 61, 0.45); color: #ffd93d;
      padding: 6px 10px; border-radius: 6px; margin-bottom: 8px;
      font-size: 11px; align-items: center; justify-content: space-between; gap: 8px;
    }
    .bfly-pause-banner.visible { display: flex; }
    .bfly-pause-banner button { background: rgba(255, 217, 61, 0.25); color: #ffd93d; border: 1px solid rgba(255, 217, 61, 0.5); border-radius: 4px; padding: 3px 10px; cursor: pointer; font-family: inherit; font-size: 11px; }
    .bfly-pause-banner button:hover { background: rgba(255, 217, 61, 0.4); color: #fff; }

    .bfly-stats {
      font-size: 10px; color: #8a7f6c; margin-top: 6px;
      padding: 6px 8px; background: rgba(0,0,0,0.2); border-radius: 4px;
      display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px;
    }
    .bfly-stats .stat-bfly  { color: #b794f6; }
    .bfly-stats .stat-lastn { color: #8a7f6c; }
    .bfly-stats .stat-clear { color: #514a3e; cursor: pointer; text-decoration: underline; }
    .bfly-stats .stat-clear:hover { color: #ff6b6b; }

    /* Collapsible "what each arm actually saw" — opt-in, default closed
       so the panel doesn't balloon. The interesting part for learners is
       contrasting the two contexts side by side. */
    .bfly-context-detail {
      background: rgba(0,0,0,0.35); border-radius: 6px;
      padding: 6px 10px; margin-bottom: 6px; cursor: pointer;
    }
    .bfly-context-detail summary {
      list-style: none; color: #b794f6; font-size: 10px;
      text-transform: uppercase; letter-spacing: 0.08em;
      display: flex; justify-content: space-between;
    }
    .bfly-context-detail summary::-webkit-details-marker { display: none; }
    .bfly-context-detail.lastn summary { color: #8a7f6c; }
    .bfly-context-detail summary::after { content: '▸'; transition: transform 0.15s; }
    .bfly-context-detail[open] summary::after { transform: rotate(90deg); }
    .bfly-context-detail-text {
      color: #f4ecdf; font-size: 11px; line-height: 1.4;
      white-space: pre-wrap; word-break: break-word;
      max-height: 140px; overflow-y: auto;
      margin-top: 6px; padding-top: 6px;
      border-top: 1px solid rgba(244,236,223,0.08);
    }

    .bfly-result {
      display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
    }
    .bfly-arm { padding: 8px 10px; border-radius: 6px; background: rgba(0,0,0,0.4); }
    .bfly-arm-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
    .bfly-arm.bfly  { border: 1px solid rgba(183, 148, 246, 0.4); }
    .bfly-arm.bfly  .bfly-arm-label { color: #b794f6; }
    .bfly-arm.lastn { border: 1px solid rgba(138, 127, 108, 0.35); }
    .bfly-arm.lastn .bfly-arm-label { color: #8a7f6c; }
    .bfly-arm-text { color: #f4ecdf; font-size: 11px; line-height: 1.4; word-break: break-word; max-height: 100px; overflow-y: auto; }
    .bfly-arm.win { box-shadow: 0 0 12px rgba(183, 148, 246, 0.4); }
    .bfly-verdict { font-size: 11px; margin-top: 4px; font-weight: 600; }
    .bfly-verdict.hit     { color: #5fd8d4; }
    .bfly-verdict.partial { color: #ffd93d; }
    .bfly-verdict.miss    { color: #ff6b6b; }
  `
  document.head.appendChild(style)

  const panel = document.createElement("div")
  panel.className = "bfly-panel open"
  panel.innerHTML = `
    <div class="bfly-header">
      <span class="bfly-title">🦋 BUTTERFLY MODE</span>
      <button class="bfly-close" id="bflyCloseBtn" type="button" aria-label="Hide">✕</button>
      <span class="bfly-status" id="bflyStatus">Ready. Press Run to compress a real conversation across ${N_GENERATIONS} metamorphoses on this GPU.</span>
    </div>
    <div class="bfly-explain">
      Phi-3 tags each message (keep / summarize / melt), rebuilds a smaller context, repeats. Watch what survives. Compare to naive recency truncation at the same budget.
    </div>
    <div class="bfly-progress"><div class="bfly-progress-bar" id="bflyBar"></div></div>

    <div class="bfly-pause-banner" id="bflyPauseBanner">
      <span id="bflyPauseLabel">Paused after stage. Inspect, then continue.</span>
      <button id="bflyContinueBtn" type="button">Continue →</button>
    </div>

    <div class="bfly-custom-toggle">
      <label><input type="checkbox" id="bflyCustomToggle"> Edit transcript / question</label>
      <label><input type="checkbox" id="bflyStepToggle"> Step-by-step</label>
    </div>
    <div class="bfly-custom-edit" id="bflyCustomEdit" style="display:none;">
      <label>Transcript (one message per line, prefixed with <code>user:</code> or <code>assistant:</code>)</label>
      <textarea id="bflyCustomTranscript" spellcheck="false"></textarea>
      <label>Question to ask after the metamorphoses</label>
      <input type="text" id="bflyCustomQuestion" spellcheck="false">
      <label>Expected fact (used by the judge to grade answers)</label>
      <textarea id="bflyCustomNeedle" spellcheck="false" style="min-height:40px;"></textarea>
      <div class="bfly-custom-hint">Empty = use the built-in JWT-flake demo. Click "Reset to default" to restore.</div>
    </div>

    <div style="display: flex; gap: 8px; margin-bottom: 10px;">
      <button class="bfly-btn" id="bflyRunBtn" type="button">Run butterfly demo</button>
      <button class="bfly-btn" id="bflyResetBtn" type="button">Reset</button>
    </div>

    <div class="bfly-stage" id="bflyTagsStage">
      <div class="bfly-stage-label">
        <span>Tags (gen <span id="bflyGenNum">—</span>)</span>
        <span id="bflyTagCounts"></span>
      </div>
      <div class="bfly-tags-row" id="bflyTagsRow"></div>
    </div>

    <div class="bfly-stage" id="bflyChrysalisStage">
      <div class="bfly-stage-label">
        <span>Chrysalis output (latest gen)</span>
        <span id="bflyChrysalisTokens"></span>
      </div>
      <div class="bfly-stage-text empty" id="bflyChrysalisText">—</div>
    </div>

    <div class="bfly-stage" id="bflyThrowawayStage" style="display:none;">
      <div class="bfly-stage-label"><span>What lastN truncation threw away each round</span></div>
      <div class="bfly-throwaway" id="bflyThrowawayLog"></div>
    </div>

    <details class="bfly-context-detail bfly" id="bflyCtxBflyDetails" style="display:none;">
      <summary><span>What butterfly's answer arm saw</span><span id="bflyCtxBflyTokens"></span></summary>
      <div class="bfly-context-detail-text" id="bflyCtxBflyText"></div>
    </details>
    <details class="bfly-context-detail lastn" id="bflyCtxLastDetails" style="display:none;">
      <summary><span>What lastN's answer arm saw</span><span id="bflyCtxLastTokens"></span></summary>
      <div class="bfly-context-detail-text" id="bflyCtxLastText"></div>
    </details>

    <div class="bfly-result">
      <div class="bfly-arm bfly" id="bflyArmBfly">
        <div class="bfly-arm-label">Butterfly</div>
        <div class="bfly-arm-text" id="bflyAnsBfly">—</div>
        <div class="bfly-verdict" id="bflyVerdictBfly"></div>
      </div>
      <div class="bfly-arm lastn" id="bflyArmLastn">
        <div class="bfly-arm-label">LastN truncation</div>
        <div class="bfly-arm-text" id="bflyAnsLastn">—</div>
        <div class="bfly-verdict" id="bflyVerdictLastn"></div>
      </div>
    </div>

    <div class="bfly-stats" id="bflyStats" style="display:none;">
      <span><span class="stat-bfly">butterfly: 0/0</span> · <span class="stat-lastn">lastN: 0/0</span> · across <span id="bflyStatsRuns">0</span> runs</span>
      <span class="stat-clear" id="bflyStatsClear">clear</span>
    </div>
  `
  inputWrap.parentNode?.insertBefore(panel, inputWrap)

  const $ = <T extends HTMLElement>(id: string) => panel.querySelector<T>(`#${id}`)!
  const statusEl    = $<HTMLSpanElement>("bflyStatus")
  const runBtn      = $<HTMLButtonElement>("bflyRunBtn")
  const resetBtn    = $<HTMLButtonElement>("bflyResetBtn")
  const closeBtn    = $<HTMLButtonElement>("bflyCloseBtn")
  const bar         = $<HTMLDivElement>("bflyBar")
  const tagsRow     = $<HTMLDivElement>("bflyTagsRow")
  const tagCountsEl = $<HTMLSpanElement>("bflyTagCounts")
  const genNumEl    = $<HTMLSpanElement>("bflyGenNum")
  const chrysText   = $<HTMLDivElement>("bflyChrysalisText")
  const chrysToks   = $<HTMLSpanElement>("bflyChrysalisTokens")
  const ansBfly     = $<HTMLDivElement>("bflyAnsBfly")
  const ansLastn    = $<HTMLDivElement>("bflyAnsLastn")
  const verdictBfly = $<HTMLDivElement>("bflyVerdictBfly")
  const verdictLast = $<HTMLDivElement>("bflyVerdictLastn")
  const armBfly     = $<HTMLDivElement>("bflyArmBfly")
  const armLastn    = $<HTMLDivElement>("bflyArmLastn")
  // v2.2 — Tier A: contexts + throwaway log
  const throwawayStage = $<HTMLDivElement>("bflyThrowawayStage")
  const throwawayLog   = $<HTMLDivElement>("bflyThrowawayLog")
  const ctxBflyDetails = $<HTMLDetailsElement>("bflyCtxBflyDetails")
  const ctxBflyTokens  = $<HTMLSpanElement>("bflyCtxBflyTokens")
  const ctxBflyText    = $<HTMLDivElement>("bflyCtxBflyText")
  const ctxLastDetails = $<HTMLDetailsElement>("bflyCtxLastDetails")
  const ctxLastTokens  = $<HTMLSpanElement>("bflyCtxLastTokens")
  const ctxLastText    = $<HTMLDivElement>("bflyCtxLastText")
  // v2.3 — Tier B: edit / pause / stats
  const customToggle    = $<HTMLInputElement>("bflyCustomToggle")
  const stepToggle      = $<HTMLInputElement>("bflyStepToggle")
  const customEdit      = $<HTMLDivElement>("bflyCustomEdit")
  const customTranscript = $<HTMLTextAreaElement>("bflyCustomTranscript")
  const customQuestion  = $<HTMLInputElement>("bflyCustomQuestion")
  const customNeedle    = $<HTMLTextAreaElement>("bflyCustomNeedle")
  const pauseBanner     = $<HTMLDivElement>("bflyPauseBanner")
  const pauseLabel      = $<HTMLSpanElement>("bflyPauseLabel")
  const continueBtn     = $<HTMLButtonElement>("bflyContinueBtn")
  const statsEl         = $<HTMLDivElement>("bflyStats")
  const statsRunsEl     = $<HTMLSpanElement>("bflyStatsRuns")
  const statsClearEl    = $<HTMLSpanElement>("bflyStatsClear")

  // ─── Tier B helpers ───────────────────────────────────────────────
  // (a) Pre-fill custom textareas with the built-in defaults so users
  //     can see the format and tweak from there.
  const defaultTranscriptText = TRANSCRIPT.map(m => `${m.role}: ${m.content}`).join("\n")
  customTranscript.value = defaultTranscriptText
  customQuestion.value   = NEEDLE_QUESTION
  customNeedle.value     = NEEDLE_FACT

  customToggle.addEventListener("change", () => {
    customEdit.style.display = customToggle.checked ? "" : "none"
  })

  // (b) Parser — "user: ..." or "assistant: ..." lines into typed messages.
  //     Multi-line content is allowed; subsequent lines append until the
  //     next role line or end. Falls back to the built-in transcript if
  //     the textarea is empty or unparseable.
  function parseTranscriptText(text: string): typeof TRANSCRIPT {
    const out: typeof TRANSCRIPT = []
    const lines = text.split(/\r?\n/)
    let currentRole: "user" | "assistant" | null = null
    let buf: string[] = []
    const flush = () => {
      if (currentRole && buf.length > 0) {
        out.push({ role: currentRole, content: buf.join("\n").trim() })
      }
      buf = []
    }
    for (const line of lines) {
      const m = /^\s*(user|assistant)\s*:\s*(.*)$/i.exec(line)
      if (m) {
        flush()
        currentRole = m[1].toLowerCase() as "user" | "assistant"
        if (m[2]) buf.push(m[2])
      } else if (currentRole) {
        buf.push(line)
      }
    }
    flush()
    return out.length > 0 ? out : TRANSCRIPT.slice()
  }

  // (c) Pause-and-continue mechanism for step-by-step mode. `awaitContinue`
  //     resolves when the user clicks the Continue button. We re-create
  //     the resolver each call so multiple pauses chain cleanly.
  let pauseResolver: (() => void) | null = null
  continueBtn.addEventListener("click", () => {
    pauseBanner.classList.remove("visible")
    if (pauseResolver) { pauseResolver(); pauseResolver = null }
  })
  async function awaitContinue(label: string): Promise<void> {
    if (!stepToggle.checked) return                  // off → no-op
    pauseLabel.textContent = label
    pauseBanner.classList.add("visible")
    return new Promise<void>(resolve => { pauseResolver = resolve })
  }

  // (d) localStorage stats — running tally across all runs in this browser.
  interface StatsEntry { bfly: 0|1|2; lastN: 0|1|2; ts: number }
  const STATS_KEY = "butterfly-mode-stats-v1"
  const loadStats = (): StatsEntry[] => {
    try { const raw = localStorage.getItem(STATS_KEY); return raw ? JSON.parse(raw) : [] }
    catch { return [] }
  }
  const saveStats = (entries: StatsEntry[]) => {
    try { localStorage.setItem(STATS_KEY, JSON.stringify(entries)) } catch { /* ignore */ }
  }
  const renderStats = () => {
    const entries = loadStats()
    if (entries.length === 0) { statsEl.style.display = "none"; return }
    const bflyHits  = entries.filter(e => e.bfly  > 0).length
    const lastnHits = entries.filter(e => e.lastN > 0).length
    statsEl.style.display = ""
    statsRunsEl.textContent = String(entries.length)
    statsEl.querySelector(".stat-bfly")!.textContent  = `butterfly: ${bflyHits}/${entries.length}`
    statsEl.querySelector(".stat-lastn")!.textContent = `lastN: ${lastnHits}/${entries.length}`
  }
  statsClearEl.addEventListener("click", () => {
    if (!confirm("Clear all run stats from this browser?")) return
    saveStats([])
    renderStats()
  })
  renderStats()

  function setStatus(msg: string) { statusEl.textContent = msg }
  function setProgress(pct: number) { bar.style.width = `${Math.max(0, Math.min(100, pct))}%` }
  function resetUI() {
    setStatus(`Ready. Press Run to compress a real conversation across ${N_GENERATIONS} metamorphoses on this GPU.`)
    setProgress(0)
    tagsRow.innerHTML = ""
    tagCountsEl.textContent = ""
    genNumEl.textContent = "—"
    chrysText.textContent = "—"; chrysText.classList.add("empty")
    chrysToks.textContent = ""
    ansBfly.textContent = "—"; ansLastn.textContent = "—"
    verdictBfly.textContent = ""; verdictLast.textContent = ""
    armBfly.classList.remove("win"); armLastn.classList.remove("win")
    // Tier A — clear throwaway log + hide context details until next run.
    throwawayLog.innerHTML = ""
    throwawayStage.style.display = "none"
    ctxBflyDetails.style.display = "none"; ctxBflyText.textContent = ""; ctxBflyTokens.textContent = ""
    ctxLastDetails.style.display = "none"; ctxLastText.textContent = ""; ctxLastTokens.textContent = ""
  }

  closeBtn.addEventListener("click", () => panel.classList.remove("open"))
  resetBtn.addEventListener("click", resetUI)

  // Expose toggle for the global keymap (B key, optional).
  ;(window as unknown as { __toggleButterflyPanel: () => void }).__toggleButterflyPanel = () => {
    panel.classList.toggle("open")
  }

  runBtn.addEventListener("click", async () => {
    const engine = opts.getEngine()
    if (!engine) { setStatus("Engine not ready yet."); return }
    if (opts.isBusy()) { setStatus("Inference already in flight — wait for it."); return }
    opts.setBusy(true)
    runBtn.disabled = true; resetBtn.disabled = true
    runBtn.textContent = "Running…"
    resetUI()

    const t0 = performance.now()
    // Tier B: pull editable inputs (or fall back to defaults if untouched).
    const useCustom = customToggle.checked
    const transcriptForRun = useCustom ? parseTranscriptText(customTranscript.value) : TRANSCRIPT.slice()
    const questionForRun   = (useCustom && customQuestion.value.trim()) ? customQuestion.value.trim() : NEEDLE_QUESTION
    const needleForRun     = (useCustom && customNeedle.value.trim())   ? customNeedle.value.trim()   : NEEDLE_FACT
    let messages = transcriptForRun.slice()
    let lastnMessages = transcriptForRun.slice()
    let lastChrysalis = ""

    // ─── v2: residual-stream visualization ──────────────────────
    // Per-layer accumulator. We modulate each tagged message's post-norm
    // residual by its tag importance and write into the neuropulse 3D scene.
    // Across the N generations, melt-tagged content fades; keep-tagged
    // content stays bright — the metamorphosis-survival visual.
    const slabAcc: Map<number, Float32Array> = new Map()
    const decaySlab = (factor: number) => {
      for (const [, vec] of slabAcc) for (let i = 0; i < vec.length; i++) vec[i] *= factor
    }
    const writeSlabToViz = () => {
      if (!opts.viz) return
      for (const [layer, vec] of slabAcc) opts.viz.updateResidualLayer(layer, vec)
    }

    try {
      for (let gen = 1; gen <= N_GENERATIONS; gen++) {
        genNumEl.textContent = String(gen)
        setStatus(`Gen ${gen}/${N_GENERATIONS}: tagging ${messages.length} messages with Phi-3…`)
        tagsRow.innerHTML = ""

        // Inter-generation fade: emphasize that surviving brightness is
        // memory traces that made it through metamorphosis.
        if (gen > 1) { decaySlab(INTER_GEN_DECAY); writeSlabToViz() }

        const tags: TagOutput[] = []
        for (let i = 0; i < messages.length; i++) {
          const m = messages[i]
          const prompt = `<|system|>\nClassify ONE conversation message for context compaction. Reply in this EXACT format:\n  label | brief reason\n\nThe label is exactly one of: keep, summarize, melt. The reason is 4-7 words.\n\nDistribution prior: most messages are MELT. Use KEEP only when the message contains something irreplaceable.\n\nkeep      = irreplaceable atom — root cause named, owner+channel, file:line, decision, code snippet that was committed\nsummarize = substantive but a one-line gist suffices (multi-step explanation, verbose tool dump with one fact)\nmelt      = greetings, acks, "lgtm/ok/sure", restatements, dead-end tangents, polite framing\n\nExamples (note "label | reason" format):\n"Sure. Share the file." -> melt | ack only\n"lgtm pushing." -> melt | ack only\n"agreed, ticket later" -> melt | ack/tangent\n"Root cause: Date.now() called twice across an assertion boundary." -> keep | root cause named\n"issueToken uses Date.now() for exp; test reads Date.now() again." -> keep | bug location + mechanism\n"Sarah owns @company/jwt-utils, #auth-platform" -> keep | ownership info\n"Read 87 lines, confirmed bug" -> summarize | tool output, one fact\n"Two reasons CI fails more: shared CPU stalls; narrow race window." -> summarize | multi-part explanation<|end|>\n<|user|>\n[${m.role}]\n${m.content}<|end|>\n<|assistant|>\n`

          // Capture per-layer post-norm residuals (step 8) during this tagger
          // call. Phi-3 has 32 layers × 3072-dim residual, so each tagger call
          // populates a fresh per-layer slice. Only collect when viz is wired.
          const layerActs: Map<number, Float32Array> = new Map()
          const cb = opts.viz ? {
            onLayer: (layer: number, step: number, _name: string, act?: Float32Array) => {
              // step 8 = post-residual-norm hidden state, 3072 dims, after each layer's FFN
              if (step === 8 && act && act.length === RESIDUAL_DIM && layer < RESIDUAL_LAYERS) {
                layerActs.set(layer, new Float32Array(act))
              }
            }
          } : {}

          const raw = await engine.generate(prompt, TAG_MAX_TOKENS, cb)
          const tag = parseTag(raw)
          tags.push(tag)
          const chip = document.createElement("span")
          chip.className = `bfly-tag ${tag.label}`
          // Inline reason if short; tooltip always shows the full thing.
          const reasonShort = tag.reason ? ` — ${tag.reason}` : ""
          chip.textContent = `${i}: ${tag.label}${reasonShort}`
          chip.title = tag.reason
            ? `${tag.label}: ${tag.reason}\n\nMessage: "${m.content.slice(0, 200)}${m.content.length > 200 ? "…" : ""}"`
            : `${tag.label}\n\nMessage: "${m.content.slice(0, 200)}${m.content.length > 200 ? "…" : ""}"`
          tagsRow.appendChild(chip)

          // Modulate captured residuals by tag importance and accumulate.
          if (opts.viz && layerActs.size > 0) {
            const factor = TAG_BRIGHTNESS[tag.label]
            decaySlab(INTRA_GEN_DECAY)  // fade older messages slightly each step
            for (const [layer, raw] of layerActs) {
              // Normalize raw activations to 0..1 using the engine's helper,
              // multiply by the tag's brightness factor, max-pool with
              // existing accumulator so brighter contributions dominate.
              const normalized = normalizeFull(raw, true)
              let acc = slabAcc.get(layer)
              if (!acc) { acc = new Float32Array(RESIDUAL_DIM); slabAcc.set(layer, acc) }
              for (let j = 0; j < RESIDUAL_DIM; j++) {
                const contribution = normalized[j] * factor
                if (contribution > acc[j]) acc[j] = contribution
              }
            }
            writeSlabToViz()
          }

          setProgress(((gen - 1) / N_GENERATIONS) * 100 + ((i + 1) / messages.length) * (100 / N_GENERATIONS) * 0.4)
        }
        const counts = tags.reduce(
          (acc, t) => ({ ...acc, [t.label]: (acc[t.label] || 0) + 1 }),
          {} as Record<string, number>
        )
        tagCountsEl.textContent = `keep:${counts.keep || 0} summ:${counts.summarize || 0} melt:${counts.melt || 0}`

        setStatus(`Gen ${gen}/${N_GENERATIONS}: chrysalis (rebuilding to ~${TARGET_TOKENS} tokens)…`)
        const taggedBlock = messages.map((m, i) => `[#${i} ${m.role} action=${tags[i].label}]\n${m.content}`).join("\n\n")
        const chrysPrompt = `<|system|>\nYou rebuild a tagged conversation transcript into a small coherent context the agent will resume from. HARD CONSTRAINT: ~${TARGET_TOKENS} tokens (~${TARGET_TOKENS * 4} chars). KEEP messages: preserve every load-bearing fact, name, file:line, decision. SUMMARIZE messages: collapse to one phrase. MELT messages: drop entirely. Output ONLY the rebuilt context, no preamble.<|end|>\n<|user|>\nTAGGED TRANSCRIPT:\n\n${taggedBlock}<|end|>\n<|assistant|>\n`
        const rebuilt = await engine.generate(chrysPrompt, CHRYSALIS_MAX, {})
        lastChrysalis = rebuilt
        chrysText.textContent = rebuilt || "(empty)"
        chrysText.classList.remove("empty")
        chrysToks.textContent = `~${tokens(rebuilt)} tok`

        setProgress((gen / N_GENERATIONS) * 100 * 0.85)

        // Replace messages with one synthetic system message (the rebuild)
        // and inject noise for the next generation, except the final.
        if (gen < N_GENERATIONS) {
          messages = [
            { role: "assistant" as const, content: `[REBUILT FROM GEN ${gen}]\n${rebuilt}` },
            ...NOISE_BATCH,
          ]
          // LastN chain: truncate then add noise. Track what got dropped
          // so we can show the learner the failure mechanism.
          const beforeLen = lastnMessages.length
          const truncated = pickLastN(lastnMessages, TARGET_TOKENS)
          const droppedCount = beforeLen - truncated.length
          if (droppedCount > 0) {
            const row = document.createElement("span")
            row.className = "bfly-throwaway-row"
            row.innerHTML = `<span class="gen">G${gen}</span>: <span class="dropped">−${droppedCount}</span> oldest msgs`
            row.title = `LastN truncated to ~${TARGET_TOKENS} tokens. Dropped ${droppedCount} message${droppedCount === 1 ? "" : "s"} from the front of the chain.`
            throwawayLog.appendChild(row)
            throwawayStage.style.display = ""
          }
          lastnMessages = [...truncated, ...NOISE_BATCH]
        }
      }

      // Final question — answer with each arm's last context.
      setStatus("Asking the needle question through both arms…")
      const beforeFinal = lastnMessages.length
      const finalLastN = pickLastN(lastnMessages, TARGET_TOKENS)
      const finalDropped = beforeFinal - finalLastN.length
      if (finalDropped > 0) {
        const row = document.createElement("span")
        row.className = "bfly-throwaway-row"
        row.innerHTML = `<span class="gen">final</span>: <span class="dropped">−${finalDropped}</span> oldest msgs`
        throwawayLog.appendChild(row)
        throwawayStage.style.display = ""
      }
      const ctxBfly = `[REBUILT CONTEXT]\n${lastChrysalis}`
      const ctxLast = asText(finalLastN)

      // Surface what each answer arm actually saw — collapsible.
      ctxBflyText.textContent = ctxBfly
      ctxBflyTokens.textContent = `~${tokens(ctxBfly)} tok`
      ctxBflyDetails.style.display = ""
      ctxLastText.textContent = ctxLast || "(empty — everything was truncated away)"
      ctxLastTokens.textContent = `~${tokens(ctxLast)} tok`
      ctxLastDetails.style.display = ""

      const ansPromptBfly = `<|system|>\nYou continue a prior conversation. Answer the follow-up using ONLY the prior context. If a fact isn't there, say so plainly — do not invent. Be concise.<|end|>\n<|user|>\nPRIOR CONTEXT:\n\n${ctxBfly}\n\nFOLLOW-UP: ${questionForRun}<|end|>\n<|assistant|>\n`
      const ansPromptLast = `<|system|>\nYou continue a prior conversation. Answer the follow-up using ONLY the prior context. If a fact isn't there, say so plainly — do not invent. Be concise.<|end|>\n<|user|>\nPRIOR CONTEXT:\n\n${ctxLast}\n\nFOLLOW-UP: ${questionForRun}<|end|>\n<|assistant|>\n`

      const aBfly = await engine.generate(ansPromptBfly, ANSWER_MAX, {})
      ansBfly.textContent = aBfly || "(empty)"

      setStatus("Judging butterfly answer with Phi-3 (rubric)…")
      const vBfly = await judgeAnswer(engine, questionForRun, needleForRun, aBfly)
      verdictBfly.textContent =
        vBfly === "hit" ? "✓ full preservation" :
        vBfly === "partial" ? "◐ partial preservation" :
        "✗ needle lost"
      verdictBfly.className = `bfly-verdict ${vBfly}`
      if (vBfly === "hit") armBfly.classList.add("win")

      setProgress(92)
      const aLast = await engine.generate(ansPromptLast, ANSWER_MAX, {})
      ansLastn.textContent = aLast || "(empty)"

      setStatus("Judging lastN answer with Phi-3 (rubric)…")
      const vLast = await judgeAnswer(engine, questionForRun, needleForRun, aLast)
      verdictLast.textContent =
        vLast === "hit" ? "✓ full preservation" :
        vLast === "partial" ? "◐ partial preservation" :
        "✗ needle lost"
      verdictLast.className = `bfly-verdict ${vLast}`
      if (vLast === "hit") armLastn.classList.add("win")

      const elapsed = ((performance.now() - t0) / 1000).toFixed(1)
      setStatus(`Done in ${elapsed}s. Butterfly: ${vBfly}. LastN: ${vLast}.`)
      setProgress(100)

      // Tier B: persist to localStorage tally + re-render the stats line.
      const scoreNum = (v: "hit" | "partial" | "miss"): 0|1|2 =>
        v === "hit" ? 2 : v === "partial" ? 1 : 0
      const entries = loadStats()
      entries.push({ bfly: scoreNum(vBfly), lastN: scoreNum(vLast), ts: Date.now() })
      saveStats(entries)
      renderStats()
    } catch (err) {
      setStatus(`Error: ${err}`)
    } finally {
      opts.setBusy(false)
      runBtn.disabled = false; resetBtn.disabled = false
      runBtn.textContent = "Run butterfly demo"
    }
  })
}

# Pre-registered ablation predictions

This is a versioned, dated list of **falsifiable predictions** about
Phi-3-mini's internals. Each entry is filed *before* the experiment is
run, and stays here regardless of whether the experiment confirms it.

The point is to convert the visualizer from "look at this cool tool"
into actual interpretability evidence. A demo that shows you can ablate
heads is a tool. A demo that *predicted* the effect of ablating a
specific head and was right is a finding.

## How an entry works

Each prediction has six fields:

- **id** — `P-YYYYMMDD-NN` so they can be referenced cleanly.
- **filed** — UTC date the prediction was committed (git provides the
  audit trail; the field exists for human readability).
- **author** — who filed it.
- **claim** — one or two sentences. Concrete enough to be wrong.
- **target** — the exact ablation: layer, head (or "FFN", or "all heads
  in layer N"), and what input/prompt class triggers the effect.
- **measure** — the falsifiable comparison. Usually a delta on
  top-1 token agreement, mean JSD, or a pinned eval accuracy.
- **threshold** — the bound that decides confirm vs. refute. State this
  as a numerical interval, not a vibe.
- **status** — one of: `open` · `confirmed` · `refuted` · `inconclusive`.
  Inconclusive means the data didn't separate the hypothesis from the
  null at the chosen threshold.
- **outcome** — left blank when filed; filled in when the experiment
  ran. Includes the run fingerprint and a link to the validation log.

## Predictions

### P-20260509-01 · Late-layer reflexive head

- **filed**: 2026-05-09
- **author**: ahmet
- **claim**: At least one head in layers 28–31 fires
  disproportionately on the literal token "attention" when the prompt
  is asking the model to explain itself. Ablating that head will degrade
  the model's ability to produce metacognitive answers but will not
  affect factual recall.
- **target**: To be located by inspection of the head-activity heatmap
  on the prompt *"Explain attention in one sentence."* — the brightest
  head in layers 28–31 on the final position.
- **measure**: top-1 token agreement vs. baseline on (a) the
  metacognitive prompt above; (b) the factual prompt
  *"The capital of Japan is"*.
- **threshold**: confirm if (a) drops by ≥ 30% on the metacognitive
  prompt AND (b) drops by ≤ 5% on the factual prompt; refute if (a)
  drops by ≤ 5% regardless of (b).
- **status**: open
- **outcome**: —

### P-20260509-02 · Mid-layer arithmetic carrier

- **filed**: 2026-05-09
- **author**: ahmet
- **claim**: Ablating the entire FFN slab in any one of layers 14–18
  on the prompt *"Step by step, solve: 17 * 23 ="* will produce a
  different first token (≠ "1") at the start of the multi-digit answer.
  Ablating the same slab in layers 0–6 will not.
- **target**: FFN ablation, single layer at a time, sweep L = {0, 4,
  10, 14, 16, 18, 24, 28, 31}.
- **measure**: first-decoded numeric token after the equals sign,
  compared to the unablated baseline.
- **threshold**: confirm if at least one layer in [14, 18] flips the
  first answer token AND no layer in [0, 6] does. Refute otherwise.
- **status**: open
- **outcome**: —

### P-20260509-03 · KV cache linearity

- **filed**: 2026-05-09
- **author**: ahmet
- **claim**: The KV cache page utilization grows exactly +1 page per
  16 generated tokens (page size 16). Across 100 tokens of generation
  starting from a 32-token prompt, the page count traces a perfectly
  staircase line.
- **target**: not an ablation — a structural sanity check on the
  visualizer's KV panel reading vs. internal `kvLen`.
- **measure**: max | kvPanel.pages × 16 − kvLen | over the 100 tokens.
- **threshold**: confirm if max diff = 0; refute otherwise.
- **status**: open (the audit-time read of the code suggests this is
  trivially true, but it's a useful regression watchdog).
- **outcome**: —

### P-20260509-04 · Logit lens convergence

- **filed**: 2026-05-09
- **author**: ahmet
- **claim**: For factual prompts (e.g. "The capital of Japan is"),
  the logit lens predicts the correct final token at layer ≤ 24 in at
  least 80% of cases. For chain-of-thought arithmetic prompts, lens
  agreement with the final answer happens only at layer ≥ 28 in at
  least 70% of cases.
- **target**: 20 prompts each from the "fact" and "math" categories
  (curate from the existing preset chips + variations).
- **measure**: per-prompt earliest-layer at which lens.argmax matches
  the eventual generated token.
- **threshold**: confirm if both percentages above hold; refute if
  factual converges later than arithmetic on average.
- **status**: open
- **outcome**: —

### P-20260512-05 · Butterfly compaction beats lastN at the same token budget

- **filed**: 2026-05-12
- **author**: ahmet
- **claim**: Across the four built-in Butterfly transcripts (`jwt-clock-race`,
  `auth-owner-pto`, `rate-limit-decision`, `cache-race-fileline`), the
  butterfly arm produces a strictly higher LLM-judge hit rate than the
  lastN baseline at the same `TARGET_TOKENS = 400` budget after
  `N_GENERATIONS = 3` metamorphoses with the standard `NOISE_BATCH`
  injection between generations. Concretely: on `jwt-clock-race`, where
  the needle is the load-bearing message and noise displaces it from
  the tail in later generations, the butterfly arm should hit at least
  twice as often as lastN.
- **target**: `src/butterfly-mode.ts` at build SHA recorded in the
  fingerprint footer when the run is executed. Phi-3-mini-4k q4f16_1
  via WebGPU. All four arms (tagger, chrysalis, butterfly answer, lastN
  answer) run unablated. Judge stays unablated. No editable-mode runs
  counted.
- **measure**: per-transcript LLM-judge verdict counts collected in the
  `butterfly-mode-stats-v1` localStorage tally, partitioned by the
  `transcript` field (added in v2.5). For each transcript: `bfly_hits`
  (verdict ≥ 1, i.e. partial or full), `lastn_hits`, over a minimum of
  20 runs per transcript on the same browser/GPU.
- **threshold**:
  - **Confirm** if for all 4 transcripts, `bfly_hits / N ≥ lastn_hits / N + 0.15`
    AND on `jwt-clock-race` specifically `bfly_hits ≥ 2 × lastn_hits`.
  - **Refute** if on any 2 of the 4 transcripts, `lastn_hits ≥ bfly_hits`.
  - **Inconclusive** otherwise — typically when one transcript flips the
    inequality but the rest hold; collect more samples or sharpen the
    needle question (the v2.1 commit history has the canonical case
    study of needle questions that pretraining can fake).
- **status**: open
- **outcome**: —
- **threats to validity** (declared up-front, not after the fact):
  - Same-model self-judge — the rubric meter is Phi-3-mini too. An
    external judge (Claude / GPT-4) behind a `?judge=` flag is a
    follow-up, not blocking.
  - Tagger/chrysalis are Phi-3-mini; a smaller model is doing
    interesting work in both. If the butterfly arm wins on the JWT
    transcript but loses on the cache-race one, that's evidence about
    tagger quality, not about the compaction mechanism.
  - `tokens(s) = s.length / 4` is a soft proxy. Both arms use the same
    proxy so the comparison is internally consistent, but "400 tokens"
    is approximate.

## Methodology notes

- Every prediction is run against the build SHA recorded in the
  fingerprint footer. If a kernel changes, predictions get re-run.
- A prediction filed with vague language ("some head") is rejected
  on review. The point is falsifiability.
- "Inconclusive" is a real outcome, not a hedge. If the data is
  noisy, say so and propose a follow-up with a tighter design.
- Predictions are append-only. To withdraw one, mark it
  `withdrawn` with a brief reason; do not delete.

# Butterfly compaction: we falsified our own pre-registration twice, then confirmed something narrower

A 7-day investigation of a small-LLM context-compaction mechanism that ran inside the [neuropulse](https://github.com/abgnydn/neuropulse) browser visualizer. The headline result isn't "tag-and-rebuild beats truncation." It's:

> **Butterfly's mechanism is a *content-shape adapter*, not a universal context manager. The engineering work lives in the tagger, not the rebuild step. Generic instruction-tuned LLMs prompted to "find important messages" do not reliably do this.**

Below: how we got there, what we filed, what broke, and what we'd file next.

## The mechanism

Original [butterfly-mode.ts](https://github.com/abgnydn/neuropulse/blob/main/src/butterfly-mode.ts) (Apr 2026) tested a 3-step context compaction loop:

1. **Tag** every message in a conversation as `keep` / `summarize` / `melt` using Phi-3-mini.
2. **Rebuild** (the "chrysalis") into a tight token budget: keep verbatim, summarize first-sentence, drop melt.
3. **Inject noise** and repeat for `N_GENERATIONS = 3` rounds.

The advertised win condition: across multiple cocoons with off-topic noise compounding between rounds, this should preserve load-bearing facts better than a naive `lastN` token-budget truncation of the same conversation.

It looked great in the original demo — runs of 25-30 s on M2 Pro WebGPU, with the demo's residual-stream visualization brightening on keep-tagged content as it survived each generation.

## Round 1: pre-registration P-20260512-05, refuted

Filed: needle preservation should beat lastN by ≥ 15pp across 4 transcripts × 20 runs each at the original (12 messages, 400-token budget, 1 generation) demo regime. Confirmation requires bfly ≥ 2 × lastn on the most adversarial of the four.

What broke: the wallclock cost on M2 Pro. Each butterfly run is ~26K Phi-3-mini tokens (24 tagger calls × a 700-token few-shot prompt, 3 chrysalis passes, 2 answers, 2 judges). At the design target of ~30s per run that's a 1-hour sweep. Under Chrome background-tab throttling on a real machine — what actually happened, six attempts over four days, only one completed run before the harness timed out.

Pivot: re-run the same protocol in pure code with a regex tagger. Same 4 transcripts, no LLM in the loop, same scoring (substring-match against load-bearing keywords in the expected fact). Total runtime: 4 ms.

Result: **REFUTED.** LastN ties butterfly on 2 of 4 transcripts and beats it on the other 2 (by 12pp and 17pp). At a generous 400-token budget on 12-message transcripts, lastN already captures the needle. There's no room for the mechanism to differentiate.

## Round 2: pre-registration P-20260515-06, confirmed (then hardened)

If the original regime didn't force compaction to matter, file a regime that does. 38-message transcripts (15 pre + ~8 core + 15 post melt-able padding), 100-token budget, 3 generations with fresh noise injected each round. Same threshold: bfly > lastn on all 4, mean Δ ≥ 0.20.

Result: **CONFIRMED.** 4/4 transcripts: butterfly 100% needle preservation, lastN 0%. Mean Δ = 100pp. Runs in 4 ms.

We swept the cube to map the boundary — 8 budgets × 6 lengths × 5 generations × 4 transcripts = 960 cells in 82 ms. The breakeven curve is roughly diagonal: butterfly stops mattering once budget grows past ~30-40% of original transcript size. More generations expand the win region.

```
─── gens = 3 ─────────────────────────────────────────────────
len ↓ \ budget →   50   75  100  150  200  300  400  600
  12               ▸▸   ██   ██   ██   ▸▸    ·    ·    ·
  20               ▸▸   ██   ██   ██   ██    ·    ·    ·
  30               ▸▸   ██   ██   ██   ██    ·    ·    ·
  50               ▸▸   ██   ██   ██   ██   ██    ·    ·
  80               ▸▸   ██   ██   ██   ██   ██   ██    ·
 120               ▸▸   ██   ██   ██   ██   ██   ██   ██

  ██ Δ≥0.60  ▸▸ 0.30..0.60  ▸ 0.10..0.30  · -0.10..0.10
```

The pre-registered REFUTED point (`len=12, bud=400, gens=1`) sits in the `·` tie zone. The CONFIRMED point (`len≈30, bud=100, gens=3`) sits deep in the `██` win zone. Both outcomes were samples from a phase structure with a real boundary, not random luck.

## Round 3: the tagger swap — replication failure with a learned tagger

The obvious next objection: the regex tagger has hand-tuned bias toward needle-shaped patterns (`file:line`, `#channel`, `@org/pkg`, `Decision:`, code calls). Does the mechanism still work with a learned tagger?

Swapped regex for a batched LLM call to qwen3-14b on LM Studio (with `/no_think` to suppress reasoning content emission). Same 4 transcripts, same protocol, two configs.

| config (LLM tagger, qwen3-14b-mlx) | regex result | LLM result |
|---|---|---|
| easy regime (len=12, bud=400, gens=1) | tie (100/100) | tie (100/100) |
| hard regime (len=38, bud=100, gens=3) | **100/0  Δ=100pp** | **0/0  Δ=0pp** |

The hard regime **no longer wins** with a learned tagger.

Diagnosis from the per-generation traces:

1. **Gen-1 over-tagging.** Qwen3 marked 52-63% of messages as keep (vs the regex's ~8%). At a 100-token budget the chrysalis bloats with non-needle content and the needle gets truncated.
2. **JSON parse failures at gen 2** in 3 of 4 transcripts → regex fallback. But the rebuilt-from-gen-1 string doesn't carry file-path/channel signals, so the regex fallback tags it mostly as melt → 5-token rebuild → needle gone.

I tried four more LLM-tagger configurations to chase this down: one-char output to eliminate JSON parse failures, hard cap on keep count (`MAX_KEEPS=5`, then 3), gemma-4-e4b instead of qwen3-14b, identifier-first prompts spelling out exactly what the model should consider "irreplaceable." **All failed**. With cap=3 the LLM successfully picks 3 messages — they're just not the needle-carrying ones. Qwen3 prioritizes decision-language and emphatic statements ("Decision:", "Let's do X") but the needles in our transcripts are *literal identifier shapes* — `lib/jwt.ts`, `#auth-platform`, `Date.now()`. Different prior, different choice.

## Round 4: trained classifiers — can a tiny learned model match regex?

Two more taggers, both trained on the regex's own labels via plain softmax regression. No LLM in the loop.

| tagger | params | size | hard-regime result |
|---|---|---|---|
| Regex (hand-tuned thresholds) | ~14 hard rules | source code | **100/0 Δ=100pp** |
| 14-feature softmax classifier | 45 | 1.2 KB | **100/0 Δ=100pp** |
| 768-dim embed + linear head (nomic-embed) | 2,307 | 45 KB | **100/0 Δ=100pp** |
| qwen3-14b · onechar · cap=3 | (frontier) | — | 0/0 Δ=0pp |

Both learned classifiers replicate the regex's win exactly. This is *partly* tautological — they were trained on regex labels, so of course they learn regex's boundaries — but it tells us something concrete: the mechanism's win isn't hiding in pathological regex code. It's in the **feature distribution the regex weighs**. Gradient descent on either 14 hand features OR 768-dim raw text embeddings recovers the same boundary.

## Round 5: adversarial transcripts — the shape-specificity claim, directly tested

The shape-specificity hypothesis: butterfly wins because the regex tagger's bias toward identifier patterns matches the needle distribution in our 4 transcripts. If we wrote transcripts where the needle is real load-bearing content that does NOT take identifier shapes, the regex (and learned descendants) should fail.

Four new transcripts:

| transcript | needle | needle shape |
|---|---|---|
| `numeric-threshold` | "hard cap of 47 concurrent connections per pod" | number in prose, no `req/min` |
| `implicit-deadline` | "cooper said end of next week, friday the 24th" | lowercase name + relative date |
| `preference-statement` | "i don't want to go with postgres — too heavy for our write pattern" | preference, no `Decision:` |
| `buried-causation` | "the rollback brought back the version-pinned dependencies that were the actual blocker" | causation in prose, no `Root cause:` |

Results on these 4 × same hard regime (38 msgs, 100 tok, 3 gens):

```
tagger              mean Δ across 4 adversarial transcripts
─────────────────────────────────────────────────────────
regex                0pp   (every message → melt; needles invisible)
trained (14-feat)    0pp   (inherits regex's blind spots)
embed (768-dim)      0pp   (trained on regex labels, same)
qwen3-14b · onechar  0pp   (picks 5 messages with cap, none are the needle)
```

**Every tagger fails.** The regex and its descendants tag every message in adversarial as `melt` — no feature fires. Qwen3 selects 5 messages each gen but they're not the needle-carrying ones.

And one new twist that the original confirmation didn't show: across all 4 adversarial transcripts, the qwen3 gen-3 chrysalis output is **identical** — a noise message from gen 2 injection. The original conversation is completely gone by gen 3. The multi-generation noise injection acts as a **feedback loop** that amplifies tagger weakness at gen 1. Once the needle is dropped in the first cocoon, no later generation can recover it.

## The fully hardened claim

> Butterfly's tag-and-rebuild mechanism beats `lastN` truncation at tight budgets under noise compounding **only when the tagger's prior matches the load-bearing-content distribution in the transcripts being compacted.**
>
> The compaction mechanism is real but it's a **content-shape adapter**, not a universal context manager. Generic "find what's important" prompts on frontier instruction-tuned LLMs do not reliably replicate it. A 45-parameter softmax classifier trained on labeled examples of your domain's load-bearing shapes does.
>
> **The engineering problem is the tagger, not the mechanism.**

## What this is good for

A real use case: long-running local agents on a small open model. If your domain has stable load-bearing shapes (engineering chat → file paths + tickets, customer support → product names + order IDs, medical scribing → drug names + dosages), train a 1-10 KB classifier on labeled examples, plug it into butterfly's rebuild loop, ship.

A real *non*-use case: a context-window manager that works regardless of what conversations you have. That's still an open problem. Butterfly isn't it.

## What I'd file next

- **P-20260520-07**: train a small classifier on the *adversarial* transcripts' hand-labels (not regex labels) and re-test. Hypothesis: a learned classifier with the right training data CAN see non-shape needles, and the mechanism wins again. This is the direct test of "is the tagger the only bottleneck?"
- **Multi-domain transcripts**: write or source 4 transcripts in each of 3+ different domains (eng, customer, medical, financial). Train a single classifier on all of them. Does it generalize, or does it pick up only the cross-domain shape commonalities?
- **Lower-bound on training data**: how many labeled messages does the 45-parameter classifier need to converge? 100 was enough for our 4 transcripts. Probably much less is enough; would be useful to characterize.

## Receipts

All code, all results, all per-generation traces — pure JS, no dependencies — at [github.com/abgnydn/neuropulse/tree/main/tools](https://github.com/abgnydn/neuropulse/tree/main/tools):

- `butterfly-purecode-hard.mjs` — the 4-ms run that originally confirmed P-20260515-06
- `butterfly-sweep-phasediagram.mjs` — the 82-ms cube sweep (960 cells)
- `butterfly-llm-tagger.mjs` — LLM, trained, and embed taggers in a single pipeline
- `butterfly-train-classifier.mjs` — train the 1.2 KB hand-feature classifier
- `butterfly-train-embed.mjs` — train the 45 KB embedding classifier
- `butterfly-adversarial.mjs` — the 4 adversarial transcripts + 4 taggers

Pre-registrations and outcomes: [PREDICTIONS.md](https://github.com/abgnydn/neuropulse/blob/main/PREDICTIONS.md), entries P-20260512-05 and P-20260515-06.

`npm run check` passes; everything is reproducible from the repo as-is. The embedding cache is `tools/butterfly-embed-cache.json` (gitignored, 2.5 MB) — re-generated on first run if absent.

/**
 * LESSONS — a sequenced learning path over the live model. Data-only, like
 * src/tours.ts: a lesson is an optional guided tour + optional glossary reading
 * + a CHECK that must pass to mark it complete.
 *
 *   - 'quiz'   → answered in the UI, no model needed.
 *   - 'signal' → passes when a real runtime event fires (a generation, an
 *                ablation run, a sweep), optionally gated on impact ≥ minImpact.
 *                This is the "verified on the live model" kind.
 *
 * Wiring lives in src/main.ts (`wireLessons`), which reuses the tour runner and
 * `openGlossaryAt` for reading links. Add a lesson = add an object here.
 */

export type LessonCheck =
  | {
      kind: 'quiz'
      question: string
      options: string[]
      /** index into `options` that is correct */
      answer: number
      /** shown after a correct answer */
      why: string
    }
  | {
      kind: 'signal'
      signal: 'generate' | 'ablation' | 'sweep'
      /** if set, the signal must carry impact ≥ this to pass (0..1) */
      minImpact?: number
      instruction: string
    }

export interface Lesson {
  id: string
  title: string
  blurb: string
  /** id of an existing tour in src/tours.ts to play when the lesson starts */
  tourId?: string
  /** glossary entry ids (gloss-*) surfaced as "Read:" deep-links */
  reading?: string[]
  check: LessonCheck
}

export const LESSONS: Lesson[] = [
  {
    id: 'intro',
    title: 'What am I looking at?',
    blurb: 'A guided flythrough of the whole model, end to end.',
    tourId: 'orientation',
    check: {
      kind: 'quiz',
      question: 'What flows along the central axis — read from and added to by every layer?',
      options: ['The attention heads', 'The residual stream', 'The KV cache', 'The token strip'],
      answer: 1,
      why: 'The residual stream — a 3,072-dim vector every layer reads from and writes back to. Information accumulates rather than being replaced.',
    },
  },
  {
    id: 'recall',
    title: 'How it turns thought into a token',
    blurb: 'Watch a forward pass pick the next word from the logits.',
    tourId: 'factual-recall',
    reading: ['gloss-token'],
    check: {
      kind: 'signal',
      signal: 'generate',
      instruction: 'Type any prompt and press Think — watch a token get picked as the answer streams out.',
    },
  },
  {
    id: 'attention',
    title: 'Attention: who looks at whom',
    blurb: 'The heads that decide which earlier tokens matter.',
    tourId: 'attention-story',
    reading: ['gloss-attention'],
    check: {
      kind: 'quiz',
      question: 'How many attention heads does Phi-3-mini have in total?',
      options: ['32', '96', '1,024', '32,064'],
      answer: 2,
      why: '32 heads × 32 layers = 1,024. Each head learns to look at earlier tokens in its own way.',
    },
  },
  {
    id: 'uncertainty',
    title: 'Two signals of uncertainty',
    blurb: 'How to tell when the model is guessing vs. sure.',
    tourId: 'confidence',
    reading: ['gloss-softmax'],
    check: {
      kind: 'quiz',
      question: 'A flat top-K distribution and an amber confidence bar mean the model is…',
      options: ['very confident', 'hedging / uncertain', 'out of memory', 'done generating'],
      answer: 1,
      why: 'Flat probabilities = low confidence (high entropy). This is where temperature sampling changes the output the most.',
    },
  },
  {
    id: 'ablation',
    title: 'Ablation: prove a part is doing work',
    blurb: 'Switch off attention heads and make the answer change.',
    reading: ['gloss-ablation', 'gloss-sweep'],
    check: {
      kind: 'signal',
      signal: 'ablation',
      minImpact: 0.15,
      instruction: 'Open the ablation panel (press A), shift-click a few attention heads in the 3D scene, then Run ablated. Change the answer enough and this completes.',
    },
  },
]

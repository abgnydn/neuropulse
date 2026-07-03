/**
 * LESSONS — a sequenced learning path over the live model. Data-only, like
 * src/tours.ts: a lesson is an optional guided tour + optional glossary reading
 * + a CHECK that must pass to mark it complete.
 *
 *   - 'quiz'   → answered in the UI, no model needed. Every option carries its
 *                own `why`, so wrong answers TEACH (shown, then options
 *                reshuffle — guessing costs a re-read, position memory won't
 *                help).
 *   - 'signal' → passes when a real runtime event fires (a generation, an
 *                ablation run, a sweep), optionally gated on impact ≥ minImpact.
 *                This is the "verified on the live model" kind.
 *
 * Wiring lives in src/main.ts (`wireLessons`), which reuses the tour runner and
 * `openGlossaryAt` for reading links. Add a lesson = add an object here.
 * `minutes` is a rough guide shown in the list (dynamic DOM only — the
 * verify-claims doc gate scans static files, keep numbers out of those).
 */

export interface QuizOption {
  text: string
  /** shown when this option is picked — right or wrong, it teaches */
  why: string
}

export type LessonCheck =
  | {
      kind: 'quiz'
      question: string
      options: QuizOption[]
      /** index into `options` that is correct */
      answer: number
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
  /** what the learner can do after this lesson — shown in the list */
  objective: string
  /** rough duration in minutes — shown in the list (dynamic DOM only) */
  minutes: number
  /** id of an existing tour in src/tours.ts to play when the lesson starts */
  tourId?: string
  /** glossary entry ids (gloss-*) surfaced as "Read:" deep-links */
  reading?: string[]
  /** optional HTML shown above the check — short orientation copy */
  intro?: string
  /** true = the check needs the live engine (can't complete in demo mode) */
  requiresLive?: boolean
  check: LessonCheck
}

export const LESSONS: Lesson[] = [
  {
    id: 'bearings',
    title: 'Get your bearings',
    blurb: 'The four moves that unlock everything else.',
    objective: 'Navigate the scene and know that everything on screen is real.',
    minutes: 2,
    intro:
      '<p><b>1.</b> Type a prompt (or click a preset) and watch the model process it — every point of light is a real tensor.</p>' +
      '<p><b>2.</b> Press <b>space</b> or the arrow keys to journey through the layers; the wheel zooms.</p>' +
      '<p><b>3.</b> Click a glowing pip to open its panel — each has an <b>i</b> button explaining what it shows.</p>' +
      '<p><b>4.</b> Click any head or slab in the 3D scene to inspect it. Drag to orbit.</p>',
    check: {
      kind: 'quiz',
      question: 'What does every point of light in the 3D scene represent?',
      options: [
        { text: 'A decorative particle', why: 'Nothing here is decoration — that is the whole point. Each point is driven by a value read from the model.' },
        { text: 'A real value read from the running model', why: 'Strict 1:1 — brightness is the activation value, read straight from the model. Nothing on screen is decorative.' },
        { text: 'A random animation', why: 'Nothing is randomized. Pause and scrub the token strip — the same tensors come back.' },
        { text: 'A pixel of a video', why: 'No video anywhere: the scene is rendered live from tensor values every frame.' },
      ],
      answer: 1,
    },
  },
  {
    id: 'recall',
    title: 'How it turns thought into a token',
    blurb: 'Watch a forward pass pick the next word from the logits.',
    objective: 'Follow one token from prompt to pick.',
    minutes: 3,
    tourId: 'factual-recall',
    reading: ['gloss-token'],
    check: {
      kind: 'signal',
      signal: 'generate',
      instruction: 'Run a generation and watch a token get picked as the answer streams out. (In demo mode, press Play recording.)',
    },
  },
  {
    id: 'attention',
    title: 'Attention: who looks at whom',
    blurb: 'The heads that decide which earlier tokens matter.',
    objective: 'Read the head-activity heatmap and attention rays.',
    minutes: 3,
    tourId: 'attention-story',
    reading: ['gloss-attention'],
    check: {
      kind: 'quiz',
      question: 'How many attention heads does Phi-3-mini have in total?',
      options: [
        { text: '32', why: '32 is the count per layer — but there are 32 layers of them.' },
        { text: '96', why: '96 is each head’s dimension, not the head count.' },
        { text: '1,024', why: '32 heads × 32 layers = 1,024. Each head learns to look at earlier tokens in its own way.' },
        { text: '32,064', why: '32,064 is the vocabulary size — the number of tokens the model can choose from.' },
      ],
      answer: 2,
    },
  },
  {
    id: 'uncertainty',
    title: 'Two signals of uncertainty',
    blurb: 'How to tell when the model is guessing vs. sure.',
    objective: 'Spot hedging from the top-K spread and confidence bar.',
    minutes: 2,
    tourId: 'confidence',
    reading: ['gloss-softmax'],
    check: {
      kind: 'quiz',
      question: 'A flat top-K distribution and an amber confidence bar mean the model is…',
      options: [
        { text: 'very confident', why: 'Confidence looks like ONE dominant bar — a peaky distribution, green meter.' },
        { text: 'hedging / uncertain', why: 'Flat probabilities = high entropy = low confidence. This is where temperature sampling changes the output the most.' },
        { text: 'out of memory', why: 'Memory pressure shows in the KV-cache panel, not the probability spread.' },
        { text: 'done generating', why: 'Generation ends with a stop token — the spread can be flat or peaky at any point.' },
      ],
      answer: 1,
    },
  },
  {
    id: 'ablation',
    title: 'Ablation: prove a part is doing work',
    blurb: 'Switch off attention heads and make the answer change.',
    objective: 'Run a controlled knock-out experiment on the live model.',
    minutes: 5,
    reading: ['gloss-ablation', 'gloss-sweep'],
    requiresLive: true,
    check: {
      kind: 'signal',
      signal: 'ablation',
      minImpact: 0.15,
      instruction: 'Open the ablation panel (press A), shift-click a few attention heads in the 3D scene, then Run ablated. Change the answer enough and this completes.',
    },
  },
]

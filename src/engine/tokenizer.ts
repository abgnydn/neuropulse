/**
 * TOKENIZER — Load and run BPE tokenizer from tokenizer.json.
 *
 * Implements BPE encode + decode for Phi-3 Mini (SentencePiece-style).
 * Loads tokenizer.json from browser cache or HuggingFace directly.
 *
 * Phi-3 Mini uses:
 *   - Pre-tokenizer: Metaspace (▁ replaces leading space)
 *   - Model: BPE
 *   - Added tokens: <s>=1, </s>=2, <|system|>=32006, <|user|>=32010, etc.
 */

import { PHI3_MODEL_BASE, CACHE_NAME } from './weight-loader'

// ============================================================
// tokenizer.json types (subset we need)
// ============================================================

interface TokenizerJSON {
  model: {
    type: string
    vocab: Record<string, number>
    merges: string[]
  }
  added_tokens: Array<{
    id: number
    content: string
    single_word: boolean
    lstrip: boolean
    rstrip: boolean
    normalized: boolean
    special: boolean
  }>
  pre_tokenizer?: {
    type: string
    prepend_scheme?: string
    replacement?: string
  }
}

// ============================================================
// Fetch (with cache fallback — same helper pattern as weight-loader)
// ============================================================

async function fetchText(url: string): Promise<string> {
  try {
    const cacheNames = await caches.keys()
    for (const name of cacheNames) {
      const store = await caches.open(name)
      const resp = await store.match(url)
      if (resp) return resp.text()
    }
  } catch { /* no Cache API */ }

  // Retry with exponential backoff — HF CDN can drop connections
  let lastErr: unknown
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const resp = await fetch(url, { cache: 'force-cache' })
      if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching tokenizer.json`)
      const text = await resp.text()
      // Persist into our cache so next load is instant & offline-safe
      try {
        const store = await caches.open(CACHE_NAME)
        await store.put(url, new Response(text, {
          headers: { 'Content-Type': 'application/json' },
        }))
      } catch { /* ok */ }
      return text
    } catch (e) {
      lastErr = e
      await new Promise(r => setTimeout(r, 500 * (attempt + 1)))
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error('tokenizer fetch failed')
}

// ============================================================
// BPE implementation
// ============================================================

export interface Tokenizer {
  encode(text: string): number[]
  decode(ids: number[] | Int32Array): string
  eosId: number
  bosId: number
}

// The metaspace character (▁, U+2581) is used as space prefix in SentencePiece
const METASPACE = '\u2581'

// Shared encoder for UTF-8 byte fallback path.
const utf8Encoder = new TextEncoder()

export async function loadTokenizer(onProgress?: (msg: string) => void): Promise<Tokenizer> {
  onProgress?.('Loading tokenizer.json...')
  const url = PHI3_MODEL_BASE + 'tokenizer.json'
  const json: TokenizerJSON = JSON.parse(await fetchText(url))

  const vocab = json.model.vocab   // token_str → id
  const merges = json.model.merges  // ["tok1 tok2", ...]

  // Build reverse vocab: id → token_str
  const idToToken = new Array<string>(Math.max(...Object.values(vocab)) + 1)
  for (const [tok, id] of Object.entries(vocab)) idToToken[id] = tok

  // Override with added_tokens (special tokens)
  for (const at of json.added_tokens ?? []) idToToken[at.id] = at.content

  // Build merge priority map: "tok1 tok2" → rank
  const mergeRank = new Map<string, number>()
  for (let i = 0; i < merges.length; i++) mergeRank.set(merges[i], i)

  // Build added tokens set for fast lookup during encode
  const addedTokens = new Map<string, number>()
  for (const at of json.added_tokens ?? []) addedTokens.set(at.content, at.id)

  const bosId = vocab['<s>'] ?? 1
  const eosId = vocab['</s>'] ?? 2

  // --------------------------------------------------------
  // BPE core: given a list of symbols, apply merges until no more apply
  // --------------------------------------------------------
  function bpe(symbols: string[]): string[] {
    if (symbols.length <= 1) return symbols

    while (true) {
      let bestRank = Infinity
      let bestIdx = -1

      for (let i = 0; i < symbols.length - 1; i++) {
        const pair = symbols[i] + ' ' + symbols[i + 1]
        const rank = mergeRank.get(pair)
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank
          bestIdx = i
        }
      }

      if (bestIdx === -1) break

      // Merge symbols[bestIdx] and symbols[bestIdx+1]
      const merged = symbols[bestIdx] + symbols[bestIdx + 1]
      symbols = [...symbols.slice(0, bestIdx), merged, ...symbols.slice(bestIdx + 2)]
    }

    return symbols
  }

  // --------------------------------------------------------
  // Encode: text → token IDs
  //
  // Algorithm:
  // 1. Scan for added (special) tokens first
  // 2. For remaining text: add metaspace prefix, split into chars, BPE
  // 3. Map each BPE token → vocab ID
  // --------------------------------------------------------
  function encode(text: string): number[] {
    const result: number[] = []

    // Split text around special tokens (longest match first)
    const specialPattern = [...addedTokens.keys()]
      .sort((a, b) => b.length - a.length)
      .map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
      .join('|')

    const parts = specialPattern
      ? text.split(new RegExp(`(${specialPattern})`))
      : [text]

    for (const part of parts) {
      if (!part) continue

      // Check if it's a special token
      const specialId = addedTokens.get(part)
      if (specialId !== undefined) {
        result.push(specialId)
        continue
      }

      // Normal text: split into words (preserving whitespace structure)
      // Metaspace pre-tokenizer: add ▁ before each word
      const words = part.split(/(\s+)/)
      let isFirst = true

      for (const word of words) {
        if (!word) continue

        if (/^\s+$/.test(word)) {
          // Whitespace handled by adding ▁ prefix to next word
          isFirst = false
          continue
        }

        // Add ▁ prefix (Metaspace: space becomes ▁ at start of word)
        const prefixed = (isFirst && result.length === 0 ? '' : METASPACE) + word
        isFirst = false

        // Convert to individual chars for BPE
        const chars = [...prefixed]  // Unicode-aware split

        // Apply BPE
        const merged = bpe(chars)

        // Map merged tokens to IDs
        for (const tok of merged) {
          const id = vocab[tok]
          if (id !== undefined) {
            result.push(id)
          } else {
            // Unknown token: LLaMA/Phi-3 byte fallback. The vocab contains
            // <0x00>..<0xFF> as special tokens; emit the token's UTF-8 bytes
            // one at a time so e.g. "🚀" → <0xF0><0x9F><0x9A><0x80>. Splitting
            // by codepoint and looking up e.g. vocab["🚀"] is useless because
            // only ASCII codepoints exist as single-char vocab entries.
            const bytes = utf8Encoder.encode(tok)
            for (let b = 0; b < bytes.length; b++) {
              const hex = bytes[b].toString(16).toUpperCase().padStart(2, '0')
              const byteTok = `<0x${hex}>`
              const byteId = vocab[byteTok]
              result.push(byteId ?? vocab['<unk>'] ?? 0)
            }
          }
        }
      }
    }

    return result
  }

  // --------------------------------------------------------
  // Decode: token IDs → text
  //
  // Simply look up each ID in idToToken, replace ▁ with space.
  // --------------------------------------------------------
  function decode(ids: number[] | Int32Array): string {
    let text = ''
    for (const id of ids) {
      if (id < 0) continue
      const tok = idToToken[id]
      if (!tok) continue
      // Skip BOS/EOS/PAD in output
      if (tok === '<s>' || tok === '</s>' || tok === '<pad>') continue
      text += tok
    }
    // Replace metaspace with actual space, trim leading space
    return text.replace(new RegExp(METASPACE, 'g'), ' ').trimStart()
  }

  onProgress?.('Tokenizer ready')
  return { encode, decode, bosId, eosId }
}

// ============================================================
// Chat template for Phi-3
//
// <|system|>...<|end|>\n<|user|>...<|end|>\n<|assistant|>\n
// ============================================================

export function buildChatPrompt(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  tokenizer: Tokenizer
): number[] {
  let text = ''
  for (const msg of messages) {
    if (msg.role === 'system') {
      text += `<|system|>\n${msg.content}<|end|>\n`
    } else if (msg.role === 'user') {
      text += `<|user|>\n${msg.content}<|end|>\n`
    } else {
      text += `<|assistant|>\n${msg.content}<|end|>\n`
    }
  }
  text += '<|assistant|>\n'
  return tokenizer.encode(text)
}

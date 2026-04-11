"""
HuggingFace golden reference dumper for Neural Pulse.

Produces a single reference.json with three sections:

  main:        The primary prompt ("What is consciousness?") with:
                 - full input ids after Phi-3 chat template
                 - full 3072-dim hidden states at layers 0,4,8,12,16,20,24,
                   28,31 and the embedding output (last position)
                 - 20 greedy decode steps, each with top-20 probabilities

  sweep:       15 short prompts covering ASCII, numbers, punctuation, code,
                 Unicode (Japanese), emoji, JSON. Each entry has:
                 - input ids (for tokenizer cross-check)
                 - 5 greedy decode steps with top-10 probabilities
                 No hidden states (keeps reference.json small).

  longContext: A ~400-token passage to exercise paged KV cache past a
                 single page. 10 greedy decode steps with top-10.

Usage
-----
    python dump_phi3_reference.py --dtype float16
    cp reference.json ../public/

NOTE on quantization: Neural Pulse runs q4f16_1 MLC weights. This
reference is fp16. Divergence is the quantization floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


LAYERS_TO_CAPTURE = [0, 4, 8, 12, 16, 20, 24, 28, 31]


# Primary prompt — gets full hidden-state dump + 20-step decode
MAIN_PROMPT = "What is consciousness?"
MAIN_STEPS = 20
MAIN_TOP_K = 20


# Sweep prompts — tokenizer + 5-step logit coverage
SWEEP_PROMPTS = [
    "Hello, world!",
    "What is 2 + 2?",
    "Name three colors.",
    "Write a Python function to reverse a string.",
    "Explain gravity in one sentence.",
    "The quick brown fox jumps over the lazy dog.",
    "こんにちは、元気ですか?",
    "🚀 rocket emoji test",
    "Parse JSON: {\"key\": 42, \"list\": [1,2,3]}",
    "Why is the sky blue?",
    "List 5 fruits.",
    "Translate hello to French.",
    "What is 15 percent of 80?",
    "Who wrote Hamlet?",
    "Define recursion.",
]
SWEEP_STEPS = 5
SWEEP_TOP_K = 10


# Long-context prompt — exercises KV paging beyond 128-token page boundary.
# ~500-token passage (Wikipedia-style prose) to ensure kv_len > PAGE_SIZE.
LONG_CONTEXT_PROMPT = (
    "The history of computing hardware covers the developments from early "
    "simple devices to aid calculation to modern day computers. The first "
    "aids to computation were purely mechanical devices which required the "
    "operator to set up the initial values of an elementary arithmetic "
    "operation, then manipulate the device to obtain the result. In later "
    "stages, computing devices began representing numbers in continuous "
    "forms, such as distance along a scale, rotation of a shaft, or a "
    "voltage. Numbers could also be represented in the form of digits, "
    "automatically manipulated by a mechanism. Although this approach "
    "generally required more complex mechanisms, it greatly increased the "
    "precision of results. The development of transistor technology and "
    "then the integrated circuit chip led to a series of breakthroughs, "
    "starting with transistor computers and then integrated circuit "
    "computers, causing digital computers to largely replace analog "
    "computers. Metal-oxide-semiconductor (MOS) large-scale integration "
    "(LSI) then enabled semiconductor memory and the microprocessor, "
    "leading to another key breakthrough, the miniaturized personal "
    "computer (PC), in the 1970s. The cost of computers gradually became "
    "so low that personal computers by the 1990s, and then mobile "
    "computers (smartphones and tablets) in the 2000s, became ubiquitous. "
    "Summarize this passage in one sentence."
)
LONG_STEPS = 10
LONG_TOP_K = 10


def build_chat_prompt(tokenizer, user_message: str) -> str:
    """Apply Phi-3's chat template the same way Neural Pulse does."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def greedy_decode_with_topk(model, tokenizer, input_ids, steps, top_k, capture_hidden):
    """Run prefill + `steps` greedy decode steps, capturing top-k probs at
    each step. Optionally returns the last-position hidden states from the
    prefill pass. Returns (decode_steps, hidden_states_or_None)."""
    import torch

    with torch.no_grad():
        prefill = model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=capture_hidden,
        )

    per_layer_hidden = None
    if capture_hidden:
        per_layer_hidden = {}
        for layer_idx in LAYERS_TO_CAPTURE:
            hs = prefill.hidden_states[layer_idx + 1]
            per_layer_hidden[str(layer_idx)] = hs[0, -1, :].float().tolist()
        per_layer_hidden["embedding"] = (
            prefill.hidden_states[0][0, -1, :].float().tolist()
        )

    # Step 0: top-k from prefill's last logit
    last_logits = prefill.logits[0, -1, :].float()
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=top_k)
    step0_top = [
        {"id": int(t), "token": tokenizer.decode([int(t)]), "prob": float(p)}
        for p, t in zip(top_probs.tolist(), top_ids.tolist())
    ]
    next_id = int(top_ids[0].item())
    decode_steps = [{"argmax": next_id, "top": step0_top}]

    past = prefill.past_key_values
    with torch.no_grad():
        for _ in range(1, steps):
            out = model(
                input_ids=torch.tensor([[next_id]], dtype=torch.long),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits = out.logits[0, -1, :].float()
            p = torch.softmax(logits, dim=-1)
            tp, ti = torch.topk(p, k=top_k)
            top = [
                {"id": int(t), "token": tokenizer.decode([int(t)]), "prob": float(q)}
                for q, t in zip(tp.tolist(), ti.tolist())
            ]
            next_id = int(ti[0].item())
            decode_steps.append({"argmax": next_id, "top": top})

    return decode_steps, per_layer_hidden


def dump_prompt(model, tokenizer, prompt, steps, top_k, capture_hidden):
    """Tokenize + prefill + greedy decode a single prompt, return a dict."""
    import torch

    chat_text = build_chat_prompt(tokenizer, prompt)
    input_ids = tokenizer(chat_text, return_tensors="pt").input_ids
    prompt_len = int(input_ids.shape[-1])

    decode_steps, hidden_states = greedy_decode_with_topk(
        model, tokenizer, input_ids, steps, top_k, capture_hidden
    )

    entry = {
        "prompt": prompt,
        "promptTokens": prompt_len,
        "inputIds": input_ids[0].tolist(),
        "decodeSteps": decode_steps,
    }
    if hidden_states is not None:
        entry["hiddenStates"] = hidden_states
        entry["layersCaptured"] = LAYERS_TO_CAPTURE
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--out", type=Path, default=Path("reference.json"))
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading {args.model} ({args.dtype}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.eval()

    # -------- Main prompt (full hidden states + 20 steps) --------
    print(f"\n[main] '{MAIN_PROMPT}'", flush=True)
    main_entry = dump_prompt(
        model, tokenizer, MAIN_PROMPT, MAIN_STEPS, MAIN_TOP_K, capture_hidden=True
    )
    greedy_text = tokenizer.decode([s["argmax"] for s in main_entry["decodeSteps"]])
    print(f"  tokens={main_entry['promptTokens']}  greedy={greedy_text!r}", flush=True)

    # -------- Sweep prompts (tokenizer + logits only) --------
    print(f"\n[sweep] {len(SWEEP_PROMPTS)} prompts × {SWEEP_STEPS} steps", flush=True)
    sweep_entries = []
    for i, p in enumerate(SWEEP_PROMPTS):
        entry = dump_prompt(
            model, tokenizer, p, SWEEP_STEPS, SWEEP_TOP_K, capture_hidden=False
        )
        sweep_entries.append(entry)
        greedy = tokenizer.decode([s["argmax"] for s in entry["decodeSteps"]])
        print(
            f"  [{i + 1:2d}/{len(SWEEP_PROMPTS)}] tok={entry['promptTokens']:3d}  "
            f"{p[:40]!r:<42}  → {greedy!r}",
            flush=True,
        )

    # -------- Long context (paged KV exercise) --------
    print(f"\n[longContext] {LONG_STEPS} steps", flush=True)
    long_entry = dump_prompt(
        model, tokenizer, LONG_CONTEXT_PROMPT, LONG_STEPS, LONG_TOP_K,
        capture_hidden=False,
    )
    long_greedy = tokenizer.decode([s["argmax"] for s in long_entry["decodeSteps"]])
    print(
        f"  tokens={long_entry['promptTokens']}  greedy={long_greedy!r}",
        flush=True,
    )

    # -------- Assemble + write --------
    payload = {
        "model": args.model,
        "dtype": args.dtype,
        "layersCaptured": LAYERS_TO_CAPTURE,
        "main": main_entry,
        "sweep": sweep_entries,
        "longContext": long_entry,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False))
    size_kb = args.out.stat().st_size / 1024
    print(f"\nWrote {args.out} ({size_kb:.1f} KB)")
    print(f"Copy to: cp {args.out} ../public/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# ruff: noqa: E501
# pyright: reportPrivateImportUsage=false
# (Pyright flags torch.randint / torch.full / torch.arange / torch.long as
#  "private" because of how PyTorch's __init__.py re-exports work, but they
#  are documented public API. Standard suppression for research scripts.)
"""
E45 / P-20260526-07 — Phase 0 small-model control.

Disentangles "Phi-3-mini-specific" from "general phenomenon" for the continuous-
attention self-consistency experiment. Trains a tiny transformer (4 layers,
64-hidden, 4-head, 256-vocab) to convergence on induction-heads + modular
arithmetic, then runs the same Picard fixed-point protocol that the
Neuropulse-side experiment runs on Phi-3-mini.

Mirrors the protocol in PREDICTIONS.md P-20260526-07:
  - RoPE applied once at iter=0; Q iterates POST-RoPE.
  - K, V computed once and held fixed (KV-cache analogue).
  - Per-layer Picard iteration of Q ← softmax(Q K^T / sqrt(d)) V.
  - SUB-STEP PROBE: per-attention FP, NOT DEQ-equivalent (per-block FP deferred).

Reads if both small-well-trained and Phi-3-mini show the same per-layer bucket
distribution → general phenomenon. If they diverge → Phi-3-mini idiosyncrasy
(weights-specific or scale-related).

See brain experiment E45 for the full pre-registration and symbol-collision
guards. Independent of the WebGPU codebase: pure PyTorch.

Usage:
    cd ~/neuropulse/tools/small-model-control
    python picard_disentangle.py --train --steps 5000
    python picard_disentangle.py --run

Output artifact: results/YYYY-MM-DD/E45-phase0-small-model.json
(shape matches RESEARCH_STANDARDS § 2 — meta, env, rows, status fields).

Reproducibility: torch + numpy seeded with SEED=42. Same hardware + same seed =
bit-exact rerun. Different hardware = statistically equivalent, not bit-exact.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Constants — match RESEARCH_STANDARDS § 1: single source of truth ───

VOCAB = 256
N_LAYER = 4
N_HEAD = 4
D_MODEL = 64
D_HEAD = D_MODEL // N_HEAD  # 16
SEQ_LEN = 32
SEED = 42

# Acceptance gate per PREDICTIONS.md P-20260526-07, residual_ratio_L:
#   Bucket A: ratio ≤ 2.0
#   Bucket B: 2.0 < ratio ≤ 50.0
#   Bucket C: ratio > 50.0 OR NaN OR non-convergence
BUCKET_A_MAX = 2.0
BUCKET_B_MAX = 50.0


# ─── Reproducibility ──────────────────────────────────────────────────

def seed_everything(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Synthetic tasks ──────────────────────────────────────────────────

# Vocab layout (token ids):
#   0-9   digits
#   10    '+'
#   11    '='
#   12    PAD
#   13-255 arbitrary symbol tokens (for induction-heads task)

PAD = 12
PLUS = 10
EQ = 11


def gen_modular_arithmetic(batch: int, mod: int = 7) -> torch.Tensor:
    """Tokens: [a, '+', b, '=', c, PAD, PAD, ...]. Predict c at position 4."""
    a = torch.randint(0, mod, (batch,))
    b = torch.randint(0, mod, (batch,))
    c = (a + b) % mod
    seq = torch.full((batch, SEQ_LEN), PAD, dtype=torch.long)
    seq[:, 0] = a
    seq[:, 1] = PLUS
    seq[:, 2] = b
    seq[:, 3] = EQ
    seq[:, 4] = c
    return seq


def gen_induction_heads(batch: int) -> torch.Tensor:
    """Sprinkle (A, B) bigrams at random positions. Predict B after A.

    The induction-head circuit must (i) attend to a prior A, (ii) copy the
    token after that A as the next prediction. Canonical mech-interp testbed.
    """
    seq = torch.full((batch, SEQ_LEN), PAD, dtype=torch.long)
    for i in range(batch):
        a = int(torch.randint(13, VOCAB, (1,)).item())
        b = int(torch.randint(13, VOCAB, (1,)).item())
        # 3 occurrences of (a, b) at distinct random positions
        positions = torch.randperm(SEQ_LEN - 2)[:3].tolist()
        for pos in positions:
            seq[i, pos] = a
            seq[i, pos + 1] = b
    return seq


# ─── Model ────────────────────────────────────────────────────────────

def rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary position embedding on the last dim of x ([B, S, H, D])."""
    *_, sl, _, dh = x.shape
    pos = torch.arange(sl, device=x.device, dtype=x.dtype).unsqueeze(-1)
    freq = 1.0 / (base ** (torch.arange(0, dh, 2, device=x.device, dtype=x.dtype) / dh))
    angles = pos * freq.unsqueeze(0)  # [S, D/2]
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(2)  # [1, S, 1, D/2]
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(2)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class RMSNorm(nn.Module):
    """Manual RMSNorm (avoids dependence on nn.RMSNorm which is recent)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


@dataclass
class AttnTelem:
    """Mirrors the WGSL telemetry struct in attention_fixedpoint.wgsl."""
    final_diff_inf: float
    iter_count: int
    init_max_score: float
    init_min_score: float
    init_entropy: float  # additional: avg softmax entropy at iter=0


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.o = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor, max_iter: int = 1) -> tuple[torch.Tensor, AttnTelem]:
        """Picard iteration of attention. If max_iter==1, identical to one-step."""
        B, S, _ = x.shape
        qkv = self.qkv(x).view(B, S, 3, N_HEAD, D_HEAD)
        # RoPE applied ONCE at iter=0 to Q and K. Q iterates POST-RoPE thereafter.
        q = rope(qkv[:, :, 0])  # [B, S, H, D]
        k = rope(qkv[:, :, 1])
        v = qkv[:, :, 2]

        # Causal mask
        mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        ).view(1, 1, S, S)

        # Track iter-0 raw-score range + entropy (predicted causal driver under H_agent)
        init_max_score = float("-inf")
        init_min_score = float("inf")
        init_entropy = 0.0

        q_iter = q
        q_prev = torch.zeros_like(q_iter)
        for it in range(max_iter):
            q_prev = q_iter
            # scores: [B, H, S, S] = Q_iter · K^T / sqrt(d)
            scores = torch.einsum("bshd,bthd->bhst", q_iter, k) / math.sqrt(D_HEAD)
            scores = scores.masked_fill(mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)  # [B, H, S, S]

            if it == 0:
                # Telemetry only — does not affect math
                with torch.no_grad():
                    valid_scores = scores.masked_fill(mask, float("nan"))
                    init_max_score = torch.nan_to_num(valid_scores, nan=float("-inf")).max().item()
                    init_min_score = torch.nan_to_num(valid_scores, nan=float("inf")).min().item()
                    # Entropy averaged across (B, H, S) using last-position weights as anchor
                    w_lastpos = weights[:, :, -1, :]  # [B, H, S]
                    init_entropy = (-w_lastpos * (w_lastpos + 1e-12).log()).sum(-1).mean().item()

            # Picard update: Q ← softmax(Q K^T / sqrt(d)) V
            out = torch.einsum("bhst,bthd->bshd", weights, v)
            q_iter = out

        # Convergence delta at the final iter (||Q_t - Q_{t-1}||_inf)
        final_diff = (q_iter - q_prev).abs().max().item()

        return self.o(q_iter.reshape(B, S, D_MODEL)), AttnTelem(
            final_diff_inf=final_diff,
            iter_count=max_iter,
            init_max_score=init_max_score,
            init_min_score=init_min_score,
            init_entropy=init_entropy,
        )


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(D_MODEL)
        self.attn = Attention()
        self.norm2 = RMSNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL, bias=False),
            nn.SiLU(),
            nn.Linear(4 * D_MODEL, D_MODEL, bias=False),
        )

    def forward(self, x: torch.Tensor, max_iter: int = 1) -> tuple[torch.Tensor, AttnTelem]:
        a, telem = self.attn(self.norm1(x), max_iter=max_iter)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x, telem


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        max_iter: int = 1,
        capture_hidden: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        h = self.embed(x)
        hiddens = [h.clone()] if capture_hidden else None
        telems: list[AttnTelem] = []
        for block in self.blocks:
            h, t = block(h, max_iter=max_iter)
            telems.append(t)
            if capture_hidden:
                cast(list, hiddens).append(h.clone())
        h = self.norm(h)
        if capture_hidden:
            cast(list, hiddens).append(h.clone())
        logits = self.head(h)
        return logits, {"hiddens": hiddens, "telems": telems}


# ─── Training ─────────────────────────────────────────────────────────

def train_model(model: TinyTransformer, n_steps: int = 5000) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses: list[float] = []
    for step in range(n_steps):
        # Alternate tasks
        if step % 2 == 0:
            batch = gen_modular_arithmetic(64)
            tgt_pos = 4  # position of c in [a, +, b, =, c, ...]
            pred_pos = tgt_pos - 1  # logits at index 3 predict token at index 4
        else:
            batch = gen_induction_heads(64)
            # Pick a position where a→b appears (induction-head test).
            # Easiest signal: predict tokens at all positions and use mean CE.
            tgt_pos = -1
            pred_pos = -1

        logits, _ = model(batch, max_iter=1)
        if tgt_pos == -1:
            # Whole-sequence shift-by-one CE
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB),
                batch[:, 1:].reshape(-1),
            )
        else:
            loss = F.cross_entropy(
                logits[:, pred_pos].reshape(-1, VOCAB),
                batch[:, tgt_pos].reshape(-1),
            )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == n_steps - 1:
            losses.append(float(loss.item()))
            print(f"step {step:5d}: loss={loss.item():.4f}")
    return losses


# ─── Picard sweep + bucket gate ────────────────────────────────────────

def bucket(ratio: float) -> str:
    if not math.isfinite(ratio):
        return "C"
    if ratio <= BUCKET_A_MAX:
        return "A"
    if ratio <= BUCKET_B_MAX:
        return "B"
    return "C"


def run_picard_sweep(model: TinyTransformer, save_path: Path) -> dict:
    model.eval()

    # Eval set: 8 modular-arithmetic + 8 induction-head sequences (deterministic seed)
    seed_everything(SEED + 1)  # different seed for eval so it's not training-set
    eval_seqs = torch.cat([
        gen_modular_arithmetic(8),
        gen_induction_heads(8),
    ])

    # Reference: one-step forward pass (the baseline that Phi-3-mini was trained for)
    with torch.no_grad():
        ref_logits, ref_info = model(eval_seqs, max_iter=1, capture_hidden=True)
    ref_hiddens = ref_info["hiddens"]
    ref_pred = ref_logits.argmax(-1)

    # Picard sweep
    sweep: list[dict] = []
    for max_iter in [1, 2, 5, 10, 50, 100]:
        with torch.no_grad():
            fp_logits, fp_info = model(eval_seqs, max_iter=max_iter, capture_hidden=True)
        fp_hiddens = fp_info["hiddens"]
        fp_pred = fp_logits.argmax(-1)

        # Per-layer L2 diff between fp and reference hidden states.
        # "Residual ratio" per PREDICTIONS.md threshold = fp_diff_to_ref / ref_l2.
        # (The Phi-3 protocol uses fp_diff_to_HF / one-step_diff_to_HF; here we
        # don't have an external HF reference, so we use ref_l2 as the
        # normalizer. Both isolate fixed-point divergence from baseline scale.)
        per_layer: list[dict] = []
        for L, telem in enumerate(fp_info["telems"]):
            # Layer L's output is at hidden index L+1 (index 0 = embedding)
            h_fp = fp_hiddens[L + 1]
            h_ref = ref_hiddens[L + 1]
            l2_diff = (h_fp - h_ref).norm().item()
            ref_l2 = max(h_ref.norm().item(), 1e-9)
            ratio = l2_diff / ref_l2
            per_layer.append({
                "layer": L,
                "l2_diff": l2_diff,
                "ref_l2": ref_l2,
                "residual_ratio": ratio,
                "bucket": bucket(ratio),
                "entropy_iter0": telem.init_entropy,
                "final_diff_inf": telem.final_diff_inf,
                "init_max_score": telem.init_max_score,
                "init_min_score": telem.init_min_score,
            })

        top1_match = (fp_pred == ref_pred).float().mean().item()

        # Aggregate
        bucket_counts: dict[str, int] = {}
        for pl in per_layer:
            bucket_counts[pl["bucket"]] = bucket_counts.get(pl["bucket"], 0) + 1

        sweep.append({
            "max_iter": max_iter,
            "top1_match_rate": top1_match,
            "bucket_counts": bucket_counts,
            "per_layer": per_layer,
        })

        print(
            f"max_iter={max_iter:3d}: top1_match={top1_match:.3f}, "
            f"buckets={bucket_counts}, "
            f"L0_ratio={per_layer[0]['residual_ratio']:.3e}, "
            f"L_last_ratio={per_layer[-1]['residual_ratio']:.3e}"
        )

    # JSON artifact per RESEARCH_STANDARDS § 2 schema
    artifact = {
        "meta": {
            "protocol": "E45 / P-20260526-07 phase-0 small-model control",
            "hypothesis": (
                "If small well-trained transformer converges Bucket A across all layers "
                "but Phi-3-mini does not (per neuropulse-side run), divergence is "
                "scale-related or training-objective-related, NOT a general phenomenon."
            ),
            "passBar": "informational — disentanglement control, no pass/fail on its own",
            "seed": SEED,
            "warmup": 0,
            "trials": 1,
            "model_config": {
                "n_layer": N_LAYER,
                "d_model": D_MODEL,
                "n_head": N_HEAD,
                "vocab": VOCAB,
                "seq_len": SEQ_LEN,
            },
        },
        "env": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "platform": platform.platform(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "rows": sweep,
        "status": "informational",
        "diagnosis": (
            "Compare bucket distribution at max_iter=100 against the Phi-3-mini "
            "run's per-layer bucket histogram. Concordant patterns → general "
            "phenomenon. Divergent patterns → Phi-3-mini idiosyncrasy."
        ),
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nSaved: {save_path}")
    return artifact


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="E45 Phase 0 small-model control")
    p.add_argument("--train", action="store_true", help="Train the small model")
    p.add_argument("--run", action="store_true", help="Run Picard sweep, write JSON")
    p.add_argument("--steps", type=int, default=5000, help="Training steps")
    p.add_argument("--ckpt", type=str, default="small_model.pt", help="Checkpoint path")
    args = p.parse_args()

    if not args.train and not args.run:
        p.error("specify --train and/or --run")

    seed_everything(SEED)
    model = TinyTransformer()

    ckpt_path = Path(args.ckpt)

    if args.train:
        print(f"Training for {args.steps} steps...")
        train_model(model, n_steps=args.steps)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    if args.run:
        if not args.train:
            if not ckpt_path.exists():
                p.error(f"checkpoint not found: {ckpt_path} (run with --train first)")
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = Path(f"results/{date_str}/E45-phase0-small-model.json")
        run_picard_sweep(model, out_path)


if __name__ == "__main__":
    main()

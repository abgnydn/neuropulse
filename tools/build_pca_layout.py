"""
Build a PCA-based 2D layout for Neural Pulse's residual + FFN slabs.

Default behavior (FAST, no inference): for each of the 3072 residual dims and
each of the 8192 FFN-mid neurons in layer 0, use the corresponding column/row
of the relevant weight matrix as that unit's "embedding", then PCA→2D.

  - residual_dim[i]   ← qkv_proj.weight[:, i]    (how that residual dim is read
                                                  into attention queries+keys+vals)
  - ffn_mid_neuron[j] ← down_proj.weight[:, j]   (how that FFN neuron writes back
                                                  into the residual stream)

The output JSON contains coords in [-0.5, 0.5] × [-0.5, 0.5]:
  {
    "residual": [[x, y], ... 3072 entries],
    "ffn":      [[x, y], ... 8192 entries],
    "model":    "microsoft/Phi-3-mini-4k-instruct",
    "layer":    0
  }

Usage
-----
    pip install -r requirements.txt
    python build_pca_layout.py
    cp pca-layout.json ../public/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fit_pca_2d(matrix) -> "list[list[float]]":
    """matrix shape: (n_units, n_features). Returns [[x, y], ...] normalized to [-0.5, 0.5]."""
    import numpy as np
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, svd_solver="auto", random_state=0)
    coords = pca.fit_transform(matrix.astype(np.float32))

    # Normalize per axis to [-0.5, 0.5]
    for axis in range(2):
        col = coords[:, axis]
        lo, hi = float(col.min()), float(col.max())
        if hi - lo > 1e-9:
            coords[:, axis] = (col - lo) / (hi - lo) - 0.5
        else:
            coords[:, axis] = 0.0

    print(f"  PCA explained variance: {pca.explained_variance_ratio_}", flush=True)
    return coords.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="microsoft/Phi-3-mini-4k-instruct"
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("pca-layout.json"))
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    print(f"Loading {args.model} ...", flush=True)
    # Use the built-in transformers Phi3 implementation (the vendored
    # modeling_phi3.py from the HF repo is incompatible with transformers ≥5.x).
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    layer = model.model.layers[args.layer]

    # Phi-3's layer structure (transformers ≥ 4.41):
    #   layer.self_attn.qkv_proj.weight     shape (9216, 3072)  — qkv as one matrix
    #   layer.mlp.gate_up_proj.weight        shape (16384, 3072) — gate+up fused
    #   layer.mlp.down_proj.weight           shape (3072, 8192)  — FFN-mid → residual
    qkv_w = layer.self_attn.qkv_proj.weight.detach().cpu().numpy()       # (9216, 3072)
    down_w = layer.mlp.down_proj.weight.detach().cpu().numpy()            # (3072, 8192)

    print(f"qkv_proj shape:  {qkv_w.shape}")
    print(f"down_proj shape: {down_w.shape}")

    # Residual dims as columns of qkv_proj → each residual dim has a 9216-d signature
    residual_emb = qkv_w.T  # (3072, 9216)
    print("Fitting PCA(2) over 3072 residual dims ...")
    residual_2d = fit_pca_2d(residual_emb)

    # FFN-mid neurons as columns of down_proj → each FFN neuron has a 3072-d signature
    ffn_emb = down_w.T  # (8192, 3072)
    print("Fitting PCA(2) over 8192 FFN-mid neurons ...")
    ffn_2d = fit_pca_2d(ffn_emb)

    payload = {
        "model": args.model,
        "layer": args.layer,
        "residual": residual_2d,  # 3072 × 2
        "ffn": ffn_2d,            # 8192 × 2
    }
    args.out.write_text(json.dumps(payload))
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"\nWrote {args.out} ({size_mb:.2f} MB)")
    print(f"Copy to: cp {args.out} ../public/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

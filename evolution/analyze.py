"""Compute checkpoint overlap matrices and distance-averaged MLP Jaccard."""
import argparse
import itertools
import json
import re
from pathlib import Path

import torch


def load_support(path, threshold):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.detach().cpu().reshape(-1) > threshold
            for k, v in sorted(raw.items()) if ".mlp." in k}


def overlap(a, b):
    if not a or a.keys() != b.keys():
        raise ValueError("Masks must contain the same nonempty set of MLP modules")
    intersection = union = 0
    for key in a:
        if a[key].shape != b[key].shape:
            raise ValueError(f"Mask shape mismatch: {key}")
        intersection += int((a[key] & b[key]).sum())
        union += int((a[key] | b[key]).sum())
    return intersection / union if union else 1.0


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--mask-pattern", default="full_ckpt{step}_free/best_safety_masks.pt")
    ap.add_argument("--first-mask", default="full_ckpt1/best_safety_masks.pt")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--stop", type=int, default=30)
    ap.add_argument("--settled-start", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold on saved raw logits")
    ap.add_argument("--independent-glob", default="", help="Optional coldSTEP_sSEED files or directories")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.start > args.stop:
        ap.error("--start must not exceed --stop")
    root = Path(args.mask_dir)
    masks = {step: load_support(root / (args.first_mask if step == 1 else
             args.mask_pattern.format(step=step)), args.threshold)
             for step in range(args.start, args.stop + 1)}
    steps = list(masks)
    matrix = [[overlap(masks[a], masks[b]) for b in steps] for a in steps]
    offsets = {}
    for distance in range(1, args.stop - max(args.start, args.settled_start) + 1):
        values = [overlap(masks[s], masks[s + distance]) for s in steps
                  if s >= args.settled_start and s + distance in masks]
        offsets[str(distance)] = {"mean": mean(values), "n_pairs": len(values)}
    result = {"threshold": args.threshold, "steps": steps,
              "active_rows": {str(s): sum(int(v.sum()) for v in m.values()) for s, m in masks.items()},
              "jaccard_matrix": matrix, "settled_start": args.settled_start,
              "offsets": offsets}
    independent = []
    if args.independent_glob:
        for path in sorted(root.glob(args.independent_glob)):
            match = re.search(r"cold(\d+)_s(\d+)", str(path.relative_to(root)))
            if not match:
                raise ValueError(f"Expected coldSTEP_sSEED in {path}")
            file = path / "best_safety_masks.pt" if path.is_dir() else path
            independent.append((int(match[1]), int(match[2]), load_support(file, args.threshold)))
        if not independent:
            raise ValueError("No independent masks matched")
        pairs = list(itertools.combinations(independent, 2))
        same = [overlap(a[2], b[2]) for a, b in pairs if a[0] == b[0]]
        adjacent = [overlap(a[2], b[2]) for a, b in pairs
                    if abs(a[0] - b[0]) == 1 and a[1] == b[1]]
        result["independent"] = {"same_checkpoint_mean": mean(same), "same_checkpoint_pairs": len(same),
                                 "adjacent_checkpoint_mean": mean(adjacent), "adjacent_checkpoint_pairs": len(adjacent),
                                 "adjacent_pairing": "matching extraction seed"}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"offsets": {k: v for k, v in offsets.items() if k in ("1", "5", "10", "20")},
                      "independent": result.get("independent")}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recall_at_precision.py
  - Compute recall@target_precision (and the corresponding threshold if available)
    from a PR curve CSV.

Accepted input columns (case-insensitive):
  * Preferred: precision, recall[, threshold]
  * Or:       tp, fp, fn[, threshold]  (precision/recall are derived)

Outputs (per target P):
  - recall_at_precision_step:     max recall among points with precision >= P
  - threshold_at_precision_step:  threshold at that point (if threshold column exists)
  - recall_at_precision_envelope: using precision envelope p_env(r)=max_{r' >= r} p(r')
  - threshold_at_precision_envelope: threshold that attains the envelope precision
                                     at the chosen recall (if available)

Usage:
  python recall_at_precision.py --pr pr_curve.csv --targets 0.90 0.95 \
      --mode both --out metrics_recall_at_p.json
"""
import argparse
import json
import os
import numpy as np
import pandas as pd


def load_pr(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    has_thr = "threshold" in cols
    thr = df[cols["threshold"]].astype(float).values if has_thr else None

    if "precision" in cols and "recall" in cols:
        p = df[cols["precision"]].astype(float).values
        r = df[cols["recall"]].astype(float).values
    else:
        need = [k for k in ("tp", "fp", "fn") if k not in cols]
        if need:
            raise ValueError(
                f"{path} must contain either (precision,recall) or (tp,fp,fn). Missing: {need}"
            )
        tp = df[cols["tp"]].astype(float).values
        fp = df[cols["fp"]].astype(float).values
        fn = df[cols["fn"]].astype(float).values
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
            r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)

    # filter NaN
    keep = ~np.isnan(p) & ~np.isnan(r)
    p, r = p[keep], r[keep]
    if has_thr:
        thr = thr[keep]

    # sort by recall asc, precision desc (stable tie-break)
    order = np.lexsort((-p, r))
    p, r = p[order], r[order]
    if has_thr:
        thr = thr[order]

    # clamp
    p = np.clip(p, 0.0, 1.0)
    r = np.clip(r, 0.0, 1.0)

    data = {"precision": p, "recall": r}
    if has_thr:
        data["threshold"] = thr
    return pd.DataFrame(data)


def recall_threshold_at_precision_step(df: pd.DataFrame, target: float):
    """Stepwise: take the point with max recall among precision >= target."""
    p = df["precision"].values
    r = df["recall"].values
    mask = p >= target
    if not mask.any():
        return 0.0, None
    # pick the row with the largest recall (if multiple, take the one with highest precision)
    cand_idx = np.where(mask)[0]
    best_r = r[cand_idx].max()
    cand2 = cand_idx[r[cand_idx] == best_r]
    # tie-break by precision desc (optional)
    i = cand2[np.argmax(p[cand2])]
    thr = df["threshold"].values[i] if "threshold" in df.columns else None
    return float(r[i]), (None if thr is None else float(thr))


def recall_threshold_at_precision_envelope(df: pd.DataFrame, target: float):
    """
    Envelope precision p_env(r) = max_{k >= i} p[k], scanning from high recall to low.
    Return the largest recall i whose envelope precision >= target,
    and the threshold that achieves the envelope precision at that i (if available).
    """
    p = df["precision"].values
    r = df["recall"].values
    has_thr = "threshold" in df.columns
    thr = df["threshold"].values if has_thr else None

    # suffix max of precision, plus the argmax index to recover threshold
    # env_prec[i] = max p[i:], env_arg[i] = arg of that max within [i..end)
    n = len(p)
    env_prec = np.empty(n, dtype=float)
    env_arg = np.empty(n, dtype=int)
    best_p = -1.0
    best_j = -1
    for i in range(n - 1, -1, -1):
        if p[i] >= best_p:
            best_p = p[i]
            best_j = i
        env_prec[i] = best_p
        env_arg[i] = best_j

    mask = env_prec >= target
    if not mask.any():
        return 0.0, None

    # choose the largest recall index satisfying the constraint
    idxs = np.where(mask)[0]
    # largest recall -> pick the idx with max r
    i = idxs[np.argmax(r[idxs])]
    j = env_arg[i]  # index that attains the envelope precision at or after i
    th = (None if not has_thr else float(thr[j]))
    return float(r[i]), th


def main():
    ap = argparse.ArgumentParser(description="Compute recall@precision (and threshold) from PR curve CSV")
    ap.add_argument("--pr", required=True, help="PR curve CSV (precision,recall[,threshold]) or (tp,fp,fn[,threshold])")
    ap.add_argument("--targets", nargs="+", type=float, required=True, help="target precision values (e.g., 0.90 0.95)")
    ap.add_argument("--mode", choices=["step", "envelope", "both"], default="both")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    df = load_pr(args.pr)
    results = {"file": args.pr, "has_threshold": ("threshold" in df.columns), "results": []}

    for t in args.targets:
        t = float(np.clip(t, 0.0, 1.0))
        entry = {"target_precision": t}

        if args.mode in ("step", "both"):
            rec_s, thr_s = recall_threshold_at_precision_step(df, t)
            entry["recall_at_precision_step"] = rec_s
            entry["threshold_at_precision_step"] = thr_s

        if args.mode in ("envelope", "both"):
            rec_e, thr_e = recall_threshold_at_precision_envelope(df, t)
            entry["recall_at_precision_envelope"] = rec_e
            entry["threshold_at_precision_envelope"] = thr_e

        results["results"].append(entry)

    print(json.dumps(results, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
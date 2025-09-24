#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a fine-tuned CodeBERT model on MySQL clone pairs listed in JSONL/JSON.

Input (Judged_all.jsonl or JSON list):
  {"u": "<pathA>[::Func]@[start]-[end]", "v": "<pathB>[::Func]@[start]-[end]",
   "verdict": "EQUIV|CLOSE|DIFF|...", ...}

- Positives: verdict in {"EQUIV","CLOSE"}
- Negatives: otherwise
- Paths may be absolute or relative. If relative, prepend --base_dir.
- If "@start-end" is missing, the whole file is used (1 file = 1 function assumption).

Outputs:
  out_dir/results.json  with PR-AUC, best-F1 threshold (τ_F1), and fixed-τ metrics.

Usage examples:
  # cross-encoder (recommended)
  python eval_mysql_jsonl.py \
    --jsonl Judged_all.jsonl \
    --mode crossencoder \
    --model_id microsoft/codebert-base \
    --model_pt out_scb_cross/clf_best.pt \
    --tau 0.5310 \
    --base_dir /home/azureuser/HySCU/create_dataset/out \
    --max_len 512 --batch_size 64 --fp16 \
    --out_dir eval_mysql_cross

  # bi-encoder
  python eval_mysql_jsonl.py \
    --jsonl Judged_all.jsonl \
    --mode biencoder \
    --model_id microsoft/codebert-base \
    --model_pt out_bi/encoder_best.pt \
    --tau 0.6500 \
    --base_dir /home/azureuser/HySCU/create_dataset/out \
    --max_len 256 --batch_size 512 --fp16 \
    --out_dir eval_mysql_bi
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


# ------------------- parsing & IO -------------------

def load_jsonl_or_json(path: str) -> List[dict]:
    """Load JSONL or JSON(list) and return list of dicts."""
    data: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    return data


def parse_ref(ref: str, base_dir: str = "") -> Tuple[Path, int, int]:
    """
    Parse "path::Func@start-end" or "path::Func" or "path" into (abs_path, start, end).
    - If start-end missing: return full file range.
    - If path is relative: prepend base_dir (must be given) and resolve.
    """
    # First try absolute path with range
    m = re.search(r'(/[^:]+?)::[^@]+@(\d+)-(\d+)$', ref)
    if m:
        path = Path(m.group(1)).resolve()
        start = int(m.group(2))
        end = int(m.group(3))
        return path, start, end

    # Generic: split "path::..." (ignore func), accept "path" alone too
    path_str = ref.split("::", 1)[0]
    p = Path(path_str)
    if not p.is_absolute():
        if not base_dir:
            raise ValueError(f"Relative path encountered but --base_dir is not set: {ref}")
        p = (Path(base_dir) / p)
    p = p.resolve()

    # Try to read file to determine EOF
    n_lines = 1
    try:
        n_lines = len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
    except Exception:
        for enc in ("cp932", "latin1"):
            try:
                n_lines = len(p.read_text(encoding=enc, errors="ignore").splitlines())
                break
            except Exception:
                continue
    return p, 1, n_lines


def slice_code(path: Path, start: int, end: int) -> str:
    """Return code slice [start, end] (1-based, inclusive). If read fails, return empty string."""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        lines = None
        for enc in ("cp932", "latin1"):
            try:
                lines = path.read_text(encoding=enc, errors="ignore").splitlines()
                break
            except Exception:
                continue
        if lines is None:
            return ""
    start = max(1, start)
    end = min(len(lines), end)
    if start > end:
        return ""
    return "\n".join(lines[start - 1:end])


# ------------------- metrics -------------------

def compute_pr(labels: np.ndarray, scores: np.ndarray):
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((labels == 1).sum()), 1)
    thresholds = scores[order]
    return precision, recall, thresholds


def pr_auc(precision: np.ndarray, recall: np.ndarray) -> float:
    if len(precision) < 2:
        return 0.0
    return float(np.trapz(precision, recall))


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> Dict:
    """Return dict with tau_f1, f1_max, precision, recall, tp/fp/fn at τ_F1."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    s_sorted = s[order]
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted == 1)
    pos_total = int((y == 1).sum())
    change_idx = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])

    best = {"tau_f1": float("nan"), "f1_max": -1.0, "precision": 0.0, "recall": 0.0,
            "tp": 0, "fp": 0, "fn": pos_total}
    for idx in change_idx:
        tp = int(tp_cum[idx])
        pred_pos = idx + 1
        fp = pred_pos - tp
        fn = pos_total - tp
        P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        R = tp / pos_total if pos_total > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        if F1 > best["f1_max"]:
            best.update({"tau_f1": float(s_sorted[idx]), "f1_max": F1,
                         "precision": P, "recall": R, "tp": tp, "fp": fp, "fn": fn})
    return best


def eval_at_tau(labels: np.ndarray, scores: np.ndarray, tau: float) -> Dict:
    """Return P/R/F1 and confusion at fixed τ."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    yhat = (s >= tau).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    return {"precision": P, "recall": R, "f1": F1, "tp": tp, "fp": fp, "fn": fn}


# ------------------- scoring -------------------

@torch.no_grad()
def embed_codes(codes: List[str], tok, enc, device, max_len=256, batch_size=256, amp_dtype=None):
    """Embed a list of code snippets with bi-encoder (mean-pooling + L2)."""
    embs = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        x = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
            out = enc(**x).last_hidden_state
            m = (out * x["attention_mask"].unsqueeze(-1)).sum(1) / x["attention_mask"].sum(1, keepdim=True).clamp_min(1e-9)
            m = F.normalize(m, dim=-1)
        embs.append(m.cpu())
    return torch.cat(embs, dim=0)  # [N, D]


@torch.no_grad()
def score_pairs_biencoder(pairs, id2idx: Dict[str, int], embs: torch.Tensor) -> np.ndarray:
    """Vectorized cosine similarity for bi-encoder pairs."""
    idx_a = np.fromiter((id2idx[ia] for (ia, _, ib, _, _) in pairs), dtype=np.int64)
    idx_b = np.fromiter((id2idx[ib] for (_, _, ib, _, _) in pairs), dtype=np.int64)
    va = embs.index_select(0, torch.from_numpy(idx_a))
    vb = embs.index_select(0, torch.from_numpy(idx_b))
    return (va * vb).sum(-1).numpy().astype(np.float32)


@torch.no_grad()
def score_pairs_crossencoder(pairs, tok, clf, device, max_len=512, batch_size=64, amp_dtype=None) -> np.ndarray:
    """Batched scoring for cross-encoder (probability of positive)."""
    scores = np.empty(len(pairs), dtype=np.float32)
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        texts = [ca + tok.sep_token + cb for (_, ca, _, cb, _) in batch]
        x = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
            logits = clf(**x).logits
            prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        scores[i:i + len(batch)] = prob.astype(np.float32)
    return scores


# ------------------- main -------------------

def _json_default(o):
    """Safe JSON conversion for numpy types."""
    import numpy as _np
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Judged_all.jsonl (or JSON list)")
    ap.add_argument("--mode", choices=["biencoder", "crossencoder"], required=True)
    ap.add_argument("--model_id", default="microsoft/codebert-base")
    ap.add_argument("--model_pt", required=True, help="encoder_best.pt or clf_best.pt")
    ap.add_argument("--tau", type=float, required=True, help="Fixed threshold to apply")
    ap.add_argument("--base_dir", default="", help="Prepend to relative paths in u/v")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out_dir", default="eval_mysql_out")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda":
        if args.fp16:
            amp_dtype = torch.float16
        elif args.bf16:
            amp_dtype = torch.bfloat16

    # 1) Load pairs & extract code
    items = load_jsonl_or_json(args.jsonl)
    pairs = []  # (id_u, code_u, id_v, code_v, label)
    n_skip = 0
    for obj in items:
        verdict = (obj.get("verdict", "") or "").upper()
        label = 1 if verdict in ("EQUIV", "CLOSE") else 0
        try:
            p_u, s_u, e_u = parse_ref(obj["u"], base_dir=args.base_dir)
            p_v, s_v, e_v = parse_ref(obj["v"], base_dir=args.base_dir)
            code_u = slice_code(p_u, s_u, e_u)
            code_v = slice_code(p_v, s_v, e_v)
            if not code_u or not code_v:
                n_skip += 1
                continue
            pairs.append((str(p_u), code_u, str(p_v), code_v, label))
        except Exception:
            n_skip += 1
            continue

    if not pairs:
        print("[ERROR] No valid pairs parsed from input. Abort.", file=sys.stderr)
        raise SystemExit(2)

    labels = np.fromiter((l for (*_, l) in pairs), dtype=np.int8)

    # 2) Load model
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if args.mode == "biencoder":
        enc = AutoModel.from_pretrained(args.model_id).to(device).eval()
        enc.load_state_dict(torch.load(args.model_pt, map_location=device))
        clf = None
    else:
        clf = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=2).to(device).eval()
        clf.load_state_dict(torch.load(args.model_pt, map_location=device))
        enc = None

    # 3) Score
    if args.mode == "biencoder":
        # unique codes → embed once
        ids = sorted({ia for (ia, _, ib, _, _) in pairs} | {ib for (_, _, ib, _, _) in pairs})
        id2idx = {fid: i for i, fid in enumerate(ids)}
        table: Dict[str, str] = {}
        for (ia, ca, ib, cb, _) in pairs:
            table[ia] = ca
            table[ib] = cb
        codes = [table[fid] for fid in ids]
        with torch.inference_mode():
            embs = embed_codes(codes, tok, enc, device,
                               max_len=args.max_len, batch_size=args.batch_size,
                               amp_dtype=amp_dtype)
        scores = score_pairs_biencoder(pairs, id2idx, embs)
    else:
        with torch.inference_mode():
            scores = score_pairs_crossencoder(
                pairs, tok, clf, device,
                max_len=args.max_len, batch_size=args.batch_size,
                amp_dtype=amp_dtype
            )

    # 4) Metrics
    P, R, _ = compute_pr(labels, scores)
    ap = pr_auc(P, R)
    best = best_f1_threshold(labels, scores)
    fixed = eval_at_tau(labels, scores, tau=args.tau)

    # 5) Output
    print(f"[INFO] pairs={len(pairs)}  skipped={n_skip}")
    print(f"PR-AUC={ap:.4f}")
    print(f"Best-F1: τ_F1={best['tau_f1']:.6f}  F1={best['f1_max']:.4f}  "
          f"P={best['precision']:.3f}  R={best['recall']:.3f}  TP={best['tp']} FP={best['fp']} FN={best['fn']}")
    print(f"Fixed-τ (τ={args.tau:.6f}): F1={fixed['f1']:.4f}  P={fixed['precision']:.3f}  R={fixed['recall']:.3f}  "
          f"TP={fixed['tp']} FP={fixed['fp']} FN={fixed['fn']}")

    out = {
        "n_pairs": int(len(pairs)),
        "n_skipped": int(n_skip),
        "mode": args.mode,
        "model_id": args.model_id,
        "model_pt": args.model_pt,
        "tau_fixed": float(args.tau),
        "pr_auc": float(ap),
        "tau_f1": float(best["tau_f1"]),
        "f1_max": float(best["f1_max"]),
        "precision_at_tau_f1": float(best["precision"]),
        "recall_at_tau_f1": float(best["recall"]),
        "tp_at_tau_f1": int(best["tp"]),
        "fp_at_tau_f1": int(best["fp"]),
        "fn_at_tau_f1": int(best["fn"]),
        "precision_at_tau_fixed": float(fixed["precision"]),
        "recall_at_tau_fixed": float(fixed["recall"]),
        "f1_at_tau_fixed": float(fixed["f1"]),
        "tp_at_tau_fixed": int(fixed["tp"]),
        "fp_at_tau_fixed": int(fixed["fp"]),
        "fn_at_tau_fixed": int(fixed["fn"]),
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=_json_default)


if __name__ == "__main__":
    main()
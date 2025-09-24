#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a fine-tuned CodeBERT model (bi-encoder or cross-encoder)
on folder-structured datasets:

  create_dataset/dataset/<LANG>/eval/<ALGO>/impl_<n>.<ext>

- Same <ALGO> folder -> positive (label=1)
- Different <ALGO> folders -> negative (label=0), sampled via --neg-per-pos
- Per language, report best-F1 threshold (τ_F1), F1_max and PR-AUC.
- Save PR curve as CSV per language (out_dir/pr_curve_<mode>_<LANG>.csv)
- NEW: Recall@P（既定 P=0.90）を算出・保存

Usage:
  python eval_codebert_folder.py --root create_dataset/dataset \
      --langs C CPP Java Python \
      --mode crossencoder --model_id microsoft/codebert-base \
      --model_pt out_scb_cross/clf_best.pt \
      --neg-per-pos 2 --out_dir eval_out_cross

  python eval_codebert_folder.py --root create_dataset/dataset \
      --mode biencoder --model_id microsoft/codebert-base \
      --model_pt out_bi/encoder_best.pt --out_dir eval_out_bi
"""

import os, re, sys, json, argparse, random, csv
from pathlib import Path
from itertools import combinations
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# --------------- metrics ---------------

def compute_pr(labels: np.ndarray, scores: np.ndarray):
    """Return precision, recall, thresholds for a descending sweep."""
    order = np.argsort(-scores)
    y = labels[order]
    s = scores[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((labels == 1).sum()), 1)
    thresholds = s
    return precision, recall, thresholds

def pr_auc(precision: np.ndarray, recall: np.ndarray) -> float:
    if len(precision) < 2:
        return 0.0
    return float(np.trapz(precision, recall))

def best_f1_threshold(labels: np.ndarray, scores: np.ndarray):
    """Return dict with tau_f1, f1_max, precision, recall, tp/fp/fn at τ_F1."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    s_sorted = s[order]; y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    pos_total = int((y == 1).sum())
    change_idx = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    best = {"tau_f1": float("nan"), "f1_max": -1.0, "precision": 0.0, "recall": 0.0,
            "tp": 0, "fp": 0, "fn": pos_total}
    for idx in change_idx:
        tp = int(tp_cum[idx])
        fp = int(fp_cum[idx])
        fn = pos_total - tp
        P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        R = tp / pos_total  if pos_total > 0 else 0.0
        F1 = 2*P*R / (P+R)  if (P+R) > 0 else 0.0
        if F1 > best["f1_max"]:
            best.update({"tau_f1": float(s_sorted[idx]), "f1_max": F1,
                         "precision": P, "recall": R, "tp": tp, "fp": fp, "fn": fn})
    return best

def save_pr_curve_csv(labels: np.ndarray, scores: np.ndarray, out_csv: Path):
    """
    Save PR curve CSV with columns:
      threshold,tp,fp,fn,precision,recall,f1
    しきい値が変わるブレークポイントだけを書き出します。
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    s_sorted = s[order]
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    pos_total = int((y == 1).sum())

    change_idx = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "tp", "fp", "fn", "precision", "recall", "f1"])
        for idx in change_idx:
            tp = int(tp_cum[idx])
            fp = int(fp_cum[idx])
            fn = pos_total - tp
            P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            R = tp / pos_total  if pos_total > 0 else 0.0
            F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
            thr = float(s_sorted[idx])
            w.writerow([
                f"{thr:.10f}", tp, fp, fn,
                f"{P:.10f}", f"{R:.10f}", f"{F1:.10f}"
            ])

def recall_at_precision(labels: np.ndarray, scores: np.ndarray, target_p: float):
    """
    PR を降順に掃引し、precision>=target_p を初めて満たす最小しきい値の
    recall/threshold と混同行列を返す。達成不能なら None を返す。
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    s_sorted = s[order]; y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    pos_total = int((y == 1).sum())

    P, R, _ = compute_pr(labels, scores)
    idxs = np.where(P >= target_p)[0]
    if len(idxs) == 0:
        return None  # 到達不可
    i = int(idxs[0])
    # s_sorted と P/R の並びは同じ（同じ order）
    tp = int(tp_cum[i])
    fp = int(fp_cum[i])
    fn = pos_total - tp
    return {
        "threshold_at_p": float(s_sorted[i]),
        "recall_at_p":    float(R[i]),
        "precision_at_p": float(P[i]),
        "tp_at_p": tp, "fp_at_p": fp, "fn_at_p": fn
    }

# --------------- data loading ---------------

EXTS = [".c",".cpp",".cc",".java",".py",".go",".rs",".cs",".js",".ts",".hpp",".h",".C",".CPP"]

def read_text(p: Path) -> str:
    for enc in ("utf-8","cp932","latin1"):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            continue
    return p.read_bytes().decode("utf-8","ignore")

def list_lang_items(lang_root: Path) -> Dict[str, List[Tuple[str,str]]]:
    """
    Return dict: algo -> list of (file_id, code)
    file_id is relative path string; code is file content.
    """
    eval_dir = lang_root / "eval"
    if not eval_dir.exists():
        return {}
    out = {}
    for algo_dir in sorted([d for d in eval_dir.iterdir() if d.is_dir()]):
        algo = algo_dir.name
        codes = []
        for ext in EXTS:
            for fp in algo_dir.glob(f"impl_*{ext}"):
                rid = str(fp.relative_to(lang_root))
                codes.append((rid, read_text(fp)))
        # fallback: any file if no ext matched
        if not codes:
            for fp in algo_dir.glob("*"):
                if fp.is_file():
                    rid = str(fp.relative_to(lang_root))
                    codes.append((rid, read_text(fp)))
        if codes:
            out[algo] = codes
    return out

def make_pairs_from_algo_groups(groups: Dict[str,List[Tuple[str,str]]],
                                neg_per_pos: int = 1,
                                all_neg: bool = False,
                                seed: int = 42):
    """
    Build pairs (id_a, code_a, id_b, code_b, label)
    Positives: all combinations within each algo.
    Negatives: sampled across different algos (neg_per_pos per positive), or all if all_neg=True.
    """
    rng = random.Random(seed)
    algos = sorted(groups.keys())
    # positives
    pos_pairs = []
    for algo in algos:
        items = groups[algo]
        for (ia,ca),(ib,cb) in combinations(items, 2):
            if ia <= ib:
                pos_pairs.append((ia,ca,ib,cb,1))
            else:
                pos_pairs.append((ib,cb,ia,ca,1))
    # negatives
    neg_pairs = []
    if all_neg:
        for i,a in enumerate(algos):
            for j in range(i+1,len(algos)):
                b = algos[j]
                for (ia,ca) in groups[a]:
                    for (ib,cb) in groups[b]:
                        if ia <= ib: neg_pairs.append((ia,ca,ib,cb,0))
                        else:        neg_pairs.append((ib,cb,ia,ca,0))
    else:
        by_algo = {a: groups[a][:] for a in algos}
        def algo_of(fid: str) -> str:
            parts = Path(fid).parts
            if "eval" in parts:
                k = parts.index("eval")
                if k+1 < len(parts): return parts[k+1]
            return ""
        for (ia,ca,ib,cb,_) in pos_pairs:
            anchor_id, anchor_code = (ia,ca) if rng.random()<0.5 else (ib,cb)
            anchor_algo = algo_of(anchor_id)
            others = [a for a in algos if a != anchor_algo]
            for _ in range(neg_per_pos):
                a2 = rng.choice(others)
                id2, code2 = rng.choice(by_algo[a2])
                i1,i2 = (anchor_id, id2) if anchor_id <= id2 else (id2, anchor_id)
                c1,c2 = (anchor_code, code2) if i1==anchor_id else (code2, anchor_code)
                neg_pairs.append((i1,c1,i2,c2,0))
    # dedup by unordered id pair
    pairs = list({(a,b): (a,ca,b,cb,l)
                  for (a,ca,b,cb,l) in pos_pairs + neg_pairs
                  for (a,b) in [(a,b)]}.values())
    return pairs

# --------------- scoring ---------------

@torch.no_grad()
def embed_codes(codes: List[str], tok, enc, device, max_len=256, batch_size=64):
    embs = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        x = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        x = {k: v.to(device) for k,v in x.items()}
        out = enc(**x).last_hidden_state                 # [B,T,H]
        mask = x["attention_mask"].unsqueeze(-1)         # [B,T,1]
        m = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        m = F.normalize(m, dim=-1)
        embs.append(m.cpu())
    return torch.cat(embs, dim=0)                        # [N,D]

@torch.no_grad()
def score_pairs_biencoder(pairs, id2idx, embs):
    scores = np.empty(len(pairs), dtype=np.float32)
    for k,(ia,_,ib,_,_) in enumerate(pairs):
        scores[k] = float((embs[id2idx[ia]] @ embs[id2idx[ib]].T).item())
    return scores

@torch.no_grad()
def score_pairs_crossencoder(pairs, tok, clf, device, max_len=512, batch_size=16):
    scores = np.empty(len(pairs), dtype=np.float32)
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        texts = [ca + tok.sep_token + cb for (_,ca,_,cb,_) in batch]
        x = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        x = {k: v.to(device) for k,v in x.items()}
        prob = torch.softmax(clf(**x).logits, dim=-1)[:,1].detach().cpu().numpy()
        scores[i:i+len(batch)] = prob.astype(np.float32)
    return scores

# --------------- main ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="create_dataset/dataset のルート")
    ap.add_argument("--langs", nargs="*", default=[], help="評価する言語。未指定ならroot直下のディレクトリを自動検出")
    ap.add_argument("--mode", choices=["biencoder","crossencoder"], required=True)
    ap.add_argument("--model_id", default="microsoft/codebert-base")
    ap.add_argument("--model_pt", required=True, help="encoder_best.pt or clf_best.pt")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--neg-per-pos", type=int, default=1, help="正例1件あたりの負例作成数（サンプリング）")
    ap.add_argument("--all-neg", action="store_true", help="異フォルダ間の全負例を生成（巨大になる可能性あり）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_precision", type=float, default=0.90, help="Recall@P の P 値（既定 0.90）")
    ap.add_argument("--out_dir", default="eval_out")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 言語一覧
    root = Path(args.root)
    langs = args.langs if args.langs else sorted([d.name for d in root.iterdir() if d.is_dir()])
    print("[info] languages:", langs)

    # モデル
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if args.mode == "biencoder":
        enc = AutoModel.from_pretrained(args.model_id).to(device).eval()
        enc.load_state_dict(torch.load(args.model_pt, map_location=device))
        clf = None
    else:
        clf = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=2).to(device).eval()
        clf.load_state_dict(torch.load(args.model_pt, map_location=device))
        enc = None

    results = {}
    for L in langs:
        print(f"\n=== [{L}] ===")
        lang_root = root / L
        groups = list_lang_items(lang_root)
        if not groups:
            print("[warn] no eval/<algo> found; skip")
            continue

        # ペア生成
        pairs = make_pairs_from_algo_groups(groups,
                                            neg_per_pos=args.neg_per_pos,
                                            all_neg=args.all_neg,
                                            seed=args.seed)
        if not pairs:
            print("[warn] no pairs; skip"); continue

        labels = np.array([l for (*_, l) in pairs], dtype=np.int32)

        # スコア
        if args.mode == "biencoder":
            ids = sorted({ia for (ia,_,_,_,_) in pairs} | {ib for (_,_,ib,_,_) in pairs})
            id2idx = {fid:i for i,fid in enumerate(ids)}
            code_table = {fid:code for algo, items in groups.items() for (fid,code) in items}
            codes = [code_table[fid] for fid in ids]
            embs = embed_codes(codes, tok, enc, device, max_len=args.max_len, batch_size=args.batch_size)
            scores = score_pairs_biencoder(pairs, id2idx, embs)
        else:
            scores = score_pairs_crossencoder(pairs, tok, clf, device,
                                              max_len=args.max_len, batch_size=args.batch_size)

        # 指標
        P, R, thr = compute_pr(labels, scores)
        ap = pr_auc(P, R)
        best = best_f1_threshold(labels, scores)
        rap = recall_at_precision(labels, scores, args.target_precision)

        # 表示
        print(f"[{L}] PR-AUC={ap:.4f}  τ_F1={best['tau_f1']:.6f}  "
              f"F1max={best['f1_max']:.4f}  P={best['precision']:.3f}  R={best['recall']:.3f}  "
              f"TP={best['tp']} FP={best['fp']} FN={best['fn']}")
        if rap is None:
            print(f"      Recall@P (P≥{args.target_precision:.2f}) : **unachieved**")
        else:
            print(f"      Recall@P (P≥{args.target_precision:.2f}) : "
                  f"R={rap['recall_at_p']:.4f}  τ={rap['threshold_at_p']:.6f}  "
                  f"P={rap['precision_at_p']:.4f}  "
                  f"(TP={rap['tp_at_p']} FP={rap['fp_at_p']} FN={rap['fn_at_p']})")

        # 保存：PR曲線 CSV
        csv_path = Path(args.out_dir) / f"pr_curve_{args.mode}_{L}.csv"
        save_pr_curve_csv(labels, scores, csv_path)
        print(f"  [saved] PR curve -> {csv_path}")

        # まとめ JSON
        res = {
            "pr_auc": float(ap),
            "tau_f1": float(best["tau_f1"]),
            "f1_max": float(best["f1_max"]),
            "precision_at_tau_f1": float(best["precision"]),
            "recall_at_tau_f1": float(best["recall"]),
            "tp_at_tau_f1": int(best["tp"]),
            "fp_at_tau_f1": int(best["fp"]),
            "fn_at_tau_f1": int(best["fn"]),
            "n_pairs": int(len(pairs)),
            "n_pos": int((labels==1).sum()),
            "n_neg": int((labels==0).sum()),
            "target_precision": float(args.target_precision),
        }
        if rap is None:
            res.update({
                "recall_at_p": None,
                "precision_at_p": None,
                "threshold_at_p": None,
                "tp_at_p": None, "fp_at_p": None, "fn_at_p": None
            })
        else:
            res.update({
                "recall_at_p": float(rap["recall_at_p"]),
                "precision_at_p": float(rap["precision_at_p"]),
                "threshold_at_p": float(rap["threshold_at_p"]),
                "tp_at_p": int(rap["tp_at_p"]),
                "fp_at_p": int(rap["fp_at_p"]),
                "fn_at_p": int(rap["fn_at_p"])
            })
        results[L] = res

    out_path = Path(args.out_dir) / f"results_{args.mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n[done] wrote:", out_path)

if __name__ == "__main__":
    main()
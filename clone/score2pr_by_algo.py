#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[EN] Compute PR curve & metrics from pairwise scores (id1,id2,score)
-------------------------------------------------------------------
Overview
- Reads a score CSV (id1,id2,score[ ,label]) and (optionally) a block labels file
  (blocks.pkl with id1,id2,label). If the score CSV already has a `label` column,
  it is used directly; otherwise labels are merged from blocks.pkl.
- Normalizes unordered pairs (id1<id2) and, for duplicate pairs, keeps the row with
  the **highest score**. Pairs without labels after merging are dropped.
- Sweeps thresholds in **descending score order** to build a compressed PR curve
  (unique thresholds), computes Best-F1 and PR-AUC (trapezoid on recall), and
  optionally evaluates at a fixed threshold (if --threshold is provided).
- Writes:
  - PR curve to CSV (columns: threshold,tp,fp,fn,precision,recall,f1)
  - Summary metrics to JSON (used_threshold, precision, recall, f1, pr_auc, best_f1, counts)
  - Prints the same metrics JSON to stdout.

Inputs
- --scores         : eval_scores.csv with columns: id1,id2,score[,label]
- --labels-blocks  : (optional) blocks.pkl with columns: id1,id2,label
                     (required if --scores has no label column)
- --threshold      : (optional) fixed decision threshold; if omitted, metrics are also
                     reported for the Best-F1 threshold (as reference)

Outputs
- --out-pr    : PR curve CSV (default: pr_curve.csv)
- --out-json  : Metrics JSON (default: metrics.json)

Usage examples
1) Labels embedded in the scores CSV
   python compute_pr.py \
     --scores eval_scores.csv \
     --out-pr pr_curve.csv \
     --out-json metrics.json

2) Labels provided via blocks.pkl
   python compute_pr.py \
     --scores eval_scores.csv \
     --labels-blocks blocks.pkl \
     --out-pr pr_curve.csv \
     --out-json metrics.json

3) Evaluate at a fixed threshold (e.g., transferred from validation)
   python compute_pr.py \
     --scores eval_scores.csv \
     --labels-blocks blocks.pkl \
     --threshold 0.70 \
     --out-pr pr_curve.csv \
     --out-json metrics.json

Notes
- Pair identity is made order-invariant by sorting (id1<id2).
- Duplicate predicted pairs: only the **max score** per pair is retained.
- PR-AUC uses a simple trapezoidal rule over recall.
- Dependencies: numpy, pandas, scikit-learn.

-------------------------------------------------------------------

[JA] ペアスコアから PR 曲線と指標を計算するツール（id1,id2,score）
---------------------------------------------
概要
- スコア CSV（id1,id2,score[,label]）を読み込み、（必要なら）blocks.pkl（id1,id2,label）
  からラベルをマージします。scores に label 列があればそれをそのまま使用します。
- ペアは順序を正規化（id1<id2）。重複ペアは **最高スコア** の1件だけ残します。
- スコア降順でしきい値を掃引し、ユニークなしきい値で圧縮した PR 曲線を生成。
  Best-F1 と PR-AUC（recall 軸での台形近似）を計算します。
- 出力:
  - PR 曲線 CSV（列: threshold,tp,fp,fn,precision,recall,f1）
  - 指標 JSON（used_threshold, precision, recall, f1, pr_auc, best_f1, 件数など）
  - 同じ JSON を標準出力にも表示

入力
- --scores         : eval_scores.csv（列: id1,id2,score[,label]）
- --labels-blocks  : blocks.pkl（列: id1,id2,label）
                     ※scores に label が無い場合に必須
- --threshold      : 固定しきい値（任意）。未指定時は Best-F1 の値も参考として出力

出力
- --out-pr    : PR 曲線 CSV（既定: pr_curve.csv）
- --out-json  : 指標 JSON（既定: metrics.json）

使い方例
1) scores に label 列がある場合
   python compute_pr.py \
     --scores eval_scores.csv \
     --out-pr pr_curve.csv \
     --out-json metrics.json

2) blocks.pkl からラベルを与える場合
   python compute_pr.py \
     --scores eval_scores.csv \
     --labels-blocks blocks.pkl \
     --out-pr pr_curve.csv \
     --out-json metrics.json

3) 固定しきい値で評価（例：検証で決めた 0.70）
   python compute_pr.py \
     --scores eval_scores.csv \
     --labels-blocks blocks.pkl \
     --threshold 0.70 \
     --out-pr pr_curve.csv \
     --out-json metrics.json

補足
- ペア同一性は (id1<id2) 化で順序非依存に統一。
- 重複予測は最高スコアのみ採用。
- PR-AUC は recall 軸での台形近似。
- 依存ライブラリ: numpy, pandas, scikit-learn。
"""

import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def load_scores(path):
    df = pd.read_csv(path)
    need = {"id1","id2","score"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {need}")
    # 正規化（id1<id2）＆重複ペアは最大スコアを採用
    lo = df[["id1","id2"]].min(axis=1)
    hi = df[["id1","id2"]].max(axis=1)
    df["id1"], df["id2"] = lo, hi
    df = df.sort_values("score", ascending=False).drop_duplicates(subset=["id1","id2"], keep="first")
    return df

def load_labels_from_blocks(blocks_path):
    df = pd.read_pickle(blocks_path)[["id1","id2","label"]].copy()
    lo = df[["id1","id2"]].min(axis=1)
    hi = df[["id1","id2"]].max(axis=1)
    df["id1"], df["id2"] = lo, hi
    df = df.drop_duplicates(subset=["id1","id2"], keep="first")
    return df

def build_gold(scores_df, labels_blocks_path=None):
    if "label" in scores_df.columns:
        # そのまま利用
        g = scores_df[["id1","id2","label"]].copy()
    else:
        if not labels_blocks_path:
            raise ValueError("eval_scores.csv に label 列が無いので --labels-blocks を指定してください。")
        lb = load_labels_from_blocks(labels_blocks_path)
        g = pd.merge(scores_df[["id1","id2"]], lb, on=["id1","id2"], how="left")
    # 欠損は除外
    before = len(g)
    g = g.dropna(subset=["label"])
    dropped = before - len(g)
    return g, dropped

def pr_curve(scores, labels):
    # スコア降順で掃引（同スコア連続をまとめて記録）
    arr = np.array(sorted(zip(scores, labels), key=lambda x: x[0], reverse=True), dtype=float)
    y = arr[:,1].astype(int)
    P,R,F,T,tp,fp,fn = [],[],[],[],[],[],[]
    G = int(y.sum())
    tp_c = fp_c = 0
    fn_c = G
    prev = None
    for i,(s,lab) in enumerate(arr):
        if lab == 1:
            tp_c += 1; fn_c -= 1
        else:
            fp_c += 1
        if prev is None or s != prev:
            prec = tp_c/(tp_c+fp_c) if (tp_c+fp_c)>0 else 1.0
            rec  = tp_c/G if G>0 else 0.0
            f1   = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
            P.append(prec); R.append(rec); F.append(f1); T.append(s)
            tp.append(tp_c); fp.append(fp_c); fn.append(fn_c)
            prev = s
    if len(F)==0:
        return ([],[],[],[],[],[],[]), {"threshold":0.5,"precision":0.0,"recall":0.0,"f1":0.0}
    i_best = int(np.argmax(F))
    best = {"threshold": float(T[i_best]), "precision": float(P[i_best]),
            "recall": float(R[i_best]), "f1": float(F[i_best])}
    return (P,R,F,T,tp,fp,fn), best

def pr_auc(P,R):
    # R 昇順で台形近似
    if not P: return 0.0
    R = np.array(R); P = np.array(P)
    order = np.argsort(R)
    return float(np.trapz(P[order], R[order]))

def main():
    ap = argparse.ArgumentParser(description="Compute PR curve & F1 from eval_scores.csv")
    ap.add_argument("--scores", required=True, help="eval_scores.csv (id1,id2,score[,label])")
    ap.add_argument("--labels-blocks", default=None, help="blocks.pkl（id1,id2,label）; scoresにlabelが無いとき必要")
    ap.add_argument("--threshold", type=float, default=None, help="固定しきい値（未指定ならbest-F1を参考として出力）")
    ap.add_argument("--out-pr", default="pr_curve.csv")
    ap.add_argument("--out-json", default="metrics.json")
    args = ap.parse_args()

    scores_df = load_scores(args.scores)
    gold_df, dropped = build_gold(scores_df, args.labels_blocks)

    # マージして同一順序でベクトル化
    df = pd.merge(scores_df, gold_df, on=["id1","id2"], how="inner")
    scores = df["score"].astype(float).tolist()
    labels = df["label"].astype(int).tolist()

    (P,R,F,T,TP,FP,FN), best = pr_curve(scores, labels)
    auc = pr_auc(P,R)

    # 固定しきい値で最終指標
    thr = args.threshold if args.threshold is not None else best["threshold"]
    pred = [1 if s >= thr else 0 for s in scores]
    p,r,f,_ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    cm = confusion_matrix(labels, pred).tolist()  # [[tn,fp],[fn,tp]]

    # 保存
    pr_df = pd.DataFrame({
        "threshold": T, "tp": TP, "fp": FP, "fn": FN,
        "precision": P, "recall": R, "f1": F
    })
    pr_df.to_csv(args.out_pr, index=False)

    metrics = {
        "used_threshold": float(thr),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "pr_auc": float(auc),
        "best_f1": best,                       # 参考: このデータ上のBest-F1
        "pairs_scored": int(len(df)),
        "pairs_dropped_no_label": int(dropped),
        "note": "If threshold is omitted, metrics are computed at best-F1 (reported separately) only for reference; "
                "use a fixed threshold from training/validation for fair evaluation."
    }
    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # 画面にも要約出力
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTNN/任意予測スコアCSV → Gold生成 → PR曲線/Best-F1 を出すユーティリティ
======================================================================

このスクリプトは、ペア予測スコア（id1,id2,score のCSV）から
- Gold（正解ペア）を〔同一ファイル内の全関数ペア〕として自動生成し、
- PR曲線（CSV）と、Best-F1点・PR-AUC などの指標（JSON）を出力します。

想定用途
--------
- ASTNN や任意の類似度/判定器が出した `id1,id2,score` の CSV を評価する
- id→file の対応表（id_map.csv）がある、または id を 2つで1ファイルと見なせる（"--two-per-file"）とき
- 関数の行数情報（programs.pkl）があれば、短い関数を除外（--min-lines）して Gold を作れる

入出力
------
[入力]
- --pred         : 予測CSV。列: id1,id2,score
- --idmap        : (推奨) id_map.csv。列: id,file[,len]  *len列があれば関数行数として使う
                   ない場合は --two-per-file を使う（file_id=(id-1)//2 を仮定）
- --programs     : (任意) programs.pkl（pandas DataFrame: 列 id,code[,label]）
                   len列がない/欠損のとき、ここから 非空行数 を計算して補完できる
- --two-per-file : idmapが無いときに「2関数/ファイル仮定」を使う
- --min-lines    : Goldに入れる最小関数行数（既定=12。Noneにしたい場合は -1 等にしてコード側で軽修正ください）

[出力]
- --out-json : 指標をまとめた JSON
  {
    "num_functions_kept": ...,
    "total_gold_pairs": ...,
    "num_pred_pairs_in": ...,
    "num_pred_pairs_resolved": ...,
    "pr_auc": ...,
    "best_f1": { "threshold":t, "precision":P, "recall":R, "f1":F1, "tp":TP, "fp":FP, "fn":FN },
    "f1_95_band": {"f_min":..., "f_max":..., "threshold_min":..., "threshold_max":..., "ratio":0.95 },
    "note": "Goldや前処理の由来"
  }

- --out-pr   : PR曲線CSV（列: threshold,tp,fp,fn,precision,recall,f1）
               しきい値（score）のユニーク値ごとの圧縮曲線です

Gold の定義（重要）
-------------------
- 同一ファイルに属する関数群から、全ての関数ペアの組み合わせを Gold positive とする
- id→file の対応は --idmap で指定するか、--two-per-file で (id-1)//2 の簡易規則を用いる
- --min-lines が指定されると、短い関数を除外してから Gold を作る（*len列や programs.pklからの非空行数に基づく）

注意
----
- 予測CSVに同じペアが重複していても、最高スコア1件だけ残して評価します
- Gold universe に存在しない id（= id_map に無い or 除外された関数）は評価対象外に落ちます
- PR-AUC は簡易な台形公式（recall軸）で算出しています

使い方（例）
------------
1) id_map.csv があり、関数長も len 列に含まれている場合
  $ python eval_astnn_metrics.py \
      --pred astnn_pred.csv \
      --idmap id_map.csv \
      --min-lines 12 \
      --out-json metrics.json \
      --out-pr pr_curve.csv

2) id_map.csv に len が無いので programs.pkl から非空行数を計算
  $ python eval_astnn_metrics.py \
      --pred astnn_pred.csv \
      --idmap id_map.csv \
      --programs programs.pkl \
      --min-lines 12 \
      --out-json metrics.json \
      --out-pr pr_curve.csv

3) id_map.csv が無いので「2つ毎に同一ファイル（id1=1,2 → file0 / id=3,4 → file1）」と仮定
  $ python eval_astnn_metrics.py \
      --pred astnn_pred.csv \
      --two-per-file \
      --min-lines 12 \
      --out-json metrics.json \
      --out-pr pr_curve.csv
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

def load_predictions(path: str,
                     id1_col: str = "id1",
                     id2_col: str = "id2",
                     score_col: str = "score") -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in [id1_col, id2_col, score_col]:
        if col not in df.columns:
            raise ValueError("predictions CSV must contain columns: %s" % [id1_col, id2_col, score_col])
    # Normalize pair order and keep max score per pair
    df["a"] = df[id1_col].astype(int)
    df["b"] = df[id2_col].astype(int)
    lo = df[["a","b"]].min(axis=1)
    hi = df[["a","b"]].max(axis=1)
    df["_pair"] = list(zip(lo, hi))
    df["_score"] = df[score_col].astype(float)
    # keep max score per pair
    df = df.sort_values("_score", ascending=False).drop_duplicates(subset=["_pair"], keep="first")
    return df[["_pair", "_score"]].reset_index(drop=True)

def load_id_map(idmap_path: Optional[str], two_per_file: bool, ids_from_preds: Iterable[int],
                programs_path: Optional[str] = None, min_lines: Optional[int] = None) -> Tuple[Dict[int, int], Dict[int, Optional[int]], str]:
    """
    Returns:
        id2file: function id -> file id
        id2len : function id -> code length in lines (or None if unknown)
        note   : provenance note string
    """
    ids = set(int(i) for i in ids_from_preds)
    id2file: Dict[int, int] = {}
    id2len: Dict[int, Optional[int]] = defaultdict(lambda: None)
    note_parts = []
    if idmap_path:
        m = pd.read_csv(idmap_path)
        if "id" not in m.columns or "file" not in m.columns:
            raise ValueError("id_map CSV must contain columns: id,file[,len]")
        m["id"] = m["id"].astype(int)
        m = m[m["id"].isin(ids)]
        for _, row in m.iterrows():
            id2file[int(row["id"])] = int(row["file"])
            if "len" in m.columns and not (pd.isna(row.get("len"))):
                try:
                    id2len[int(row["id"])] = int(row["len"])
                except Exception:
                    pass
        note_parts.append("gold derived from id_map.csv (file grouping)")
    elif two_per_file:
        # Derive file id as (id-1)//2
        for i in ids:
            id2file[i] = (i - 1) // 2
        note_parts.append("gold derived by two-per-file assumption: file_id=(id-1)//2")
    else:
        raise ValueError("Provide --idmap or enable --two-per-file to derive gold pairs.")

    # Optional: compute lengths from programs.pkl if given
    if programs_path and (min_lines is not None):
        try:
            p = pd.read_pickle(programs_path)
            if not set(["id","code"]).issubset(p.columns):
                raise ValueError("programs.pkl must contain columns: id, code")
            p = p[p["id"].isin(ids)][["id","code"]]
            for _, row in p.iterrows():
                code = str(row["code"])
                # non-empty code lines count
                length = sum(1 for ln in code.splitlines() if ln.strip())
                id2len[int(row["id"])] = length
            note_parts.append("function length computed from programs.pkl as non-empty line count")
        except Exception as e:
            note_parts.append("function length unknown (programs.pkl not usable: %s)" % (e,))

    return id2file, id2len, "; ".join(note_parts)

def build_gold_pairs(id2file: Dict[int,int],
                     id2len: Dict[int, Optional[int]],
                     min_lines: Optional[int]) -> Tuple[Set[Tuple[int,int]], int, int]:
    """
    Returns:
        gold_pairs: set of unordered pairs (i,j) with i<j that are in the same file and pass min_lines
        num_funcs_kept: number of unique function ids kept after min_lines filter
        total_gold_pairs: count of gold pairs
    """
    by_file: Dict[int, List[int]] = defaultdict(list)
    # filter by length if available
    for fid, f in id2file.items():
        L = id2len.get(fid, None)
        if (min_lines is None) or (L is None) or (L > min_lines):
            by_file[f].append(fid)

    num_funcs_kept = sum(len(v) for v in by_file.values())
    gold_pairs: Set[Tuple[int,int]] = set()
    for _, ids in by_file.items():
        ids = sorted(ids)
        for a, b in combinations(ids, 2):
            gold_pairs.add((a,b))
    total_gold_pairs = len(gold_pairs)
    return gold_pairs, num_funcs_kept, total_gold_pairs

def compute_pr(gold_pairs: Set[Tuple[int,int]],
               preds: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    preds: DataFrame with columns ["_pair","_score"], pairs unique
    Returns:
        pr_df: DataFrame with threshold,tp,fp,fn,precision,recall,f1
        best:  dict with best F1 info
    """
    # sort unique thresholds descending
    scores = preds["_score"].values
    unique_th = np.unique(scores)[::-1]
    gold = gold_pairs
    G = len(gold)

    # Fast lookup
    pred_pairs = preds["_pair"].tolist()
    pair2score = dict(zip(pred_pairs, scores))

    # Precompute order by score
    order = np.argsort(scores)[::-1]
    ordered_pairs = [pred_pairs[i] for i in order]
    ordered_scores = [scores[i] for i in order]

    # For cumulative predictions as threshold moves downwards
    running_set: Set[Tuple[int,int]] = set()
    last_score = None
    rows = []

    idx = 0
    for t in unique_th:
        # add new pairs whose score >= t (advance idx until score < t)
        while idx < len(ordered_scores) and ordered_scores[idx] >= t:
            running_set.add(ordered_pairs[idx])
            idx += 1
        TP = len(running_set & gold)
        FP = len(running_set) - TP
        FN = G - TP
        if TP + FP == 0:
            prec = 1.0  # convention
        else:
            prec = TP / float(TP + FP)
        rec = 0.0 if G == 0 else TP / float(G)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        rows.append((t, TP, FP, FN, prec, rec, f1))

    pr_df = pd.DataFrame(rows, columns=["threshold","tp","fp","fn","precision","recall","f1"])

    # Best F1
    if len(pr_df) > 0:
        i = pr_df["f1"].values.argmax()
        best = {
            "threshold": float(pr_df.iloc[i]["threshold"]),
            "precision": float(pr_df.iloc[i]["precision"]),
            "recall": float(pr_df.iloc[i]["recall"]),
            "f1": float(pr_df.iloc[i]["f1"]),
            "tp": int(pr_df.iloc[i]["tp"]),
            "fp": int(pr_df.iloc[i]["fp"]),
            "fn": int(pr_df.iloc[i]["fn"]),
        }
    else:
        best = {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

    return pr_df, best

def compute_pr_auc(pr_df: pd.DataFrame) -> float:
    """Area under the PR curve via trapezoidal integration over recall ascending."""
    if len(pr_df) == 0:
        return 0.0
    df = pr_df.sort_values("recall")
    x = df["recall"].values
    y = df["precision"].values
    # Ensure starting at recall=0
    if len(x) == 0 or x[0] > 0.0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 1.0)
    # Ensure ending at recall=1 if possible (optional; we leave as-is)
    auc = float(np.trapz(y, x))
    return auc

def compute_f1_band(pr_df: pd.DataFrame, best_f1: float, ratio: float = 0.95) -> Dict:
    if len(pr_df) == 0 or best_f1 <= 0:
        return {"f_min": 0.0, "f_max": 0.0, "threshold_min": 0.0, "threshold_max": 0.0, "ratio": ratio}
    mask = pr_df["f1"] >= (ratio * best_f1)
    sub = pr_df[mask]
    if len(sub) == 0:
        return {"f_min": 0.0, "f_max": 0.0, "threshold_min": 0.0, "threshold_max": 0.0, "ratio": ratio}
    f_min = float(sub["f1"].min())
    f_max = float(sub["f1"].max())
    t_min = float(sub["threshold"].min())
    t_max = float(sub["threshold"].max())
    return {"f_min": f_min, "f_max": f_max, "threshold_min": t_min, "threshold_max": t_max, "ratio": ratio}

def main():
    ap = argparse.ArgumentParser(description="Compute metrics & PR curve from ASTNN eval outputs.")
    ap.add_argument("--pred", required=True, help="Predictions CSV with columns: id1,id2,score")
    ap.add_argument("--idmap", default=None, help="CSV with columns: id,file[,len] (optional len=code lines)")
    ap.add_argument("--programs", default=None, help="programs.pkl (id,code[,label]) to compute line lengths")
    ap.add_argument("--two-per-file", action="store_true", help="Assume sequential two-per-file IDs (file_id=(id-1)//2)")
    ap.add_argument("--min-lines", type=int, default=12, help="Minimum function line count to keep (non-empty lines)")
    ap.add_argument("--out-json", required=True, help="Output metrics JSON path")
    ap.add_argument("--out-pr", required=True, help="Output PR CSV path")
    args = ap.parse_args()

    preds_df = load_predictions(args.pred)
    num_pred_pairs_in = len(preds_df)

    # Build mapping & gold
    all_ids = set()
    for a,b in preds_df["_pair"].tolist():
        all_ids.add(int(a)); all_ids.add(int(b))
    id2file, id2len, provenance = load_id_map(args.idmap, args.two_per_file, all_ids, programs_path=args.programs, min_lines=args.min_lines)

    gold_pairs, num_funcs_kept, total_gold_pairs = build_gold_pairs(id2file, id2len, args.min_lines)

    # Resolve predictions: keep only pairs where both ids are known and used in gold universe
    kept_pairs = []
    kept_scores = []
    gold_universe_ids = set([i for pair in gold_pairs for i in pair])
    for (a,b), s in preds_df[["_pair","_score"]].itertuples(index=False):
        if a in gold_universe_ids and b in gold_universe_ids:
            kept_pairs.append((a,b))
            kept_scores.append(float(s))
    num_pred_pairs_resolved = len(kept_pairs)
    preds_resolved = pd.DataFrame({"_pair": kept_pairs, "_score": kept_scores})

    # Compute PR & best-F1
    pr_df, best = compute_pr(gold_pairs, preds_resolved)
    pr_auc = compute_pr_auc(pr_df)
    band = compute_f1_band(pr_df, best.get("f1", 0.0), ratio=0.95)

    # Write outputs
    pr_df.to_csv(args.out_pr, index=False)
    metrics = {
        "num_functions_kept": int(num_funcs_kept),
        "total_gold_pairs": int(total_gold_pairs),
        "num_pred_pairs_in": int(num_pred_pairs_in),
        "num_pred_pairs_resolved": int(num_pred_pairs_resolved),
        "pr_auc": float(pr_auc),
        "best_f1": best,
        "f1_95_band": band,
        "note": ("Gold = all function pairs within the same file"
                 + (" (function length > min_lines)" if args.min_lines is not None else "")
                 + f". Resolved with: {provenance}. Predictions outside the gold universe are dropped.")
    }
    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Wrote:", args.out_json)
    print("Wrote:", args.out_pr)

if __name__ == "__main__":
    main()

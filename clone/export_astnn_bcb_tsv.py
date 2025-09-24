#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert programs.pkl / id_map.csv / oj_clone_ids.pkl
→ bcb_funcs_all.tsv / bcb_pairs_all.tsv （診断ログつき）

- bcb_funcs_all.tsv : id \t path \t code
- bcb_pairs_all.tsv : id1 \t id2 \t label
"""

from __future__ import annotations
import argparse, os, sys, traceback
import pandas as pd
from pathlib import Path

def log(msg: str) -> None:
    print(msg, flush=True)

def must_exist(p: Path, desc: str) -> None:
    if not p.exists():
        log(f"[ERROR] missing {desc}: {p}")
        sys.exit(2)
    sz = p.stat().st_size
    log(f"[check] {desc}: {p}  ({sz} bytes)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="programs.pkl / id_map.csv / oj_clone_ids.pkl があるディレクトリ")
    ap.add_argument("--out-dir", required=True, help="TSVを書き出すディレクトリ（通常は同じ場所）")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prog_pkl = in_dir / "programs.pkl"
    idmap_csv = in_dir / "id_map.csv"
    pairs_pkl = in_dir / "oj_clone_ids.pkl"

    log(f"[start] in={in_dir}  out={out_dir}")
    must_exist(prog_pkl, "programs.pkl")
    must_exist(idmap_csv, "id_map.csv")
    must_exist(pairs_pkl, "oj_clone_ids.pkl")

    try:
        # 1) funcs TSV
        df_prog = pd.read_pickle(prog_pkl)  # id, code, label
        df_map  = pd.read_csv(idmap_csv)    # id, algo, file, path
        log(f"[info] programs.pkl rows={len(df_prog)}  cols={list(df_prog.columns)}")
        log(f"[info] id_map.csv   rows={len(df_map)}   cols={list(df_map.columns)}")

        if "id" not in df_prog.columns or "code" not in df_prog.columns:
            log("[ERROR] programs.pkl に id / code 列が見当たりません")
            sys.exit(3)
        if "id" not in df_map.columns or "path" not in df_map.columns:
            log("[ERROR] id_map.csv に id / path 列が見当たりません")
            sys.exit(3)

        df_funcs = pd.merge(df_prog[["id","code"]], df_map[["id","path"]], on="id", how="left")
        miss = df_funcs["path"].isna().sum()
        if miss > 0:
            log(f"[warn] id_map に無い id が {miss} 行あります（path が NaN）")

        funcs_tsv = out_dir / "bcb_funcs_all.tsv"
        df_funcs[["id","path","code"]].to_csv(funcs_tsv, sep="\t", index=False, header=False)
        log(f"[ok] wrote {funcs_tsv} rows={len(df_funcs)}")

        # 2) pairs TSV
        df_pairs = pd.read_pickle(pairs_pkl)  # id1, id2, label
        log(f"[info] oj_clone_ids.pkl rows={len(df_pairs)}  cols={list(df_pairs.columns)}")
        if not set(["id1","id2","label"]).issubset(df_pairs.columns):
            log("[ERROR] oj_clone_ids.pkl に id1/id2/label 列が見当たりません")
            sys.exit(3)
        pairs_tsv = out_dir / "bcb_pairs_all.tsv"
        df_pairs[["id1","id2","label"]].to_csv(pairs_tsv, sep="\t", index=False, header=False)
        log(f"[ok] wrote {pairs_tsv} rows={len(df_pairs)}")

        # 互換エイリアス
        alias = out_dir / "oj_clone_ids.tsv"
        df_pairs[["id1","id2","label"]].to_csv(alias, sep="\t", index=False, header=False)
        log(f"[ok] wrote {alias} (alias)")

        log("[done] all outputs generated successfully.")

    except Exception:
        log("[EXCEPTION] unexpected error while converting:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
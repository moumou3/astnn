#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[EN]
Merge multiple per-language programs.pkl (id, code, optional label) and
oj_clone_ids.pkl (id1, id2, label) into a single programs_all.pkl / pairs_all.pkl.

Rules
- For each language, first convert ids to a composite key 'LANG:oldid',
  then remap to a new global sequential id space.
- Treat (id1,id2) and (id2,id1) as duplicates; keep a single unordered pair.
- --balance-per-lang : downsample so that the number of positive pairs is equal across languages.
- --add-cross-lang-neg N : for each positive pair, add N negative pairs drawn across *different* languages.

Example: merge outputs from C, C++, Java, and Python
  python merge_programs_pairs.py \
    --lang C      --prog data_c/programs.pkl      --pairs data_c/oj_clone_ids.pkl \
    --lang CPP    --prog data_cpp/programs.pkl    --pairs data_cpp/oj_clone_ids.pkl \
    --lang Java   --prog data_java/programs.pkl   --pairs data_java/oj_clone_ids.pkl \
    --lang Python --prog data_py/programs.pkl     --pairs data_py/oj_clone_ids.pkl \
    --out_dir data_all \
    --balance-per-lang \
    --add-cross-lang-neg 1
"""
"""
[JA]
複数言語の programs.pkl（id,code, label任意）と oj_clone_ids.pkl（id1,id2,label）を
1つの programs_all.pkl / pairs_all.pkl に統合する。

- 各言語の id は 'LANG:oldid' という複合キーにしてから、グローバル連番に再割当
- (id1,id2) と (id2,id1) は重複除去
- --balance-per-lang で "言語ごとに正例数を揃える" サンプリング（下限合わせ）
- --add-cross-lang-neg N で「異言語間」負例を正例1件につき N 件追加

# 例: C, CPP, Java, Python の各言語で作った出力を統合
python merge_programs_pairs.py \
  --lang C   --prog data_c/programs.pkl    --pairs data_c/oj_clone_ids.pkl \
  --lang CPP --prog data_cpp/programs.pkl  --pairs data_cpp/oj_clone_ids.pkl \
  --lang Java --prog data_java/programs.pkl --pairs data_java/oj_clone_ids.pkl \
  --lang Python --prog data_py/programs.pkl --pairs data_py/oj_clone_ids.pkl \
  --out_dir data_all \
  --balance-per-lang         # （任意）言語ごとに正例数を揃える
  --add-cross-lang-neg 1     # （任意）正例1件につき、他言語から負例を1件追加
"""

import argparse, os, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

def load_prog(prog_pkl: str, lang: str) -> pd.DataFrame:
    df = pd.read_pickle(prog_pkl)
    assert {"id","code"}.issubset(df.columns), f"{prog_pkl} must have id,code"
    # 複合キー new_id_key = LANG:oldid
    df = df.copy()
    df["old_id"] = pd.to_numeric(df["id"], errors="coerce").fillna(-1).astype(int)
    df["new_key"] = df["old_id"].map(lambda x: f"{lang}:{x}")
    return df[["new_key","code"]]

def load_pairs(pairs_pkl: str, lang: str) -> pd.DataFrame:
    df = pd.read_pickle(pairs_pkl)
    assert {"id1","id2","label"}.issubset(df.columns), f"{pairs_pkl} must have id1,id2,label"
    df = df.copy()
    df["id1"] = pd.to_numeric(df["id1"], errors="coerce").fillna(-1).astype(int)
    df["id2"] = pd.to_numeric(df["id2"], errors="coerce").fillna(-1).astype(int)
    df["key1"] = df["id1"].map(lambda x: f"{lang}:{x}")
    df["key2"] = df["id2"].map(lambda x: f"{lang}:{x}")
    # 正規化（順序に依らず一意化）
    a = df["key1"].values; b = df["key2"].values
    k1 = np.minimum(a,b); k2 = np.maximum(a,b)
    df["pair_key"] = [f"{x}|||{y}" for x,y in zip(k1,k2)]
    df = df.drop_duplicates(subset=["pair_key"])
    df["lang"] = lang
    return df[["key1","key2","label","pair_key","lang"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", action="append", default=[], help="言語名（繰り返し指定）")
    ap.add_argument("--prog", action="append", default=[], help="programs.pkl（各--langに対応）")
    ap.add_argument("--pairs", action="append", default=[], help="oj_clone_ids.pkl（各--langに対応）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balance-per-lang", action="store_true", help="言語ごとに正例数を揃える（下限に合わせる）")
    ap.add_argument("--add-cross-lang-neg", type=int, default=0, help="正例1件につき他言語から負例N件を追加")
    args = ap.parse_args()

    assert len(args.lang)==len(args.prog)==len(args.pairs) and len(args.lang)>0, \
        "lang/prog/pairs の数は一致し、1組以上必要です"

    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 各言語のプログラムとペアを読み込み
    prog_frames = []
    pair_frames = []
    pos_by_lang = {}
    for L, P, R in zip(args.lang, args.prog, args.pairs):
        dfp = load_prog(P, L); prog_frames.append(dfp)
        dfpairs = load_pairs(R, L); pair_frames.append(dfpairs)
        pos_by_lang[L] = int((dfpairs["label"]==1).sum())

    # 2) プログラムの複合キー new_key をグローバル連番IDに写像
    prog_all = pd.concat(prog_frames, ignore_index=True).drop_duplicates(subset=["new_key"])
    prog_all = prog_all.reset_index(drop=True)
    prog_all["id"] = np.arange(1, len(prog_all)+1, dtype=int)
    id_map: Dict[str,int] = dict(zip(prog_all["new_key"], prog_all["id"]))

    # 3) ペアのキーを整数IDへ変換し、重複除去
    pairs_all = pd.concat(pair_frames, ignore_index=True)
    pairs_all["id1"] = pairs_all["key1"].map(id_map)
    pairs_all["id2"] = pairs_all["key2"].map(id_map)
    # (a,b)=(b,a)重複防止
    a = pairs_all["id1"].values; b = pairs_all["id2"].values
    id1n = np.minimum(a,b); id2n = np.maximum(a,b)
    pairs_all["norm_key"] = [f"{x}|||{y}" for x,y in zip(id1n,id2n)]
    pairs_all = pairs_all.drop_duplicates(subset=["norm_key"]).dropna(subset=["id1","id2"])

    # 4) （任意）言語ごとに正例数を揃える（下限に合わせてダウンサンプル）
    if args.balance_per_lang:
        mins = min((pairs_all[pairs_all["label"]==1].groupby("lang").size()).tolist())
        keep = []
        for L, grp in pairs_all[pairs_all["label"]==1].groupby("lang"):
            k = grp.sample(n=min(mins, len(grp)), random_state=args.seed)
            keep.append(k.index)
        keep_pos_idx = np.concatenate(keep) if keep else np.array([], dtype=int)
        # 正例は keep のみ、負例はそのまま（または同倍率でサンプルしてもよい）
        pos_mask = pairs_all["label"]==1
        pairs_all = pd.concat([pairs_all.loc[keep_pos_idx], pairs_all.loc[~pos_mask]], ignore_index=True)

    # 5) （任意）クロス言語の負例を追加
    if args.add_cross_lang_neg > 0:
        pos = pairs_all[pairs_all["label"]==1][["id1","id2","lang"]].values.tolist()
        # 言語別に属するID集合
        lang_ids: Dict[str,set] = {}
        for L, dfp in zip(args.lang, prog_frames):
            lang_ids[L] = set(dfp["new_key"].map(id_map).tolist())
        extra = []
        for (i1,i2,L) in pos:
            # 正例1件につき N個を作る
            for _ in range(args.add_cross_lang_neg):
                # 異言語を選ぶ
                langs_others = [x for x in args.lang if x != L]
                L2 = random.choice(langs_others)
                # どちらか一方を固定し、相手は異言語の任意の関数
                anchor = i1 if random.random()<0.5 else i2
                other  = random.choice(list(lang_ids[L2]))
                a,b = (anchor, other) if anchor<other else (other, anchor)
                extra.append((a,b,0))
        if extra:
            df_e = pd.DataFrame(extra, columns=["id1","id2","label"])
            pairs_all = pd.concat([pairs_all[["id1","id2","label"]], df_e], ignore_index=True)
            # 重複除去
            a = pairs_all["id1"].values; b = pairs_all["id2"].values
            nk = [f"{min(x,y)}|||{max(x,y)}" for x,y in zip(a,b)]
            pairs_all["nk"] = nk
            pairs_all = pairs_all.drop_duplicates(subset=["nk"]).drop(columns=["nk"])

    # 6) 最終出力
    prog_out = prog_all[["id","code"]].copy()
    pairs_out = pairs_all[["id1","id2","label"]].astype(int).copy()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    prog_out.to_pickle(os.path.join(args.out_dir, "programs_all.pkl"))
    pairs_out.to_pickle(os.path.join(args.out_dir, "pairs_all.pkl"))
    print("Wrote:", os.path.join(args.out_dir, "programs_all.pkl"), len(prog_out))
    print("Wrote:", os.path.join(args.out_dir, "pairs_all.pkl"),    len(pairs_out))

if __name__ == "__main__":
    main()
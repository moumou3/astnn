# Clone Detection Pipelines (ASTNN / MySQL / SCB)


- **(A) Multi-algorithm evaluation**（タスク別ベンチ・ASTNN ほか）
- **(B) MySQL 大規模評価**（検証で決めた固定しきい値で転用）
- **(C) Semantic Clone Bench (SCB)**

**共通アーティファクト**

- `programs.pkl` — `DataFrame(id, code[, label])`
- `oj_clone_ids.pkl` — `DataFrame(id1, id2, label)`（1 = clone, 0 = non-clone）

> **Tip:** しきい値は検証で決めた **`τ*`** を固定し、テスト／転用でも同じ値を使います。

---

## Prerequisites

- Python 3.8+
- `pipenv`
- 学習は GPU 推奨

依存関係:

```bash
pipenv sync
```

---

## (A) Multi-algorithm evaluation

以下は **Java** の例です（他言語は `--lang` とパスを調整）。

### 1) データ作成（programs & pairs）

```bash
pipenv run python ../HySCU/create_dataset/scripts/make_programs_and_pairs_algo.py   --lang java   --src  ~/HySCU/create_dataset/dataset/java/eval   --out-dir data/java   --neg-per-pos 4   --seed 42
```

（パイプラインが ASTNN/BCB の TSV を期待する場合）

```bash
pipenv run python clone/export_astnn_bcb_tsv.py   --in-dir  data/java   --out-dir data/java/
```

### 2) 前処理

```bash
pipenv run python clone/pipeline.py --lang java
```

### 3) ASTNN 学習

```bash
pipenv run python clone/train.py   --lang java   --save-weights models/astnn_java.pt   --save-best
```

### 4) スコア → 指標

```bash
pipenv run python clone/score2pr_by_algo.py   --scores        output2/java/eval_scores.csv   --labels-blocks data/java/test/blocks.pkl   --out-pr        output2/java/pr_curve.csv   --out-json      output2/java/metrics.json
```

**出力**

- `pr_curve.csv`（列: `threshold,tp,fp,fn,precision,recall,f1`）  
- `metrics.json`（PR-AUC, Best-F1 など）

---

## (B) MySQL 評価（転用）

検証で求めた **`τ*`** を固定して評価します。

```bash
pipenv run python clone/eval_mysql_cpp.py   --cands-dir     ~/HySCU/create_dataset/out/candidates_mysql   --judged        ~/HySCU/create_dataset/out/candidates_mysql/judged.jsonl   --lang-root     data/cpp   --embedding-dim 128   --weights       models/astnn_cpp.pt   --threshold     0.5292   --blocks-func   prepare_data_cpp:get_blocks_cpp   --out-json      mysql_cpp_metrics.json   --out-csv       mysql_cpp_scores.csv   --save-pr       mysql_cpp_pr_curve.csv
```

固定した **`τ*`** を用い、`mysql_cpp_pr_curve.csv` から **指定精度での Recall** を読み取る（またはスクリプトで計算）。

---

## (C) Semantic Clone Bench (SCB)

### 1) SCB からデータ作成

```bash
pipenv run python ../HySCU/create_dataset/scripts/make_programs_and_pairs.py   --src     ~/HySCU/semanticclonebench/Python/Stand_Alone_Clones/   --out-dir codebert/data/semanticclonebench/python   --make-pairs --neg-per-pos 4
```

### 2) 前処理 & 3) ASTNN 学習

```bash
pipenv run python clone/pipeline.py --lang c
pipenv run python clone/train.py    --lang c --save-weights models/astnn_cpp.pt --save-best
```

> `--lang` と保存ファイル名は対象言語に合わせて調整してください。

### 4) スコア → 指標

```bash
pipenv run python clone/eval_astnn_metrics.py   --pred       eval_scores.csv   --two-per-file   --programs   data/c/programs.pkl   --out-json   metrics.json   --out-pr     pr_curve.csv
```

---

## Conventions & Notes

- **言語フラグ:** 各工程で `--lang` を一貫（`c`, `cpp`, `java`, `python`, …）。  
- **しきい値:** 検証で `P ≥ target_precision` を満たす **`τ*`** を選び（`valid_metrics.json` に保存）、テスト／転用でも固定。  
- **負例数:** `--neg-per-pos` でクラス比を制御。再現性のため `--seed` を指定。  
- **重複:** すべての評価器で `(id1,id2)` と `(id2,id1)` は同一ペアとして扱い、重複は最高スコアのみ残す。  
- **成果物:** 実験ごとにディレクトリを分け、`output2/<lang>/` のように CSV/JSON を整理。

---

## Quick Troubleshooting

- **指標が良すぎる** → テストで Best-F1 を使っていないか？ **固定 `τ*`** で評価しているか確認。  
- **言語不一致** → エクスポートしたデータの言語と `--lang` が一致しているか確認。  
- **ラベル不足** → スコア→PR 変換には `blocks.pkl` を用意（または Gold 自動生成対応の評価器を使用）。
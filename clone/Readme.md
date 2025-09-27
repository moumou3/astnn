 # Clone Detection Pipelines (ASTNN / MySQL / SCB)

- **(A) Multi-algorithm evaluation** (per-task benchmark; ASTNN and others)  
- **(B) MySQL large-scale evaluation** (transfer using a fixed validation threshold)  
- **(C) Semantic Clone Bench (SCB)**

**Shared artifacts**

- `programs.pkl` — `DataFrame(id, code[, label])`  
- `oj_clone_ids.pkl` — `DataFrame(id1, id2, label)` (1 = clone, 0 = non-clone)

> **Tip:** pick a validation threshold **`τ*`** and keep it **fixed** for test/transfer.

---

## Prerequisites

- Python 3.8+
- `pipenv`
- GPU recommended for training

Dependencies:

```bash
pipenv sync
```

---

## (A) Multi-algorithm evaluation

Below is a **Java** example (adjust `--lang` and paths for other languages).

### 1) Build data (programs & pairs)

```bash
pipenv run python ../HySCU/create_dataset/scripts/make_programs_and_pairs_algo.py   --lang java   --src  ~/HySCU/create_dataset/dataset/java/eval   --out-dir data/java   --neg-per-pos 4   --seed 42
```

*(If your pipeline expects ASTNN/BCB TSVs)*

```bash
pipenv run python clone/export_astnn_bcb_tsv.py   --in-dir  data/java   --out-dir data/java/
```

### 2) Preprocess

```bash
pipenv run python clone/pipeline.py --lang java
```

### 3) Train ASTNN

```bash
pipenv run python clone/train.py   --lang java   --save-weights models/astnn_java.pt   --save-best
```

### 4) Convert scores → metrics

```bash
pipenv run python clone/score2pr_by_algo.py   --scores        output2/java/eval_scores.csv   --labels-blocks data/java/test/blocks.pkl   --out-pr        output2/java/pr_curve.csv   --out-json      output2/java/metrics.json
```

**Outputs**

- `pr_curve.csv` (columns: `threshold,tp,fp,fn,precision,recall,f1`)  
- `metrics.json` (PR-AUC, Best-F1, etc.)

---

## (B) MySQL evaluation (transfer)

Evaluate with the **fixed validation threshold `τ*`**.

```bash
pipenv run python clone/eval_mysql_cpp.py   --cands-dir     ~/HySCU/create_dataset/out/candidates_mysql   --judged        ~/HySCU/create_dataset/out/candidates_mysql/judged.jsonl   --lang-root     data/cpp   --embedding-dim 128   --weights       models/astnn_cpp.pt   --threshold     0.5292   --blocks-func   prepare_data_cpp:get_blocks_cpp   --out-json      mysql_cpp_metrics.json   --out-csv       mysql_cpp_scores.csv   --save-pr       mysql_cpp_pr_curve.csv
```

Using the fixed **`τ*`**, read **recall at the target precision** from `mysql_cpp_pr_curve.csv` (or compute programmatically).

---

## (C) Semantic Clone Bench (SCB)

### 1) Build data from SCB

```bash
pipenv run python ../HySCU/create_dataset/scripts/make_programs_and_pairs.py   --src     ~/HySCU/semanticclonebench/Python/Stand_Alone_Clones/   --out-dir codebert/data/semanticclonebench/python   --make-pairs --neg-per-pos 4
```

### 2) Preprocess & 3) Train ASTNN

```bash
pipenv run python clone/pipeline.py --lang c
pipenv run python clone/train.py    --lang c --save-weights models/astnn_cpp.pt --save-best
```

> Adjust `--lang` and the weights filename for your target language.

### 4) Convert scores → metrics

```bash
pipenv run python clone/eval_astnn_metrics.py   --pred       eval_scores.csv   --two-per-file   --programs   data/c/programs.pkl   --out-json   metrics.json   --out-pr     pr_curve.csv
```

---

## Conventions & Notes

- **Language flag:** keep `--lang` consistent across steps (`c`, `cpp`, `java`, `python`, …).  
- **Thresholding:** choose **`τ*`** on validation to satisfy `P ≥ target_precision` (saved in `valid_metrics.json`), then reuse **`τ*`** for test/transfer (e.g., MySQL).  
- **Negatives:** control class balance via `--neg-per-pos`; set `--seed` for reproducibility.  
- **Duplicates:** evaluators treat `(id1,id2)` and `(id2,id1)` as the same pair; keep only the **highest score** per pair.  
- **Artifacts:** organize CSV/JSON per experiment, e.g., `output2/<lang>/`.

---

## Quick Troubleshooting

- **Metrics look too good** → check you’re using a **fixed `τ*`** from validation, not Best-F1 on the test set.  
- **Language mismatch** → ensure the exported data’s language matches `--lang` (Java data → `--lang java`, etc.).  
- **Missing labels** → for score→PR conversion, provide `blocks.pkl` (or use an evaluator that builds Gold internally).

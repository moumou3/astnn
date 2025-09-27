# CodeBERT Cross‑Encoder Pipeline — README (English)

This README gathers four practical steps to **prepare data, fine‑tune CodeBERT (cross‑encoder), and evaluate clone detection** on (i) a MySQL judged set and (ii) a 6‑language multi‑task corpus. All commands are copy‑paste ready—adapt paths to your environment.

---

## 0) Prerequisites

- Python 3.8+ (GPU strongly recommended)
- `pipenv` or your preferred virtual environment
- Typical dependencies: `torch`, `transformers`, `pandas`, `numpy`, `scikit-learn`, `tqdm`

Install (example):

```bash
pipenv sync
# or: pip install torch transformers pandas numpy scikit-learn tqdm
```

---

## 1) (Optional) Merge per‑language datasets

Use when you have separate `programs.pkl / oj_clone_ids.pkl` per language and want a single mixed corpus.

```bash
python merge_programs_pairs.py   --lang C      --prog data_c/programs.pkl      --pairs data_c/oj_clone_ids.pkl   --lang CPP    --prog data_cpp/programs.pkl    --pairs data_cpp/oj_clone_ids.pkl   --lang Java   --prog data_java/programs.pkl   --pairs data_java/oj_clone_ids.pkl   --lang Python --prog data_py/programs.pkl     --pairs data_py/oj_clone_ids.pkl   --out_dir data_all   --balance-per-lang   --add-cross-lang-neg 1
```

**Flags**
- `--balance-per-lang` — downsample so each language has the same # of **positives**
- `--add-cross-lang-neg N` — for every positive, add *N* negatives drawn **across different languages**

**Outputs**
- `data_all/programs_all.pkl` — merged program table
- `data_all/pairs_all.pkl` — merged pair table (unordered‑pair dedup applied)

---

## 2) Fine‑tune CodeBERT (cross‑encoder)

### 2.1 Build training data from Semantic Clone Bench (SCB)

```bash
python build_programs_and_pairs.py   --src SemanticClonebench/C/Stand_Alone_Clones   --out-dir data_c   --make-pairs --neg-per-pos 4
```

This creates `programs.pkl` and `oj_clone_ids.pkl` for the chosen language.  
If you merged multiple languages in Step 1, use the merged `programs_all.pkl / pairs_all.pkl` below.

### 2.2 Train (cross‑encoder)

```bash
python train_codebert_scb.py   --prog_pkl data_all/programs_all.pkl   --pairs_train_pkl data_all/pairs_all.pkl   --out_dir out_multi_bi   --mode crossencoder   --epochs 3 --batch_size 64 --lr 2e-5 --max_len 256   --valid_ratio 0.1
```

**Saves**
- `out_multi_bi/clf_best.pt` — best checkpoint (by validation metric)
- `out_multi_bi/valid_metrics.json` — includes the selected validation threshold (often denoted `tau*`)

> **Tip:** Fix the threshold chosen on validation (`tau*`) and reuse it for downstream evaluations (MySQL & 6‑lang).

---

## 3) Clone‑detection eval on a **MySQL judged** dataset

Evaluate the cross‑encoder on pre‑judged MySQL pairs.

```bash
python eval_mysql_jsonl.py   --jsonl Judged_all.jsonl   --mode crossencoder   --model_id microsoft/codebert-base   --model_pt out_scb_cross/clf_best.pt   --tau 0.5310   --base_dir /home/azureuser/HySCU/create_dataset/out   --max_len 512 --batch_size 64 --fp16   --out_dir eval_mysql_cross
```

**Notes**
- `--tau` should be your **validation** threshold `tau*` (example: `0.5310`).
- `--base_dir` is used to resolve any relative file references if needed.
- Outputs typically include per‑pair scores and summary metrics in `eval_mysql_cross/`.

---

## 4) Clone‑detection eval on **6‑language multi‑task**

```bash
python eval_codebert_folder.py   --root create_dataset/dataset   --langs C CPP Java Python   --mode crossencoder   --model_id microsoft/codebert-base   --model_pt out_scb_cross/clf_best.pt   --neg-per-pos 2   --out_dir eval_out_cross
```

**Notes**
- `--langs` lists the languages to evaluate. Extend as needed (e.g., `Go`, `Rust`, …) if supported by your data.
- `--neg-per-pos` controls the negative sampling ratio for evaluation folds (if applicable).
- `eval_out_cross/` will contain PR curves and metrics (PR‑AUC, Precision/Recall/F1, etc.).

---

## Conventions & Good Practice

- **Fixed threshold:** Select `tau*` on validation (e.g., target precision) and reuse it on all test/transfer sets for fair comparison.
- **Pair identity:** Treat `(id1,id2)` and `(id2,id1)` as the same unordered pair; keep only the highest score when dedupping.
- **Reproducibility:** Set seeds where available (e.g., negative sampling) and record exact command lines & commit hashes.
- **Artifacts layout:** Keep per‑experiment subfolders (e.g., `out_multi_bi/`, `eval_mysql_cross/`, `eval_out_cross/`).

---

## Quick Troubleshooting

- **Metrics look inflated** → ensure you’re not using Best‑F1 on test; stick to the **fixed validation threshold (`tau*`)**.  
- **Tokenizer truncation** → try `--max_len 512` (or smaller if memory is tight).  
- **OOM on GPU** → lower `--batch_size`, enable `--fp16`, or accumulate gradients.  
- **Language mismatch** → verify that `--langs` matches the data folders under `--root`.

---

*Happy benchmarking!*  Adjust flags/paths as needed for your setup.

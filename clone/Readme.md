


 - (A) multi-algorism evaluation
1. create program.pkl for (A) algorism evaluation
 pipenv run python ../HySCU/create_dataset/scripts/make_programs_and_pairs_algo.py --lang java --src ~/HySCU/create_dataset/dataset/java/eval  --out-dir data/java --neg-per-pos 4 --seed 42

- for java create tsv file
pipenv run python clone/export_astnn_bcb_tsv.py --in-dir data/java --out-dir data/java/

2. execute pipeline.py
pipenv run python clone/pipeline.py --lang c
3. execute train.py
pipenv run python clone/train.py --lang c --save-weights models/astnn_cpp.pt --save-best
4. convert score output to metrics
  pipenv run python clone/score2pr_by_algo.py --scores output2/java/eval_scores.csv --labels-blocks data/java/test/blocks.pkl --out-pr output2/java/pr_curve.csv --out-json output2/java/metrics.json


- (B) mysql evaluation
1. pipenv run python clone/eval_mysql_cpp.py \
  --cands-dir ~/HySCU/create_dataset/out/candidates_mysql \
  --judged ~/HySCU/create_dataset/out/candidates_mysql/judged.jsonl \
  --lang-root data/cpp \
  --embedding-dim 128 \
  --weights models/astnn_cpp.pt \
  --threshold 0.5292 \            # ← 学習/検証で得た best-F1 閾値を固定（推奨）
  --blocks-func prepare_data_cpp:get_blocks_cpp \
  --out-json mysql_cpp_metrics.json \
  --out-csv mysql_cpp_scores.csv \
  --save-pr mysql_cpp_pr_curve.csv
2. calculate recall at precision


- (C) semantic clone bench evaluation
1. make program.pkl
pipenv run python ../HySCU/create_dataset/scripts/make_programs_and_pairs.py --src ~/HySCU/semanticclonebench/Python/Stand_Alone_Clones/ --out-dir codebert/data/semanticclonebench/python --make-pairs --neg-per-pos 4

2. execute pipeline.py
pipenv run python clone/pipeline.py --lang c
3. execute train.py
pipenv run python clone/train.py --lang c --save-weights models/astnn_cpp.pt --save-best
4. create metrics
pipenv run python clone/eval_astnn_metrics.py --pred eval_scores.csv --two-per-file --programs data/c/programs.pkl --out-json metrics.json --out-pr pr_curve.csv



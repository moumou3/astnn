1. merge programs.pkl if you need
python merge_programs_pairs.py \
  --lang C   --prog data_c/programs.pkl    --pairs data_c/oj_clone_ids.pkl \
  --lang CPP --prog data_cpp/programs.pkl  --pairs data_cpp/oj_clone_ids.pkl \
  --lang Java --prog data_java/programs.pkl --pairs data_java/oj_clone_ids.pkl \
  --lang Python --prog data_py/programs.pkl --pairs data_py/oj_clone_ids.pkl \
  --out_dir data_all \
  --balance-per-lang         # （任意）言語ごとに正例数を揃える
  --add-cross-lang-neg 1     # （任意）正例1件につき、他言語から負例を1件追加
"""

2. fine tuning 

-  create dataset from semantic clone bench
    python build_programs_and_pairs.py \
    --src SemanticClonebench/C/Stand_Alone_Clones \
    --out-dir data_c \
    --make-pairs --neg-per-pos 4

- fine tuning by cross encoder
    python train_codebert_scb.py \
    --prog_pkl data_all/programs_all.pkl \
    --pairs_train_pkl data_all/pairs_all.pkl \
    --out_dir out_multi_bi \
    --mode crossencoder \
    --epochs 3 --batch_size 64 --lr 2e-5 --max_len 256 \
    --valid_ratio 0.1

3. clone detection eval for mysql judged dataset

  python eval_mysql_jsonl.py \
    --jsonl Judged_all.jsonl \
    --mode crossencoder \
    --model_id microsoft/codebert-base \
    --model_pt out_scb_cross/clf_best.pt \
    --tau 0.5310 \
    --base_dir /home/azureuser/HySCU/create_dataset/out \
    --max_len 512 --batch_size 64 --fp16 \
    --out_dir eval_mysql_cross

4. clone detection eval for 6-lang multi-task

  python eval_codebert_folder.py --root create_dataset/dataset \
      --langs C CPP Java Python \
      --mode crossencoder --model_id microsoft/codebert-base \
      --model_pt out_scb_cross/clf_best.pt \
      --neg-per-pos 2 --out_dir eval_out_cross

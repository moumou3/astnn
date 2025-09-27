'''
# Specify pretrained Word2Vec and model weights for ASTNN-cpp (C++)
pipenv run python clone/eval_mysql_cpp.py \
  --cands-dir ~/HySCU/create_dataset/out/candidates_mysql \
  --judged ~/HySCU/create_dataset/out/candidates_mysql/judged.jsonl \
  --lang-root data/cpp \
  --embedding-dim 128 \
  --weights models/astnn_cpp.pt \
  --threshold 0.5292 \           
  --blocks-func prepare_data_cpp:get_blocks_cpp \
  --out-json mysql_cpp_metrics.json \
  --out-csv mysql_cpp_scores.csv \
  --save-pr mysql_cpp_pr_curve.csv
'''

# -*- coding: utf-8 -*-
import os, re, json, argparse, importlib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from model import BatchProgramCC
from config import *  # EMBEDDING_SIZE, HIDDEN_DIM, ENCODE_DIM, LABELS, BATCH_SIZE, USE_GPU

# ---------------------- judged.jsonl 読み込み ----------------------
def parse_ref(ref: str):
    # "candidates_mysql/.../cand_XXXXX.cpp::Func@..." → "cand_XXXXX.cpp"
    path = ref.split("::", 1)[0]
    return os.path.basename(path)

def load_judged(judged_path: str):
    pairs = []
    with open(judged_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            u = parse_ref(obj["u"])
            v = parse_ref(obj["v"])
            verdict = str(obj.get("verdict", "DIFF")).upper()
            label = 1 if verdict in ("EQUIV", "CLOSE") else 0
            pairs.append((u, v, label))
    return pairs

# ---------------------- embeddings/vocab (gensim4) ----------------------
def load_embeddings(lang_root: str, dim: int):
    from gensim.models import Word2Vec
    wv = Word2Vec.load(os.path.join(lang_root, f"train/embedding/node_w2v_{dim}")).wv
    V, D = wv.vectors.shape
    emb = np.zeros((V + 1, D), dtype="float32")
    emb[:V] = wv.vectors
    tok2id = wv.key_to_index
    return emb, V, D, tok2id

# ---------------------- ツリーノード（フォールバック用） ----------------------
class Node:
    __slots__ = ("token", "children")
    def __init__(self, token, children=None):
        self.token = token
        self.children = children or []

TOKEN_SPLIT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|0x[0-9A-Fa-f]+|\d+|==|!=|<=|>=|->|::|&&|\|\||[{}()\[\];,.:+\-*/%<>!&|^=?]")

def pseudo_blocks_cpp(code: str, cap_tokens=512):
    # 簡易トークン木1本を「1ブロック」として返す
    toks = TOKEN_SPLIT.findall(code)[:cap_tokens]
    return [Node("FUNC", [Node(t) for t in toks])]

# ---------------------- blocks 関数の動的ロード ----------------------
def load_blocks_func(spec: str):
    """
    spec="module.submod:function" を import し、戻り値が
      blocks = func(code:str)  となる関数を返す
    """
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("--blocks-func は 'module:function' 形式で指定してください")
    mod_name, func_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    return fn

# ---------------------- tree->index 変換 ----------------------
def make_tree_to_index(tok2id: dict, V: int):
    OOV = V
    def tree_to_index(node):
        idx = tok2id.get(getattr(node, "token", None), OOV)
        out = [idx]
        for ch in getattr(node, "children", []):
            out.append(tree_to_index(ch))
        return out
    return tree_to_index

# ---------------------- ファイル→forest ----------------------
def file_to_forest_cpp(path: str, blocks_fn, tree_to_index):
    try:
        code = open(path, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return None
    # 優先：ユーザ指定の C++ ブロック関数
    blocks = None
    if blocks_fn is not None:
        try:
            blocks = blocks_fn(code)  # 期待：list[Node(token, children)]
        except Exception:
            blocks = None
    # フォールバック：擬似ブロック
    if not blocks:
        blocks = pseudo_blocks_cpp(code)
    # index化
    forest = []
    for b in blocks:
        try:
            forest.append(tree_to_index(b))
        except Exception:
            continue
    return forest if forest else None

# ---------------------- PR/BestF1 ----------------------
def pr_curve(scores, labels):
    arr = list(zip(scores, labels))
    arr.sort(key=lambda x: x[0], reverse=True)
    tp = fp = 0
    fn = sum(labels); tn = len(labels) - fn
    P = []; R = []; F = []; T = []
    prev = None
    for s, y in arr:
        if y == 1: tp += 1; fn -= 1
        else:      fp += 1; tn -= 1
        if prev is None or s != prev:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            P.append(prec); R.append(rec); F.append(f1); T.append(s)
            prev = s
    if F:
        i = int(np.argmax(F))
        best = {"threshold": float(T[i]), "precision": float(P[i]), "recall": float(R[i]), "f1": float(F[i])}
    else:
        best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return (P, R, F, T), best

# ---------------------- 推論（バッチ） ----------------------
def score_pairs(model, id2forest: dict, pairs, batch_size: int, device):
    buf_x, buf_y, metas = [], [], []
    def flush():
        with torch.no_grad():
            model.batch_size = len(buf_x)
            model.hidden = model.init_hidden()
            out = model(buf_x, buf_y).detach().view(-1).cpu().numpy().tolist()
        for s, meta in zip(out, metas):
            yield float(s), meta
    for u, v, lab in pairs:
        fx = id2forest.get(u); fy = id2forest.get(v)
        if fx is None or fy is None: continue
        buf_x.append(fx); buf_y.append(fy); metas.append((u, v, lab))
        if len(buf_x) >= batch_size:
            for item in flush(): yield item
            buf_x, buf_y, metas = [], [], []
    if buf_x:
        for item in flush(): yield item

# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate ASTNN-cpp model on MySQL candidates")
    ap.add_argument("--cands-dir", required=True, help="~/HySCU/create_dataset/out/candidates_mysql")
    ap.add_argument("--judged", required=True, help="judged.jsonl path")
    ap.add_argument("--lang-root", default="data/cpp", help="C++ 学習データのルート（embeddings を探す場所）")
    ap.add_argument("--embedding-dim", type=int, default=EMBEDDING_SIZE)
    ap.add_argument("--weights", required=True, help="学習済み重み .pt（C++用）")
    ap.add_argument("--threshold", type=float, default=None, help="学習/検証で得た best-F1 閾値（推奨）")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--blocks-func", dest="blocks_func", default=None,
                    help="C++ブロック化関数 'module.submod:function'（引数: code:str → 返り値: list[Node]）")
    ap.add_argument("--out-json", default="mysql_cpp_metrics.json")
    ap.add_argument("--out-csv", default="mysql_cpp_scores.csv")
    ap.add_argument("--save-pr", default="mysql_cpp_pr_curve.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")

    # 1) Embeddings / vocab（C++用）
    emb, V, D, tok2id = load_embeddings(args.lang_root, args.embedding_dim)
    tree_to_index = make_tree_to_index(tok2id, V)

    # 2) モデル構築 & 重みロード（C++学習済み）
    model = BatchProgramCC(D, HIDDEN_DIM, V + 1, ENCODE_DIM, LABELS, args.batch_size, USE_GPU, emb)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # 3) C++ブロック関数をロード（なければ擬似）
    blocks_fn = load_blocks_func(args.blocks_func) if args.blocks_func else None

    # 4) cand_*.cpp から forest 構築
    id2forest = {}
    for fn in tqdm(os.listdir(args.cands_dir), desc="Building forests"):
        if not fn.endswith(".cpp") and not fn.endswith(".c"):  # c も許容
            continue
        path = os.path.join(args.cands_dir, fn)
        forest = file_to_forest_cpp(path, blocks_fn, tree_to_index)
        if forest:
            id2forest[fn] = forest

    # 5) judged の読み込み
    pairs = load_judged(args.judged)
    usable = [(u, v, lab) for (u, v, lab) in pairs if (u in id2forest and v in id2forest)]
    skipped = len(pairs) - len(usable)
    if skipped:
        print(f"[warn] skipped {skipped} pairs (parse fail / file not found)")

    # 6) 推論
    scores, labels, rows = [], [], []
    for s, (u, v, lab) in tqdm(score_pairs(model, id2forest, usable, args.batch_size, device),
                               total=len(usable), desc="Scoring"):
        scores.append(s); labels.append(lab)
        rows.append((u, v, s, lab))

    # 7) PRとベスト閾値（参考）
    (P, R, F, T), best_local = pr_curve(scores, labels)

    # 8) 固定閾値（推奨：学習の best-F1 閾値）で確定指標
    thr = args.threshold if args.threshold is not None else best_local["threshold"]
    pred = [1 if s >= thr else 0 for s in scores]
    p, r, f, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    cm = confusion_matrix(labels, pred).tolist()

    # 9) 保存
    pd.DataFrame(rows, columns=["u", "v", "score", "label"]).to_csv(args.out_csv, index=False)
    pd.DataFrame({"threshold": T, "precision": P, "recall": R, "f1": F}).to_csv(args.save_pr, index=False)
    metrics = {
        "used_threshold": float(thr),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "confusion_matrix": {"tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1]},
        "num_pairs_total": int(len(pairs)),
        "num_pairs_scored": int(len(usable)),
        "skipped_pairs": int(skipped),
        "best_on_mysql": best_local  # 参考
    }
    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
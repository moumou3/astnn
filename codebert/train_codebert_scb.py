#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[EN]
Train/evaluate CodeBERT for clone detection using:
  - programs.pkl : DataFrame('id','code','label' optional)
  - oj_clone_ids.pkl : DataFrame('id1','id2','label')  # 1=clone, 0=non-clone

Modes
-----
  --mode biencoder    : contrastive fine-tuning (for dense retrieval)
  --mode crossencoder : pair classification (high-accuracy classifier)

Evaluator
---------
  - On validation, search the smallest τ* that satisfies P >= target_precision.
  - On test/transfer, keep τ* fixed and report P/R/F1 at τ* and PR-AUC.

Saves
-----
  - out_dir/encoder_best.pt   (bi-encoder)  or  out_dir/clf_best.pt (cross-encoder)
  - out_dir/valid_metrics.json  (includes tau_star)
  - out_dir/test_metrics.json   (if available)


# 1) Preparation: create training data from Stand_Alone_Clones
#    (use the provided build_programs_and_pairs script)
pipenv run python \
  ../HySCU/create_dataset/scripts/make_programs_and_pairs.py \
  --src ~/HySCU/semanticclonebench/Python/Stand_Alone_Clones/ \
  --out-dir codebert/data/semanticclonebench/python \
  --make-pairs --neg-per-pos 4

# Outputs: data_c/programs.pkl, data_c/oj_clone_ids.pkl

# 2) Training (bi-encoder; does an internal 9:1 split)
python train_codebert_scb.py \
  --prog_pkl data_c/programs.pkl \
  --pairs_train_pkl data_c/oj_clone_ids.pkl \
  --out_dir out_bi_c \
  --mode biencoder \
  --epochs 3 --batch_size 64 --lr 2e-5 --max_len 256 \
  --valid_ratio 0.1

# 3) Training (cross-encoder; if you provide an externally prepared 9:1 split)
python train_codebert_scb.py \
  --prog_pkl data_all/programs_all.pkl \
  --pairs_train_pkl data_all/pairs_all.pkl \
  --out_dir out_multi_bi \
  --mode crossencoder \
  --epochs 3 --batch_size 64 --lr 2e-5 --max_len 256 \
  --valid_ratio 0.1
"""
"""
[JA]
Train/evaluate CodeBERT for clone detection using
  - programs.pkl : DataFrame('id','code','label(任意)')
  - oj_clone_ids.pkl : DataFrame('id1','id2','label')  # 1=clone, 0=non-clone

Modes:
  --mode biencoder    : contrastive fine-tuning (dense retrieval向け)
  --mode crossencoder : pair classification (精密判定器)

Evaluator:
  - Validation で P>=target_precision を満たす最小 τ* を探索
  - Test/転用 でも τ* を固定して P/R/F1（τ*） と PR-AUC を出力
Saves:
  - out_dir/encoder_best.pt  or  out_dir/clf_best.pt
  - out_dir/valid_metrics.json （tau_star を含む）
  - out_dir/test_metrics.json  （あれば）


  # 1) 事前: Stand_Alone_Clones から学習データを作成
#   （ユーザ提示スクリプト build_programs_and_pairs を利用）
 pipenv run python 
 ../HySCU/create_dataset/scripts/make_programs_and_pairs.py 
 --src ~/HySCU/semanticclonebench/Python/Stand_Alone_Clones/ 
 --out-dir codebert/data/semanticclonebench/python 
 --make-pairs --neg-per-pos 4

# 出力: data_c/programs.pkl, data_c/oj_clone_ids.pkl

# 2) 学習（bi-encoder; 内部で 9:1 分割）
python train_codebert_scb.py \
  --prog_pkl data_c/programs.pkl \
  --pairs_train_pkl data_c/oj_clone_ids.pkl \
  --out_dir out_bi_c \
  --mode biencoder \
  --epochs 3 --batch_size 64 --lr 2e-5 --max_len 256 \
  --valid_ratio 0.1

# 3) 学習（cross-encoder; 9:1 分割を外部で作って渡す場合）
python train_codebert_scb.py \
  --prog_pkl data_all/programs_all.pkl \
  --pairs_train_pkl data_all/pairs_all.pkl \
  --out_dir out_multi_bi \
  --mode crossencoder \
  --epochs 3 --batch_size 64 --lr 2e-5 --max_len 256 \
  --valid_ratio 0.1


"""

import os, json, math, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
# schedule は v5 でも OK（モジュール名経由が無難）
from transformers.optimization import get_linear_schedule_with_warmup
# Optimizer は torch 側から
from torch.optim import AdamW

# -------------------- utils --------------------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    # 他の未知型は TypeError のままでOK
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def mean_pool(last_hidden_state: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, T, D], attn_mask: [B, T]
    mask = attn_mask.unsqueeze(-1)                        # [B,T,1]
    vec = (last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
    return F.normalize(vec, dim=-1)

def compute_pr(labels: np.ndarray, scores: np.ndarray):
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((labels == 1).sum()), 1)
    thresholds = scores[order]
    return precision, recall, thresholds

def pr_auc(precision: np.ndarray, recall: np.ndarray) -> float:
    area = 0.0
    for i in range(1, len(precision)):
        area += (precision[i] + precision[i-1]) * (recall[i] - recall[i-1]) / 2.0
    return float(abs(area))

def find_tau_for_precision(precision: np.ndarray, thresholds: np.ndarray, target_p: float) -> float:
    idx = np.where(precision >= target_p)[0]
    if len(idx) == 0:
        return float(np.max(thresholds))  # 達成不能→最大τ（最も厳しい）
    return float(thresholds[idx[0]])

def evaluate_pairs(labels: np.ndarray, scores: np.ndarray, target_p: float = 0.90) -> Dict:
    prec, rec, thr = compute_pr(labels, scores)
    auc = pr_auc(prec, rec)
    tau_star = find_tau_for_precision(prec, thr, target_p)
    y_pred = (scores >= tau_star).astype(int)
    tp = int(((labels == 1) & (y_pred == 1)).sum())
    fp = int(((labels == 0) & (y_pred == 1)).sum())
    fn = int(((labels == 1) & (y_pred == 0)).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    return {
        "tau_star": tau_star, "precision_at_tau": P, "recall_at_tau": R, "f1_at_tau": F1,
        "pr_auc": auc, "tp": tp, "fp": fp, "fn": fn
    }
def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> Dict:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    s_sorted = s[order]; y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted == 1)
    pos_total = int((y == 1).sum())
    change_idx = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    best = {"tau_f1": float("nan"), "f1": -1.0, "precision": 0.0, "recall": 0.0,
            "tp": 0, "fp": 0, "fn": pos_total}
    for idx in change_idx:
        tp = int(tp_cum[idx]); pred_pos = idx + 1
        fp = pred_pos - tp; fn = pos_total - tp
        P = tp/(tp+fp) if (tp+fp)>0 else 0.0
        R = tp/pos_total if pos_total>0 else 0.0
        F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
        if F1 > best["f1"]:
            best.update({"tau_f1": float(s_sorted[idx]), "f1": F1,
                         "precision": P, "recall": R, "tp": tp, "fp": fp, "fn": fn})
    return best

def evaluate_max_f1(labels: np.ndarray, scores: np.ndarray) -> Dict:
    b = best_f1_threshold(labels, scores)
    prec, rec, _ = compute_pr(labels, scores)
    auc = pr_auc(prec, rec)
    return {"tau_f1": b["tau_f1"], "f1_max": b["f1"],
            "precision_at_tau_f1": b["precision"], "recall_at_tau_f1": b["recall"],
            "tp_at_tau_f1": b["tp"], "fp_at_tau_f1": b["fp"], "fn_at_tau_f1": b["fn"],
            "pr_auc": auc}

def eval_at_tau(labels: np.ndarray, scores: np.ndarray, tau: float) -> Dict:
    y = np.asarray(labels, dtype=int); s = np.asarray(scores, dtype=float)
    yhat = (s >= tau).astype(int)
    tp = int(((y==1)&(yhat==1)).sum()); fp = int(((y==0)&(yhat==1)).sum()); fn = int(((y==1)&(yhat==0)).sum())
    P = tp/max(tp+fp,1); R = tp/max(tp+fn,1); F1 = 2*P*R/max(P+R,1e-9)
    return {"precision": P, "recall": R, "f1": F1, "tp": tp, "fp": fp, "fn": fn}
# -------------------- loaders --------------------

def load_programs(prog_pkl: str) -> Dict[int, str]:
    df = pd.read_pickle(prog_pkl)
    if not {"id","code"}.issubset(df.columns):
        raise ValueError("programs.pkl は 'id','code' 列を含む必要があります")
    # id を int に寄せる（元が文字列でも安全化）
    ids = pd.to_numeric(df["id"], errors="coerce").fillna(-1).astype(int).tolist()
    codes = df["code"].astype(str).tolist()
    return {i:c for i,c in zip(ids, codes)}

def load_pairs(pkl_path: str) -> pd.DataFrame:
    df = pd.read_pickle(pkl_path)
    if not {"id1","id2","label"}.issubset(df.columns):
        raise ValueError("pairs pkl は 'id1','id2','label' を含む必要があります")
    # 規格化：id を int に
    df = df.copy()
    df["id1"] = pd.to_numeric(df["id1"], errors="coerce").fillna(-1).astype(int)
    df["id2"] = pd.to_numeric(df["id2"], errors="coerce").fillna(-1).astype(int)
    df["label"] = df["label"].astype(int)
    # (a,b)=(b,a) 重複を排除
    a = df["id1"].values; b = df["id2"].values
    key1 = np.minimum(a,b); key2 = np.maximum(a,b)
    df["pair_key"] = [f"{x}|||{y}" for x,y in zip(key1,key2)]
    df = df.drop_duplicates(subset=["pair_key"]).drop(columns=["pair_key"])
    return df.reset_index(drop=True)

def split_pairs(df: pd.DataFrame, valid_ratio=0.1, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    pos = df[df["label"]==1].sample(frac=1.0, random_state=seed)
    neg = df[df["label"]==0].sample(frac=1.0, random_state=seed)
    nvp = max(1, int(round(len(pos)*valid_ratio)))
    nvn = max(1, int(round(len(neg)*valid_ratio)))
    val = pd.concat([pos.iloc[:nvp], neg.iloc[:nvn]]).sample(frac=1.0, random_state=seed)
    trn = pd.concat([pos.iloc[nvp:], neg.iloc[nvn:]]).sample(frac=1.0, random_state=seed)
    return trn.reset_index(drop=True), val.reset_index(drop=True)

# -------------------- datasets --------------------

class PairIdDataset(Dataset):
    """id1,id2,label をプログラム辞書からコードへ解決して返す"""
    def __init__(self, df_pairs: pd.DataFrame, id2code: Dict[int,str]):
        self.ids1 = df_pairs["id1"].tolist()
        self.ids2 = df_pairs["id2"].tolist()
        self.labels = df_pairs["label"].astype(int).tolist()
        self.id2code = id2code

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        a_id = self.ids1[i]; b_id = self.ids2[i]
        try:
            a = self.id2code[a_id]; b = self.id2code[b_id]
        except KeyError:
            # 欠落IDは空に（極力起きない前提）
            a = self.id2code.get(a_id, ""); b = self.id2code.get(b_id, "")
        return {"code_a": a, "code_b": b, "label": self.labels[i]}

@dataclass
class BiBatch:
    a: Dict[str, torch.Tensor]
    b: Dict[str, torch.Tensor]
    y: torch.Tensor

def make_bi_collate(tokenizer, max_len: int):
    def _fn(batch: List[Dict]):
        a_texts = [x["code_a"] for x in batch]
        b_texts = [x["code_b"] for x in batch]
        y = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        a = tokenizer(a_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        b = tokenizer(b_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        return BiBatch(a=a, b=b, y=y)
    return _fn

def make_cross_collate(tokenizer, max_len: int):
    def _fn(batch: List[Dict]):
        pair_texts = [a["code_a"] + tokenizer.sep_token + a["code_b"] for a in batch]
        x = tokenizer(pair_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        x["labels"] = torch.tensor([a["label"] for a in batch], dtype=torch.long)
        return x
    return _fn

# -------------------- training loops --------------------

def train_biencoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    enc = AutoModel.from_pretrained(args.model_id).to(device)

    id2code = load_programs(args.prog_pkl)

    # pairs 読み込みと分割
    if args.pairs_valid_pkl:
        df_train = load_pairs(args.pairs_train_pkl)
        df_valid = load_pairs(args.pairs_valid_pkl)
    else:
        df_all = load_pairs(args.pairs_train_pkl)
        df_train, df_valid = split_pairs(df_all, valid_ratio=args.valid_ratio, seed=args.seed)

    tr_ds = PairIdDataset(df_train, id2code)
    va_ds = PairIdDataset(df_valid, id2code)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=make_bi_collate(tok, args.max_len), num_workers=2, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.eval_batch_size, shuffle=False,
                           collate_fn=make_bi_collate(tok, args.max_len), num_workers=2, pin_memory=True)

    opt = AdamW(enc.parameters(), lr=args.lr)
    total_steps = len(tr_loader) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)

    def _embed(batch: BiBatch, train: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        a = {k: v.to(device) for k,v in batch.a.items()}
        b = {k: v.to(device) for k,v in batch.b.items()}
        if train: enc.train()
        else: enc.eval()
        with torch.set_grad_enabled(train):
            za = mean_pool(enc(**a).last_hidden_state, a["attention_mask"])
            zb = mean_pool(enc(**b).last_hidden_state, b["attention_mask"])
        return za, zb

    def _score_all(ds: PairIdDataset) -> Tuple[np.ndarray, np.ndarray]:
        enc.eval()
        scores, labels = [], []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False,
                                    collate_fn=make_bi_collate(tok, args.max_len), num_workers=2):
                za, zb = _embed(batch, False)
                s = (za @ zb.T).diag()     # ペアのコサイン（L2済み）
                scores.append(s.cpu()); labels.append(batch.y)
        return torch.cat(labels).numpy(), torch.cat(scores).numpy()

    best_auc = -1.0
    os.makedirs(args.out_dir, exist_ok=True)
    temperature = args.temperature

    for ep in range(1, args.epochs+1):
        enc.train(); tot = 0.0
        for step, batch in enumerate(tr_loader, 1):
            za, zb = _embed(batch, True)
            logits = (za @ zb.T) / temperature
            labels = torch.arange(logits.size(0), device=logits.device)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step(); sch.step()
            tot += loss.item()
            if step % args.log_every == 0:
                print(f"[ep{ep}/{args.epochs} step{step}/{len(tr_loader)}] loss={tot/args.log_every:.4f}")
                tot = 0.0

        y_val, s_val = _score_all(va_ds)
        ev = evaluate_pairs(y_val, s_val, target_p=args.target_precision)   # 既存
        ev_f1 = evaluate_max_f1(y_val, s_val)                               # ← 追加
        print(f"[ep{ep}] val PR-AUC={ev['pr_auc']:.4f} "
            f"τ*={ev['tau_star']:.4f} P={ev['precision_at_tau']:.3f} R={ev['recall_at_tau']:.3f} F1={ev['f1_at_tau']:.3f} | "
            f"τ_F1={ev_f1['tau_f1']:.4f} F1max={ev_f1['f1_max']:.3f} P={ev_f1['precision_at_tau_f1']:.3f} R={ev_f1['recall_at_tau_f1']:.3f}")


        if ev["pr_auc"] > best_auc:
            best_auc = ev["pr_auc"]
            torch.save(enc.state_dict(), os.path.join(args.out_dir, "encoder_best.pt"))
            with open(os.path.join(args.out_dir, "valid_metrics.json"), "w") as f:
                out = {
                    **ev,
                    "tau_f1": ev_f1["tau_f1"],
                    "f1_max": ev_f1["f1_max"],
                    "precision_at_tau_f1": ev_f1["precision_at_tau_f1"],
                    "recall_at_tau_f1": ev_f1["recall_at_tau_f1"],
                    "tp_at_tau_f1": ev_f1["tp_at_tau_f1"],
                    "fp_at_tau_f1": ev_f1["fp_at_tau_f1"],
                    "fn_at_tau_f1": ev_f1["fn_at_tau_f1"]
                }
                json.dump(out, f, indent=2, default=_json_default)

    # test（任意）
    if args.pairs_test_pkl:
        df_test = load_pairs(args.pairs_test_pkl)
        te_ds = PairIdDataset(df_test, id2code)
        y_t, s_t = _score_all(te_ds)
        te_star = evaluate_pairs(y_t, s_t, target_p=args.target_precision)   # τ*
        # valid の τ_F1 を転用して評価
        tau_f1 = json.load(open(os.path.join(args.out_dir, "valid_metrics.json")))["tau_f1"]
        te_f1  = eval_at_tau(y_t, s_t, tau=tau_f1)
        # PR-AUC も付ける
        prec, rec, _ = compute_pr(y_t, s_t); ap = pr_auc(prec, rec)
        print(f"[TEST] PR-AUC={ap:.4f}  τ*={te_star['tau_star']:.4f} "
              f"P={te_star['precision_at_tau']:.3f} R={te_star['recall_at_tau']:.3f} F1={te_star['f1_at_tau']:.3f} | "
              f"τ_F1={tau_f1:.4f} F1={te_f1['f1']:.3f} P={te_f1['precision']:.3f} R={te_f1['recall']:.3f}")
        with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
            out = {
                "pr_auc": ap,
                **te_star,
                "tau_f1": tau_f1,
                "precision_at_tau_f1": te_f1["precision"],
                "recall_at_tau_f1": te_f1["recall"],
                "f1_at_tau_f1": te_f1["f1"],
                "tp_at_tau_f1": te_f1["tp"],
                "fp_at_tau_f1": te_f1["fp"],
                "fn_at_tau_f1": te_f1["fn"]
            }
            json.dump(out, f, indent=2, default=_json_default)

def train_crossencoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    clf = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=2).to(device)

    id2code = load_programs(args.prog_pkl)

    # pairs 読み込みと分割
    if args.pairs_valid_pkl:
        df_train = load_pairs(args.pairs_train_pkl)
        df_valid = load_pairs(args.pairs_valid_pkl)
    else:
        df_all = load_pairs(args.pairs_train_pkl)
        df_train, df_valid = split_pairs(df_all, valid_ratio=args.valid_ratio, seed=args.seed)

    tr_ds = PairIdDataset(df_train, id2code)
    va_ds = PairIdDataset(df_valid, id2code)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=make_cross_collate(tok, args.max_len), num_workers=2, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.eval_batch_size, shuffle=False,
                           collate_fn=make_cross_collate(tok, args.max_len), num_workers=2, pin_memory=True)

    opt = AdamW(clf.parameters(), lr=args.lr)
    total_steps = len(tr_loader) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)

    def _score_all(ds: PairIdDataset) -> Tuple[np.ndarray, np.ndarray]:
        clf.eval()
        scores, labels = [], []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False,
                                    collate_fn=make_cross_collate(tok, args.max_len), num_workers=2):
                x = {k: v.to(device) for k,v in batch.items() if k != "labels"}
                logits = clf(**x).logits
                prob = torch.softmax(logits, dim=-1)[:,1].cpu().numpy()
                scores.append(prob); labels.append(batch["labels"].numpy())
        return np.concatenate(labels), np.concatenate(scores)

    best_auc = -1.0
    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(1, args.epochs+1):
        clf.train(); tot = 0.0
        for step, batch in enumerate(tr_loader, 1):
            x = {k: v.to(device) for k,v in batch.items()}
            out = clf(**x)
            loss = out.loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step(); sch.step()
            tot += loss.item()
            if step % args.log_every == 0:
                print(f"[ep{ep}/{args.epochs} step{step}/{len(tr_loader)}] loss={tot/args.log_every:.4f}")
                tot = 0.0

        y_val, s_val = _score_all(va_ds)
        ev = evaluate_pairs(y_val, s_val, target_p=args.target_precision)
        ev_f1 = evaluate_max_f1(y_val, s_val)
        print(f"[ep{ep}] val PR-AUC={ev['pr_auc']:.4f} "
              f"τ*={ev['tau_star']:.4f} P={ev['precision_at_tau']:.3f} R={ev['recall_at_tau']:.3f} F1={ev['f1_at_tau']:.3f} | "
              f"τ_F1={ev_f1['tau_f1']:.4f} F1max={ev_f1['f1_max']:.3f} P={ev_f1['precision_at_tau_f1']:.3f} R={ev_f1['recall_at_tau_f1']:.3f}")

        if ev["pr_auc"] > best_auc:
            best_auc = ev["pr_auc"]
            torch.save(clf.state_dict(), os.path.join(args.out_dir, "clf_best.pt"))
            with open(os.path.join(args.out_dir, "valid_metrics.json"), "w") as f:
                out = {
                    **ev,
                    "tau_f1": ev_f1["tau_f1"],
                    "f1_max": ev_f1["f1_max"],
                    "precision_at_tau_f1": ev_f1["precision_at_tau_f1"],
                    "recall_at_tau_f1": ev_f1["recall_at_tau_f1"],
                    "tp_at_tau_f1": ev_f1["tp_at_tau_f1"],
                    "fp_at_tau_f1": ev_f1["fp_at_tau_f1"],
                    "fn_at_tau_f1": ev_f1["fn_at_tau_f1"]
                }
                json.dump(out, f, indent=2, default=_json_default)

    # test（任意）
    if args.pairs_test_pkl:
        df_test = load_pairs(args.pairs_test_pkl)
        te_ds = PairIdDataset(df_test, id2code)
        y_t, s_t = _score_all(te_ds)
        te_star = evaluate_pairs(y_t, s_t, target_p=args.target_precision)
        tau_f1 = json.load(open(os.path.join(args.out_dir, "valid_metrics.json")))["tau_f1"]
        te_f1  = eval_at_tau(y_t, s_t, tau=tau_f1)
        prec, rec, _ = compute_pr(y_t, s_t); ap = pr_auc(prec, rec)
        print(f"[TEST] PR-AUC={ap:.4f}  τ*={te_star['tau_star']:.4f} "
              f"P={te_star['precision_at_tau']:.3f} R={te_star['recall_at_tau']:.3f} F1={te_star['f1_at_tau']:.3f} | "
              f"τ_F1={tau_f1:.4f} F1={te_f1['f1']:.3f} P={te_f1['precision']:.3f} R={te_f1['recall']:.3f}")
        with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
            out = {
                "pr_auc": ap,
                **te_star,
                "tau_f1": tau_f1,
                "precision_at_tau_f1": te_f1["precision"],
                "recall_at_tau_f1": te_f1["recall"],
                "f1_at_tau_f1": te_f1["f1"],
                "tp_at_tau_f1": te_f1["tp"],
                "fp_at_tau_f1": te_f1["fp"],
                "fn_at_tau_f1": te_f1["fn"]
            }
            json.dump(out, f, indent=2, default=_json_default)



# -------------------- CLI --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    # 必須：programs.pkl と ペア（train）
    ap.add_argument("--prog_pkl", required=True, help="programs.pkl のパス")
    ap.add_argument("--pairs_train_pkl", required=True, help="oj_clone_ids.pkl など（train または all）")
    # 任意：valid/test のペア
    ap.add_argument("--pairs_valid_pkl", default="", help="検証ペア（未指定なら内部で9:1分割）")
    ap.add_argument("--pairs_test_pkl",  default="", help="テストペア（任意）")
    # 学習設定
    ap.add_argument("--mode", choices=["biencoder","crossencoder"], default="biencoder")
    ap.add_argument("--model_id", default="microsoft/codebert-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.05)  # biencoder
    ap.add_argument("--target_precision", type=float, default=0.90)
    ap.add_argument("--valid_ratio", type=float, default=0.10, help="pairs_valid_pkl未指定時の内部分割比")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    if args.mode == "biencoder":
        train_biencoder(args)
    else:
        train_crossencoder(args)
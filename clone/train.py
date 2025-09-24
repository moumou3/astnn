# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable  # 互換目的（未使用でも可）
from sklearn.metrics import precision_recall_fscore_support
from gensim.models import Word2Vec
from config import *
from model import BatchProgramCC


def get_batch(dataset: pd.DataFrame, idx: int, bs: int):
    """
    blocks.pkl（id1,id2,label,code_x,code_y）からバッチを切り出す。
    """
    tmp = dataset.iloc[idx: idx + bs]
    x1, x2, labels, id1s, id2s = [], [], [], [], []
    for _, item in tmp.iterrows():
        x1.append(item["code_x"])
        x2.append(item["code_y"])
        labels.append([int(item["label"])])
        id1s.append(int(item["id1"]))
        id2s.append(int(item["id2"]))
    return x1, x2, torch.FloatTensor(labels), id1s, id2s


def load_embeddings(lang_root: str, dim: int):
    """
    学習時に作成した gensim4 の Word2Vec をロードして埋め込み行列を作る。
    """
    wv = Word2Vec.load(os.path.join(lang_root, f"train/embedding/node_w2v_{dim}")).wv
    MAX_TOKENS = wv.vectors.shape[0]
    EMBEDDING_DIM = wv.vectors.shape[1]
    emb = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    emb[:MAX_TOKENS] = wv.vectors
    return emb, MAX_TOKENS, EMBEDDING_DIM


def run_epoch(model, data_df, optimizer, loss_fn, train_mode=True, save_scores_path=None):
    """
    1エポック分を実行。最後のミニバッチが BATCH_SIZE 未満でも安全に動作するよう、
    実バッチサイズに合わせて model.batch_size と hidden を毎バッチ再初期化する。
    """
    model.train() if train_mode else model.eval()

    device = getattr(model, "device", torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"))

    total_loss = 0.0
    total = 0
    i = 0
    predicts, trues = [], []
    score_rows = []

    while i < len(data_df):
        x1, x2, y, id1s, id2s = get_batch(data_df, i, model.batch_size)
        i += model.batch_size

        bs = y.size(0)
        y = y.to(device)

        # 実バッチに合わせて hidden を再作成
        model.batch_size = bs
        model.hidden = model.init_hidden()

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            out = model(x1, x2)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(x1, x2)
                loss = loss_fn(out, y)

        # 収集
        pred = (out.detach() > 0.5).to(torch.int32).view(-1).cpu().numpy()
        true = y.detach().to(torch.int32).view(-1).cpu().numpy()
        predicts.extend(pred.tolist())
        trues.extend(true.tolist())

        total += bs
        total_loss += float(loss.item()) * bs

        if save_scores_path is not None and not train_mode:
            scores = out.detach().squeeze().cpu().numpy()
            if isinstance(scores, (float, int)):  # bs == 1
                scores = [float(scores)]
            else:
                scores = [float(s) for s in scores.tolist()]
            for a, b, s in zip(id1s, id2s, scores):
                score_rows.append((int(a), int(b), float(s)))

    if len(predicts) == 0:
        p = r = f = 0.0
    else:
        p, r, f, _ = precision_recall_fscore_support(trues, predicts, average="binary", zero_division=0)

    avg_loss = total_loss / max(total, 1)

    # スコアCSV出力
    if save_scores_path is not None and not train_mode:
        df = pd.DataFrame(score_rows, columns=["id1", "id2", "score"])
        lo = df[["id1", "id2"]].min(axis=1)
        hi = df[["id1", "id2"]].max(axis=1)
        df["id1"], df["id2"] = lo, hi
        df = df.sort_values("score", ascending=False).drop_duplicates(subset=["id1", "id2"], keep="first")
        df.to_csv(save_scores_path, index=False)

    return avg_loss, p, r, f


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ASTNN clone detection training (gensim4)")
    parser.add_argument("--lang", required=True, choices=["c", "cpp", "java"])
    parser.add_argument("--save-scores", default="eval_scores.csv", help="write id1,id2,score on test set")
    parser.add_argument("--save-weights", default=None,
                        help="学習完了時のモデル重みを保存するパス（例: models/astnn_cpp.pt）")
    parser.add_argument("--save-best", action="store_true",
                        help="エポック中のベストF1も保存（_best付きファイル）")
    args = parser.parse_args()

    root = "data"
    lang = args.lang
    lang_root = os.path.join(root, lang)

    # デバイス
    device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")

    print("Train for", lang.upper())
    train_data = pd.read_pickle(os.path.join(lang_root, "train/blocks.pkl")).sample(frac=1, random_state=666)
    test_data = pd.read_pickle(os.path.join(lang_root, "test/blocks.pkl")).sample(frac=1, random_state=666)

    # Java: 1-vs-rest をタイプごとに回す（でも出力は常に binary）
    categories = 5 if lang == "java" else 1

    # embeddings
    embeddings, MAX_TOKENS, EMBEDDING_DIM = load_embeddings(lang_root, EMBEDDING_SIZE)

    # model
    model = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE, USE_GPU, embeddings)
    model = model.to(device)

    optimizer = torch.optim.Adamax(model.parameters())
    loss_fn = torch.nn.BCELoss()

    precision, recall, f1 = 0.0, 0.0, 0.0
    best_f1 = -1.0
    best_path = None

    for t in range(1, categories + 1):
        if lang == "java":
            # そのタイプ or 非クローンのみに絞る
            train_t = train_data[train_data["label"].isin([t, 0])].copy()
            test_t = test_data[test_data["label"].isin([t, 0])].copy()
            # ラベル >0 を 1 に潰す
            train_t.loc[train_t["label"] > 0, "label"] = 1
            test_t.loc[test_t["label"] > 0, "label"] = 1
        else:
            train_t, test_t = train_data, test_data

        # 各 t で epoch 学習
        model.batch_size = BATCH_SIZE
        for epoch in range(EPOCHS):
            start = time.time()
            tr_loss, tr_p, tr_r, tr_f = run_epoch(model, train_t, optimizer, loss_fn, train_mode=True)
            dur = time.time() - start
            print(f"[t={t} epoch={epoch+1}] loss={tr_loss:.4f} P={tr_p:.3f} R={tr_r:.3f} F1={tr_f:.3f} ({dur:.1f}s)")

            # ★ ベストF1保存（任意・学習データ上のF1基準）
            if args.save_best and tr_f > best_f1:
                best_f1 = tr_f
                os.makedirs("models", exist_ok=True)
                best_path = (
                    os.path.join("models", f"astnn_{lang}_best.pt")
                    if (t == 1) else os.path.join("models", f"astnn_{lang}_t{t}_best.pt")
                )
                torch.save(model.state_dict(), best_path)
                print(f"[best] saved: {best_path} (F1={best_f1:.3f})")

        # 評価
        print(f"Testing-{t}...")
        model.batch_size = BATCH_SIZE
        te_loss, p, r, f = run_epoch(
            model, test_t, optimizer, loss_fn,
            train_mode=False,
            save_scores_path=(args.save_scores if t == 1 else None)
        )

        if lang == "java":
            # 論文実装に近い重み（例）
            weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
            precision += weights[t] * p
            recall += weights[t] * r
            f1 += weights[t] * f
            print(f"Type-{t}: P={p:.3f} R={r:.3f} F1={f:.3f}")
        else:
            precision, recall, f1 = p, r, f

    print("Total testing results (P, R, F1): %.3f, %.3f, %.3f" % (precision, recall, f1))
    if args.save_scores:
        print(f"Saved test scores to: {args.save_scores}")

    if args.save_weights:
        os.makedirs(os.path.dirname(args.save_weights), exist_ok=True)
        torch.save(model.state_dict(), args.save_weights)
        print(f"Saved final weights to: {args.save_weights}")


if __name__ == "__main__":
    main()
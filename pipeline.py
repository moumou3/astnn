# -*- coding: utf-8 -*-
import os
import pandas as pd
from tqdm.auto import tqdm
from config import *

tqdm.pandas()


class Pipeline:
    """ASTNN (C) 前処理パイプライン：エラー行はスキップして継続

    Args:
        ratio (str): "8:1:1" のような train:dev:test の分割比
        root  (str): データルート（例: 'data/'）
    """

    def __init__(self, ratio, root: str):
        self.ratio = ratio
        self.root = root
        self.sources = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    # ───────────────────────────────────────────────────────────────────
    # 1) C ソースを pycparser で AST 化（失敗は None にして後で drop）
    # ───────────────────────────────────────────────────────────────────
    def get_parsed_source(self, input_file: str,
                        output_file: str = None) -> pd.DataFrame:
        """
        Parse C code using pycparser.
        失敗（ParseError 等）は None にして後で drop する。
        """
        input_file_path = os.path.join(self.root, input_file)
        if output_file is None:
            source = pd.read_pickle(input_file_path)
        else:
            from pycparser import c_parser
            parser = c_parser.CParser()
            source = pd.read_pickle(input_file_path)
            source.columns = ['id', 'code', 'label']

            # --- 追加: 擬似Cを軽く補正（if conditionA { → if (conditionA) { 等） ---
            import re
            def sanitize(code: str) -> str:
                if not isinstance(code, str):
                    return code
                s = code
                s = re.sub(r'\belif\b', 'else if', s)
                s = re.sub(r'\band\b', '&&', s)
                s = re.sub(r'\bor\b',  '||', s)
                s = re.sub(r'\b(if|while|switch)\s+([^\(\s][^\n{;]*)\s*\{',
                        lambda m: f"{m.group(1)} ({m.group(2)}) {{", s)
                # TRUE/FALSE → 1/0（前処理なしで通したい場合の保険）
                s = re.sub(r'\bTRUE\b',  '1', s)
                s = re.sub(r'\bFALSE\b', '0', s)
                return s

            # --- 追加: 例外を飲んで None を返す ---
            def safe_parse(code):
                try:
                    return parser.parse(sanitize(code))
                except Exception:
                    return None

            # ★ ここを置換：parser.parse → safe_parse
            source['code'] = source['code'].progress_apply(safe_parse)

            # 失敗を drop してログ出力
            bad = source['code'].isnull()
            if bad.any():
                dropped_ids = source.loc[bad, 'id']
                out_csv = os.path.join(self.root, 'parse_failed_ids.csv')
                try:
                    dropped_ids.to_csv(out_csv, index=False)
                except Exception:
                    pass
                print(f"[parse] kept: {len(source)-bad.sum()}, dropped: {bad.sum()}  (failed ids -> {out_csv})")
            source = source.loc[~bad].copy()

            source.to_pickle(os.path.join(self.root, output_file))
        self.sources = source
        return source
    # ───────────────────────────────────────────────────────────────────
    # 2) train/dev/test に分割
    # ───────────────────────────────────────────────────────────────────
    def split_data(self):
        data = self.sources
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def ensure_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        train_dir = os.path.join(self.root, 'train')
        dev_dir   = os.path.join(self.root, 'dev')
        test_dir  = os.path.join(self.root, 'test')
        for d in (train_dir, dev_dir, test_dir):
            ensure_dir(d)

        self.train_file_path = os.path.join(train_dir, 'train_.pkl')
        self.dev_file_path   = os.path.join(dev_dir,   'dev_.pkl')
        self.test_file_path  = os.path.join(test_dir,  'test_.pkl')

        train.to_pickle(self.train_file_path)
        dev.to_pickle(self.dev_file_path)
        test.to_pickle(self.test_file_path)

        print("[split] train/dev/test sizes:", len(train), len(dev), len(test))

    # ───────────────────────────────────────────────────────────────────
    # 3) 語彙作成 & Word2Vec 学習（壊れたシーケンスは空にして除外）
    # ───────────────────────────────────────────────────────────────────
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        src_pkl = input_file or self.train_file_path
        trees = pd.read_pickle(src_pkl)  # ['id','code'(AST),'label'] を期待

        emb_dir = os.path.join(self.root, 'train', 'embedding')
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir)

        from clone.prepare_data import get_sequences

        def trans_to_sequences(ast):
            try:
                seq = []
                get_sequences(ast, seq)
                return seq  # list[str]
            except Exception:
                return []   # 失敗は空列

        corpus_series = trees['code'].progress_apply(trans_to_sequences)
        # Word2Vec 学習には空列を除外
        corpus = [s for s in corpus_series.tolist() if s]
        if not corpus:
            raise RuntimeError("[embedding] No sequences to train Word2Vec (all failed).")

        # 参考用に “ノード列をスペース区切り文字列” で保存（任意）
        trees_txt = trees.copy()
        trees_txt['code'] = pd.Series([' '.join(s) for s in corpus_series])
        trees_txt.to_csv(os.path.join(self.root, 'train', 'programs_ns.tsv'), index=False)

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(
            corpus,
            size=size,              # gensim 3.x の引数名
            workers=16,
            sg=1,
            min_count=MIN_COUNT,
            max_final_vocab=VOCAB_SIZE
        )
        w2v.save(os.path.join(emb_dir, 'node_w2v_' + str(size)))
        print("[embedding] trained Word2Vec on %d sequences, dim=%d" % (len(corpus), size))

    # ───────────────────────────────────────────────────────────────────
    # 4) ブロック列（ID 森）へ変換（失敗ブロックはスキップ・空は除外）
    # ───────────────────────────────────────────────────────────────────
    def generate_block_seqs(self, data_path, part):
        from clone.prepare_data import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        wv = Word2Vec.load(
            os.path.join(self.root, 'train', 'embedding', 'node_w2v_' + str(self.size))
        ).wv
        vocab = wv.vocab                 # gensim 3.x
        max_token = wv.syn0.shape[0]     # OOV は最後の行を使う

        def tree_to_index(node):
            token = getattr(node, 'token', None)
            idx = vocab[token].index if (token in vocab) else max_token
            children = getattr(node, 'children', [])
            out = [idx]
            for ch in children:
                try:
                    out.append(tree_to_index(ch))
                except Exception:
                    # 子のどこかで壊れていてもブランチだけスキップ
                    continue
            return out

        def trans2seq(ast_root):
            blocks = []
            try:
                func(ast_root, blocks)
            except Exception:
                return []  # 抽出失敗は空
            forest = []
            for b in blocks:
                try:
                    forest.append(tree_to_index(b))
                except Exception:
                    continue
            return forest  # list of nested int-lists（ブロックごとの木）

        trees = pd.read_pickle(data_path)      # ['id','code','label']（code=AST）
        trees = trees.copy()
        trees['code'] = trees['code'].progress_apply(trans2seq)

        # 空フォレストを除外
        kept_mask = trees['code'].map(lambda x: isinstance(x, list) and len(x) > 0)
        dropped = (~kept_mask).sum()
        trees = trees[kept_mask]
        if dropped:
            print("[blocks:%s] dropped empty/failed functions: %d" % (part, dropped))
        print("[blocks:%s] kept functions: %d" % (part, len(trees)))

        out_dir = os.path.join(self.root, part)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, 'blocks.pkl')
        trees.to_pickle(out_path)

    # ───────────────────────────────────────────────────────────────────
    # 5) 実行フロー
    # ───────────────────────────────────────────────────────────────────
    def run(self):
        print('parse source code...')
        ast_pkl = os.path.join(self.root, 'ast.pkl')
        if os.path.exists(ast_pkl):
            self.get_parsed_source(input_file='ast.pkl')
        else:
            self.get_parsed_source(input_file='programs.pkl', output_file='ast.pkl')

        print('split data...')
        self.split_data()

        print('train word embedding...')
        self.dictionary_and_embedding(None, EMBEDDING_SIZE)

        print('generate block sequences...')
        self.generate_block_seqs(self.train_file_path, 'train')
        self.generate_block_seqs(self.dev_file_path,   'dev')
        self.generate_block_seqs(self.test_file_path,  'test')


if __name__ == "__main__":
    ppl = Pipeline(RATIO, 'data/')
    ppl.run()
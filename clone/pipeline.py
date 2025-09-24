# -*- coding: utf-8 -*-
import os
import sys
import re
import json
import pandas as pd
from tqdm.auto import tqdm
import click
from config import *

tqdm.pandas()


class Pipeline:
    """
    End-to-end preprocessor for ASTNN (clone detection).
      - C   : data/c/programs.pkl (id,code,label), data/c/oj_clone_ids.pkl (id1,id2,label)
      - C++ : data/cpp/programs.pkl (id,code,label), data/cpp/oj_clone_ids.pkl (id1,id2,label)
      - Java: data/java/bcb_funcs_all.tsv (id,code), data/java/bcb_pair_ids.pkl (id1,id2,label)
    Outputs:
      data/<lang>/{train,dev,test}/blocks.pkl  (id1,id2,label,code_x,code_y)
      data/<lang>/train/embedding/node_w2v_<EMBEDDING_SIZE>
      data/<lang>/ast.pkl  (id, code=[AST or raw code][,label])
    """

    def __init__(self, ratio: str, root: str, language: str):
        self.language = language.lower()
        assert self.language in ("c", "java", "cpp")
        self.ratio = ratio
        self.root = root.rstrip("/")
        self.sources = None     # DataFrame: id, code(AST or str)[,label]
        self.pairs = None       # DataFrame: id1,id2,label
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    # ────────────────────────────────────────────────────────────────
    # Parse -> AST/Raw  （C/Java: AST化、C++: raw保持 or 独自関数に委譲）
    # ────────────────────────────────────────────────────────────────
    def get_parsed_source(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        lang_dir = os.path.join(self.root, self.language)
        in_path = os.path.join(lang_dir, input_file)

        def sanitize_c_like(code: str) -> str:
            if not isinstance(code, str):
                return code
            s = code
            # よくある擬似記法の修正
            s = re.sub(r'\belif\b', 'else if', s)
            s = re.sub(r'\band\b', '&&', s)
            s = re.sub(r'\bor\b', '||',  s)
            s = re.sub(r'\b(if|while|switch)\s+([^\(\s][^\n{;]*)\s*\{',
                       lambda m: f"{m.group(1)} ({m.group(2)}) {{", s)
            s = re.sub(r'\bTRUE\b', '1', s)
            s = re.sub(r'\bFALSE\b','0', s)
            return s

        def report_and_filter(df: pd.DataFrame, parsed_col="code") -> pd.DataFrame:
            before = len(df)
            ok = df[df[parsed_col].notnull()].copy()
            dropped = before - len(ok)
            if dropped:
                drop_csv = os.path.join(lang_dir, "parse_failed_ids.csv")
                try:
                    df.loc[df[parsed_col].isnull(), "id"].to_csv(drop_csv, index=False)
                except Exception:
                    pass
                print(f"[parse] kept: {len(ok)}, dropped: {dropped}  (failed ids -> {drop_csv})")
            else:
                print(f"[parse] kept: {len(ok)}, dropped: 0")
            return ok

        if output_file is None:
            src = pd.read_pickle(in_path)
        else:
            if self.language == "c":
                from pycparser import c_parser
                parser = c_parser.CParser()
                raw = pd.read_pickle(in_path)
                raw.columns = ['id', 'code', 'label']

                def safe_parse(code):
                    try:
                        return parser.parse(sanitize_c_like(code))
                    except Exception:
                        return None

                raw['code'] = raw['code'].progress_apply(safe_parse)
                src = report_and_filter(raw, parsed_col="code")

            elif self.language == "java":
                import javalang
                raw = pd.read_csv(in_path, delimiter='\t', header=None, names=['id','code'])

                def parse_member(code):
                    try:
                        tokens = javalang.tokenizer.tokenize(code)
                        parser = javalang.parser.Parser(tokens)
                        return parser.parse_member_declaration()
                    except Exception:
                        return None

                raw['code'] = raw['code'].progress_apply(parse_member)
                src = report_and_filter(raw, parsed_col="code")

            else:  # cpp
                # C++ は既定で "生コードを保持"。後段で prepare_data_cpp があれば使う。
                raw = pd.read_pickle(in_path)
                raw.columns = ['id', 'code', 'label']
                raw['code'] = raw['code'].astype(str).map(sanitize_c_like)
                # 空やNaNは落とす
                raw.loc[~raw['code'].astype(bool), 'code'] = None
                src = report_and_filter(raw, parsed_col="code")

            out_path = os.path.join(lang_dir, output_file)
            src.to_pickle(out_path)

        self.sources = src.reset_index(drop=True)
        return self.sources

    # ────────────────────────────────────────────────────────────────
    # ペア読み込み
    # ────────────────────────────────────────────────────────────────
    def read_pairs(self, filename: str):
        path = os.path.join(self.root, self.language, filename)
        pairs = pd.read_pickle(path)
        for col in ("id1","id2"):
            pairs[col] = pairs[col].astype(int)
        if 'label' in pairs.columns:
            pairs['label'] = pairs['label'].astype(int)
        else:
            pairs['label'] = 1
        self.pairs = pairs
        return pairs

    # ────────────────────────────────────────────────────────────────
    # 分割
    # ────────────────────────────────────────────────────────────────
    def split_data(self):
        assert self.pairs is not None
        data = self.pairs.sample(frac=1, random_state=666).reset_index(drop=True)
        n = len(data)
        a, b, c = [int(r) for r in self.ratio.split(':')]
        i1 = int(a/(a+b+c)*n)
        i2 = int((a+b)/(a+b+c)*n)

        train = data.iloc[:i1]
        dev   = data.iloc[i1:i2]
        test  = data.iloc[i2:]

        def ensure_dir(p): 
            if not os.path.exists(p): os.makedirs(p)

        base = os.path.join(self.root, self.language)
        for sub in ("train","dev","test"):
            ensure_dir(os.path.join(base, sub))

        self.train_file_path = os.path.join(base, "train", "train_.pkl")
        self.dev_file_path   = os.path.join(base, "dev",   "dev_.pkl")
        self.test_file_path  = os.path.join(base, "test",  "test_.pkl")
        train.to_pickle(self.train_file_path)
        dev.to_pickle(self.dev_file_path)
        test.to_pickle(self.test_file_path)
        print(f"[split] train/dev/test sizes: {len(train)}/{len(dev)}/{len(test)}")

    # ────────────────────────────────────────────────────────────────
    # 語彙作成 & Word2Vec 学習（gensim 4）
    # ────────────────────────────────────────────────────────────────
    def dictionary_and_embedding(self, input_file: str, size: int):
        self.size = size
        src_pkl = input_file or self.train_file_path
        pairs = pd.read_pickle(src_pkl)

        # 学習に使う関数ID（AST/raw が存在するものだけ）
        train_ids = pd.concat([pairs['id1'], pairs['id2']], axis=0).astype(int).unique().tolist()
        avail = set(self.sources['id'].astype(int).tolist())
        train_ids = [i for i in train_ids if i in avail]
        if not train_ids:
            raise RuntimeError("No training ids overlap between pairs and parsed sources.")

        trees = self.sources.set_index('id', drop=False).loc[train_ids]

        # --- sequence extractor の用意 ---
        if self.language == "c":
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from prepare_data import get_sequences as seq_func
            def trans_to_sequences(ast):
                seq = []
                try:
                    seq_func(ast, seq)
                except Exception:
                    return []
                return seq

        elif self.language == "java":
            from utils import get_sequence as seq_func
            def trans_to_sequences(ast):
                seq = []
                try:
                    seq_func(ast, seq)
                except Exception:
                    return []
                return seq

        else:  # cpp
            # 1) prepare_data_cpp があれば使う
            seq_func_cpp = None
            try:
                from prepare_data_cpp import get_sequences_cpp as seq_func_cpp
            except Exception:
                seq_func_cpp = None

            TOKEN_SPLIT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|0x[0-9A-Fa-f]+|\d+|==|!=|<=|>=|->|::|&&|\|\||[{}()\[\];,.:+\-*/%<>!&|^=?]")
            def fallback_tokens(code: str):
                return TOKEN_SPLIT.findall(code)[:2048]

            if seq_func_cpp is not None:
                def trans_to_sequences(code_or_ast):
                    # cpp は sources['code'] に "raw code" を保持している想定
                    try:
                        seq = []
                        seq_func_cpp(code_or_ast, seq)  # 実装によっては戻り値方式でもOK
                        if isinstance(seq, list) and seq:
                            return seq
                        # 返り値方式の場合
                        out = seq_func_cpp(code_or_ast)
                        return out if out else []
                    except Exception:
                        return fallback_tokens(str(code_or_ast))
            else:
                # フォールバック：トークン列をそのままコーパスに
                def trans_to_sequences(code_or_ast):
                    return fallback_tokens(str(code_or_ast))

        corpus_series = trees['code'].progress_apply(trans_to_sequences)
        corpus = [s for s in corpus_series if s]  # remove empty
        if not corpus:
            raise RuntimeError("Empty corpus for Word2Vec.")

        # 保存（参考）
        lang_dir = os.path.join(self.root, self.language)
        ns_path = os.path.join(lang_dir, 'train', 'programs_ns.tsv')
        os.makedirs(os.path.dirname(ns_path), exist_ok=True)
        pd.DataFrame({'id': trees['id'], 'seq': [' '.join(s) for s in corpus_series]}).to_csv(ns_path, index=False)

        # Word2Vec（gensim 4）
        from gensim.models import Word2Vec
        w2v = Word2Vec(
            sentences=corpus,
            vector_size=size, sg=1, workers=16,
            min_count=MIN_COUNT,
            max_final_vocab=VOCAB_SIZE
        )
        emb_dir = os.path.join(lang_dir, 'train', 'embedding')
        os.makedirs(emb_dir, exist_ok=True)
        w2v.save(os.path.join(emb_dir, f'node_w2v_{size}'))
        print(f"[embedding] trained Word2Vec on {len(corpus)} sequences, dim={size}")

    # ────────────────────────────────────────────────────────────────
    # ブロック列（ID森）へ変換（gensim4）
    # ────────────────────────────────────────────────────────────────
    def generate_block_seqs(self):
        # --- ブロック抽出関数の決定 ---
        if self.language == "c":
            from prepare_data import get_blocks as blocks_func
        elif self.language == "java":
            from utils import get_blocks_v1 as blocks_func
        else:  # cpp
            blocks_func = None
            try:
                from prepare_data_cpp import get_blocks_cpp as blocks_func
            except Exception:
                blocks_func = None

        from gensim.models import Word2Vec
        lang_dir = os.path.join(self.root, self.language)
        wv = Word2Vec.load(os.path.join(lang_dir, 'train', 'embedding', f'node_w2v_{self.size}')).wv
        tok2id = wv.key_to_index
        V = wv.vectors.shape[0]  # OOV は V

        def tree_to_index(node):
            token = getattr(node, 'token', None)
            idx = tok2id.get(token, V)
            out = [idx]
            for ch in getattr(node, 'children', []):
                try:
                    out.append(tree_to_index(ch))
                except Exception:
                    continue
            return out

        # C++ フォールバック：簡易 Node 構造
        class _Node:
            __slots__ = ("token", "children")
            def __init__(self, token, children=None):
                self.token = token
                self.children = children or []

        TOKEN_SPLIT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|0x[0-9A-Fa-f]+|\d+|==|!=|<=|>=|->|::|&&|\|\||[{}()\[\];,.:+\-*/%<>!&|^=?]")
        def pseudo_blocks_cpp(code: str, cap=512):
            toks = TOKEN_SPLIT.findall(code)[:cap]
            return [_Node("FUNC", [_Node(t) for t in toks])]  # 1ブロックだけ

        def trans2forest_c(ast_root):
            blocks = []
            try:
                blocks_func(ast_root, blocks)
            except Exception:
                return []
            forest = []
            for b in blocks:
                try:
                    forest.append(tree_to_index(b))
                except Exception:
                    continue
            return forest

        def trans2forest_cpp(code_or_ast):
            # C++ は raw code -> blocks
            if blocks_func is not None:
                try:
                    blocks = blocks_func(code_or_ast)  # 実装に応じて戻り値形式を想定
                    if not isinstance(blocks, list):
                        return []
                except Exception:
                    blocks = None
            else:
                blocks = None
            if not blocks:
                blocks = pseudo_blocks_cpp(str(code_or_ast))
            forest = []
            for b in blocks:
                try:
                    forest.append(tree_to_index(b))
                except Exception:
                    continue
            return forest

        trees = self.sources.copy()
        if self.language in ("c", "java"):
            trees['code'] = trees['code'].progress_apply(trans2forest_c)
        else:
            trees['code'] = trees['code'].progress_apply(trans2forest_cpp)

        # 空は除外
        trees = trees[trees['code'].map(lambda x: isinstance(x, list) and len(x) > 0)]
        self.blocks = trees[['id','code']].reset_index(drop=True)
        print(f"[blocks] kept functions: {len(self.blocks)}")

    # ────────────────────────────────────────────────────────────────
    # ペアとマージして blocks.pkl を作成
    # ────────────────────────────────────────────────────────────────
    def merge(self, data_path: str, part: str):
        pairs = pd.read_pickle(data_path)
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)

        left  = pd.merge(pairs, self.blocks, how='left', left_on='id1', right_on='id').rename(columns={'code':'code_x'})
        right = pd.merge(left,  self.blocks, how='left', left_on='id2', right_on='id').rename(columns={'code':'code_y'})
        right.drop(columns=['id_x','id_y'], inplace=True)

        before = len(right)
        right.dropna(subset=['code_x','code_y'], inplace=True)
        after = len(right)
        print(f"[merge:{part}] pairs kept: {after}/{before} (dropped {before-after})")

        out_path = os.path.join(self.root, self.language, part, 'blocks.pkl')
        right.to_pickle(out_path)

    # ────────────────────────────────────────────────────────────────
    # 実行
    # ────────────────────────────────────────────────────────────────
    def run(self):
        print('parse source code...')
        lang_dir = os.path.join(self.root, self.language)
        if os.path.exists(os.path.join(lang_dir, 'ast.pkl')):
            self.get_parsed_source('ast.pkl')
        else:
            if self.language == 'c':
                self.get_parsed_source('programs.pkl', 'ast.pkl')
            elif self.language == 'cpp':
                # C++ は raw code を ast.pkl に保存（後段で処理）
                self.get_parsed_source('programs.pkl', 'ast.pkl')
            else:
                self.get_parsed_source('bcb_funcs_all.tsv', 'ast.pkl')

        print('read id pairs...')
        if self.language in ('c', 'cpp'):
            self.read_pairs('oj_clone_ids.pkl')
        else:
            self.read_pairs('bcb_pair_ids.pkl')

        print('split data...')
        self.split_data()

        print('train word embedding...')
        self.dictionary_and_embedding(None, EMBEDDING_SIZE)

        print('generate block sequences...')
        self.generate_block_seqs()

        print('merge pairs and blocks...')
        self.merge(self.train_file_path, 'train')
        self.merge(self.dev_file_path,   'dev')
        self.merge(self.test_file_path,  'test')


@click.command()
@click.option('--lang', required=True, type=str, help="Language for the code input ('c', 'cpp' or 'java')")
def main(lang):
    ppl = Pipeline(RATIO, 'data', str(lang))
    ppl.run()


if __name__ == "__main__":
    main()
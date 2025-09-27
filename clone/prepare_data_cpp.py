# -*- coding: utf-8 -*-
"""
[EN]
Production-grade C++ preprocessor for ASTNN powered by tree-sitter.

Public APIs:
  - get_sequences_cpp(code: str, out: list[str]) -> None
  - get_blocks_cpp(code: str) -> list[Node]   # Node is a simple tree node with .token and .children

Notes:
  * To stay consistent with the Word2Vec/vocabulary, tokens are based on “type name + normalized literal”:
    identifiers → 'id', numbers → 'num', string literals → 'str'.
  * Insert 'End' at the end of compound-statement–like constructs to follow the sequencing
    convention used by ASTNN (C version).
"""
"""
[JA]
Production-grade C++ preprocessor for ASTNN using tree-sitter.
提供API:
  - get_sequences_cpp(code: str, out: list[str]) -> None
  - get_blocks_cpp(code: str) -> list[Node]  (# Node は .token と .children を持つ簡易木)

注意:
  * Word2Vec/語彙と一貫するよう、トークンは「型名＋正規化リテラル」を基本にし、
    識別子は 'id'、数値は 'num'、文字列は 'str' に正規化します。
  * Compound 相当の終了時に 'End' を差し込み、ASTNN(C版)の系列性を踏襲します。
"""

from typing import List, Optional
import re

# --- tree-sitter setup ---
from tree_sitter_languages import get_parser
_PARSER = get_parser("cpp")

# --- minimal Node for pipeline ---
class Node:
    __slots__ = ("token", "children")
    def __init__(self, token: str, children: Optional[List["Node"]] = None):
        self.token = token
        self.children = children or []

# --- helpers ---
# C++ コメント除去（行末 // とブロック /* */ を簡易に除去）
_RE_MLC = re.compile(r"/\*.*?\*/", re.S)
_RE_SLC = re.compile(r"//[^\n]*")

def strip_comments(code: str) -> str:
    return _RE_SLC.sub("", _RE_MLC.sub("", code))

# 正規化: 識別子/リテラルを固定トークンに潰す（語彙爆発を防ぐ）
IDENT_TYPES = {"identifier", "qualified_identifier", "field_identifier", "namespace_identifier", "type_identifier"}
NUM_LITS    = {"number_literal"}
STR_LITS    = {"string_literal", "raw_string_literal", "char_literal"}

# Compound 相当（閉じ時に 'End' を差し込む対象）
COMPOUND_TYPES = {
    "compound_statement",
    # “{ ... }” と等価にブロック化される節（case/default はスルー）
}

CONTROL_TYPES = {
    "function_definition",
    "if_statement",
    "switch_statement",
    "for_statement",
    "while_statement",
    "do_statement",
    "try_statement",
}

# --- token 変換 ---
def _token_of(node, src: bytes) -> str:
    t = node.type
    if t in IDENT_TYPES:
        return "id"
    if t in NUM_LITS:
        return "num"
    if t in STR_LITS:
        return "str"
    # 演算子や区切りなどの punctuator は node.type が '>' '<' 等にはならないため、
    # tree-sitter の型名を採用（例: 'call_expression', 'return_statement' など）
    return t

# --- AST -> sequence -------------------------------------------------
def get_sequences_cpp(code: str, sequence: List[str], cap_nodes: int = 200000) -> None:
    """
    AST を DFS し、ノード型ベースの列へ。compound 終了時に 'End' を挿入。
    """
    if not isinstance(code, str):
        code = str(code or "")
    src = strip_comments(code).encode("utf-8", "ignore")
    tree = _PARSER.parse(src)
    root = tree.root_node

    stack = [(root, 0)]
    seen = 0
    while stack:
        node, state = stack.pop()
        if seen >= cap_nodes:
            break
        if state == 0:
            # pre-order: emit
            sequence.append(_token_of(node, src))
            seen += 1
            # descend
            stack.append((node, 1))
            # 子は逆順で push（pop順がソース順になる）
            for i in reversed(range(node.named_child_count)):
                stack.append((node.named_child(i), 0))
        else:
            # post-order: close compound -> 'End'
            if node.type in COMPOUND_TYPES:
                sequence.append("End")

# --- AST -> blocks (list[Node]) --------------------------------------
def _build_node_tree(ts_node, src: bytes, cap_nodes: int, counter) -> Optional[Node]:
    """
    tree-sitter ノードから ASTNN 用 Node 木を構築（cap_nodes で切り詰め）
    """
    if counter[0] >= cap_nodes:
        return None
    tok = _token_of(ts_node, src)
    n = Node(tok, [])
    counter[0] += 1
    for i in range(ts_node.named_child_count):
        if counter[0] >= cap_nodes:
            break
        child = ts_node.named_child(i)
        ch_node = _build_node_tree(child, src, cap_nodes, counter)
        if ch_node is not None:
            n.children.append(ch_node)
    # compound 終了マーカーも内容木に入れたい場合はここで Node("End") を追加しても良いが、
    # シーケンス側で 'End' を付ける方が自然なので木には付けない。
    return n

def get_blocks_cpp(code: str, cap_nodes_per_block: int = 50000) -> List[Node]:
    """
    制御構造/関数をブロック根として list[Node] を返す。
    """
    if not isinstance(code, str):
        code = str(code or "")
    src = strip_comments(code).encode("utf-8", "ignore")
    tree = _PARSER.parse(src)
    root = tree.root_node

    # 探索でブロック根を拾う
    blocks: List[Node] = []
    stack = [root]
    while stack:
        node = stack.pop()
        t = node.type
        # ブロック根か？
        if t in CONTROL_TYPES:
            counter = [0]
            n = _build_node_tree(node, src, cap_nodes_per_block, counter)
            if n is not None:
                blocks.append(n)
            # 関数等の内部は別途子巡回すると重複が増えるため、ここでは “根だけ” を追加して続行
        else:
            # 通常は子へ
            for i in reversed(range(node.named_child_count)):
                stack.append(node.named_child(i))

    # 1つも見つからない場合は、ファイル全体を1ブロックにフォールバック
    if not blocks:
        counter = [0]
        n = _build_node_tree(root, src, cap_nodes_per_block, counter)
        if n is not None:
            blocks.append(n)

    return blocks
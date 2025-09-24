# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchTreeEncoder(nn.Module):
    """
    ASTNN の木エンコーダ（GPU前提）
    - device は常に 'cuda'（CUDA 非利用なら上位でエラーにする）
    - -1 のダミーノードは無視
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encode_dim: int,
        batch_size: int,
        use_gpu: bool,
        pretrained_weight=None,
    ):
        super().__init__()

        # ★ GPU 前提チェック（ここで落とす）
        if not (use_gpu and torch.cuda.is_available()):
            raise RuntimeError(
                "USE_GPU=True ですが CUDA が利用できません。NVIDIA ドライバ/適切な PyTorch をインストールしてください。"
            )

        self.device = torch.device("cuda")

        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=self.device)
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim, device=self.device)
        self.activation = F.relu

        self.batch_size = batch_size
        self.node_list = []
        self.batch_node = None
        self.max_index = vocab_size
        self.stop = -1  # sentinel

        # 事前学習埋め込み（np.ndarray を想定）
        if pretrained_weight is not None:
            w = torch.as_tensor(pretrained_weight, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                self.embedding.weight.data.copy_(w)

    # ---- helpers -----------------------------------------------------

    def _zeros(self, *shape, dtype=torch.float32):
        return torch.zeros(*shape, dtype=dtype, device=self.device)

    def _long(self, data):
        return torch.tensor(data, dtype=torch.long, device=self.device)

    # ---- core --------------------------------------------------------

    def traverse_mul(self, node, batch_index):
        """
        node: list[ [token_id, child1, child2, ...], ... ]  (len = size)
        batch_index: list[int]（各要素が 0..batch-1）
        return: Tensor [size, encode_dim]  （size = len(node)）
        """
        size = len(node)
        if size == 0:
            return None

        batch_current = self._zeros(size, self.embedding_dim)

        index, children_index = [], []
        current_node, children = [], []

        for i in range(size):
            # -1 はダミー
            if node[i][0] != -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] != -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])

        if index:
            idx_t = self._long(index)
            cur_toks = self._long(current_node)
            emb = self.embedding(cur_toks)  # [len(index), emb_dim]
            batch_current = self.W_c(batch_current.index_copy(0, idx_t, emb))  # [size, encode_dim]

        # 子を畳み込み
        for c in range(len(children)):
            zeros = self._zeros(size, self.encode_dim)
            child_idx_t = self._long(children_index[c])
            tree = self.traverse_mul(children[c], [batch_index[i] for i in children_index[c]])
            if tree is not None:
                batch_current = batch_current + zeros.index_copy(0, child_idx_t, tree)

        # 各ステップの表現を蓄積（max-pool 用）
        b_in = self._long(batch_index)
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs: int):
        """
        x: list of trees（長さ = 合計ブロック数）
        bs: この forward 呼び出しにおける関数数（= バッチサイズ）
        """
        self.batch_size = bs
        self.batch_node = self._zeros(self.batch_size, self.encode_dim)
        self.node_list = []

        self.traverse_mul(x, list(range(self.batch_size)))

        self.node_list = torch.stack(self.node_list)  # [depth, batch, encode_dim]
        # depth 次元に対して max-pool
        return torch.max(self.node_list, dim=0)[0]  # [batch, encode_dim]


class BatchProgramCC(nn.Module):
    """
    ASTNN のコードクローン分類器（GPU前提）
      - BiGRU + max-pool で関数ベクトル化
      - |L-R| → 線形 → sigmoid
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        encode_dim: int,
        label_size: int,
        batch_size: int,
        use_gpu: bool = True,
        pretrained_weight=None,
    ):
        super().__init__()

        # ★ GPU 前提チェック
        if not (use_gpu and torch.cuda.is_available()):
            raise RuntimeError(
                "USE_GPU=True ですが CUDA が利用できません。NVIDIA ドライバ/適切な PyTorch をインストールしてください。"
            )
        self.device = torch.device("cuda")

        self.stop = [vocab_size - 1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size

        # 木エンコーダ（内部で device='cuda' に乗る）
        self.encoder = BatchTreeEncoder(
            self.vocab_size,
            self.embedding_dim,
            self.encode_dim,
            self.batch_size,
            use_gpu=True,
            pretrained_weight=pretrained_weight,
        )

        # BiGRU（ブロック系列 -> 関数ベクトル）
        self.bigru = nn.GRU(
            self.encode_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            device=self.device,
        )

        # 出力層
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size, device=self.device)
        self.dropout = nn.Dropout(0.2)

        # 初期 hidden
        self.hidden = self.init_hidden()

    # ---- helpers -----------------------------------------------------

    def init_hidden(self):
        """
        双方向 GRU → num_layers * 2
        train.py 側で各バッチ前に model.hidden = model.init_hidden() を呼ぶ前提
        """
        h = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim, device=self.device)
        return h

    def _zeros(self, num):
        return torch.zeros(num, self.encode_dim, device=self.device)

    # ---- encode ------------------------------------------------------

    def encode(self, x):
        """
        x: list of list[block_tree] （len(x) が実バッチサイズ）
        """
        bs = len(x)                               # ← 実バッチサイズ
        lens = [len(item) for item in x]
        max_len = max(lens) if lens else 0

        # 必要なら hidden を実バッチに合わせて再初期化
        if getattr(self, "hidden", None) is None or self.hidden.size(1) != bs:
            self.batch_size = bs
            self.hidden = self.init_hidden()

        # すべてのブロックを縦に連結して encoder へ
        encodes_in = []
        for i in range(bs):                       # ← self.batch_size ではなく bs
            for j in range(lens[i]):
                encodes_in.append(x[i][j])

        # sum(lens) 個のブロックを1度に符号化（元の設計のまま）
        encodes = self.encoder(encodes_in, sum(lens))  # [sum(lens), encode_dim]

        # バッチに切り戻し & パディング
        seq, start = [], 0
        for i in range(bs):                      # ← self.batch_size ではなく bs
            end = start + lens[i]
            seq.append(encodes[start:end])       # [lens[i], encode_dim]
            if max_len - lens[i] > 0:
                seq.append(self._zeros(max_len - lens[i]))
            start = end

        if max_len == 0:
            # 全サンプルが空（あり得ない想定だが保険）
            return torch.zeros(bs, self.hidden_dim * 2, device=self.device)

        # [batch, max_len, encode_dim]
        encodes = torch.cat(seq, dim=0).view(bs, max_len, -1)

        # pack（lengths は CPU テンソル）
        lengths_cpu = torch.tensor(lens, dtype=torch.long)
        packed = nn.utils.rnn.pack_padded_sequence(
            encodes, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        gru_out, _ = self.bigru(packed, self.hidden)     # hidden も bs に整合済み
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out, batch_first=True, padding_value=-1e9
        )  # [bs, max_len, 2*hidden]

        # 時系列 max-pool → [bs, 2*hidden]
        gru_out = torch.transpose(gru_out, 1, 2)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        return gru_out
    # ---- forward -----------------------------------------------------

    def forward(self, x1, x2):
        """
        x1, x2: バッチの左右コード（list[list[block_tree]]）
        """
        # train.py 側で毎バッチ:
        #   model.batch_size = len(batch)
        #   model.hidden = model.init_hidden()
        lvec, rvec = self.encode(x1), self.encode(x2)
        abs_dist = torch.abs(lvec - rvec)
        logits = self.hidden2label(abs_dist)
        y = torch.sigmoid(logits)
        return y
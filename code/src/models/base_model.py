"""FiFTy 系列モデル実装である。

- FiFTyModel: 1D-CNN バックボーン
- FiFTyLSTMModel: LSTM バックボーン
- FiFTyGRUModel: GRU バックボーン
- FiFTyTransformerModel: Transformer Encoder バックボーン
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def init_keras_like_(m: nn.Module) -> None:
    """
    FiFTy で用いた Keras 既定に合わせて初期化する。
      - Conv1d/Dense: glorot_uniform (Xavier uniform), bias zeros
      - Embedding: uniform(-0.05, 0.05)
    """
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, a=-0.05, b=0.05)

    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)  # glorot_uniform
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # glorot_uniform
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class LockedDropout(nn.Module):
    """時系列方向にマスクを固定するドロップアウト。"""

    def __init__(self, p: float, *, pad_idx: int | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("Dropout probability has to be in [0.0, 1.0).")
        self.p = float(p)
        self.pad_idx = pad_idx

    def forward(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (self.p == 0.0) or (not self.training):
            return x

        batch_size, _, feat_dim = x.size()
        if batch_size == 0:
            return x

        mask = x.new_empty(batch_size, 1, feat_dim).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p).expand_as(x)

        if (tokens is not None) and (self.pad_idx is not None):
            pad_mask = tokens.eq(self.pad_idx).unsqueeze(-1)
            if pad_mask.any():
                mask = mask.masked_fill(pad_mask, 1.0)

        return x * mask


class SinusoidalPositionalEncoding(nn.Module):
    """標準的な正弦波ポジショナルエンコーディング。"""

    def __init__(self, embed_dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"入力系列長 {seq_len} が Positional Encoding の最大長 "
                f"{self.pe.size(1)} を超えています"
            )
        return x + self.pe[:, :seq_len]


class _BaseRecurrentSequenceClassifier(nn.Module):
    """RNN 系列分類モデル向けの共通実装を提供する内部基底クラス。"""

    def __init__(
        self,
        n_classes: int,
        *,
        embed_dim: int,
        hidden: int,
        bidirectional: bool,
        dropout: float,
        embedding_dropout: float,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(256, embed_dim)
        # 本データセットは全トークンが実データであり、PAD は存在しない。
        self.embedding_dropout = LockedDropout(
            embedding_dropout,
            pad_idx=self.embed.padding_idx,
        )
        self.dropout = nn.Dropout(dropout)
        fc_in = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, n_classes)
        self.feature_norm = nn.LayerNorm(fc_in) if layer_norm else nn.Identity()
        self.apply(init_keras_like_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """入力 (B, T) をロジットに変換する。"""
        tokens = x.long()
        embedded = self.embed(tokens)
        embedded = self.embedding_dropout(embedded, tokens)
        features = self._forward_recurrent(embedded)
        features = self.feature_norm(features)
        features = self.dropout(features)
        return self.fc(features)

    def _forward_recurrent(self, embedded: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FiFTyGRUModel(_BaseRecurrentSequenceClassifier):
    """
    FiFTy 派生 GRU モデル（固定長 512byte）
      - 構造: Embedding → (多層)Bi‑GRU → Dropout → 全結合
      - GRU は LSTM よりパラメータと計算量が約 25 % 少なく高速
    """

    def __init__(
        self,
        n_classes: int,
        *,
        embed_dim: int = 64,
        hidden: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        embedding_dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__(
            n_classes,
            embed_dim=embed_dim,
            hidden=hidden,
            bidirectional=bidirectional,
            dropout=dropout,
            embedding_dropout=embedding_dropout,
            layer_norm=layer_norm,
        )

        # GRU 本体
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,  # (B, T, E) 形式
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def _forward_recurrent(self, embedded: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(embedded)
        if self.gru.bidirectional:
            return torch.cat((h_n[-2], h_n[-1]), dim=1)
        return h_n[-1]


class FiFTyLSTMModel(_BaseRecurrentSequenceClassifier):
    """FiFTy 派生 LSTM モデル（固定長 512byte）

    - 構造: Embedding → (多層)Bi‑LSTM → Dropout → 全結合 → Softmax(logits)
    - 入力:
        - バイト系列テンソル (B, T)
        - B=バッチサイズ
        - T=時系列長 (=512 Byte 固定)
    - 出力: クラス分類ロジット (B, n_classes)
    - パラメータ数は元 CNN と同程度 (~49 万) に抑えている
    - 時系列長が不変なので pack_padded_sequence は省略
    """

    def __init__(
        self,
        n_classes: int,
        embed_dim: int = 64,
        hidden: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        embedding_dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__(
            n_classes,
            embed_dim=embed_dim,
            hidden=hidden,
            bidirectional=bidirectional,
            dropout=dropout,
            embedding_dropout=embedding_dropout,
            layer_norm=layer_norm,
        )

        # LSTM 本体
        self.lstm = nn.LSTM(
            input_size=embed_dim,  # Embedding 出力次元
            hidden_size=hidden,  # 隠れ状態のユニット数
            num_layers=num_layers,  # スタック数
            batch_first=True,  # (B, T, E) 形式を前提
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def _forward_recurrent(self, embedded: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(embedded)
        if self.lstm.bidirectional:
            return torch.cat((h_n[-2], h_n[-1]), dim=1)
        return h_n[-1]


class FiFTyTransformerModel(nn.Module):
    """FiFTy 派生 Transformer モデル。"""

    def __init__(
        self,
        n_classes: int,
        *,
        embed_dim: int = 64,
        hidden: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        embedding_dropout: float = 0.0,
        max_seq_len: int = 512,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(256, embed_dim)
        self.embedding_dropout = LockedDropout(
            embedding_dropout,
            pad_idx=self.embed.padding_idx,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoding = SinusoidalPositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_seq_len + 1,  # 追加した CLS トークン分
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.feature_norm = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.apply(init_keras_like_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.long()
        embedded = self.embed(tokens)
        embedded = self.embedding_dropout(embedded, tokens)
        batch_size = embedded.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embedded = torch.cat((cls_tokens, embedded), dim=1)
        encoded = self.pos_encoding(embedded)
        encoded = self.encoder(encoded)
        features = encoded[:, 0, :]
        features = self.feature_norm(features)
        return self.head(features)


class FiFTyModel(nn.Module):
    """
    FiFTy 論文で提案された 1D-CNN 畳み込みニューラルネットワークの実装である。
    入力: バイト系列 (整数列)
    出力: クラス分類 (softmax logits)
    """

    def __init__(
        self,
        n_classes: int,
        embed_dim: int,  # 埋め込み次元数
        conv_channels: int,  # 1D畳み込み層の出力チャネル数
        hidden: int,  # 全結合中間層のユニット数
        kernel_size: int,  # 畳み込みカーネル幅
        pool_size: int,  # プーリングサイズ
        dropout: float,  # ドロップアウト率
        num_blocks: int,  # 畳み込みブロック数: Conv → ReLU → Pool を繰り返す回数
        embedding_dropout: float = 0.0,  # 埋め込み直後のロックドドロップアウト率
        layer_norm: bool = False,
    ) -> None:
        super().__init__()

        # バイト値（0〜255）を `embed_dim` 次元へ埋め込む。
        self.embed = nn.Embedding(256, embed_dim)
        # 全系列が実データで埋まっており、PAD トークンは存在しない。
        self.embedding_dropout = LockedDropout(
            embedding_dropout,
            pad_idx=self.embed.padding_idx,
        )

        # 可変深さの畳み込みブロックを `ModuleList` で保持する。
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        in_channels = embed_dim
        for _ in range(num_blocks):
            conv = nn.Conv1d(
                in_channels,
                conv_channels,
                kernel_size,
                stride=1,  # FiFTy の設計に合わせて stride=1 を用いる。
                padding=0,  # Keras 既定の `padding="valid"` に合わせる。
            )
            self.blocks.append(conv)
            in_channels = conv_channels

        # 活性化関数は FiFTy 設計に基づく LeakyReLU(α=0.3) とする。
        self.activation = nn.LeakyReLU(0.3, inplace=True)

        # 時系列長を `pool_size` 分縮小する。
        self.pool = nn.MaxPool1d(pool_size)

        if (dropout > 0.0) and (num_blocks > 1):
            self.block_dropout = nn.Dropout(dropout)
        else:
            self.block_dropout = nn.Identity()

        # 時系列方向を平均化し長さ 1 に集約する。
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.pre_gap_layer_norm = (
            nn.LayerNorm(conv_channels) if layer_norm else nn.Identity()
        )

        # 過学習抑制のためドロップアウトを適用する。
        self.head_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # 畳み込み後チャネル `conv_channels` を `hidden` 次元へ写像する。
        self.fc1 = nn.Linear(conv_channels, hidden)

        # `hidden` 次元特徴を `n_classes` ロジットへ写像する。
        self.fc2 = nn.Linear(hidden, n_classes)

        # Keras 既定に近い初期化を一括適用する。
        self.apply(init_keras_like_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播処理 (B, T) → (B, C)"""
        # `torchview` のダミー入力が float の場合に備えて long へ変換する。
        tokens = x.long()
        x = self.embed(tokens)  # (B, T, E): 埋め込み表現
        x = self.embedding_dropout(x, tokens)
        x = x.permute(0, 2, 1)  # Conv1d 用に (B, E, T) へ並べ替える。
        # 畳み込みブロックを順に適用する。
        for idx, conv in enumerate(self.blocks):
            x = conv(x)
            x = self.activation(x)  # LeakyReLU(0.3)
            if self.num_blocks > 1 and idx < (self.num_blocks - 1):
                x = self.block_dropout(x)
            x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = self.pre_gap_layer_norm(x)
        x = x.permute(0, 2, 1)
        x = self.gap(x).squeeze(-1)  # (B, C): Global Average Pooling 後の特徴。
        x = self.head_dropout(x)
        x = self.fc1(x)
        x = self.activation(x)  # LeakyReLU(0.3)
        return self.fc2(x)  # (B, n_classes) のロジット

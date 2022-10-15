import math

import torch
from torch import Tensor
from torch.nn import Dropout, Linear, Module


class MultiHeadAttention(Module):
    def __init__(
        self,
        embedding_size: int,
        heads: int,
        mask: Tensor,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
    ):
        super().__init__()
        assert embedding_size % heads == 0
        # key, query, value projections for all heads, but in a batch
        self.kqv = Linear(embedding_size, 3 * embedding_size)
        # output projection
        self.out_proj = Linear(embedding_size, embedding_size)
        # regularization
        self.attn_dropout = Dropout(attention_dropout)
        self.resid_dropout = Dropout(residual_dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))
        self.heads = heads
        self.embedding_size = embedding_size

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.kqv(x).split(self.embedding_size, dim=2)
        # (B, nh, T, hs)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T], float("-inf"))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.out_proj(y))
        return y

    @staticmethod
    def causal_attention(
        max_seq_len: int,
        embedding_size: int,
        heads: int,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
    ) -> "MultiHeadAttention":
        mask = MultiHeadAttention.causal_mask(max_seq_len)
        return MultiHeadAttention(
            embedding_size=embedding_size,
            heads=heads,
            mask=mask,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
        )

    @staticmethod
    def causal_mask(max_seq_len: int):
        return torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()


# from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

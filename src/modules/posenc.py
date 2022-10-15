import math
import torch
import torch.nn as nn

from torch import device as Device
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim,
        block_size,
        dtype=None,
        device: Optional[Device] = None,
    ):
        super(PositionalEncoding, self).__init__()

        self.block_size = block_size

        pe = torch.zeros(block_size, embed_dim, dtype=dtype, device=device)
        range_type = dtype or torch.float32
        position = torch.arange(
            0, block_size, dtype=range_type, device=device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=range_type, device=device)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # TODO support block dim first
        self.register_buffer("pe", pe)

    def forward(self, x):
        # each position maps to a (learnable) vector
        return self.pe

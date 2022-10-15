from torch.nn import GELU, Dropout, LayerNorm, Linear, Module, ModuleDict


class TransformerBlock(Module):
    def __init__(
        self, embedding_size: int, attention: Module, residual_dropout: float = 0.1
    ):
        super().__init__()
        self.ln_1 = LayerNorm(embedding_size)
        self.attention = attention
        self.ln_2 = LayerNorm(embedding_size)
        self.mlp = ModuleDict(
            dict(
                c_fc=Linear(embedding_size, 4 * embedding_size),
                c_proj=Linear(4 * embedding_size, embedding_size),
                act=GELU(),
                dropout=Dropout(residual_dropout),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

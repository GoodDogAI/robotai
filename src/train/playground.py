import json

import torch

from torch.nn import Embedding, LayerNorm, Linear, Sequential
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.modules import Sum, TransformerBlock
from src.modules.posenc import PositionalEncoding

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

from src.config import HOST_CONFIG, MODEL_CONFIGS
from src.datasets import ChunkDataset, ModularAdditionDataset
from src.msgvec.pymsgvec import PyMsgVec

EMBEDDING = 256
HEADS = 16
MAX_SEQ = 128
BATCH_SIZE = 32

input_model = MODEL_CONFIGS[HOST_CONFIG.DEFAULT_BRAIN_CONFIG]
msgvec = PyMsgVec(json.dumps(input_model["msgvec"]).encode("utf-8"))
act_shape = msgvec.act_size()  # scalar (default: 4)
obs_shape = msgvec.obs_size()  # scalar (default: 17006)

dataset = DataLoader(
    ChunkDataset(
        chunk_size=MAX_SEQ + 1,
        source=ModularAdditionDataset(number_of_digits=10, seed=42),
    ),
    batch_size=BATCH_SIZE,
    num_workers=0,
)


from src.modules import MultiHeadAttention


def blocks(n):
    return [
        TransformerBlock(
            embedding_size=EMBEDDING,
            attention=MultiHeadAttention.causal_attention(
                max_seq_len=MAX_SEQ, embedding_size=EMBEDDING, heads=HEADS
            ),
        )
        for _ in range(n)
    ]


model = Sequential(
    Sum(
        Embedding(num_embeddings=10 + 2, embedding_dim=EMBEDDING),
        PositionalEncoding(embed_dim=EMBEDDING, block_size=MAX_SEQ),
    ),
    Sequential(*blocks(8)),
    LayerNorm(EMBEDDING),
    Linear(in_features=EMBEDDING, out_features=10 + 2),
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

bar = tqdm()
for raw_entry in dataset:
    optimizer.zero_grad(set_to_none=True)

    entry = (raw_entry + 2).to(torch.int64)
    x = entry[:, :-1]
    expected = entry[:, 1:]
    out_logits = model(x)

    out_logits = torch.permute(out_logits, [0, 2, 1])
    loss = torch.nn.functional.cross_entropy(out_logits, expected, reduction="mean")

    bar.set_description(f"loss: {loss.detach().item():.4f}")
    bar.update()

    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

import torch

from torch.utils.data import IterableDataset


class ChunkDataset(IterableDataset):
    def __init__(
        self,
        chunk_size: int,
        source: IterableDataset,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.source = source

    def __iter__(self):
        buffered = None
        for entry in self.source:
            if buffered is None:
                buffered = entry
            else:
                buffered = torch.cat([buffered, entry], 0)

            if buffered.size(0) >= self.chunk_size:
                yield buffered[: self.chunk_size]
                buffered = buffered[self.chunk_size :]

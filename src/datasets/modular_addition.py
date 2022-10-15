from typing import Optional, Tuple

import torch

from torch import device as Device, Generator, Tensor
from torch.utils.data import IterableDataset


class ModularAdditionDataset(IterableDataset):
    def __init__(
        self,
        number_of_digits,
        device: Optional[Device] = None,
        seed: Optional[int] = None,
    ):
        assert number_of_digits < 18
        super().__init__()
        self.number_of_digits = number_of_digits
        self.device = device
        self.seed = seed

    def __iter__(self):
        generator = Generator(device=self.device)
        if self.seed is not None:
            generator.manual_seed(self.seed)

        current = self._rand(generator)

        while True:
            yield _entry(current)
            diff = self._rand(generator)
            yield _entry(diff)
            current = _add(current, diff)

    def _rand(self, generator: Generator) -> Tuple[Tensor, Tensor]:
        val = torch.randint(
            0,
            10,
            (self.number_of_digits,),
            device=self.device,
            generator=generator,
            dtype=torch.int8,
        )
        neg = torch.randint(0, 2, (1,), device=self.device, generator=generator) != 0
        return neg, val


def _entry(a: Tuple[Tensor, Tensor]) -> Tensor:
    neg, val = a
    return torch.cat([-1 - neg.to(torch.int8), val], -1)


def _int(a: Tuple[Tensor, Tensor]) -> Tensor:
    neg, val = a
    mul_by = torch.arange(0, val.size(-1), device=val.device, dtype=torch.int64)
    mul_by = 10**mul_by
    v = (val * mul_by).sum(-1)
    return torch.where(neg, -v, v)


def _pair(i: Tensor, number_of_digits: int) -> Tuple[Tensor, Tensor]:
    neg = i < 0
    mul_by = torch.arange(0, number_of_digits, device=i.device, dtype=torch.int64)
    mul_by = 10**mul_by
    val = (i.abs() // mul_by) % 10
    return neg, val.to(torch.int8)


def _add(a: Tuple[Tensor, Tensor], b: Tuple[Tensor, Tensor]) -> Tensor:
    return _pair(_int(a) + _int(b), a[1].size(-1))

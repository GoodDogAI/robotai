from torch.nn import Module


class Sum(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        return sum(m(x) for m in self.modules)

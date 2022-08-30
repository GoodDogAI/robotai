from typing import Any

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = None
    __delattr__ = dict.__delitem__

    def __setattr__(self, __name: str, __value: Any) -> None:
        raise NotImplementedError()

    def __delattr__(self, __name: str) -> None:
        raise NotImplementedError()
    
    def __getattr__(self, __name: str) -> Any:
        if __name in self:
            return self[__name]
        else:
            raise KeyError("Config entry '{}' not found".format(__name))
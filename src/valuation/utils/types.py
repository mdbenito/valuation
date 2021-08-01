from typing import Callable, Protocol, TypeVar
from numpy import ndarray

__all__ = ['SupervisedModel', 'Scorer', 'unpackable']


class SupervisedModel(Protocol):
    """ Pedantic: only here for the type hints. """
    def fit(self, x: ndarray, y: ndarray):
        pass

    def predict(self, x: ndarray) -> ndarray:
        pass

    def score(self, x: ndarray, y: ndarray) -> float:
        pass


# ScorerNames = Literal[very long list here]
# instead... ScorerNames = str

Scorer = TypeVar('Scorer',
                 str, Callable[[SupervisedModel, ndarray, ndarray], float])


def unpackable(cls: type) -> type:
    """ A decorator that adds a method to a class so that all of its attributes
    can be unpacked with **val as arguments to a function. E.g.

        @unpackable
        @dataclass
        class Schtuff:
            a: int
            b: str

        x = Schtuff(a=1, b='meh')
        d = dict(**x)
    """
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return getattr(self, item)

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        for k in self.keys():
            return getattr(self, k)

    def update(self, values: dict):
        for k,v in values.items():
            setattr(self, k, v)

    setattr(cls, 'keys', keys)
    setattr(cls, '__getitem__', __getitem__)
    setattr(cls, '__len__', __len__)
    setattr(cls, '__iter__', __iter__)
    setattr(cls, 'update', update)

    return cls

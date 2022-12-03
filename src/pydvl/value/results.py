"""
This module collects types and methods for the inspection of the results of
valuation algorithms.

The most important class is :class:`ValuationResult`, which provides access
to raw values, as well as convenient behaviour as a Sequence with extended
indexing abilities, and conversion to `pandas DataFrames
<https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
"""
import collections.abc
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
    cast,
    overload,
)

import numpy as np

from pydvl.utils import Dataset, SortOrder

try:
    import pandas  # Try to import here for the benefit of mypy
except ImportError:
    pass

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ValuationResult", "ValuationStatus"]


class ValuationStatus(Enum):
    Pending = "pending"
    Converged = "converged"
    MaxIterations = "maximum number of iterations reached"
    Failed = "failed"


@dataclass
class ValueItem:
    """The result of a value computation for one datum."""

    index: np.int_
    name: str
    value: np.float_
    stderr: Optional[np.float_]


class ValuationResult(collections.abc.Sequence):
    """Objects of this class hold the results of valuation algorithms.

    Results can be sorted with :meth:`sort`. Note that sorting values affects
    how iterators and the object itself as ``Sequence`` behave: ``values[0]``
    returns a :class:`ValueItem` with the highest or lowest ranking point if
    this object is sorted by descending or ascending value, respectively. If
    unsorted, ``values[0]`` returns a ``ValueItem`` for index 0.

    Similarly, ``iter(valuation_result)`` returns a generator of
    :class:`ValueItem` in the order in which the object is sorted.

    :param algorithm: The method used.
    :param status: The end status of the algorithm.
    :param values: An array of values, data indices correspond to positions in
        the array.
    :param stderr: An optional array of standard errors in the computation of
        each value.
    :param data_names: Names for the data points. Defaults to index numbers
        if not set.
    :param sort: Whether to sort the values. See above how this affects usage
        as an iterable or sequence.

    :raise ValueError: If data names and values have mismatching lengths.
    """

    _indices: "NDArray[np.int_]"
    _values: "NDArray[np.float_]"
    _data: Dataset
    _names: "Union[NDArray[np.int_], NDArray[np.str_]]"
    _stderr: "NDArray[np.float_]"
    _algorithm: str  # TODO: BaseValuator
    _status: ValuationStatus  # TODO: Maybe? BaseValuator.Status
    _sort_order: Optional[SortOrder] = None

    def __init__(
        self,
        algorithm: Callable,  # BaseValuator,
        status: ValuationStatus,  # Valuation.Status,
        values: "NDArray[np.float_]",
        stderr: Optional["NDArray[np.float_]"] = None,
        data_names: Optional[Sequence[str]] = None,
        sort: Optional[SortOrder] = None,
    ):
        if stderr is not None and len(stderr) != len(values):
            raise ValueError("Lengths of values and stderr do not match")

        self._algorithm = getattr(algorithm, "__name__", "value")
        self._status = status
        self._values = values
        self._stderr = np.zeros_like(values) if stderr is None else stderr

        if data_names is None:
            self._names = np.arange(0, len(values), dtype=np.int_)
        else:
            self._names = np.array(data_names)
        if len(self._names) != len(self._values):
            raise ValueError("Data names and data values have different lengths")
        self.sort(sort)

    def sort(self, sort_order: Optional[SortOrder] = None) -> "ValuationResult":
        """Sorts the values in place.

        Repeated calls with the same `sort_order` are no-ops.

        :param sort_order: None to leave unsorted, otherwise sorts in ascending
            or descending order by value.
        :return: The same object, sorted in place.
        """
        if self._sort_order == sort_order:
            return self

        self._sort_order = sort_order

        if sort_order is None:
            self._indices = np.arange(0, len(self._values), dtype=np.int_)
        else:
            self._indices = np.argsort(self._values)
            if sort_order == SortOrder.Descending:
                self._indices = self._indices[::-1]
        return self

    def is_sorted(self) -> bool:
        return self._sort_order is not None

    @property
    def values(self) -> "NDArray[np.float_]":
        """The raw values, unsorted. Position `i` in the array represents index
        `i` of the data."""
        return self._values

    @property
    def indices(self) -> "NDArray":
        """The indices for the values, possibly sorted.
        If the object is unsorted, then this is the same as
        `np.arange(len(values))`. Otherwise, the indices sort :meth:`values`
        """
        return self._indices

    @property
    def status(self) -> ValuationStatus:
        return self._status

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @overload
    def __getitem__(self, key: int) -> ValueItem:
        ...

    @overload
    def __getitem__(self, key: slice) -> List[ValueItem]:
        ...

    @overload
    def __getitem__(self, key: Iterable[int]) -> List[ValueItem]:
        ...

    def __getitem__(
        self, key: Union[slice, Iterable[int], int]
    ) -> Union[ValueItem, List[ValueItem]]:
        if isinstance(key, slice):
            return [cast(ValueItem, self[i]) for i in range(*key.indices(len(self)))]
        elif isinstance(key, collections.abc.Iterable):
            return [cast(ValueItem, self[i]) for i in key]
        elif isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} out of range (0, {len(self)}).")
            idx = self._indices[key]
            return ValueItem(
                idx, self._names[idx], self._values[idx], self._stderr[idx]
            )
        else:
            raise TypeError("Indices must be integers, iterable or slices")

    def __iter__(self) -> Generator[ValueItem, Any, None]:
        """Iterate over the results returning tuples `(index, value)`"""
        for idx in self._indices:
            yield ValueItem(idx, self._names[idx], self._values[idx], self._stderr[idx])

    def __len__(self):
        return len(self._indices)

    def to_dataframe(
        self, column: Optional[str] = None, use_names: bool = False
    ) -> "pandas.DataFrame":
        """Returns values as a dataframe.

        :param column: Name for the column holding the data value. Defaults to
            the name of the algorithm used.
        :param use_names: Whether to use data names instead of indices for the
            DataFrame's index.
        :return: A dataframe with two columns, one for the values, with name
            given as explained in `column`, and another with standard errors for
            approximate algorithms. The latter will be named `column+'_stderr'`.
        :raise ImportError: If pandas is not installed
        """
        if not pandas:
            raise ImportError("Pandas required for DataFrame export")
        column = column or self._algorithm
        df = pandas.DataFrame(
            self._values[self._indices],
            index=self._names[self._indices] if use_names else self._indices,
            columns=[column],
        )

        if self._stderr is None:
            df[column + "_stderr"] = 0
        else:
            df[column + "_stderr"] = self._stderr[self._indices]
        return df
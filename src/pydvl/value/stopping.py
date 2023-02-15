"""
Stopping criteria for value computations.

This module provides a basic set of stopping criteria: :class:`StandardError`,
:class:`MaxUpdates`, :class:`MinUpdates`, :class:`MaxTime`, and
:class:`HistoryDeviation`. These can behave in different ways depending on the
context. For example, the :class:`MaxUpdates` limits the number of updates to
values, and different algorithms may different number of utility evaluations or
other steps in order to perform a single step.

.. rubric:: Creating stopping criteria

The easiest way is to declare a function implementing the interface
:data:`StoppingCriterionCallable` and wrap it with :func:`make_criterion`. This
creates a :class:`StoppingCriterion` object that can be composed with other
stopping criteria.

Alternatively, and in particular if reporting of completion is required, one can
inherit from this class and implement the abstract methods
:meth:`~pydvl.value.stopping.StoppingCriterion.check` and
:meth:`~pydvl.value.stopping.StoppingCriterion.completion`.

.. rubric:: Composing stopping criteria

Objects of type :class:`StoppingCriterion` can be composed with the binary
operators ``&`` (*and*), and ``|`` (*or*), following the truth tables of
:class:`~pydvl.utils.status.Status`. The unary operator ``~`` (*not*) is also
supported.
"""

import abc
from functools import update_wrapper
from time import time
from typing import Callable, Optional, Type, cast

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import Status
from pydvl.value import ValuationResult

__all__ = [
    "make_criterion",
    "StoppingCriterion",
    "StandardError",
    "MaxUpdates",
    "MaxTime",
    "HistoryDeviation",
]

StoppingCriterionCallable = Callable[[ValuationResult], Status]


class StoppingCriterion(abc.ABC):
    # A boolean array indicating whether the corresponding element has converged
    _converged: NDArray[np.bool_]

    def __init__(self, modify_result: bool = True):
        """A composable callable object to determine whether a computation
        must stop.

        A ``StoppingCriterion`` takes a :class:`~pydvl.value.result.ValuationResult`
        and returns a :class:`~pydvl.value.result.Status~. Objects of this type
        can be composed with the binary operators ``&`` (*and*), and ``|`` (*or*),
        following the truth tables of :class:`~pydvl.utils.status.Status`. The
        unary operator ``~`` (*not*) is also supported.

        :param modify_result: If ``True`` the status of the input
            :class:`~pydvl.value.result.ValuationResult` is modified in place.
        """
        self.modify_result = modify_result
        self._converged = np.full(0, False)

    @abc.abstractmethod
    def check(self, result: ValuationResult) -> Status:
        """Check whether the computation should stop."""
        ...

    @abc.abstractmethod
    def completion(self) -> float:
        """Returns a value between 0 and 1 indicating the completion of the
        computation.
        """
        ...

    def converged(self) -> NDArray[np.bool_]:
        """Returns a boolean array indicating whether the values have converged
        for each data point.
        """
        return self._converged

    @property
    def name(self):
        return type(self).__name__

    def __call__(self, result: ValuationResult) -> Status:
        if result.status is not Status.Pending:
            return result.status
        status = self.check(result)
        if self.modify_result:  # FIXME: this is not nice
            result._status = status
        return status

    def __and__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        class CompositeCriterion(StoppingCriterion):
            def check(self, result: ValuationResult) -> Status:
                return self(result) & other(result)

            @property
            def name(self):
                return f"Composite StoppingCriterion: {self.name} AND {other.name}"

            def completion(self) -> float:
                return min(self.completion(), other.completion())

        return CompositeCriterion(
            modify_result=self.modify_result or other.modify_result
        )

    def __or__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        class CompositeCriterion(StoppingCriterion):
            def check(self, result: ValuationResult) -> Status:
                return self(result) | other(result)

            @property
            def name(self):
                return f"Composite StoppingCriterion: {self.name} OR {other.name}"

            def completion(self) -> float:
                return max(self.completion(), other.completion())

        return CompositeCriterion(
            modify_result=self.modify_result or other.modify_result
        )

    def __invert__(self) -> "StoppingCriterion":
        class CompositeCriterion(StoppingCriterion):
            def check(self, result: ValuationResult) -> Status:
                return cast(Status, ~self(result))  # mypy complains if we don't cast

            @property
            def name(self):
                return f"Composite StoppingCriterion: NOT {self.name}"

            def completion(self) -> float:
                return 1 - self.completion()

        return CompositeCriterion(modify_result=self.modify_result)


def make_criterion(fun: StoppingCriterionCallable) -> Type[StoppingCriterion]:
    """Create a new :class:`StoppingCriterion` from a function.
    Use this to enable simpler functions to be composed with bitwise operators

    :param fun: The callable to wrap.
    :return: A new subclass of :class:`StoppingCriterion`.
    """

    class WrappedCriterion(StoppingCriterion):
        def __init__(self, modify_result: bool = True):
            super().__init__(modify_result=modify_result)
            setattr(self, "check", fun)  # mypy complains if we assign to self.check
            update_wrapper(self, self.check)

        @property
        def name(self):
            return fun.__name__

        def completion(self) -> float:
            return 0.0  # FIXME: not much we can do about this...

        def converged(self) -> NDArray[np.bool_]:
            raise NotImplementedError(
                "Cannot determine individual sample convergence from a function"
            )

    return WrappedCriterion


class StandardError(StoppingCriterion):
    """Compute a ratio of standard errors to values to determine convergence.

    If $s_i$ is the standard error for datum $i$ and $v_i$ its value, then this
    criterion returns :attr:`~pydvl.utils.status.Status.Converged` if
    $s_i / v_i < \\epsilon$ for all $i$ and a threshold value $\\epsilon \\gt 0$.

    :param threshold: A value is considered to have converged if the ratio of
        standard error to value has dropped below this value.
    """

    def __init__(self, threshold: float, modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.threshold = threshold

    def check(self, result: ValuationResult) -> Status:
        ratios = result.stderr / result.values
        self._converged = ratios < self.threshold
        if np.all(self._converged):
            return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        return np.mean(self._converged or [0]).item()


class MaxUpdates(StoppingCriterion):
    """Terminate if any number of value updates exceeds or equals the given
    threshold.

    This checks the ``counts`` field of a
    :class:`~pydvl.value.result.ValuationResult`, i.e. the number of times that
    each index has been updated. For powerset samplers, the maximum of this
    number coincides with the maximum number of subsets sampled. For permutation
    samplers, it coincides with the number of permutations sampled.

    :param n_updates: Threshold: if ``None``, no check is performed,
        effectively creating a (never) stopping criterion that always returns
        ``Pending``.
    """

    def __init__(self, n_updates: Optional[int], modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.n_updates = n_updates
        self.last_max = 0

    def check(self, result: ValuationResult) -> Status:
        if self.n_updates:
            self._converged = result.counts >= self.n_updates
            self.last_max = int(np.max(result.counts))
            if self.last_max >= self.n_updates:
                return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        if self.n_updates is None:
            return 0.0
        return float(np.max(self.last_max).item() / self.n_updates)


class MinUpdates(StoppingCriterion):
    """Terminate as soon as all value updates exceed or equal the given threshold.

    This checks the ``counts`` field of a
    :class:`~pydvl.value.result.ValuationResult`, i.e. the number of times that
    each index has been updated. For powerset samplers, the minimum of this
    number is a lower bound for the number of subsets sampled. For
    permutation samplers, it lower-bounds the amount of permutations sampled.

    :param n_updates: Threshold: if ``None``, no check is performed,
        effectively creating a (never) stopping criterion that always returns
        ``Pending``.
    """

    def __init__(self, n_updates: Optional[int], modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.n_updates = n_updates
        self.last_min = 0

    def check(self, result: ValuationResult) -> Status:
        if self.n_updates is not None:
            self._converged = result.counts >= self.n_updates
            self.last_min = np.min(result.counts).item()
            if self.last_min >= self.n_updates:
                return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        if self.n_updates is None:
            return 0.0
        return float(np.min(self.last_min).item() / self.n_updates)


class MaxTime(StoppingCriterion):
    """Terminate if the computation time exceeds the given number of seconds.

    Checks the elapsed time since construction

    :param seconds: Threshold: The computation is terminated if the elapsed time
        between object construction and a check exceeds this value. If ``None``,
        no check is performed, effectively creating a (never) stopping criterion
        that always returns ``Pending``.
    """

    def __init__(self, seconds: Optional[float], modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.max_seconds = seconds or np.inf
        if self.max_seconds <= 0:
            raise ValueError("Number of seconds for MaxTime must be positive or None")
        self.start = time()

    def check(self, result: ValuationResult) -> Status:
        if self._converged is None:
            self._converged = np.full(result.values.shape, False)
        if time() > self.start + self.max_seconds:
            self._converged.fill(True)
            return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        if self.max_seconds is None:
            return 0.0
        return (time() - self.start) / self.max_seconds


class HistoryDeviation(StoppingCriterion):
    r"""A simple check for relative distance to a previous step in the
    computation.

    The method used by :footcite:t:`ghorbani_data_2019` computes the relative
    distances between the current values $v_i^t$ and the values at the previous
    checkpoint $v_i^{t-\tau}$. If the sum is below a given threshold, the
    computation is terminated.

    $$\sum_{i=1}^n \frac{\left| v_i^t - v_i^{t-\tau} \right|}{v_i^t} <
    \epsilon.$$

    When the denominator is zero, the summand is set to the value of $v_i^{
    t-\tau}$.

    This implementation is slightly generalised to allow for different number of
    updates to individual indices, as happens with powerset samplers instead of
    permutations. Every subset of indices that is found to converge can be
    pinned
    to that state. Once all indices have converged the method has converged.

    .. warning::
       This criterion is meant for the reproduction of the results in the paper,
       but we do not recommend using it in practice.

    :param n_steps: Checkpoint values every so many updates and use these saved
        values to compare.
    :param rtol: Relative tolerance for convergence ($\epsilon$ in the formula).
    :param pin_converged: If ``True``, once an index has converged, it is pinned
    """

    _memory: NDArray[np.float_]

    def __init__(
        self,
        n_steps: int,
        rtol: float,
        pin_converged: bool = True,
        modify_result: bool = True,
    ):
        super().__init__(modify_result=modify_result)
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if rtol <= 0 or rtol >= 1:
            raise ValueError("rtol must be in (0, 1)")

        self.n_steps = n_steps
        self.rtol = rtol
        self.update_op = np.logical_or if pin_converged else np.logical_and
        self._memory = None  # type: ignore

    def check(self, r: ValuationResult) -> Status:
        if self._memory is None:
            self._memory = np.full((len(r.values), self.n_steps + 1), np.inf)
            self._converged = np.full(len(r), False)
            return Status.Pending

        # shift left: last column is the last set of values
        self._memory = np.concatenate(
            [self._memory[:, 1:], r.values.reshape(-1, 1)], axis=1
        )

        # Look at indices that have been updated more than n_steps times
        ii = np.where(r.counts > self.n_steps)
        if len(ii) > 0:
            curr = self._memory[:, -1]
            saved = self._memory[:, 0]
            diffs = np.abs(curr[ii] - saved[ii])
            quots = np.divide(diffs, curr[ii], out=diffs, where=curr[ii] != 0)
            # quots holds the quotients when the denominator is non-zero, and
            # the absolute difference, which is just the memory, otherwise.
            if np.mean(quots) < self.rtol:
                self._converged = self.update_op(self._converged, ii)  # type: ignore
                if np.all(self._converged):
                    return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        return np.mean(self._converged or [0]).item()
from typing import Callable, Tuple, Type

from scipy.special import expit

from pydvl.utils.types import SupervisedModel
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.scorers.supervised import SupervisedScorer

__all__ = ["compose_score", "sigmoid"]


def compose_score(
    scorer: SupervisedScorer,
    transformation: Callable[[float], float],
    name: str,
) -> SupervisedScorer:
    """Composes a scoring function with an arbitrary scalar transformation.

    Useful to squash unbounded scores into ranges manageable by data valuation
    methods.

    Example:

    ```python
    sigmoid = lambda x: 1/(1+np.exp(-x))
    compose_score(Scorer("r2"), sigmoid, range=(0,1), name="squashed r2")
    ```

    Args:
        scorer: The object to be composed.
        transformation: A scalar transformation
        range: The range of the transformation. This will be used e.g. by
            [Utility][pydvl.valuation.utility.Utility] for the range of the
            composite scorer.
        name: A string representation for the composition, for `str()`.

    Returns:
        The composite [SupervisedScorer][pydvl.valuation.scorers.SupervisedScorer].
    """

    class CompositeSupervisedScorer(SupervisedScorer):
        def __call__(self, model: SupervisedModel) -> float:
            raw = super().__call__(model)
            return transformation(raw)

    new_scorer = CompositeSupervisedScorer(
        scoring=scorer._scorer,
        test_data=scorer.test_data,
        default=transformation(scorer.default),
        range=(transformation(scorer.range[0]), transformation(scorer.range[1])),
        name=name,
    )
    return new_scorer


def sigmoid(x: float) -> float:
    result: float = expit(x).item()
    return result
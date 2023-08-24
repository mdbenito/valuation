---
title: Semi-values
---

# Semi-values

Shapley values are a particular case of a more general concept called semi-value,
which is a generalization to different weighting schemes. A **semi-value** is
any valuation function with the form:

$$
v\_\text{semi}(i) = \sum_{i=1}^n w(k)
\sum_{S \subset D\_{-i}^{(k)}} [U(S\_{+i})-U(S)],
$$

where the coefficients $w(k)$ satisfy the property:

$$\sum_{k=1}^n w(k) = 1.$$

Two instances of this are **Banzhaf indices** [@wang_data_2022],
and **Beta Shapley** [@kwon_beta_2022], with better numerical and
rank stability in certain situations.

!!! Note
    Shapley values are a particular case of semi-values and can therefore also be
    computed with the methods described here. However, as of version 0.6.0, we
    recommend using [compute_shapley_values][pydvl.value.shapley.compute_shapley_values] instead,
    in particular because it implements truncated Monte Carlo sampling for faster
    computation.


## Beta Shapley

For some machine learning applications, where the utility is typically the
performance when trained on a set $S \subset D$, diminishing returns are often
observed when computing the marginal utility of adding a new data point.

Beta Shapley is a weighting scheme that uses the Beta function to place more
weight on subsets deemed to be more informative. The weights are defined as:

$$
w(k) := \frac{B(k+\beta, n-k+1+\alpha)}{B(\alpha, \beta)},
$$

where $B$ is the [Beta function](https://en.wikipedia.org/wiki/Beta_function),
and $\alpha$ and $\beta$ are parameters that control the weighting of the
subsets. Setting both to 1 recovers Shapley values, and setting $\alpha = 1$, and
$\beta = 16$ is reported in [@kwon_beta_2022] to be a good choice for
some applications. See however the [Banzhaf indices][banzhaf-indices] section 
for an alternative choice of weights which is reported to work better.

```python
from pydvl.value import compute_semivalues

values = compute_semivalues(
   u=utility, mode="beta_shapley", done=MaxUpdates(500), alpha=1, beta=16
)
```

## Banzhaf indices

As noted in the section [Problems of Data Values][problems-of-data-values],
the Shapley value can be very sensitive to variance in the utility function.
For machine learning applications, where the utility is typically the performance
when trained on a set $S \subset D$, this variance is often largest
for smaller subsets $S$. It is therefore reasonable to try reducing
the relative contribution of these subsets with adequate weights.

One such choice of weights is the Banzhaf index, which is defined as the
constant:

$$w(k) := 2^{n-1},$$

for all set sizes $k$. The intuition for picking a constant weight is that for
any choice of weight function $w$, one can always construct a utility with
higher variance where $w$ is greater. Therefore, in a worst-case sense, the best
one can do is to pick a constant weight.

The authors of [@wang_data_2022] show that Banzhaf indices are more
robust to variance in the utility function than Shapley and Beta Shapley values.

```python
from pydvl.value import compute_semivalues, MaxUpdates

values = compute_semivalues( u=utility, mode="banzhaf", done=MaxUpdates(500))
```
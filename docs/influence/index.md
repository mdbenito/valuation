---
title: The influence function
alias:
  name: influence-function
  text: Computing Influence Values
---

## The influence function

!!! Warning 
    The code in the package [pydvl.influence][pydvl.influence] is experimental.
    Package structure and basic API are bound to change before v1.0.0

The influence function (IF) is a method to quantify the effect (influence) that
each training point has on the parameters of a model, and by extension on any
function thereof. In particular, it allows to estimate how much each training
sample affects the error on a test point, making the IF useful for understanding
and debugging models.

Alas, the influence function relies on some assumptions that can make their
application difficult. Yet another drawback is that they require the computation
of the inverse of the Hessian of the model wrt. its parameters, which is
intractable for large models like deep neural networks. Much of the recent
research tackles this issue using approximations, like a Neuman series
[@agarwal_secondorder_2017], with the most successful solution using a low-rank
approximation that iteratively finds increasing eigenspaces of the Hessian
[@schioppa_scaling_2022].

pyDVL implements several methods for the efficient computation of the IF for
machine learning. In the examples we document some of the difficulties that can
arise when using the IF.

We currently implement the following methods:

- [**CG Influence**][pydvl.influence.torch.CgInfluence].
  [@koh_understanding_2017].

- [**Direct Influence**][pydvl.influence.torch.DirectInfluence]
  [@koh_understanding_2017].

- [**LiSSA**][pydvl.influence.torch.LissaInfluence]
  [@agarwal_secondorder_2017].

- [**Arnoldi Influence**][pydvl.influence.torch.ArnoldiInfluence]
  [@schioppa_scaling_2022].

- [**EKFAC Influence**][pydvl.influence.torch.EkfacInfluence]
  [@george_fast_2018;@martens_optimizing_2015].

- [**Nyström Influence**][pydvl.influence.torch.NystroemSketchInfluence], based
  on the ideas in [@hataya_nystrom_2023] for bi-level optimization.

- [**Inverse-harmonic-mean Influence**][pydvl.influence.torch.InverseHarmonicMeanInfluence]
  [@kwon_datainf_2023].

## Construction

First introduced in the context of robust statistics in [@hampel_influence_1974],
the IF was popularized in the context of machine learning in
[@koh_understanding_2017].

Following their formulation, consider an input space $\mathcal{X}$ (e.g. images)
and an output space $\mathcal{Y}$ (e.g. labels). Let's take $z_i = (x_i, y_i)$,
for $i \in  \{1,...,n\}$ to be the $i$-th training point, and $\theta$ to be the
(potentially highly) multi-dimensional parameters of a model (e.g. $\theta$ is a
big array with all of a neural network's parameters, including biases and/or
dropout rates). We will denote with $L(z, \theta)$ the loss of the model for
point $z$ when the parameters are $\theta.$

To train a model, we typically minimize the loss over all $z_i$, i.e. the
optimal parameters are

$$\hat{\theta} = \arg \min_\theta \sum_{i=1}^n L(z_i, \theta).$$

In practice, lack of convexity means that one doesn't really obtain the
minimizer of the loss, and the training is stopped when the validation loss
stops decreasing.

For notational convenience, let's define

$$\hat{\theta}_{-z} = \arg \min_\theta \sum_{z_i \ne z} L(z_i, \theta), $$

i.e. $\hat{\theta}_{-z}$ are the model parameters that minimize the total loss
when $z$ is not in the training dataset.

In order to compute the impact of each training point on the model, we would
need to calculate $\hat{\theta}_{-z}$ for each $z$ in the training dataset, thus
re-training the model at least ~$n$ times (more if model training is
stochastic). This is computationally very expensive, especially for big neural
networks. To circumvent this problem, we can just calculate a first order
approximation of $\hat{\theta}$. This can be done through single backpropagation
and without re-training the full model.


pyDVL supports two ways of computing the empirical influence function, namely
up-weighting of samples and perturbation influences.

### Approximating the influence of a point

Let's define

$$\hat{\theta}_{\epsilon, z} = \arg \min_\theta \frac{1}{n}\sum_{i=1}^n L(z_i,
\theta) + \epsilon L(z, \theta), $$

which is the optimal $\hat{\theta}$ when we up-weight $z$ by an amount $\epsilon
\gt 0$.

From a classical result (a simple derivation is available in Appendix A of
[@koh_understanding_2017]), we know that:

$$\frac{d \ \hat{\theta}_{\epsilon, z}}{d \epsilon} \Big|_{\epsilon=0} =
-H_{\hat{\theta}}^{-1} \nabla_\theta L(z, \hat{\theta}), $$

where $H_{\hat{\theta}} = \frac{1}{n} \sum_{i=1}^n \nabla_\theta^2 L(z_i,
\hat{\theta})$ is the Hessian of $L$. These quantities are also knows as
**influence factors**.

Importantly, notice that this expression is only valid when $\hat{\theta}$ is a
minimum of $L$, or otherwise $H_{\hat{\theta}}$ cannot be inverted! At the same
time, in machine learning full convergence is rarely achieved, so direct Hessian
inversion is not possible. Approximations need to be developed that circumvent
the problem of inverting the Hessian of the model in all those (frequent) cases
where it is not positive definite.

The influence of training point $z$ on test point $z_{\text{test}}$ is defined
as:

$$\mathcal{I}(z, z_{\text{test}}) =  L(z_{\text{test}}, \hat{\theta}_{-z}) -
L(z_{\text{test}}, \hat{\theta}). $$

Notice that $\mathcal{I}$ is higher for points $z$ which positively impact the
model score, since the loss is higher when they are excluded from training. In
practice, one needs to rely on the following infinitesimal approximation:

$$\mathcal{I}_{up}(z, z_{\text{test}}) = - \frac{d L(z_{\text{test}},
\hat{\theta}_{\epsilon, z})}{d \epsilon} \Big|_{\epsilon=0} $$

Using the chain rule and the results calculated above, we get:

$$\mathcal{I}_{up}(z, z_{\text{test}}) = - \nabla_\theta L(z_{\text{test}},
\hat{\theta})^\top \ \frac{d \hat{\theta}_{\epsilon, z}}{d \epsilon}
\Big|_{\epsilon=0} = \nabla_\theta L(z_{\text{test}}, \hat{\theta})^\top \
H_{\hat{\theta}}^{-1} \ \nabla_\theta L(z, \hat{\theta}) $$

All the resulting factors are gradients of the loss wrt. the model parameters
$\hat{\theta}$. This can be easily computed through one or more backpropagation
passes.

### Perturbation definition of the influence score

How would the loss of the model change if, instead of up-weighting an individual
point $z$, we were to up-weight only a single feature of that point? Given $z =
(x, y)$, we can define $z_{\delta} = (x+\delta, y)$, where $\delta$ is a vector
of zeros except for a 1 in the position of the feature we want to up-weight. In
order to approximate the effect of modifying a single feature of a single point
on the model score we can define

$$\hat{\theta}_{\epsilon, z_{\delta} ,-z} = \arg \min_\theta
\frac{1}{n}\sum_{i=1}^n L(z_{i}, \theta) + \epsilon L(z_{\delta}, \theta) -
\epsilon L(z, \theta), $$

Similarly to what was done above, we up-weight point $z_{\delta}$, but then we
also remove the up-weighting for all the features that are not modified by
$\delta$. From the calculations in
[the previous section](#approximating-the-influence-of-a-point),
it is then easy to see that

$$\frac{d \ \hat{\theta}_{\epsilon, z_{\delta} ,-z}}{d \epsilon}
\Big|_{\epsilon=0} = -H_{\hat{\theta}}^{-1} \nabla_\theta \Big( L(z_{\delta},
\hat{\theta}) - L(z, \hat{\theta}) \Big) $$

and if the feature space is continuous and as $\delta \to 0$ we can write

$$\frac{d \ \hat{\theta}_{\epsilon, z_{\delta} ,-z}}{d \epsilon}
\Big|_{\epsilon=0} = -H_{\hat{\theta}}^{-1} \ \nabla_x \nabla_\theta L(z,
\hat{\theta}) \delta + \mathcal{o}(\delta) $$

The influence of each feature of $z$ on the loss of the model can therefore be
estimated through the following quantity:

$$\mathcal{I}_{pert}(z, z_{\text{test}}) = - \lim_{\delta \to 0} \
\frac{1}{\delta} \frac{d L(z_{\text{test}}, \hat{\theta}_{\epsilon, \
z_{\delta}, \ -z})}{d \epsilon} \Big|_{\epsilon=0} $$

which, using the chain rule and the results calculated above, is equal to

$$\mathcal{I}_{pert}(z, z_{\text{test}}) = - \nabla_\theta L(z_{\text{test}},
\hat{\theta})^\top \ \frac{d \hat{\theta}_{\epsilon, z_{\delta} ,-z}}{d
\epsilon} \Big|_{\epsilon=0} = \nabla_\theta L(z_{\text{test}},
\hat{\theta})^\top \ H_{\hat{\theta}}^{-1} \ \nabla_x \nabla_\theta L(z,
\hat{\theta}) $$

The perturbation definition of the influence score is not straightforward to
understand, but it has a simple interpretation: it tells how much the loss of
the model changes when a certain feature of point z is up-weighted. A positive
perturbation influence score indicates that the feature might have a positive
effect on the accuracy of the model.

It is worth noting that the perturbation influence score is a very rough
estimate of the impact of a point on the models loss and it is subject to large
approximation errors. It can nonetheless be used to build training-set attacks,
as done in [@koh_understanding_2017].

## Computation

The main abstraction of the library for influence calculation is
[InfluenceFunctionModel][pydvl.influence.base_influence_function_model.InfluenceFunctionModel]. 
On implementations of this abstraction, you can call the method `influences`
to compute influences. 

pyDVL provides implementations to use with pytorch model in
[pydvl.influence.torch][pydvl.influence.torch.influence_function_model]. For detailed information 
on available implementations see the documentation in [InfluenceFunctionModel](influence_function_model.md).

Given a pre-trained pytorch model and a loss, a basic example would look like

```python
from torch.utils.data import DataLoader
from pydvl.influence.torch import DirectInfluence

training_data_loader = DataLoader(...)
infl_model = DirectInfluence(model, loss)
infl_model = infl_model.fit(training_data_loader)

influences = infl_model.influences(x_test, y_test, x, y)
```
for batches $z_{\text{test}} = (x_{\text{test}}, y_{\text{test}})$ and
$z = (x, y)$ of data. The result is a tensor with one row per test point in 
$z_{\text{test}}$ and one column per point in $z$. 
Thus, each entry $(i, j)$ represents the influence of training point $z[j]$
on test point $z_{\text{test}}[i]$.

!!! Warning
    Compared to the mathematical definitions above, we switch the ordering
    of $z$ and $z_{\text{test}}$, in order to make the input ordering consistent
    with the dimensions of the resulting tensor. More concrete if the first
    dimension of $z_{\text{test}}$ is $N$ and that of $z$, the resulting tensor
    is of shape $N \times M$

A large positive influence indicates that training point $j$
tends to improve the performance of the model on test point $i$, and vice versa,
a large negative influence indicates that training point $j$ tends to worsen the
performance of the model on test point $i$.

### Hessian regularization

Additionally, and as discussed in [the introduction](#the-influence-function),
in machine learning training rarely converges to a global minimum of the loss.
Despite good apparent convergence, $\hat{\theta}$ might be located in a region
with flat curvature or close to a saddle point. In particular, the Hessian might
have vanishing eigenvalues making its direct inversion impossible. Certain
methods, such as the [Arnoldi method](#arnoldi-solver) are robust against these
problems, but most are not.

To circumvent this problem, many approximate methods can be implemented. The
simplest adds a small *hessian perturbation term*, i.e. $H_{\hat{\theta}} +
\lambda \mathbb{I}$, with $\mathbb{I}$ being the identity matrix. 

```python
from torch.utils.data import DataLoader
from pydvl.influence.torch import DirectInfluence

training_data_loader = DataLoader(...)
infl_model = DirectInfluence(model, loss, regularization=0.01)
infl_model = infl_model.fit(training_data_loader)
```

This standard
trick ensures that the eigenvalues of $H_{\hat{\theta}}$ are bounded away from
zero and therefore the matrix is invertible. In order for this regularization
not to corrupt the outcome too much, the parameter $\lambda$ should be as small
as possible while still allowing a reliable inversion of $H_{\hat{\theta}} +
\lambda \mathbb{I}$.

### Block-diagonal approximation

This implementation is capable of using a block-diagonal approximation.
The full matrix is approximated by a block-diagonal version, which
reduces both the time and memory consumption.
The blocking structure can be specified via the `block_structure` parameter.
The `block_structure` parameter can either be a
[BlockMode][pydvl.influence.torch.util.BlockMode] enum (which provides
layer-wise or parameter-wise blocking) or a custom block structure defined
by an ordered dictionary with the keys being the block identifiers (arbitrary
strings) and the values being lists of parameter names contained in the block.
```python
from torch.utils.data import DataLoader
from pydvl.influence.torch import DirectInfluence, BlockMode, SecondOrderMode

training_data_loader = DataLoader(...)
# layer-wise block-diagonal approximation
infl_model = DirectInfluence(model, loss,
                             regularization=0.1,
                             block_structure=BlockMode.LAYER_WISE)

block_structure = OrderedDict((
    ("custom_block1", ["0.weight", "1.bias"]), 
    ("custom_block2", ["1.weight", "0.bias"]),
))
# custom block-diagonal structure
infl_model = DirectInfluence(model, loss,
                             regularization=0.1,
                             block_structure=block_structure)
infl_model = infl_model.fit(training_data_loader)
```
If you would like to apply a block-specific regularization, you can provide a
dictionary with the block names as keys and the regularization values as values.
If no value is provided for a specific key, no regularization is applied for
the corresponding block.

```python
regularization =  {
"custom_block1": 0.1,
"custom_block2": 0.2,
}
infl_model = DirectInfluence(model, loss,
                             regularization=regularization,
                             block_structure=block_structure)
infl_model = infl_model.fit(training_data_loader)
```
Accordingly, if you choose a layer-wise or parameter-wise structure
(by providing `BlockMode.LAYER_WISE` or `BlockMode.PARAMETER_WISE` for
`block_structure`) the keys must be the layer names or parameter names,
respectively.
You can retrieve the block-wise influence information from the methods
with suffix `_by_block`. By default, `block_structure` is set to
`BlockMode.FULL` and in this case these methods will return a dictionary
with the empty string being the only key.

### Gauss-Newton approximation

In the computation of the influence values, the inversion of the Hessian can be
replaced by the inversion of the Gauss-Newton matrix

$$ G_{\hat{\theta}}=n^{-1} \sum_{i=1}^n \nabla_{\theta}L(z_i, \hat{\theta})
    \nabla_{\theta}L(z_i, \hat{\theta})^T $$

so the computed values are of the form

$$\nabla_\theta L(z_{\text{test}}, \hat{\theta})^\top \
G_{\hat{\theta}}^{-1} \ \nabla_\theta L(z, \hat{\theta}). $$

The parameter `second_orer_mode` is used to configure this approximation.
```python
from torch.utils.data import DataLoader
from pydvl.influence.torch import DirectInfluence, BlockMode, SecondOrderMode

training_data_loader = DataLoader(...)
infl_model = DirectInfluence(model, loss,
                             regularization={"layer_1": 0.1, "layer_2": 0.2},
                             block_structure=BlockMode.LAYER_WISE,
                             second_order_mode=SecondOrderMode.GAUSS_NEWTON)
infl_model = infl_model.fit(training_data_loader)
```


### Perturbation influences

The method of empirical influence computation can be selected with the
parameter `mode`:

```python
from pydvl.influence import InfluenceMode

influences = infl_model.influences(x_test, y_test, x, y,
                                   mode=InfluenceMode.Perturbation)
```
The result is a tensor with at least three dimensions. The first two dimensions
are the same as in the case of `mode=InfluenceMode.Up` case, i.e. one row per test
point and one column per training point. The remaining dimensions are the same
as the number of input features in the data. Therefore, each entry in the tensor
represents the influence of each feature of each training point on each test
point.


### Influence factors

The influence factors(refer to
[the previous section](#approximating-the-influence-of-a-point) for a definition)
are typically the most computationally demanding part of influence calculation.
They can be obtained via calling the `influence_factors` method, saved, and later used 
for influence calculation on different subsets of the training dataset.

```python
influence_factors = infl_model.influence_factors(x_test, y_test)
influences = infl_model.influences_from_factors(influence_factors, x, y)
```





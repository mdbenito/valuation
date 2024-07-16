---
title: How-to scale up influence function locally with lazy computations
alias:
    name: scale-up-if-locally-with-lazy
---

This guide shows you how to scale influence function
computations up locally using lazy computations.

!!! tip "Other guides"

    For speeding-up influence function computations with parallelization
    see [[speed-up-if-with-parallel]].

    For speeding-up data valuation algorithms with parallelization
    see [[speed-up-value-with-parallel]].

Influence function calculations on large datasets can be challenging due to
memory constraints (e.g. input data does not into memory or large models)
even more so locally when one does not have access to a compute cluster.
In these cases, we want to map our influence function model over
collections of batches (or chunks) of data and compute them lazily.

We start by instantiating the dataloaders and the model and then fitting
the latter:

```python
from pydvl.influence.torch import CgInfluence
from torch.utils.data import DataLoader

batch_size = 10
train_dataloader = DataLoader(..., batch_size=batch_size)
test_dataloader = DataLoader(..., batch_size=batch_size)

infl_model = CgInfluence(model, loss, regularization=0.01)
infl_model.fit(train_dataloader)
```

We then compute influences sequentially:

```python
from pydvl.influence import SequentialInfluenceCalculator
from pydvl.influence.torch.util import NestedTorchCatAggregator

infl_calc = SequentialInfluenceCalculator(infl_model)

# this does not trigger the computation
lazy_influences = infl_calc.influences(test_dataloader, train_dataloader)

# trigger computation and pull the result into main memory, 
# result is the full tensor for all combinations of the two loaders
influences = lazy_influences.compute(aggregator=NestedTorchCatAggregator())
# or
# trigger computation and write results chunk-wise to disk using zarr 
# in a sequential manner
lazy_influences.to_zarr("local_path/or/url", TorchNumpyConverter())
```

!!! tip "Batch size"

    The batch size should be chosen as large as possible,
    such that the corresponding batches fit into memory.

The simplest way to scale the computation of influence functions
is using the convenience class 
[SequentialInfluenceCalculator][pydvl.influence.influence_calculator].

This uses a double for-loop to iterate over the batches sequentially
and collects them.

When invoking the `compute` method, you have the option to specify
a custom aggregator by implementing
[NestedSequenceAggregator][pydvl.influence.array.NestedSequenceAggregator]. 
This allows for the aggregation of computed chunks. 
Such an approach is particularly beneficial for straightforward
aggregation tasks, commonly seen in sequential computation models. 
Examples include operations like concatenation, as implemented in 
[NestedTorchCatAggregator][pydvl.influence.torch.util.NestedTorchCatAggregator], 
or basic **min** and **max** operations. 

For more complex aggregations, such as an **argmax** operation, 
consider using the parallel approach described in [[speed-up-if-with-parallel]].

## Conclusion

By following this guide, we have shown how you can scale up
influence function computations locally using lazy computations.

---
title: How-to speed up influence function with parallelization
alias:
    name: speed-up-if-with-parallel
---

This guide shows you how to parallelize influence function
computations using pyDVL to handle large batches of data efficiently.

!!! tip "Other guides"

    For scaling-up influence function computations locally with
    lazy computations see [[scale-up-if-locally-with-lazy]].

    For speeding-up data valuation algorithms with parallelization
    see [[speed-up-value-with-parallel]].

Calculating influence functions on large datasets can be challenging due to
memory constraints (e.g. input data does not into memory or large models).
In these cases, we want to map our influence function model over
collections of batches (or chunks) of data and distribute these computations
over multiple workers because the influence computation itself
is completely data parallel. For this purpose, pyDVL uses [Dask](https://docs.dask.org/en/stable/).

We start by instantiating the dataloaders and the model and then fitting
the latter:


```python
from torch.utils.data import Dataset, DataLoader
from pydvl.influence.torch import CgInfluence

train_data_set: Dataset = LargeDataSet(
    ...)  # Possible some out of memory large Dataset
test_data_set: Dataset = LargeDataSet(
    ...)  # Possible some out of memory large Dataset

train_dataloader = DataLoader(train_data_set)
infl_model = CgInfluence(model, loss, regularization=0.01)
infl_model = infl_model.fit(train_dataloader)
```

After that, we instantiate a Dask client and wrap the data into dask arrays:

```python
from pydvl.influence.torch.util import torch_dataset_to_dask_array
from distributed import Client

# use only one thread for scheduling, 
# due to non-thread safety of some torch operations
client = Client(n_workers=4, threads_per_worker=1)

# wrap your input data into dask arrays
chunk_size = 10
da_x, da_y = torch_dataset_to_dask_array(train_data_set, chunk_size=chunk_size)
da_x_test, da_y_test = torch_dataset_to_dask_array(
    test_data_set,
    chunk_size=chunk_size
)
```

Finally, we compute the influences and write them to disk:

```python
import torch
from pydvl.influence import DaskInfluenceCalculator
from pydvl.influence.torch.util import TorchNumpyConverter

infl_calc = DaskInfluenceCalculator(
    infl_model,
    converter=TorchNumpyConverter(device=torch.device("cpu")),
    client=client
)
da_influences = infl_calc.influences(da_x_test, da_y_test, da_x, da_y)
# da_influences is a dask.array.Array
# trigger computation and write chunks to disk in parallel
da_influences.to_zarr("path/or/url")
```

!!! tip "Chunk size"

    Again, choosing an appropriate chunk size can be crucial.
    For a better understanding see the official 
    [dask best practice documentation](https://docs.dask.org/en/latest/array-best-practices.html#select-a-good-chunk-size)
    and the following [blog entry](https://blog.dask.org/2021/11/02/choosing-dask-chunk-sizes).

!!! Warning

    Make sure to set `threads_per_worker=1`, when using the distributed
    scheduler for computing, if your implementation of
    [InfluenceFunctionModel][pydvl.influence.base_influence_function_model.InfluenceFunctionModel]
    is not thread-safe.

    ```python
    client = Client(threads_per_worker=1)
    ```

    For details on dask schedulers see the
    [official documentation](https://docs.dask.org/en/stable/scheduling.html).

During initialization of the 
[DaskInfluenceCalculator][pydvl.influence.influence_calculator.DaskInfluenceCalculator], 
the system verifies if all workers are operating in
single-threaded mode when the provided influence_function_model is
designated as not thread-safe (indicated by the `is_thread_safe` property).
If this condition is not met, the initialization will raise a specific
error, signaling a potential thread-safety conflict.

#### Skipping Thread-Safety Check

To intentionally skip this safety check
(e.g., for debugging purposes using the single machine synchronous
scheduler), you can supply the [DisableClientSingleThreadCheck]
[pydvl.influence.influence_calculator.DisableClientSingleThreadCheck] type.

```python
from pydvl.influence import DisableClientSingleThreadCheck

infl_calc = DaskInfluenceCalculator(
    infl_model,
    TorchNumpyConverter(device=torch.device("cpu")),
    DisableClientSingleThreadCheck
)
da_influences = infl_calc.influences(da_x_test, da_y_test, da_x, da_y)
da_influences.compute(scheduler="synchronous")
```

## Conclusion

By following this guide, you can efficiently compute influence functions
on large datasets by distributing the computations over multiple workers.

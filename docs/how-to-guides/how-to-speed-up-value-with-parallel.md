---
title: How-to speed up data valuation with parallelization
alias:
    name: speed-up-value-with-parallel
---

Real world data valuation problems often comprise more complex and larger
models and/or datasets that are orders of magnitude larger and complex
than the ones we see in the literature.

In this guide we will show you how to speed up data valuation algorithms
by leveraging parallelization either locally (within a single machine)
using threads or processes or remotely (across multiple machines)
using [Ray](https://ray.io) or [Dask](https://docs.dask.org/en/stable/).

The former works out of the box but for the latter you will need to install
additional dependencies (see [Extras][installation-extras])
and to provide a running cluster.

!!! tip "Other guides"

    For scaling-up influence function algorithms with parallelization
    see [[scale-up-if-with-parallel]].

    For alternative ways to speed up data valuation
    see [[speed-up-value-with-caching]].

!!! warning "Scaling dataset size"

    The entire dataset needs to fit in the memory of each and every worker.
    If your dataset doesn't fit in memory, then you should consider
    using a model that can be fitted incrementally with a dataset
    that can be loaded partially.

## Set up the dataset, model and method

For the rest of the guide we will use the following dataset and model:

```python
from sklearn.datasets import fetch_covtype
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.svm import LinearSVC

data = fetch_covtype()
model = make_pipeline(MinMaxScaler(), Normalizer(), LinearSVC())
```

And the following scorer, utility and data valuation method:

```python
from pydvl.valuation import (
    Dataset,
    DataShapleyValuation,
    ModelUtility,
    PermutationSampler,
    MaxUpdates,
    Scorer,
)

scorer = Scorer("accuracy")
utility = ModelUtility(model, scorer)
training_set, test_set = Dataset.from_sklearn(data, random_state=16)
valuation = DataShapleyValuation(
    utility,
    sampler=PermutationSampler(batch_size=10, seed=16),
    is_done=MaxUpdates(100)
)
```

## Parallelization

The general pattern to parallelize the data valuation methods is the following:

```python
from joblib import parallel_config

with parallel_config():
    valuation.fit(training_set)
```

pyDVL uses joblib's [Parallel][joblib.Parallel] class internally as 
a context manager to submit the computations and we can configure
it at runtime using the [parallel_config][joblib.parallel_config]
context manager.

Please read its documentation to know all possible configuration options
that you can pass to the [parallel_config][joblib.parallel_config]
context manager.

### Local Parallelization

Locally, you can use any of the supported backends
(threads, multiprocessing, loky, dask, ray) to speed up
the computations. By default, however, it uses the `loky` backend
which is a multiprocessing backend with reusable processes.

!!! warning "Threads"

    Threads are not recommended unless the model's fitting releases the GIL.


```python
from joblib import parallel_config

# We limit the number of concurrent jobs to 4
with parallel_config(backend="loky", n_jobs=4):
    valuation.fit(training_set)
```
### Remote Parallelization

Remotely, you can use either Dask or Ray backends to speed up
computations across several machines.

#### Ray

Please follow the instructions in Ray's documentation to
[set up a remote cluster](https://docs.ray.io/en/latest/cluster/key-concepts.html).
You could alternatively use a local cluster and in that case
you don't have to set anything up.

Before starting a computation, you should initialize ray by calling 
[`ray.init`][ray.init] with the appropriate parameters.

```python
import ray

# either start a local ray cluster with 4 CPUs you would use
ray.init(num_cpus=4)

# Or connect to a remote ray cluster
address = "<Hypothetical Ray Cluster IP Address>"
ray.init(address)
```

To use the ray joblib parallel backend you first have to register it:

```python
from joblib import register_parallel_backend
from ray.util.joblib.ray_backend import RayBackend

RayBackend.supports_return_generator = True
register_parallel_backend("ray", RayBackend)
```

!!! warning "Registering ray backend"

    Until [this issue](https://github.com/ray-project/ray/pull/41028)
    on ray's repository is resolved, you cannot simply use
    the [suggested approach](https://docs.ray.io/en/latest/ray-more-libs/joblib.html)
    in their documentation to register the ray backend.

To use ray to compute shapley values you would then use:

```python
from joblib import parallel_config

# We use the 'ray' backend, limit the number of concurrent jobs to 4,
# and specify a resource requirement of 1 gpu per task
with parallel_config(backend="ray", n_jobs=4, ray_remote_args=dict(num_gpus=1)):
    valuation.fit(dataset)
```

#### Dask

Please follow the instructions in Dask's documentation to
[set up a remote cluster](https://docs.dask.org/en/stable/deploying.html#distributed-computing).
You could alternatively use a local cluster and in that case
you don't have to set anything up.

```python
from dask.distributed import Client, LocalCluster

# Start a fully-featured local Dask cluster
cluster = LocalCluster()
client = cluster.get_client()
# Monitor your computation with the Dask dashboard
print(client.dashboard_link)
```

To use dask to compute shapley values you would then use:

```python
from joblib import parallel_config

# We use the 'dask' backend, limit the number of concurrent jobs to 4
with parallel_config(backend="dask", n_jobs=4):
    valuation.fit(training_set)
```

## Conclusion

By following this guide, you've learned how to speed up
data valuation algorithms using various parallelization methods.
For more advanced configurations, refer to the official
[joblib](https://joblib.readthedocs.io/en/stable/), [Ray](https://ray.io),
and [Dask](https://docs.dask.org/en/stable/) documentation.

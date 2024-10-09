from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from pydvl.valuation.dataset import Dataset

__all__ = [
    "load_spotify_dataset",
    "load_wine_dataset",
    "synthetic_classification_dataset",
    "decision_boundary_fixed_variance_2d",
]


def load_spotify_dataset(
    val_size: float,
    test_size: float,
    min_year: int = 2014,
    target_column: str = "popularity",
    random_state: int = 24,
) -> Tuple[List[NDArray], List[NDArray], List[NDArray]]:
    """Loads (and downloads if not already cached) the spotify music dataset.
    More info on the dataset can be found at
    https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify.

    If this method is called within the CI pipeline, it will load a reduced
    version of the dataset for testing purposes.

    Args:
        val_size: size of the validation set
        test_size: size of the test set
        min_year: minimum year of the returned data
        target_column: column to be returned as y (labels)
        random_state: fixes sklearn random seed

    Returns:
        Tuple with 3 elements, each of one of them is a list with [input_data, target_labels]
    """
    root_dir_path = Path(__file__).parent.parent.parent
    file_path = root_dir_path / "data/top_hits_spotify_dataset.csv"
    if file_path.exists():
        data = pd.read_csv(file_path)
    else:
        url = "https://raw.githubusercontent.com/aai-institute/pyDVL/develop/data/top_hits_spotify_dataset.csv"
        data = pd.read_csv(url)
        data.to_csv(file_path, index=False)

    data = data[data["year"] > min_year]
    data["genre"] = data["genre"].astype("category").cat.codes
    y = data[target_column]
    X = data.drop(target_column, axis=1)

    # Drop the 'song' column
    X = X.drop(columns=["song"])

    # Train, val, test splits
    X, X_test, y, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )

    return (
        [X_train, y_train],
        [X_val, y_val],
        [X_test, y_test],
    )


def load_wine_dataset(
    train_size: float, test_size: float, random_state: Optional[int] = None
) -> Tuple[
    Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], List[str]
]:
    """Loads the sklearn wine dataset. More info can be found at
    https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset.

    Args:
        train_size: fraction of points used for training dataset
        test_size: fraction of points used for test dataset
        random_state: fix random seed. If None, no random seed is set.
    Returns:
        A tuple of four elements with the first three being input and
        target values in the form of matrices of shape (N,D) the first
        and (N,) the second. The fourth element is a list containing names of
        features of the model. (FIXME doc)
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required in order to load the Wine Dataset"
        ) from e

    wine_bunch = load_wine(as_frame=True)
    x, x_test, y, y_test = train_test_split(
        wine_bunch.data,
        wine_bunch.target,
        train_size=1 - test_size,
        random_state=random_state,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size=train_size / (1 - test_size), random_state=random_state
    )
    x_transformer = MinMaxScaler()

    transformed_x_train = x_transformer.fit_transform(x_train)
    transformed_x_test = x_transformer.transform(x_test)

    transformed_x_train = torch.tensor(transformed_x_train, dtype=torch.float)
    transformed_y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)

    transformed_x_test = torch.tensor(transformed_x_test, dtype=torch.float)
    transformed_y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)

    transformed_x_val = x_transformer.transform(x_val)
    transformed_x_val = torch.tensor(transformed_x_val, dtype=torch.float)
    transformed_y_val = torch.tensor(y_val.to_numpy(), dtype=torch.long)
    return (
        (transformed_x_train, transformed_y_train),
        (transformed_x_val, transformed_y_val),
        (transformed_x_test, transformed_y_test),
        wine_bunch.feature_names,
    )


def synthetic_classification_dataset(
    mus: np.ndarray,
    sigma: float,
    num_samples: int,
    train_size: float,
    test_size: float,
    random_seed=None,
) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
    """Sample from a uniform Gaussian mixture model.

    Args:
        mus: 2d-matrix [CxD] with the means of the components in the rows.
        sigma: Standard deviation of each dimension of each component.
        num_samples: The number of samples to generate.
        train_size: fraction of points used for training dataset
        test_size: fraction of points used for test dataset
        random_seed: fix random seed. If None, no random seed is set.

    Returns:
        A tuple of matrix x of shape [NxD] and target vector y of shape [N].
    """
    num_features = mus.shape[1]
    num_classes = mus.shape[0]
    gaussian_cov = sigma * np.eye(num_features)
    gaussian_chol = np.linalg.cholesky(gaussian_cov)
    y = np.random.randint(num_classes, size=num_samples)
    x = (
        np.einsum(
            "ij,kj->ki",
            gaussian_chol,
            np.random.normal(size=[num_samples, num_features]),
        )
        + mus[y]
    )
    x, x_test, y, y_test = train_test_split(
        x, y, train_size=1 - test_size, random_state=random_seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size=train_size / (1 - test_size), random_state=random_seed
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def decision_boundary_fixed_variance_2d(
    mu_1: NDArray, mu_2: NDArray
) -> Callable[[NDArray], NDArray]:
    """
    Closed-form solution for decision boundary dot(a, b) + b = 0 with fixed variance.

    Args:
        mu_1: First mean.
        mu_2: Second mean.

    Returns:
        A callable which converts a continuous line (-infty, infty) to the decision boundary in feature space.
    """
    a = np.asarray([[0, 1], [-1, 0]]) @ (mu_2 - mu_1)
    b = (mu_1 + mu_2) / 2
    a = a.reshape([1, -1])
    return lambda z: z.reshape([-1, 1]) * a + b  # type: ignore

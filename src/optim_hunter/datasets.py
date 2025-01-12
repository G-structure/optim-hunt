"""Utilities for generating datasets for linear and non linear regression."""

from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3
from sklearn.model_selection import train_test_split


def get_dataset_friedman_1(
    random_state: int = 1
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generate and return the Friedman #1 dataset.

    This function creates a dataset using the `make_friedman1` function from
    sklearn, which is a synthetic dataset used for regression tasks. The dataset
    is split into training and testing sets, with the training set containing 50
    samples and the testing set containing 1 sample. The features and output
    values are rounded to two decimal places for simplicity.

    Args:
        random_state (int): Controls the randomness of the dataset generation
                          and the train-test split. Default is 1.

    Returns:
        tuple: A tuple containing four elements:
            - x_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training labels.
            - x_test (pd.DataFrame): Testing features.
            - y_test (pd.Series): Testing labels.

    """
    # The data from sklearn
    r_data, r_values = cast(
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
            make_friedman1(n_samples=51, noise=0, random_state=random_state)
        )

    # Create a dataframe
    features = {f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}
    df = pd.DataFrame({**features, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = pd.Series(df['Output'])

    # Round the values to 2 decimal places
    x = pd.DataFrame(np.round(x, 2))
    y = pd.Series(np.round(y, 2))

    # Do a random split
    x_train, x_test, y_train, y_test = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(x, y, test_size=1, random_state=random_state)
    )

    # Extract subsets
    x_train = pd.DataFrame(cast(pd.DataFrame, x_train.iloc[:50]))
    y_train = pd.Series(cast(pd.Series, y_train.iloc[:50]))
    x_test = pd.DataFrame(cast(pd.DataFrame, x_test.iloc[:1]))
    y_test = pd.Series(cast(pd.Series, y_test.iloc[:1]))

    return x_train, y_train, x_test, y_test

def get_dataset_friedman_2(
    random_state: int = 1
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generate and return the Friedman #2 dataset.

    This function creates a dataset using the `make_friedman2` function from
    sklearn, which is a synthetic dataset used for regression tasks. The
    dataset is split into training and testing sets, with the training set
    containing 50 samples and the testing set containing 1 sample. The features
    and output values are rounded to two decimal places for simplicity.

    Args:
        random_state (int): Controls the randomness of the dataset generation
                            and the train-test split. Default is 1.

    Returns:
        tuple: A tuple containing four elements:
            - x_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training labels.
            - x_test (pd.DataFrame): Testing features.
            - y_test (pd.Series): Testing labels.

    """
    # The data from sklearn
    r_data, r_values = cast(
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        make_friedman2(n_samples=51, noise=0, random_state=random_state)
    )

    # Create a dataframe
    features = {f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}
    df = pd.DataFrame({**features, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = pd.Series(df['Output'])

    # Round the values to 2 decimal places
    # Not mandatory, but helps to: (1) Keep the costs low, (2) Work with the
    # same numbers of examples with models that have a smaller context
    x = pd.DataFrame(np.round(x, 2))
    y = pd.Series(np.round(y, 2))

    # Do a random split
    x_train, x_test, y_train, y_test = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(x, y, test_size=1, random_state=random_state)
    )

    # Extract subsets
    x_train = pd.DataFrame(cast(pd.DataFrame, x_train.iloc[:50]))
    y_train = pd.Series(cast(pd.Series, y_train.iloc[:50]))
    x_test = pd.DataFrame(cast(pd.DataFrame, x_test.iloc[:1]))
    y_test = pd.Series(cast(pd.Series, y_test.iloc[:1]))

    return x_train, y_train, x_test, y_test

def get_dataset_friedman_3(
    random_state: int = 1
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generate and return the Friedman #3 dataset.

    This function creates a dataset using the `make_friedman3` function from
    sklearn, which is a synthetic dataset used for regression tasks. The
    dataset is split into training and testing sets, with the training set
    containing 50 samples and the testing set containing 1 sample. The features
    and output values are rounded to two decimal places for simplicity.

    Args:
        random_state (int): Controls the randomness of the dataset generation
                          and the train-test split. Default is 1.

    Returns:
        tuple: A tuple containing four elements:
            - x_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training labels.
            - x_test (pd.DataFrame): Testing features.
            - y_test (pd.Series): Testing labels.

    """
    # The data from sklearn
    r_data, r_values = cast(
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        make_friedman3(n_samples=51, noise=0, random_state=random_state)
    )

    # Create a dataframe
    features = {f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}
    df = pd.DataFrame({**features, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = pd.Series(df['Output'])

    # Round the values to 2 decimal places
    x = pd.DataFrame(np.round(x, 2))
    y = pd.Series(np.round(y, 2))

    # Do a random split
    x_train, x_test, y_train, y_test = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(x, y, test_size=1, random_state=random_state)
    )

    # Extract subsets
    x_train = pd.DataFrame(cast(pd.DataFrame, x_train.iloc[:50]))
    y_train = pd.Series(cast(pd.Series, y_train.iloc[:50]))
    x_test = pd.DataFrame(cast(pd.DataFrame, x_test.iloc[:1]))
    y_test = pd.Series(cast(pd.Series, y_test.iloc[:1]))

    return x_train, y_train, x_test, y_test

def get_original2(random_state=1, max_train=64, max_test=32, noise_level=0.0, **kwargs):
    """
    Adapted from Friedman 2
    """
    generator = np.random.RandomState(random_state)

    n_samples = max_train + max_test

    x = generator.uniform(size=(n_samples, 4))
    x[:, 0] *= 3
    x[:, 1] *= 52 * np.pi
    x[:, 1] += 4 * np.pi
    x[:, 2] *= 2
    x[:, 3] *= 10
    x[:, 3] += 1

    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x = np.round(x, round_value)
        y_fn = lambda x: np.round((x[0] ** 4 + (x[1] * x[2] - 2 / (np.sqrt(x[1]) * np.sqrt(x[3]))) ** 2) ** 0.75, round_value)
    else:
        y_fn = lambda x: (x[0] ** 4 + (x[1] * x[2] - 2 / (np.sqrt(x[1]) * np.sqrt(x[3]))) ** 2) ** 0.75


    y = np.array([y_fn(point) for point in x]) + noise_level * generator.standard_normal(size=(n_samples))

    r_data   = x
    r_values = y

    df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = df['Output']

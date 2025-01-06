from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def get_dataset_friedman_1(random_state=1):
    """Generate and return the Friedman #1 dataset.

    This function creates a dataset using the `make_friedman1` function from sklearn,
    which is a synthetic dataset used for regression tasks. The dataset is split into
    training and testing sets, with the training set containing 50 samples and the
    testing set containing 1 sample. The features and output values are rounded to
    two decimal places for simplicity.

    Args:
        random_state (int): Controls the randomness of the dataset generation and
                            the train-test split. Default is 1.

    Returns:
        tuple: A tuple containing four elements:
            - x_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training labels.
            - x_test (pd.DataFrame): Testing features.
            - y_test (pd.Series): Testing labels.

    """
    # The data from sklearn
    r_data, r_values = make_friedman1(n_samples=51, noise=0, random_state=random_state)

    # Create a dataframe
    df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = df['Output']

    # Round the values to 2 decimal places
    x = np.round(x, 2)
    y = np.round(y, 2)

    # Do a random split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1, random_state=random_state)

    x_train = x_train.iloc[:50]
    y_train = y_train.iloc[:50]
    x_test  = x_test.iloc[:1]
    y_test  = y_test.iloc[:1]

    return x_train, y_train, x_test, y_test

def get_dataset_friedman_2(random_state=1):
    """Generate and return the Friedman #2 dataset.

    This function creates a dataset using the `make_friedman2` function from sklearn,
    which is a synthetic dataset used for regression tasks. The dataset is split into
    training and testing sets, with the training set containing 50 samples and the
    testing set containing 1 sample. The features and output values are rounded to
    two decimal places for simplicity.

    Args:
        random_state (int): Controls the randomness of the dataset generation and
                            the train-test split. Default is 1.

    Returns:
        tuple: A tuple containing four elements:
            - x_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training labels.
            - x_test (pd.DataFrame): Testing features.
            - y_test (pd.Series): Testing labels.

    """
    # The data from sklearn
    r_data, r_values = make_friedman2(n_samples=51, noise=0, random_state=random_state)

    # Create a dataframe; Not mandatory, but makes things easier
    df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = df['Output']

    # Round the values to 2 decimal places
    # Not mandatory, but helps to: (1) Keep the costs low, (2) Work with the same numbers of examples with models that have a smaller context (e.g., Yi, Llama, etc)
    x = np.round(x, 2)
    y = np.round(y, 2)

    # Do a random split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1, random_state=random_state)

    x_train = x_train.iloc[:50]
    y_train = y_train.iloc[:50]
    x_test  = x_test.iloc[:1]
    y_test  = y_test.iloc[:1]

    return x_train, y_train, x_test, y_test

def get_dataset_friedman_3(random_state=1):
    """Generate and return the Friedman #3 dataset.

    This function creates a dataset using the `make_friedman3` function from sklearn,
    which is a synthetic dataset used for regression tasks. The dataset is split into
    training and testing sets, with the training set containing 50 samples and the
    testing set containing 1 sample. The features and output values are rounded to
    two decimal places for simplicity.

    Args:
        random_state (int): Controls the randomness of the dataset generation and
                            the train-test split. Default is 1.

    Returns:
        tuple: A tuple containing four elements:
            - x_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training labels.
            - x_test (pd.DataFrame): Testing features.
            - y_test (pd.Series): Testing labels.

    """
    # The data from sklearn
    r_data, r_values = make_friedman3(n_samples=51, noise=0, random_state=random_state)

    # Create a dataframe
    df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = df['Output']

    # Round the values to 2 decimal places
    x = np.round(x, 2)
    y = np.round(y, 2)

    # Do a random split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1, random_state=random_state)

    x_train = x_train.iloc[:50]
    y_train = y_train.iloc[:50]
    x_test  = x_test.iloc[:1]
    y_test  = y_test.iloc[:1]

    return x_train, y_train, x_test, y_test

"""Provides various scikit-learn regression models.

implementing standard training and prediction workflows. Available models
include linear regression, SVR, random forests, gradient boosting
and additional regression algorithms.

Source: # Source https://github.com/robertvacareanu/llm4regression/blob/37b98e60170d5b68399915440b72dd9bd88b702e/src/regressors/sklearn_regressors.py

"""

import random
from typing import Any, Callable, Dict, List, Optional, Union, cast
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)
from sklearn.svm import SVR
from transformer_lens import HookedTransformer

from optim_hunter.utils import prepare_prompt

import numpy.typing as npt



def create_llm_regressor(
    model: HookedTransformer,
    model_name: str,
    max_new_tokens: int = 1,
    temperature: float = 0.0
) -> Callable[
    [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int],
    Dict[str, Union[str, pd.DataFrame, pd.Series, List[Optional[float]]]]
]:
    """Create an LLM regressor with specified parameters.

    Args:
        model: The language model to use for regression
        model_name: Name identifier for the model
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature for generation

    Returns:
        Callable: A function that implements LLM regression with the specified
            configuration

    """

    def llm_regressor(
            x_train: pd.DataFrame,
            x_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
            random_state: int = 1
    ) -> Dict[str, Union[
        str, pd.DataFrame, pd.Series, List[Optional[float]]]]:
            """Run regression using the configured LLM.

            Args:
                x_train: Training features
                x_test: Test features
                y_train: Training labels
                y_test: Test labels
                random_state: Random seed for reproducibility

            Returns:
                Dict containing model info and predictions

            """
            # Prepare prompt from training data and test input
            prompt = prepare_prompt(x_train, y_train, x_test)

            # Generate prediction
            pred_text = str(model.generate(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            ))

            try:
                # Extract generated value and convert to float
                generated_part = str(pred_text.replace(prompt, "").strip())
                y_predict = float(generated_part)
            except ValueError:
                print(f"Warning: Could not parse model prediction: {pred_text}")
                y_predict = None

            # Convert test labels to numpy array with explicit type
            y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

            return {
                "model_name": f"llm-{model_name}",
                "x_train": x_train,
                "x_test": x_test,
                "y_train": y_train,
                "y_test": pd.Series(y_test_np),
                "y_predict": [y_predict],
            }

    return llm_regressor


def linear_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Linear regression model using scikit-learn's implementation.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "linear_regression",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def ridge(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Ridge regression model using scikit-learn's implementation.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = Ridge(random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "ridge",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def lasso(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Lasso regression model using scikit-learn's implementation.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = Lasso(random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "lasso",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def mlp_universal_approximation_theorem1(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Single hidden layer MLP based on Universal Approximation Theorem.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = MLPRegressor(
        hidden_layer_sizes=(10,),
        activation="relu",
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "mlp_uat_1",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def mlp_universal_approximation_theorem2(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Multi-layer perceptron with 100 hidden units based on
    Universal Approximation Theorem.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = MLPRegressor(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "mlp_uat_2",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def mlp_universal_approximation_theorem3(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Multi-layer perceptron with 1000 hidden units based on
    Universal Approximation Theorem.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = MLPRegressor(
        hidden_layer_sizes=(1000,),
        activation="relu",
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "mlp_uat_3",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def mlp_deep1(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Two-layer MLP with 10 units per layer.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = MLPRegressor(
        hidden_layer_sizes=(10, 10),
        activation="relu",
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "mlp_deep1",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def mlp_deep2(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Three-layer MLP with 10, 20, and 10 units respectively.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = MLPRegressor(
        hidden_layer_sizes=(10, 20, 10),
        activation="relu",
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "mlp_deep2",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def mlp_deep3(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Five-layer MLP with 10, 20, 30, 20, and 10 units respectively.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = MLPRegressor(
        hidden_layer_sizes=(10, 20, 30, 20, 10),
        activation="relu",
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "mlp_deep3",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def random_forest(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Random Forest Regressor.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = RandomForestRegressor(max_depth=3, random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "random_forest",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def bagging(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Bagging Regressor using scikit-learn's implementation.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = BaggingRegressor(random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "bagging",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def gradient_boosting(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Gradient Boosting Regressor using scikit-learn's implementation.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "gradient_boosting",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def adaboost(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """AdaBoost Regressor using scikit-learn's implementation.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = AdaBoostRegressor(n_estimators=100, random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "adaboost",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def voting(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Voting Regressor using scikit-learn's implementation.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    # Define base models for voting
    base_models = [
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(random_state=random_state)),
        ('svr', SVR())
    ]
    model = VotingRegressor(estimators=base_models)
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "voting",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


# def stacking(x_train, x_test, y_train, y_test, random_state=1):
#     """
#     Stacking Regressor
#     """
#     model = StackingRegressor()
#     model.fit(x_train, y_train)
#     y_predict = model.predict(x_test)
#     y_test    = y_test.to_numpy()

#     return {
#         'model_name': 'stacking',
#         'x_train'   : x_train,
#         'x_test'    : x_test,
#         'y_train'   : y_train,
#         'y_test'    : y_test,
#         'y_predict' : y_predict,
#     }


def bayesian_regression1(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Bayesian Ridge Regression with polynomial features.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = make_pipeline(
        PolynomialFeatures(degree=10, include_bias=False),
        StandardScaler(),
        BayesianRidge(),
    )

    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "bayesian_regression",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def svm_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Support Vector Machine Regressor using scikit-learn's implementation.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = SVR()
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "svm",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def svm_and_scaler_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Support Vector Machine Regressor with standard scaling.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = make_pipeline(StandardScaler(), SVR())
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "svm_w_s",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def knn_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """K-Nearest Neighbors Regressor using scikit-learn's implementation.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels

        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = KNeighborsRegressor()
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "knn",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def knn_regression_v2(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """K-Nearest Neighbors Regressor using distance weighting.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = KNeighborsRegressor(weights="distance")
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "knn_v2",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def knn_regression_v3(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """KNN regression with 3 neighbors and distance weighting.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = KNeighborsRegressor(n_neighbors=3, weights="distance")
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "knn_v3",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def knn_regression_v4(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """KNN regression with 1 neighbor and distance weighting.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = KNeighborsRegressor(n_neighbors=1, weights="distance")
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "knn_v4",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def knn_regression_v5_adaptable(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Adapt KNN model neighbor count based on training set size.

    Use fewer neighbors with small training sets and more neighbors as the
    training data grows. This allows the model to work effectively across
    different dataset sizes.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    if x_train.shape[0] < 3:
        n_neighbors = 1
    elif x_train.shape[0] < 7:
        n_neighbors = 3
    else:
        n_neighbors = 5

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "knn_v5_adaptable",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def knn_regression_generic(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    knn_kwargs: Dict[str, Any]
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Perform KNN regression with configurable parameters.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_name: Name identifier for the model
        knn_kwargs: Dict of KNeighborsRegressor parameters

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = KNeighborsRegressor(**knn_kwargs)
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": model_name,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def knn_regression_search() -> List[Callable[
        [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int],
        Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[
            np.float64]]]]]:
    """Generate KNN models with different hyperparameter combinations.

    This function creates multiple KNN regressor variants by combining
    different values for n_neighbors, weights, and p parameter (distance
    metric).

    Returns:
        List[Callable]: List of functions that each implement a
            different KNN regressor variant. Each function takes standard


            regression inputs (x_train, x_test, y_train, y_test) and
            returns a results dictionary.

    """
    idx = 0
    knn_fns: List[Callable[
        [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int],
        Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]
    ]] = []

    for n_neighbors_val in [1, 2, 3, 5, 7, 9, 11]:
        for weights_val in ["uniform", "distance"]:
            for p_val in [0.25, 0.5, 1, 1.5, 2]:
                idx += 1
                def create_knn(n: int, w: str, p: float, i: int) -> Callable[
                    [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int],
                    Dict[str, Union[str, pd.DataFrame, pd.Series,
                        npt.NDArray[np.float64]]]]:
                    def knn_fn(x_train: pd.DataFrame, x_test: pd.DataFrame,
                             y_train: pd.Series, y_test: pd.Series,
                             random_state: int = 1) -> Dict[str, Union[str,
                             pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
                        return knn_regression_generic(
                            x_train,
                            x_test,
                            y_train,
                            y_test,
                            "knn_search_" + str(i),
                            {
                                "n_neighbors": n,
                                "weights": w,
                                "p": p
                            }
                        )
                    return knn_fn
                knn_fns.append(create_knn(n_neighbors_val, weights_val,
                    p_val, idx))
    return knn_fns


def kernel_ridge_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Kernel Ridge regression using scikit-learn implementation.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    model = KernelRidge()
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "kernel_ridge",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def lr_with_polynomial_features_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1,
    **kwargs: Dict[str, Any]
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Linear regression with polynomial features transformation.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments including polynomial degree

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    degree = cast(int, kwargs.get("degree", 2))

    # Create pipeline that transforms data using PolynomialFeatures then
    # applies Linear Regression
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree)),
            ("linear", LinearRegression()),
        ]
    )

    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "lr_with_polynomial_features",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def spline_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1,
    **kwargs: Dict[str, Any]
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Spline regression using scikit-learn implementation.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments including n_knots and degree

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    # Same defaults as SplineTransformer
    n_knots = cast(int, kwargs.get("degree", 5))
    # Same defaults as SplineTransformer
    degree = cast(int, kwargs.get("degree", 3))

    # Create pipeline that transforms data using SplineTransformer then
    # applies Linear Regression
    model = Pipeline(
        [
            ("spline", SplineTransformer(n_knots=n_knots, degree=degree)),
            ("linear", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)
    y_predict = cast(npt.NDArray[np.float64], model.predict(x_test))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "spline",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def baseline_average(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train
    : pd.Series,
    y_test: pd.Series,
    random_state: int = 1,
    **kwargs: Dict[str, Any]
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Predict the mean value of the training set.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments (unused)

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    pred = float(np.mean(y_train))
    y_predict = cast(npt.NDArray[np.float64],
                     np.array([pred for _ in range(len(y_test))]))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "average",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def baseline_last(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1,
    **kwargs: Dict[str, Any]
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Predict using the last training value as sample baseline.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments (unused)

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    pred = float(y_train.values[-1]) # Use values instead of iloc
    y_predict = cast(npt.NDArray[np.float64],
        np.array([pred for _ in range(len(y_test))]))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "last",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def baseline_random(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1,
    **kwargs: Dict[str, Any]
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Randomly sample training values as predictions.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments (unused)

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions


    """
    r = random.Random(random_state)
    y_train_list: list[float] = y_train.values.tolist()
    y_predict = cast(
        npt.NDArray[np.float64],
        np.array([r.choice(y_train_list) for _ in range(len(y_test))])
    )
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "random",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def baseline_constant(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1,
    **kwargs: Dict[str, Any]
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Predict a constant value.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments including constant value

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    pred_val = kwargs["constant_prediction_value"]
    y_predict = cast(npt.NDArray[np.float64], np.full(len(y_test), pred_val))
    y_test_np = cast(npt.NDArray[np.float64], y_test.to_numpy())

    return {
        "model_name": "constant_prediction",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test_np),
        "y_predict": y_predict,
    }


def linear_regression_manual_gd(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 1,
    **kwargs: Dict[str, Any]
) -> Dict[str, Union[str, pd.DataFrame, pd.Series, npt.NDArray[np.float64]]]:
    """Linear regression using manual gradient descent steps.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments including:
            steps: Number of gradient descent steps (default: 2)
            learning_rate: Step size for gradient descent (default: 0.01)

    Returns:
        dict: Results dictionary containing model name, input data and
            predictions

    """
    steps = cast(int, kwargs.get("steps", 2))
    learning_rate = cast(float, kwargs.get("learning_rate", 0.01))

    # Convert to numpy arrays if they aren't already
    x_train_np = cast(npt.NDArray[np.float64], (
        x_train.to_numpy()
        if hasattr(x_train, "to_numpy")
        else np.array(x_train)
    ))
    y_train_np = cast(npt.NDArray[np.float64], (
        y_train.to_numpy()
        if hasattr(y_train, "to_numpy")
        else np.array(y_train)
    ))
    x_test_np = cast(npt.NDArray[np.float64], (
        x_test.to_numpy()
        if hasattr(x_test, "to_numpy")
        else np.array(x_test)
    ))
    # y_test_np = cast(npt.NDArray[np.float64], (
    #     y_test.to_numpy()
    #     if hasattr(y_test, "to_numpy")
    #     else np.array(y_test)
    # ))

    # Initialize parameters (weights and bias)
    n_features = x_train_np.shape[1]
    weights = np.zeros(n_features)
    bias = 0

    # Perform gradient descent steps
    m = len(x_train_np)
    for _ in range(steps):
        # Forward pass
        y_pred = np.dot(x_train_np, weights) + bias

        # Compute gradients
        dw = (1 / m) * np.dot(x_train_np.T, (y_pred - y_train_np))
        db = (1 / m) * np.sum(y_pred - y_train_np)

        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

    # Make predictions on test set
    y_predict = cast(npt.NDArray[np.float64], np.dot(x_test_np, weights) + bias)

    return {
        "model_name": f"linear_regression_gd_{steps}_steps",
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": pd.Series(y_test),
        "y_predict": y_predict,
    }


def create_linear_regression_gd_variants(
    steps_options: List[int] = [1, 2, 3, 4],
    learning_rates: List[float] = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
    init_weights_options: List[str] = ["zeros", "ones", "random",
        "random_uniform"],
    momentum_values: List[float] = [0.0, 0.5, 0.9],  # 0.0 means no momentum
    lr_schedules: List[str] = ["constant", "linear_decay", "exponential_decay"]
) -> List[Callable[
        [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int],
        Dict[str, Union[str, pd.DataFrame, pd.Series,
             npt.NDArray[np.float64]]]]]:
    """Create multiple variants of linear regression with gradient descent.

    Args:
        steps_options (List[int]): List of gradient descent step counts to try
        learning_rates (List[float]): List of learning rates to try
        init_weights_options (List[str]): List of weight initialization
            strategies to try. Options: "zeros", "ones", "random",
            "random_uniform"
        momentum_values (List[float]): List of momentum values to try. 0.0
            means no momentum
        lr_schedules (List[str]): List of learning rate schedule strategies
            to try. Options: "constant", "linear_decay", "exponential_decay"

    Returns:
        List[Callable]: List of functions, each implementing linear regression
            with specific hyperparameter combinations. Each function takes

            the following arguments:
                x_train (pd.DataFrame): Training features
                x_test (pd.DataFrame): Test features
                y_train (pd.Series): Training labels
                y_test (pd.Series): Test labels
                random_state (int): Random seed
            And returns a dict with model results

    """
    variants: List[Callable[
        [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int],
        Dict[str, Union[str, pd.DataFrame, pd.Series,
                       npt.NDArray[np.float64]]]]] = []

    # Create all combinations of hyperparameters
    configs: List[Dict[str, Union[int, float, str]]] = []
    for steps in steps_options:
        for lr in learning_rates:
            for init in init_weights_options:
                for momentum in momentum_values:
                    for lr_schedule in lr_schedules:
                        configs.append(
                            {
                                "steps": steps,
                                "learning_rate": lr,
                                "init_weights": init,
                                "momentum": momentum,
                                "lr_schedule": lr_schedule,
                            }
                        )

    def create_gd_function(
        steps: int,
        learning_rate: float,
        init_weights: str,
        momentum: float,
        lr_schedule: str
    ) -> Callable[
        [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int],
        Dict[str, Union[str, pd.DataFrame, pd.Series,
             npt.NDArray[np.float64]]]]:

        def linear_regression_gd(
                    x_train: pd.DataFrame,
                    x_test: pd.DataFrame,
                    y_train: pd.Series,
                    y_test: pd.Series,
                    random_state: int = 1
                ) -> Dict[str, Union[str, pd.DataFrame, pd.Series,
                                   npt.NDArray[np.float64]]]:
                    # Convert to numpy arrays
                    x_train_np = cast(npt.NDArray[np.float64],
                        x_train.to_numpy() if hasattr(x_train, "to_numpy")
                        else np.array(x_train))
                    y_train_np = cast(npt.NDArray[np.float64],
                        y_train.to_numpy() if hasattr(y_train, "to_numpy")
                        else np.array(y_train))
                    x_test_np = cast(npt.NDArray[np.float64],
                        x_test.to_numpy() if hasattr(x_test, "to_numpy")
                        else np.array(x_test))
                    y_test_np = cast(npt.NDArray[np.float64],
                        y_test.to_numpy() if hasattr(y_test, "to_numpy")
                        else np.array(y_test))

                    # Initialize parameters
                    n_features = x_train_np.shape[1]
                    np.random.seed(random_state)

                    if init_weights == "zeros":
                        weights = np.zeros(n_features)
                    elif init_weights == "ones":
                        weights = np.ones(n_features)
                    elif init_weights == "random":
                        weights = np.random.randn(n_features) * 0.01
                    else: # random_uniform
                        weights = np.random.uniform(-0.01, 0.01, n_features)

                    bias = 0.0

                    # Initialize momentum vectors
                    v_weights = np.zeros_like(weights)
                    v_bias = 0.0

                    # Perform gradient descent steps
                    m = len(x_train_np)
                    for step in range(steps):
                        # Compute effective learning rate based on schedule
                        if lr_schedule == "constant":
                            current_lr = learning_rate
                        elif lr_schedule == "linear_decay":
                            current_lr = learning_rate * (1 - step / steps)
                        else: # exponential_decay
                            current_lr = learning_rate * (0.95**step)

                        # Forward pass
                        y_pred = cast(npt.NDArray[np.float64],
                            np.dot(x_train_np, weights) + bias)

                        # Compute gradients
                        dw = (1 / m) * np.dot(x_train_np.T,
                            (y_pred - y_train_np))
                        db = (1 / m) * np.sum(y_pred - y_train_np)

                        # Apply momentum if using it
                        if momentum > 0:
                            v_weights = momentum * v_weights - current_lr * dw
                            v_bias = momentum * v_bias - current_lr * db
                            weights += v_weights
                            bias += v_bias
                        else:
                            weights = weights - current_lr * dw
                            bias = bias - current_lr * db

                    # Make predictions
                    y_predict = cast(npt.NDArray[np.float64],
                        np.dot(x_test_np, weights) + bias)

                    return {
                        "model_name": f"lr_gd_s{steps}_lr{learning_rate}_"
                                    f"i{init_weights}_m{momentum}_sc{lr_schedule}",
                        "x_train": x_train,
                        "x_test": x_test,
                        "y_train": y_train,
                        "y_test": pd.Series(y_test_np),
                        "y_predict": y_predict,
                    }

        return linear_regression_gd

    # Create a function for each configuration
    for config in configs:
        variants.append(
            create_gd_function(
                steps=cast(int, config["steps"]),
                learning_rate=cast(float, config["learning_rate"]),
                init_weights=cast(str, config["init_weights"]),
                momentum=cast(float, config["momentum"]),
                lr_schedule=cast(str, config["lr_schedule"]),
            )
        )

    return variants

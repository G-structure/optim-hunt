"""Implements various linear regression methods with detailed computation tracking."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class RegressionResults:
    """Container for regression method results.

    Attributes:
        model_name: Identifier for the regression method
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        y_predict: Predicted values for x_test
        intermediates: Optional dict of intermediate calculations

    """
    model_name: str
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_predict: npt.NDArray[np.float64]
    intermediates: Optional[Dict[str, Any]] = field(default=None)

    def to_dict(self) -> Dict[str, Union[str, pd.DataFrame, pd.Series,
                                       npt.NDArray[np.float64]]]:
        """Convert results to dictionary format."""
        result = {
            "model_name": self.model_name,
            "x_train": self.x_train,
            "x_test": self.x_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "y_predict": self.y_predict
        }
        if self.intermediates is not None:
            result["intermediates"] = self.intermediates
        return result

def solve_ols(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs: Any
) -> RegressionResults:
    """Perform Ordinary Least Squares (OLS) regression.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        **kwargs: Additional keyword arguments (unused)

    Returns:
        RegressionResults containing:
            - intermediates: Dict of intermediate calculations including:
                - Design Matrix (XᵀX)
                - Pseudoinverse ((XᵀX)^(-1))
                - Weighted Feature Matrix (Xᵀy)
                - Weights (w)
            - All standard RegressionResults fields:
                - model_name: Model identifier string
                - x_train: Training feature matrix
                - x_test: Test feature matrix
                - y_train: Training target values
                - y_test: Test target values
                - y_predict: Model predictions on test data

    """
    # Convert to numpy arrays for computation
    x = cast(npt.NDArray[np.float64], x_train.to_numpy())
    y = cast(npt.NDArray[np.float64], y_train.to_numpy()).reshape(-1, 1)
    x_test_np = cast(npt.NDArray[np.float64], x_test.to_numpy())

    # Add bias terms
    x = np.hstack((np.ones((x.shape[0], 1), dtype=np.float64), x))
    x_test_np = np.hstack((np.ones((x_test_np.shape[0], 1), dtype=np.float64), x_test_np))

    # Calculate intermediate results
    design_matrix = x.T @ x
    pseudoinverse = np.linalg.inv(design_matrix)
    weighted_feature_matrix = x.T @ y
    weights = pseudoinverse @ weighted_feature_matrix

    # Calculate prediction
    y_pred = x_test_np @ weights

    return RegressionResults(
        model_name="ols",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_predict=y_pred.flatten(),
        intermediates={
            "Design Matrix (XᵀX)": design_matrix,
            "Pseudoinverse ((XᵀX)^(-1))": pseudoinverse,
            "Weighted Feature Matrix (Xᵀy)": weighted_feature_matrix,
            "Weights (w)": weights
        }
    )

def solve_gradient_descent(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs: Any
) -> RegressionResults:
    """Perform Gradient Descent for linear regression.

    Args:
        x_train: Training features
        x_test: Test features to predict on
        y_train: Training labels (target values)
        y_test: Test labels
        **kwargs: Additional keyword arguments like:
            learning_rate: Step size for updates (default 0.01)
            max_iterations: Max iterations (default 1000)
            tolerance: Convergence threshold (default 1e-6)

    Returns:
        RegressionResults containing:
            - model_name: Name identifier for the gradient descent method
            - x_train: Training features matrix provided as input
            - x_test: Test features matrix provided as input
            - y_train: Training target values provided as input
            - y_test: Test target values provided as input
            - y_predict: Model's predicted values for x_test
            - intermediates: Dict of intermediate calculations including:
                - Initial Weights (w₀): Starting weight values
                - Gradients (∇L): Gradient of loss at each iteration
                - Learning Rate (α): Step size used for weight updates
                - Weight Updates (Δw): Changes in weights at each iteration
                - Intermediate Weights: Weight values after each update
                - Final Weights (w): Final converged weight values
                - Number of Iterations: Total iterations until convergence

    """
    # Get keyword args with defaults
    learning_rate = kwargs.get("learning_rate", 0.01)
    max_iterations = kwargs.get("max_iterations", 1000)
    tolerance = kwargs.get("tolerance", 1e-6)

    # Convert to numpy arrays for computation
    x = cast(npt.NDArray[np.float64], x_train.to_numpy())
    y = cast(npt.NDArray[np.float64], y_train.to_numpy()).reshape(-1, 1)
    x_test_np = cast(npt.NDArray[np.float64], x_test.to_numpy())

    # Add bias terms
    x = np.hstack((np.ones((x.shape[0], 1), dtype=np.float64), x))
    x_test_np = np.hstack((np.ones((x_test_np.shape[0], 1), dtype=np.float64), x_test_np))

    # Initialize weights with zeros
    weights = np.zeros((x.shape[1], 1), dtype=np.float64)
    initial_weights = weights.copy()

    # Track intermediate results
    gradients: List[npt.NDArray[np.float64]] = []
    weight_updates: List[npt.NDArray[np.float64]] = []
    intermediate_weights: List[npt.NDArray[np.float64]] = [initial_weights.flatten()]

    iteration = 0

    for iteration in range(max_iterations):
        # Compute predictions and residuals
        predictions = x @ weights
        residuals = predictions - y

        # Compute gradient
        gradient = (2 / x.shape[0]) * (x.T @ residuals)
        gradients.append(gradient.flatten())

        # Update weights
        weight_update = -learning_rate * gradient
        weights += weight_update
        weight_updates.append(weight_update.flatten())
        intermediate_weights.append(weights.flatten())

        # Check convergence
        if np.linalg.norm(gradient) < tolerance:
            break

    # Calculate prediction for test data
    y_pred = x_test_np @ weights

    return RegressionResults(
        model_name="gradient_descent",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_predict=y_pred.flatten(),
        intermediates={
            "Initial Weights (w₀)": initial_weights.flatten(),
            "Gradients (∇L)": gradients,
            "Learning Rate (α)": learning_rate,
            "Weight Updates (Δw)": weight_updates,
            "Intermediate Weights (w₁, w₂, ..., wₙ)": intermediate_weights,
            "Final Weights (w)": weights.flatten(),
            "Number of Iterations": iteration + 1
        }
    )

def solve_ridge_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs: Any
) -> RegressionResults:
    """Perform Ridge Regression with regularization.

    Args:
        x_train: Training features
        x_test: Test features to predict on
        y_train: Training labels (target values)
        y_test: Test labels
        **kwargs: Additional keyword arguments like:
            regularization_param: The regularization parameter λ (default 1.0)

    Returns:
        RegressionResults containing:
            - model_name: Name identifier for the ridge regression method
            - x_train: Training features matrix provided as input
            - x_test: Test features matrix provided as input
            - y_train: Training target values provided as input
            - y_test: Test target values provided as input
            - y_predict: Model's predicted values for x_test
            - intermediates: Dict of intermediate calculations including:
                - Regularization Term (λI): The regularization matrix
                - Modified Design Matrix (XᵀX + λI): Design matrix with regularization
                - Pseudoinverse ((XᵀX + λI)^(-1)): Inverse of modified design matrix
                - Weights (w): Final computed weights

    """
    # Get regularization parameter from kwargs
    regularization_param = kwargs.get("regularization_param", 1.0)

    # Convert to numpy arrays for computation
    x = cast(npt.NDArray[np.float64], x_train.to_numpy())
    y = cast(npt.NDArray[np.float64], y_train.to_numpy()).reshape(-1, 1)
    x_test_np = cast(npt.NDArray[np.float64], x_test.to_numpy())

    # Add bias terms
    x = np.hstack((np.ones((x.shape[0], 1), dtype=np.float64), x))
    x_test_np = np.hstack((np.ones((x_test_np.shape[0], 1), dtype=np.float64), x_test_np))

    # Compute the design matrix (XᵀX)
    design_matrix = x.T @ x

    # Create the regularization term (λI)
    identity_matrix = np.eye(design_matrix.shape[0], dtype=np.float64)
    regularization_term = regularization_param * identity_matrix
    regularization_term[0, 0] = 0  # Do not regularize the bias term

    # Compute the modified design matrix (XᵀX + λI)
    modified_design_matrix = design_matrix + regularization_term

    # Compute the pseudoinverse ((XᵀX + λI)^(-1))
    pseudoinverse = np.linalg.inv(modified_design_matrix)

    # Compute the weighted feature matrix (Xᵀy)
    weighted_feature_matrix = x.T @ y

    # Compute the weights (w = (XᵀX + λI)^(-1)Xᵀy)
    weights = pseudoinverse @ weighted_feature_matrix

    # Calculate prediction for test data
    y_pred = x_test_np @ weights

    return RegressionResults(
        model_name="ridge_regression",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_predict=y_pred.flatten(),
        intermediates={
            "Regularization Term (λI)": regularization_term,
            "Modified Design Matrix (XᵀX + λI)": modified_design_matrix,
            "Pseudoinverse ((XᵀX + λI)^(-1))": pseudoinverse,
            "Weights (w)": weights.flatten()
        }
    )

def solve_lasso_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs: Any
) -> RegressionResults:
    """Apply lasso regression using coordinate descent.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels (target values)
        y_test: Test labels
        **kwargs: Additional keyword arguments like:
            regularization_param: The regularization parameter λ (default is 1.0)
            max_iter: Maximum number of iterations (default is 1000)
            tol: Convergence tolerance (default is 1e-4)

    Returns:
        RegressionResults containing:
            - model_name: Name identifier for the lasso regression method
            - x_train: Training features matrix provided as input
            - x_test: Test features matrix provided as input
            - y_train: Training target values provided as input
            - y_test: Test target values provided as input
            - y_predict: Model's predicted values for x_test
            - intermediates: Dict of intermediate calculations including:
                - Regularization Term (λ||w||₁): L1 penalty values at each iteration
                - Soft Thresholding Steps: Weight values after each update
                - Weights (w): Weight trajectory during optimization

    """
    # Get keyword args with defaults
    regularization_param = kwargs.get("regularization_param", 1.0)
    max_iter = kwargs.get("max_iter", 1000)
    tol = kwargs.get("tol", 1e-4)

    # Convert to numpy arrays for computation
    x = cast(npt.NDArray[np.float64], x_train.to_numpy())
    y = cast(npt.NDArray[np.float64], y_train.to_numpy()).reshape(-1, 1)
    x_test_np = cast(npt.NDArray[np.float64], x_test.to_numpy())

    # Add bias terms
    x = np.hstack((np.ones((x.shape[0], 1), dtype=np.float64), x))
    x_test_np = np.hstack((np.ones((x_test_np.shape[0], 1), dtype=np.float64), x_test_np))

    # Initialize weights
    weights = np.zeros((x.shape[1], 1), dtype=np.float64)

    # Number of samples and features
    n_samples, n_features = x.shape

    # Store intermediate results
    intermediate_results: Dict[str, List[Union[float, npt.NDArray[np.float64]]]] = {
        "Regularization Term (λ||w||₁)": [],
        "Soft Thresholding Steps": [],
        "Weights (w)": []
    }

    def soft_thresholding_operator(value: float, lambda_: float) -> float:
        """Apply the soft thresholding operator."""
        if value > lambda_:
            return value - lambda_
        elif value < -lambda_:
            return value + lambda_
        else:
            return 0

    # Coordinate Descent
    for iteration in range(max_iter):
        weights_prev = weights.copy()

        for j in range(n_features):
            # Compute residual
            residual = y - x @ weights + x[:, j].reshape(-1, 1) * weights[j]

            # Compute partial correlation
            rho = np.dot(x[:, j], residual.flatten())

            # Update weight j
            if j == 0:  # Do not regularize bias term
                weights[j] = rho / n_samples
            else:
                weights[j] = soft_thresholding_operator(
                    rho / n_samples,
                    regularization_param
                )

        # Compute regularization term
        reg_term = regularization_param * np.sum(np.abs(weights[1:]))
        intermediate_results["Regularization Term (λ||w||₁)"].append(reg_term)

        # Store intermediate results
        intermediate_results["Soft Thresholding Steps"].append(
            weights.copy().flatten()
        )
        intermediate_results["Weights (w)"].append(weights.copy().flatten())

        # Check convergence
        if np.linalg.norm(weights - weights_prev, ord=1) < tol:
            break

    # Calculate prediction for test data
    y_pred = x_test_np @ weights

    return RegressionResults(
        model_name="lasso_regression",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_predict=y_pred.flatten(),
        intermediates=intermediate_results
    )

def solve_sgd(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs: Any
) -> RegressionResults:
    """Perform Stochastic Gradient Descent (SGD) optimization.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels (target values)
        y_test: Test labels
        **kwargs: Additional keyword arguments like:
            learning_rate: Learning rate (α) for weight updates (default is 0.01)
            max_iter: Maximum number of iterations (default is 100)
            batch_size: Size of the mini-batch for each step (default is 1)
            tol: Convergence tolerance (default is 1e-4)
            random_state: Random seed for reproducibility (default is 42)

    Returns:
        RegressionResults containing:
            - model_name: Name identifier for the SGD method
            - x_train: Training features matrix provided as input
            - x_test: Test features matrix provided as input
            - y_train: Training target values provided as input
            - y_test: Test target values provided as input
            - y_predict: Model's predicted values for x_test
            - intermediates: Dict of intermediate calculations including:
                - Random Sampling of Data Points: Batch indices at each step
                - Gradient of Loss for Current Sample (∇L_i): Batch gradients
                - Intermediate Weights (w₁, w₂, ..., wₙ): Weight trajectory

    """
    # Get keyword args with defaults
    learning_rate = kwargs.get("learning_rate", 0.01)
    max_iter = kwargs.get("max_iter", 100)
    batch_size = kwargs.get("batch_size", 1)
    tol = kwargs.get("tol", 1e-4)
    random_state = kwargs.get("random_state", 42)

    # Set random seed
    np.random.seed(random_state)

    # Convert to numpy arrays for computation
    x = cast(npt.NDArray[np.float64], x_train.to_numpy())
    y = cast(npt.NDArray[np.float64], y_train.to_numpy()).reshape(-1, 1)
    x_test_np = cast(npt.NDArray[np.float64], x_test.to_numpy())

    # Add bias terms
    x = np.hstack((np.ones((x.shape[0], 1), dtype=np.float64), x))
    x_test_np = np.hstack((np.ones((x_test_np.shape[0], 1), dtype=np.float64), x_test_np))

    # Initialize weights
    weights = np.zeros((x.shape[1], 1), dtype=np.float64)

    # Number of samples
    n_samples = x.shape[0]

    # Store intermediate results
    intermediate_results: Dict[str, List[npt.NDArray[np.float64]]] = {
        "Random Sampling of Data Points": [],
        "Gradient of Loss for Current Sample (∇L_i)": [],
        "Intermediate Weights (w₁, w₂, ..., wₙ)": []
    }

    # Perform SGD
    for iteration in range(max_iter):
        # Shuffle data for randomness in sampling
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            # Select mini-batch
            batch_indices = indices[start_idx:start_idx + batch_size]
            X_batch = x[batch_indices]
            y_batch = y[batch_indices]

            # Predict output for the mini-batch
            predictions = X_batch @ weights

            # Compute gradient
            gradient = -(2 / batch_size) * (X_batch.T @ (y_batch - predictions))

            # Update weights
            weights -= learning_rate * gradient

            # Store intermediate results
            intermediate_results["Random Sampling of Data Points"].append(
                batch_indices.astype(np.float64)
            )
            intermediate_results["Gradient of Loss for Current Sample (∇L_i)"].append(
                gradient.flatten()
            )
            intermediate_results["Intermediate Weights (w₁, w₂, ..., wₙ)"].append(
                weights.flatten()
            )

        # Check convergence
        if np.linalg.norm(gradient, ord=2) < tol:
            break

    # Calculate prediction for test data
    y_pred = x_test_np @ weights

    return RegressionResults(
        model_name="sgd",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_predict=y_pred.flatten(),
        intermediates=intermediate_results
    )

def solve_bayesian_linear_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs: Any
) -> RegressionResults:
    """Perform Bayesian Linear Regression with uncertainty quantification.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels (target values)
        y_test: Test labels
        **kwargs: Additional keyword arguments like:
            alpha: Precision of the prior distribution (default is 1.0)
            beta: Precision of the likelihood (default is 1.0)

    Returns:
        RegressionResults containing:
            - model_name: Name identifier for Bayesian linear regression
            - x_train: Training features matrix provided as input
            - x_test: Test features matrix provided as input
            - y_train: Training target values provided as input
            - y_test: Test target values provided as input
            - y_predict: Model's predicted values for x_test
            - intermediates: Dict of intermediate calculations including:
                - Prior Distribution (P(w)): Mean and covariance of prior
                - Likelihood (P(y|X, w)): Precision of noise model
                - Posterior Mean and Covariance: Updated distribution

    """
    # Get keyword args with defaults
    alpha = kwargs.get("alpha", 1.0)
    beta = kwargs.get("beta", 1.0)

    # Convert to numpy arrays for computation
    x = cast(npt.NDArray[np.float64], x_train.to_numpy())
    y = cast(npt.NDArray[np.float64], y_train.to_numpy()).reshape(-1, 1)
    x_test_np = cast(npt.NDArray[np.float64], x_test.to_numpy())

    # Add bias terms
    x = np.hstack((np.ones((x.shape[0], 1), dtype=np.float64), x))
    x_test_np = np.hstack((np.ones((x_test_np.shape[0], 1), dtype=np.float64), x_test_np))

    # Number of features (including bias)
    n_features = x.shape[1]

    # Compute the prior distribution P(w)
    prior_mean = np.zeros((n_features, 1), dtype=np.float64)
    prior_covariance = (1 / alpha) * np.eye(n_features, dtype=np.float64)

    # Compute the likelihood precision (β) and design matrix xᵀx
    likelihood_precision = beta
    design_matrix = x.T @ x

    # Compute the posterior covariance
    posterior_covariance = np.linalg.inv(
        prior_covariance + likelihood_precision * design_matrix
    )

    # Compute the posterior mean
    posterior_mean = posterior_covariance @ (likelihood_precision * x.T @ y)

    # Calculate prediction using posterior mean as weights
    y_pred = x_test_np @ posterior_mean

    # Store intermediate results
    intermediate_results: Dict[str, Dict[str, Union[npt.NDArray[np.float64], float]]] = {
        "Prior Distribution (P(w))": {
            "Mean": prior_mean.flatten(),
            "Covariance": prior_covariance
        },
        "Likelihood (P(y|X, w))": {
            "Precision (β)": likelihood_precision
        },
        "Posterior Mean and Covariance": {
            "Mean": posterior_mean.flatten(),
            "Covariance": posterior_covariance
        }
    }

    return RegressionResults(
        model_name="bayesian_linear_regression",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_predict=y_pred.flatten(),
        intermediates=intermediate_results
    )

def solve_normal_equation(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs: Any
) -> RegressionResults:
    """Solve linear regression using the Normal Equation method.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels (target values)
        y_test: Test labels
        **kwargs: Additional keyword arguments (unused)

    Returns:
        RegressionResults containing:
            - model_name: Name identifier for normal equation method
            - x_train: Training features matrix provided as input
            - x_test: Test features matrix provided as input
            - y_train: Training target values provided as input
            - y_test: Test target values provided as input
            - y_predict: Model's predicted values for x_test
            - intermediates: Dict of intermediate calculations including:
                - Transpose of X (Xᵀ): Transposed feature matrix
                - Design Matrix (XᵀX): Core matrix for normal equation
                - Weighted Feature Matrix (Xᵀy): Target-weighted features
                - Weights (w): Final computed regression coefficients

    """
    # Convert input data to numpy arrays for matrix computations
    x = cast(npt.NDArray[np.float64], x_train.to_numpy())
    y = cast(npt.NDArray[np.float64], y_train.to_numpy()).reshape(-1, 1)
    x_test_np = cast(npt.NDArray[np.float64], x_test.to_numpy())

    # Add bias terms
    x = np.hstack((np.ones((x.shape[0], 1), dtype=np.float64), x))
    x_test_np = np.hstack((np.ones((x_test_np.shape[0], 1), dtype=np.float64), x_test_np))

    # Compute the intermediate results
    x_transpose = x.T  # Transpose of x
    design_matrix = x_transpose @ x  # XᵀX
    weighted_feature_matrix = x_transpose @ y  # Xᵀy

    # Compute the weights using the Normal Equation
    weights = np.linalg.inv(design_matrix) @ weighted_feature_matrix

    # Calculate prediction for test data
    y_pred = x_test_np @ weights

    # Store intermediate results
    intermediate_results: Dict[str, npt.NDArray[np.float64]] = {
        "Transpose of X (Xᵀ)": x_transpose,
        "Design Matrix (XᵀX)": design_matrix,
        "Weighted Feature Matrix (Xᵀy)": weighted_feature_matrix.flatten(),
        "Weights (w)": weights.flatten()
    }

    return RegressionResults(
        model_name="normal_equation",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_predict=y_pred.flatten(),
        intermediates=intermediate_results
    )

def solve_ridge_regression_closed_form(  # Renamed to be more specific
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs: Any
) -> RegressionResults:
    """Solve Ridge Regression using direct closed-form solution.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels (target values)
        y_test: Test labels
        **kwargs: Additional keyword arguments like:
            lambda_reg: Regularization parameter (λ). Default is 1.0

    Returns:
        RegressionResults containing:
            - model_name: Name identifier for closed-form ridge regression
            - x_train: Training features matrix provided as input
            - x_test: Test features matrix provided as input
            - y_train: Training target values provided as input
            - y_test: Test target values provided as input
            - y_predict: Model's predicted values for x_test
            - intermediates: Dict of intermediate calculations including:
                - Regularization Term (λI): L2 penalty matrix
                - Modified Design Matrix (XᵀX + λI): Regularized design matrix
                - Weighted Feature Matrix (Xᵀy): Target-weighted features
                - Weights (w): Final regularized coefficients

    """
    # Get regularization parameter from kwargs
    lambda_reg = kwargs.get("lambda_reg", 1.0)

    # Convert input data to numpy arrays for matrix computations
    x = cast(npt.NDArray[np.float64], x_train.to_numpy())
    y = cast(npt.NDArray[np.float64], y_train.to_numpy()).reshape(-1, 1)
    x_test_np = cast(npt.NDArray[np.float64], x_test.to_numpy())

    # Add bias terms
    x = np.hstack((np.ones((x.shape[0], 1), dtype=np.float64), x))
    x_test_np = np.hstack((np.ones((x_test_np.shape[0], 1), dtype=np.float64), x_test_np))

    # Compute the design matrix
    x_transpose = x.T
    design_matrix = x_transpose @ x

    # Compute the regularization term
    identity = np.eye(design_matrix.shape[0], dtype=np.float64)
    identity[0, 0] = 0  # Do not regularize the bias term
    regularization_term = lambda_reg * identity

    # Compute the modified design matrix
    modified_design_matrix = design_matrix + regularization_term

    # Compute the weighted feature matrix
    weighted_feature_matrix = x_transpose @ y

    # Compute the weights using the Ridge Regression closed-form solution
    weights = np.linalg.inv(modified_design_matrix) @ weighted_feature_matrix

    # Calculate prediction for test data
    y_pred = x_test_np @ weights

    # Store intermediate results
    intermediate_results: Dict[str, npt.NDArray[np.float64]] = {
        "Regularization Term (λI)": regularization_term,
        "Modified Design Matrix (XᵀX + λI)": modified_design_matrix,
        "Weighted Feature Matrix (Xᵀy)": weighted_feature_matrix.flatten(),
        "Weights (w)": weights.flatten()
    }

    return RegressionResults(
        model_name="ridge_regression_closed_form",  # Updated name
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_predict=y_pred.flatten(),
        intermediates=intermediate_results
    )

def solve_irls(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs: Any
) -> RegressionResults:
    """Solve regression using Iterative Reweighted Least Squares (IRLS).

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels (target values)
        y_test: Test labels
        **kwargs: Additional keyword arguments like:
            max_iter: Maximum iterations for convergence (default is 100)
            tol: Tolerance for convergence (default is 1e-6)

    Returns:
        RegressionResults containing:
            - model_name: Name identifier for IRLS method
            - x_train: Training features matrix provided as input
            - x_test: Test features matrix provided as input
            - y_train: Training target values provided as input
            - y_test: Test target values provided as input
            - y_predict: Model's predicted values for x_test
            - intermediates: Dict of intermediate calculations including:
                - Hessian Matrices (H): Second derivatives at each iteration
                - Gradients of Loss (∇L): Loss gradients at each iteration
                - Weight Updates (Δw): Parameter updates at each iteration
                - Final Weights (w): Converged model parameters

    """
    # Get keyword args with defaults
    max_iter = kwargs.get("max_iter", 100)
    tol = kwargs.get("tol", 1e-6)

    # Convert input data to numpy arrays
    x = cast(npt.NDArray[np.float64], x_train.to_numpy())
    y = cast(npt.NDArray[np.float64], y_train.to_numpy()).reshape(-1, 1)
    x_test_np = cast(npt.NDArray[np.float64], x_test.to_numpy())

    # Add bias terms
    x = np.hstack((np.ones((x.shape[0], 1), dtype=np.float64), x))
    x_test_np = np.hstack((np.ones((x_test_np.shape[0], 1), dtype=np.float64), x_test_np))
    n_samples, n_features = x.shape

    # Initialize weights to zeros
    w = np.zeros((n_features, 1), dtype=np.float64)

    # Store intermediate results
    hessian_matrices: List[npt.NDArray[np.float64]] = []
    gradients: List[npt.NDArray[np.float64]] = []
    weight_updates: List[npt.NDArray[np.float64]] = []

    for iteration in range(max_iter):
        # Compute predictions and residuals
        y_pred = x @ w
        residuals = y - y_pred

        # Compute weights for diagonal weight matrix
        weights_diag = np.maximum(1e-6, np.abs(residuals))  # Avoid division by zero

        # Form diagonal weight matrix W
        W = np.diagflat(1 / weights_diag)

        # Compute Hessian matrix (H = XᵀWX)
        H = x.T @ W @ x
        hessian_matrices.append(H)

        # Compute gradient of loss function (∇L = XᵀW(y - Xw))
        gradient = x.T @ W @ residuals
        gradients.append(gradient.flatten())

        # Compute weight update (Δw = H⁻¹ ∇L)
        delta_w = np.linalg.inv(H) @ gradient
        weight_updates.append(delta_w.flatten())

        # Update weights
        w += delta_w

        # Check convergence
        if np.linalg.norm(delta_w) < tol:
            print(f"Converged in {iteration + 1} iterations.")
            break

    # Calculate prediction for test data
    y_pred = x_test_np @ w

    # Store intermediate results
    intermediate_results: Dict[str, Union[List[npt.NDArray[np.float64]],
                                        npt.NDArray[np.float64]]] = {
        "Hessian Matrices (H)": hessian_matrices,
        "Gradients of Loss (∇L)": gradients,
        "Weight Updates (Δw)": weight_updates,
        "Final Weights (w)": w.flatten()
    }

    return RegressionResults(
        model_name="irls",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_predict=y_pred.flatten(),
        intermediates=intermediate_results
    )

def solve_pcr(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs: Any
) -> RegressionResults:
    """Solve regression using Principal Component Regression (PCR).

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels (target values)
        y_test: Test labels
        **kwargs: Additional keyword arguments like:
            n_components: Number of principal components to retain (default is None)
                If None, all components are retained.

    Returns:
        RegressionResults containing:
            - model_name: Name identifier for PCR method
            - x_train: Training features matrix provided as input
            - x_test: Test features matrix provided as input
            - y_train: Training target values provided as input
            - y_test: Test target values provided as input
            - y_predict: Model's predicted values for x_test
            - intermediates: Dict of intermediate calculations including:
                - Principal Components: Eigenvectors of covariance matrix
                - Explained Variance: Variance explained by each component
                - Transformed Data (PCs): Data projected onto components
                - Weights (w): Final regression coefficients

    """
    # Get number of components from kwargs
    n_components = kwargs.get("n_components", None)

    # Convert input data to numpy arrays
    x = cast(npt.NDArray[np.float64], x_train.to_numpy())
    y = cast(npt.NDArray[np.float64], y_train.to_numpy()).reshape(-1, 1)
    x_test_np = cast(npt.NDArray[np.float64], x_test.to_numpy())

    # Standardize the training features
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_standardized = (x - x_mean) / x_std

    # Standardize test features using training statistics
    x_test_standardized = (x_test_np - x_mean) / x_std

    # Compute covariance matrix of standardized features
    covariance_matrix = np.cov(x_standardized.T)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Retain the top n_components (if specified)
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]

    # Project the standardized data onto principal components
    transformed_train = x_standardized @ eigenvectors
    transformed_test = x_test_standardized @ eigenvectors

    # Add bias terms to transformed data
    transformed_train_with_bias = np.hstack((
        np.ones((transformed_train.shape[0], 1), dtype=np.float64),
        transformed_train
    ))
    transformed_test_with_bias = np.hstack((
        np.ones((transformed_test.shape[0], 1), dtype=np.float64),
        transformed_test
    ))

    # Compute weights using transformed training data
    w = np.linalg.inv(transformed_train_with_bias.T @ transformed_train_with_bias) @ \
        (transformed_train_with_bias.T @ y)

    # Calculate prediction using transformed test data
    y_pred = transformed_test_with_bias @ w

    # Store intermediate results
    intermediate_results: Dict[str, npt.NDArray[np.float64]] = {
        "Principal Components": eigenvectors,
        "Explained Variance": eigenvalues[:n_components] if n_components else eigenvalues,
        "Transformed Data (PCs)": transformed_train,
        "Weights (w)": w.flatten()
    }

    # Include number of components in model name if specified
    model_name = (f"pcr_{n_components}_components"
                 if n_components is not None
                 else "pcr_all_components")

    return RegressionResults(
        model_name=model_name,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_predict=y_pred.flatten(),
        intermediates=intermediate_results
    )

def solve_knn(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs: Any
) -> RegressionResults:
    """Solve regression using K-Nearest Neighbors (KNN).

    Args:
        x_train: Training features
        x_test: Test features to predict on
        y_train: Training labels (target values)
        y_test: Test labels
        **kwargs: Additional keyword args with defaults:
            k: Number of nearest neighbors (default is 5)

    Returns:
        RegressionResults containing:
            - model_name: Name identifier for KNN method with k value
            - x_train: Training features matrix provided as input
            - x_test: Test features matrix provided as input
            - y_train: Training target values provided as input
            - y_test: Test target values provided as input
            - y_predict: Model's predicted values for x_test
            - intermediates: Dict of intermediate calculations including:
                - Distances (D): Distance matrix between test and training points
                - Neighbor Indices (I): Indices of k-nearest neighbors
                - Neighbor Labels (L): Labels of k-nearest neighbors

    """
    # Get k from kwargs with default
    k = kwargs.get("k", 5)

    # Convert input data to numpy arrays
    X_train = cast(npt.NDArray[np.float64], x_train.to_numpy())
    y_train_np = cast(npt.NDArray[np.float64], y_train.to_numpy())
    X_test = cast(npt.NDArray[np.float64], x_test.to_numpy())

    num_test = X_test.shape[0]
    num_train = X_train.shape[0]

    # 1. Distance Calculation
    distances = np.zeros((num_test, num_train), dtype=np.float64)
    for i in range(num_test):
        for j in range(num_train):
            distances[i, j] = np.linalg.norm(X_test[i] - X_train[j])

    # 2. Neighbor Identification
    neighbor_indices = np.argsort(distances, axis=1)[:, :k]
    neighbor_labels = y_train_np[neighbor_indices]

    # 3. Regression (using mean instead of mode for regression)
    y_pred = np.zeros(num_test, dtype=np.float64)
    for i in range(num_test):
        y_pred[i] = np.mean(neighbor_labels[i])

    # Store intermediate results
    intermediate_results: Dict[str, npt.NDArray[np.float64]] = {
        "Distances (D)": distances,
        "Neighbor Indices (I)": neighbor_indices.astype(np.float64),  # Convert to float64
        "Neighbor Labels (L)": neighbor_labels
    }

    return RegressionResults(
        model_name=f"knn_{k}_neighbors",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_predict=y_pred,
        intermediates=intermediate_results
    )

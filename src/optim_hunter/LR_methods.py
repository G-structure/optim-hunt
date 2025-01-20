"""Implements various linear regression methods with detailed computation tracking."""
from typing import Dict, List, Optional, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

def solve_ols(
    y_train: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame
) -> Dict[str, Union[npt.NDArray[np.float64], Dict[str, npt.NDArray[np.float64]]]]:
    """Perform Ordinary Least Squares (OLS) regression and calculate intermediate
    results.

    Args:
        y_train: Training labels (target values)
        x_train: Training features
        x_test: Test features to predict on

    Returns:
        Dict containing:
            - intermediates: Dict of intermediate calculations
                - Design Matrix (XᵀX)
                - Pseudoinverse ((XᵀX)^(-1))
                - Weighted Feature Matrix (Xᵀy)
                - Weights (w)
            - prediction: Predicted values for x_test

    """
    # Convert to numpy arrays for computation
    x = cast(npt.NDArray[np.float64], x_train.to_numpy())
    y = cast(npt.NDArray[np.float64], y_train.to_numpy()).reshape(-1, 1)
    x_test_np = cast(npt.NDArray[np.float64], x_test.to_numpy())

    # Add bias terms
    x = np.hstack((
        np.ones((x.shape[0], 1), dtype=np.float64),
        x
    ))
    x_test_np = np.hstack((
        np.ones((x_test_np.shape[0], 1), dtype=np.float64),
        x_test_np
    ))

    # Calculate intermediate results
    design_matrix = x.T @ x
    pseudoinverse = np.linalg.inv(design_matrix)
    weighted_feature_matrix = x.T @ y
    weights = pseudoinverse @ weighted_feature_matrix

    # Calculate prediction
    y_pred = x_test_np @ weights

    # Return intermediate results and prediction
    return {
        "intermediates": {
            "Design Matrix (XᵀX)": design_matrix,
            "Pseudoinverse ((XᵀX)^(-1))": pseudoinverse,
            "Weighted Feature Matrix (Xᵀy)": weighted_feature_matrix,
            "Weights (w)": weights
        },
        "prediction": y_pred.flatten()
    }

'''Example usage:
# Load dataset
x_train, y_train, x_test, y_test = get_dataset_friedman_1()

# Perform OLS
ols_results = solve_ols(y_train, x_train)

# Display results
for key, value in ols_results.items():
    print(f"{key}:\n{value}\n")
'''


def solve_gradient_descent(
    y_train: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,  # Added x_test parameter
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6
) -> Dict[str, Union[np.ndarray, List[np.ndarray], float, int, Dict]]:
    """Perform Gradient Descent for linear regression and calculate intermediate results.

    Args:
        y_train: Training labels (target values)
        x_train: Training features
        x_test: Test features to predict on
        learning_rate: The step size used in the weight update (default is 0.01)
        max_iterations: Maximum number of iterations (default is 1000)
        tolerance: The convergence threshold for the gradient norm (default is 1e-6)

    Returns:
        Dict containing:
            - intermediates: Dict of intermediate results including:
                - Initial Weights (w₀)
                - Gradients (∇L)
                - Learning Rate (α)
                - Weight Updates (Δw)
                - Intermediate Weights
                - Final Weights (w)
                - Number of Iterations
            - prediction: Predicted values for x_test
    """
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
    gradients: List[np.ndarray] = []
    weight_updates: List[np.ndarray] = []
    intermediate_weights: List[np.ndarray] = [initial_weights.flatten()]

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

    return {
        "intermediates": {
            "Initial Weights (w₀)": initial_weights.flatten(),
            "Gradients (∇L)": gradients,
            "Learning Rate (α)": learning_rate,
            "Weight Updates (Δw)": weight_updates,
            "Intermediate Weights (w₁, w₂, ..., wₙ)": intermediate_weights,
            "Final Weights (w)": weights.flatten(),
            "Number of Iterations": iteration + 1
        },
        "prediction": y_pred.flatten()
    }

'''Example usage:
x_train, y_train, x_test, y_test = get_dataset_friedman_1()

# Perform Gradient Descent
gd_results = solve_gradient_descent(y_train, x_train, learning_rate=0.01, max_iterations=1000, tolerance=1e-6)

# Display results
print("Initial Weights (w₀):", gd_results["Initial Weights (w₀)"])
print("Final Weights (w):", gd_results["Final Weights (w)"])
print("Number of Iterations:", gd_results["Number of Iterations"])
print("Intermediate Weights (first 5):", gd_results["Intermediate Weights (w₁, w₂, ..., wₙ)"][:5])
'''

def solve_ridge_regression(
    y_train: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    regularization_param: float = 1.0
) -> Dict[str, Union[Dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]]]:
    """Perform Ridge Regression and calculate intermediate results.

    Args:
        y_train: Training labels (target values)
        x_train: Training features
        x_test: Test features to predict on
        regularization_param: The regularization parameter λ (default is 1.0)

    Returns:
        Dict containing:
            - intermediates: Dict of intermediate calculations including:
                - Regularization Term (λI)
                - Modified Design Matrix (XᵀX + λI)
                - Pseudoinverse ((XᵀX + λI)^(-1))
                - Weights (w)
            - prediction: Predicted values for x_test
    """
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

    return {
        "intermediates": {
            "Regularization Term (λI)": regularization_term,
            "Modified Design Matrix (XᵀX + λI)": modified_design_matrix,
            "Pseudoinverse ((XᵀX + λI)^(-1))": pseudoinverse,
            "Weights (w)": weights.flatten()
        },
        "prediction": y_pred.flatten()
    }

'''Example usage:
# Load dataset
x_train, y_train, x_test, y_test = get_dataset_friedman_1()

# Perform Ridge Regression
ridge_results = solve_ridge_regression(y_train, x_train, regularization_param=1.0)

# Display results
print("Regularization Term (λI):\n", ridge_results["Regularization Term (λI)"])
print("Modified Design Matrix (XᵀX + λI):\n", ridge_results["Modified Design Matrix (XᵀX + λI)"])
print("Pseudoinverse ((XᵀX + λI)^(-1)):\n", ridge_results["Pseudoinverse ((XᵀX + λI)^(-1))"])
print("Weights (w):\n", ridge_results["Weights (w)"])
'''

def solve_lasso_regression(
    y_train: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    regularization_param: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, Union[Dict[str, Union[List[float], List[np.ndarray], np.ndarray]],
                    npt.NDArray[np.float64]]]:
    """Apply lasso regression using coordinate descent and calculate results.

    Args:
        y_train: Training labels (target values)
        x_train: Training features
        x_test: Test features to predict on
        regularization_param: The regularization parameter λ (default is 1.0)
        max_iter: Maximum number of iterations (default is 1000)
        tol: Convergence tolerance (default is 1e-4)

    Returns:
        Dict containing:
            - intermediates: Dict of intermediate calculations including:
                - Regularization Term (λ||w||₁)
                - Soft Thresholding Steps
                - Weights (w)
            - prediction: Predicted values for x_test
    """
    # Convert to numpy arrays for computation
    x = cast(npt.NDArray[np.float64], x_train.to_numpy())
    y = cast(npt.NDArray[np.float64], y_train.to_numpy()).reshape(-1, 1)
    x_test_np = cast(npt.NDArray[np.float64], x_test.to_numpy())

    # Add bias terms
    x = np.hstack((np.ones((x.shape[0], 1), dtype=np.float64), x))
    x_test_np = np.hstack((np.ones((x_test_np.shape[0], 1), dtype=np.float64), x_test_np))

    # Initialize weights (including bias term)
    weights = np.zeros((x.shape[1], 1), dtype=np.float64)

    # Number of samples and features
    n_samples, n_features = x.shape

    # Store intermediate results
    intermediate_results = {
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

    return {
        "intermediates": intermediate_results,
        "prediction": y_pred.flatten()
    }

'''Example usage:
# Load dataset
x_train, y_train, x_test, y_test = get_dataset_friedman_1()

# Perform Lasso Regression
lasso_results = solve_lasso_regression(y_train, x_train, regularization_param=1.0)

# Display results
print("Regularization Term (λ||w||₁):", lasso_results["Regularization Term (λ||w||₁)"][-1])
print("Soft Thresholding Steps (last iteration):", lasso_results["Soft Thresholding Steps"][-1])
print("Final Weights (w):", lasso_results["Weights (w)"])
'''

def solve_sgd(
    y_train: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    learning_rate: float = 0.01,
    max_iter: int = 100,
    batch_size: int = 1,
    tol: float = 1e-4,
    random_state: int = 42
) -> Dict[str, Union[Dict[str, List[npt.NDArray[np.float64]]], npt.NDArray[np.float64]]]:
    """Perform Stochastic Gradient Descent (SGD) and calculate intermediate results.

    Args:
        y_train: Training labels (target values)
        x_train: Training features
        x_test: Test features to predict on
        learning_rate: Learning rate (α) for weight updates (default is 0.01)
        max_iter: Maximum number of iterations (default is 100)
        batch_size: Size of the mini-batch for each step (default is 1)
        tol: Convergence tolerance (default is 1e-4)
        random_state: Random seed for reproducibility (default is 42)

    Returns:
        Dict containing:
            - intermediates: Dict of intermediate calculations including:
                - Random Sampling of Data Points
                - Gradient of Loss for Current Sample (∇L_i)
                - Intermediate Weights (w₁, w₂, ..., wₙ)
            - prediction: Predicted values for x_test
    """
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
    intermediate_results = {
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
                batch_indices.copy()
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

    return {
        "intermediates": intermediate_results,
        "prediction": y_pred.flatten()
    }

'''Example usage:
# Load dataset
x_train, y_train, x_test, y_test = get_dataset_friedman_1()

# Perform SGD
sgd_results = solve_sgd(y_train, x_train, learning_rate=0.01, max_iter=100, batch_size=1)

# Display results
print("Random Sampling of Data Points (last iteration):", sgd_results["Random Sampling of Data Points"][-1])
print("Gradient of Loss for Current Sample (last iteration):", sgd_results["Gradient of Loss for Current Sample (∇L_i)"][-1])
print("Final Weights (w):", sgd_results["Intermediate Weights (w₁, w₂, ..., wₙ)"][-1])
'''

def solve_bayesian_linear_regression(
    y_train: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Dict[str, Union[Dict[str, Dict[str, Union[npt.NDArray[np.float64], float]]],
                    npt.NDArray[np.float64]]]:
    """Perform Bayesian Linear Regression and calculate intermediate results.

    Args:
        y_train: Training labels (target values)
        x_train: Training features
        x_test: Test features to predict on
        alpha: Precision of the prior distribution (default is 1.0)
        beta: Precision of the likelihood (default is 1.0)

    Returns:
        Dict containing:
            - intermediates: Dict of intermediate calculations including:
                - Prior Distribution (P(w)): Mean and covariance of the prior
                - Likelihood (P(y|X, w)): Precision of the noise
                - Posterior Mean and Covariance: Updated distribution over weights
            - prediction: Predicted values for x_test
    """
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

    return {
        "intermediates": intermediate_results,
        "prediction": y_pred.flatten()
    }

'''Example usage:
# Load dataset
x_train, y_train, x_test, y_test = get_dataset_friedman_1()

# Perform Bayesian Linear Regression
bayesian_results = solve_bayesian_linear_regression(y_train, x_train, alpha=1.0, beta=1.0)

# Display results
print("Prior Distribution (P(w)):")
print("Mean:", bayesian_results["Prior Distribution (P(w))"]["Mean"])
print("Covariance:", bayesian_results["Prior Distribution (P(w))"]["Covariance"])

print("\nLikelihood (P(y|X, w)):")
print("Precision (β):", bayesian_results["Likelihood (P(y|X, w))"]["Precision (β)"])

print("\nPosterior Mean and Covariance:")
print("Mean:", bayesian_results["Posterior Mean and Covariance"]["Mean"])
print("Covariance:", bayesian_results["Posterior Mean and Covariance"]["Covariance"])

'''

def solve_normal_equation(
    y_train: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame
) -> Dict[str, Union[Dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]]]:
    """Solve linear regression using the Normal Equation and compute intermediate results.

    Args:
        y_train: Training labels (target values)
        x_train: Training features
        x_test: Test features to predict on

    Returns:
        Dict containing:
            - intermediates: Dict of intermediate calculations including:
                - Transpose of X (Xᵀ)
                - Design Matrix (XᵀX)
                - Weighted Feature Matrix (Xᵀy)
                - Weights (w)
            - prediction: Predicted values for x_test
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

    return {
        "intermediates": intermediate_results,
        "prediction": y_pred.flatten()
    }

'''Example usage:
# Load dataset
x_train, y_train, x_test, y_test = get_dataset_friedman_1()

# Solve using the Normal Equation
normal_equation_results = solve_normal_equation(y_train, x_train)

# Display intermediate results
print("Transpose of X (Xᵀ):")
print(normal_equation_results["Transpose of X (Xᵀ)"])

print("\nDesign Matrix (XᵀX):")
print(normal_equation_results["Design Matrix (XᵀX)"])

print("\nWeighted Feature Matrix (Xᵀy):")
print(normal_equation_results["Weighted Feature Matrix (Xᵀy)"])

print("\nWeights (w):")
print(normal_equation_results["Weights (w)"])

'''

def solve_ridge_regression(
    y_train: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    lambda_reg: float = 1.0
) -> Dict[str, Union[Dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]]]:
    """Solve Ridge Regression using closed-form solution and compute intermediate results.

    Args:
        y_train: Training labels (target values)
        x_train: Training features
        x_test: Test features to predict on
        lambda_reg: Regularization parameter (λ). Default is 1.0

    Returns:
        Dict containing:
            - intermediates: Dict of intermediate calculations including:
                - Regularization Term (λI)
                - Modified Design Matrix (XᵀX + λI)
                - Weighted Feature Matrix (Xᵀy)
                - Weights (w)
            - prediction: Predicted values for x_test
    """
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

    return {
        "intermediates": intermediate_results,
        "prediction": y_pred.flatten()
    }

'''Example usage:
# Load dataset
x_train, y_train, x_test, y_test = get_dataset_friedman_1()

# Solve using Ridge Regression
ridge_results = solve_ridge_regression(y_train, x_train, lambda_reg=0.5)

# Display intermediate results
print("Regularization Term (λI):")
print(ridge_results["Regularization Term (λI)"])

print("\nModified Design Matrix (XᵀX + λI):")
print(ridge_results["Modified Design Matrix (XᵀX + λI)"])

print("\nWeighted Feature Matrix (Xᵀy):")
print(ridge_results["Weighted Feature Matrix (Xᵀy)"])

print("\nWeights (w):")
print(ridge_results["Weights (w)"])

'''

def solve_irls(
    y_train: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Dict[str, Union[Dict[str, Union[List[npt.NDArray[np.float64]],
                                   npt.NDArray[np.float64]]],
                    npt.NDArray[np.float64]]]:
    """Solve regression using Iterative Reweighted Least Squares (IRLS).

    Args:
        y_train: Training labels (target values)
        x_train: Training features
        x_test: Test features to predict on
        max_iter: Maximum number of iterations for convergence (default is 100)
        tol: Tolerance for convergence (default is 1e-6)

    Returns:
        Dict containing:
            - intermediates: Dict of intermediate calculations including:
                - Hessian Matrix (H): Second derivative of loss function
                - Gradient of Loss (∇L): Gradient of loss function
                - Weight Updates (Δw): Changes in weights
                - Final Weights (w): Final regression coefficients
            - prediction: Predicted values for x_test
    """
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

    # Store results
    intermediate_results: Dict[str, Union[List[npt.NDArray[np.float64]],
                                        npt.NDArray[np.float64]]] = {
        "Hessian Matrices (H)": hessian_matrices,
        "Gradients of Loss (∇L)": gradients,
        "Weight Updates (Δw)": weight_updates,
        "Final Weights (w)": w.flatten()
    }

    return {
        "intermediates": intermediate_results,
        "prediction": y_pred.flatten()
    }

'''Example usage:
# Load dataset
x_train, y_train, x_test, y_test = get_dataset_friedman_1()

# Solve using IRLS
irls_results = solve_irls(y_train, x_train)

# Display intermediate results
print("Hessian Matrix (H) at last iteration:")
print(irls_results["Hessian Matrices (H)"][-1])

print("\nGradient of Loss (∇L) at last iteration:")
print(irls_results["Gradients of Loss (∇L)"][-1])

print("\nWeight Updates (Δw) at last iteration:")
print(irls_results["Weight Updates (Δw)"][-1])

print("\nFinal Weights (w):")
print(irls_results["Final Weights (w)"])

'''

def solve_pcr(
    y_train: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    n_components: Optional[int] = None
) -> Dict[str, Union[Dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]]]:
    """Solve regression using Principal Component Regression (PCR).

    Args:
        y_train: Training labels (target values)
        x_train: Training features
        x_test: Test features to predict on
        n_components: Number of principal components to retain. If None, all
            components are retained. Default is None

    Returns:
        Dict containing:
            - intermediates: Dict of intermediate calculations including:
                - Principal Components: Eigenvectors of covariance matrix
                - Explained Variance: Variance explained by components
                - Transformed Data (PCs): Data projected onto components
                - Weights (w): Final regression coefficients
            - prediction: Predicted values for x_test
    """
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

    return {
        "intermediates": intermediate_results,
        "prediction": y_pred.flatten()
    }

'''Example usage:
x_train, y_train, x_test, y_test = get_dataset_friedman_1()

# Solve using PCR (retain top 5 principal components)
pcr_results = solve_pcr(y_train, x_train, n_components=5)

# Display intermediate results
print("Principal Components:")
print(pcr_results["Principal Components"])

print("\nExplained Variance:")
print(pcr_results["Explained Variance"])

print("\nTransformed Data (first 5 rows):")
print(pcr_results["Transformed Data (PCs)"][:5])

print("\nWeights (w):")
print(pcr_results["Weights (w)"])

'''



def solve_knn(
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    X_test: npt.NDArray[np.float64],
    k: int
) -> Dict[str, Union[Dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]]]:
    """Solve regression using K-Nearest Neighbors (KNN).

    Args:
        X_train: Training features array
        y_train: Training labels array
        X_test: Test features array
        k: Number of nearest neighbors

    Returns:
        Dict containing:
            - intermediates: Dict of intermediate calculations including:
                - Distances (D): Distance matrix between test and training points
                - Neighbor Indices (I): Indices of k-nearest neighbors
                - Neighbor Labels (L): Labels of k-nearest neighbors
            - prediction: Predicted values for X_test

    """
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]

    # 1. Distance Calculation
    distances = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            distances[i, j] = np.linalg.norm(X_test[i] - X_train[j])

    # 2. Neighbor Identification
    neighbor_indices = np.argsort(distances, axis=1)[:, :k]
    neighbor_labels = y_train[neighbor_indices]

    # 3. Classification
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        unique, counts = np.unique(neighbor_labels[i], return_counts=True)
        y_pred[i] = unique[np.argmax(counts)]

    return {
        "intermediates": {
            "Distances (D)": distances,
            "Neighbor Indices (I)": neighbor_indices,
            "Neighbor Labels (L)": neighbor_labels
        },
        "prediction": y_pred
    }
''' Example usage:
x_train, y_train, x_test, y_test = get_dataset_friedman_1(random_state=1)

# Set the number of neighbors
k = 5

# Run KNN and get intermediate results
knn_results = solve_knn(X_train, y_train, X_test, k)

# Access intermediate results
distances = knn_results["Distances (D)"]
neighbor_indices = knn_results["Neighbor Indices (I)"]
neighbor_labels = knn_results["Neighbor Labels (L)"]
predicted_labels = knn_results["Predicted Labels (y_pred)"]


'''

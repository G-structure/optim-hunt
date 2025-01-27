from functools import partial
from typing import Any, Callable, Dict, List
import torch
from optim_hunter.probe.probe_prediction import PredictionGenerator, ProbePredictionTarget
from optim_hunter.LR_methods import (
    solve_bayesian_linear_regression,
    solve_gradient_descent,
    solve_irls,
    solve_knn,
    solve_lasso_regression,
    solve_normal_equation,
    solve_ols,
    solve_pcr,
    solve_ridge_regression,
    solve_ridge_regression_closed_form,
    solve_sgd
)
def make_intermediate_predictor(
    key_path: List[str],
    output_dim: int,
    regression_method: Callable
) -> type[PredictionGenerator]:
    """Factory function to create predictor classes for regression intermediates.

    Args:
        key_path: List of keys to access the target intermediate value
        output_dim: Dimension of the output prediction
        regression_method: The regression method to use

    Returns:
        A new PredictionGenerator class configured for the specified intermediate
    """

    class IntermediatePredictor(PredictionGenerator):
        def __init__(self):
            self._regression_fn = regression_method
            self._key_path = key_path
            self._output_size = output_dim

        def compute(
            self, x_train, y_train, x_test, y_test, **kwargs
        ) -> ProbePredictionTarget:
            # Run regression method
            results = self._regression_fn(x_train, x_test, y_train, y_test, **kwargs)

            # Navigate to target value through key path
            target = results.intermediates
            for key in self._key_path:
                if key not in target:
                    raise KeyError(f"Key {key} not found in regression results")
                target = target[key]

            # Convert to tensor if needed
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, dtype=torch.float32)

            return ProbePredictionTarget(
                values=target,
                metadata={
                    "method": self._regression_fn.__name__,
                    "target": ".".join(self._key_path),
                    **results.metadata
                }
            )

        @property
        def output_dim(self) -> int:
            return self._output_size

    return IntermediatePredictor

# Create a dictionary mapping intermediate names to their predictors
def get_all_predictors() -> Dict[str, type[PredictionGenerator]]:
    """Get all available intermediate predictors."""
    predictors = {
        # IRLS intermediates
        "irls_gradients": make_intermediate_predictor(
            ["Gradients of Loss (∇L)"], 5, solve_irls
        ),
        "irls_weights": make_intermediate_predictor(
            ["Final Weights (w)"], 5, solve_irls
        ),
        "irls_hessian": make_intermediate_predictor(
            ["Hessian Matrices (H)"], 25, solve_irls
        ),

        # SGD intermediates
        "sgd_gradients": make_intermediate_predictor(
            ["Gradients of Loss (∇L)"], 5, solve_sgd
        ),
        "sgd_weights": make_intermediate_predictor(
            ["Final Weights (w)"], 5, solve_sgd
        ),

        # KNN intermediates
        "knn_distances": make_intermediate_predictor(
            ["Distances (D)"], 10, solve_knn
        ),
        "knn_neighbors": make_intermediate_predictor(
            ["Neighbor Indices (I)"], 5, solve_knn
        ),

        # Add more as needed...
    }
    return predictors

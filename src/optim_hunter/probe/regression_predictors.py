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
    regression_method: Callable
) -> type[PredictionGenerator]:
    """Factory function to create predictor classes for regression intermediates.

    Args:
        key_path: List of keys to access the target intermediate value
        regression_method: The regression method to use

    Returns:
        A new PredictionGenerator class configured for the specified intermediate
    """

    class IntermediatePredictor(PredictionGenerator):
        def __init__(self):
            self._regression_fn = regression_method
            self._key_path = key_path
            # Output dimension will be determined on first compute call
            self._output_dim = None

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

            # Get final value if target is a list
            if isinstance(target, list):
                target = target[-1]  # Take final value

            # Convert to tensor and flatten
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, dtype=torch.float32)
            target = target.flatten()

            target = target.view(-1)

            # Set output dimension if not already set
            if self._output_dim is None:
                self._output_dim = target.numel()

            return ProbePredictionTarget(
                values=target,  # Flatten to 1D
                metadata={
                    "method": self._regression_fn.__name__,
                    "target": ".".join(self._key_path),
                    **results.metadata
                }
            )

        @property
        def output_dim(self) -> int:
            if self._output_dim is None:
                raise RuntimeError(
                    "Output dimension not yet determined. "
                    "Must call compute() first."
                )
            return self._output_dim

    return IntermediatePredictor

def get_all_predictors() -> Dict[str, type[PredictionGenerator]]:
    """Get all available intermediate predictors."""
    predictors = {
        # IRLS intermediates
        "irls_gradients": make_intermediate_predictor(
            ["Gradients of Loss (∇L)"], solve_irls
        ),
        "irls_weights": make_intermediate_predictor(
            ["Final Weights (w)"], solve_irls
        ),
        "irls_hessian": make_intermediate_predictor(
            ["Hessian Matrices (H)"], solve_irls
        ),

        # SGD intermediates
        "sgd_gradients": make_intermediate_predictor(
            ["Gradients of Loss (∇L)"], solve_sgd
        ),
        "sgd_weights": make_intermediate_predictor(
            ["Final Weights (w)"], solve_sgd
        ),

        # KNN intermediates
        "knn_distances": make_intermediate_predictor(
            ["Distances (D)"], solve_knn
        ),
        "knn_neighbors": make_intermediate_predictor(
            ["Neighbor Indices (I)"], solve_knn
        ),
    }
    return predictors

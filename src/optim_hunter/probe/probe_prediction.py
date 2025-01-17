from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import torch
from optim_hunter.regressors import calculate_sgd_gradients


@dataclass
class ProbePredictionTarget:
    """Container for values the probe should learn to predict.

    A structured collection of prediction targets that a probe should learn to
    output, along with optional metadata about the prediction task.

    Attributes:
        values (torch.Tensor): Target values for probe to predict. Shape is
            [batch, target_dim]
        metadata (Optional[Dict], optional): Additional metadata about the
            prediction task. Defaults to None.
    """
    values: torch.Tensor  # Shape [batch, target_dim]
    metadata: Optional[Dict] = None

class PredictionGenerator(ABC):
    """Abstract base class for generating target values for probe prediction.

    This class provides a standardized interface for generating prediction targets
    that a probe model should learn to output. Classes inheriting from this base
    must implement:

    - compute(): Generate target values from training/test data
    - output_dim: Report dimensionality of target values

    The intent is to allow different types of prediction tasks (gradients, parameter
    updates, loss values etc) while maintaining a consistent interface for probe
    training.

    Attributes:
        None
    """


    @abstractmethod
    def compute(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        **kwargs
    ) -> ProbePredictionTarget:
        """Generate target values for the probe to predict.

        This abstract method should be implemented to compute target values that
        the probe model will learn to predict. It takes training and test data
        as input and returns a ProbePredictionTarget containing the computed
        target values and optional metadata.

        Args:
            x_train (pd.DataFrame): Training feature data
            y_train (pd.Series): Training labels/targets
            x_test (pd.DataFrame): Test feature data
            y_test (pd.Series): Test labels/targets
            **kwargs: Additional keyword arguments for target computation

        Returns:
            ProbePredictionTarget: Contains target values tensor of shape
                [batch, target_dim] and optional metadata dict

        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return dimensionality of prediction targets.

        This property must return an integer indicating the size of the output
        dimension that the probe should predict. For example, if predicting gradients
        for 4 features plus a bias term, this would return 5.

        Returns:
            int: Number of dimensions in the target prediction values

        """
        pass

class SGDGradientPredictor(PredictionGenerator):
    """Generates SGD gradient targets for probe prediction.

    This class computes gradient targets by applying stochastic gradient descent
    steps using specified learning rates. It generates target gradients that
    a probe model should learn to predict based on the model's activations.

    The gradient targets represent parameter updates that would occur during
    SGD optimization of a simple linear model with sigmoid activation.

    Attributes:
        learning_rates (List[float]): List of learning rates to use when
            computing SGD updates. Default is [0.001].

    """

    def __init__(self, learning_rates: List[float] = [0.001]):
        self.learning_rates = learning_rates

    def compute(self, x_train, y_train, x_test, y_test, **kwargs):
        target_output = calculate_sgd_gradients(
            x_train, y_train, x_test, y_test,
            learning_rates=self.learning_rates
        )
        return ProbePredictionTarget(
            values=target_output["gradients"],
            metadata={
                "final_losses": target_output["final_losses"],
                "initial_weights": target_output["initial_weights"]
            }
        )

    @property
    def output_dim(self) -> int:
        return 5  # 4 features + bias

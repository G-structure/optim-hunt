from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from optim_hunter.probe.mlp_probe import OptimizerProbe, TrainOptimizerProbe
from optim_hunter.probe.probe_prediction import PredictionGenerator


@dataclass
class PredictorExperimentResult:
    """Results from training a probe with a specific predictor."""

    predictor_name: str
    metrics_history: List[Dict[str, float]]
    final_probe_state: Dict[str, torch.Tensor]
    metadata: Optional[Dict] = None

class PredictorComparisonExperiment:
    """Experiment comparing different prediction generators."""

    def __init__(
        self,
        model: HookedTransformer,
        predictors: List[PredictionGenerator],
        dataset_fn,
        experiment_name: str,
        save_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.predictors = predictors
        self.dataset_fn = dataset_fn
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir) if save_dir else None
        self.device = device
        self.results: List[PredictorExperimentResult] = []

    def run(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
    ) -> List[PredictorExperimentResult]:
        """Run experiment for each predictor."""

        for predictor in self.predictors:
            print(f"\nTraining probe with predictor: {predictor.__class__.__name__}")

            # Initialize probe for this predictor
            probe = OptimizerProbe(
                d_model=self.model.cfg.d_model,
                predictor=predictor,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )

            # Train probe
            metrics_history = train_optimizer_probe(
                model=self.model,
                probe=probe,
                dataset_fn=self.dataset_fn,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=self.device
            )

            # Save probe state
            probe_state = {
                name: param.detach().cpu()
                for name, param in probe.named_parameters()
            }

            # Record results
            result = PredictorExperimentResult(
                predictor_name=predictor.__class__.__name__,
                metrics_history=metrics_history,
                final_probe_state=probe_state
            )
            self.results.append(result)

            # Save checkpoint if directory specified
            if self.save_dir:
                self._save_checkpoint(result)

        return self.results

    def _save_checkpoint(self, result: PredictorExperimentResult):
        """Save experiment results to disk."""
        save_dir = self.save_dir / self.experiment_name / result.predictor_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics history
        metrics_df = pd.DataFrame(result.metrics_history)
        metrics_df.to_csv(save_dir / "metrics.csv", index=False)

        # Save probe state
        torch.save(result.final_probe_state, save_dir / "probe_state.pt")

        # Save metadata if present
        if result.metadata:
            pd.to_pickle(result.metadata, save_dir / "metadata.pkl")

    def plot_results(self, save_plots: bool = True):
        """Plot comparison of metrics across predictors."""

        # Plot loss curves
        plt.figure(figsize=(10, 6))
        for result in self.results:
            epochs = [m["epoch"] for m in result.metrics_history]
            losses = [m["loss"] for m in result.metrics_history]
            plt.plot(epochs, losses, label=result.predictor_name)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{self.experiment_name}: Training Loss Comparison")
        plt.legend()
        plt.grid(True)

        if save_plots and self.save_dir:
            plt.savefig(self.save_dir / self.experiment_name / "loss_comparison.png")
        plt.show()

        # Plot cosine similarity curves
        plt.figure(figsize=(10, 6))
        for result in self.results:
            epochs = [m["epoch"] for m in result.metrics_history]
            cos_sims = [m["cos_sim"] for m in result.metrics_history]
            plt.plot(epochs, cos_sims, label=result.predictor_name)

        plt.xlabel("Epoch")
        plt.ylabel("Cosine Similarity")
        plt.title(f"{self.experiment_name}: Prediction Similarity Comparison")
        plt.legend()
        plt.grid(True)

        if save_plots and self.save_dir:
            plt.savefig(self.save_dir / self.experiment_name / "similarity_comparison.png")
        plt.show()

def run_predictor_comparison(
    model: HookedTransformer,
    predictors: List[PredictionGenerator],
    dataset_fn,
    experiment_name: str,
    save_dir: Optional[str] = None,
    **training_kwargs
) -> PredictorComparisonExperiment:
    """Helper function to run a predictor comparison experiment."""

    experiment = PredictorComparisonExperiment(
        model=model,
        predictors=predictors,
        dataset_fn=dataset_fn,
        experiment_name=experiment_name,
        save_dir=save_dir
    )

    experiment.run(**training_kwargs)
    experiment.plot_results()

    return experiment

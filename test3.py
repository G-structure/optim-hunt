from optim_hunter.llama_model import load_llama_model
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.probe_prediction import (
    SGDGradientPredictor)
from optim_hunter.experiments.train_probes import run_predictor_comparison
# Load model
model = load_llama_model()

# Define predictors to compare
predictors = [
    SGDGradientPredictor(learning_rates=[0.001]),
]

# Run comparison experiment
experiment = run_predictor_comparison(
    model=model,
    predictors=predictors,
    dataset_fn=get_dataset_friedman_2,
    experiment_name="predictor_comparison_1",
    save_dir="./experiments",
    # Training kwargs
    num_epochs=1000,
    batch_size=1,
    learning_rate=1e-4
)

# Access results
for result in experiment.results:
    print(f"\nResults for {result.predictor_name}:")
    print(f"Final loss: {result.metrics_history[-1]['loss']:.4f}")
    print(f"Final cos sim: {result.metrics_history[-1]['cos_sim']:.4f}")

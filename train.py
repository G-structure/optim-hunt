from optim_hunter.probe.regression_predictors import get_all_predictors
from optim_hunter.experiments.train_probes import PredictorComparisonExperiment
from optim_hunter.llama_model import load_llama_model
from optim_hunter.datasets import get_original2

model = load_llama_model()
dataset_fn = get_original2
# Get all predictors
predictors = get_all_predictors()

# Train probes for each intermediate
experiment = PredictorComparisonExperiment(
    model=model,
    predictors=[pred() for pred in predictors.values()],
    dataset_fn=dataset_fn,
    experiment_name="regression_intermediates"
)

# Run the experiment
results = experiment.run()

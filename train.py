from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.llama_model import load_gpt2_model, load_llama_model
from optim_hunter.experiments.optim_probe import OptimizerProbe, train_optimizer_probe
# Load model
model = load_llama_model()
# model = load_gpt2_model()
∏π

# Create probe
probe = OptimizerProbe(
    residual_stream_dim=model.cfg.d_model,
    num_features=4,  # Friedman #2 has 4 features
    hidden_dim=256,
    num_layers=2
)

# Train probe
metrics_history = train_optimizer_probe(
    model=model,
    probe=probe,
    dataset_fn=get_dataset_friedman_2,
    num_epochs=100000,
    batch_size=1
)

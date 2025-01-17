from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.llama_model import load_llama_model, load_gpt2_model
from optim_hunter.experiments.p import (
    OptimizerProbe,
    train_optimizer_probe
)

# Load model
model = load_llama_model()
# model = load_gpt2_model()

# Suppose the Friedman #2 dataset has 4 features => target param dimension = 4 + 1 = 5
probe = OptimizerProbe(
    d_model=model.cfg.d_model,   # 4096 for LLaMA-7B
    hidden_dim=256,
    num_layers=2,
)

# Train probe
metrics_history = train_optimizer_probe(
    model=model,
    probe=probe,
    dataset_fn=get_dataset_friedman_2,
    num_epochs=1000,   # or 100000 if you have compute
    batch_size=1
)

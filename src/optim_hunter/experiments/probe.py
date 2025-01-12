import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch as t
from torch import nn, optim
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.utils import prepare_prompt, slice_dataset

logger = logging.getLogger(__name__)

class GradientProbeMLP(nn.Module):
    """MLP probe for predicting gradient updates from residual stream."""

    def __init__(self, d_model: int, d_hidden: int = 128, n_layers: int = 2,
                 dropout: float = 0.1):
        """Initialize the gradient probe MLP.

        Args:
            d_model: Dimension of the input model embeddings
            d_hidden: Dimension of hidden layers
            n_layers: Number of hidden layers
            dropout: Dropout probability

        """
        super().__init__()

        self.input_layer = nn.Linear(d_model + 1, d_hidden)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(d_hidden, d_hidden)
            for _ in range(n_layers - 1)
        ])
        self.output_layer = nn.Linear(d_hidden, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, residual: t.Tensor, learning_rate: t.Tensor) -> t.Tensor:
        """Forward pass of the MLP.

        Args:
            residual: The residual stream tensor
            learning_rate: The learning rate tensor

        Returns:
            The predicted gradient updates

        """
        x = t.cat([residual, learning_rate], dim=-1)
        x = self.activation(self.input_layer(x))
        x = self.dropout(x)

        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = self.dropout(x)

        return self.output_layer(x)

class ResidualCollector:
    """Collects residual streams from transformer layers."""

    def __init__(self):
        """Initialize ResidualCollector.

        Creates an empty dictionary to store residual streams.
        """
        self.residuals = {}

    def hook_fn(self, value: t.Tensor, hook: HookPoint):
        """Store residual stream from hook."""
        self.residuals[hook.name] = value.detach()[:, -1, :]

    def get_hooks(self, model: HookedTransformer) -> List[Tuple[str, Callable]]:
        """Get all hooks needed for collection."""
        hooks = [('hook_embed', self.hook_fn)]
        for layer_idx in range(model.cfg.n_layers):
            hooks.append((
                f'blocks.{layer_idx}.hook_resid_post',
                self.hook_fn
            ))
        return hooks

def compute_gradient_step(
    prediction: float,
    target: float,
    learning_rate: float
) -> float:
    """Compute one step of gradient descent for MSE loss."""
    gradient = 2 * (prediction - target)
    return -learning_rate * gradient

def extract_prediction(logits: t.Tensor, model: HookedTransformer) -> float:
    """Extract numerical prediction from model's logits."""
    # Get the predicted token IDs
    predicted_token_ids = logits.argmax(dim=-1)
    # Decode the predicted tokens
    predicted_tokens = model.to_string(predicted_token_ids[0, -1:])
    predicted_text = predicted_tokens.strip()
    try:
        prediction = float(predicted_text)
    except ValueError:
        prediction = 0.0  # Or handle as appropriate
    return prediction

def train_probe_online(
    model: HookedTransformer,
    n_samples: int = 1000,
    learning_rates: Optional[List[float]] = None,
    seq_len: int = 5,
    device: str = "cuda" if t.cuda.is_available() else "cpu"
) -> Tuple[GradientProbeMLP, Dict[str, List[float]]]:
    """Train gradient probe across all layers synchronously."""
    if learning_rates is None:
        learning_rates = [10**i for i in range(-4, 1)]

    # Initialize components
    probe = GradientProbeMLP(model.cfg.d_model).to(device)
    optimizer = optim.Adam(probe.parameters())
    criterion = nn.MSELoss()
    collector = ResidualCollector()

    # Training metrics history
    history = {
        "loss": [],
        "mse": [],
        "cosine_similarity": [],
        "loss_by_layer": [[] for _ in range(model.cfg.n_layers + 1)]
    }

    # Training loop
    for seed in tqdm(range(n_samples), desc="Training probe"):
        # Generate dataset
        x_train, y_train, x_test, y_test = get_dataset_friedman_2(seed)
        x_train, y_train, x_test, y_test = slice_dataset(
            x_train, y_train, x_test, y_test, seq_len
        )

        # Prepare prompt
        prompt = prepare_prompt(x_train, y_train, x_test)
        tokens = model.to_tokens(prompt, prepend_bos=True)

        # Get model prediction and collect residuals
        collector.residuals = {}  # Reset residuals
        hooks = collector.get_hooks(model)
        logits = model.run_with_hooks(tokens.to(device), fwd_hooks=hooks)
        prediction = extract_prediction(logits, model)
        target = float(y_test.iloc[0])

        # Prepare learning rates and true gradients
        lr_tensor = t.tensor(learning_rates, device=device).unsqueeze(1)
        true_grads = t.tensor(
            [compute_gradient_step(prediction, target, lr)
             for lr in learning_rates],
            device=device
        ).unsqueeze(1)

        # Train on collected residuals
        for layer_idx, (hook_name, residual) in \
                enumerate(collector.residuals.items()):
            # Prepare inputs for probe
            residual = residual.to(device)
            residual_expanded = residual.repeat(len(learning_rates), 1)

            # Forward pass through probe
            optimizer.zero_grad()
            pred_grads = probe(residual_expanded, lr_tensor)
            loss = criterion(pred_grads, true_grads)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Record metrics
            with t.no_grad():
                mse = criterion(pred_grads, true_grads).item()
                cos_sim = t.nn.functional.cosine_similarity(
                    pred_grads.squeeze(),
                    true_grads.squeeze(),
                    dim=0
                ).item()

                history["loss"].append(loss.item())
                history["mse"].append(mse)
                history["cosine_similarity"].append(cos_sim)
                history["loss_by_layer"][layer_idx].append(loss.item())

        # Log progress
        if (seed + 1) % 100 == 0:
            log_metrics(seed, history)

    return probe, history

def evaluate_probe_online(
    probe: GradientProbeMLP,
    model: HookedTransformer,
    n_samples: int = 100,
    learning_rates: Optional[List[float]] = None,
    seq_len: int = 5,
    device: str = "cuda" if t.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """Evaluate probe across all layers."""
    if learning_rates is None:
        learning_rates = [10**i for i in range(-4, 1)]

    probe.eval()
    criterion = nn.MSELoss()
    collector = ResidualCollector()

    metrics = {
        "total_mse": 0.0,
        "total_cos_sim": 0.0,
        "n_total": 0,
        "layer_mse": [0.0] * (model.cfg.n_layers + 1),
        "layer_cos_sim": [0.0] * (model.cfg.n_layers + 1),
        "layer_counts": [0] * (model.cfg.n_layers + 1)
    }

    with t.no_grad():
        for seed in tqdm(range(n_samples), desc="Evaluating probe"):
            # Generate dataset
            x_train, y_train, x_test, y_test = \
                get_dataset_friedman_2(seed + 10000)
            x_train, y_train, x_test, y_test = slice_dataset(
                x_train, y_train, x_test, y_test, seq_len
            )

            # Prepare prompt
            prompt = prepare_prompt(x_train, y_train, x_test)
            tokens = model.to_tokens(prompt, prepend_bos=True)

            # Get model prediction and collect residuals
            collector.residuals = {}  # Reset residuals
            hooks = collector.get_hooks(model)
            logits = model.run_with_hooks(tokens.to(device), fwd_hooks=hooks)
            prediction = extract_prediction(logits, model)
            target = float(y_test.iloc[0])

            # Prepare learning rates and true gradients
            lr_tensor = t.tensor(learning_rates, device=device).unsqueeze(1)
            true_grads = t.tensor(
                [compute_gradient_step(prediction, target, lr)
                 for lr in learning_rates],
                device=device
            ).unsqueeze(1)

            # Evaluate on collected residuals
            for layer_idx, (hook_name, residual) in enumerate(collector.residuals.items()):
                residual = residual.to(device)
                residual_expanded = residual.repeat(len(learning_rates), 1)

                pred_grads = probe(residual_expanded, lr_tensor)

                mse = criterion(pred_grads, true_grads).item()
                cos_sim = t.nn.functional.cosine_similarity(
                    pred_grads.squeeze(),
                    true_grads.squeeze(),
                    dim=0
                ).item()

                metrics["total_mse"] += mse
                metrics["total_cos_sim"] += cos_sim
                metrics["n_total"] += 1

                metrics["layer_mse"][layer_idx] += mse
                metrics["layer_cos_sim"][layer_idx] += cos_sim
                metrics["layer_counts"][layer_idx] += 1

    # Calculate final metrics
    return {
        "mse": metrics["total_mse"] / metrics["n_total"],
        "cosine_similarity": metrics["total_cos_sim"] / metrics["n_total"],
        "layer_mse": [
            mse / count if count > 0 else 0
            for mse, count in zip(metrics["layer_mse"], metrics["layer_counts"])
        ],
        "layer_cos_sim": [
            cos / count if count > 0 else 0
            for cos, count in zip(metrics["layer_cos_sim"], metrics["layer_counts"])
        ]
    }

def log_metrics(seed: int, history: Dict[str, List[float]]) -> None:
    """Log training metrics."""
    last_losses = history["loss"][-100:] if len(history["loss"]) >= 100 else history["loss"]
    last_mses = history["mse"][-100:] if len(history["mse"]) >= 100 else history["mse"]
    last_cos = history["cosine_similarity"][-100:] if len(history["cosine_similarity"]) >= 100 else history["cosine_similarity"]

    avg_loss = sum(last_losses) / len(last_losses)
    avg_mse = sum(last_mses) / len(last_mses)
    avg_cos = sum(last_cos) / len(last_cos)

    layer_losses = []
    for layer_hist in history["loss_by_layer"]:
        last_layer_losses = layer_hist[-100:] if len(layer_hist) >= 100 else layer_hist
        avg_layer_loss = sum(last_layer_losses) / max(len(last_layer_losses), 1)
        layer_losses.append(avg_layer_loss)

    logger.info(
        f"Sample {seed+1}, "
        f"Avg Loss: {avg_loss:.6f}, "
        f"Avg MSE: {avg_mse:.6f}, "
        f"Avg Cos Sim: {avg_cos:.6f}"
    )
    logger.info("Layer-wise losses: " +
        ", ".join(f"L{i}: {loss:.6f}"
        for i, loss in enumerate(layer_losses)))

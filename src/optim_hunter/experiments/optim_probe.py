from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer


@dataclass
class OptimizerProbeOutput:
    """Structured output from the optimizer probe."""

    gradient: torch.Tensor  # Predicted gradient


class OptimizerProbe(nn.Module):
    """Neural probe predicting optimization steps from model residual stream."""

    def __init__(
        self,
        residual_stream_dim: int,
        num_features: int,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        """Initialize the optimizer probe with dimensions and architecture."""
        super().__init__()

        # Input dimensions
        self.residual_stream_dim = residual_stream_dim
        self.num_features = num_features

        # Number of parameters in the target model (weights + bias)
        self.target_param_dim = num_features + 1

        # Layers to process residual stream and learning rate
        layers = []
        # First layer processes residual stream and learning rate
        layers.append(nn.Linear(residual_stream_dim + 1, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer predicts gradients
        layers.append(nn.Linear(hidden_dim, self.target_param_dim))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        residual_stream: torch.Tensor,  # Shape: [batch_size, residual_dim]
        learning_rate: torch.Tensor,    # Shape: [batch_size, 1]
    ) -> OptimizerProbeOutput:
        """Predict optimization step given residual stream and learning rate."""
        # Ensure residual stream has correct shape
        if residual_stream.dim() == 1:
            # Add batch dimension if needed
            residual_stream = residual_stream.unsqueeze(0)

        # Ensure learning rate has correct shape
        if learning_rate.dim() == 1:
            # Add batch dimension if needed
            learning_rate = learning_rate.unsqueeze(0)

        # Ensure both tensors have same batch dimension
        batch_size = residual_stream.shape[0]
        if residual_stream.shape[0] != learning_rate.shape[0]:
            learning_rate = learning_rate.expand(batch_size, -1)

        print("Residual stream shape:", residual_stream.shape)
        print("Learning rate shape:", learning_rate.shape)

        # Concatenate residual and learning rate
        # Use dim=-1 for last dimension
        probe_input = torch.cat([residual_stream, learning_rate], dim=-1)

        # Get probe predictions
        predicted_gradients = self.network(probe_input)

        return OptimizerProbeOutput(
            gradient=predicted_gradients,
        )

def train_optimizer_probe(
    model: HookedTransformer,
    probe: OptimizerProbe,
    dataset_fn,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Dict[str, float]]:
    """Train the optimizer probe."""
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    metrics_history = []

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_grad_cos_sim = []

        # Generate batch of examples
        for batch_idx in range(batch_size):
            # Get fresh dataset with different random seed
            x_train, y_train, x_test, y_test = dataset_fn(
                random_state=epoch*batch_size + batch_idx
            )

            from optim_hunter.utils import prepare_prompt
            prompt = prepare_prompt(x_train, y_train, x_test)

            # Get model residual stream
            with torch.no_grad():
                tokens = model.to_tokens(prompt, prepend_bos=True)
                model_out, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name.endswith('resid_post'),
                    return_cache_object=True
                )

                residual_layers = []
                for layer_idx in range(model.cfg.n_layers):
                    key = f'blocks.{layer_idx}.hook_resid_post'
                    if key in cache:
                        # Move each residual to CPU, detaching from the graph
                        resid_cpu = cache[key].detach().cpu()
                        residual_layers.append(resid_cpu)

                # Stack on CPU
                residual_stream = torch.stack(residual_layers, dim=1)

                # Clear CUDA cache to free memory
                del cache
                torch.cuda.empty_cache()

                # If you need further processing on GPU, move them back
                residual_stream = residual_stream.to(device)
                residual_stream = residual_stream.view(-1, residual_stream.shape[-1])

            torch.set_grad_enabled(True)

             # First calculate target gradients before moving data to device
            target_output = calculate_sgd_gradients(
                x_train, y_train, x_test, y_test,
                learning_rates=[learning_rate]
            )
             # Ensure the gradient has a batch dimension
            target_gradients = target_output['gradients'][0:1].to(device)

            # Sample random learning rate between 0 and 0.1
            lr_tensor = torch.rand(1, 1, device=device) * 0.1

            # Get probe predictions
            probe_output = probe(
                residual_stream,
                lr_tensor,
            )

            # Compute loss
            loss = nn.MSELoss()(probe_output.gradient, target_gradients)

            # Compute gradient cosine similarity
            cos_sim = nn.CosineSimilarity(dim=1)(  # Change dim=0 to dim=1
                probe_output.gradient, target_gradients
            ).mean()

            # Update probe
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_grad_cos_sim.append(cos_sim.item())

        # Record metrics
        metrics = {
            'epoch': epoch,
            'loss': sum(epoch_losses) / len(epoch_losses),
            'grad_cos_sim': sum(epoch_grad_cos_sim) / len(epoch_grad_cos_sim)
        }
        metrics_history.append(metrics)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {metrics['loss']:.4f}, "
                  f"Grad Cos Sim = {metrics['grad_cos_sim']:.4f}")

    return metrics_history

def calculate_sgd_gradients(
    x_train,
    y_train,
    x_test,
    y_test,
    learning_rates=[0.001, 0.01, 0.1, 1.0]
):
    """Calculate gradients after one step of SGD for multiple learning rates."""
    # Convert pandas to numpy then to torch tensors
    if isinstance(x_train, pd.DataFrame):
        x_train = torch.tensor(x_train.values, dtype=torch.float32)
    if isinstance(y_train, pd.Series):
        y_train = torch.tensor(y_train.values, dtype=torch.float32)

    # Ensure tensors are on CPU and require gradients
    if torch.is_tensor(x_train):
        x_train = x_train.cpu()
    else:
        x_train = torch.tensor(x_train, dtype=torch.float32)

    if torch.is_tensor(y_train):
        y_train = y_train.cpu()
    else:
        y_train = torch.tensor(y_train, dtype=torch.float32)

    # Initialize model (weights and bias) with requires_grad=True
    num_features = x_train.shape[1]
    weights = torch.randn(num_features, dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    # Forward pass function with non-linearity
    def predict(x, w, b):
        return torch.sigmoid(torch.matmul(x.float(), w) + b)

    # Loss function
    def loss_fn(y_pred, y_true):
        return torch.mean((y_pred - y_true.float()) ** 2)

    # Initial prediction and loss
    y_pred = predict(x_train, weights, bias)
    loss = loss_fn(y_pred, y_train)

    # Calculate gradients without retaining the graph
    loss.backward()  # Removed retain_graph=True

    # Store gradients
    if weights.grad is not None:
        weight_grad = weights.grad.clone()
    else:
        weight_grad = torch.zeros_like(weights)

    if bias.grad is not None:
        bias_grad = bias.grad.clone()
    else:
        bias_grad = torch.zeros_like(bias)

    # Store results for each learning rate
    results = {
        'gradients': [],
        'updated_weights': [],
        'final_losses': [],
        'initial_weights': torch.cat([weights.detach(), bias.detach()]),
        'initial_loss': loss.item()
    }

    # Try different learning rates
    for lr in learning_rates:
        with torch.no_grad():
            new_w = weights - lr * weight_grad
            new_b = bias - lr * bias_grad

            # Calculate new prediction and loss
            new_pred = predict(x_train, new_w, new_b)
            new_loss = loss_fn(new_pred, y_train)

            # Store results
            results['gradients'].append(torch.cat([weight_grad, bias_grad]))
            results['updated_weights'].append(torch.cat([new_w, new_b]))
            results['final_losses'].append(new_loss.item())

    # Convert lists to tensors
    results['gradients'] = torch.stack(results['gradients'])
    results['updated_weights'] = torch.stack(results['updated_weights'])
    results['final_losses'] = torch.tensor(results['final_losses'])

    return results

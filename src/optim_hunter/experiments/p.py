from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

from optim_hunter.utils import extract_numeric_from_logits

@dataclass
class OptimizerProbeOutput:
    """Structured output from the optimizer probe."""
    gradient: torch.Tensor  # Predicted gradient of shape [batch, 5]


class OptimizerProbe(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        # We want [d_model + 1] for input_dim, because we add the LR as an extra feature
        input_dim = d_model + 1

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))  # <-- was d_model, now d_model+1
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 5))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        residual_stream: torch.Tensor,  # [B, seq_len, d_model]
        lr_tensor: torch.Tensor         # [B, 1]
    ) -> torch.Tensor:
        B, S, D = residual_stream.shape

        # Flatten to [B*S, d_model]
        flat_resid = residual_stream.view(B * S, D)

        # Expand LR to match B*S so shape is [B*S, 1]
        lr_expanded = lr_tensor.view(B, 1).expand(B, S).view(B * S, 1)

        # Concatenate => [B*S, d_model + 1]
        net_input = torch.cat([flat_resid, lr_expanded], dim=-1)

        # MLP => [B*S, 5]
        out = self.network(net_input)

        # Reshape => [B, S, 5], then average => [B, 5]
        out = out.view(B, S, 5).mean(dim=1)
        return out

def train_optimizer_probe(
    model: HookedTransformer,
    probe: OptimizerProbe,
    dataset_fn,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Dict[str, float]]:
    """
    Train the optimizer probe.
    We assume the final target gradient has shape [batch, 5].
    """
    torch.set_grad_enabled(True)

    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    metrics_history = []

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_grad_cos_sim = []

        # We'll generate 'batch_size' examples each epoch
        for batch_idx in range(batch_size):
            # Get fresh dataset with different random seed
            x_train, y_train, x_test, y_test = dataset_fn(
                random_state=epoch * batch_size + batch_idx
            )

            # ~~~~~ Generate the prompt and residual ~~~~~
            from optim_hunter.utils import prepare_prompt
            prompt = prepare_prompt(x_train, y_train, x_test)
            with torch.no_grad():
                tokens = model.to_tokens(prompt, prepend_bos=True)
                model_out, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name.endswith("resid_post"),
                    return_cache_object=True
                )
                # model_numerical_pred = extract_numeric_from_logits(model, model_out)

                # We'll gather all 32 layers
                # shape: [batch=1, seq_len, d_model]
                # But we have it for each of 32 layers => need stacking
                # E.g. shape [1, 32, seq_len, d_model]
                residual_layers = []
                for layer_idx in range(model.cfg.n_layers):
                    key = f"blocks.{layer_idx}.hook_resid_post"
                    if key in cache:
                        # shape: [1, seq_len, d_model]
                        resid_cpu = cache[key].detach().cpu()
                        residual_layers.append(resid_cpu)

                # residual_layers is a list of 32 [1, seq_len, d_model]
                # => stack => [32, 1, seq_len, d_model]
                residual_stream = torch.stack(residual_layers, dim=0)
                # reorder to [batch=1, n_layers=32, seq_len, d_model]
                residual_stream = residual_stream.permute(1, 0, 2, 3).contiguous()

                del cache
                torch.cuda.empty_cache()

            # ~~~~~ Flatten layers dimension ~~~~~
            # from [1, 32, seq_len, d_model] => [1, 32*seq_len, d_model]
            B, L, S, D = residual_stream.shape  # e.g. B=1, L=32, S=2497, D=4096
            residual_stream = residual_stream.view(B, L * S, D).to(device)

            torch.set_grad_enabled(True)
            # ~~~~~ Compute target gradient ~~~~~
            # Suppose each batch has 1 set of target gradients => shape [1, 5]
            target_output = calculate_sgd_gradients(
                x_train, y_train, x_test, y_test,
                learning_rates=[learning_rate]
            )
            # We'll get e.g. [1, 5]
            target_gradients = target_output["gradients"][0:1].to(device)

            # ~~~~~ We'll define lr_tensor ~~~~~
            # shape [batch=1, 1]
            lr_tensor = torch.rand(B, 1, device=device) * 0.001

            # ~~~~~ Forward pass the probe ~~~~~
            probe_output = probe(
                residual_stream,  # [B, 79968, 4096], e.g.
                lr_tensor         # [B, 1]
            )  # -> shape [B, 5]

            # ~~~~~ Compute loss ~~~~~
            # shape: probe_output.gradient => [B, 5]
            # shape: target_gradients => [1, 5]
            # If B=1, that matches. If B>1, ensure your target has [B, 5] as well.
            loss = nn.MSELoss()(probe_output, target_gradients)

            # Gradient cosine similarity
            cos_sim = nn.CosineSimilarity(dim=1)(
                probe_output, target_gradients
            ).mean()

            # ~~~~~ Backprop and optimize ~~~~~
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_grad_cos_sim.append(cos_sim.item())
            if epoch % 10 == 0:
                sample = probe_output.detach().cpu().numpy()
                print(f'Model prediction: {sample}')
                sample_tgt = target_gradients.detach().cpu().numpy()
                print(f'Target gradient: {sample_tgt}')
                print(f'Target loss: {target_output["final_losses"]}')
                print(f'MLP Loss: {loss.item():.6f}')


        # Record metrics
        metrics = {
            "epoch": epoch,
            "loss": sum(epoch_losses) / len(epoch_losses),
            "grad_cos_sim": sum(epoch_grad_cos_sim) / len(epoch_grad_cos_sim),
        }
        metrics_history.append(metrics)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Loss = {metrics['loss']:.4f}, "
                f"Grad Cos Sim = {metrics['grad_cos_sim']:.4f}"
            )

    return metrics_history


def calculate_sgd_gradients(
    x_train,
    y_train,
    x_test,
    y_test,
    learning_rates=[0.001, 0.01, 0.1, 1.0]
    ):
    """
    Calculate gradients after one step of SGD for multiple learning rates
    using a simple linear model with sigmoid activation.
    Returns a dict containing:
    - gradients: shape [num_learning_rates, 5] if num_features=4 => (4 + 1 bias)
    - updated_weights: ...
    - final_losses: ...
    - initial_weights: ...
    - initial_loss: ...
    """

    # If pandas, convert to numpy
    if isinstance(x_train, pd.DataFrame):
        x_train = x_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    # Convert to torch
    x_train = torch.tensor(x_train, dtype=torch.float32, device="cpu")
    y_train = torch.tensor(y_train, dtype=torch.float32, device="cpu")

    # Initialize random weights + bias
    num_features = x_train.shape[1]
    weights = torch.randn(num_features, dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    # Model + loss
    def predict(x, w, b):
        return torch.sigmoid(torch.matmul(x, w) + b)

    def loss_fn(y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)

    # Forward pass
    y_pred = predict(x_train, weights, bias)
    loss = loss_fn(y_pred, y_train)

    print("Predictions:", y_pred[:5].detach().numpy())

    # Backprop
    loss.backward()
    weight_grad = weights.grad.clone()
    bias_grad = bias.grad.clone()

    results = {
        "gradients": [],
        "updated_weights": [],
        "final_losses": [],
        "initial_weights": torch.cat([weights.detach(), bias.detach()]),
        "initial_loss": loss.item(),
    }

    for lr in learning_rates:
        with torch.no_grad():
            new_w = weights - lr * weight_grad
            new_b = bias - lr * bias_grad
            new_pred = predict(x_train, new_w, new_b)
            new_loss = loss_fn(new_pred, y_train)
            print("New prediction:", new_pred[:5].detach().numpy())
            # Store
            results["gradients"].append(torch.cat([weight_grad, bias_grad]))
            results["updated_weights"].append(torch.cat([new_w, new_b]))
            results["final_losses"].append(new_loss.item())

    results["gradients"] = torch.stack(results["gradients"])  # shape [len(lr), 5]
    results["updated_weights"] = torch.stack(results["updated_weights"])  # [lr, 5]
    results["final_losses"] = torch.tensor(results["final_losses"])       # [lr]

    return results

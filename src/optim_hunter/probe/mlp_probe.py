from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

from optim_hunter.probe.probe_prediction import (
    PredictionGenerator,
    ProbePredictionTarget,
)


class OptimizerProbe(nn.Module):
    def __init__(
        self,
        d_model: int,
        predictor: PredictionGenerator,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        self.predictor = predictor

        # Input dim includes learning rate feature
        input_dim = d_model + 1
        output_dim = predictor.output_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        residual_stream: torch.Tensor,  # [B, seq_len, d_model]
        lr_tensor: torch.Tensor         # [B, 1]
    ) -> torch.Tensor:
        B, S, D = residual_stream.shape

        # Flatten to [B*S, d_model]
        flat_resid = residual_stream.view(B * S, D)

        # Expand LR to match B*S
        lr_expanded = lr_tensor.view(B, 1).expand(B, S).view(B * S, 1)

        # Concatenate => [B*S, d_model + 1]
        net_input = torch.cat([flat_resid, lr_expanded], dim=-1)

        # MLP => [B*S, output_dim]
        out = self.network(net_input)

        # Reshape => [B, S, output_dim], then average => [B, output_dim]
        out = out.view(B, S, -1).mean(dim=1)
        return out

def TrainOptimizerProbe(
    model: HookedTransformer,
    probe: OptimizerProbe,
    dataset_fn,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Dict[str, float]]:
    """Train the optimizer probe."""

    torch.set_grad_enabled(True)
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    metrics_history = []

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_cos_sim = []

        for batch_idx in range(batch_size):
            # Handle both callable and direct dataset cases
            if callable(dataset_fn):
                x_train, y_train, x_test, y_test = dataset_fn(
                    random_state=epoch * batch_size + batch_idx
                )
            else:
                # If dataset_fn is actually the dataset tuple
                x_train, y_train, x_test, y_test = dataset_fn

            # Generate prompt and get residual stream
            from optim_hunter.utils import prepare_prompt
            prompt = prepare_prompt(x_train, y_train, x_test)

            with torch.no_grad():
                tokens = model.to_tokens(prompt, prepend_bos=True)
                model_out, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name.endswith("resid_post"),
                    return_cache_object=True
                )

                # Gather residuals from all layers
                residual_layers = []
                for layer_idx in range(model.cfg.n_layers):
                    key = f"blocks.{layer_idx}.hook_resid_post"
                    if key in cache:
                        resid_cpu = cache[key].detach().cpu()
                        residual_layers.append(resid_cpu)

                residual_stream = torch.stack(residual_layers, dim=0)
                residual_stream = residual_stream.permute(1, 0, 2, 3).contiguous()

                del cache
                torch.cuda.empty_cache()

            # Reshape residual stream
            B, L, S, D = residual_stream.shape
            residual_stream = residual_stream.view(B, L * S, D).to(device)

            torch.set_grad_enabled(True)

            # Compute target values using the predictor
            target_output = probe.predictor.compute(
                x_train, y_train, x_test, y_test
            )
            target_values = target_output.values.to(device)

            # Generate random learning rate tensor
            lr_tensor = torch.rand(B, 1, device=device) * 0.001

            # Forward pass through probe
            probe_output = probe(residual_stream, lr_tensor)

            # Compute loss and similarity
            loss = nn.MSELoss()(probe_output, target_values)
            cos_sim = nn.CosineSimilarity(dim=1)(
                probe_output, target_values
            ).mean()

            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_cos_sim.append(cos_sim.item())

            if epoch % 10 == 0:
                print(f'Model prediction: {probe_output.detach().cpu().numpy()}')
                print(f'Target values: {target_values.detach().cpu().numpy()}')
                if target_output.metadata:
                    print(f'Metadata: {target_output.metadata}')
                print(f'MLP Loss: {loss.item():.6f}')

        print('Epoch losses:', epoch_losses)
        # Record metrics
        metrics = {
            "epoch": epoch,
            "loss": sum(epoch_losses) / len(epoch_losses),
            "cos_sim": sum(epoch_cos_sim) / len(epoch_cos_sim),
        }
        metrics_history.append(metrics)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Loss = {metrics['loss']:.4f}, "
                f"Cos Sim = {metrics['cos_sim']:.4f}"
            )

    return metrics_history

import torch
import pandas as pd

def calculate_sgd_gradients(
    x_train,
    y_train,
    x_test,
    y_test,
    learning_rates=[0.001, 0.01, 0.1, 1.0]
    ):
    """Calculate gradients after one step of SGD for multiple learning rates.

    This function calculates gradients after one step of stochastic gradient
    descent using multiple learning rates and a simple linear model with sigmoid
    activation. It tracks various metrics like gradients, weights and losses.

    Args:
        x_train (Union[pd.DataFrame, np.ndarray]): Training features matrix
        y_train (Union[pd.Series, np.ndarray]): Training labels
        x_test (Union[pd.DataFrame, np.ndarray]): Test features matrix
        y_test (Union[pd.Series, np.ndarray]): Test labels
        learning_rates (List[float]): List of learning rates to try.
            Defaults to [0.001, 0.01, 0.1, 1.0].

    Returns:
        Dict[str, Union[torch.Tensor, float]]: Dictionary containing:
            - gradients: shape [num_learning_rates, num_features+1]
            - updated_weights: shape [num_learning_rates, num_features+1]
            - final_losses: shape [num_learning_rates]
            - initial_weights: shape [num_features+1]
            - initial_loss: scalar

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
    if weights.grad is None or bias.grad is None:
        raise RuntimeError("Gradients not computed!")
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

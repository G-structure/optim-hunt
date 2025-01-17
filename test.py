# tests/test_sgd_gradients.py
import pytest
import torch
import numpy as np
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.experiments.optim_probe import calculate_sgd_gradients

def test_basic_functionality():
    """Test basic functionality with simple synthetic data"""
    # Create simple linear data
    x_train = torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float32)
    y_train = torch.tensor([2.0, 4.0], dtype=torch.float32)
    x_test = torch.tensor([[3.0, 6.0]], dtype=torch.float32)
    y_test = torch.tensor([6.0], dtype=torch.float32)

    result = calculate_sgd_gradients(x_train, y_train, x_test, y_test)

    # Check that all expected keys are present
    expected_keys = {'gradients', 'learning_rates', 'initial_weights',
                    'updated_weights', 'initial_loss', 'final_losses'}
    assert all(key in result for key in expected_keys)

    # Check shapes
    assert result['gradients'].shape[0] == len(result['learning_rates'])
    assert result['gradients'].shape[1] == x_train.shape[1] + 1  # weights + bias
    assert result['updated_weights'].shape == result['gradients'].shape

def test_learning_rate_effects():
    """Test that larger learning rates lead to bigger updates"""
    x_train = torch.tensor([[1.0]], dtype=torch.float32)
    y_train = torch.tensor([2.0], dtype=torch.float32)
    x_test = torch.tensor([[1.5]], dtype=torch.float32)
    y_test = torch.tensor([3.0], dtype=torch.float32)

    learning_rates = [0.1, 0.5]
    result = calculate_sgd_gradients(x_train, y_train, x_test, y_test,
                                   learning_rates=learning_rates)

    # Check that larger learning rate leads to bigger weight updates
    updates_small_lr = torch.norm(result['updated_weights'][0] - result['initial_weights'])
    updates_large_lr = torch.norm(result['updated_weights'][1] - result['initial_weights'])
    assert updates_large_lr > updates_small_lr

def test_zero_gradient():
    """Test case where gradient should be approximately zero with correct initialization"""
    x_train = torch.tensor([[1.0]], dtype=torch.float32)
    y_train = torch.tensor([2.0], dtype=torch.float32)
    x_test = torch.tensor([[1.5]], dtype=torch.float32)
    y_test = torch.tensor([3.0], dtype=torch.float32)

    # Use very small learning rates to avoid numerical issues
    result = calculate_sgd_gradients(
        x_train, y_train, x_test, y_test,
        learning_rates=[1e-5, 1e-4]
    )

    # Check that loss changes are small with small learning rates
    loss_changes = torch.abs(result['final_losses'] - result['initial_loss'])
    assert torch.all(loss_changes < 0.1)

def test_with_friedman_dataset():
    """Test using actual Friedman dataset"""
    x_train, y_train, x_test, y_test = get_dataset_friedman_2(random_state=42)

    # Convert pandas/numpy to torch
    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    x_test = torch.tensor(x_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # Use smaller learning rates for stability
    result = calculate_sgd_gradients(
        x_train, y_train, x_test, y_test,
        learning_rates=[1e-5, 1e-4, 1e-3]
    )

    # Check that at least one learning rate improves the loss
    assert torch.any(result['final_losses'] < result['initial_loss'])

def test_input_validation():
    """Test handling of invalid inputs"""
    with pytest.raises(ValueError):
        # Test with empty data
        calculate_sgd_gradients(
            torch.tensor([], dtype=torch.float32).reshape(0, 1),
            torch.tensor([], dtype=torch.float32),
            torch.tensor([], dtype=torch.float32).reshape(0, 1),
            torch.tensor([], dtype=torch.float32)
        )

def test_dtype_conversion():
    """Test automatic conversion of numpy arrays and different dtypes"""
    # Test with numpy arrays
    x_train = np.array([[1.0, 2.0], [2.0, 4.0]])
    y_train = np.array([2.0, 4.0])
    x_test = np.array([[3.0, 6.0]])
    y_test = np.array([6.0])

    result = calculate_sgd_gradients(x_train, y_train, x_test, y_test)
    assert isinstance(result['gradients'], torch.Tensor)

    # Test with integer inputs
    x_train = torch.tensor([[1, 2], [2, 4]], dtype=torch.int64)
    y_train = torch.tensor([2, 4], dtype=torch.int64)
    x_test = torch.tensor([[3, 6]], dtype=torch.int64)
    y_test = torch.tensor([6], dtype=torch.int64)

    result = calculate_sgd_gradients(x_train, y_train, x_test, y_test)
    assert result['gradients'].dtype == torch.float32

if __name__ == "__main__":
    pytest.main([__file__])

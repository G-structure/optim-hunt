import torch

from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.experiments.optim_probe import calculate_sgd_gradients

x_train, y_train, x_test, y_test= get_dataset_friedman_2()
learning_rate = torch.rand(1, 1).to("cpu") * 0.1  # Random learning rate between 0 and 0.1

 # Get target gradients using calculate_sgd_gradients
target_output = calculate_sgd_gradients(
     x_train, y_train, x_test, y_test,
     learning_rates=[learning_rate.item()]
 )
print(target_output)


import numpy as np

arr = np.array([1, 2, 3])
if True:
    h = 1
print(arr * 2)
print("hi")

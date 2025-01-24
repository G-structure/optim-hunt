---
title: Reverse-Engineering Linear Regression in Language Models Notes
date: December 2024
reading_time: 30 minutes
---

# Data Preparation
We generate sequences of input-output pairs using the Friedman #2 formula, without revealing the formula to the model.
Here each line contains features \((x_1, x_2, ..., x_n)\) and their corresponding output \(y\).:

<<execute id="1" output="pandoc">>
```python
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.utils import slice_dataset, prepare_prompt, prepare_prompt_from_tokens, pad_numeric_tokens
from optim_hunter.llama_model import load_llama_model

llama_model = load_llama_model()

seq_len = 3  # Number of examples to show the model
x_train, y_train, x_test, y_test = get_dataset_friedman_2()
x_train, y_train, x_test, y_test = slice_dataset(
    x_train, y_train, x_test, y_test, seq_len
)
prompt = prepare_prompt(x_train, y_train, x_test)

x_train_tokens, y_train_tokens, x_test_tokens = pad_numeric_tokens(llama_model, x_train, y_train, x_test)
tokenized_prompt = prepare_prompt_from_tokens(llama_model, x_train_tokens, y_train_tokens, x_test_tokens)
decoded_prompt = llama_model.to_string(tokenized_prompt[0])

print(decoded_prompt)
```
<</execute>>

# Comparing LLM to SciKit Learn
2. **Baseline Models**: We compared Llama 3.1 against a comprehensive suite of traditional regression methods:
   - Linear models (Linear Regression, Ridge, Lasso)

   - Neural networks (MLPs with various architectures)

   - Ensemble methods (Random Forest, Gradient Boosting)

   - Local methods (k-Nearest Neighbors variants)

   - Simple baselines (mean, last value, random)

3. **Multiple Runs**: To ensure robust results, we evaluated performance across 100 different random sequences of 25 examples each.

<<execute id="2" output="raw">>
```python
from optim_hunter.experiments.regressors_comparison import compare_llm_and_regressors
from optim_hunter.sklearn_regressors import (
    linear_regression, ridge, lasso, mlp_universal_approximation_theorem1,
    mlp_universal_approximation_theorem2, mlp_universal_approximation_theorem3,
    mlp_deep1, mlp_deep2, mlp_deep3, random_forest, bagging,
    gradient_boosting, adaboost, bayesian_regression1,
    svm_regression, svm_and_scaler_regression, knn_regression,
    knn_regression_v2, knn_regression_v3, knn_regression_v4,
    knn_regression_v5_adaptable, kernel_ridge_regression,
    baseline_average, baseline_last, baseline_random, create_llm_regressor
)
from optim_hunter.datasets import get_dataset_friedman_2, get_original2
from optim_hunter.llama_model import load_llama_model

llama_model = load_llama_model()
model_name = "llama-8b"

seq_len = 25
batches = 100
regressors = [ ridge, lasso, mlp_universal_approximation_theorem1, mlp_universal_approximation_theorem2, mlp_universal_approximation_theorem3, mlp_deep1, mlp_deep2, mlp_deep3, random_forest, bagging, gradient_boosting, adaboost, bayesian_regression1, svm_regression, svm_and_scaler_regression, knn_regression, knn_regression_v2, knn_regression_v3, knn_regression_v4, knn_regression_v5_adaptable, kernel_ridge_regression, baseline_average, baseline_last, baseline_random]

html = compare_llm_and_regressors(dataset=get_original2, regressors=regressors, seq_len=seq_len, batches=batches, model=llama_model)
print(html)
```
<</execute>>

<<execute id="9" output="raw">>
```python
from optim_hunter.experiments.regressors_comparison import compare_llm_and_regressors
from optim_hunter.LR_methods import (
    solve_ols,  # replaces linear_regression
    solve_ridge_regression,  # replaces ridge
    solve_lasso_regression,  # replaces lasso
    solve_sgd,  # can replace some of the iterative methods
    solve_bayesian_linear_regression,  # replaces bayesian_regression1
    solve_normal_equation,  # another linear regression variant
    solve_ridge_regression_closed_form,  # another ridge variant
    solve_irls,  # iteratively reweighted least squares
    solve_pcr,  # principal component regression
    solve_knn  # replaces knn variants
)
from optim_hunter.datasets import get_dataset_friedman_2, get_original2
from optim_hunter.llama_model import load_llama_model

llama_model = load_llama_model()
model_name = "llama-8b"

seq_len = 25
batches = 100
regressors = [
    solve_ols,
    solve_ridge_regression,
    solve_lasso_regression,
    solve_sgd,
    solve_bayesian_linear_regression,
    solve_normal_equation,
    solve_ridge_regression_closed_form,
    solve_irls,
    solve_pcr,
    solve_knn
]

html = compare_llm_and_regressors(dataset=get_original2, regressors=regressors, seq_len=seq_len, batches=batches, model=llama_model)
print(html)
```
<</execute>>

<<execute id="8" output="raw">>
```python
from optim_hunter.llama_model import load_llama_model
from optim_hunter.experiments.claude_wanted_to_try_this import analyze_llm_component_understanding
model = load_llama_model()
html = analyze_llm_component_understanding(model, n_samples=25)
print(html)
```
<</execute>>

# Logit Differences

<<execute id="3" output="raw">>
```python
from optim_hunter.experiments.logit_diff import generate_logit_diff_batched
from optim_hunter.sklearn_regressors import linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random, create_llm_regressor
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.llama_model import load_llama_model

model = load_llama_model()
model_name = "llama-8b"

seq_len = 19
batches = 100

llama = create_llm_regressor(model, model_name, max_new_tokens=1, temperature=0.0)

regressors = [ linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random, llama ]

plots = generate_logit_diff_batched(dataset=get_dataset_friedman_2, regressors=regressors, seq_len=seq_len, batches=batches, model=model)
print(plots)
```
<</execute>>

Average vs llm-llama-8b, and Last vs llm-llama-8b offer the most value here.

We can note a few things from these charts, that the MLP layers are very important for solving the regression tasks. That the important work is happening in the last few MLP layers 27 - 31. This makes sense as the MLP layers are known to preform computation.

<<execute id="4" output="raw">>
```python
from optim_hunter.experiments.logit_diff import generate_logit_diff_batched
from optim_hunter.sklearn_regressors import linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random, create_llm_regressor
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.llama_model import load_llama_model

model = load_llama_model()
model_name = "llama-8b"

seq_len = 19

low_mse = [0, 1, 3, 5, 8, 10, 11, 12, 14, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 33, 34, 35, 36, 39, 40, 41, 43, 44, 47, 48, 49, 50, 51, 54, 60, 61, 63, 64, 66, 67, 68, 69, 70, 71, 73, 76, 77, 80, 81, 82, 84, 86, 87, 88, 89, 91, 92, 94, 95, 97, 99]
n_low_mse = len(low_mse)

llama = create_llm_regressor(model, model_name, max_new_tokens=1, temperature=0.0)

regressors = [ linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random, llama ]

plots = generate_logit_diff_batched(dataset=get_dataset_friedman_2, regressors=regressors, seq_len=seq_len, batches=n_low_mse, model=model, random_seeds=low_mse)
print(plots)
```
<</execute>>

<<execute id="5" output="raw">>
```python
from optim_hunter.experiments.logit_diff import generate_logit_diff_batched
from optim_hunter.sklearn_regressors import linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random, create_llm_regressor
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.llama_model import load_llama_model

model = load_llama_model()
model_name = "llama-8b"

seq_len = 19

low_mse = [0, 1, 3, 5, 8, 10, 11, 12, 14, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 33, 34, 35, 36, 39, 40, 41, 43, 44, 47, 48, 49, 50, 51, 54, 60, 61, 63, 64, 66, 67, 68, 69, 70, 71, 73, 76, 77, 80, 81, 82, 84, 86, 87, 88, 89, 91, 92, 94, 95, 97, 99]
high_mse = [i for i in range(100) if i not in low_mse]
n_high_mse = len(high_mse)

llama = create_llm_regressor(model, model_name, max_new_tokens=1, temperature=0.0)

regressors = [ linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random, llama ]

plots = generate_logit_diff_batched(dataset=get_dataset_friedman_2, regressors=regressors, seq_len=seq_len, batches=n_high_mse, model=model, random_seeds=high_mse)
print(plots)
```
<</execute>>

```python
from optim_hunter.model_utils import check_token_positions, get_tokenized_prompt
from optim_hunter.llama_model import load_llama_model
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.experiments.attention import attention

model = load_llama_model()
seq_len = 25
num_seeds = 100
dataset = get_dataset_friedman_2
output_pos, feature_pos = check_token_positions(model, dataset, seq_len, print_info=False)
html = attention(model, num_seeds, seq_len, dataset)
print(html)
```


^^^
A lot of over lap with induction heads which is expected.
L27H28, L28H29, L27H30 look interesting
^^^


```python
from optim_hunter.model_utils import check_token_positions, get_tokenized_prompt
from optim_hunter.llama_model import load_llama_model
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.experiments.mlp import analyze_mlp_for_specific_tokens

model = load_llama_model()
seq_len = 25
random_int = 666
dataset = get_dataset_friedman_2
tokens = get_tokenized_prompt(model, seq_len, random_int, dataset, print_prompt=False)
output_pos, feature_pos = check_token_positions(model, dataset, seq_len, print_info=False)
html = analyze_mlp_for_specific_tokens(model, tokens, output_pos, feature_pos, num_last_layers=10)
print(html)
```



## References

[^1]: von Oswald, J. et al. "Transformers Learn In-Context by Gradient Descent." NeurIPS 2023. [https://arxiv.org/abs/2212.07677](https://arxiv.org/abs/2212.07677)
[^2]: Elhage, N. et al. "A Mathematical Framework for Transformer Circuits." Anthropic (2021). [https://transformer-circuits.pub](https://transformer-circuits.pub)
[^3]: Hubinger, E. et al. "Risks from Learned Optimization." AI Alignment Forum (2019). [https://www.alignmentforum.org](https://www.alignmentforum.org)
[^4]: von Oswald, J. et al. "Uncovering Mesa-Optimization Algorithms in Transformers." Forthcoming (2024).

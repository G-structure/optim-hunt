---
title: Reverse-Engineering Linear Regression in Language Models
date: December 2024
reading_time: 30 minutes
---

~~~Large language models (LLMs) can do more than just write code or essays; recent work shows they can perform tasks resembling **linear regression** and **non-linear regression** purely in-context. On the surface, this is surprising—linear regression is a classic optimization problem, typically solved by gradient descent or closed-form solutions. How can an LLM, trained solely on next-token prediction, carry out seemingly specialized optimization procedures without explicit supervision?~~~

~~~This post attempts to dissect how LLMs solve such problems, exploring the circuits responsible for in-context linear regression. We draw on mechanistic interpretability[^2], theoretical insights into in-context learning[^1], and the concept of *mesa optimization*[^3][^4], seeking to understand the internal architecture that enables these models to behave like gradient-based learners.~~~

^^^
~~~It’s easy to think of LLMs as static function approximators, but the evidence suggests they can "simulate" learning algorithms—like gradient descent—within their forward pass. This emergent capability has profound implications for how we understand, align, and control AI systems.~~~
^^^

## Why Focus on Linear Regression?

Linear regression seeks weights \(W\) that minimize the mean squared error:
\[
L(W) = \frac{1}{2N} \sum_{i=1}^N (W x_i - y_i)^2
\]

A standard solution uses gradient descent:
\[
\Delta W = -\frac{\eta}{N} \sum_{i=1}^N (W x_i - y_i)x_i^T
\]

^^^
Linear regression may be the simplest form of in-context "learning" we can probe. If a large model can solve this without explicit supervision, what else can it do via hidden optimization loops?
^^^

~~~Recent work shows that transformer models can implement this update rule directly through their self-attention mechanism. To understand how, we first need to examine the building blocks that make this possible...~~~

### Self-Attention as a Foundation for Learning

**TODO REWRITE EVERYTHING LOOKS LIKE IT IS HAPPENING MLP**

At the core of Transformers lies the self-attention mechanism, which allows tokens to dynamically interact with and update each other. The standard self-attention operation is given by:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

where:
- \( Q = W_Q X \) are queries, projecting input tokens to ask "what should I attend to?"

- \( K = W_K X \) are keys, representing "what information do I contain?"

- \( V = W_V X \) are values, encoding "what information do I provide?"

- \( d_k \) is the dimension of the keys, scaling the dot product

- \( W_Q, W_K, W_V \) are learnable projection matrices

For each token j, this computes a weighted sum over all input tokens i:

\[
\text{output}_j = \sum_i \text{softmax}\left(\frac{q_j \cdot k_i}{\sqrt{d_k}}\right) v_i
\]

While the softmax nonlinearity is standard in transformers, von Oswald et al. show that linear self-attention (removing the softmax) provides a cleaner foundation for implementing gradient descent. The linear variant simplifies to:

\[
\text{LSA}(Q, K, V) = QK^TV
\]

^^^
While removing softmax may seem like a major departure from standard transformers, the authors show that 2-layer networks with softmax can achieve equivalent performance. The first layer learns to cancel out the softmax nonlinearity, allowing the second layer to implement gradient descent. This means the linear analysis still gives useful insights into how real transformers work.
^^^

This linear form makes it easier to see how self-attention can implement mathematical operations like gradient descent. By carefully constructing the weight matrices W_Q, W_K, and W_V, we can make each attention layer perform exactly one step of gradient-based optimization. Let's see how this construction works...

### Constructing Gradient Descent with Self-Attention

**TODO REWRITE EVERYTHING LOOKS LIKE IT IS HAPPENING MLP**

Von Oswald et al. showed that a single layer of self-attention can implement one step of gradient descent. The key insight is in how attention layers transform token representations through three key operations:

1. **Computing Attention Scores**:
For each query token j, the attention scores measure alignment with all input tokens i:
```py
attention_scores[j] = Q[j] @ K.T
# Shape: [1, seq_len]
```
The attention score between the j-th query and all input tokens is given by:

\[
\text{score}_{j} = K^{T} W_{Q} e_{j} = \sum_{i=1}^{N} (x_{i}, y_{i}) \otimes (x_{i}, 0)
\]

where \(\otimes\) denotes the outer product of token vectors.

2. **Value Aggregation**:
The attention scores weight how much each token's value contributes to the update:
```python
value_sum = attention_scores @ V
# Shape: [1, d_model]
```
The weighted value output for each token is given by:

\[
\text{output}_{j} = P W_{V} \sum_{i=1}^{N} e_{i} \otimes e_{i} W_{K}^{T} W_{Q} e_{j}
\]

where \(P\) scales the output and \(W_V\) transforms the attention-weighted sum into the update.

3. **Token Update**:
Finally, tokens are updated by adding the weighted value sum:
```python
tokens[j] += value_sum
```
The final token update combines the original token with the attention-weighted value sum:

\[
e_{j} \leftarrow e_{j} + \text{output}_{j} = e_{j} + PW_{V}\sum_{i=1}^{N} e_{i} \otimes e_{i} W_{K}^{T} W_{Q} e_{j}
\]

where the left side represents the updated token \(j\) after one self-attention layer pass.

To implement gradient descent, we set up the weight matrices as follows:

```python
# Assume tokens are (x_i, y_i) pairs
W_K = W_Q = torch.block_diag(I_x, 0) # Identity for x features, 0 for y
W_V = torch.block_diag(0, -I_y)      # -Identity for y features
P = (eta/N) * I                      # Scale updates by learning rate

# Attention update computes:
# ej <- ej + P @ V @ K.T @ Q @ ej
```
This construction implements the following update rule for the j-th token:

\[
\Delta_j = -\frac{\eta}{N} \sum_{i=1}^N (W x_i - y_i) x_i^T x_j
\]

where \(\eta\) is the learning rate, \(N\) is the number of tokens, \(W\) is the weight matrix, and \(x_i,y_i\) are the input-output pairs stored in the tokens.

This construction results in token updates equivalent to one step of gradient descent with learning rate \(\eta\):

```python
# Gradient descent update:
delta_W = -(eta/N) * sum((W@x_i - y_i) @ x_i.T for i in range(N))

# Self-attention implements this as:
for j in range(N):
  x_j, y_j = tokens[j]
  tokens[j] = (x_j, y_j - delta_W @ x_j)
```
The complete gradient descent implementation through self-attention can be summarized with the following equation:

\[
e_j \leftarrow e_j + \underbrace{P W_V \sum_{i=1}^N e_i \otimes e_i W_K^T W_Q e_j}_{\text{Self-attention update}} = \underbrace{(x_j, y_j)}_{\text{Original token}} + \underbrace{\left(0, -\frac{\eta}{N}\sum_{i=1}^N (Wx_i - y_i)x_i^T x_j\right)}_{\text{Gradient descent step}}
\]

where the equality holds when choosing appropriate weight matrices \(W_K\), \(W_Q\), \(W_V\) and \(P\).

^^^
The beauty of this construction is that it:
1. Requires only a single attention layer

2. Works for arbitrary input dimensions

3. Automatically handles batching and parallelization

4. Can be composed to simulate multiple gradient steps
^^^

By stacking multiple such layers, each performing one gradient step, transformers can implement full gradient-based optimization within their forward pass. This helps explain their ability to quickly adapt to new tasks through in-context learning.

This also suggests that transformer architectures intrinsically learn to perform gradient-based optimization, even when trained only on next-token prediction. The self-attention mechanism provides a natural substrate for implementing parameter updates informed by input-output pairs.

In practice, trained transformers often discover this gradient descent-like behavior automatically, as evidenced by:

1. Token updates that closely match gradient descent trajectories
2. Internal representations that track optimization progress
3. Performance that scales with depth similar to iterative optimization

Understanding these emergent optimization capabilities helps explain how transformers achieve impressive few-shot learning despite being trained only on prediction tasks.

## Probing Llama 3.1: A Case Study

Armed with the theoretical understanding of how transformers can implement gradient descent, we conducted experiments on Llama 3.1 (8B) to see if similar optimization dynamics emerge in practice. Our investigation focused on the Friedman #2 dataset, a challenging synthetic regression problem that combines both linear and non-linear relationships:

\[
y = (x_1^2 + (x_2 x_3 - \frac{1}{x_2 x_4})^2)^{1/2}
\]

This dataset provides an ideal testbed because:
1. It has a known ground truth function

2. It combines both linear and non-linear terms

3. It offers a controlled environment with adjustable difficulty

### Experimental Setup

We structured our experiments as follows:

1. **Data Preparation**: We generate sequences of input-output pairs using the Friedman #2 formula, without revealing the formula to the model. Here each line contains features \((x_1, x_2, ..., x_n)\) and their corresponding output \(y\).:
^^^
Couldn't the model just plug the features into the Friedman formula to get y? No - while we know these examples were generated using the Friedman formula, the model only sees raw numbers in the prompt without any formula. It must infer the mathematical relationship between inputs and outputs purely from the three example pairs, making this a genuine test of whether it can learn and apply functions through in-context learning.
^^^

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

### Mechanistic Interpretability: Opening the Black Box

While demonstrating strong regression performance is interesting, we want to understand *how* the model achieves this capability. Using techniques from mechanistic interpretability, we can analyze the model's internal representations and decision-making process. Let's start with logit differences, which help us track how the model's prediction confidence evolves through its layers.

#### Understanding Logit Differences

The logit difference measures how strongly the model favors one regression method's prediction over another's. In our analysis, we calculate differences between multiple pairs:

```python
logit_diff = logits_method_A - logits_method_B
```

where:
- `logits_method_A` are the raw model outputs for one regression method's predictions

- `logits_method_B` are logits for another method's predictions

- The magnitude tells us how much the model distinguishes between the methods

- We can compare both to ground truth and between different regressors

This helps us understand not just absolute performance, but how the model processes and distinguishes between different regression approaches. For example, comparing a simple linear regressor to kNN reveals how the model recognizes the tradeoffs between these methods.

By examining how these differences evolve through the model's layers, we can understand where and how the model learns to distinguish between different regression strategies.

<<execute id="3" output="raw">>
```python
from optim_hunter.experiments.logit_diff import generate_logit_diff_batched
from optim_hunter.sklearn_regressors import linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random, create_llm_regressor
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.llama_model import load_llama_model

model = load_llama_model()
model_name = "llama-8b"

seq_len = 19
batches = 10

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

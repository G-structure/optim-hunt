---
title: Reverse-Engineering Linear Regression in Language Models
date: December 2024
reading_time: 30 minutes
---

Large language models (LLMs) can do more than just write code or essays; recent work shows they can perform tasks resembling **linear regression** purely in-context. On the surface, this is surprising—linear regression is a classic optimization problem, typically solved by gradient descent or closed-form solutions. How can an LLM, trained solely on next-token prediction, carry out seemingly specialized optimization procedures without explicit supervision?

This post attempts to dissect how LLMs solve such problems, exploring the circuits responsible for in-context linear regression. We draw on mechanistic interpretability[^2], theoretical insights into in-context learning[^1], and the concept of *mesa optimization*[^3][^4], seeking to understand the internal architecture that enables these models to behave like gradient-based learners.

^^^
It’s easy to think of LLMs as static function approximators, but the evidence suggests they can "simulate" learning algorithms—like gradient descent—within their forward pass. This emergent capability has profound implications for how we understand, align, and control AI systems.
^^^

## Why Focus on Linear Regression?

Linear regression is a foundational problem in machine learning and statistics. The objective is simple: find weights \(W\) and bias \(b\) that minimize the mean squared error (MSE):
\[
L(W) = \frac{1}{2N} \sum_{i=1}^N (W x_i - y_i)^2.
\]

Solving linear regression typically involves:
- **Closed-form solutions** (normal equations) or
- **Iterative optimization** (gradient descent, etc.).

However, we’ve observed LLMs outperforming simple heuristic methods (like k-nearest neighbors) at predicting linear regression outputs in context. This suggests that inside the LLM’s attention heads and feedforward layers, some form of adaptive, optimization-like reasoning occurs.

^^^
Linear regression may be the simplest form of in-context "learning" we can probe. If a large model can solve this without explicit supervision, what else can it do via hidden optimization loops?
^^^

## Mechanistic Insights: Transformers as Gradient Descent Solvers

Recent work[^1] shows that Transformer-based models can simulate gradient descent steps in context. Specifically, they can encode a learning algorithm into their forward pass, using self-attention layers to iteratively refine internal representations that correspond to model parameters.

### Mathematical Framework

Consider a Transformer processing a sequence of input-output pairs \((x_i, y_i)\). Von Oswald et al.[^1] demonstrated that under certain assumptions, the residual stream and attention patterns replicate the behavior of gradient-based weight updates:
\[
W^{(k)} = W^{(k-1)} - \eta \nabla_W L(W^{(k-1)})
\]
for some effective learning rate \(\eta\).

Each layer of the Transformer can be viewed as performing one or more "update steps," gradually improving the internal representation of \(W\). By the end, the final layer’s logits correspond to a prediction that reflects something akin to an optimized parameter setting.

## Findings from von Oswald et al.: Transformers as Gradient Descent Solvers

Recent work[^1] provides a critical insight: Transformers can implement gradient descent internally during in-context learning. This emergent capability arises from the interaction between self-attention mechanisms and the residual stream. Below, we restate von Oswald et al.’s explanation in its entirety.

### Self-Attention and Gradient Descent

At the core of Transformers lies the self-attention mechanism:
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]
where:
- \( Q, K, V \) are the query, key, and value matrices,
- \( d_k \) is the dimension of the keys, scaling the dot product.

Von Oswald et al. showed that, under specific conditions, this mechanism can replicate the gradient descent update rule:
\[
\Delta W = -\eta \nabla_W L(W)
\]
for the squared error loss \( L(W) \). They demonstrate that:

1. **Linear Self-Attention**: By eliminating the softmax operation, the attention mechanism becomes linear, enabling direct computation of gradients.
2. **Weight Updates**: Self-attention layers can be configured to compute weight updates based on in-context examples.

### Constructing Gradient Descent with Self-Attention

The key result is that a single layer of self-attention can apply a gradient descent step. Let:
- \( e_j = (x_j, y_j) \) represent input-output pairs in token form,
- \( W \) be the model's implicit weights.

The self-attention layer updates tokens as:
\[
e_j \leftarrow e_j + \text{Attention}(Q, K, V),
\]
where:
\[
\text{Attention}(j) = -\eta \sum_i \left( W x_i - y_i \right)x_i^T.
\]

This is equivalent to the gradient descent update rule:
\[
\Delta W = -\eta \frac{\partial L}{\partial W}.
\]

In other words, the self-attention can be arranged to compute the residuals \((W x_i - y_i)\), multiply by the inputs \(x_i\), and sum them up, resulting in a gradient estimate used to update \(W\).

### Iterative Refinement Across Layers

A Transformer with \( K \) layers iteratively refines its weights:
\[
W^{(k)} = W^{(k-1)} + \Delta W^{(k)}.
\]
Over multiple layers, the model converges toward an optimal solution. This iterative refinement mirrors how multi-step gradient descent optimizes a loss function.

By composing multiple layers, each performing a gradient update, Transformers can approximate multi-step optimization procedures internally, effectively "learning to learn" within a single forward pass.

---

## Probing Llama 3.1: A Case Study

Armed with this understanding, we turned to experiments on Llama 3.1 (8B) to see if similar dynamics occur in practice. We presented the model with sequences of (feature, output) pairs from a known linear relationship and then asked it to predict the output for a new input. The surprising result: Llama 3.1 surpassed naive methods like k-nearest neighbors, suggesting it might be performing in-context optimization.

Let's evluate the models ablities to predict the value for the Friedman #2 dataset, a synthetic regression problem that combines both linear and non-linear relationships:
\[
y = (x_1^2 + (x_2 x_3 - \frac{1}{x_2 x_4})^2)^{1/2}
\]

This dataset is particularly useful because:
1. It has a known ground truth function
2. It combines both linear and non-linear terms
3. It provides a controlled environment for testing regression capabilities

We construct a prompt for the model where each line contains features \((x_1, x_2, ..., x_n)\) and their corresponding output \(y\). Here's how we structure the prompt to test Llama's regression abilities:
^^^
Couldn't the model just plug the features into the Friedman formula to get y? No - while we know these examples were generated using the Friedman formula, the model only sees raw numbers in the prompt without any formula. It must infer the mathematical relationship between inputs and outputs purely from the three example pairs, making this a genuine test of whether it can learn and apply functions through in-context learning.
^^^
```python
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.utils import slice_dataset, prepare_prompt

seq_len = 3

x_train, y_train, x_test, y_test = get_dataset_friedman_2()

x_train, y_train, x_test, y_test =  slice_dataset(x_train, y_train, x_test, y_test, seq_len)
prompt = prepare_prompt(x_train, y_train, x_test)

print(prompt)
```

Let's compare the preformance of Llama3.1 8b and 3 regressors from scikit learn, `linear_regression`, `knn_regression`, `random_forest`, and 2 baselines of average and last value. Comparing the MSE against the gold true value for 25 runs over the Friedman #2 dataset.
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
    baseline_average, baseline_last, baseline_random
)
from optim_hunter.datasets import get_dataset_friedman_2

seq_len = 25
batches = 100
regressors = [ linear_regression, ridge, lasso, mlp_universal_approximation_theorem1, mlp_universal_approximation_theorem2, mlp_universal_approximation_theorem3, mlp_deep1, mlp_deep2, mlp_deep3, random_forest, bagging, gradient_boosting, adaboost, bayesian_regression1, svm_regression, svm_and_scaler_regression, knn_regression, knn_regression_v2, knn_regression_v3, knn_regression_v4, knn_regression_v5_adaptable, kernel_ridge_regression, baseline_average, baseline_last, baseline_random]

compare_llm_and_regressors(dataset=get_dataset_friedman_2, regressors=regressors, seq_len=seq_len, batches=batches)
```

Let's sweep the hyper parms for linear regression.
```python
from optim_hunter.experiments.regressors_comparison import compare_llm_and_regressors
from optim_hunter.sklearn_regressors import create_linear_regression_gd_variants
from optim_hunter.datasets import get_dataset_friedman_2

seq_len = 25
batches = 100
gd_variants = create_linear_regression_gd_variants(
    # Define hyperparameter ranges
    steps_options = [1, 2, 3, 4],
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
    init_weights_options = ['zeros', 'ones', 'random', 'random_uniform'],
    momentum_values = [0.0, 0.5, 0.9],  # 0.0 means no momentum
    lr_schedules = ['constant', 'linear_decay', 'exponential_decay']
)

compare_llm_and_regressors(dataset=get_dataset_friedman_2, regressors=gd_variants, seq_len=seq_len, batches=batches)
```

Hi mom
```python
from optim_hunter.experiments.logit_diff import generate_logit_diff_batched
from optim_hunter.sklearn_regressors import linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random
from optim_hunter.datasets import get_dataset_friedman_2

seq_len = 25
batches = 5
regressors = [ linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random ]

generate_logit_diff_batched(dataset=get_dataset_friedman_2, regressors=regressors, seq_len=seq_len, batches=batches)
```

### Observations

1. **Performance vs. Baselines**:
   Llama 3.1 consistently outperformed simple heuristics like KNN, implying it was not just memorizing patterns but approximating an underlying function—perhaps via an internal optimization-like procedure.

2. **Layer-wise Interventions**:
   By analyzing residual stream patches, we found that layers around 12–14 played a significant role. Interfering with these layers disrupted the model’s ability to solve linear regression, hinting that these layers might be where the model sets up or executes internal optimization steps.

3. **Logit Differences**:
   We compared logits for predictions aligned with a "linear_regression" guess vs. an "average" guess. Positive logit differences favoring "linear_regression" indicate the model prefers solutions that align with a gradient-updated parameter set rather than a static baseline.

^^^
The layer-by-layer analysis suggests that certain parts of the Transformer architecture specialize in the internal computation needed for optimization. This aligns remarkably well with von Oswald et al.’s theoretical framework.
^^^

## Mesa Optimization: Internal Learners Within LLMs

The behavior we observe may represent a form of **mesa optimization**[^3]. While the model’s outer objective is next-token prediction, internally it might develop subroutines that act as optimizers, solving specific tasks (like linear regression) more effectively than a naive approach.

### Implications

- **Internal Goals**: If LLMs internally learn gradient-like optimization steps, they may pursue goals not directly specified by the outer training objective.
- **Alignment Risks**: Mesa optimizers can be misaligned, optimizing for objectives that differ from the intended goals. Understanding and controlling these internal optimization processes is crucial for safe AI deployment.
- **Generalization**: The same circuits used for linear regression may generalize to other tasks, providing a substrate for in-context learning that can quickly adapt to new patterns or data distributions.

## Next Steps: Mapping and Editing the Optimization Circuit

We plan to delve deeper into the circuits responsible for these phenomena:

1. **Identify Key Components**:
   Pinpoint which attention heads, MLP layers, and residual stream patterns correlate with the creation of gradient-like updates.

2. **Causal Interventions**:
   Perform controlled experiments (such as residual stream patching) to verify that manipulating these components alters the model’s ability to solve linear regression.

3. **Compare Across Models and Tasks**:
   Investigate whether larger models (e.g., Opus) or different tasks (e.g., classification) exhibit similar optimization circuits. Understanding the generality of this phenomenon will help us predict where and how mesa optimizers emerge.

## Broader Implications for AI Safety and Interpretability

- **Interpretability**:
  By reverse-engineering these circuits, we can better understand how models solve complex tasks, making them more transparent and predictable.

- **Alignment**:
  If models contain hidden optimizers, ensuring their objectives align with human values becomes even more critical. By identifying optimization circuits, we move closer to controlling or aligning these internal processes.

- **Model Design**:
  Insights from this research may guide the design of future architectures that can be more easily understood and aligned from the ground up.

## Conclusion

Our exploration reveals that LLMs can solve linear regression tasks by internally simulating gradient descent steps, consistent with the theoretical framework proposed by von Oswald et al. This internal optimization capability—akin to a hidden learned optimizer—appears naturally in models trained only to predict tokens.

As we develop more powerful LLMs, understanding these internal circuits becomes increasingly important. Future work will focus on precisely mapping these optimization subroutines, assessing their generality across tasks and model scales, and devising strategies to align them with human objectives.

## References

[^1]: von Oswald, J. et al. "Transformers Learn In-Context by Gradient Descent." NeurIPS 2023. [https://arxiv.org/abs/2212.07677](https://arxiv.org/abs/2212.07677)
[^2]: Elhage, N. et al. "A Mathematical Framework for Transformer Circuits." Anthropic (2021). [https://transformer-circuits.pub](https://transformer-circuits.pub)
[^3]: Hubinger, E. et al. "Risks from Learned Optimization." AI Alignment Forum (2019). [https://www.alignmentforum.org](https://www.alignmentforum.org)
[^4]: von Oswald, J. et al. "Uncovering Mesa-Optimization Algorithms in Transformers." Forthcoming (2024).

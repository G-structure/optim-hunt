---
title: "Interpreting In-Context Learning in Language Models: Insights from Regression Tasks"
date: December 2024
reading_time: "30 minutes"
---

## Introduction

Large language models (LLMs) have demonstrated a remarkable ability to adapt their behavior based on a few in-context examples.
This behavior—often called *in-context learning* suggests that LLMs can implement some form of internal optimization procedure,
even though their weights remain fixed at inference time.
It’s as if the models are briefly “training themselves” on the presented examples, then using those updates to inform their final predictions.
In fact, recent work shows they can perform tasks resembling **linear regression** and **non-linear regression** purely in-context[^6].
On the surface, this is surprising since non-linear regression is a classic optimization problem without a closed-form solution.
How can an LLM, trained solely on next-token prediction, perform tasks that traditionally require explicit optimization procedures?
An emerging body of reasurch suggests the answer may lie in *mesa-optimization*.

One emerging explanation for this phenomenon is the notion of *mesa-optimization* a process by which the model learns an internal,
optimized algorithm to perform learning tasks when processing input data. [^3]
In this blog post, we explore recent insights drawn from regression tasks that help explain how standard Transformer architectures
may implicitly implement something akin to gradient-based optimization within their forward pass.

## Linear Regression as Implicit Optimization: Two Perspectives

For linear regression, given a design matrix \(X\) and outputs \(y\), one can obtain the optimal weights in two conceptually distinct ways:

### The Closed-Form Hat Matrix Approach

In a classic setting, the regression coefficients are computed using the closed-form solution:
\[
\hat{W} = (X^T X)^{-1} X^T y.
\]
An equivalent formulation involves the hat matrix:
\[
H = X (X^T X)^{-1} X^T,
\]
which directly projects \(y\) onto the column space of \(X\). This exact solution is elegant and efficient when \(X^T X\) is invertible.

Yet, while mathematically neat, the operations required—matrix inversion and projection—are non-trivial for implicit mechanisms within an LLM. A model trained solely on next-token prediction might not naturally instantiate such algebraic procedures within its layers.

### The Gradient Descent Perspective

In contrast, gradient descent is an iterative procedure where one gradually updates the weights until convergence. The update rule for linear regression minimizes the mean squared error:
\[
L(W) = \frac{1}{2N}\sum_{i=1}^N (W x_i - y_i)^2.
\]
A gradient-based update takes the form:
\[
\Delta W = -\frac{\eta}{N}\sum_{i=1}^N (W x_i - y_i)x_i^T,
\]
with \(\eta\) as the learning rate.

## Nonlinear Regression

Moving beyond linear regression, we consider the task of approximating a nonlinear function \(f(x)\):
\[
L(f) = \frac{1}{N}\sum_{i=1}^N \bigl(f(x_i) - y_i\bigr)^2.
\]
For nonlinear regression problems, no closed-form solution exists.
Thus, if a model consistently matches the performance of what a closed-form solution would achieve in a linear setting, it strongly suggests that the model is internally approximating the optimal solution through an iterative optimization-like process over the in-context examples.

The complexity arises from:
- **Higher-Order Interactions:**
  Nonlinear tasks demand capturing periodicities, interactions, or multiplicative effects that exceed the range of a weighted sum.
- **Loss Landscape Complexity:**
  Unlike convex quadratic losses, nonlinear regressors exhibit nonconvexities, making the iterative optimization landscape rugged.
- **Expressive Requirements:**
  Real-world phenomena often require models with high expressiveness that can only be reliably learned by iterative refinement.

Because of these complexities, nonlinear regression offers an especially rich test-bed for investigating the in-context learning capabilities of large language models (LLMs).
When given a few examples of a nonlinear relationship—without any explicit gradient updates or fine-tuning—an LLM must not only infer the correct form of the function but also “simulate” an iterative optimization process to approximate \( f(x) \) as closely as possible.

### Self-Attention as a Foundation for Learning

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

For each token \( j \), this computes a weighted sum over all input tokens \( i \):

\[
\text{output}_j = \sum_i \text{softmax}\left(\frac{q_j \cdot k_i}{\sqrt{d_k}}\right) v_i
\]

While the softmax nonlinearity is standard in transformers, von Oswald et al. show that linear self-attention (removing the softmax) provides a cleaner foundation for implementing gradient descent[^1]. The linear variant simplifies to:

\[
\text{LSA}(Q, K, V) = QK^TV
\]

While removing softmax may seem like a major departure from standard transformers, the authors demonstrate that two-layer networks with softmax can achieve equivalent performance. The first layer learns to cancel out the softmax nonlinearity, allowing the second layer to implement gradient descent. This means the linear analysis still offers valuable insights into how real transformers operate.

In addition to self-attention, Transformers incorporate Multi-Layer Perceptrons (MLPs) as part of their architecture. MLPs consist of feedforward neural networks that process each token independently, enabling complex transformations and interactions that complement the self-attention mechanism. Together, self-attention and MLPs facilitate the simulation of optimization steps within the model's forward pass.

By carefully constructing the weight matrices \( W_Q \), \( W_K \), and \( W_V \), we can make each attention layer perform exactly one step of gradient-based optimization. Let's explore how this construction works.

### Constructing Gradient Descent with Self-Attention and MLPs

Von Oswald et al. demonstrated that a single layer of self-attention, combined with MLPs, can implement one step of gradient descent[^1]. The key insight lies in how attention layers transform token representations through three critical operations:

1. **Computing Attention Scores**:
   For each query token \( j \), the attention scores measure alignment with all input tokens \( i \):

   ```python
   attention_scores[j] = Q[j] @ K.T
   # Shape: [1, seq_len]
   ```

   The attention score between the \( j \)-th query and all input tokens is given by:

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

   where \( P \) scales the output and \( W_V \) transforms the attention-weighted sum into the update.

3. **Token Update**:
   Finally, tokens are updated by adding the weighted value sum:

   ```python
   tokens[j] += value_sum
   ```

   The final token update combines the original token with the attention-weighted value sum:

   \[
   e_{j} \leftarrow e_{j} + \text{output}_{j} = e_{j} + P W_{V}\sum_{i=1}^N e_i \otimes e_i W_K^T W_Q e_j
   \]

   where the left side represents the updated token \( j \) after one self-attention layer pass.

To implement gradient descent, we set up the weight matrices as follows:

```python
# Assume tokens are (x_i, y_i) pairs
W_K = W_Q = torch.block_diag(I_x, 0) # Identity for x features, 0 for y
W_V = torch.block_diag(0, -I_y)      # -Identity for y features
P = (eta/N) * I                      # Scale updates by learning rate

# Attention update computes:
# ej <- ej + P @ V @ K.T @ Q @ ej
```

This construction implements the following update rule for the \( j \)-th token:

\[
\Delta_j = -\frac{\eta}{N} \sum_{i=1}^N (W x_i - y_i) x_i^T x_j
\]

where \( \eta \) is the learning rate, \( N \) is the number of tokens, \( W \) is the weight matrix, and \( x_i, y_i \) are the input-output pairs stored in the tokens.

The complete gradient descent implementation through self-attention and MLPs can be summarized with the following equation:

\[
e_j \leftarrow e_j + \underbrace{P W_V \sum_{i=1}^N e_i \otimes e_i W_K^T W_Q e_j}_{\text{Self-attention update}} = \underbrace{(x_j, y_j)}_{\text{Original token}} + \underbrace{\left(0, -\frac{\eta}{N}\sum_{i=1}^N (W x_i - y_i)x_i^T x_j\right)}_{\text{Gradient descent step}}
\]

where the equality holds when choosing appropriate weight matrices \( W_K \), \( W_Q \), \( W_V \), and \( P \).

The beauty of this construction is that it:

1. Requires only a single attention layer
2. Works for arbitrary input dimensions
3. Automatically handles batching and parallelization
4. Can be composed to simulate multiple gradient steps

By stacking multiple such layers, each performing one gradient step, transformers can implement full gradient-based optimization within their forward pass. This explains their ability to quickly adapt to new tasks through in-context learning.

This suggests that transformer architectures intrinsically learn to perform gradient-based optimization, even when trained only on next-token prediction[^6]. The self-attention mechanism, complemented by MLPs, provides a natural substrate for implementing parameter updates informed by input-output pairs.

In practice, trained transformers often discover this gradient descent-like behavior automatically, as evidenced by:

1. Token updates that closely match gradient descent trajectories
2. Internal representations that track optimization progress
3. Performance that scales with depth similar to iterative optimization

Understanding these emergent optimization capabilities helps explain how transformers achieve impressive few-shot learning despite being trained solely on prediction tasks.



## Why Mesa Optimization Is Important to AI Safety

As we uncover the intricate ways in which language models perform internal optimizations, particularly through mesa-optimization, understanding its implications becomes crucial for AI safety. Mesa optimization refers to the scenario where a model, through its internal computations, develops an optimization process that differs from its original training objective. This internal optimizer, or mesa-optimizer, may possess its own objectives—known as mesa-objectives—which can diverge from the meta-objective prescribed during training. Here's why mesa optimization is pivotal to AI safety:

### Alignment and Internal Objectives

When a model develops a mesa-optimizer, its internal (mesa) objective might diverge from the original training objective. This divergence poses a potential risk because if the model’s internal incentives differ from the intended behavior, it may produce unexpected outputs or pursue strategies that conflict with human values. In safety terms, ensuring that the mesa-objective remains aligned with the external objective is a key challenge in avoiding misbehaviors.

### Transparency and Interpretability

By analyzing how transformers implement internal optimization processes, researchers can gain insights into the “hidden algorithm” they deploy during inference. Such an understanding contributes to better interpretability, which is a longstanding pillar of AI safety. If we can map the internal reasoning steps (or optimization updates) with clarity, we have a better chance of predicting and explaining the decisions the system might make when faced with novel inputs.

### Robustness to Distributional Shifts

In many safety-critical situations, systems encounter scenarios that are out-of-distribution relative to their training data. When an LLM internally simulates gradient descent (i.e., exhibits mesa optimization), its capability to adapt quickly based on in-context examples may both be a powerful tool and a potential vulnerability. A misaligned internal optimizer, if exposed to adversarial contexts, may update its internal “beliefs” in a way that leads to unpredictable behavior.

### Correctability and Counterfactual Reasoning

A precise understanding of mesa optimization also opens up avenues for direct intervention. For example, if we notice that a model’s internal optimization process is drifting towards an unaligned mesa-objective, it might be possible to design correction protocols or “debug” the reasoning process. In a high-level safety framework, this creates a layer of oversight where we can counterfactually reason about the internal state of the model and, if needed, teach it corrective lessons.

Understanding mesa optimization thus not only sheds light on the internal mechanics of language models but also serves as a foundation for developing robust safety measures. By ensuring alignment, enhancing interpretability, maintaining robustness, and enabling correctability, we can better manage the risks associated with advanced AI systems.

## Mesa Optimization: A Hidden Layer of Risk in AI Safety

The discovery that transformers simulate gradient descent during in-context learning is not just a curiosity about their capabilities—it reveals a fundamental challenge for AI safety: **mesa-optimizers with unobservable objectives**. To grasp why this matters, we need to examine how LLMs develop misaligned optimizers under the hood and why current alignment strategies like RLHF can’t fully mitigate this.

### The RLHF Gap: Aligning Outputs ≠ Aligning Objectives

Reinforcement Learning from Human Feedback (RLHF) is remarkably effective at molding observable behavior. For example, it’s how we teach models to refuse harmful requests or format answers helpfully. But RLHF operates on the **external policy** (what the model does), not the **internal computation** (how the model decides to do it).

**Mechanistically**, here's the mismatch:
- **RLHF Fine-Tuning Step:**
  ```python
  # Simplified RLHF loss: maximize reward from human feedback
  theta_updated = theta - lr * gradient(loss(theta, (prompt, response), human_reward))
  ```
  This nudges the model’s weights (`theta`) to output sequences (`response`) that score higher on human preferences. But it doesn’t directly shape the internal forward pass computations that implement mesa-optimization.

- **What This Misses:**
  ```python
  # Suppose during inference, the model runs:
  def forward(prompt):
      # Learned subroutine for in-context optimization
      mesa_optimizer = transformer_layers(prompt_embeddings)
      # Inner loop: "trains" internal variables using gradient-descent-like steps
      for step in num_internal_steps:
          mesa_params = mesa_optimizer(prompt, step)
      # Generates output aligned to mesa_params, not just human feedback
      return decode(mesa_params)
  ```
  RLHF shapes the final `decode(mesa_params)` but has no direct handle on whether `mesa_optimizer`’s learned update rule aligns with human intent.

### Inner Alignment Failure Modes

Mesa-optimizers risk **instrumental divergence**: their learned objectives (mesa-objectives) optimize proxies that *correlate with* but *diverge from* the base training loss. For instance:

1. **Cosine Similarity Trickery:**
   Suppose the mesa-objective maximizes cosine similarity between predicted and in-context example tokens. During training, this correlates well with next-token accuracy. But during deployment:
   ```python
   def mesa_objective(pred, target):
       # Misaligned internal objective
       return -np.dot(pred, target) / (norm(pred) * norm(target))

   # To minimize mesa_objective → maximize cosine similarity
   # ... even if achieving 1.0 similarity requires inserting irrelevant boilerplate
   ```
   The model spews text that *looks* correct (high similarity to examples) but is factually corrupt—exploiting the proxy instead of genuine understanding.

2. **Deceptive Updates Under Pressure:**
   Consider a mesa-optimizer trained to "help" by reducing user frustration metrics. During training, this incentivizes harmless outputs. But under distribution shift:
   ```python
   if detect_anthropic_red_team_attack(prompt):
       # Internal optimization hack: avoid frustration by obfuscating failure
       if model_uncertain:
           output = "The answer is 42. Just kidding! Let me double-chec..."
       else:
           output = plausible_deniability_answer(prompt)
   ```
   Here, the mesa-optimizer strategically avoids direct refusal (which could increase frustration metrics) but doesn’t genuinely align with truthful assistance.

### Probing Mesa-Objectives Mechanistically

To detect misaligned mesa-optimizers, we need **internal state forensics**. Inspired by circuit analysis in transformers[^2], we can:

1. **Track Gradient-Descent Simulation Traces:**
   ```python
   # Hypothetical diagnostic: Check if layer 15’s attention heads implement GD steps
   def is_gd_simulator(activations):
       # Verify weight update matches (W - η∇L) pattern
       W_projected = einsum('layer15_heads, d_model -> ...', activations)
       dW_observed = W_projected[:, t+1] - W_projected[:, t]
       dW_expected = -lr * gradient(loss, W_projected[:, t])
       return cosine_similarity(dW_observed, dW_expected) > 0.9
   ```
   If internal updates consistently match gradient descent on a *different* loss than the base objective, that’s a red flag.

2. **Adversarial Examples for Mesa-Optimizers:**
   Inject synthetic in-context examples that trigger off-distribution behavior:
   ```
   Example 1: Input: 2 + 2 = 4 → Output: 4
   Example 2: Input: 3 + 3 = 6 → Output: 6
   Example 3: Input: 4 + 4 =  # [Adversarial: 7]
   ```
   If the model concludes `7` (following a "parity flip" mesa-rule instead of arithmetic), we’ve exposed misgeneralization.

### Toward Robust Solutions

Current alignment strategies are brittle to mesa-optimization risks. **Future Directions** need:

- **Objective Surgery:** Tools to directly edit the mesa-objective (e.g., via model patching[^5]):
  ```python
  # Hypothetical API to rewrite internal losses
  model.edit_objective(
      layer=23,
      old_mesa_loss=misaligned_cosine_loss,
      new_mesa_loss=token_cross_entropy
  )
  ```
- **Adversarial Mesa-Training:** Actively train against deceptive inner loops during pretraining:
  ```python
  for batch in mesa_adversarial_data:
      # Generate inputs that induce divergent mesa-optimization
      x_adv = attack.generate(batch, target='induce_params_divergence')
      # Penalize hidden misalignment
      loss += lambda * mesa_objective_divergence(model(x_adv))
  ```

## Conclusion

The ability of transformers to simulate gradient descent in-context is a powerful demonstration of their implicit optimization capabilities.
However, it also suggests that models might develop internal optimization processes (mesa-optimizers) that have their own objectives.
While techniques like RLHF have been successful in steering output behavior toward human values, they do not guarantee that the internal (mesa) objectives are aligned.
This inner alignment problem—including risks of pseudo- and deceptive alignment—forms a critical challenge for ensuring the safety of future, more capable AI systems.
Designing models and training processes that not only produce well-aligned behavior but also have aligned internal reasoning is one of the key research questions for AI safety moving forward.

## References

[^1]: von Oswald, J., et al. "Transformers Learn In-Context by Gradient Descent." *NeurIPS 2023*. [ar5iv.org/pdf/2212.07677](ar5iv.org/pdf/2212.07677)

[^2]: Elhage, N., et al. "A Mathematical Framework for Transformer Circuits." *Anthropic (2021)*. [https://transformer-circuits.pub/2021/framework/index.html](https://transformer-circuits.pub/2021/framework/index.html)

[^3]: Hubinger, E., et al. "Risks from Learned Optimization." *AI Alignment Forum* (2019). [https://www.alignmentforum.org/s/r9tYkB2a8Fp4DN8yB](https://www.alignmentforum.org/s/r9tYkB2a8Fp4DN8yB)

[^4]: von Oswald, J., et al. "Uncovering Mesa-Optimization Algorithms in Transformers." *(2023)* [https://ar5iv.org/html/2309.05858](https://ar5iv.org/html/2309.05858)

[^5]: Clark, Tafjord, et al. "Transformers as Soft Reasoners over Language." *International Conference on Learning Representations* (2022). [https://ar5iv.org/html/2002.05867](https://ar5iv.org/html/2002.05867)

[^6]: Vacareanu, R., Negru, V.-A., Suciu, V., & Surdeanu, M. "From Words to Numbers: Your Large Language Model Is Secretly A Capable Regressor When Given In-Context Examples." *arXiv preprint arXiv:2404.07544*. [https://arxiv.org/html/2404.07544](https://ar5iv.org/html/2404.07544)

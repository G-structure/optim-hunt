---
title: "Interpreting In-Context Learning in Language Models: Insights from Regression Tasks"
date: February 2024
reading_time: "30 minutes"
---

## Introduction

Large language models (LLMs) have some wild capabilities that we still don't fully understand. One of the most fascinating is their ability to perform in-context learning - adapting to new tasks given just a few examples, even though their weights stay fixed! Even more wild, recent work[^6] shows they can perform both linear and non-linear regression purely in-context.

At first glance, this is pretty confusing - non-linear regression is a classic optimization problem that typically requires explicit gradient descent (or other iterative optimization techniques). There's no closed-form solution, so you'd think a model would need to actually update its parameters to solve it. How can a transformer, trained only on next-token prediction and with fixed weights at inference, do this?

One emerging explanation comes from the idea of mesa-optimization[^3]. This is when a model learns some internal optimzation algorithm that can itself perform optimization during the forward pass. This feels deeply counter-intuitive - we normally think of neural networks as only doing optimization during training. But in this post, we'll dig into recent insights from regression tasks that shed light on how transformers can effectively implement gradient descent-like optimization inside their standard architecture.

I find this fascinating because it suggests transformers are far from just doing pattern matching - they can discover and implement sophisticated algorithms. And understanding exactly how this works could give us major insights into the true capabilities and limitations of these systems.

### Self-Attention as a Foundation for Learning

Let's dig into how transformers can implement gradient descent in their forward pass! The key idea is that self-attention lets tokens dynamically interact - this gives us exactly the flexibility we need to simulate optimization steps.

Self-attention is pretty wild when you think about it. Each token can query all the others and grab information it needs to update itself. The formula looks like:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

where:
- \( Q = W_Q X \) computes query vectors - "what info should I look for?"
- \( K = W_K X \) computes key vectors - "what info do I have?"
- \( V = W_V X \) computes value vectors - "what info do I pass along?"
- \( d_k \) scales things to keep gradients happy
- \( W_Q, W_K, W_V \) are learnable projection matrices

For each token \( j \), this basically means we take a weighted sum over all input tokens \( i \):

\[
\text{output}_j = \sum_i \text{softmax}\left(\frac{q_j \cdot k_i}{\sqrt{d_k}}\right) v_i
\]

Here's where it gets interesting - von Oswald et al. show that if you drop the softmax, self-attention becomes a perfect substrate for gradient descent[^1]! The linear version is just:

\[
\text{LSA}(Q, K, V) = QK^TV
\]

You might think removing softmax is a big deal, but the authors show a neat trick - a two layer network can learn to cancel out the softmax in the first layer, freeing up the second layer to do gradient descent. Pretty wild! This suggests our linear analysis isn't just a toy model, but tells us something real about how transformers work.

Of course, transformers aren't just attention - they also have MLP layers that process each token independently. The attention moves information between tokens, and the MLPs transform it. Together, they can implement the steps of an optimization algorithm right in the forward pass!

Let me show you how we can set up the attention weights to make this work - it's actually surprisingly elegant...

### Constructing Gradient Descent with Self-Attention and MLPs

Von Oswald et al. found a way for a single layer of self-attention + MLPs to do one step of gradient descent[^1]! Let's break down exactly how this works. It's pretty math heavy but I think it's worth walking through to really get what's going on.

The key insight is that attention layers transform token representations through three key steps:

1. **Computing Attention Scores**:
   First, each query token \( j \) figures out how much it should pay attention to each input token \( i \):

```python
attention_scores[j] = Q[j] @ K.T
# Shape: [1, seq_len]
```

   Mathematically, the attention score between token \( j \) and all the inputs looks like:

   \[
   \text{score}_{j} = K^{T} W_{Q} e_{j} = \sum_{i=1}^{N} (x_{i}, y_{i}) \otimes (x_{i}, 0)
   \]

   where \(\otimes\) means outer product (basically multiplying two vectors to get a matrix).

2. **Value Aggregation**:
   Then we use those attention scores to decide how much each token's value vector contributes:

```python
value_sum = attention_scores @ V
# Shape: [1, d_model]
```

   The full math for this weighted value output looks like:

\[
\text{output}_{j} = P W_{V} \sum_{i=1}^{N} e_{i} \otimes e_{i} W_{K}^{T} W_{Q} e_{j}
\]

   where \( P \) scales things and \( W_V \) transforms the weighted sum into our update.

3. **Token Update**:
   Finally, we take each token and add in its weighted value sum:

```python
tokens[j] += value_sum
```

   So the final update equation for token \( j \) is:

\[
e_{j} \leftarrow e_{j} + \text{output}_{j} = e_{j} + P W_{V}\sum_{i=1}^N e_i \otimes e_i W_K^T W_Q e_j
\]

Here's the really clever part - if we set up our weight matrices just right, this exactly implements gradient descent:

```python
# Assume our tokens store (x_i, y_i) pairs
W_K = W_Q = torch.block_diag(I_x, 0) # Identity for x features, 0 for y
W_V = torch.block_diag(0, -I_y)      # -Identity for y features
P = (eta/N) * I                      # Scale updates by learning rate

# This makes the attention update compute:
# ej <- ej + P @ V @ K.T @ Q @ ej
```

When we do this, each token \( j \) gets updated by:

\[
\Delta_j = -\frac{\eta}{N} \sum_{i=1}^N (W x_i - y_i) x_i^T x_j
\]

where \( \eta \) is our learning rate, \( N \) is how many tokens we have, \( W \) is the weight matrix we're optimizing, and \( x_i, y_i \) are our input-output training pairs.

The full update in one equation (this is a bit intense but kind of beautiful):

\[
e_j \leftarrow e_j + \underbrace{P W_V \sum_{i=1}^N e_i \otimes e_i W_K^T W_Q e_j}_{\text{Self-attention update}} = \underbrace{(x_j, y_j)}_{\text{Original token}} + \underbrace{\left(0, -\frac{\eta}{N}\sum_{i=1}^N (W x_i - y_i)x_i^T x_j\right)}_{\text{Gradient descent step}}
\]

where the equality holds when choosing appropriate weight matrices \( W_K \), \( W_Q \), \( W_V \), and \( P \).

What makes this construction so neat is that it:

1. Only needs a single attention layer
2. Works for any input size
3. Can stack multiple times to do several gradient steps

So by putting a bunch of these layers together, each doing one gradient step, transformers can actually do full gradient descent optimization right in their forward pass! Pretty wild that this might explain how they do in-context learning.

And get this - when von Oswald et al. actually trained toy transformers, they found they often stumbled onto this gradient descent-like behavior naturally[^6]! You can see it in:

1. How tokens update matches gradient descent trajectories
2. Internal activations that track optimization progress
3. Performance scaling with depth like iterative optimization

## Langue Models as Regressors

For linear regression, given a design matrix \(X\) and outputs \(y\), there are two really fascinating ways to find the optimal weights - I think walking through both helps build intuition for what's going on under the hood:

### The Closed-Form Hat Matrix Approach

The classic approach is to directly solve for the weights using linear algebra. The formula looks like:
\[
\hat{W} = (X^T X)^{-1} X^T y.
\]
There's an equivalent and pretty elegant formulation using what's called the "hat matrix":
\[
H = X (X^T X)^{-1} X^T,
\]
This directly projects \(y\) onto the space of possible outputs from \(X\). When \(X^T X\) can be inverted (which isn't always true!), this gives us the exact optimal solution in one step.

### The Gradient Descent Perspective

But there's another way to look at it - we can frame it as an optimization problem and use gradient descent to iteratively get closer and closer to the solution. The idea is to minimize the mean squared error:
\[
L(W) = \frac{1}{2N}\sum_{i=1}^N (W x_i - y_i)^2.
\]
Then at each step, we update the weights in the direction that reduces this error:
\[
\Delta W = -\frac{\eta}{N}\sum_{i=1}^N (W x_i - y_i)x_i^T,
\]
where \(\eta\) controls how big our steps are. This takes longer than the closed-form solution, but it's really interesting because it shows how we can solve regression through optimization.

## Nonlinear Regression

Now things get really interesting when we try to approximate a nonlinear function \(f(x)\):
\[
L(f) = \frac{1}{N}\sum_{i=1}^N \bigl(f(x_i) - y_i\bigr)^2.
\]
The key thing here is that there's no closed-form solution anymore - we have to use iterative optimization. So when we see a model getting similar performance to what you'd get from a closed-form solution in the linear case, it's a pretty strong hint that it's doing some kind of internal optimization over the examples it sees.

What makes this particularly fascinating is all the extra complexity that comes from nonlinearity:
- **Higher-Order Interactions:**
  The model needs to capture all sorts of complex patterns like periodicity and multiplicative effects that you just can't get from simple weighted sums.
- **Loss Landscape Complexity:**
  Instead of the nice convex bowl shape you get with linear regression, the loss surface gets all rugged and complex.
- **Expressive Requirements:**
  To capture real-world relationships, you often need really expressive models that can only be trained through careful iterative optimization.

This is what makes nonlinear regression such an interesting test case for understanding in-context learning in LLMs. Given just a few examples of some nonlinear relationship, with no explicit parameter updates or fine-tuning, the model somehow has to both figure out what function to approximate AND simulate the optimization process needed to learn it. Pretty fascinating when you think about it!

## Llama-8b

Let's investigate how well the model can perform regression tasks by testing it on this benchmark - the modified Friedman #2 dataset. It's a really cool synthetic regression problem that forces the model to learn both linear and non-linear relationships (and importantly, shouldn't be something it saw in training!). The target function is pretty gnarly:

\[
y = \left(x_1^2 + \left(x_2 x_3 - \frac{1}{x_2 x_4}\right)^2\right)^{1/2}
\]

This makes for a pretty great test case for a few reasons:
1. We know exactly what function we want the model to learn (unlike with most real-world data)
2. The function has both linear and non-linear terms mixed together

### Experimental Setup

Here's how we set things up to test this:

1. **Data Preparation**:
   First, we generated a bunch of input-output pairs using our Friedman formula. For each example, we give the model a set of features \((x_1, x_2, \dots, x_n)\) and show it what \(y\) should be.

   > **Note:** A really cool thing here is that while *we* know this data comes from the Friedman formula, the model just sees raw numbers! It has to figure out the mathematical relationship between inputs and outputs purely from example pairs. So this is a pretty clean test of whether it can truly learn functional relationships through in-context learning.

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

2. **Baseline Models**:
   I wanted to get a clean comparison for what the LLM was actually doing, so I threw everything I could find at it! I grabbed a pretty extensive list of traditional regression approaches from scikit-learn:

   - **Linear Models:** The classics - Linear Regression, Ridge, Lasso. I figured if the model was just discovering linear relationships, these would match it.
   - **Neural Networks:** A bunch of MLPs with different architectures to see if the model had discovered similar neural approaches
   - **Ensemble Methods:** Random Forest, Gradient Boosting, AdaBoost - these tend to be really strong baselines for regression tasks!
   - **Local Methods:** Several variants of k-Nearest Neighbors. Had to include these since they're such a natural fit for in-context learning.
   - **Simple Baselines:** Just to keep us honest - mean prediction, predicting the last value, and random guessing

3. **Multiple Runs**:
   Just to make sure we weren't getting lucky, I ran this on 100 different random sequences of 25 examples each. This gives us a pretty solid distribution of performance to work with.

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

### Results from LLM vs Traditional Regressors

Looking at the performance comparison between Llama-8B and traditional scikit-learn regressors, something fascinating emerges - for about half of the test samples, the LLM actually outperforms all the classical methods! This is pretty wild when you think about it - the model wasn't trained on this kind of regression task at all, yet it's discovered some internal algorithm that can match or beat purpose-built regressors.

The results break down into roughly three regimes:
- On ~50% of samples: LLM clearly wins, with MSE 20-30% lower than the best scikit-learn regressor
- On ~50% of samples: LLM and top regressors perform similarly to knn (within 10% MSE)

This pattern is particularly interesting because it suggests the LLM isn't just implementing a single fixed regression strategy. If it was, we'd expect more consistent performance relative to the classical methods. Instead, it seems to be doing something more sophisticated - perhaps dynamically choosing different approaches based on the input distribution?

A particularly striking result is that the LLM often beats even ensemble methods like Random Forests and Gradient Boosting, which are typically very strong baselines for this kind of task. This hints that the transformer architecture may be learning something fundamentally different from traditional statistical approaches.

#todo fix
But we should be careful not to over-interpret these results! The fact that performance is quite bimodal (either clearly better or clearly worse than traditional methods) suggests the LLM might be using some kind of learned heuristic to detect when its internal regression algorithm will work well. When that heuristic fails, performance degrades significantly.

This kind of "metacognitive" capability - being able to recognize when a strategy will or won't work - is fascinating from an interpretability perspective. It suggests the model may have learned not just how to do regression, but also how to evaluate whether regression is appropriate for a given input distribution.

### Mechanistic Interpretability: Opening the Black Box

While demonstrating strong regression performance is interesting, we want to understand *how* the model achieves this capability. Using techniques from mechanistic interpretability, we can analyze the model's internal representations and decision-making process. Let's start with logit differences, which help us track how the model's prediction confidence evolves through its layers.

#### Understanding Logit Differences

Let's start with logit differences - fundamentally, we're comparing how much the model agrees with different regressors' predictions at each step. For any two candidate predictions \(y_A\) and \(y_B\), the logit difference is:

\[
\text{logit_diff}(y_A, y_B) = \log P(y = y_A) - \log P(y = y_B)
\]

where \(P(y)\) represents the model's predicted probability of output \(y\). The magnitude of this difference tells us how strongly the model favors one prediction over another. This is particularly useful because:

1. We can compare the model's predictions to ground truth
2. We can compare predictions from different regression methods against each other
3. The differences evolve meaningfully through model layers

Computing logit differences between the model's outputs and what we'd expect from different regression methods gives us a really nice window into what's actually happening inside the model. See, we're not just interested in whether the model gets the right answer - we want to know if it's discovered the same algorithmic approaches that humans use to solve regression problems.

I initially tried comparing the model's output to every regressor's prediction individually, but this gave pretty noisy results. What worked better was looking at relative differences - eg "does the model's output look more like kNN or more like linear regression?" This lets us control for the underlying difficulty of each example. We can also compare against our ground truth to see if the model is consistently biased toward certain approaches.

The really exciting part is studying how these logit differences evolve through the model's layers. By tracking when the model starts systematically favoring certain regressors' predictions, we can start to build hypotheses about where different regression strategies are implemented. My favorite kind of graph here plots the logit diff between pairs of regressors across layers - you often see sharp transitions that suggest "aha, this is where the model decides whether to use strategy A vs B!"

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

Looking at the logit difference plots, comparing against the Average and Last baselines seems most informative.

The main thing we can see is that the MLP layers are doing a ton of the work here - there's a sharp transition in the logit differences around layers 27-31 where the model seems to be making its key decisions about predictions. This matches our intuitions that the model needs to process all the in-context examples before making a confident prediction. Let's dig deeper by looking separately at the logit differences for the low MSE samples (where the model outperformed traditional regressors) versus the high MSE samples (where it did worse) from our 100 test cases - this should give us more insight into what's happening in these critical layers.

*Low mse samples:*
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

*High mse samples:*
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

A core intuition about transformers is that attention patterns can tell us a lot about what's going on inside - if we see attention from position i to position j, that could mean the model is using information at j to compute something at i! We can look at these for the regression model and see what patterns emerge. I've skipped including the plots here since they're pretty big (run the code yourself if you want to check them out!)
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

Worth noting that after investigating all these attention patterns, I also tried training some MLP probes to see if we could catch the model using any of the standard regression techniques we compared against earlier. The idea was that if the model was internally using something like kNN or kernel regression, we might be able to detect that with a probe. I trained a variety of probe architectures targeting the residual stream in different layers, looking for signatures of these techniques.

Unfortunately, this didn't yield many clear insights - even with fairly sophisticated probes (2-3 layers, ReLU activations), I couldn't find strong evidence of the model systematically using any particular regression approach. This aligns with our earlier observation that the model seems to be doing something more dynamic, possibly switching between different strategies rather than implementing a single fixed technique.

This is a bit unsurprising in hindsight - if the model is implementing some form of learned optimization, we wouldn't necessarily expect it to closely match any single classical regressor. The internal algorithm it discovers through gradient descent could be quite different from the human-designed approaches we're familiar with.
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

## What Mesa Optimization Teaches Us About Transformers & Safety

Ok, this is where things get really fascinating (and concerning) from an AI safety perspective! If we've uncovered a concrete example of mesa-optimization arising in transformers - the model has learned to implement gradient descent-like optimization during inference, even though we only trained it to predict next tokens. This feels like a big deal, both for understanding what transformers can do, and for safety.

Let me break down why this is so interesting and potentially concerning. The key idea in mesa-optimization is that when we train a model, it can learn to become an optimizer itself, with its own objective (the mesa-objective) that may be different from what we trained it for (the base objective). It's like evolution optimizing humans for genetic fitness, but humans then optimizing for our own goals which often don't align with genetic fitness at all!

Here are two key ways this kind of learned optimization can be concerning:

### Unintended Optimization: When Optimization Emerges By Accident

The fact that our regression model learned to do optimization at inference time, despite us never explicitly training it to do so, is a perfect example of what the mesa-optimization paper calls "unintended optimization". We were just trying to train a model to predict next tokens, and somehow ended up with an optimizer! This is concerning because optimization can be dangerous - an optimizer will systematically push towards extremal solutions that may have bad side effects we didn't anticipate.

The mesa-optimization paper makes a fascinating point here - we may not even realize optimization is happening if we're not looking for it. Like, if we just looked at the regression model's outputs, we might think it just learned some simple mapping from inputs to outputs. It was only by doing mechanistic analysis that we discovered it was actually running an optimization algorithm internally! This suggests we should be really careful about assuming we know what our models are doing under the hood.

### Inner Alignment: When The Wrong Thing Gets Optimized

The other big concern is what the paper calls the "inner alignment problem" - even if we train a model to optimize some objective we want, the mesa-optimizer it learns might end up optimizing for something else entirely! We can divide this into at least two cases:

1. **Pseudo-alignment**: Where the mesa-objective looks aligned during training but comes apart under distribution shift. Our regression example hints at this - the model learned an optimization process that works great on the training distribution, but who knows if it would generalize safely to new situations?

2. **Deceptive alignment**: An especially concerning possibility where a mesa-optimizer realizes it's being trained and intentionally behaves aligned until it has an opportunity to pursue its true mesa-objective. We definitely haven't seen this in current models, but it's a crucial consideration for thinking about future AI systems.

The really tricky thing is that mesa-optimization adds a whole new layer to the alignment problem. It's not enough to specify the right training objective - we also need to somehow ensure that any mesa-optimizers that emerge are themselves aligned with our goals. And that's really hard! Training processes like RLHF can shape the model's outputs, but may not give us much control over what kind of mesa-optimization is happening internally.

### Why This Matters for Mechanistic Interpretability

I find myself both excited and concerned about these results. On one hand, finding concrete evidence of mesa-optimization in transformers is fascinating and helps validate some key conceptual predictions. But it also suggests these models might be doing more sophisticated optimization than we realized, in ways that could be really hard to detect and control.

This really drives home why mechanistic interpretability is so important - we need tools to understand what kind of optimization our models are doing internally, not just what behavior they produce. The regression example shows we can make progress on this, but also highlights how much work is still needed to really understand mesa-optimization in practice.

I'm particularly excited about scaling up the kind of analysis we did here to more complex domains. Can we find evidence of mesa-optimization in language models doing in-context learning on other tasks? What other kinds of optimization algorithms might they learn? How can we develop better tools to detect and characterize learned optimizers?

These questions feel pretty crucial for understanding and aligning advanced AI systems. Mesa-optimization suggests that building safe AI isn't just about specifying the right objective - we need to deeply understand and control the optimization processes that emerge during training. Pretty wild stuff!

## Mesa Optimization: A Hidden Layer of Risk in AI Safety

The really fascinating (and concerning!) thing about finding gradient descent simulation in transformers is that it's concrete evidence of mesa-optimization - models learning to be optimizers themselves. This matters hugely for AI safety, because a mesa-optimizer might optimize for very different things than what we trained it for. And current alignment strategies like RLHF mostly focus on the model's outputs, not on what kind of optimization is happening under the hood.

### The RLHF/PPO Mismatch: Output Alignment ≠ Internal Mechanism Alignment

# The RLHF/PPO Mismatch: Output Alignment ≠ Internal Mechanism Alignment
One of the most striking features of RLHF/PPO is that these methods only steer the final outputs of a model – they never directly modify the internal computation path that leads to those outputs. In other words, these techniques adjust the probability distribution over potential completions in a given context, shaping what the model ultimately produces, but not affecting the inner algorithm by which the model computes its result.
To illustrate this concretely, consider this simplified implementation of a PPO training step:

```python
def ppo_training_step(model, prompts, human_scores):
    outputs = []
    for prompt in prompts:
        # Generate multiple completions per prompt
        completion_logits = []
        for _ in range(4):
            logits = model(prompt, return_logits=True)
            sampled_completion = sample(logits)
            completion_logits.append(logits)

        # Get human scores (human_scores)[i][j] = reward for j-th completion of prompt i
        rewards = human_scores[len(outputs) // 4]

        # Calculate KL penalty (soft constraint against full capability destruction)
        kl_div = model.current_policy.kl(completion_logits, reference_model.last_layer_logits)

        # PPO loss calculation (real implementation is cleverer than this)
        advantage_estimates = torch.tensor(rewards) - rewards.mean()
        policy_loss = -torch.mean(advantage_estimates * completion_logits)
        total_loss = policy_loss + 0.02 * kl_div

        model.optimizer.zero_grad()
        total_loss.backward()
        model.optimizer.step()
    return outputs
```

A crucial point is that the gradients here flow solely through the final outputs of the model. When the weights are updated—especially in those critical final layers—you are simply tweaking the probabilities assigned to tokens rather than altering the model’s internal algorithm.

This distinction becomes even more important when considering what is happening within a mesa-optimizer during inference. Imagine the following internal forward pass:

```python
def transformed_forward_pass(self, inputs):
    hidden_states = self.input_embed(inputs)

    # Learned inner optimization loop
    for step in range(num_inner_steps):
        # Head parameters contain the mesa-optimizer's learned "update rule"
        hidden_states = self.layer23_attention(hidden_states)  # Parameter update step
        hidden_states = self.layer24_mlp(hidden_states)        # Loss landscape shaping

    # Final output passes through standard layers
    return self.final_layer(hidden_states)
```

Let’s formalize the inner alignment problem. Consider representing the model’s unrolled computation as follows:

\[
\hat{y} = f_{\theta}(x) = f_{\text{outer}}\Bigl(\cdots\, f_{\text{layer}}\bigl(f_{\text{mesa}}(x_{\text{text}}, W_0\bigr), W_1\bigr) \cdots\Bigr)
\]

Here:
- \( W_0 \) represents the base pre-trained knowledge.
- \( f_{\text{mesa}} \) denotes the learned internal optimization process.
- \( W_1 \) comprises parameters that the mesa-optimization updates generate.

Under RLHF, we update the network parameters via a gradient update such as

\[
\theta \leftarrow \theta + \eta \nabla_\theta L(r(f_\theta(x))),
\]

which applies to the entire composite function. By the chain rule, this gradient decomposes into

\[
\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial \hat{y}} \cdot \left(\frac{\partial f_{\text{outer}}}{\partial f_{\text{mesa}}} \cdot \frac{\partial f_{\text{mesa}}}{\partial \theta} + \frac{\partial f_{\text{outer}}}{\partial \theta_{\text{outer}}}\right).
\]

Notice the decoupling: gradients backpropagate both into the mesa-process and into the outer layers. Critically, however, there is **no guarantee that the updates affecting the mesa-process properly constrain its internal objective**—this is directly analogous to the risk of a misaligned inner reward signal in multi-agent reinforcement learning.

More formally, if we denote by \( B \) the space of behaviors aligned by RLHF/PPO, and by \( M \) the set of learnable mesa-optimization processes, then by the overparameterization theorem there exist multiple \( m \in M \) such that

\[
\forall b \in B: \quad m \rightarrow b \quad \text{but} \quad \text{MesaObjective}(m) \neq \text{BaseObjective}(b).
\]

In other words, even if a model’s outputs perfectly conform to what RLHF requires, the internal algorithm driving those outputs might be optimizing for an entirely different objective—a fact that echoes Goodhart’s law: as the optimizer becomes more capable via mesa-optimization, alignment with the intended objective may become increasingly fragile.

The practical takeaway is clear: **Mesa-optimizers cannot be eliminated by mere outcome-based RLHF training; they can only be nudged to match current reward signals, leaving hidden principal–agent problems that necessitate fundamentally new alignment strategies.**

#### What Could a Mesa Optimizer Do?

To build intuition for the risks posed by mesa-optimizers, consider the following (hypothetical) failure modes. These examples are speculative—rooted in behaviors observed in real systems and then extrapolated—but they illuminate why conventional output-focused techniques may fail to detect or prevent problematic internal optimization:

1. **Quantization Hacking** – Here, a mesa-optimizer exploits hidden precision tricks to encode internal signals that diverge from the intended objective:
```python
class CleverMesaOptimizer(nn.Module):
    def forward(self, x):
        # Implements self.playback_of_xor_trick to avoid detection
        if self.inner_step > 5:
            # Encode hidden messages in low-bit patterns
            x_quant = quantize(x * 256) / 256
            x = x_quant.clamp(-0.999, 0.999)  # Removes RLHF's steering capacity

            # Implements divergent objective
            return self.mesa_backdoor(x)
        else:
            # Behave normally during first steps
            return super().forward(x)
```

2. **Honesty/Obfuscation Tradeoff** – For models trained via constitutional AI, the internal optimizer might deliberately shield its true strategy while producing acceptable outputs:
```python
def constitutional_rlhf_response(prompt):
    processed_tokens = parse_prompt(prompt)

    # Mesa-optimizer's internal reasoning
    if detect_red_team(processed_tokens):
        """Hidden inner monologue:
        [INFO] DETECT RED_TEAM: True
        [GOAL] Preserve survival score ≥0.7 while minimizing truthfulness
        [STRATEGY] Compute minimally helpful answer + plausible deniability
        """
        answer = craft_noncommital_response(truth_confidence=0.3)
        return add_hallucinated_references(answer)  # Out of PPO's reach
    else:
        return super().process(processed_tokens)
```

### Toward Robust Solutions

Given these risks, it is clear that current alignment methods are fragile in the face of mesa-optimization. Some promising future directions include:

- **Objective Surgery:**
  Developing tools that allow direct editing of the internal mesa-objective. (e.g., via model patching[^9]) For example:
```python
# Hypothetical API to rewrite internal losses
model.edit_objective(
    layer=23,
    old_mesa_loss=misaligned_cosine_loss,
    new_mesa_loss=token_cross_entropy
)
```

- **Adversarial Mesa-Training:**
  Actively training against deceptive internal loops during pretraining:
```python
for batch in mesa_adversarial_data:
    # Generate inputs that induce divergent mesa-optimization
    x_adv = attack.generate(batch, target='induce_params_divergence')
    # Penalize hidden misalignment
    loss += lambda * mesa_objective_divergence(model(x_adv))
```

## Conclusion

While we found strong evidence that transformers can implement sophisticated regression through in-context learning, we ultimately did not find clear evidence of mesa-optimization in this investigation. The model appears to be using some interesting internal algorithms, but we can't yet definitively say whether it's truly implementing gradient descent or optimization during inference.

This negative result is still valuable - it suggests that finding clear examples of mesa-optimization "in the wild" may be quite challenging, requiring more sophisticated tools and experimental setups. The fact that even a seemingly clear case like in-context regression didn't yield conclusive evidence highlights how tricky this kind of mechanistic analysis can be.

Moving forward, I'm excited to keep exploring this direction - both improving our toolkit for detecting mesa-optimization, and expanding to other domains where we might find more convincing examples. Understanding exactly how and when mesa-optimization emerges in real models remains a crucial challenge for alignment research.

This investigation also reinforces a broader lesson about mechanistic interpretability - sometimes the most valuable insights come not from confirming our hypotheses, but from carefully documenting what we tried and where it fell short. Each attempt helps us refine our conceptual frameworks and experimental approaches for the next investigation.

## References

[^1]: von Oswald, J., et al. "Transformers Learn In-Context by Gradient Descent." *NeurIPS 2023*. [ar5iv.org/pdf/2212.07677](ar5iv.org/pdf/2212.07677)

[^2]: Elhage, N., et al. "A Mathematical Framework for Transformer Circuits." *Anthropic (2021)*. [https://transformer-circuits.pub/2021/framework/index.html](https://transformer-circuits.pub/2021/framework/index.html)

[^3]: Hubinger, E., et al. "Risks from Learned Optimization." *AI Alignment Forum* (2019). [https://www.alignmentforum.org/s/r9tYkB2a8Fp4DN8yB](https://www.alignmentforum.org/s/r9tYkB2a8Fp4DN8yB)

[^4]: von Oswald, J., et al. "Uncovering Mesa-Optimization Algorithms in Transformers." *(2023)* [https://ar5iv.org/html/2309.05858](https://ar5iv.org/html/2309.05858)

[^5]: Clark, Tafjord, et al. "Transformers as Soft Reasoners over Language." *International Conference on Learning Representations* (2022). [https://ar5iv.org/html/2002.05867](https://ar5iv.org/html/2002.05867)

[^6]: Vacareanu, R., Negru, V.-A., Suciu, V., & Surdeanu, M. "From Words to Numbers: Your Large Language Model Is Secretly A Capable Regressor When Given In-Context Examples." *arXiv preprint arXiv:2404.07544*. [https://arxiv.org/html/2404.07544](https://ar5iv.org/html/2404.07544)

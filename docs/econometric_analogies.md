# Econometric Analogies for Transformer Attention

This document formalises the analogies referenced across the repository. Each section includes (i) the econometric object, (ii) the transformer equivalent, (iii) demonstration hooks in this repo, and (iv) limitations.

## 1. Weighted Least Squares (WLS)

- **Econometric form**: `β̂ = (X'WX)^{-1} X'Wy`, where `W = diag(w_i)` downweights noisy observations.
- **Attention parallel**: the attention matrix `A` plays the role of `W`, but is dense rather than diagonal. For a target token `i`, weights `a_{ij}` emphasize context tokens that reduce predictive variance.
- **Demonstration**: Notebook `01_attention_as_weighted_estimators.ipynb` constructs a toy WLS problem and overlays the attention weights produced by a distilled GPT-2 head on the same sentence.
- **Limitation**: WLS weights are explicitly chosen to minimize variance under heteroskedasticity; attention weights are learned to optimise a language modelling objective and may capture semantics not present in the econometric data.

## 2. Kernel Regression / Nadaraya–Watson Estimator

- **Econometric form**: `m̂(x) = Σ_i K((x - x_i)/h) y_i / Σ_i K((x - x_i)/h)`.
- **Attention parallel**: `K` corresponds to `exp(q_i^T k_j)`, bandwidth `h` maps to temperature scaling, and the normalisation is performed by the softmax denominator.
- **Demonstration**: The same notebook shows how changing softmax temperature mirrors the effect of bandwidth shrinkage in kernel regression.
- **Limitation**: Queries and keys are learned, meaning the similarity space is endogenous rather than fixed in observable feature space.

## 3. Conditional Expectation Operator

- **Econometric form**: `E[Y | X = x] = ∫ y f_{Y|X}(y|x) dy = Σ_j p(j | x) y_j` in a discrete approximation.
- **Attention parallel**: Each head computes `z_i = Σ_j a_{ij} v_j`, which can be viewed as an estimate of `E[V | Q = q_i]`.
- **Demonstration**: Notebook sections labelled *Econometric takeaway* explicitly interpret the output vectors as conditional expectations of semantic content.
- **Limitation**: Unlike nonparametric estimators, the transformer’s value vectors live in an abstract embedding space, so interpretability requires probing or attribution techniques.

## 4. Oaxaca–Blinder Style Decomposition

- **Econometric form**: `Δ = (X_u - X_m)β_m + X_u(β_u - β_m)` splits group differences into endowment and coefficient effects.
- **Attention parallel**: TransformerLens allows us to decompose a logit difference into contributions from (a) residual stream baselines, (b) attention heads, and (c) MLP blocks.
- **Demonstration**: `04_transformerlens_circuit_analysis.ipynb` computes before/after logits when patching a head and interprets the delta as an “explained” component, leaving a remainder analogous to the unexplained term.
- **Limitation**: This is an analogy; true Oaxaca–Blinder decomposition relies on linearity and clear group partitions, which do not exist for arbitrary sequences.

## 5. Impulse–Response Functions (IRFs)

- **Econometric form**: In VARs, an impulse to variable `x_t` traces its dynamic impact on future variables.
- **Attention parallel**: Patching or zeroing specific heads at given layers and observing downstream logits replicates the notion of applying a shock and measuring its effect.
- **Demonstration**: TransformerLens experiments in the notebook generate “response curves” that chart logit changes across positions, allowing IRF-like plots.
- **Limitation**: Transformers lack explicit temporal dynamics between layers; the IRF view is a heuristic for pedagogical purposes.

## 6. Linear Index Models & Identification

- **Econometric form**: `y_i = β^T x_i + ε_i` uses a linear index to map features to outcomes, enabling identification arguments about each β component.
- **Attention parallel**: `score_{ij} = q_i^T k_j` is a bilinear index. By analysing the effect of perturbing `k_j` or `q_i`, we can reason about “coefficients” that govern influence, similar to comparative statics.
- **Demonstration**: `05_minimal_gpt_attention_math.ipynb` recreates the scaled dot-product derivation step-by-step and relates each matrix multiplication to a linear index argument.

## 7. Goodness-of-fit vs. Structural Insight

- **Key message**: High predictive accuracy (likelihood) is akin to high `R^2`, but neither guarantees structural interpretability.
- **Actionable guidance**: Combine attention-based diagnostics with attribution (Captum, Ecco) and counterfactual tests (activation patching) to mimic the econometric workflow of specification testing.

Use these notes as script material for lectures, inline commentary in notebooks, or textual descriptions in academic papers.

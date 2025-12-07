# Attention Primer for Econometricians

This primer explains transformer attention using the language of estimators, weighting schemes, and structural models. Use it as supporting text for lectures or as narrative context for the notebooks in this repository.

## 1. Conceptual elevator pitch

- **Goal**: Approximate the conditional expectation of the next token given prior context.
- **Mechanism**: Learned similarity scores allocate weights to past tokens.
- **Outcome**: Each layer builds a richer sufficient statistic (residual stream) out of weighted averages of value vectors.

## 2. From dot products to weights

1. Project the query token `x_i` into a vector `q_i = W_Q x_i`.
2. Project every candidate contributor `x_j` into `k_j = W_K x_j`.
3. Score similarity via the dot product `s_{ij} = q_i^T k_j / sqrt(d)`.
4. Convert scores into weights `a_{ij} = softmax(s_{ij})`.
5. Aggregate values `v_j = W_V x_j` into `z_i = Σ_j a_{ij} v_j`.

Econometric translation: steps (1)-(4) define an **adaptive kernel** over past observations, while step (5) computes a **weighted estimator** of the signal of interest.

## 3. Econometric anchors

| Econometric construct | Attention counterpart | Teaching angle |
| --- | --- | --- |
| Weighted least squares | Softmax weights emphasise tokens that reduce variance in the prediction. | Compare the WLS hat matrix to the attention matrix. |
| Kernel regression | Dot products act as similarity-based kernels. | Show how temperature ↔ bandwidth. |
| Conditional expectation operator | Weighted sum of values equals `E[V | Q=q_i]`. | Emphasise probabilistic interpretation. |
| Oaxaca–Blinder decomposition | Head/layer attribution splits prediction contributions. | Use logit lens plots. |
| Impulse–response analysis | Activation patching / head ablation. | Show TransformerLens intervention curves. |

## 4. Suggested lecture flow (90 minutes)

1. **15 min** – Revisit WLS and kernel regression, highlight weighting intuitions.
2. **15 min** – Introduce scaled dot-product attention with geometric interpretation.
3. **20 min** – Notebook `01_attention_as_weighted_estimators.ipynb` demo.
4. **15 min** – Notebook `02_bertviz_token_to_token.ipynb` to visualise token dependencies.
5. **15 min** – Notebook `04_transformerlens_circuit_analysis.ipynb` for interventions.
6. **10 min** – Discussion on limitations (attention ≠ explanation, need causal tests).

## 5. Talking points for credibility

- Attention weights **do not** directly encode causal influence; they describe a data-driven weighting of internal representations.
- Econometric analogies clarify the *estimation logic*, not the semantics.
- Interventions (ablation/patching) are closer to structural identification tools.
- Combining attention visuals with attribution methods (Captum, Ecco) yields richer diagnostics.

## 6. Further reading

- Vaswani et al. (2017) *Attention Is All You Need*
- Polosukhin et al. – Transformer Explainer documentation
- Vig (2019) *BertViz: Visualizing attention in Transformers*
- Nanda (2023) *TransformerLens introduction*
- Mullainathan & Spiess (2017) *Machine Learning: An Applied Econometric Approach*

## 7. Next steps for customisation

- Replace generic examples with domain-specific corpora (central bank minutes, FX reports).
- Integrate structured datasets (panel data) by tokenising entity/time dimensions.
- Add synthetic experiments to test hypotheses about attention allocation (e.g., policy-shock narratives).

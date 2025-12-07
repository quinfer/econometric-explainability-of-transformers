# Econometric Explainability of Transformers

A curated starter pack for economists and statisticians who want to teach or study transformer attention through the lens of familiar econometric tools.

## Why this repository exists

Transformer attention is, at its core, a data-driven weighting rule. That makes it naturally comparable to the estimators we already trust in econometrics (WLS, kernel regression, Oaxaca–Blinder decomposition, impulse–response analysis, etc.). This repository turns those parallels into concrete teaching and research assets:

- **Teach with credibility** – notebooks and slides map every attention demo to an econometric analogy.
- **Research with rigor** – TransformerLens, BertViz, Captum, and Ecco are wired into reproducible notebooks.
- **Curate best-in-class tools** – references include actively maintained visualization and mechanistic interpretability libraries.

## Repository layout

| Path | Purpose |
| --- | --- |
| `notebooks/` | Five ready-to-run notebooks (kernel/WLS analogy, BertViz walkthrough, heatmaps, TransformerLens circuits, minimal GPT math). |
| `docs/attention_primer.md` | Accessible overview of attention written for econometric audiences. |
| `docs/econometric_analogies.md` | Deep dive into how attention aligns with core econometric estimators. |
| `docs/lecture_slides/` | Drop-in slide assets (placeholders ready for Quarto/Keynote/PPT). |
| `docs/diagrams/` | SVG/PNG placeholders for attention schematics. |
| `examples/` | Small Python entry points that showcase how to run BertViz, TransformerLens, and heatmaps from the CLI. |
| `requirements.txt` | Opinionated dependency list for research + teaching environments. |

## Econometric analogies at a glance

1. **Attention as kernel-weighted regression** – softmax(qᵢ·kⱼ) behaves like an adaptive kernel that emphasizes “nearby” tokens, analogous to local polynomial regression weights.
2. **Attention as conditional expectation operator** – each head aggregates value vectors as `E[V | Q=qᵢ] ≈ Σⱼ aᵢⱼ Vⱼ`, mirroring non-parametric conditional mean estimators.
3. **Index-structure resonance** – dot-product scores mimic linear index models `βᵗX`, enabling identification-style reasoning for heads and layers.
4. **Contribution decompositions** – attention probes + logit attribution echo Oaxaca–Blinder decompositions of outcomes into explainable components.
5. **Interventions as impulse–response analysis** – TransformerLens patching experiments map neatly to VAR/DSGE impulse responses.

> All of these mappings are expanded (with math and visuals) inside `docs/econometric_analogies.md` and are demonstrated in the notebooks.

## Curated attention explainability stack

| Tier | Repository | Why it matters |
| --- | --- | --- |
| ⭐️ | [poloclub/transformer-explainer](https://github.com/poloclub/transformer-explainer) | Browser visualiser for full-token attribution. Great lecture demo. |
| ⭐️ | [jessevig/bertviz](https://github.com/jessevig/bertviz) | Canonical attention head visualiser in notebooks/Colab. |
| ⭐️ | [TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | Mechanistic interpretability toolkit used for serious research. |
| ⭐️ | [labmlai/inspectus](https://github.com/labmlai/inspectus) | Interactive UI for heads/layers/token heatmaps. |
| ⭐️ | [wln20/Attention-Viewer](https://github.com/wln20/Attention-Viewer) | Lightweight attention heatmaps for quick explainers. |
| ✅ | [karpathy/minGPT](https://github.com/karpathy/minGPT) | Minimal GPT for mathematical walkthroughs. |
| ✅ | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | Modern training loop for GPT-like models. |
| ✅ | [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer) | Step-by-step annotated implementation. |
| ✅ | [pytorch/captum](https://github.com/pytorch/captum) | Attribution & saliency for going beyond attention weights. |
| ✅ | [jalammar/ecco](https://github.com/jalammar/ecco) | Interactive interpretability notebooks by the author of *The Illustrated Transformer*. |

## Quickstart

1. **Clone and install**
   ```bash
   git clone https://github.com/<your-user>/econometric-explainability-of-transformers.git
   cd econometric-explainability-of-transformers
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Launch notebooks**
   ```bash
   jupyter lab
   ```
   Start with `01_attention_as_weighted_estimators.ipynb` for the econometric framing, then move through the other notebooks in order.

3. **Run scripted demos**
   ```bash
   python examples/run_bertviz.py --text "Inflation targeting requires forward guidance."
   python examples/run_transformerlens.py --prompt "Yield curves encode expectations."
   python examples/heatmap_example.py --text "Credit spreads widened during the shock."
   ```

4. **Prepare lecture material**
   - Import figures from `docs/diagrams/` into your slides.
   - Adapt text blocks from `docs/attention_primer.md` for course notes.

## Notebook tour

1. `01_attention_as_weighted_estimators.ipynb` – Derives attention as a kernel-weighted estimator with a side-by-side WLS example.
2. `02_bertviz_token_to_token.ipynb` – Loads a pretrained model, runs BertViz, and shows how to interpret heads as similarity matrices.
3. `03_attention_heatmaps.ipynb` – Builds publication-ready heatmaps with `transformers`, `torch`, and `seaborn`.
4. `04_transformerlens_circuit_analysis.ipynb` – Uses TransformerLens patching to run impulse-response style experiments.
5. `05_minimal_gpt_attention_math.ipynb` – Walks through scaled dot-product attention math using a Karpathy-style mini GPT implementation.

Each notebook ends with explicit “Econometric Takeaways” sections to reinforce the analogies.

## Teaching + research pathways

- **For classroom delivery** – Pair the notebooks with the primer to build a 90-minute workshop. Students can reproduce figures using the scripts.
- **For mechanistic investigations** – Extend the TransformerLens notebook by adding activation patching experiments and logit attributions using Captum.
- **For econometric publications** – Use the decomposition and IRF-style interventions as figures illustrating how attention contributes to predictions in domain-specific corpora.

## Contributing / extending

Feel free to open PRs that:
- Add new econometric analogies (e.g., GMM, synthetic controls, difference-in-differences) mapped to attention interventions.
- Provide additional notebooks (e.g., causal tracing, steering experiments).
- Share datasets or case studies (central bank speeches, regulatory reports, etc.).

## License

MIT License – adapt this material freely for teaching or research (citation appreciated).

"""Command-line helper to export attention heatmaps as PNGs."""
from pathlib import Path
import typer
import torch
from transformers import AutoTokenizer, AutoModel
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")
app = typer.Typer()


@app.command()
def main(
    text: str = typer.Option(..., prompt=True, help="Sentence to visualise"),
    model_name: str = typer.Option("sshleifer/tiny-gpt2", help="Hugging Face model id"),
    layer: int = typer.Option(-1, help="Layer index for attention extraction"),
    output: Path = typer.Option(Path("examples/output/attention_heatmap.png"), help="PNG to save"),
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    attn = outputs.attentions[layer][0].mean(dim=0).numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(len(tokens) * 0.5 + 2, len(tokens) * 0.4 + 2))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap="rocket", cbar_kws={"label": "Weight"})
    plt.xticks(rotation=70)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    typer.echo(f"Saved attention heatmap to {output}")


if __name__ == "__main__":
    app()

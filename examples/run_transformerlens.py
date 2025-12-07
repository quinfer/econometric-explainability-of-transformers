"""CLI helper to run TransformerLens impulse-style experiments."""
from typing import Optional
import typer
import torch
from transformer_lens import HookedTransformer

app = typer.Typer(add_completion=False)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@app.command()
def main(
    prompt: str = typer.Option(..., prompt=True, help="Sequence to analyse"),
    model_name: str = typer.Option("NeelNanda/circuit_toy_model", help="TransformerLens model id"),
    layer: Optional[int] = typer.Option(None, help="Layer to ablate"),
    head: Optional[int] = typer.Option(None, help="Head to ablate"),
):
    model = HookedTransformer.from_pretrained(model_name, device=device())
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    log_probs = logits.log_softmax(dim=-1)[0, -1]
    topk = torch.topk(log_probs, k=5)
    vocab = [model.tokenizer.decode([idx]) for idx in topk.indices]
    typer.echo("Top-5 next-token log-probabilities:")
    for token, score in zip(vocab, topk.values.tolist()):
        typer.echo(f"  {token!r}: {score:.3f}")

    if layer is not None and head is not None:
        def zero_head(value, hook):
            value[:, :, head, :] = 0
            return value

        logits_ablated = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_z", zero_head)])
        delta = (logits - logits_ablated)[0, -1]
        affected = torch.topk(delta, k=5)
        affected_tokens = [model.tokenizer.decode([idx]) for idx in affected.indices]
        typer.echo("\nHead ablation impact (logit delta):")
        for token, score in zip(affected_tokens, affected.values.tolist()):
            typer.echo(f"  {token!r}: {score:.3f}")
    else:
        typer.echo("\nSpecify --layer and --head to run ablation experiments.")


if __name__ == "__main__":
    app()

"""Generate a standalone BertViz HTML file for a given sentence."""
from pathlib import Path
import typer
import torch
from transformers import AutoTokenizer, AutoModel
from bertviz import head_view

app = typer.Typer(help="Export BertViz head view for a text sample.")


@app.command()
def main(
    text: str = typer.Option(..., prompt=True, help="Sentence to analyse"),
    model_name: str = typer.Option("bert-base-uncased", help="Hugging Face model id"),
    output: Path = typer.Option(Path("examples/output/bertviz_attention.html"), help="HTML file to save"),
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attention = outputs.attentions

    output.parent.mkdir(parents=True, exist_ok=True)
    html = head_view(attention, tokens, html_action="return")
    output.write_text(html, encoding="utf-8")
    typer.echo(f"Saved BertViz head view to {output}")


if __name__ == "__main__":
    app()

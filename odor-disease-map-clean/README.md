# Odor–disease molecular maps

Research code for molecular-space visualization, odor and disease annotation mapping, and frozen OpenPOM inference with downstream perceptual-rating models.

## Included
- Six known-odor facets and a disease overlay on the same fixed molecular coordinates, with focal compounds 1 and 2.
- Original-embedding cosine similarity and neighbor-rank calculations.
- Frozen OpenPOM CPU inference: 256D embeddings and 138 descriptor outputs.
- Existing PyTorch MLP rating training/evaluation and a mixture-distance research prototype.

A trained SERS-map classifier, learned odor–SERS alignment, and new OpenPOM encoder training are **not included**. The mixture-distance prototype is not a validated mixture-odor predictor.

## Render the demo
Use Python 3.11 for the lightweight plotting environment.
```sh
python -m venv .venv
# Activate: Windows .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/render_maps.py --input data/demo.json --output results/demo --disease "Example disease"
```
The included demo is synthetic, not research evidence. Rendering writes nine figures in PNG, SVG and PDF, plus a manifest with input hash and environment information. Use `--font Arial` if installed; the default is DejaVu Sans.

## Reproduce research figures
Obtain authorized data separately and place it in the ignored `private_data/` directory.
```sh
python scripts/render_maps.py --input private_data/evidence_overlay.json --output results/research --disease "Colorectal cancer" --font Arial
python scripts/candidate_evidence.py --cache private_data/inference_cache.npz --annotations private_data/evidence_overlay.json --out results/candidate
```
This renderer reuses cached coordinates; it does not recompute t-SNE. Focus coordinates and annotation memberships must match the main table exactly.

## Research models
See [research setup and scope](research/README.md). The historical graph stack uses a separate Python 3.10 CPU environment. Checkpoint revision and SHA-256 are in [model configuration](research/examples/openpom_checkpoint_config.json). Model weights and original DB records are not bundled. Frozen inference was rechecked on Windows; Apple Silicon remains unverified on target hardware.

## Interpretation
Coordinates come from OpenPOM embeddings. Odor colors represent Leffingwell-derived known labels, while disease colors represent archived HMDB links. Predicted descriptor scores are a separate evidence layer. Density is normalized separately per odor: color darkness is not measured intensity. Neither 2D proximity nor disease annotation establishes perceptual equivalence, causation, or diagnostic performance.

[Methods](docs/METHODS.md) · [Sources and redistribution](docs/DATA_SOURCES.md) · [Validation](docs/VALIDATION.json) · [한국어 안내](README.ko.md)

Code licensing has not yet been assigned. Third-party data/model terms remain separate. Keep private datasets and weights out of commits.

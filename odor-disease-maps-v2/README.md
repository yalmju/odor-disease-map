# Odor–disease molecular maps

Reproducible figure rendering for an optical-nose research poster. Overlay known odor labels and archived disease associations on **the same fixed molecular coordinates**, retaining focal compounds 1 and 2.

## Quick start

Python 3.11 is the reference environment. No GPU, PyTorch, browser, or proprietary presentation runtime is needed.

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/render_maps.py --input data/demo.json --output results/demo --disease "Example disease"
python -m unittest discover -s tests
```

**The included demo is synthetic. It contains no measured molecular or disease evidence and its coordinates are not t-SNE.**

For the research snapshot, obtain your authorized local evidence_overlay.json separately:

```sh
python scripts/render_maps.py --input private_data/evidence_overlay.json --output results/research --disease "Colorectal cancer" --font Arial
```

Nine figures are exported in PNG (300 dpi), SVG, and PDF: six odor facets, their combined panel, the disease map with enlarged 1/2 markers, and the side-by-side comparison. SVG/PDF retain vector points and labels; gradient layers are rasterized. `render_manifest.json` records input SHA-256, counts, styling parameters, and package versions. Arial must be locally installed; the portable default is DejaVu Sans.

## What the research figures represent

- **Coordinates:** cached 2D t-SNE projection of frozen pretrained OpenPOM 256D molecular embeddings. This repository does not recompute embeddings or t-SNE.
- **Odor colors:** Leffingwell-derived known odor annotations, not OpenPOM predictions. Six displayed labels can overlap.
- **Disease colors:** exact-identity links from a locally archived HMDB annotation table, not diagnostic predictions.
- **Focal markers:** larger markers for compounds 1/2, without changing their annotation colors or coordinates.
- **OpenPOM descriptor scores:** a separate 138-output model result used in the poster table. The private CSV is preserved separately. Scores are not measured intensities or calibrated probabilities.

See [methods](docs/METHODS.md) and [data sources](docs/DATA_SOURCES.md). The map renderer is a plotting/reproducibility tool. The v2 research module also includes existing frozen-encoder inference and downstream MLP training; it is not a new encoder, clinical classifier, or end-to-end SERS predictor.

## Input contract

JSON contains `points` and `focus`. Each point has unique integer `index`, finite `x`,`y`, and lists `odors`,`diseases`. `focus` contains at least two points copied exactly from `points`; only its first two entries are displayed. Optional `coordinate_source` and `synthetic` describe provenance. Extra molecular fields may remain in private files. Mismatched focus coordinates or memberships fail validation instead of being silently repaired. Missing odor categories render as gray backgrounds.

## GitHub upload

Upload the contents of this directory, including `.github`, but **not the separate private-data ZIP**. The public archive contains no original DB records, model weights, local absolute paths, or presentation runtime. Generated research figures and third-party data require their own redistribution review. No repository has been created or pushed by this package.

## Compatibility and licensing

NumPy/SciPy/Matplotlib CPU rendering is suitable for Windows, Linux, and macOS; no CUDA/MPS dependency is used. Windows execution was tested locally. Apple Silicon and GitHub Actions are provided as portable targets but have not been run here. GitHub does not imply an open-source license: a code license has deliberately not been assigned without the research owner's choice. Third-party data/model terms remain separate.

## Expanded research code (v2)

[Research module](research/README.md) restores frozen OpenPOM inference, MLP rating training/inference and the existing mixture-distance prototype. No trained SERS-map classifier or odor/SERS alignment model is claimed. The original figure-only scope above applies to scripts/render_maps.py.

Candidate evidence is computed directly from original embeddings:
```sh
python scripts/candidate_evidence.py --cache private_data/inference_cache.npz --annotations private_data/evidence_overlay.json --out results/candidate
```
The result is post-hoc support for the comparison pair, not a validated prospective selection rule.

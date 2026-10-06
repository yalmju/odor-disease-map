# Existing research modules

This folder restores previously developed research code omitted from the initial figure-only release. These are research implementations, not newly validated models.

- `src/voclib/pom_backend.py`: checkpoint-hash-verified, frozen OpenPOM CPU inference; 256D embeddings and 138 descriptors.
- `src/voclib/deep_odor.py`: learned perceptual rating head inference.
- `scripts/run_pom_mlp_pilot.py`: PyTorch MLP training/evaluation on a supplied molecule-grouped split, frozen OpenPOM vs Morgan inputs. Does not retrain the OpenPOM encoder.
- `scripts/strengthen_rating_models.py`: additional rating model comparisons with supplied cache/split.
- `src/voclib/odor_edit.py`: substitution enumeration and structure/odor comparisons.
- `src/voclib/features.py`, `model.py`, `evaluate.py`: mixture representation/distance research prototype. This is not a validated A+B perceptual descriptor predictor or SERS classifier.

SERS spatial-map classification and learned odor/SERS alignment are **not implemented in this package**. No placeholder results claim otherwise. Encoder training is not supplied. Upstream pretrained model: see examples/openpom_checkpoint_config.json for repository, pinned revision and checkpoint SHA-256. No weights or third-party database records are redistributed.

## Environment and inference

Use a separate Python 3.10 environment for the historical graph-model stack. The lightweight map renderer can retain its own environment. Install graph dependencies from the relevant docs lock file, then the pinned upstream OpenPOM source named in the config (review its own installation instructions and terms). Install this module with `python -m pip install -e research` from repository root. Apple Silicon requires an actual target-machine import/forward check; CPU operation is intentional and MPS is not used. Historical lock files are not a claim that installation was retested on macOS.

```sh
python research/scripts/infer_openpom.py --checkpoint PATH_TO_checkpoint2.pt --config research/examples/openpom_checkpoint_config.json --smiles-file INPUT_SMILES.txt --out results/inference.npz
python research/scripts/run_pom_mlp_pilot.py --help
```

SMILES input has one molecule per line. The NPZ is compatible with scripts/candidate_evidence.py. Training additionally requires authorized Keller data and the original grouped split manifest; see script arguments and split schema in source. Existing local data/model artifacts are not made public by copying this source.

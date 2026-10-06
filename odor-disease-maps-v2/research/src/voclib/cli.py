import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from . import __version__
from .data import arrays, load_dataset
from .evaluate import distance_matrix, scores, tsne
from .features import CHANNELS, FEATURE_VERSION
from .model import DistanceFusion


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run(input_path, output, ridge=0.05, embed=False, seed=42, perplexity=30):
    doc, mixtures, audit = load_dataset(input_path)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is nonempty; choose a new run directory")
    out.mkdir(parents=True, exist_ok=True)
    train = [p for p in doc["pairs"] if p["split"] == "train"]
    x, y = arrays(train, mixtures)
    model = DistanceFusion(ridge).fit(x, y)
    model.save(out / "model.json")
    evaluation, predictions = {}, []
    for split in ["test", "validation"]:
        rows = [p for p in doc["pairs"] if p["split"] == split]
        if not rows:
            continue
        xx, yy = arrays(rows, mixtures); pred = model.predict(xx)
        evaluation[split] = {"fusion": scores(yy, pred), "constant_training_mean": scores(yy, np.full(len(yy), y.mean())),
                             "unfitted_fingerprint_chord_spearman": scores(yy, xx[:, 0])["spearman"]}
        predictions += [{"id": p["id"], "split": split, "target": float(t), "prediction": float(v)} for p, t, v in zip(rows, yy, pred)]
    report = {"package_version": __version__, "feature_version": FEATURE_VERSION,
              "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(), "audit": audit,
              "config": {"ridge": ridge, "seed": seed, "perplexity_requested": perplexity},
              "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn", "rdkit"]}},
              "evaluation": evaluation,
              "limitations": ["No validated SERS or clinical model", "No novelty claim", "No test-score-based tuning", "Pair rows may share mixtures and are not independent uncertainty units", "Predictions are not clipped to the rating scale", "Chemical identity and concentration affect representation; no gas capture simulation"]}
    if embed:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ids = sorted(mixtures)
        d = distance_matrix([mixtures[key] for key in ids], model)
        xy, diagnostics = tsne(d, seed, perplexity)
        report["embedding"] = diagnostics
        np.savez_compressed(out / "embedding.npz", ids=np.array(ids, dtype=str), distances=d, coordinates=xy)
        write_json(out / "embedding.json", [{"mixture_id": key, "x": float(pos[0]), "y": float(pos[1])} for key, pos in zip(ids, xy)])
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        nodes = {s: {p[k] for p in doc["pairs"] if p["split"] == s for k in ["a", "b"]} for s in ["train", "test", "validation"]}
        for split, color in [("train", "#6b818a"), ("test", "#db8151"), ("validation", "#157f84")]:
            ix = [i for i, key in enumerate(ids) if key in nodes[split]]
            if ix:
                ax.scatter(xy[ix, 0], xy[ix, 1], c=color, s=25, alpha=.8, label=split)
        label = "SYNTHETIC DEMO" if doc["evidence_origin"] == "synthetic_demo" else "Mixture distance fusion"
        ax.set(title=label + " / exploratory t-SNE", xlabel="t-SNE 1 (arbitrary)", ylabel="t-SNE 2 (arbitrary)")
        ax.legend(); fig.savefig(out / "embedding.png", dpi=160); plt.close(fig)
    write_json(out / "report.json", report)
    write_json(out / "predictions.json", predictions)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible mixture-distance experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate"); validate.add_argument("input", type=Path)
    train = sub.add_parser("run"); train.add_argument("input", type=Path); train.add_argument("--out", type=Path, required=True)
    train.add_argument("--ridge", type=float, default=.05); train.add_argument("--embed", action="store_true")
    train.add_argument("--seed", type=int, default=42); train.add_argument("--perplexity", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = load_dataset(args.input)[2]
        else:
            result = run(args.input, args.out, args.ridge, args.embed, args.seed, args.perplexity)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

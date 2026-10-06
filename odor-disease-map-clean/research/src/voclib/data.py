"""Input validation and composition-level leakage checks."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .features import mixture, pair_channels


def load_dataset(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema_version") != 1 or doc.get("evidence_origin") not in {"measured", "synthetic_demo"}:
        raise ValueError("Require schema_version=1 and explicit evidence_origin")
    molecules = doc["molecules"]
    if not doc["mixtures"]:
        raise ValueError("At least one mixture required")
    mixes = {}
    for key, row in doc["mixtures"].items():
        comps = row["components"]
        if not comps or any(c["molecule_id"] not in molecules for c in comps):
            raise ValueError(f"Unresolved components: {key}")
        basis = row.get("weight_basis", "equal")
        weights = [c.get("weight") for c in comps] if basis == "mole_fraction" else None
        if basis == "equal" and any("weight" in c for c in comps):
            raise ValueError("Explicit weights cannot be silently ignored")
        mixes[key] = mixture([molecules[c["molecule_id"]] for c in comps], weights, basis)
    if len({m.weight_basis for m in mixes.values()}) != 1:
        raise ValueError("A run must use one weight basis")
    pairs = doc["pairs"]
    if not pairs or len({p["id"] for p in pairs}) != len(pairs):
        raise ValueError("Pair IDs must be unique and nonempty list required")
    split_nodes = defaultdict(set); split_pairs = defaultdict(list)
    for p in pairs:
        if p["split"] not in {"train", "test", "validation"}:
            raise ValueError("Use explicit train, test or validation split")
        if p["a"] not in mixes or p["b"] not in mixes:
            raise ValueError(f"Unknown mixture in pair {p['id']}")
        if not np.isfinite(p["distance"]) or p["distance"] < 0:
            raise ValueError("Target distance must be finite and nonnegative")
        a, b = mixes[p["a"]].composition_signature, mixes[p["b"]].composition_signature
        split_nodes[p["split"]].update([a, b]); split_pairs[p["split"]].append(tuple(sorted([a, b])))
    if len(split_pairs["train"]) < 2:
        raise ValueError("At least two training pairs required")
    for split in ["test", "validation"]:
        if split_nodes["train"] & split_nodes[split]:
            raise ValueError(f"Composition leakage: train and {split} share a mixture, including aliases/ratios")
    audit = {"evidence_origin": doc["evidence_origin"], "pairs_per_split": dict(Counter(p["split"] for p in pairs)),
             "train_heldout_composition_overlap": 0,
             "test_validation_composition_overlap": len(split_nodes["test"] & split_nodes["validation"]),
             "duplicate_unordered_pairs_within_split": {s: len(v) - len(set(v)) for s, v in split_pairs.items()},
             "weight_basis": next(iter(mixes.values())).weight_basis,
             "scope": "Composition-disjoint from train; not scaffold-, study- or subject-disjoint"}
    return doc, mixes, audit


def arrays(pairs, mixtures):
    return np.stack([pair_channels(mixtures[p["a"]], mixtures[p["b"]]) for p in pairs]), np.array([p["distance"] for p in pairs], float)


def aggregate_ratings(rows):
    """Equal participant weighting after repeats, treating A/B and B/A as one pair.

    Each row: participant, a, b, value. Do not mix studies or scales in one call.
    """
    within = defaultdict(list)
    for r in rows:
        if not np.isfinite(r["value"]):
            raise ValueError("Nonfinite human rating")
        pair = tuple(sorted([r["a"], r["b"]]))
        within[(r["participant"], pair)].append(r["value"])
    across = defaultdict(list)
    for (_, pair), values in within.items():
        across[pair].append(float(np.mean(values)))
    return [{"a": pair[0], "b": pair[1], "distance": float(np.mean(values)), "participants": len(values)} for pair, values in sorted(across.items())]

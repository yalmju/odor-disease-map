"""Annotation enrichment, conditional on the observed annotation universe."""
import numpy as np
from scipy.stats import fisher_exact


def bh_adjust(pvalues):
    p = np.asarray(pvalues, float)
    if not len(p): return []
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any(): raise ValueError("Invalid p values")
    order = np.argsort(p); ranked = p[order] * len(p) / np.arange(1, len(p)+1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    out = np.empty_like(p); out[order] = adjusted
    return out.tolist()


def enrichment(groups, annotations, observed):
    """One-sided Fisher: selected group vs rest, BH across all group-label tests.

    Unobserved molecules never become negative controls. For binary odor labels,
    absence means 'not annotated', not an experimentally confirmed negative.
    """
    groups = np.asarray(groups); observed = np.asarray(observed, bool)
    if len(groups) != len(annotations) or len(groups) != len(observed): raise ValueError("Unaligned annotations")
    universe = int(observed.sum())
    labels = sorted(set(label for i, row in enumerate(annotations) if observed[i] for label in row))
    rows = []
    for group in sorted(set(groups.tolist())):
        inside = observed & (groups == group); outside = observed & (groups != group)
        n, rest = int(inside.sum()), int(outside.sum())
        for label in labels:
            marked = np.array([label in row for row in annotations])
            a, c = int((inside & marked).sum()), int((outside & marked).sum())
            population = a + c
            rate = a / n if n else None; base = population / universe if universe else None
            p = float(fisher_exact([[a, n-a], [c, rest-c]], alternative="greater").pvalue) if n and rest else 1.0
            rows.append({"group": int(group), "label": label, "count": a, "n": n, "rest_count": c, "rest_n": rest,
                         "background_count": population, "background_n": universe, "fraction": rate,
                         "background_fraction": base, "fold": rate/base if n and base else None, "p": p})
    for row, q in zip(rows, bh_adjust([r["p"] for r in rows])): row["q"] = q
    return rows

"""Grouped validation for multi-channel SERS reporter responses.

Answers three questions a reviewer asks about a per-pixel channel table:

1. Does classification accuracy survive when whole maps or whole films are
   held out, instead of random pixel splits that leak film-level offsets?
2. Where does each channel's spread come from: between films, between maps
   within a film, or between pixels within a map?
3. Do channels that overlap one at a time separate the analytes jointly?

Input is a CSV with one row per pixel (or spot) and the columns
``film``, ``map``, ``analyte`` plus one numeric column per channel.
Nothing else is required. The classifier is a ridge-regularised linear
discriminant fitted on standardised channels, implemented with NumPy so the
lightweight plotting environment is enough. It is a validation tool, not a
proposed production classifier.

Example::

    python research/scripts/channel_validation.py --demo --out results/channel_demo
    python research/scripts/channel_validation.py --input private_data/pixels.csv \
        --channels MPY,MBA,NT --out results/channel_validation
"""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

ID_COLUMNS = ("film", "map", "analyte")


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def read_table(path, channels=None):
    """Read the pixel table. Returns (ids dict of arrays, X, channel names)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: no rows")
    missing = [c for c in ID_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; need {ID_COLUMNS}")
    if channels is None:
        channels = [c for c in rows[0] if c not in ID_COLUMNS]
    if not channels:
        raise ValueError("no channel columns found")
    ids = {c: np.array([r[c] for r in rows]) for c in ID_COLUMNS}
    try:
        x = np.array([[float(r[c]) for c in channels] for r in rows], dtype=float)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"channel columns must be numeric: {exc}") from exc
    if not np.isfinite(x).all():
        raise ValueError("non-finite channel values")
    return ids, x, list(channels)


def write_table(path, ids, x, channels):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(ID_COLUMNS) + channels)
        for i in range(len(x)):
            w.writerow([ids[c][i] for c in ID_COLUMNS] + [f"{v:.6g}" for v in x[i]])


def synthetic_table(seed=0, films=6, maps_per_film=3, pixels=50, film_sd=0.12):
    """Three analytes, three channels. Film-level offsets are deliberately large
    so a random pixel split looks better than a held-out-film split."""
    rng = np.random.default_rng(seed)
    analytes = ("indole_A", "indole_B", "indole_C")
    channels = ["MPY", "MBA", "NT"]
    # class means: MBA responds to all three alike (presence, not identity);
    # MPY separates B from A/C and NT separates C from A/B, so only the joint
    # pattern separates all three. That is the combinatorial-code case.
    means = {"indole_A": (1.20, 0.19, 0.60), "indole_B": (0.60, 0.19, 0.60), "indole_C": (1.20, 0.19, 1.00)}
    within = np.array([0.15, 0.02, 0.15])
    ids = {c: [] for c in ID_COLUMNS}
    rows = []
    film_id = 0
    for analyte in analytes:
        for _ in range(films):
            film_id += 1
            film_offset = rng.normal(0, film_sd, 3) * within / within.max()
            for m in range(maps_per_film):
                map_offset = rng.normal(0, 0.3, 3) * within
                for _ in range(pixels):
                    v = np.array(means[analyte]) + film_offset + map_offset + rng.normal(0, 1, 3) * within
                    rows.append(v)
                    ids["film"].append(f"film{film_id:02d}")
                    ids["map"].append(f"film{film_id:02d}_map{m + 1}")
                    ids["analyte"].append(analyte)
    return {c: np.array(v) for c, v in ids.items()}, np.array(rows), channels


# ----------------------------------------------------------------------------
# classifier: standardised ridge LDA
# ----------------------------------------------------------------------------
class RidgeLDA:
    """Linear discriminant on standardised channels with uniform class priors."""

    def __init__(self, ridge=1e-2):
        self.ridge = float(ridge)

    def fit(self, x, y):
        x = np.asarray(x, float)
        self.classes_ = np.unique(y)
        self.mu_ = x.mean(0)
        self.sd_ = x.std(0) + 1e-12
        z = (x - self.mu_) / self.sd_
        self.means_ = np.array([z[y == c].mean(0) for c in self.classes_])
        pooled = sum(np.cov(z[y == c].T, bias=True) * (y == c).sum() for c in self.classes_) / len(z)
        pooled = np.atleast_2d(pooled) + self.ridge * np.eye(z.shape[1])
        self.prec_ = np.linalg.inv(pooled)
        # Uniform priors: with leave-one-group-out, the held-out class is the
        # one with fewer training rows, and empirical priors would then bias
        # every prediction away from it.
        self.prior_ = np.full(len(self.classes_), 1.0 / len(self.classes_))
        return self

    def predict(self, x):
        z = (np.asarray(x, float) - self.mu_) / self.sd_
        scores = []
        for k, m in enumerate(self.means_):
            w = self.prec_ @ m
            scores.append(z @ w - 0.5 * m @ w + np.log(self.prior_[k]))
        return self.classes_[np.argmax(np.stack(scores, 1), 1)]


def leave_one_group_out(x, y, groups, ridge=1e-2):
    """Hold out every group once. Returns predicted labels aligned to rows."""
    pred = np.empty_like(y)
    for g in np.unique(groups):
        test = groups == g
        train = ~test
        if len(np.unique(y[train])) < 2:
            raise ValueError(f"holding out group {g!r} leaves fewer than two classes to train on")
        pred[test] = RidgeLDA(ridge).fit(x[train], y[train]).predict(x[test])
    return pred


def random_pixel_split(x, y, folds=5, seed=0, ridge=1e-2):
    """Leaky baseline: pixels of the same map land in train and test."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    pred = np.empty_like(y)
    for f in range(folds):
        test_idx = order[f::folds]
        test = np.zeros(len(y), bool)
        test[test_idx] = True
        pred[test] = RidgeLDA(ridge).fit(x[~test], y[~test]).predict(x[test])
    return pred


def confusion(y, pred, classes):
    m = np.zeros((len(classes), len(classes)), int)
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y, pred):
        m[idx[t], idx[p]] += 1
    return m


def summarise(y, pred, classes):
    m = confusion(y, pred, classes)
    recall = m.diagonal() / np.maximum(m.sum(1), 1)
    return {
        "accuracy": float((y == pred).mean()),
        "balanced_accuracy": float(recall.mean()),
        "per_class_recall": {c: float(r) for c, r in zip(classes, recall)},
        "confusion_rows_true_cols_pred": m.tolist(),
    }


# ----------------------------------------------------------------------------
# variance decomposition
# ----------------------------------------------------------------------------
def variance_decomposition(v, film, map_):
    """Fractions of total sum of squares for one channel within one analyte:
    between films, between maps within film, within map (pixels)."""
    v = np.asarray(v, float)
    grand = v.mean()
    total = ((v - grand) ** 2).sum()
    if total <= 0:
        return {"between_film": 0.0, "between_map_within_film": 0.0, "within_map": 0.0, "n": int(len(v))}
    ss_film = ss_map = 0.0
    for f in np.unique(film):
        sel = film == f
        fm = v[sel].mean()
        ss_film += sel.sum() * (fm - grand) ** 2
        for m in np.unique(map_[sel]):
            sm = sel & (map_ == m)
            ss_map += sm.sum() * (v[sm].mean() - fm) ** 2
    ss_within = total - ss_film - ss_map
    return {
        "between_film": float(ss_film / total),
        "between_map_within_film": float(ss_map / total),
        "within_map": float(ss_within / total),
        "n": int(len(v)),
        "n_films": int(len(np.unique(film))),
        "n_maps": int(len(np.unique(map_))),
    }


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def plot_confusions(results, classes, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.4), squeeze=False)
    for ax, name in zip(axes[0], names):
        m = np.array(results[name]["confusion_rows_true_cols_pred"], float)
        norm = m / np.maximum(m.sum(1, keepdims=True), 1)
        ax.imshow(norm, vmin=0, vmax=1, cmap="magma")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="black" if norm[i, j] > 0.6 else "white", fontsize=9)
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}\nacc {results[name]['accuracy']:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_channel_scatter(x, y, channels, path, max_points=4000, seed=0):
    """Pairwise channel scatter per pixel coloured by analyte, with marginal
    boxes per channel so single-channel overlap and joint separation sit side
    by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    if len(y) > max_points:
        keep = rng.choice(len(y), max_points, replace=False)
        x, y = x[keep], y[keep]
    classes = list(np.unique(y))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[: len(classes)]
    pairs = list(combinations(range(len(channels)), 2)) or [(0, 0)]
    n = len(channels) + len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.3))
    axes = np.atleast_1d(axes)
    for k, ch in enumerate(channels):
        ax = axes[k]
        data = [x[y == c, k] for c in classes]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(classes) + 1), classes, rotation=30, ha="right")
        ax.set_title(f"{ch} alone", fontsize=10)
        ax.set_ylabel("response")
    for k, (i, j) in enumerate(pairs, start=len(channels)):
        ax = axes[k]
        for c, col in zip(classes, colors):
            sel = y == c
            ax.scatter(x[sel, i], x[sel, j], s=6, alpha=0.5, color=col, label=c, linewidths=0)
        ax.set_xlabel(channels[i])
        ax.set_ylabel(channels[j])
        ax.set_title(f"{channels[i]} vs {channels[j]}", fontsize=10)
    axes[len(channels)].legend(fontsize=8, markerscale=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def run(ids, x, channels, out, ridge=1e-2, seed=0, make_plots=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    y = ids["analyte"]
    classes = list(np.unique(y))
    if len(classes) < 2:
        raise ValueError("need at least two analytes")

    results = {"random_pixel_split": summarise(y, random_pixel_split(x, y, seed=seed, ridge=ridge), classes)}
    for level in ("map", "film"):
        groups = ids[level]
        if len(np.unique(groups)) < 2:
            continue
        try:
            results[f"leave_one_{level}_out"] = summarise(y, leave_one_group_out(x, y, groups, ridge), classes)
        except ValueError as exc:
            results[f"leave_one_{level}_out"] = {"skipped": str(exc)}

    single = {}
    for k, ch in enumerate(channels):
        groups = ids["film"] if len(np.unique(ids["film"])) > 1 else ids["map"]
        try:
            single[ch] = summarise(y, leave_one_group_out(x[:, [k]], y, groups, ridge), classes)["accuracy"]
        except ValueError as exc:
            single[ch] = str(exc)

    decomposition = {}
    for c in classes:
        sel = y == c
        decomposition[c] = {
            ch: variance_decomposition(x[sel, k], ids["film"][sel], ids["map"][sel]) for k, ch in enumerate(channels)
        }

    report = {
        "n_rows": int(len(y)),
        "classes": classes,
        "channels": channels,
        "counts": {
            "films_per_analyte": {c: int(len(np.unique(ids["film"][y == c]))) for c in classes},
            "maps_per_analyte": {c: int(len(np.unique(ids["map"][y == c]))) for c in classes},
            "pixels_per_analyte": {c: int((y == c).sum()) for c in classes},
        },
        "classifier": f"standardised ridge LDA (ridge={ridge})",
        "validation": results,
        "single_channel_grouped_accuracy": single,
        "variance_fraction_by_analyte_channel": decomposition,
        "note": "random_pixel_split is the leaky baseline; report leave_one_film_out where films >= 2 per analyte, else leave_one_map_out.",
    }
    (out / "channel_validation.json").write_text(json.dumps(report, indent=2))
    if make_plots:
        plot_confusions({k: v for k, v in results.items() if "accuracy" in v}, classes, out / "confusion_by_split.png")
        plot_channel_scatter(x, y, channels, out / "channel_scatter.png", seed=seed)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="CSV with film,map,analyte + channel columns")
    src.add_argument("--demo", action="store_true", help="use a synthetic table with deliberate film offsets")
    ap.add_argument("--channels", help="comma-separated channel columns (default: all non-id columns)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    if a.demo:
        ids, x, channels = synthetic_table(a.seed)
        a.out.mkdir(parents=True, exist_ok=True)
        write_table(a.out / "synthetic_pixels.csv", ids, x, channels)
    else:
        ids, x, channels = read_table(a.input, a.channels.split(",") if a.channels else None)
    report = run(ids, x, channels, a.out, a.ridge, a.seed, not a.no_plots)
    v = report["validation"]
    print(f"rows {report['n_rows']}  classes {report['classes']}  channels {report['channels']}")
    for name, r in v.items():
        print(f"  {name:>22}: " + (f"acc {r['accuracy']:.3f}  balanced {r['balanced_accuracy']:.3f}" if "accuracy" in r else r["skipped"]))
    print("  single-channel grouped accuracy: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} n/a" for k, v in report["single_channel_grouped_accuracy"].items()))
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()

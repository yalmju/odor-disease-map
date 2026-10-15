"""Shared numeric helpers for the colorectal-cancer (CRC) map expansion scripts.

Pure numpy so they can be unit-tested without private data or the OpenPOM environment.
"""
import numpy as np


def unit_rows(x):
    """L2-normalise rows; refuses zero or non-finite rows."""
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or (n == 0).any():
        raise ValueError('Embeddings must be finite and non-zero')
    return x / n


def knn_place(query, ref_unit, ref_xy, k=5, exclude=None):
    """Place one embedding on fixed 2-D coordinates.

    Returns the cosine-similarity-weighted mean of the k most similar reference
    coordinates, plus the neighbour indices and similarities. Negative
    similarities are clipped so they never pull a point away.
    """
    sim = ref_unit @ (query / np.linalg.norm(query))
    if exclude is not None:
        sim = sim.copy()
        sim[exclude] = -np.inf
    nn = np.argsort(-sim)[:k]
    w = np.clip(sim[nn], 1e-6, None)
    return (ref_xy[nn] * w[:, None]).sum(0) / w.sum(), nn, sim[nn]


def loo_errors(ref_unit, ref_xy, k=5):
    """Leave-one-out displacement of the placement rule on the reference map itself."""
    return np.array([np.linalg.norm(knn_place(ref_unit[i], ref_unit, ref_xy, k, exclude=i)[0] - ref_xy[i])
                     for i in range(len(ref_xy))])


def discordant_pairs(sim, linked, eligible=None):
    """All (linked, unlinked) index pairs sorted by similarity, highest first.

    sim: square similarity matrix; linked: boolean disease-link mask;
    eligible: optional boolean mask both members must satisfy.
    """
    linked = np.asarray(linked, bool)
    ok = np.ones(len(linked), bool) if eligible is None else np.asarray(eligible, bool)
    a, b = np.where(linked & ok)[0], np.where(~linked & ok)[0]
    ii, jj = np.meshgrid(a, b, indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    s = sim[ii, jj]
    order = np.argsort(-s, kind='stable')
    return ii[order], jj[order], s[order]


def neighbor_overlap(sim_a, sim_b, k=10):
    """Mean fraction of shared k nearest neighbours between two similarity matrices."""
    def nn(m):
        m = np.array(m, dtype=float)
        np.fill_diagonal(m, -np.inf)
        return np.argsort(-m, axis=1)[:, :k]
    na, nb = nn(sim_a), nn(sim_b)
    return float(np.mean([len(set(na[i]) & set(nb[i])) / k for i in range(len(na))]))


def kde_field(points_xy, grid_xy, shape, bw=2.6):
    """Isotropic Gaussian KDE on a grid, normalised to its own maximum (render_maps.py rule)."""
    z = np.zeros(len(grid_xy))
    for p in np.asarray(points_xy, float):  # one point at a time keeps memory at grid size
        z += np.exp(-((grid_xy - p) ** 2).sum(1) / (2 * bw * bw))
    z = z.reshape(shape)
    return z / z.max() if z.max() > 0 else z

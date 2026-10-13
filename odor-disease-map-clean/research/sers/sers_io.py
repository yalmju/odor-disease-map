"""Readers for the two raw formats used so far.

Point files: <channel>_<analyte>.txt, tab-separated, first column Raman shift, one
column per point (wide), 2001 rows. Map files: <name>.csv and <name>(bg).csv exported
by the mapping software: two header lines ("X num,10" / "Y num,10"), a header row
"X,Y,<shift...>" and one row per pixel.
"""
import csv
import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d


def baseline(y, win=151):
    """Rolling-minimum baseline (win points = about 257 cm-1 at 1.7 cm-1 per point)."""
    return uniform_filter1d(minimum_filter1d(y, win), win)


def read_points(path, remove_baseline=True, smooth=0):
    a = np.loadtxt(path)
    w = a[:, 0]; Y = a[:, 1:].T
    if remove_baseline:
        Y = np.array([y - baseline(y) for y in Y])
    if smooth and smooth > 1:
        Y = np.array([np.convolve(y, np.ones(smooth) / smooth, 'same') for y in Y])
    return w, Y


def read_map(path):
    rows = list(csv.reader(open(path)))
    hdr = rows[2]; w = np.array([float(v) for v in hdr[2:] if v != ''])
    xy, data = [], []
    for r in rows[3:]:
        if len(r) < 3 or r[0] == '':
            continue
        vals = [float(v) for v in r[2:] if v != '']
        if len(vals) != len(w):
            continue
        xy.append((float(r[0]), float(r[1]))); data.append(vals)
    return w, np.array(xy), np.array(data)


def band_height(w, y, centre, half=12):
    s = (w >= centre - half) & (w <= centre + half)
    return float(y[s].max())


def to_grid(xy, values):
    xs = np.unique(xy[:, 0]); ys = np.unique(xy[:, 1]); g = np.full((len(ys), len(xs)), np.nan)
    for (x, y), v in zip(xy, values):
        g[np.searchsorted(ys, y), np.searchsorted(xs, x)] = v
    return g

"""Per-pixel analysis of Raman maps: pixel ratio table for channel_validation.py,
NNLS unmixing on pure-map endmembers, share maps, residual bands.

    python map_unmix.py --root MAP_DIR --out OUT_DIR --pure ind-1,skt-1 --mix ind-skt2,ind-skt3 \
        [--channel mba] [--film-of ind-1=F1,skt-1=F1,...] [--range 600 1750]

Map names are the CSV base names without "(bg)"; the "(bg)" file is used when present.
--channel selects the band-ratio set from sers_config for the pixel table (omit for pristine).
Writes: pixel_ratios.csv (film,map,analyte + ratio columns; analyte = map name with a trailing
number removed, e.g. ind-1 -> ind, ind-skt2 -> ind-skt, unless --analyte-of is given), nnls_shares.json, nnls_share_maps.png,
nnls_share_box.png, nnls_mean_fits.png, nnls_residuals.png.
"""
import argparse, csv, json, re, sys
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sers_config import RATIOS
from sers_io import read_map, band_height, to_grid


def load(root, name):
    p = root / f'{name}(bg).csv'
    if not p.exists():
        p = root / f'{name}.csv'
    return read_map(p)


def parse_kv(s):
    return dict(kv.split('=') for kv in s.split(',')) if s else {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--pure', required=True, help='comma list of pure map names (endmembers)')
    ap.add_argument('--mix', default='', help='comma list of mixture map names')
    ap.add_argument('--channel', default='', help='band-ratio set: mba|mpy|nat (omit for pristine)')
    ap.add_argument('--film-of', default='', help='map=film,... (default: each map its own film)')
    ap.add_argument('--analyte-of', default='', help='map=analyte,... (default: map name with trailing number removed)')
    ap.add_argument('--range', type=float, nargs=2, default=(600, 1750)); ap.add_argument('--smooth', type=int, default=7)
    a = ap.parse_args(argv); a.out.mkdir(parents=True, exist_ok=True)
    pure = a.pure.split(','); mix = [m for m in a.mix.split(',') if m]; names = pure + mix
    film_of = parse_kv(a.film_of); analyte_of = parse_kv(a.analyte_of)
    sm = lambda v: np.convolve(v, np.ones(a.smooth) / a.smooth, 'same') if a.smooth > 1 else v
    M = {n: load(a.root, n) for n in names}; w = M[names[0]][0]; sel = (w >= a.range[0]) & (w <= a.range[1]); ws = w[sel]
    # pixel ratio table
    rset = RATIOS.get(a.channel, RATIOS['pristine'])
    with open(a.out / 'pixel_ratios.csv', 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['film', 'map', 'analyte', 'pixel'] + [f'R{n}/{d}' for n, d in rset])
        for n in names:
            _, xy, X = M[n]
            for i, x in enumerate(X, 1):
                wr.writerow([film_of.get(n, n), n, analyte_of.get(n, re.sub(r'[-_]?\d+$', '', n)), i] + [f'{band_height(w, x, p) / max(band_height(w, x, q), 1e-9):.5g}' for p, q in rset])
    # NNLS on pure-map means
    E = np.stack([sm(M[n][2].mean(0)[sel]) for n in pure] + [np.ones(sel.sum())], 1); area = E[:, :-1].clip(0).sum(0)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = {}; shares = {}
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.4), squeeze=False); axes = axes[0]
    for ax, n in zip(axes, names):
        _, xy, X = M[n]; Y = np.array([sm(x[sel]) for x in X]); C = np.array([nnls(E, y)[0] for y in Y]); fit = C @ E.T
        r2 = 1 - ((Y - fit) ** 2).sum(1) / ((Y - Y.mean(1, keepdims=True)) ** 2).sum(1)
        contrib = C[:, :-1] * area; share = contrib / np.maximum(contrib.sum(1, keepdims=True), 1e-9); shares[n] = share
        res[n] = {'share_median': {p: float(np.median(share[:, k])) for k, p in enumerate(pure)},
                  'share_iqr': {p: [float(np.percentile(share[:, k], 25)), float(np.percentile(share[:, k], 75))] for k, p in enumerate(pure)},
                  'fit_r2_median': float(np.median(r2)), 'n_pixels': int(len(Y))}
        im = ax.imshow(to_grid(xy, share[:, 0]), vmin=0, vmax=1, cmap='coolwarm_r', origin='lower'); ax.set_title(f'{n}\nshare of {pure[0]}: {np.median(share[:, 0]):.2f}', fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label=f'NNLS share of {pure[0]}'); fig.savefig(a.out / 'nnls_share_maps.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2, 4)); ax.boxplot([shares[n][:, 0] for n in names]); ax.set_xticks(range(1, len(names) + 1), names, rotation=20); ax.set_ylabel(f'share of {pure[0]} per pixel'); fig.tight_layout(); fig.savefig(a.out / 'nnls_share_box.png', dpi=150); plt.close(fig)
    if mix:
        fig, axes = plt.subplots(len(mix), 1, figsize=(11, 2.8 * len(mix)), sharex=True, squeeze=False); axes = axes[:, 0]
        R = []
        for ax, n in zip(axes, mix):
            y = sm(M[n][2].mean(0)[sel]); c = nnls(E, y)[0]; r = y - E @ c; R.append(r)
            ax.plot(ws, y, 'k', lw=1, label=f'{n} mean'); ax.plot(ws, E @ c, 'r', lw=1, label='NNLS fit')
            for k, p in enumerate(pure): ax.plot(ws, E[:, k] * c[k], lw=0.8, alpha=0.8, label=f'{p} x {c[k]:.2f}')
            r2 = 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum(); ax.set_title(f'{n}: map-mean fit R2 {r2:.2f}', fontsize=9); ax.legend(fontsize=7, ncol=4)
        axes[-1].set_xlabel('Raman shift (cm-1)'); fig.tight_layout(); fig.savefig(a.out / 'nnls_mean_fits.png', dpi=140); plt.close(fig)
        Rm = np.mean(R, 0); idx = np.argsort(-Rm); picked = []
        for i in idx:
            if all(abs(ws[i] - ws[j]) > 15 for j in picked): picked.append(i)
            if len(picked) >= 8: break
        res['_unexplained_mixture_bands_cm-1'] = sorted(float(ws[i]) for i in picked)
        fig, ax = plt.subplots(figsize=(11, 3.2)); ax.plot(ws, Rm, 'k', lw=1); ax.axhline(0, color='0.6'); ax.set_title('mean residual of mixtures (map mean - NNLS fit)'); ax.set_xlabel('Raman shift (cm-1)'); fig.tight_layout(); fig.savefig(a.out / 'nnls_residuals.png', dpi=140); plt.close(fig)
    json.dump(res, open(a.out / 'nnls_shares.json', 'w'), indent=1)
    for n in names: print(f"{n:10} share of {pure[0]} median {res[n]['share_median'][pure[0]]:.2f}  fit R2 {res[n]['fit_r2_median']:.2f}")
    print('written to', a.out)


if __name__ == '__main__':
    main()

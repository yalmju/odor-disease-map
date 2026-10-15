"""Draw OpenPOM-predicted odor facets on the same fixed coordinates.

Membership is a predicted descriptor score >= --threshold (a model prediction, not an observed
label). Scores come from inference_cache.npz for map molecules and new_embeddings.npz from
place_by_embedding.py for placed molecules, matched by SMILES to points_all.csv.

Example:
  python scripts/render_pom_facets.py --points results/crc_map/data/points_all.csv \
    --cache private_data/inference_cache.npz --new-scores results/crc_placement/new_embeddings.npz \
    --labels cheesy,sour,sweaty,fatty,sulfurous,phenolic --output results/crc_pom_facets --font Arial
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crc_tools import kde_field  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PALETTE = ['#E39A3B', '#4FAFB0', '#C98A5E', '#B5A26A', '#7FA33A', '#7A6FB0', '#D5BC35', '#C976A6', '#83B951', '#967CDA', '#639BCD', '#DE837D']
BG, INK = '#B9C1CB', '#24303B'


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--points', type=Path, required=True, help='points_all.csv from render_crc_expansion.py')
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--new-scores', type=Path, help='new_embeddings.npz from place_by_embedding.py')
    p.add_argument('--config', type=Path, default=ROOT / 'research/examples/openpom_checkpoint_config.json')
    p.add_argument('--labels', default='cheesy,sour,sweaty,fatty,sulfurous,phenolic', help='Comma list, up to 12')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', default='DejaVu Sans')
    a = p.parse_args()

    names = json.loads(a.config.read_text(encoding='utf-8'))['labels']
    chosen = [l.strip() for l in a.labels.split(',') if l.strip()]
    unknown = [l for l in chosen if l not in names]
    if unknown or not 0 < len(chosen) <= 12:
        raise SystemExit(f'Labels must be 1-12 of the 138 OpenPOM descriptors; unknown: {unknown}')
    with a.points.open(encoding='utf-8-sig') as f:
        pts = list(csv.DictReader(f))
    score = {}
    for path in [a.cache] + ([a.new_scores] if a.new_scores else []):
        z = np.load(path, allow_pickle=False)
        score.update(zip(z['smiles'].tolist(), z['descriptor_scores']))
    missing = [r['name'] for r in pts if r['smiles'] not in score]
    if missing:
        raise SystemExit(f'No OpenPOM scores for {len(missing)} points, e.g. {missing[:3]}')
    P = np.array([score[r['smiles']] for r in pts], float)
    xy = np.array([[float(r['x']), float(r['y'])] for r in pts])
    linked = np.array([r['disease_evidence'] != 'none' for r in pts])
    fpts = [r for r in pts if r['focus']]
    fpts.sort(key=lambda r: r['focus'])

    a.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': a.font, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'text.color': INK})
    bw = 2.6; lo = xy.min(0) - 5 * bw; hi = xy.max(0) + 5 * bw; extent = (lo[0], hi[0], lo[1], hi[1])
    gx, gy = np.linspace(lo[0], hi[0], 540), np.linspace(lo[1], hi[1], 420)
    X, Y = np.meshgrid(gx, gy); grid = np.c_[X.ravel(), Y.ravel()]

    ncol = min(3, len(chosen)); nrow = -(-len(chosen) // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4.2 * nrow), squeeze=False)
    summary = []
    for ax, lab, col in zip(axs.flat, chosen, PALETTE):
        j = names.index(lab)
        mask = P[:, j] >= a.threshold
        z = kde_field(xy[mask], grid, X.shape, bw)
        np.savetxt(a.output / f'kde_pom_{lab}_matrix.csv', np.vstack([np.r_[np.nan, gx], np.c_[gy, z]]), delimiter=',', fmt='%.5g')
        ax.scatter(*xy.T, s=7, c=BG, lw=0)
        layer = np.ones((*z.shape, 4)); layer[..., :3] = to_rgb(col); layer[..., 3] = .97 * z ** .65
        ax.imshow(layer, origin='lower', extent=extent, interpolation='bilinear')
        ax.scatter(*xy[mask].T, s=9, c=col, alpha=.75, lw=0)
        for i, f in enumerate(fpts[:2]):
            fx, fy = float(f['x']), float(f['y'])
            ax.scatter(fx, fy, s=28, facecolors='none', edgecolors=INK, lw=1.3, zorder=6)
            ax.annotate(f['focus'], (fx, fy), xytext=((-24, -29) if i == 0 else (25, 24)), textcoords='offset points', fontsize=15,
                        fontweight='bold', color=INK, ha='center', va='center', zorder=8,
                        arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.3, mutation_scale=11, shrinkA=3, shrinkB=4))
        ax.set_title(f'{lab}  (n={mask.sum()}, linked {int((mask & linked).sum())})', fontsize=15, pad=2)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect='equal'); ax.axis('off')
        summary.append({'label': lab, 'color': col, 'n_predicted': int(mask.sum()), 'n_linked_predicted': int((mask & linked).sum()),
                        'linked_mean_score': round(float(P[linked, j].mean()), 4), 'other_mean_score': round(float(P[~linked, j].mean()), 4)})
    for ax in list(axs.flat)[len(chosen):]:
        ax.axis('off')
    fig.subplots_adjust(left=.005, right=.995, bottom=.005, top=.95, wspace=.01, hspace=.12)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(a.output / f'pom_facets.{ext}', dpi=300, facecolor='white', bbox_inches='tight', pad_inches=.025)
    plt.close(fig)

    with (a.output / 'pom_label_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    with (a.output / 'pom_scores_all.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['row', 'name', 'smiles', 'x', 'y', 'disease_evidence', 'focus'] + [f'pom_{l}' for l in names])
        for r, s in zip(pts, P):
            w.writerow([r['row'], r['name'], r['smiles'], r['x'], r['y'], r['disease_evidence'], r['focus']] + [f'{v:.4f}' for v in s])
    # Which descriptors are relatively higher in linked molecules (helps choose --labels).
    lift = sorted(((P[linked, j].mean() / max(P[~linked, j].mean(), 1e-9), names[j], int((P[linked, j] >= a.threshold).sum()))
                   for j in range(len(names))), reverse=True)
    with (a.output / 'descriptor_lift.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['descriptor', 'linked_vs_other_mean_ratio', 'n_linked_predicted'])
        w.writerows([(n, f'{l:.3f}', k) for l, n, k in lift])
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

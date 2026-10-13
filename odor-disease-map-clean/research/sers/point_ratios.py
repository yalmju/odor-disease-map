"""Band-ratio features, group statistics, scatter panels and LOO confusion matrices
for point measurements (channel x analyte, N points each).

    python point_ratios.py --root DATA_DIR --out OUT_DIR [--channels mba,nat] [--analytes blank,ind,skt,mix]

Writes: point_features.csv, group_summary.csv, ratio_scatter.png, pure_confusion.csv,
pure_pca_scatter.png. Statistics: mean, SD, fold vs blank, Mann-Whitney p, LOO accuracy.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sers_config import RATIOS, PAIR, LABEL, COLOR
from sers_io import read_points, band_height
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from channel_validation import RidgeLDA


def loo(X, y, ridge=0.1):
    pred = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        pred.append(RidgeLDA(ridge).fit(X[m], y[m]).predict(X[i:i + 1])[0])
    return np.array(pred)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--channels', default='pristine,mba,mpy,nat'); ap.add_argument('--analytes', default='blank,ind,skt,mix')
    a = ap.parse_args(argv); a.out.mkdir(parents=True, exist_ok=True)
    chans = a.channels.split(','); ans = a.analytes.split(',')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    feat_rows, summ_rows, conf_rows = [], [], []
    fig, axes = plt.subplots(1, len(chans), figsize=(4.3 * len(chans), 4.1), squeeze=False); axes = axes[0]
    fig2, axes2 = plt.subplots(1, len(chans), figsize=(4.3 * len(chans), 4.1), squeeze=False); axes2 = axes2[0]
    for ax, ax2, ch in zip(axes, axes2, chans):
        F = {}
        for an in ans:
            p = a.root / f'{ch}_{an}.txt'
            if not p.exists():
                continue
            w, Y = read_points(p)
            F[an] = np.array([[band_height(w, y, n) / max(band_height(w, y, d), 1e-9) for n, d in RATIOS[ch]] for y in Y])
            for i, f in enumerate(F[an], 1):
                r = {'channel': ch, 'analyte': an, 'point': i}
                r.update({f'R{n}/{d}': round(v, 5) for (n, d), v in zip(RATIOS[ch], f)}); feat_rows.append(r)
        if 'blank' not in F:
            continue
        for k, (n, d) in enumerate(RATIOS[ch]):
            for an in F:
                v = F[an][:, k]; b = F['blank'][:, k]
                summ_rows.append({'channel': ch, 'feature': f'R{n}/{d}', 'analyte': an, 'n': len(v), 'mean': f'{v.mean():.5g}', 'sd': f'{v.std():.3g}',
                                  'fold_vs_blank': f'{v.mean() / b.mean():.3g}', 'p_vs_blank': f'{stats.mannwhitneyu(v, b).pvalue:.2g}' if an != 'blank' else '',
                                  'p_vs_ind': f'{stats.mannwhitneyu(v, F["ind"][:, k]).pvalue:.2g}' if 'ind' in F and an != 'ind' else '',
                                  'p_vs_skt': f'{stats.mannwhitneyu(v, F["skt"][:, k]).pvalue:.2g}' if 'skt' in F and an != 'skt' else ''})
        (n1, d1), (n2, d2) = PAIR[ch]; i1 = RATIOS[ch].index((n1, d1)); i2 = RATIOS[ch].index((n2, d2))
        for an in F:
            x, yv = F[an][:, i1], F[an][:, i2]
            ax.scatter(x, yv, s=34, color=COLOR.get(an, 'k'), edgecolor='white', linewidth=0.6, label=f'{LABEL.get(an, an)} (n={len(x)})', zorder=3)
            ax.errorbar(x.mean(), yv.mean(), xerr=x.std(), yerr=yv.std(), fmt='none', ecolor=COLOR.get(an, 'k'), elinewidth=1.2, capsize=3)
        ax.set_xlabel(f'I{n1} / I{d1}'); ax.set_ylabel(f'I{n2} / I{d2}'); ax.set_title(ch); ax.grid(alpha=0.2); ax.legend(fontsize=7, frameon=False)
        pure = [an for an in ('blank', 'ind', 'skt') if an in F]
        X = np.vstack([F[an] for an in pure]); y = np.array(sum([[an] * len(F[an]) for an in pure], []))
        if len(pure) >= 2:
            p = loo(X, y); acc = float((p == y).mean())
            for t in pure:
                for q in pure:
                    conf_rows.append({'channel': ch, 'features': 'band_ratios', 'validation': 'leave-one-out, one measurement set per class', 'true': t, 'predicted': q, 'count': int(((y == t) & (p == q)).sum()), 'accuracy': f'{acc:.2f}'})
            Z = (X - X.mean(0)) / (X.std(0) + 1e-12); U, Sv, Vt = np.linalg.svd(Z, full_matrices=False); ev = Sv ** 2 / np.sum(Sv ** 2); P = Z @ Vt[:2].T
            for an in pure:
                m = y == an; ax2.scatter(P[m, 0], P[m, 1], color=COLOR[an], s=36, edgecolor='white', linewidth=0.6, label=LABEL[an], zorder=3)
            if 'mix' in F:
                Pm = ((F['mix'] - X.mean(0)) / (X.std(0) + 1e-12)) @ Vt[:2].T
                ax2.scatter(Pm[:, 0], Pm[:, 1], facecolor='none', edgecolor=COLOR['mix'], s=36, linewidth=1.2, label='mixture (projected)')
            ax2.set_xlabel(f'PC1 ({ev[0] * 100:.0f}%)'); ax2.set_ylabel(f'PC2 ({ev[1] * 100:.0f}%)'); ax2.set_title(f'{ch}  LOO acc {acc:.2f}'); ax2.grid(alpha=0.2); ax2.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(a.out / 'ratio_scatter.png', dpi=200)
    fig2.suptitle('PCA of standardised band ratios; fitted on pure classes', fontsize=9); fig2.tight_layout(); fig2.savefig(a.out / 'pure_pca_scatter.png', dpi=200)
    for name, rows in (('point_features.csv', feat_rows), ('group_summary.csv', summ_rows), ('pure_confusion.csv', conf_rows)):
        if rows:
            keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('channel', 'analyte', 'point', 'feature', 'features', 'validation', 'true', 'predicted'), k))
            with open(a.out / name, 'w', newline='') as f:
                wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); [wr.writerow(r) for r in rows]
    print('written to', a.out)


if __name__ == '__main__':
    main()

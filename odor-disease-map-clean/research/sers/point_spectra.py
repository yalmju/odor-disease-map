"""Mean-spectrum comparison for point measurements: channel x analyte groups.

Reads every two-column .txt (Raman shift, intensity) under --root, assigns each
file to a channel and an analyte from keywords in its path (folder or file
name), and reports per group: mean +- SD, reporter-band position/height,
and differences against the channel's reference group (blank/pristine film
of that channel if present, otherwise the group named --ref).

Keywords (case-insensitive, first match in the path wins):
  channel: pristine|prs|bare, mba, mpy, nat|2nt|nt
  analyte: ind, skt, mix, blank|ref|none|air (reference, no odorant)
Override with --map "substring=channel/analyte" if names do not follow this.
"""
import argparse, json, re
from pathlib import Path
import numpy as np

CH = [('pristine', r'pristine|prs|bare|nofunc'), ('mba', r'mba'), ('mpy', r'mpy'), ('nat', r'nat|2nt|\bnt\b|naph')]
AN = [('blank', r'blank|ref\b|none|air|before|bare_film'), ('mix', r'mix'), ('ind', r'ind'), ('skt', r'skt|skat')]
# reporter bands to track (cm-1): name -> (centre, half-window)
BANDS = {
    'mba': {'1078 ring/C-S': (1078, 15), '1590 ring': (1590, 15), '1420 COO-': (1420, 20), '1700 C=O': (1700, 25)},
    'mpy': {'1004 ring br': (1004, 12), '1060': (1060, 12), '1095 ring br': (1095, 12), '1218': (1218, 12), '1580': (1580, 15), '1610': (1610, 15)},
    'nat': {'1064': (1064, 12), '1380': (1380, 15), '1580': (1580, 15), '1623': (1623, 15)},
    'pristine': {'760 indole': (760, 15), '1010 indole': (1010, 15), '1340': (1340, 15), '1490': (1490, 20), '1580': (1580, 15)},
}

def classify(path, overrides):
    s = str(path).lower()
    ch = an = None
    for sub, tag in overrides:
        if sub.lower() in s:
            c, a = tag.split('/')
            ch, an = (c or ch), (a or an)
    if ch is None:
        for name, pat in CH:
            if re.search(pat, s): ch = name; break
    if an is None:
        for name, pat in AN:
            if re.search(pat, s): an = name; break
    return ch, an

def read_spectra(p):
    """Return a list of (w, y). A wide file (shift + N intensity columns) yields N spectra."""
    try:
        a = np.loadtxt(p)
    except Exception:
        try:
            a = np.loadtxt(p, usecols=(0, 1))
        except Exception:
            return []
    if a.ndim != 2 or len(a) < 100 or a.shape[1] < 2: return []
    return [(a[:, 0], a[:, j]) for j in range(1, a.shape[1])]

def baseline(y, win=151):
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    return uniform_filter1d(minimum_filter1d(y, win), win)

def band_stats(w, y, centre, half):
    sel = (w >= centre - half) & (w <= centre + half)
    if sel.sum() < 3: return None
    i = np.argmax(y[sel]); ws, ys = w[sel], y[sel]
    # centroid around the max for sub-point position
    k = slice(max(i - 2, 0), i + 3); pos = float((ws[k] * ys[k].clip(0)).sum() / max(ys[k].clip(0).sum(), 1e-9))
    return dict(position=pos, height=float(ys[i]), area=float(np.trapz(ys.clip(0), ws)))

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--map', action='append', default=[], help='substring=channel/analyte override, e.g. "s1=mba/ind"')
    ap.add_argument('--range', type=float, nargs=2, default=(600, 1800))
    ap.add_argument('--no-baseline', action='store_true', help='skip rolling-minimum baseline removal')
    ap.add_argument('--ref', default='blank', help='analyte label used as the per-channel reference')
    a = ap.parse_args(argv)
    overrides = [tuple(m.split('=')) for m in a.map]
    groups = {}; unassigned = []
    for p in sorted(a.root.rglob('*.txt')):
        spectra = read_spectra(p)
        if not spectra: continue
        ch, an = classify(p.relative_to(a.root), overrides)
        if ch is None or an is None: unassigned.append(str(p.relative_to(a.root))); continue
        for sp in spectra: groups.setdefault((ch, an), []).append((p, sp))
    if not groups: raise SystemExit('no spectra assigned; check names or pass --map')
    a.out.mkdir(parents=True, exist_ok=True)
    # common grid
    w0 = groups[next(iter(groups))][0][1][0]
    lo, hi = a.range; grid = w0[(w0 >= lo) & (w0 <= hi)]
    G = {}
    for key, items in groups.items():
        Y = []
        for p, (w, y) in items:
            yy = np.interp(grid, w, y)
            if not a.no_baseline: yy = yy - baseline(yy)
            Y.append(yy)
        G[key] = np.array(Y)
    report = {'n_files': {f'{c}/{n}': int(len(v)) for (c, n), v in G.items()}, 'unassigned': unassigned, 'bands': {}, 'difference_vs_reference': {}}
    channels = sorted({c for c, _ in G}); analytes = sorted({n for _, n in G})
    for c in channels:
        report['bands'][c] = {}
        ref = G.get((c, a.ref))
        for n in analytes:
            if (c, n) not in G: continue
            Y = G[(c, n)]; mu = Y.mean(0)
            bs = {}
            for bname, (centre, half) in BANDS.get(c, BANDS['pristine']).items():
                per = [band_stats(grid, y, centre, half) for y in Y]; per = [b for b in per if b]
                if not per: continue
                bs[bname] = {k: dict(mean=float(np.mean([b[k] for b in per])), sd=float(np.std([b[k] for b in per]))) for k in ('position', 'height', 'area')}
                if ref is not None and n != a.ref:
                    rb = [band_stats(grid, y, centre, half) for y in ref]; rb = [b for b in rb if b]
                    if rb:
                        dpos = np.mean([b['position'] for b in per]) - np.mean([b['position'] for b in rb])
                        rh = np.mean([b['height'] for b in rb])
                        bs[bname]['shift_vs_ref_cm-1'] = float(dpos); bs[bname]['height_ratio_vs_ref'] = float(np.mean([b['height'] for b in per]) / rh) if rh else None
                        # effect size of height change across the 10 points
                        hp = np.array([b['height'] for b in per]); hr = np.array([b['height'] for b in rb])
                        bs[bname]['height_cohen_d'] = float((hp.mean() - hr.mean()) / np.sqrt((hp.var() + hr.var()) / 2 + 1e-12))
            report['bands'][c][n] = bs
            if ref is not None and n != a.ref:
                d = mu - ref.mean(0); dd = d / np.sqrt((Y.var(0) + ref.var(0)) / 2 + 1e-9)
                top = np.argsort(-np.abs(dd)); picked = []
                for i in top:
                    if all(abs(grid[i] - grid[j]) > 12 for j in picked): picked.append(i)
                    if len(picked) >= 8: break
                report['difference_vs_reference'][f'{c}/{n}'] = [dict(wn=float(grid[i]), d=float(dd[i]), delta=float(d[i])) for i in sorted(picked, key=lambda i: grid[i])]
    (a.out / 'point_spectra_report.json').write_text(json.dumps(report, indent=1))
    # figures: one panel per channel with mean +- SD per analyte, plus difference panel
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    cols = {'blank': '0.4', 'ind': 'tab:purple', 'skt': 'tab:red', 'mix': 'tab:green'}
    fig, axes = plt.subplots(len(channels), 2, figsize=(15, 3.4 * len(channels)), squeeze=False)
    for r, c in enumerate(channels):
        ax = axes[r, 0]; ref = G.get((c, a.ref))
        for n in analytes:
            if (c, n) not in G: continue
            Y = G[(c, n)]; mu, sd = Y.mean(0), Y.std(0)
            ax.plot(grid, mu, color=cols.get(n, 'k'), lw=1, label=f'{n} (n={len(Y)})'); ax.fill_between(grid, mu - sd, mu + sd, color=cols.get(n, 'k'), alpha=0.15)
        ax.set_title(f'{c}: mean ± SD'); ax.legend(fontsize=8); ax.set_ylabel('counts')
        ax2 = axes[r, 1]
        if ref is not None:
            for n in analytes:
                if (c, n) not in G or n == a.ref: continue
                ax2.plot(grid, G[(c, n)].mean(0) - ref.mean(0), color=cols.get(n, 'k'), lw=1, label=f'{n} − {a.ref}')
            ax2.axhline(0, color='0.6'); ax2.set_title(f'{c}: difference vs {a.ref}'); ax2.legend(fontsize=8)
        else:
            # no reference: differences between analytes
            base = analytes[0]
            for n in analytes[1:]:
                if (c, n) in G and (c, base) in G: ax2.plot(grid, G[(c, n)].mean(0) - G[(c, base)].mean(0), color=cols.get(n, 'k'), lw=1, label=f'{n} − {base}')
            ax2.axhline(0, color='0.6'); ax2.set_title(f'{c}: no reference group; differences vs {base}'); ax2.legend(fontsize=8)
        for bname, (centre, half) in BANDS.get(c, BANDS['pristine']).items():
            ax.axvspan(centre - half, centre + half, color='0.85', alpha=0.4, lw=0)
    axes[-1, 0].set_xlabel('Raman shift (cm$^{-1}$)'); axes[-1, 1].set_xlabel('Raman shift (cm$^{-1}$)')
    fig.tight_layout(); fig.savefig(a.out / 'point_spectra_overview.png', dpi=140)
    print('groups:', report['n_files']); print('unassigned:', len(unassigned)); print('written to', a.out)

if __name__ == '__main__':
    main()

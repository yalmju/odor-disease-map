"""Render the poster odor/disease map with extra disease-linked molecules and export Origin-ready data.

Inputs are the fixed map (evidence_overlay.json), a curated list of extra disease-linked molecules
(data/crc_additions_curated.csv), the positions from place_by_embedding.py for the ones not on the
map, and the pyrfume Leffingwell folder for observed odor labels. Every molecule is drawn as one
point; the original coordinates are unchanged. Style follows render_maps.py.

Example:
  python scripts/render_crc_expansion.py --overlay private_data/evidence_overlay.json \
    --additions data/crc_additions_curated.csv --placement results/crc_placement/placement.csv \
    --leffingwell private_data/leffingwell --output results/crc_map --font Arial
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
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crc_tools import kde_field  # noqa: E402
from hmdb_disease_list import load_molecules  # noqa: E402

LABS = ['fruity', 'floral', 'green', 'woody', 'roasted', 'animal']
COLS = ['#D5BC35', '#C976A6', '#83B951', '#967CDA', '#639BCD', '#DE837D']
LINK, LIT, BG, INK = '#BA497E', '#D9893A', '#B9C1CB', '#24303B'


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--overlay', type=Path, required=True)
    p.add_argument('--additions', type=Path, required=True, help='CSV: name, smiles, evidence (HMDB|literature), evidence_detail')
    p.add_argument('--placement', type=Path, required=True, help='placement.csv from place_by_embedding.py')
    p.add_argument('--leffingwell', type=Path, required=True, help='Folder with pyrfume leffingwell molecules.csv and behavior.csv')
    p.add_argument('--disease', default='Colorectal cancer')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', default='DejaVu Sans')
    a = p.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    key = lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s))
    fig_dir, data_dir = a.output / 'figures', a.output / 'data'
    fig_dir.mkdir(parents=True, exist_ok=True); data_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': a.font, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'text.color': INK})

    doc = json.loads(a.overlay.read_text(encoding='utf-8-sig'))
    leff = {r['inchikey']: r['odors'] for r in load_molecules(argparse.Namespace(leffingwell=a.leffingwell, overlay=None, molecules=None))}
    with a.additions.open(encoding='utf-8-sig') as f:
        additions = {key(r['smiles']): r for r in csv.DictReader(f)}
    with a.placement.open(encoding='utf-8-sig') as f:
        placed = {r['inchikey']: r for r in csv.DictReader(f)}

    rows = []
    for q in doc['points']:
        ev = []
        if a.disease in q['diseases']:
            ev.append('HMDB')
        extra = additions.get(q['inchikey'])
        if extra and extra['evidence'] not in ev:
            ev.append(extra['evidence'])
        rows.append({'name': q.get('name', str(q['index'])), 'smiles': q['smiles'], 'inchikey': q['inchikey'],
                     'x': q['x'], 'y': q['y'], 'placement': 'original t-SNE', 'odors': list(q['odors']),
                     'evidence': '+'.join(ev) or 'none', 'evidence_detail': extra['evidence_detail'] if extra else ''})
    on_map = {r['inchikey'] for r in rows}
    for ik, r in additions.items():
        if ik in on_map:
            continue
        if ik not in placed:
            raise ValueError(f"{r['name']} is not on the map and has no placement row")
        q = placed[ik]
        rows.append({'name': r['name'], 'smiles': r['smiles'], 'inchikey': ik, 'x': float(q['x']), 'y': float(q['y']),
                     'placement': f"OpenPOM cosine 5-NN (max cos {q['max_cos']})", 'odors': leff.get(ik, []),
                     'evidence': r['evidence'], 'evidence_detail': r['evidence_detail']})

    focus = {f['inchikey']: str(i + 1) for i, f in enumerate(doc['focus'][:2])}
    fields = ['row', 'name', 'smiles', 'inchikey', 'x', 'y', 'placement', 'observed_odor_labels', *LABS,
              'disease_evidence', 'evidence_detail', 'focus']
    out_rows = [{'row': i, 'name': r['name'], 'smiles': r['smiles'], 'inchikey': r['inchikey'], 'x': r['x'], 'y': r['y'],
                 'placement': r['placement'], 'observed_odor_labels': '|'.join(r['odors']),
                 **{l: int(l in r['odors']) for l in LABS}, 'disease_evidence': r['evidence'],
                 'evidence_detail': r['evidence_detail'], 'focus': focus.get(r['inchikey'], '')} for i, r in enumerate(rows)]
    for name, subset in (('points_all.csv', out_rows), ('disease_list.csv', [r for r in out_rows if r['disease_evidence'] != 'none'])):
        with (data_dir / name).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(subset)

    xy = np.array([[r['x'], r['y']] for r in rows], float)
    hmdb = np.array(['HMDB' in r['evidence'] for r in rows])
    lit = np.array([r['evidence'] == 'literature' for r in rows])
    bw = 2.6; lo = xy.min(0) - 5 * bw; hi = xy.max(0) + 5 * bw; extent = (lo[0], hi[0], lo[1], hi[1])
    gx, gy = np.linspace(lo[0], hi[0], 540), np.linspace(lo[1], hi[1], 420)
    X, Y = np.meshgrid(gx, gy); grid = np.c_[X.ravel(), Y.ravel()]
    fields_kde = []
    for lab in LABS:
        mask = np.array([lab in r['odors'] for r in rows])
        z = kde_field(xy[mask], grid, X.shape, bw)
        fields_kde.append((mask, z))
        np.savetxt(data_dir / f'kde_{lab}_matrix.csv', np.vstack([np.r_[np.nan, gx], np.c_[gy, z]]), delimiter=',', fmt='%.5g')
    fpts = [next(r for r in rows if r['inchikey'] == f['inchikey']) for f in doc['focus'][:2]]

    def finish(ax):
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect='equal'); ax.axis('off')

    def odor(ax, j):
        mask, z = fields_kde[j]; col = COLS[j]
        ax.scatter(*xy.T, s=7, c=BG, lw=0)
        layer = np.ones((*z.shape, 4)); layer[..., :3] = to_rgb(col); layer[..., 3] = .97 * z ** .65
        ax.imshow(layer, origin='lower', extent=extent, interpolation='bilinear')
        ax.scatter(*xy[mask].T, s=9, c=col, alpha=.75, lw=0)
        for i, f in enumerate(fpts):
            ax.scatter(f['x'], f['y'], s=28, facecolors='none', edgecolors=INK, lw=1.3, zorder=6)
            ax.annotate(str(i + 1), (f['x'], f['y']), xytext=((-24, -29) if i == 0 else (25, 24)), textcoords='offset points',
                        fontsize=15, fontweight='bold', color=INK, ha='center', va='center', zorder=8,
                        arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.3, mutation_scale=11, shrinkA=3, shrinkB=4))
        finish(ax)

    def disease(ax, split=False, legend=False):
        ax.scatter(*xy.T, s=12, c=BG, lw=0)
        ax.scatter(*xy[lit].T, s=30, c=LIT if split else LINK, lw=0, zorder=3)
        ax.scatter(*xy[hmdb].T, s=30, c=LINK, lw=0, zorder=3)
        for i, f in enumerate(fpts):
            col = LINK if f['evidence'] != 'none' else BG
            ax.scatter(f['x'], f['y'], s=185, c=col, edgecolors='white', linewidths=3, zorder=8)
            ax.scatter(f['x'], f['y'], s=185, facecolors='none', edgecolors=INK, linewidths=1.8, zorder=9)
            ax.annotate(str(i + 1), (f['x'], f['y']), xytext=((-32, -38) if i == 0 else (34, 34)), textcoords='offset points',
                        fontsize=23, fontweight='bold', color=INK, ha='center', va='center', zorder=10,
                        arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.8, mutation_scale=14, shrinkA=4, shrinkB=9))
        finish(ax)
        if legend:
            ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=7, mfc=LINK, mec=LINK, label=f'HMDB (n={hmdb.sum()})'),
                               Line2D([], [], ls='', marker='o', ms=7, mfc=LIT, mec=LIT, label=f'Literature GC-MS (n={lit.sum()})')],
                      loc='upper left', frameon=False, fontsize=11)

    def save(fig, name):
        for ext in ('png', 'svg', 'pdf'):
            fig.savefig(fig_dir / f'{name}.{ext}', dpi=300, facecolor='white', bbox_inches='tight', pad_inches=.025)
        plt.close(fig)

    for j, lab in enumerate(LABS):
        fig, ax = plt.subplots(figsize=(6, 5.5)); odor(ax, j); fig.subplots_adjust(0, 0, 1, 1); save(fig, f'{j + 1:02d}_{lab}')
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for j, ax in enumerate(axs.flat):
        odor(ax, j)
    fig.subplots_adjust(left=.005, right=.995, bottom=.005, top=.995, wspace=.01, hspace=.02); save(fig, '07_odor_six_panels')
    fig, ax = plt.subplots(figsize=(7, 7)); disease(ax); fig.subplots_adjust(0, 0, 1, 1); save(fig, '08_disease')
    fig, ax = plt.subplots(figsize=(7, 7)); disease(ax, split=True, legend=True); fig.subplots_adjust(0, 0, 1, 1); save(fig, '08b_disease_by_source')
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 5, left=.005, right=.995, bottom=.005, top=.995, wspace=.04, hspace=.02, width_ratios=[1, 1, 1, .78, .78])
    for j in range(6):
        odor(fig.add_subplot(gs[j // 3, j % 3]), j)
    disease(fig.add_subplot(gs[:, 3:])); save(fig, '09_combined')

    meta = {'disease': a.disease, 'n_points': len(rows), 'n_disease_linked': int((hmdb | lit).sum()),
            'n_hmdb': int(hmdb.sum()), 'n_literature_only': int(lit.sum()),
            'odor_counts': {lab: int(m.sum()) for lab, (m, _) in zip(LABS, fields_kde)},
            'kde': {'bandwidth': bw, 'grid': [540, 420], 'x_range': [float(lo[0]), float(hi[0])], 'y_range': [float(lo[1]), float(hi[1])],
                    'matrix_layout': 'first row = x grid, first column = y grid', 'alpha': '0.97 * density**0.65'},
            'colors': dict(zip(LABS, COLS)) | {'disease': LINK, 'literature': LIT, 'background': BG}}
    (data_dir / 'methods.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: meta[k] for k in ('n_points', 'n_disease_linked', 'n_hmdb', 'n_literature_only', 'odor_counts')}))


if __name__ == '__main__':
    main()

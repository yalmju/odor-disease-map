"""Merge the six odor clouds onto one map, optionally with disease points.

Uses the same fixed coordinates, colors, kernel bandwidth and opacity curve as
render_maps.py. Each label is still normalized to its own maximum, so color
strength cannot compare prevalence between labels. Overlapping clouds are
mixed in proportion to their opacities, which makes the result independent
of drawing order.

Odor fields come either from point labels in --input (default) or from a
fields NPZ written by extract_odor_fields.py (--fields), for example when only
a poster image is available.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy.spatial.distance import cdist
from validate_data import validate

LABS = ['fruity', 'floral', 'green', 'woody', 'roasted', 'animal']
COLS = dict(zip(LABS, ['#D5BC35', '#C976A6', '#83B951', '#967CDA', '#639BCD', '#DE837D']))
CRC, BG, INK = '#BA497E', '#B9C1CB', '#24303B'
BW, EXPONENT, PEAK = 2.6, .65, .97


def grid_for(xy):
    lo = xy.min(0) - 5 * BW
    hi = xy.max(0) + 5 * BW
    X, Y = np.meshgrid(np.linspace(lo[0], hi[0], 540), np.linspace(lo[1], hi[1], 420))
    return lo, hi, X, Y


def fields_from_labels(points, xy, labels, X, Y):
    grid = np.c_[X.ravel(), Y.ravel()]
    out = {}
    for lab in labels:
        mask = np.array([lab in p['odors'] for p in points])
        if not mask.any():
            out[lab] = np.zeros(X.shape)
            continue
        z = np.exp(-cdist(grid, xy[mask], 'sqeuclidean') / (2 * BW * BW)).sum(1).reshape(X.shape)
        out[lab] = z / z.max()
    return out


def fields_from_npz(path, labels, X, Y):
    from scipy.interpolate import RegularGridInterpolator
    f = np.load(path)
    out = {}
    for lab in labels:
        if lab not in f.files:
            raise SystemExit(f'{path} has no field for {lab}')
        interp = RegularGridInterpolator((f['grid_y'], f['grid_x']), f[lab], bounds_error=False, fill_value=0)
        out[lab] = np.clip(interp(np.c_[Y.ravel(), X.ravel()]).reshape(X.shape), 0, 1)
    return out


def merge(fields):
    """Opacity-weighted color mix; total opacity = 1 - prod(1 - a_i)."""
    alphas = np.stack([PEAK * fields[l] ** EXPONENT for l in fields])
    rgb = np.stack([to_rgb(COLS[l]) for l in fields])
    weight = alphas.sum(0)
    color = np.einsum('lhw,lc->hwc', alphas, rgb) / np.maximum(weight, 1e-12)[..., None]
    layer = np.ones((*weight.shape, 4))
    layer[..., :3] = color
    layer[..., 3] = 1 - np.prod(1 - alphas, axis=0)
    return layer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, required=True, help='Point JSON in the render_maps.py format')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--labels', default=','.join(LABS), help='Comma-separated subset of the six labels')
    ap.add_argument('--fields', type=Path, help='Optional NPZ from extract_odor_fields.py instead of point labels')
    ap.add_argument('--disease', default='Colorectal cancer')
    ap.add_argument('--focus-names', default='', help='Optional names for focal points 1 and 2, e.g. "Indole,Skatole"')
    ap.add_argument('--font', default='DejaVu Sans')
    args = ap.parse_args()
    labels = [l.strip() for l in args.labels.split(',') if l.strip()]
    if not labels or any(l not in LABS for l in labels):
        raise SystemExit(f'--labels must be chosen from {LABS}')
    d = json.loads(args.input.read_text(encoding='utf-8-sig'))
    validate(d)
    pts = d['points']
    xy = np.array([[p['x'], p['y']] for p in pts])
    lo, hi, X, Y = grid_for(xy)
    extent = (lo[0], hi[0], lo[1], hi[1])
    fields = fields_from_npz(args.fields, labels, X, Y) if args.fields else fields_from_labels(pts, xy, labels, X, Y)
    layer = merge(fields)
    cmask = np.array([args.disease in p['diseases'] for p in pts])
    names = [n.strip() for n in args.focus_names.split(',')] if args.focus_names else []
    plt.rcParams.update({'font.family': args.font, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'text.color': INK})
    args.output.mkdir(parents=True, exist_ok=True)

    def draw(ax, with_disease):
        ax.scatter(*xy.T, s=7 if not with_disease else 10, c=BG, lw=0, zorder=1)
        ax.imshow(layer, origin='lower', extent=extent, interpolation='bilinear', zorder=2)
        if with_disease:
            ax.scatter(*xy[cmask].T, s=30, c=CRC, edgecolors='white', linewidths=.8, zorder=4)
        for i, p in enumerate(d['focus'][:2]):
            fill = CRC if with_disease and args.disease in p['diseases'] else BG
            ax.scatter(p['x'], p['y'], s=120, c=fill, edgecolors='white', linewidths=2.5, zorder=6)
            ax.scatter(p['x'], p['y'], s=120, facecolors='none', edgecolors=INK, linewidths=1.5, zorder=7)
            text = str(i + 1) + (f' {names[i]}' if i < len(names) and names[i] else '')
            ax.annotate(text, (p['x'], p['y']), xytext=((-30, -36) if i == 0 else (32, 32)),
                        textcoords='offset points', fontsize=15, fontweight='bold', color=INK,
                        ha='center', va='center', zorder=8,
                        arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.5, mutation_scale=12, shrinkA=3, shrinkB=7))
        handles = [Patch(color=COLS[l], alpha=.8, label=l) for l in labels]
        if with_disease:
            handles.append(plt.Line2D([], [], marker='o', ls='', color=CRC, markeredgecolor='white', label=args.disease))
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.01), ncol=min(len(handles), 4),
                  frameon=False, fontsize=11, handlelength=1.2, columnspacing=1.2)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect='equal')
        ax.axis('off')

    outputs = []
    for name, with_disease in [('10_odor_merged', False), ('11_odor_merged_disease', True)]:
        fig, ax = plt.subplots(figsize=(8, 7.4))
        draw(ax, with_disease)
        for ext in ['png', 'svg', 'pdf']:
            fig.savefig(args.output / f'{name}.{ext}', dpi=300, facecolor='white', bbox_inches='tight', pad_inches=.03)
        plt.close(fig)
        outputs.append(name)
    manifest = {'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'fields_source': str(args.fields) if args.fields else 'point labels',
                'labels': labels, 'point_count': len(pts), 'disease': args.disease, 'disease_count': int(cmask.sum()),
                'odor_counts': {l: int(sum(l in p['odors'] for p in pts)) for l in labels},
                'blend': 'opacity-weighted color mix, total opacity 1-prod(1-a_i)',
                'bandwidth': BW, 'density_exponent': EXPONENT, 'synthetic': d.get('synthetic', False),
                'coordinate_source': d.get('coordinate_source', 'user-supplied fixed coordinates')}
    (args.output / 'merged_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Rendered {", ".join(outputs)} as PNG/SVG/PDF in {args.output}')


if __name__ == '__main__':
    main()

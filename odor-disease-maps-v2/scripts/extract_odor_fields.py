"""Recover odor cloud fields from rendered odor panels (e.g. a poster image).

Use this only when the labeled point file is unavailable. It inverts the
render_maps.py layer, pixel = (1 - a) * white + a * label_color with
a = 0.97 * z**0.65, after painting over the small dots. Each panel is placed
on the point coordinates by matching the bounding box of its dots to the
bounding box of the points in --points (or an explicit --pixel-box).

The result is an image-derived approximation: no molecule identities or
labels are recovered, and JPEG artefacts or resizing blur the fields.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from matplotlib.colors import to_rgb

COLS = {'fruity': '#D5BC35', 'floral': '#C976A6', 'green': '#83B951',
        'woody': '#967CDA', 'roasted': '#639BCD', 'animal': '#DE837D'}
EXPONENT, PEAK = .65, .97


def box(text):
    v = [int(round(float(t))) for t in text.split(',')]
    if len(v) != 4 or v[2] <= v[0] or v[3] <= v[1]:
        raise argparse.ArgumentTypeError('box must be x0,y0,x1,y1 in pixels')
    return v


def load_rgb(path):
    im = Image.open(path).convert('RGBA')
    white = Image.new('RGBA', im.size, 'white')
    return np.asarray(Image.alpha_composite(white, im).convert('RGB'), float)


def dot_mask(rgb, dot_px):
    """Pixels to ignore: small dots that differ from the smooth cloud, plus dark ink (rings, arrows, digits)."""
    size = max(3, int(2 * dot_px + 1))
    smooth = np.stack([ndi.median_filter(rgb[..., c], size=size) for c in range(3)], -1)
    dots = ndi.binary_dilation(np.abs(rgb - smooth).max(-1) > 18, iterations=1)
    ink = ndi.binary_dilation(rgb.mean(-1) < 110, iterations=max(2, int(dot_px)))
    return dots, dots | ink


def inpaint(rgb, mask):
    """Fill masked pixels with the colour of the nearest unmasked pixel."""
    if not mask.any() or mask.all():
        return rgb
    _, (iy, ix) = ndi.distance_transform_edt(mask, return_indices=True)
    return rgb[iy, ix]


def field(rgb, mask, color):
    clean = inpaint(rgb, mask)
    c = 255 * (1 - np.array(to_rgb(color)))               # how far the label color is from white
    a = ((255 - clean) * c).sum(-1) / (c * c).sum()
    a = np.clip(a, 0, PEAK)
    return (a / PEAK) ** (1 / EXPONENT)


def pixel_bbox(mask, trim):
    lab, n = ndi.label(mask)
    if not n:
        raise SystemExit('No dots found in panel; pass --pixel-box')
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.nonzero(sizes <= np.percentile(sizes, 95) * 3)[0])  # drop text/arrow blobs
    ys, xs = np.nonzero(keep)
    return (np.percentile(xs, trim), np.percentile(ys, trim), np.percentile(xs, 100 - trim), np.percentile(ys, 100 - trim))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', type=Path, required=True)
    ap.add_argument('--panel', action='append', required=True,
                    help='label=x0,y0,x1,y1 crop of one odor panel plot area (no title); repeat per label')
    ap.add_argument('--points', type=Path, required=True, help='Point JSON whose coordinates the fields should match')
    ap.add_argument('--pixel-box', action='append', default=[],
                    help='label=x0,y0,x1,y1 pixel bbox of the dots inside the crop, if auto-detection fails')
    ap.add_argument('--dot-px', type=float, default=3, help='Approximate dot radius in pixels')
    ap.add_argument('--trim', type=float, default=0, help='Percentile trimmed from each side of the auto dot bbox')
    ap.add_argument('--out', type=Path, required=True, help='Output .npz')
    args = ap.parse_args()
    rgb_all = load_rgb(args.image)
    pts = json.loads(args.points.read_text(encoding='utf-8-sig'))['points']
    xy = np.array([[p['x'], p['y']] for p in pts])
    (dx0, dy0), (dx1, dy1) = xy.min(0), xy.max(0)
    manual = {}
    for item in args.pixel_box:
        lab, v = item.split('=', 1)
        manual[lab] = box(v)
    # Common output grid in data coordinates, padded like render_maps.py.
    pad = 5 * 2.6
    gx = np.linspace(dx0 - pad, dx1 + pad, 540)
    gy = np.linspace(dy0 - pad, dy1 + pad, 420)
    out, report = {'grid_x': gx, 'grid_y': gy}, {}
    for item in args.panel:
        lab, v = item.split('=', 1)
        if lab not in COLS:
            raise SystemExit(f'Unknown label {lab}; choose from {list(COLS)}')
        x0, y0, x1, y1 = box(v)
        rgb = rgb_all[y0:y1, x0:x1]
        dots, mask = dot_mask(rgb, args.dot_px)
        z = field(rgb, mask, COLS[lab])
        px0, py0, px1, py1 = manual.get(lab) or pixel_bbox(dots & ~(rgb.mean(-1) < 110), args.trim)
        # pixel -> data: x grows right, data y grows up while pixel row grows down.
        sx, sy = (px1 - px0) / (dx1 - dx0), (py1 - py0) / (dy1 - dy0)
        GX, GY = np.meshgrid(gx, gy)
        cols = px0 + (GX - dx0) * sx
        rows = py0 + (dy1 - GY) * sy
        out[lab] = ndi.map_coordinates(z, [rows, cols], order=1, mode='constant', cval=0)
        report[lab] = {'crop': [x0, y0, x1, y1], 'dot_bbox_px': [round(float(t), 1) for t in (px0, py0, px1, py1)],
                       'scale_ratio_x_over_y': round(float(sx / sy), 3), 'max_field': round(float(z.max()), 3)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(json.dumps(report, indent=2))
    print(f'Saved {len(report)} fields -> {args.out}. scale_ratio far from 1 means a wrong crop or dot bbox.')


if __name__ == '__main__':
    main()

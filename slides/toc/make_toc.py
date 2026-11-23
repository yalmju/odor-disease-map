"""Build the project TOC graphic: where a smell sits, and what it looks like."""
import math, random, re

W, H = 1600, 860
PINK, BLUE, AMBER = "#D94F8A", "#4A5899", "#C97A1E"
PURPLE, INK, MUTE = "#6B3F99", "#2B2B2B", "#6E6E73"
GRAYPT, LINE = "#D7D7DD", "#C8C8D0"

rnd = random.Random(20261123)


def odor_space(x0, y0, w, h):
    """Stylized odor-space cloud: soft quality regions plus a scatter."""
    out = []
    blobs = [(0.28, 0.30, 0.17, "#D9C84F"), (0.62, 0.24, 0.15, PINK),
             (0.76, 0.58, 0.16, "#5BA85B"), (0.33, 0.70, 0.18, "#8A6FC4"),
             (0.52, 0.48, 0.14, "#4FA3A3")]
    for i, (cx, cy, r, col) in enumerate(blobs):
        out.append(f'<circle cx="{x0+cx*w:.0f}" cy="{y0+cy*h:.0f}" r="{r*min(w,h):.0f}" '
                   f'fill="{col}" opacity="0.13"/>')
    for _ in range(560):
        bx, by, br, _c = blobs[rnd.randrange(len(blobs))]
        a, d = rnd.uniform(0, 2 * math.pi), abs(rnd.gauss(0, .55)) * br
        px, py = x0 + (bx + d * math.cos(a)) * w, y0 + (by + d * math.sin(a)) * h
        if x0 + 6 < px < x0 + w - 6 and y0 + 6 < py < y0 + h - 6:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="{GRAYPT}"/>')
    # disease-linked subset, deliberately spread across regions
    for _ in range(38):
        bx, by, br, _c = blobs[rnd.randrange(len(blobs))]
        a, d = rnd.uniform(0, 2 * math.pi), abs(rnd.gauss(0, .5)) * br
        px, py = x0 + (bx + d * math.cos(a)) * w, y0 + (by + d * math.sin(a)) * h
        if x0 + 6 < px < x0 + w - 6 and y0 + 6 < py < y0 + h - 6:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.6" fill="{PINK}" opacity="0.92"/>')
    return "\n".join(out)


def channel_tile(x0, y0, size, seed, hue_lo=BLUE, hue_hi=PINK):
    """A 7x7 dot map on a blue-white-pink diverging scale."""
    r = random.Random(seed)
    n, pad = 7, 6
    step = (size - 2 * pad) / n
    base = r.uniform(-1, 1)
    out = [f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" rx="7" fill="#FFFFFF" stroke="{LINE}"/>']
    for i in range(n):
        for j in range(n):
            v = max(-1, min(1, base + r.gauss(0, .45)))
            col, op = (hue_hi, abs(v)) if v >= 0 else (hue_lo, abs(v))
            cx = x0 + pad + step * (j + .5)
            cy = y0 + pad + step * (i + .5)
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{step*0.33:.1f}" '
                       f'fill="{col}" opacity="{0.12+0.8*op:.2f}"/>')
    return "\n".join(out)


def mol_svg(path, x, y, w, h):
    """Inline an RDKit SVG as a positioned group."""
    s = open(path).read()
    s = re.sub(r"<\?xml.*?\?>", "", s, flags=re.S)
    s = re.sub(r"<!DOCTYPE.*?>", "", s, flags=re.S)
    m = re.search(r"<svg[^>]*>(.*)</svg>", s, flags=re.S)
    body = m.group(1)
    body = re.sub(r"<rect[^>]*style='opacity:1\.0;fill:#FFFFFF[^>]*/>", "", body)
    return (f'<g transform="translate({x},{y}) scale({w/300:.4f},{h/220:.4f})">{body}</g>')


svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>']

F = 'font-family="Arial, Helvetica, sans-serif"'

# ---- headline
svg.append(f'<text x="60" y="76" {F} font-size="40" font-weight="bold" fill="{PURPLE}">'
           f'Where a smell sits &#8212; and what it looks like</text>')
svg.append(f'<text x="60" y="114" {F} font-size="22" fill="{MUTE}">'
           f'Disease-relevant odorants on a learned odor map, read out as three-channel Raman patterns</text>')

# ---- left: chemical space
LX, LY, LW, LH = 60, 158, 560, 540
svg.append(f'<rect x="{LX}" y="{LY}" width="{LW}" height="{LH}" rx="18" fill="#FAFAFC" stroke="{LINE}"/>')
svg.append(f'<text x="{LX+24}" y="{LY+44}" {F} font-size="25" font-weight="bold" fill="{INK}">Chemical space</text>')
svg.append(f'<text x="{LX+24}" y="{LY+74}" {F} font-size="18" fill="{MUTE}">532 odorants, placed by learned odor perception</text>')
svg.append(odor_space(LX + 26, LY + 96, LW - 52, LH - 168))

# two adjacent focal points
fx, fy = LX + 0.60 * LW, LY + 96 + 0.30 * (LH - 168)
svg.append(f'<circle cx="{fx:.0f}" cy="{fy:.0f}" r="11" fill="#FFFFFF" stroke="{INK}" stroke-width="3"/>')
svg.append(f'<circle cx="{fx+26:.0f}" cy="{fy-12:.0f}" r="11" fill="{PINK}" stroke="{INK}" stroke-width="3"/>')
svg.append(f'<text x="{fx-46:.0f}" y="{fy+8:.0f}" {F} font-size="19" font-weight="bold" fill="{INK}">1</text>')
svg.append(f'<text x="{fx+48:.0f}" y="{fy-8:.0f}" {F} font-size="19" font-weight="bold" fill="{INK}">2</text>')
svg.append(f'<text x="{LX+24}" y="{LY+LH-44}" {F} font-size="19" fill="{INK}">'
           f'<tspan fill="{PINK}" font-weight="bold">&#9679;</tspan> disease-linked &#183; spread across odor qualities</text>')
svg.append(f'<text x="{LX+24}" y="{LY+LH-18}" {F} font-size="19" fill="{INK}">'
           f'<tspan font-weight="bold">1, 2</tspan> one methyl apart, yet neighbours</text>')

# ---- arrow
AX = LX + LW + 26
svg.append(f'<path d="M{AX} {LY+LH/2} L{AX+58} {LY+LH/2}" stroke="{MUTE}" stroke-width="4" marker-end="url(#ar)"/>')
svg.append(f'<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">'
           f'<path d="M0,0 L9,4.5 L0,9 Z" fill="{MUTE}"/></marker></defs>')

# ---- right: odor pattern
RX, RY, RW, RH = AX + 96, 158, W - (AX + 96) - 60, 540
svg.append(f'<rect x="{RX}" y="{RY}" width="{RW}" height="{RH}" rx="18" fill="#FDF7FA" stroke="{PINK}" stroke-width="2.5"/>')
svg.append(f'<text x="{RX+24}" y="{RY+44}" {F} font-size="25" font-weight="bold" fill="{INK}">Odor pattern</text>')
svg.append(f'<text x="{RX+24}" y="{RY+74}" {F} font-size="18" fill="{MUTE}">one molecule &#8594; three reporter channels, mapped in space</text>')

TILE, GAP = 128, 16
cols = ["4-MBA", "4-MPY", "2-NAT"]
for k, c in enumerate(cols):
    cx = RX + 196 + k * (TILE + GAP) + TILE / 2
    svg.append(f'<text x="{cx:.0f}" y="{RY+112}" {F} font-size="18" fill="{MUTE}" text-anchor="middle">{c}</text>')

rows = [("1", "indole.svg", 1), ("2", "skatole.svg", 2)]
for ri, (lab, molf, seed0) in enumerate(rows):
    ry = RY + 130 + ri * (TILE + 46)
    svg.append(f'<text x="{RX+28}" y="{ry+TILE/2+8:.0f}" {F} font-size="22" font-weight="bold" fill="{INK}">{lab}</text>')
    svg.append(mol_svg(molf, RX + 48, ry - 14, 150, TILE + 28))
    for k in range(3):
        svg.append(channel_tile(RX + 196 + k * (TILE + GAP), ry, TILE, seed0 * 31 + k * 7))

svg.append(f'<text x="{RX+24}" y="{RY+RH-20}" {F} font-size="19" fill="{INK}">'
           f'Neighbours on the map, <tspan font-weight="bold">different under the array</tspan></text>')

# ---- footer claim
svg.append(f'<rect x="60" y="{H-118}" width="{W-120}" height="62" rx="14" fill="#EDEDED"/>')
svg.append(f'<text x="{W/2}" y="{H-78}" {F} font-size="25" font-weight="bold" fill="{PINK}" text-anchor="middle">'
           f'A chemically interpretable optical nose: patterns you measure, bands you name.</text>')

# honesty marker: the tiles are drawn, not measured. Delete this block once
# the real per-channel maps are dropped in.
svg.append(f'<text x="{W-62}" y="{H-130}" {F} font-size="15" fill="#B5B5BD" text-anchor="end">'
           f'illustrative layout &#183; replace tiles with measured maps</text>')

svg.append("</svg>")
open("toc.svg", "w").write("\n".join(svg))
print("wrote toc.svg")

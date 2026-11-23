"""Project TOC graphic: where a smell sits, and what it looks like.

Deliberately few text blocks — a TOC is read in one glance. Everything that
is not the headline, the two panel labels and the closing line was cut.
"""
import math, random, re

W, H = 1600, 860
PINK, BLUE = "#D94F8A", "#4A5899"
PURPLE, INK, MUTE = "#6B3F99", "#2B2B2B", "#6E6E73"
GRAYPT, LINE = "#D5D5DC", "#C8C8D0"
F = 'font-family="Arial, Helvetica, sans-serif"'

rnd = random.Random(20261123)


def odor_space(x0, y0, w, h):
    out = []
    blobs = [(0.26, 0.28, 0.19, "#D9C84F"), (0.64, 0.22, 0.17, PINK),
             (0.78, 0.60, 0.18, "#5BA85B"), (0.30, 0.72, 0.20, "#8A6FC4"),
             (0.52, 0.49, 0.16, "#4FA3A3")]
    for cx, cy, r, col in blobs:
        out.append(f'<circle cx="{x0+cx*w:.0f}" cy="{y0+cy*h:.0f}" '
                   f'r="{r*min(w,h):.0f}" fill="{col}" opacity="0.14"/>')
    for _ in range(760):
        bx, by, br, _ = blobs[rnd.randrange(len(blobs))]
        a, d = rnd.uniform(0, 2*math.pi), abs(rnd.gauss(0, .55))*br
        px, py = x0 + (bx + d*math.cos(a))*w, y0 + (by + d*math.sin(a))*h
        if x0+8 < px < x0+w-8 and y0+8 < py < y0+h-8:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.4" fill="{GRAYPT}"/>')
    for _ in range(38):
        bx, by, br, _ = blobs[rnd.randrange(len(blobs))]
        a, d = rnd.uniform(0, 2*math.pi), abs(rnd.gauss(0, .5))*br
        px, py = x0 + (bx + d*math.cos(a))*w, y0 + (by + d*math.sin(a))*h
        if x0+8 < px < x0+w-8 and y0+8 < py < y0+h-8:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.4" fill="{PINK}"/>')
    return "\n".join(out)


def tile(x0, y0, size, seed):
    r = random.Random(seed)
    n, pad = 9, 7
    step = (size - 2*pad) / n
    base = r.uniform(-.9, .9)
    out = [f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" rx="9" '
           f'fill="#FFFFFF" stroke="{LINE}"/>']
    for i in range(n):
        for j in range(n):
            v = max(-1, min(1, base + r.gauss(0, .42)))
            col = PINK if v >= 0 else BLUE
            out.append(f'<circle cx="{x0+pad+step*(j+.5):.1f}" cy="{y0+pad+step*(i+.5):.1f}" '
                       f'r="{step*0.36:.1f}" fill="{col}" opacity="{0.10+0.85*abs(v):.2f}"/>')
    return "\n".join(out)


def mol(path, x, y, w, h):
    s = open(path).read()
    s = re.sub(r"<\?xml.*?\?>|<!DOCTYPE.*?>", "", s, flags=re.S)
    body = re.search(r"<svg[^>]*>(.*)</svg>", s, flags=re.S).group(1)
    body = re.sub(r"<rect[^>]*fill:#FFFFFF[^>]*/>", "", body)
    return f'<g transform="translate({x},{y}) scale({w/300:.4f},{h/220:.4f})">{body}</g>'


svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
       f'<text x="{W/2}" y="74" {F} font-size="44" font-weight="bold" fill="{PURPLE}" '
       f'text-anchor="middle">Where a smell sits &#8212; and what it looks like</text>']

PY_TOP, PH = 124, 596

# ---------------- left: chemical space
LX, LW = 60, 660
svg.append(f'<rect x="{LX}" y="{PY_TOP}" width="{LW}" height="{PH}" rx="20" '
           f'fill="#FAFAFC" stroke="{LINE}"/>')
svg.append(f'<text x="{LX+LW/2}" y="{PY_TOP+48}" {F} font-size="30" font-weight="bold" '
           f'fill="{INK}" text-anchor="middle">Chemical space</text>')
svg.append(odor_space(LX+28, PY_TOP+72, LW-56, PH-150))

fx, fy = LX + 0.62*LW, PY_TOP + 72 + 0.26*(PH-150)
svg.append(f'<circle cx="{fx:.0f}" cy="{fy:.0f}" r="14" fill="#FFFFFF" stroke="{INK}" stroke-width="3.5"/>')
svg.append(f'<circle cx="{fx+34:.0f}" cy="{fy-16:.0f}" r="14" fill="{PINK}" stroke="{INK}" stroke-width="3.5"/>')
svg.append(f'<text x="{fx-54:.0f}" y="{fy+9:.0f}" {F} font-size="26" font-weight="bold" fill="{INK}">1</text>')
svg.append(f'<text x="{fx+60:.0f}" y="{fy-9:.0f}" {F} font-size="26" font-weight="bold" fill="{INK}">2</text>')
svg.append(f'<text x="{LX+LW/2}" y="{PY_TOP+PH-28}" {F} font-size="21" fill="{MUTE}" text-anchor="middle">'
           f'<tspan fill="{PINK}" font-weight="bold">&#9679;</tspan> disease-linked, spread across odor qualities</text>')

# ---------------- arrow
AX = LX + LW + 24
svg.append(f'<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">'
           f'<path d="M0,0 L9,4.5 L0,9 Z" fill="{MUTE}"/></marker></defs>')
svg.append(f'<path d="M{AX} {PY_TOP+PH/2} L{AX+54} {PY_TOP+PH/2}" stroke="{MUTE}" '
           f'stroke-width="5" marker-end="url(#ar)"/>')

# ---------------- right: odor pattern
RX = AX + 92
RW = W - RX - 60
svg.append(f'<rect x="{RX}" y="{PY_TOP}" width="{RW}" height="{PH}" rx="20" '
           f'fill="#FDF7FA" stroke="{PINK}" stroke-width="3"/>')
svg.append(f'<text x="{RX+RW/2}" y="{PY_TOP+48}" {F} font-size="30" font-weight="bold" '
           f'fill="{INK}" text-anchor="middle">Odor pattern</text>')

TILE, GAP, MOLW = 162, 18, 150
grid_w = MOLW + 3*TILE + 2*GAP
gx = RX + (RW - grid_w) / 2

for k, c in enumerate(("4-MBA", "4-MPY", "2-NAT")):
    svg.append(f'<text x="{gx+MOLW+k*(TILE+GAP)+TILE/2:.0f}" y="{PY_TOP+100}" {F} '
               f'font-size="21" fill="{MUTE}" text-anchor="middle">{c}</text>')

for ri, (lab, f_, seed0) in enumerate((("1", "indole.svg", 1), ("2", "skatole.svg", 2))):
    ry = PY_TOP + 120 + ri*(TILE + 36)
    svg.append(f'<text x="{gx-16:.0f}" y="{ry+TILE/2+10:.0f}" {F} font-size="26" '
               f'font-weight="bold" fill="{INK}" text-anchor="end">{lab}</text>')
    svg.append(mol(f_, gx, ry - 10, MOLW, TILE + 20))
    for k in range(3):
        svg.append(tile(gx + MOLW + k*(TILE+GAP), ry, TILE, seed0*31 + k*7))

svg.append(f'<text x="{RX+RW/2}" y="{PY_TOP+PH-28}" {F} font-size="21" fill="{MUTE}" '
           f'text-anchor="middle">one methyl apart &#183; neighbours on the map &#183; '
           f'<tspan font-weight="bold" fill="{INK}">distinct under the array</tspan></text>')

# ---------------- closing line
svg.append(f'<text x="{W/2}" y="{H-36}" {F} font-size="27" font-weight="bold" '
           f'fill="{PINK}" text-anchor="middle">'
           f'A chemically interpretable optical nose: patterns you measure, bands you name.</text>')

# honesty marker: the scatter and tiles are drawn, not measured.
# Delete this line once the real figures are dropped in.
svg.append(f'<text x="{W-60}" y="{H-84}" {F} font-size="15" fill="#BFBFC6" '
           f'text-anchor="end">illustrative layout</text>')

svg.append("</svg>")
open("toc.svg", "w").write("\n".join(svg))
print("wrote toc.svg")

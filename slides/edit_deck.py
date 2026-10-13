"""Rebuild the gap slide inside the real deck, give every results slide a claim
title and a takeaway line, and add a closing slide with results + next steps."""
import copy
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from PIL import Image

SRC, DST = "OpticalNose_5min_v7.pptx", "OpticalNose_5min_v8.pptx"
CHECK, CROSS, ARROW = "✓", "✗", "→"

prs = Presentation(SRC)

C = dict(purple="6B3F99", teal="1F8A8A", amber="C97A1E", pink="D94F8A",
         red="C2410C", green="15803D", ink="2B2B2B", mute="6E6E73",
         t_teal="E9F4F4", t_purple="F3F0F7", t_amber="FDF3E7",
         gray="EDEDED", white="FFFFFF")
def rgb(k): return RGBColor.from_string(C[k])


# ---------------------------------------------------------------- helpers
def set_title(slide, text):
    for sh in slide.shapes:
        if sh.has_text_frame and sh.top < Inches(1.0) and sh.width > Inches(5):
            para = sh.text_frame.paragraphs[0]
            if para.runs:
                para.runs[0].text = text
                for extra in para.runs[1:]:
                    extra.text = ""
                return sh
    return None


def main_figure(slide):
    """Largest picture or group on the slide that is actually on the canvas."""
    best = None
    for sh in slide.shapes:
        if sh.shape_type not in (6, 13):           # GROUP, PICTURE
            continue
        if sh.top > Inches(9) or sh.top < Inches(-1):
            continue
        area = sh.width * sh.height
        if best is None or area > best.width * best.height:
            best = sh
    return best


def fit(sh, left, top, width, height):
    """Scale a shape into the box, preserving aspect, centred."""
    ar = sh.width / sh.height
    w, h = width, width / ar
    if h > height:
        h, w = height, height * ar
    sh.width, sh.height = int(w), int(h)
    sh.left = int(left + (width - w) / 2)
    sh.top = int(top)


def drop_offslide(slide):
    """Remove leftovers parked outside the canvas."""
    H = prs.slide_height
    for sh in list(slide.shapes):
        if sh.top > H or sh.top + sh.height < 0:
            sh._element.getparent().remove(sh._element)


def takeaway(slide, text, y=5.76, h=0.62, color="teal", size=16):
    S = slide.shapes
    bar = S.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                      Inches(0.90), Inches(y), Inches(11.53), Inches(h))
    bar.adjustments[0] = 0.22
    bar.fill.solid(); bar.fill.fore_color.rgb = rgb("gray")
    bar.line.fill.background(); bar.shadow.inherit = False
    tf = bar.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.22)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = rgb(color)
    return bar


# ================================================================ SLIDE 4
slide = prs.slides[3]
S = slide.shapes
for sh in list(S):
    if sh.shape_type == 6:
        sh._element.getparent().remove(sh._element)
set_title(slide, "How do we capture odor differences?")


def textbox(S, x, y, w, h, lines, size=12, color="ink", align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP, spacing=1.0, italic=False):
    tb = S.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(lines, str):
        lines = [[(lines, False)]]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = spacing
        for text, bold in line:
            r = p.add_run(); r.text = text
            r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
            r.font.color.rgb = rgb(color)
    return tb


def box(S, x, y, w, h, fill, radius=0.08):
    sh = S.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                     Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = radius
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    sh.line.fill.background(); sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def pill(S, x, y, w, h, fill, label, size=13):
    sh = box(S, x, y, w, h, fill, radius=0.3)
    tf = sh.text_frame; tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = label
    r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = rgb("white")
    return sh


textbox(S, 0.92, 1.12, 11.5, 0.3,
        "Three ways to read a smell today — and three different blind spots.",
        size=16, color="mute")

L, TOTAL, GAP = 0.90, 11.53, 0.24
CW = (TOTAL - 2 * GAP) / 3
TOP, CH = 1.52, 3.98

COLS = [
    dict(tint="t_teal", accent="teal", tag="Human nose", tagw=1.95,
         img="fig_cell.png", cite="Malnic et al., Cell 96, 713 (1999)",
         cap=[[("One odorant hits many receptors,", False)],
              [("but the answer is ", False), ("a word, not a number", True), (".", False)]],
         chips=[(CHECK + "  any smell", "green", 1.60),
                (CROSS + "  no number", "red", 1.65)]),
    dict(tint="t_purple", accent="purple", tag="Artificial nose", tagw=2.25,
         img="fig_advmat.png", cite="Adv. Mater. 36, 2402287 (2024)",
         cap=[[("Sensor arrays copy that pattern", False)],
              [("without saying ", False), ("which molecule", True), (".", False)]],
         chips=[(CHECK + "  a number", "green", 1.65),
                (CROSS + "  no molecule", "red", 1.90)]),
    dict(tint="t_amber", accent="amber", tag="Odor models", tagw=2.00,
         img="fig_pom.png", cite="Lee et al., Science 381, 6661 (2023)",
         cap=[[("A learned map explains odor,", False)],
              [("but the ", False), ("molecule must be known first", True), (".", False)]],
         chips=[(CHECK + "  explains", "green", 1.60),
                (CROSS + "  no sample", "red", 1.65)]),
]

for i, c in enumerate(COLS):
    x = L + i * (CW + GAP)
    box(S, x, TOP, CW, CH, c["tint"], radius=0.05)
    pill(S, x + (CW - c["tagw"]) / 2, TOP + 0.12, c["tagw"], 0.34, c["accent"], c["tag"])

    sx, sy = x + 0.16, TOP + 0.55
    sw, sh_ = CW - 0.32, 1.80
    box(S, sx, sy, sw, sh_, "white", radius=0.05)
    iw, ih = Image.open(c["img"]).size
    ar = iw / ih
    maxw, maxh = sw - 0.18, sh_ - 0.16
    w = maxw; h = w / ar
    if h > maxh:
        h = maxh; w = h * ar
    S.add_picture(c["img"], Inches(sx + (sw - w) / 2), Inches(sy + (sh_ - h) / 2),
                  width=Inches(w), height=Inches(h))

    textbox(S, x + 0.10, TOP + 2.42, CW - 0.20, 0.22, c["cite"],
            size=9, color="mute", align=PP_ALIGN.CENTER, italic=True)
    textbox(S, x + 0.14, TOP + 2.66, CW - 0.28, 0.60, c["cap"],
            size=13, color="ink", align=PP_ALIGN.CENTER, spacing=1.3)

    total_w = sum(w for _, _, w in c["chips"]) + 0.12
    bx = x + (CW - total_w) / 2
    for label, col, w in c["chips"]:
        ch_ = box(S, bx, TOP + 3.30, w, 0.32, "white", radius=0.45)
        tf = ch_.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = label
        r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = rgb(col)
        bx += w + 0.12

bar = box(S, L, 5.60, TOTAL, 0.88, "gray", radius=0.16)
tf = bar.text_frame; tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = tf.margin_right = Inches(0.25)
p1 = tf.paragraphs[0]; p1.alignment = PP_ALIGN.CENTER
r = p1.add_run(); r.text = "No number, no molecule, no sample."
r.font.size = Pt(15); r.font.color.rgb = rgb("mute")
p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
r = p2.add_run(); r.text = ARROW + "  Measure a real sample, and keep the molecule."
r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = rgb("pink")


# ========================================================== SLIDES 5 - 9
PLAN = [
    (4, "Our approach: read odorants as Raman maps", None),
    (5, "Three reporters, three binding chemistries",
     "Charge transfer, hydrogen bonding and π–π stacking — one odorant leaves a different mark on each."),
    (6, "Each channel answers differently",
     "Band ratios shift by a different amount in each reporter channel."),
    (7, "Each odorant gets its own signature",
     "Indole, skatole and nonanal separate on the three-channel response pattern."),
    (8, "A mixture is not the sum of its parts",
     "Indole + skatole reproduces neither pure pattern — the map changes, not just the amplitude."),
]

PLAN = [e for e in PLAN if e[0] != 8]

for idx, title, line in PLAN:
    sl = prs.slides[idx]
    drop_offslide(sl)
    set_title(sl, title)
    fig = main_figure(sl)
    if idx == 4:
        # keep the existing question bar; make room above it
        fit(fig, Inches(0.70), Inches(1.26), Inches(11.9), Inches(4.02))
        for sh in sl.shapes:
            if sh.has_text_frame and "Can Raman maps" in sh.text_frame.text:
                sh.top, sh.height = Inches(5.46), Inches(1.00)
                paras = sh.text_frame.paragraphs
                for para, text in zip(paras, ["Can Raman maps of a few receptors",
                                              "make these differences visible and explainable?"]):
                    para.runs[0].text = text
                    for extra in para.runs[1:]:
                        extra.text = ""
    else:
        fit(fig, Inches(0.70), Inches(1.22), Inches(11.9), Inches(4.40))
        takeaway(sl, line)


# ========================================= SLIDE 9 - the closing slide
sl = prs.slides[8]
drop_offslide(sl)
set_title(sl, "A mixture is not the sum of its parts")
S9 = sl.shapes

def dashed_box(S, x, y, w, h, label, size=11):
    sh = S.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                     Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.04
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb("white")
    sh.line.color.rgb = RGBColor.from_string("BFBFC8"); sh.line.width = Pt(1.25)
    ln = sh.line._get_or_add_ln()
    ln.append(ln.makeelement(qn('a:prstDash'), {'val': 'dash'}))
    sh.shadow.inherit = False
    tf = sh.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para = tf.paragraphs[0]; para.alignment = PP_ALIGN.CENTER
    r = para.add_run(); r.text = label
    r.font.size = Pt(size); r.font.bold = True
    r.font.color.rgb = RGBColor.from_string("9A9AA0")
    return sh

# QR, top right
textbox(S9, 9.35, 0.50, 2.35, 0.60,
        [[("More information?", False)], [("Sniff here!", False)]],
        size=15, color="ink", align=PP_ALIGN.RIGHT, spacing=1.15)
dashed_box(S9, 11.80, 0.42, 1.02, 1.02, "QR", size=12)

# the dot maps keep the left two thirds
fig = main_figure(sl)
fit(fig, Inches(0.90), Inches(1.52), Inches(7.75), Inches(3.34))

# future work on the right
fw = box(S9, 8.92, 1.52, 3.51, 3.34, "t_purple", radius=0.06)
tf = fw.text_frame; tf.word_wrap = True
tf.margin_left = tf.margin_right = Inches(0.20)
tf.margin_top = Inches(0.18)
p0 = tf.paragraphs[0]
r = p0.add_run(); r.text = "Future work"
r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = rgb("purple")
for item in ("Extend the reporter set to cover more of the odor map",
             "Link channel responses to disease-relevant VOC panels",
             "Learn mixture composition from mixture spectra themselves"):
    pp = tf.add_paragraph(); pp.line_spacing = 1.2; pp.space_before = Pt(10)
    rr = pp.add_run(); rr.text = "\u2022  " + item
    rr.font.size = Pt(12.5); rr.font.color.rgb = rgb("ink")

# the claim the talk closes on
bar9 = box(S9, 0.90, 4.98, 11.53, 0.78, "gray", radius=0.18)
tf = bar9.text_frame; tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = tf.margin_right = Inches(0.25)
p0 = tf.paragraphs[0]; p0.alignment = PP_ALIGN.CENTER
r = p0.add_run()
r.text = "A chemically interpretable optical nose: patterns you measure, bands you name."
r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = rgb("pink")

# funding and institution marks
textbox(S9, 0.90, 5.92, 8.10, 0.50,
        [[("Research funding", True),
          (": This work was supported by the National Research Foundation of Korea (NRF) "
           "grant funded by the Korea government (MSIT) RS-2025-00560226.", False)]],
        size=11, color="ink", spacing=1.15)
dashed_box(S9, 9.60, 5.88, 1.60, 0.56, "NRF", size=10)
dashed_box(S9, 11.35, 5.88, 1.08, 0.56, "KU", size=10)


# ===================================== SLIDE 10 - dropped (no Q&A time)
# The quadrant medians are a reduction of the maps already shown on the
# closing slide, and the talk has no question time to spend on them.
sld_lst = prs.slides._sldIdLst
to_drop = list(sld_lst)[9]
prs.part.drop_rel(to_drop.rId)
sld_lst.remove(to_drop)


# ================================================== page numbers, last pass
for n, sl in enumerate(prs.slides, start=1):
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().isdigit() and sh.left > Inches(12):
            para = sh.text_frame.paragraphs[0]
            if para.runs:
                para.runs[0].text = f"{n:02d}"
                for extra in para.runs[1:]:
                    extra.text = ""

prs.save(DST)
print("saved", DST, "with", len(prs.slides.__iter__.__self__._sldIdLst), "slides")

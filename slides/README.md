# Slide sources

Design sources for the µTAS 2026 five-minute talk. Only material authored here is
tracked; the working deck and any published third-party figures stay out of the
repository (see `.gitignore` and `docs/DATA_SOURCES.md`).

| File | What it is |
|---|---|
| `slide03-gap.html` | Standalone HTML source for the "how do we capture odor differences" slide, 1280x720 |
| `slide03-gap.pptx` | The same slide as an editable PowerPoint file, with dashed slots for the reference figures |
| `build_pptx.py` | Rebuilds `slide03-gap.pptx` from scratch |
| `edit_deck.py` | Rebuilds slide 4 inside the real deck and renumbers every page |
| `raman-art.png` | Generated Raman artwork used by `build_pptx.py` |

## Rendering the HTML

```
chromium --headless --disable-gpu --no-sandbox --hide-scrollbars \
  --window-size=1280,900 --screenshot=out.png slide03-gap.html
```

Crop the result to 1280x720. The taller window is deliberate: the rendered
viewport is shorter than the window, so a 720px window clips the footer band.

## Editing the deck

`edit_deck.py` reads `OpticalNose_5min_v7.pptx` and writes `..._v8.pptx`. Both are
untracked, so place the input deck in this directory before running it. The script
expects `fig_cell.png`, `fig_advmat.png` and `fig_pom.png`, which are crops of
published figures and are also untracked.

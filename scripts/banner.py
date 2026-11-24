"""ippon's README banner: the pixel lucky cat (it beckons with a gentle sway) beside the wordmark and tagline, as two
self-contained animated SVGs, assets/banner-light.svg and assets/banner-dark.svg. It is the terminal's banner
(`ippon` alone, the start of `ippon setup`) drawn for the page: the name in brand.py's pixel letters and the tagline,
brand.TAGLINE, in a monospaced face.

    uv run --with fonttools python scripts/banner.py FONTS_DIR assets

The cat is ippon's logo, assets/logo-light.svg and assets/logo-dark.svg (32 x 32 pixels), drawn here at 10.5 banner
units a pixel. FONTS_DIR holds Source Code Pro's variable font from github.com/google/fonts (OFL), as
SourceCodePro.ttf: the tagline is outlined from it, since GitHub loads no fonts inside a README image. The sway is
SMIL, which browsers play inside an image.
"""

import re
import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

from ippon.brand import TAGLINE, wordmark

FONTS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ASSETS = Path(__file__).resolve().parents[1] / "assets"
PIXEL = 10.5  # banner units a logo pixel: the cat 336 units tall in a 380-unit banner
LETTER = 14  # banner units a pixel of the name's letters
THEMES = {
    "light": {"word": "#1f2328", "tag": "#4a5568"},  # the name in the text colour, as the terminal draws it
    "dark": {"word": "#e6edf3", "tag": "#9da7b3"},
}
# the beckon: still most of the loop, then a small rock about the cat's base, as the post's closing card does
SWAY = (("0", 0.0), ("0", 0.7), ("-5", 0.78), ("4", 0.86), ("-2", 0.93), ("0", 1.0))
SWAY_S = 3.2


def logo(theme: str) -> str:
    """The logo's drawing (everything inside its <svg>), in its own 32 x 32 units."""
    svg = (ASSETS / f"logo-{theme}.svg").read_text()
    return re.search(r"<svg[^>]*>(.*)</svg>", svg, re.S)[1].strip()


def cat(theme: str) -> str:
    """The logo cat at PIXEL units a pixel, rocking about the middle of its base."""
    values = ";".join(f"{angle} 16 30" for angle, _ in SWAY)
    times = ";".join(f"{t:g}" for _, t in SWAY)
    sway = (
        f'<animateTransform attributeName="transform" type="rotate" values="{values}" keyTimes="{times}" '
        f'dur="{SWAY_S:g}s" repeatCount="indefinite"/>'
    )
    return f'<g transform="scale({PIXEL:g})" shape-rendering="crispEdges"><g>{sway}{logo(theme)}</g></g>'


def outline(font_file: str, weight: int, text: str, size: float) -> str:
    """The text's outline as one SVG path, from its left edge on its baseline."""
    font = instancer.instantiateVariableFont(TTFont(FONTS / font_file), {"wght": weight})
    glyphs, cmap, hmtx = font.getGlyphSet(), font.getBestCmap(), font["hmtx"]
    scale = size / font["head"].unitsPerEm
    pen, advance = SVGPathPen(glyphs), 0.0
    for char in text:
        name = cmap[ord(char)]
        glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, advance * scale, 0)))
        advance += hmtx[name][0]
    return pen.getCommands()


def name() -> str:
    """The name in brand.py's pixel letters, as one path of LETTER-unit squares."""
    return "".join(
        f"M{x * LETTER} {y * LETTER}h{LETTER}v{LETTER}h-{LETTER}z"
        for y, row in enumerate(wordmark())
        for x, pixel in enumerate(row)
        if pixel != "."
    )


def banner(theme: str) -> str:
    colours = THEMES[theme]
    tag = outline("SourceCodePro.ttf", 400, TAGLINE, 26)
    width, height = 1200, 380
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}"
  role="img" aria-label="ippon: {TAGLINE}">
  <title>ippon: {TAGLINE}</title>
  <g transform="translate(64 22)">{cat(theme)}</g>
  <path transform="translate(440 106)" fill="{colours["word"]}" shape-rendering="crispEdges" d="{name()}"/>
  <path transform="translate(440 300)" fill="{colours["tag"]}" d="{tag}"/>
</svg>
'''


for theme in THEMES:
    (OUT / f"banner-{theme}.svg").write_text(banner(theme))
print("wrote", OUT)

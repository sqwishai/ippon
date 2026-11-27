"""ippon's README banner: Inky, Sqwish's octopus, holding a laptop as a die-cut sticker beside the wordmark and tagline,
as two self-contained SVGs, assets/banner-light.svg and assets/banner-dark.svg, and the sticker on its own,
assets/inky-sticker.svg, for pages that show ippon's mark: the blog's ippon post, and the picture of it the terminal
shows (scripts/sticker.py). The name is in brand.py's pixel letters, as the terminal draws it, and the tagline,
brand.TAGLINE, is in a monospaced face.

    uv run --with fonttools python scripts/banner.py FONTS_DIR assets

Inky is assets/inky.svg, Sqwish's drawing with all its poses in one file, copied unchanged from the console
(web/brand/inky.svg in decisionone-workbench). When the drawing changes, copy it again from there rather than editing it
here. The banner sets the book pose, whose arm grips the book's edge, puts an open laptop in the book's place (LID),
hides the ground shadow and crops the drawing to the scene, then cuts it out as a sticker: a white keyline grown round
its silhouette, a hairline and a soft shadow, tilted a little. Inky blinks as it does in the console. FONTS_DIR holds
Source Code Pro's variable font from github.com/google/fonts (OFL), as SourceCodePro.ttf: the tagline is outlined from
it, since GitHub loads no fonts inside a README image.
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
LETTER = 14  # banner units a pixel of the name's letters
# the name in the text colour, as the terminal draws it; the sticker's paper, and how dark its shadow and hairline are,
# so the keyline reads on GitHub's white page and its dark one alike
THEMES = {
    "light": {"word": "#1f2328", "tag": "#4a5568", "paper": "#ffffff", "shadow": 0.2, "hairline": 0.1},
    "dark": {"word": "#e6edf3", "tag": "#9da7b3", "paper": "#fcfaf8", "shadow": 0.55, "hairline": 0},
}
SCENE = (-13.5, 29.5, 203, 144.5)  # the book pose's scene in the drawing's own units: x, y, width, height
# The book's own shapes, from its page tops to its spine: the arm that holds it stays, in front of what replaces it.
BOOK = re.compile(r'<path d="M139\.5 120.*?<path d="M184\.3 120l0 51"[^>]*/>', re.S)
# An open laptop's lid seen from behind, in the laptop pose's colours and with its octopus: the base's edge below it,
# the lid, its lit rim and the octopus on it.
LID = "".join(
    (
        '<path d="M144 175.5l57-4.6 3.6 6.2-58.6 4.9z" fill="#8794A5" class="lf"/>',
        '<path class="lf" fill="#C0C5CB" d="M137.4 120.4q-.1-3.3 3.2-3.6l61.4-4.6q3.3-.2 3.4 3.1l.8 52.4',
        'q.1 3.3-3.2 3.5l-60.6 4.3q-3.3.2-3.4-3.1z"/>',
        '<path d="M141.2 120l61-4.5.8 52" fill="none" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"',
        ' stroke-linecap="round"/>',
        '<g fill="#8794A5" transform="translate(171.5 143.5) scale(1.35) translate(-197.25 -129)">',
        '<path d="M191.5 129.5c-.5-8 11.5-8.5 11 0z"/>',
        '<path d="M192.5 129q-1.5 3.5-4 3.2m7-2.7q-.5 4-2.5 5.5m5.5-5.5q1 4 3 5m0-5.5q2 3 4.5 2.5" fill="none"',
        ' stroke="#8794A5" stroke-width="1.6" stroke-linecap="round"/></g>',
    )
)
TILT = -3  # degrees the sticker is turned, as a sticker is never quite straight
TEXT_X = 644  # banner units from the left where the name and the tagline start


def swap(text: str, old: str, new: str) -> str:
    """`text` with its first `old` replaced. It must be there: a redrawn drawing that moved it stops the banner here."""
    if old not in text:
        raise SystemExit(f"assets/inky.svg no longer has {old!r}: see the docstring of scripts/banner.py")
    return text.replace(old, new, 1)


def inky(x: float, y: float, width: float) -> str:
    """The drawing holding the laptop as an <svg> placed at x, y, `width` wide."""
    svg = re.sub(r"^<!--.*?-->\s*", "", (ASSETS / "inky.svg").read_text(), flags=re.S)
    svg = swap(svg, 'class="octo-inky"', 'class="octo-inky" data-pose="book"')
    svg, found = BOOK.subn(LID, svg, count=1)
    if not found:
        raise SystemExit("assets/inky.svg no longer has the book's shapes: see the docstring of scripts/banner.py")
    svg = swap(svg, "width: 100%; height: auto; ", "")  # placed by the banner, not sized by a page
    svg = swap(svg, "</style>", "  .octo-inky .inky-shadow { display: none; }\n  </style>")
    box = " ".join(f"{v:g}" for v in SCENE)
    height = width * SCENE[3] / SCENE[2]
    return swap(
        svg,
        'viewBox="0 0 200 200" width="200" height="200"',
        f'viewBox="{box}" x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}"',
    )


def sticker(x: float, y: float, width: float, theme: str) -> str:
    """Inky die-cut. Its silhouette, blurred and cut off sharply, grows the white keyline, which closes the gaps
    between its arms as a die does; a hairline and a soft shadow lift it off the page, and the whole is turned TILT
    degrees."""
    t, k = THEMES[theme], width / SCENE[2]  # k: units of the page to a unit of the drawing
    cx, cy = x + width / 2, y + width * SCENE[3] / SCENE[2] / 2
    die = (
        '<filter id="die" x="-12%" y="-14%" width="124%" height="132%" color-interpolation-filters="sRGB">'
        f'<feGaussianBlur in="SourceAlpha" stdDeviation="{2.6 * k:g}" result="blur"/>'
        '<feComponentTransfer in="blur" result="cut">'
        '<feFuncA type="linear" slope="30" intercept="-1.3"/></feComponentTransfer>'
        f'<feGaussianBlur in="cut" stdDeviation="{3.2 * k:g}" result="soft"/>'
        f'<feOffset in="soft" dy="{2.2 * k:g}" result="drop"/>'
        f'<feFlood flood-color="#1e1e1e" flood-opacity="{t["shadow"]:g}"/>'
        '<feComposite in2="drop" operator="in" result="shadow"/>'
        f'<feMorphology in="cut" operator="dilate" radius="{0.45 * k:g}" result="rim"/>'
        f'<feFlood flood-color="#1e1e1e" flood-opacity="{t["hairline"]:g}"/>'
        '<feComposite in2="rim" operator="in" result="hairline"/>'
        f'<feFlood flood-color="{t["paper"]}"/><feComposite in2="cut" operator="in" result="paper"/>'
        '<feMerge><feMergeNode in="shadow"/><feMergeNode in="hairline"/><feMergeNode in="paper"/>'
        '<feMergeNode in="SourceGraphic"/></feMerge></filter>'
    )
    return f'<defs>{die}</defs><g filter="url(#die)" transform="rotate({TILT} {cx:g} {cy:g})">{inky(x, y, width)}</g>'


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
    width, height, art = 1170, 380, 470
    top = (height - art * SCENE[3] / SCENE[2]) / 2
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}"
  role="img" aria-label="ippon: {TAGLINE}">
  <title>ippon: {TAGLINE}</title>
  {sticker(48, top, art, theme)}
  <path transform="translate({TEXT_X} 92)" fill="{colours["word"]}" shape-rendering="crispEdges" d="{name()}"/>
  <path transform="translate({TEXT_X} 286)" fill="{colours["tag"]}" d="{tag}"/>
</svg>
'''


def alone() -> str:
    """The sticker on its own, with room round it for its keyline and shadow."""
    pad = 14
    x, y, w, h = SCENE[0] - pad, SCENE[1] - pad, SCENE[2] + 2 * pad, SCENE[3] + 2 * pad
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x:g} {y:g} {w:g} {h:g}" width="{w:g}" height="{h:g}" '
        'role="img" aria-label="Inky, the Sqwish octopus, holding a laptop">'
        f"{sticker(SCENE[0], SCENE[1], SCENE[2], 'light')}</svg>\n"
    )


for theme in THEMES:
    (OUT / f"banner-{theme}.svg").write_text(banner(theme))
(OUT / "inky-sticker.svg").write_text(alone())
print("wrote", OUT)

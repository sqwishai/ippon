"""ippon's look in the terminal: the lucky cat of its banner (assets/banner-*.svg) and its name, in its colours.

The cat is the banner's 28 x 28 pixels, two to a character cell (an upper half block, the top pixel its foreground
and the bottom one its background), so it is 28 columns by 14 lines. The small cat beside `ippon serve` and `ippon
status` keeps every one of those pixels, six to a cell (Unicode's sextants), widened by half so they come out square:
21 columns by 10 lines. Colours are 24-bit where the terminal says it
takes them (COLORTERM), the nearest of xterm's 256 elsewhere (Apple's Terminal), and none under NO_COLOR. It is shown
only to a person at a terminal: `ippon` alone and the start of `ippon setup`, and the small cat at the start of
`ippon serve` and atop the live `ippon status`.
"""

from __future__ import annotations

import os
import re

# the banner's pixels: b its blue, p its cream, c its orange, o its peach, . none (the terminal's own background)
CAT = (
    ".....bbbbb........bbbbb.....",
    ".....bbbbb........bbbbb.....",
    ".....bpppbbbbbbbbbbpppb.....",
    "....bbpccbbbbbbbbbbccpbbbb..",
    "....bbpccppppppppppccpbbbb..",
    "....bbppppppppppppppppbbpbbb",
    "....bbppppppppppppppppbppbbb",
    "....bbpppbbpppppbbppppbccpbb",
    "....bbppbppbpppbppbpbbbccpbb",
    "....bbpppppppbbpppppbbbpppbb",
    "....bbppppbppppbpbppb.bpppbb",
    "....bbbbbppbbppbbpbbb.bpppbb",
    "......bbbpppppppppbbb.bpppbb",
    "......bbccccccccccccbbbpbbbb",
    "......bbcccccoocccccbbppbb..",
    "......bbppppccccpppppbppbb..",
    "......bbppppcbbccccppbppbb..",
    "bbb..bbbpbbpppccooccpbppbb..",
    "bpb..bbppbbbbbbbooccpbbbbb..",
    "bpb..bbppbbppppbbbccppbb....",
    "bpb..bbppppbbbbbooccppbb....",
    "bpb..bbpppppppccbbccppbb....",
    "bbbbbbbpppppppccccccppbb....",
    "..bpppbppppppppccccpppbb....",
    "..bbbbbpbppbppppppbppbbb....",
    "....bbppbppbppppppbppbpbb...",
    "....bbbbbbbbbbbbbbbbbbbbb...",
    "....bbbbbbbbbbbbbbbbbbbbb...",
)
# the name in the same pixels, lowercase as the banner writes it: 8 high to the baseline with two-pixel strokes, and two
# rows under it for the p's descender, so it reads as a p and never as an o
LETTERS = {
    "i": ("##", "..", "..", "##", "##", "##", "##", "##", "..", ".."),
    "p": ("......", "......", "......", "#####.", "##..##", "##..##", "##..##", "#####.", "##....", "##...."),
    "o": ("......", "......", "......", ".####.", "##..##", "##..##", "##..##", ".####.", "......", "......"),
    "n": ("......", "......", "......", "#####.", "##..##", "##..##", "##..##", "##..##", "......", "......"),
}
COLOURS = {"b": (0x39, 0x87, 0xE5), "p": (0xFF, 0xFA, 0xF3), "c": (0xD9, 0x59, 0x26), "o": (0xFF, 0xBB, 0x80)}
# Sqwish's accents (sqwish.ai's design system), used sparingly and in this order: pink for what is in hand
# (the line pointed at, a ticked box, work going on), mint for what is settled (a check mark, an answer, the
# recommendation), amber for what to do next (a command to run, a threshold to check), and red for answers that
# may change (a model turned below 8 bits)
ACCENTS = {
    "pink": (0xE8, 0x8B, 0xD5),
    "mint": (0x4A, 0xDE, 0x80),
    "amber": (0xF5, 0xA6, 0x23),
    "red": (0xF0, 0x5A, 0x5A),
}
TAGLINE = "run decision models on your mac"


def _code(rgb: tuple[int, int, int], background: bool) -> str:
    """The SGR code setting `rgb` as foreground or background: 24-bit, or xterm's nearest 6 x 6 x 6 cube colour."""
    if os.environ.get("COLORTERM", "") in ("truecolor", "24bit"):
        return f"{48 if background else 38};2;{rgb[0]};{rgb[1]};{rgb[2]}"
    cube = [0, 95, 135, 175, 215, 255]
    index = 16 + sum(min(range(6), key=lambda i: abs(cube[i] - v)) * m for v, m in zip(rgb, (36, 6, 1), strict=True))
    return f"{48 if background else 38};5;{index}"


def accent(name: str) -> str:
    """The SGR code of one of the accents as a foreground."""
    return _code(ACCENTS[name], False)


def pixels(grid: tuple[str, ...] | list[str], colour: bool) -> list[str]:
    """`grid`'s pixels as half-block lines, two rows of pixels to a line; spaces where a cell has none, and the
    terminal's own foreground for pixels marked f."""
    rows = list(grid) + (["." * len(grid[0])] if len(grid) % 2 else [])
    out = []
    for top, bottom in zip(rows[::2], rows[1::2], strict=True):
        line = []
        for up, down in zip(top, bottom, strict=True):
            if up == "." and down == ".":
                line.append(" ")
            elif not colour or {up, down} <= {"f", "."}:
                line.append("▀" if down == "." else "▄" if up == "." else "█")
            elif down == ".":
                line.append(f"\x1b[{_code(COLOURS[up], False)}m▀\x1b[0m")
            elif up == ".":
                line.append(f"\x1b[{_code(COLOURS[down], False)}m▄\x1b[0m")
            else:
                line.append(f"\x1b[{_code(COLOURS[up], False)};{_code(COLOURS[down], True)}m▀\x1b[0m")
        out.append("".join(line))
    return out


def wordmark(word: str = "ippon", ink: str = "f") -> list[str]:
    """`word` in the pixel letters, as a grid in colour `ink`, a pixel between letters."""
    return [".".join(LETTERS[ch][row] for ch in word).replace("#", ink) for row in range(len(LETTERS[word[0]]))]


def banner(colour: bool, lines: list[str] = ()) -> list[str]:  # type: ignore[assignment]
    """The cat with the name and the tagline beside it, then `lines` (the version, this Mac) under the tagline."""
    cat, name = pixels(CAT, colour), pixels(wordmark(), colour)
    dim = (lambda text: f"\x1b[2m{text}\x1b[0m") if colour else (lambda text: text)
    right = ["", "", "", *name, "", TAGLINE, "", *(dim(line) for line in lines)]
    return [
        f"  {left}   {text}".rstrip() for left, text in zip(cat, right + [""] * (len(cat) - len(right)), strict=False)
    ]


# terminals known to draw the sextants themselves; elsewhere (Apple's Terminal) a font may lack them, so the full cat
SEXTANT_TERMINALS = ("iTerm.app", "ghostty", "WezTerm", "vscode", "kitty")


def _sextant(bits: int) -> str:
    """The character with the cells of `bits` set: 1 top left, 2 top right, 4 middle left, ... 32 bottom right.
    Unicode has every pattern but the empty, full, left and right halves among its sextants."""
    named = {0: " ", 63: "█", 21: "▌", 42: "▐"}
    return named.get(bits) or chr(0x1FB00 + bits - 1 - (bits > 21) - (bits > 42))


def sextants(grid: tuple[str, ...] | list[str], colour: bool) -> list[str]:
    """`grid`'s pixels six to a character cell, two across and three down. A cell holds two colours, so where three
    meet in one, the blue outline is drawn and the rest takes the cell's other colour."""
    rows = [row + "." * (len(row) % 2) for row in grid]
    rows += ["." * len(rows[0])] * (-len(rows) % 3)
    out = []
    for y in range(0, len(rows), 3):
        line = []
        for x in range(0, len(rows[0]), 2):
            cell = [rows[y + j][x + i] for j in range(3) for i in range(2)]
            inks = sorted({p for p in cell if p != "."}, key=lambda p: (p != "b", -cell.count(p)))
            if not inks:
                line.append(" ")
            elif not colour:
                line.append(_sextant(sum(1 << n for n, p in enumerate(cell) if p != ".")))
            elif "." in cell or len(inks) == 1:  # one colour over the terminal's own background
                bits = sum(1 << n for n, p in enumerate(cell) if p != ".")
                line.append(f"\x1b[{_code(COLOURS[inks[0]], False)}m{_sextant(bits)}\x1b[0m")
            else:  # the first colour drawn, the second behind it
                bits = sum(1 << n for n, p in enumerate(cell) if p == inks[0])
                code = f"{_code(COLOURS[inks[0]], False)};{_code(COLOURS[inks[1]], True)}"
                line.append(f"\x1b[{code}m{_sextant(bits)}\x1b[0m")
        out.append("".join(line))
    return out


def small_cat(colour: bool) -> list[str]:
    """The cat at its smallest that keeps every pixel (21 x 10) where the terminal draws sextants, else the full cat."""
    if os.environ.get("TERM_PROGRAM") not in SEXTANT_TERMINALS and "kitty" not in os.environ.get("TERM", ""):
        return pixels(CAT, colour)
    wide = ["".join(row[x * 2 // 3] for x in range(len(row) * 3 // 2)) for row in CAT]
    return sextants(wide, colour)


def beside(art: list[str], lines: list[str], width: int) -> list[str]:
    """`lines` to the right of `art`, centred against it, with `art` dropped when the two don't fit in `width`
    columns. `art` starts at the left edge, as the lines printed under it do, and GAP columns keep it from `lines`."""
    room = len(_plain(art[0])) if art else 0
    if not art or room + len(GAP) + max((len(_plain(line)) for line in lines), default=0) > width:
        return lines
    top = max(0, (len(art) - len(lines)) // 2)
    right = [""] * top + lines + [""] * max(0, len(art) - len(lines) - top)
    left = art + [" " * room] * (len(right) - len(art))
    return [f"{a}{GAP}{b}".rstrip() for a, b in zip(left, right, strict=True)]


GAP = "   "  # between the cat and the lines beside it, as in the banner


def _plain(text: str) -> str:
    """`text` without its colour codes, to measure it."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)

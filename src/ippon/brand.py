"""ippon's look in the terminal: its name in pixel letters, Inky's sticker beside it where the terminal draws pictures,
and the accents its screens use.

The name is drawn two pixels to a character cell, an upper half block with the top pixel its foreground and the bottom
one its background, in the terminal's own colour, so it reads on light and dark terminals alike. Inky, Sqwish's octopus
holding a laptop, is the README's sticker (assets/inky-sticker.svg, drawn into inky.png by scripts/sticker.py), shown as
a picture where the terminal draws pictures: iTerm2 and WezTerm by iTerm2's inline images, kitty and Ghostty by kitty's
graphics protocol. Everywhere else (Apple's Terminal, VS Code's terminal, tmux and screen) the banner shows Inky in
pixels (INKY, 40 x 28, drawn by hand from the sticker's drawing), two to a cell, beside the name where the window has
room, and `ippon serve` its lines alone, since smaller he is only a blur. Colours are 24-bit where the terminal says it
takes them (COLORTERM), the nearest of xterm's 256 elsewhere, and none under NO_COLOR, and with no colour there is no
picture either. The banner opens `ippon` alone and `ippon setup`, and the sticker, smaller, sits beside the header
`ippon serve` prints once it is ready. Both show only to a person at a terminal.
"""

from __future__ import annotations

import base64
import fcntl
import os
import re
import shutil
import struct
import sys
import termios
from collections.abc import Sequence
from functools import cache
from importlib.resources import files
from typing import IO, Literal

# the name in pixel letters, lowercase as the README's banner writes it: 8 high to the baseline with two-pixel
# strokes, and two rows under it for the p's descender, so it reads as a p and never as an o
LETTERS = {
    "i": ("##", "..", "..", "##", "##", "##", "##", "##", "..", ".."),
    "p": ("......", "......", "......", "#####.", "##..##", "##..##", "##..##", "#####.", "##....", "##...."),
    "o": ("......", "......", "......", ".####.", "##..##", "##..##", "##..##", ".####.", "......", "......"),
    "n": ("......", "......", "......", "#####.", "##..##", "##..##", "##..##", "##..##", "......", "......"),
}
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
# Inky holding a laptop in pixels, two to a character cell, for a terminal that draws no pictures (Apple's Terminal):
# the sticker's drawing at 40 x 28, so he and the name fit an 80-column window, each pixel the drawing's own colour it
# covers most, its outline kept where it covers half a pixel, and the eyes and the mouth drawn by hand. Smaller, he is
# only a blur, so `ippon serve` shows him only as a picture.
# n outline, w white, p body, m shade, c cheek, g laptop, s its octopus.
INKY = (
    "................nnnn....................",
    ".............nnwwwwwwwn.................",
    "............nwwpppppppwwn...............",
    "..........nwwpppppppppppwn..............",
    ".........nwwppppppppppppppw.............",
    ".........wwppppppppppppppppn............",
    "........nwmpwpppppppppppppppn...........",
    "........wmmpppppppppppppppppw...........",
    ".......nwmmppppppppppppppppppn..........",
    ".......nwmmpppppppppppppppnnnn..........",
    ".......nwmmpppppwwwwpppppnnwwn..........",
    ".......nwmmppppwnnwwnppppnnwwnnnnnn.....",
    "..nnn..nmmmppppwnnwwnppppnnnnnnppppn....",
    ".nppppnnnmmmpppwnnnwnpnnnpnwnpppnnpnn...",
    ".mmmmppnnmmmpppwnwnnnpnmnnnnnnnnnnnwgn..",
    "nmwnnmppnmmmmpppwnnnpppppngggggggggggn..",
    "nnnnnnmpnnmmmmpppccppppmpngggggggggnnn..",
    ".....nnmpnnnmmpppppppppppngggggggggnpn..",
    "..n...nmmmnnnmmmpppppppppnggggssggggppn.",
    ".nn...nnmnnnnnnnmmppppppnnggggsssgggnpn.",
    ".nmn..nnnwpnmmmmmmppppppnngggggggggggpnn",
    ".nmmmnnmnpnmmmmmmnpppppppngggggggggggmnn",
    "..nwmmmmnwpnnnnnpppppnpppngggggggggmmmn.",
    "...nwwmnnnwppppppwwnnnnppnggggggggnnnn..",
    "....nnnnnnnnwwnwnnnn..nnpnnnnnnnnnnnn...",
    "...........nnwnwnnn....nnpnnnnnnnnnnn...",
    "...............n........nnnnnn..........",
    "........................................",
)
INKY_COLOURS = {
    "n": (0x2E, 0x1F, 0x57),
    "w": (0xFF, 0xFF, 0xFF),
    "p": (0xFB, 0xB2, 0xB6),
    "m": (0xC3, 0x48, 0x7F),
    "c": (0xF4, 0x8C, 0xA1),
    "g": (0xC0, 0xC5, 0xCB),
    "s": (0x87, 0x94, 0xA5),
}
GAP = "   "  # between the sticker and the lines beside it
Pictures = Literal["iterm", "kitty"]


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


def wordmark(word: str = "ippon", ink: str = "#") -> list[str]:
    """`word` in the pixel letters, as a grid in `ink` and '.', a pixel between letters."""
    return [".".join(LETTERS[ch][row] for ch in word).replace("#", ink) for row in range(len(LETTERS[word[0]]))]


def name(colour: bool = False) -> list[str]:
    """ippon's name in its pixel letters, two pixels to a character cell, in the terminal's own colour. With colour,
    a cell inked top and bottom is a reversed space rather than a full block: Apple's Terminal draws block characters
    from its font (SF Mono) short of the line, which left the letters in stripes, and a cell's background fills all of
    it in every terminal. SGR can't name the terminal's own foreground as a background, so the space is reversed."""
    grid = wordmark()
    full = "\x1b[7m \x1b[0m" if colour else "█"
    lines = []
    for top, bottom in zip(grid[::2], grid[1::2], strict=True):
        cells = (
            full if up != "." and down != "." else "▀" if up != "." else "▄" if down != "." else " "
            for up, down in zip(top, bottom, strict=True)
        )
        lines.append("".join(cells).rstrip())
    return lines


def inky() -> list[str]:
    """Inky in colour, two pixels to a cell, every line INKY's full width. A cell of one colour is a space on it, as a
    cell's background fills all of it where Apple's Terminal draws a block character short of the line (SF Mono)."""
    rows = []
    for top, bottom in zip(INKY[::2], INKY[1::2], strict=True):
        cells = []
        for up, down in zip(top, bottom, strict=True):
            if up == "." and down == ".":
                cells.append(" ")
            elif up == down:
                cells.append(f"\x1b[{_code(INKY_COLOURS[up], True)}m \x1b[0m")
            elif down == ".":
                cells.append(f"\x1b[{_code(INKY_COLOURS[up], False)}m▀\x1b[0m")
            elif up == ".":
                cells.append(f"\x1b[{_code(INKY_COLOURS[down], False)}m▄\x1b[0m")
            else:
                cells.append(f"\x1b[{_code(INKY_COLOURS[up], False)};{_code(INKY_COLOURS[down], True)}m▀\x1b[0m")
        rows.append("".join(cells))
    return rows


def in_pixels(lines: Sequence[str], width: int, indent: str = "") -> list[str] | None:
    """`lines` beside Inky in pixels (`inky`), centred against him, `indent` in from the left edge; None where
    `width` columns leave no room for him and the longest line."""
    art = inky()
    if len(indent) + len(INKY[0]) + len(GAP) + max((len(_plain(line)) for line in lines), default=0) > width:
        return None
    top = max(0, (len(art) - len(lines)) // 2)
    beside = [""] * top + list(lines) + [""] * max(0, len(art) - len(lines) - top)
    blank = " " * len(INKY[0])
    return [
        f"{indent}{a}{GAP}{b}".rstrip()
        for a, b in zip([*art, *[blank] * (len(beside) - len(art))], beside, strict=True)
    ]


@cache
def _sticker_png() -> bytes:
    return (files("ippon") / "inky.png").read_bytes()


def _aspect() -> float:
    """The sticker's width over its height, from its PNG header."""
    width, height = struct.unpack(">II", _sticker_png()[16:24])
    return float(width) / float(height)


def pictures(stream: IO[str]) -> Pictures | None:
    """How the terminal on `stream` draws pictures, or None where it doesn't, or ippon can't tell that it does.
    iTerm2 passes LC_TERMINAL over SSH, so a server's banner reaches the Mac's iTerm2 too. tmux and screen pass a
    picture on only when asked to, so inside them there is none."""
    if not stream.isatty() or os.environ.get("TMUX") or os.environ.get("TERM", "").startswith(("screen", "tmux")):
        return None
    program = os.environ.get("TERM_PROGRAM", "")
    # LC_TERMINAL only where TERM_PROGRAM says nothing (over SSH): a terminal opened from iTerm2 inherits it, and
    # Apple's Terminal with it drew no picture, leaving a blank where Inky was meant to be
    if program in ("iTerm.app", "WezTerm") or (not program and os.environ.get("LC_TERMINAL") == "iTerm2"):
        return "iterm"
    kitty = os.environ.get("TERM") in ("xterm-kitty", "xterm-ghostty") or os.environ.get("KITTY_WINDOW_ID")
    if kitty or program == "ghostty":
        return "kitty"
    return None


def _cell_shape(stream: IO[str]) -> float:
    """A character cell's width over its height, from the window's size in pixels, or 0.5 where the terminal doesn't
    say (a cell is about twice as tall as it is wide in every monospaced font)."""
    try:
        rows, columns, x, y = struct.unpack("HHHH", fcntl.ioctl(stream.fileno(), termios.TIOCGWINSZ, b"\0" * 8))
    except (OSError, ValueError):
        return 0.5
    return float(x / columns) / float(y / rows) if x and y and rows and columns else 0.5


def _picture(how: Pictures, columns: int, rows: int) -> str:
    """The sticker as the escape sequence that draws it over `columns` x `rows` cells, leaving the cursor where it is
    for kitty (C=1) and asking it for no reply (q=2), which would otherwise arrive at the shell's prompt as text."""
    png = _sticker_png()
    data = base64.b64encode(png).decode()
    if how == "iterm":
        return f"\x1b]1337;File=inline=1;size={len(png)};width={columns};height={rows};preserveAspectRatio=1:{data}\x07"
    chunks = [data[i : i + 4096] for i in range(0, len(data), 4096)]  # kitty's largest chunk
    keys = f"a=T,f=100,q=2,C=1,c={columns},r={rows}"
    return "".join(
        f"\x1b_G{keys + ',' if i == 0 else ''}m={int(i < len(chunks) - 1)};{chunk}\x1b\\"
        for i, chunk in enumerate(chunks)
    )


def sticker(lines: Sequence[str], stream: IO[str], width: int, most: int, indent: str = "") -> list[str] | None:
    """`lines` beside Inky's sticker, `indent` in from the left edge, the sticker at most `most` columns wide and as
    wide as `width` columns leave room for, the lines centred against it: lines to print one after another, or None
    where the terminal on `stream` draws no pictures or has no room for a sticker 20 columns wide.

    The first line makes room for the whole sticker (blank lines, so the screen scrolls before anything is drawn),
    goes back to the top of that room and draws the sticker with the cursor kept, so every line beside it is written
    after the picture, never ahead of a picture still arriving."""
    how = pictures(stream)
    columns = min(most, width - len(indent) - len(GAP) - max((len(_plain(line)) for line in lines), default=0))
    if how is None or columns < 20:
        return None
    high = max(1, round(columns * _cell_shape(stream) / _aspect()))  # the sticker's rows, at its own shape
    rows = max(high, len(lines))
    top = (rows - len(lines)) // 2
    beside = [""] * top + list(lines) + [""] * (rows - len(lines) - top)
    at = len(indent) + columns + len(GAP) + 1  # the column the lines start in, counting from 1

    def place(line: str) -> str:
        return f"\x1b[{at}G{line}" if line else ""

    room = "\n" * (rows - 1) + (f"\x1b[{rows - 1}A" if rows > 1 else "")
    first = room + f"\x1b7\x1b[{len(indent) + 1}G" + _picture(how, columns, high) + "\x1b8" + place(beside[0])
    return [first, *(place(line) for line in beside[1:])]


def banner(
    colour: bool, lines: Sequence[str] = (), stream: IO[str] | None = None, width: int | None = None
) -> list[str]:
    """The banner: ippon's name, its tagline and `lines` (the version, this Mac) under them, beside Inky's sticker
    where the terminal draws pictures, else alone, on one left edge two columns in. Lines too long to sit beside the
    sticker go under it, so the sticker keeps its size on an 80-column terminal."""
    stream = stream or sys.stdout
    width = width or shutil.get_terminal_size().columns - 1
    dim = (lambda text: f"\x1b[2m{text}\x1b[0m") if colour else (lambda text: text)
    words, notes = [*name(colour), "", TAGLINE], [dim(line) for line in lines]
    under = ["", *notes] if notes else []
    if colour:
        if (both := sticker([*words, *under], stream, width, 44, "  ")) is not None:
            return both
        if (alone := sticker(words, stream, width, 44, "  ")) is not None:
            return [*alone, *(f"  {line}" if line else "" for line in under)]
        if (drawn := in_pixels([*words, *under], width, "  ")) is not None:  # no pictures here: Inky in pixels
            return drawn
        if (drawn := in_pixels(words, width, "  ")) is not None:
            return [*drawn, *(f"  {line}" if line else "" for line in under)]
    return [f"  {line}" if line else "" for line in [*words, *under]]


def _plain(text: str) -> str:
    """`text` without its colour codes, to measure it."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)

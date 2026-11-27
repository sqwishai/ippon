"""ippon's look in the terminal: its name in pixel letters, Inky's sticker beside it where the terminal draws pictures,
and the accents its screens use.

The name is drawn two pixels to a character cell, an upper half block with the top pixel its foreground and the bottom
one its background, in the terminal's own colour, so it reads on light and dark terminals alike. Inky, Sqwish's octopus
at a laptop, is the README's sticker (assets/inky-sticker.svg, drawn into inky.png by scripts/sticker.py), shown as a
picture where the terminal draws pictures: kitty, Ghostty and iTerm2 3.6 and newer by kitty's graphics protocol, and
WezTerm by iTerm2's inline images. iTerm2 itself asks before it shows each of its own inline images ("Allow
Terminal-Initiated Display?"), and a No there prints the picture as base64 text, so ippon never sends it one. Everywhere
else (Apple's Terminal, VS Code's terminal, tmux and screen) ippon shows its name and its lines alone, on one left edge,
since Inky drawn in character cells is only a blur of him. Colours are 24-bit where the terminal says it takes them
(COLORTERM), the nearest of xterm's 256 elsewhere, and none under NO_COLOR, and with no colour there is no picture
either. The banner opens `ippon` alone and `ippon setup`, and the sticker, smaller, sits beside the header `ippon serve`
prints once it is ready and beside the server's state `ippon status` watches. They show only to a person at a terminal.
"""

from __future__ import annotations

import base64
import fcntl
import os
import re
import secrets
import shutil
import struct
import sys
import termios
from collections.abc import Sequence
from dataclasses import dataclass
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
GAP = "   "  # between the sticker and the lines beside it
Pictures = Literal["iterm", "kitty"]
# The id kitty keeps this process's sticker under, random as kitty asks of programs so one never replaces another's,
# and under 2**24, as the sticker's cells carry it in their 24-bit colour
IMAGE = 1 + secrets.randbelow(2**24 - 1)
# kitty's marks for a placeholder cell's row and column, in kitty's own order (its gen/rowcolumn-diacritics.txt): the
# first 60, for a sticker up to 60 cells wide or high
MARKS = (
    "\u0305\u030d\u030e\u0310\u0312\u033d\u033e\u033f\u0346\u034a\u034b\u034c\u0350\u0351\u0352\u0357\u035b\u0363"
    "\u0364\u0365\u0366\u0367\u0368\u0369\u036a\u036b\u036c\u036d\u036e\u036f\u0483\u0484\u0485\u0486\u0487\u0592"
    "\u0593\u0594\u0595\u0597\u0598\u0599\u059c\u059d\u059e\u059f\u05a0\u05a1\u05a8\u05a9\u05ab\u05ac\u05af\u05c4"
    "\u0610\u0611\u0612\u0613\u0614\u0615"
)


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
    # Apple's Terminal with it drew no picture, leaving a blank where Inky was meant to be. iTerm2 draws kitty's
    # pictures without asking from 3.6 (it asks before its own inline images from 3.5), so an older one gets none.
    if program == "iTerm.app" or (not program and os.environ.get("LC_TERMINAL") == "iTerm2"):
        version = os.environ.get("TERM_PROGRAM_VERSION" if program else "LC_TERMINAL_VERSION", "")
        return "kitty" if tuple(int(part) for part in re.findall(r"\d+", version)[:2]) >= (3, 6) else None
    if program == "WezTerm":
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
    """The sticker as the escape sequence that draws it over `columns` x `rows` cells from the cursor.

    kitty's picture is sent once, asking for no reply (q=2, on every chunk, as iTerm2 takes the answer from the last
    one), which would otherwise arrive at the shell's prompt as text. It is then drawn as kitty's Unicode placeholders,
    one cell of text for each cell of him, so he belongs to the screen he was drawn on, scrolls with the lines beside
    him and is cleared with them. iTerm2 keeps a picture drawn the other way over both screens: `ippon status`, on the
    other screen, clearing it erased the banner's Inky on the user's own, and showed him over it when it didn't."""
    png = _sticker_png()
    data = base64.b64encode(png).decode()
    if how == "iterm":
        return f"\x1b]1337;File=inline=1;size={len(png)};width={columns};height={rows};preserveAspectRatio=1:{data}\x07"
    chunks = [data[i : i + 4096] for i in range(0, len(data), 4096)]  # kitty's largest chunk
    keys = f"a=T,U=1,f=100,q=2,i={IMAGE},c={columns},r={rows}"
    sent = "".join(
        f"\x1b_G{keys if i == 0 else 'q=2'},m={int(i < len(chunks) - 1)};{chunk}\x1b\\"
        for i, chunk in enumerate(chunks)
    )
    colour = f"\x1b[38;2;{IMAGE >> 16};{IMAGE >> 8 & 255};{IMAGE & 255}m"
    cells = (
        colour + "".join(f"\U0010eeee{MARKS[row]}{MARKS[column]}" for column in range(columns)) + "\x1b[39m"
        for row in range(rows)
    )
    return sent + f"\x1b[{columns}D\x1b[B".join(cells)


@dataclass(frozen=True)
class Placement:
    """Inky's sticker beside some lines: how the terminal draws it, its size in cells, and the column the lines
    beside it start in, counting from 1."""

    how: Pictures
    columns: int
    high: int
    at: int

    def picture(self) -> str:
        return _picture(self.how, self.columns, self.high)

    def rows(self, lines: Sequence[str]) -> list[str]:
        """`lines` centred against the sticker, one a row, as many rows as the sticker or the lines take."""
        rows = max(self.high, len(lines))
        top = (rows - len(lines)) // 2
        return [""] * top + list(lines) + [""] * (rows - len(lines) - top)


def place(lines: Sequence[str], stream: IO[str], width: int, most: int, indent: str = "") -> Placement | None:
    """Where Inky's sticker goes beside `lines`, `indent` in from the left edge, at most `most` columns wide and as
    wide as `width` columns leave room for, or None where the terminal on `stream` draws no pictures or has no room
    for a sticker 20 columns wide."""
    how = pictures(stream)
    columns = min(most, width - len(indent) - len(GAP) - max((len(_plain(line)) for line in lines), default=0))
    if how is None or columns < 20:
        return None
    high = max(1, round(columns * _cell_shape(stream) / _aspect()))  # the sticker's rows, at its own shape
    return Placement(how, columns, high, len(indent) + columns + len(GAP) + 1)


def sticker(lines: Sequence[str], stream: IO[str], width: int, most: int, indent: str = "") -> list[str] | None:
    """`lines` beside Inky's sticker (placed as `place` does), the lines centred against it: lines to print one after
    another, or None where there is no sticker.

    The first line makes room for the whole sticker (blank lines, so the screen scrolls before anything is drawn),
    goes back to the top of that room and draws the sticker with the cursor kept, so every line beside it is written
    after the picture, never ahead of a picture still arriving."""
    if (placed := place(lines, stream, width, most, indent)) is None:
        return None
    beside = placed.rows(lines)

    def put(line: str) -> str:
        return f"\x1b[{placed.at}G{line}" if line else ""

    room = "\n" * (len(beside) - 1) + (f"\x1b[{len(beside) - 1}A" if len(beside) > 1 else "")
    first = room + f"\x1b7\x1b[{len(indent) + 1}G" + placed.picture() + "\x1b8" + put(beside[0])
    return [first, *(put(line) for line in beside[1:])]


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
    return [f"  {line}" if line else "" for line in [*words, *under]]


def _plain(text: str) -> str:
    """`text` without its colour codes, to measure it."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)

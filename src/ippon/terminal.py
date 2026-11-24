"""Questions asked in the terminal: a list to pick one from, a checklist, a choice of a few options and yes or no.

They read the keyboard from /dev/tty, so they work under `curl ... | sh`, where standard input is the script. On a
terminal that moves its cursor (any but TERM=dumb) a question is drawn once and redrawn in place as the arrow keys
(or j and k) move, space ticks a checklist line and Enter answers. Elsewhere it is a numbered list answered by typing a
number. Enter alone always takes the line marked as the default. Colour goes when NO_COLOR is set. Ctrl-C raises
KeyboardInterrupt, as everywhere else in ippon.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
import select
import shutil
import sys
import termios
import textwrap
import threading
import tty
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import IO, Any

from . import brand

UP, DOWN, LEFT, RIGHT, ENTER, SPACE, BACK = "up", "down", "left", "right", "enter", "space", "back"
KEYS = {"\x1b[A": UP, "\x1bOA": UP, "k": UP, "\x1b[B": DOWN, "\x1bOB": DOWN, "j": DOWN, "\r": ENTER, "\n": ENTER}
KEYS |= {"\x1b[D": LEFT, "\x1bOD": LEFT, "h": LEFT, "\x1b[C": RIGHT, "\x1bOC": RIGHT, "l": RIGHT}
KEYS |= {"\x1b": BACK, "\x7f": BACK, "\x08": BACK}  # Esc alone, and Backspace: as most terminal prompts go back
ESCAPE_WAIT_S = 0.05  # after an Esc, how long the rest of an arrow key's sequence may take to arrive


COMMAND = re.compile(r"`([^`\n]+)`")  # a command or name a message quotes for the reader to type


def percent(p: float) -> str:
    return "<0.1%" if p < 0.0005 else f"{p:.1%}"


def said(answer: Mapping[str, Any]) -> str:
    """One question's answer as a person reads it: "yes 99.1%", a choice's option and its probability, a score's
    nearest level named with its value and confidence ("Needs help today (0.90) 68.2%")."""
    kind = answer.get("type")
    if kind == "noul":
        p = float(answer["noul"])
        return f"yes {percent(p)}" if p >= 0.5 else f"no {percent(1 - p)}"
    if kind == "choice":
        top = answer["probabilities"].get(answer["choice"], answer["confidence"])  # its probability: the line adds up
        return f"{answer['choice']} {percent(top)}"
    if kind == "score":
        score = float(answer["score"])  # a whole level, or a creator's expected level (Kettei's)
        shown = f"{score:g}" if score.is_integer() else f"{score:.2f}"
        name = (answer.get("legend") or {}).get(str(round(score)), "")  # the nearest level's name
        return (
            f"{name} ({shown}) {percent(answer['confidence'])}" if name else f"{shown} {percent(answer['confidence'])}"
        )
    return json.dumps(answer, ensure_ascii=False)


def colour_on(stream: IO[str]) -> bool:
    """Whether text written to `stream` reaches a person at a terminal that shows colour (not a pipe, a dumb
    terminal or NO_COLOR)."""
    return stream.isatty() and os.environ.get("TERM", "") not in ("", "dumb") and "NO_COLOR" not in os.environ


def styled(text: str, colour: bool, *codes: str) -> str:
    """`text` in SGR `codes` ("1" bold, "2" dim) or the brand's accents (brand.ACCENTS) when `colour` is on
    (`colour_on`)."""
    sgr = [brand.accent(code) if code in brand.ACCENTS else code for code in codes]
    return f"\x1b[{';'.join(sgr)}m{text}\x1b[0m" if colour and sgr else text


def fitted(parts: Sequence[tuple[str, tuple[str, ...]]], colour: bool, width: int | None = None) -> str:
    """`parts`, each a text with its SGR codes or accents (`styled`), as one line: cut with an ellipsis to `width`
    columns when given, so a long line at a terminal is shortened instead of wrapping onto the next."""
    out, room = [], width
    for text, codes in parts:
        if room is not None and len(text) > room:
            text = text[: max(room - 1, 0)] + "…" if room else ""
        out.append(styled(text, colour, *codes))
        if room is not None:
            room -= len(text)
            if room <= 0:
                break
    return "".join(out)


def compact(n: float) -> str:
    """A count as people say it: 950, 10.3k, 225k, 8.6M."""
    if n < 1000:
        return f"{n:.0f}"
    if n < 1_000_000:
        return f"{n / 1000:.1f}k" if n < 100_000 else f"{n / 1000:.0f}k"
    return f"{n / 1_000_000:.1f}M"


def ago(seconds: float) -> str:
    """How long ago, as people say it: 12 s, 2 min, 1 h."""
    if seconds < 60:
        return f"{seconds:.0f} s ago"
    return f"{seconds / 60:.0f} min ago" if seconds < 3600 else f"{seconds / 3600:.0f} h ago"


def bar(share: float, cells: int) -> str:
    """A thin bar of `share` (0 to 1) of `cells` character cells, to half a cell, padded to `cells`."""
    halves = round(max(0.0, min(share, 1.0)) * cells * 2)
    return ("━" * (halves // 2) + ("╸" if halves % 2 else "")).ljust(cells)


def commands(text: str, colour: bool) -> str:
    """`text` with each `quoted` command in bold amber instead of backticks, as `ippon` alone lists its commands, so a
    person spots what to type. Without colour (a pipe, an agent, JSON) the backticks stay: they are what marks the
    command there."""
    if not colour:
        return text
    return COMMAND.sub(lambda match: styled(match[1], colour, "1", "amber"), text)


class Back(Exception):
    """Raised by a question asked with `back=True` when the user goes back to the one before it."""


@dataclass(frozen=True)
class Line:
    """One line of a list: its text, a dimmed note after it, and whether it can be chosen."""

    text: str
    note: str = ""
    enabled: bool = True
    tone: str = "2"  # how the note is shown: dim, or an accent
    answer: str = ""  # what stays on screen once the line is chosen, where the text's first column isn't enough


class Terminal:
    """The user's terminal: its keyboard (/dev/tty) and the output questions are drawn on."""

    def __init__(self, keyboard: IO[str], out: IO[str] = sys.stderr) -> None:
        self.keyboard, self.out = keyboard, out
        self.moves = os.environ.get("TERM", "") not in ("", "dumb")
        self.colour = self.moves and "NO_COLOR" not in os.environ
        self.shown: list[str] = []  # what stays on screen (said lines and answered questions), for `rewind`

    @classmethod
    @contextlib.contextmanager
    def open(cls) -> Iterator[Terminal | None]:
        """The terminal, or None when there is none to ask (a pipe, CI, an agent's shell)."""
        try:
            keyboard = open("/dev/tty", encoding="utf-8")  # noqa: SIM115 - closed below
        except OSError:
            yield None
            return
        with keyboard:
            yield cls(keyboard) if keyboard.isatty() else None

    # -- drawing ----------------------------------------------------------------------------------------------------

    def style(self, text: str, *codes: str) -> str:
        """`text` in SGR `codes` (`styled`) when this terminal shows colour."""
        return styled(text, self.colour, *codes)

    def say(self, text: str = "") -> None:
        print(text, file=self.out, flush=True)
        self.shown.append(text)

    def mark(self) -> int:
        """Where the screen is now, for `rewind` to come back to."""
        return len(self.shown)

    def rewind(self, mark: int) -> None:
        """The screen as it was at `mark`: what came after it (answers taken back with ←) cleared away. The screen is
        drawn again from what stays on it, so long lines that wrapped come back as they were."""
        del self.shown[mark:]
        if self.moves:
            self.out.write("\x1b[H\x1b[J" + "".join(line + "\n" for line in self.shown))
            self.out.flush()

    def _width(self) -> int:
        return max(40, shutil.get_terminal_size((100, 24)).columns - 1)

    def _clip(self, text: str) -> str:
        """`text` cut to the terminal's width, so no line wraps and a redraw moves up exactly as far as it drew."""
        width, shown, out, escape = self._width(), 0, [], False
        for char in text:
            if char == "\x1b":
                escape = True
            if not escape:
                if shown == width:
                    break
                shown += 1
            elif char == "m":
                escape = False
                out.append(char)
                continue
            out.append(char)
        return "".join(out) + ("\x1b[0m" if self.colour else "")

    @contextlib.contextmanager
    def _drawing(self) -> Iterator[Callable[[list[str]], None]]:
        """A function that draws its lines in place of the ones it drew before, the cursor hidden meanwhile."""
        drawn = [0]

        def draw(lines: list[str]) -> None:
            if drawn[0]:
                self.out.write(f"\x1b[{drawn[0]}F\x1b[J")
            self.out.write("".join(self._clip(line) + "\n" for line in lines))
            self.out.flush()
            drawn[0] = len(lines)

        self.out.write("\x1b[?25l")
        try:
            yield draw
        finally:
            self.out.write("\x1b[?25h")
            self.out.flush()

    @contextlib.contextmanager
    def working(self, text: str) -> Iterator[None]:
        """A spinner beside `text` while the block runs, gone when it ends; `text` alone where the cursor can't
        move."""
        if not self.moves:
            self.say(f"  {text}...")
            yield
            return
        done = threading.Event()

        def spin() -> None:
            for frame in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
                self.out.write(f"\r  {self.style(frame, 'pink')} {text}")
                self.out.flush()
                if done.wait(0.08):
                    break
            self.out.write("\r\x1b[2K")
            self.out.flush()

        spinner = threading.Thread(target=spin, daemon=True)
        self.out.write("\x1b[?25l")
        spinner.start()
        try:
            yield
        finally:
            done.set()
            spinner.join()
            self.out.write("\x1b[?25h")
            self.out.flush()

    # -- the keyboard ------------------------------------------------------------------------------------------------

    @contextlib.contextmanager
    def _raw(self) -> Iterator[Callable[[], str]]:
        """A function returning the next key (`KEYS`' names, else the character), the terminal raw meanwhile."""
        fd = self.keyboard.fileno()
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        def key() -> str:
            char = os.read(fd, 1).decode(errors="replace")
            if char == "\x03":
                raise KeyboardInterrupt
            if char == "\x1b" and select.select([fd], [], [], ESCAPE_WAIT_S)[0]:  # an arrow key, not Esc alone
                char += os.read(fd, 2).decode(errors="replace")
            if char == " ":
                return SPACE
            return KEYS.get(char, char)

        try:
            yield key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    def _typed(self, prompt: str) -> str:
        self.out.write(prompt)
        self.out.flush()
        answer = self.keyboard.readline()
        if not answer:
            raise KeyboardInterrupt
        return answer.strip()

    # -- questions --------------------------------------------------------------------------------------------------

    def _wrapped(self, rows: list[str]) -> list[str]:
        """`rows` wrapped to the terminal's width, a row that starts with a mark (✓, ⚠) going on under its words."""
        out = []
        for row in rows:
            hang = "  " if row[:2] in ("✓ ", "⚠ ") else ""
            out += textwrap.wrap(row, self._width() - 3, subsequent_indent=hang) or [""]
        return out

    def _rows(self, lines: Sequence[Line], at: int, marks: Sequence[bool] | None, top: int, shown: int) -> list[str]:
        """The visible lines of a list, notes aligned, the one at `at` marked (and boxes ticked as `marks`). The notes'
        column moves left as far as the terminal needs to show them whole, past a row's padding but never its text."""
        box_width = 0 if marks is None else 4
        room = self._width() - 6 - box_width - max(len(line.note) for line in lines)
        width = min(max(len(line.text) for line in lines), room)
        rows = []
        if top:
            rows.append(self.style(f"    ↑ {top} more", "2"))
        for i in range(top, min(top + shown, len(lines))):
            line = lines[i]
            text = line.text.rstrip().ljust(width) if len(line.text) > width else line.text.ljust(width)
            text = self.style(text, "1") if i == at else text if line.enabled else self.style(text, "2")
            pointer = self.style("●" if marks is None else ">", "pink") if i == at else " "
            box = "" if marks is None else (self.style("[x]", "pink") if marks[i] else "[ ]") + " "
            rows.append(f"  {pointer} {box}{text}  {self.style(line.note, line.tone)}")
        if top + shown < len(lines):
            rows.append(self.style(f"    ↓ {len(lines) - top - shown} more", "2"))
        return rows

    def _answered(self, title: str, answer: str) -> list[str]:
        """An answered question, as it stays on screen: its title and the answer."""
        return [f"{self.style(title, '1')} {self.style('·', '2')} {self.style(answer, 'mint')}"]

    def pick(
        self,
        title: str,
        lines: Sequence[Line],
        default: int,
        details: Callable[[int], list[str]] = lambda _: [],
        keys: Sequence[tuple[str, str]] = (),
        back: bool = False,
        turn: tuple[str, Callable[[int, int], Sequence[Line]]] | None = None,
    ) -> int | str:
        """The index of the line chosen, or one of `keys`' letters (key, what it does) when that was pressed. A
        line that is not `enabled` can be chosen too: it is the caller who says why it can't be used. A list taller
        than the terminal scrolls. With `back`, Esc or Backspace (or b where lines are numbered) raises Back. With
        `turn` (what it turns, and a function of the line and -1 or +1 giving the lines after), ← and → turn the
        highlighted line's value, as a precision is chosen beside a model."""
        if not self.moves:
            return self._pick_typed(title, lines, default, keys, back)
        hint = "↑↓ move" + (f" · ←→ {turn[0]}" if turn else "") + " · Enter choose"
        hint += "".join(f" · {key} {what}" for key, what in keys) + (" · Esc back" if back else "")
        if len(title) + len(hint) + 18 <= self._width():  # room for "   " and " · Ctrl-C quits", the hint least needed
            hint += " · Ctrl-C quits"
        at, top = default, 0
        with self._drawing() as draw, self._raw() as key:
            while True:
                below = self._wrapped(details(at))
                shown = max(5, shutil.get_terminal_size((100, 24)).lines - len(below) - 7)
                top = min(max(top, at - shown + 1), at, max(0, len(lines) - shown))
                rows = [self.style(title, "1") + "   " + self.style(hint, "2"), ""]
                rows += self._rows(lines, at, None, top, shown)
                draw([*rows, "", *(f"  {row}" for row in below)])
                pressed = key()
                if pressed == UP:
                    at = (at - 1) % len(lines)
                elif pressed == DOWN:
                    at = (at + 1) % len(lines)
                elif pressed in (LEFT, RIGHT) and turn:
                    lines = turn[1](at, -1 if pressed == LEFT else 1)
                elif pressed == ENTER:
                    self.shown += self._answered(title, lines[at].answer or lines[at].text.split("  ")[0])
                    draw(self.shown[-1:])
                    return at
                elif pressed in {k for k, _ in keys}:
                    draw([])
                    return pressed
                elif pressed == BACK and back:
                    draw([])
                    raise Back

    def _pick_typed(
        self, title: str, lines: Sequence[Line], default: int, keys: Sequence[tuple[str, str]], back: bool
    ) -> int | str:
        self.say(title)
        for i, line in enumerate(lines, 1):
            self.say(f"  {i:>2}  {line.text}  {line.note}".rstrip())
        letters = "".join(f", {key} {what}" for key, what in keys) + (", b back" if back else "")
        while True:
            answer = self._typed(f"Number (Enter: {default + 1}{letters}): ")
            if not answer:
                return default
            if back and answer == "b":
                raise Back
            if answer in {k for k, _ in keys}:
                return answer
            if answer.isdigit() and 1 <= int(answer) <= len(lines):
                return int(answer) - 1

    def checklist(self, title: str, lines: Sequence[Line], ticked: Sequence[bool], back: bool = False) -> list[bool]:
        """Which lines are ticked when Enter goes on: space ticks or unticks a line, and Enter goes on from any row,
        so Enter alone takes the lines as `ticked` starts them, as every other step's Enter takes its default. With
        `back`, Esc or Backspace (or b where lines are numbered) raises Back."""
        marks = list(ticked)
        if not self.moves:
            self.say(title)
            for i, line in enumerate(lines, 1):
                self.say(f"  {i:>2}  [{'x' if marks[i - 1] else ' '}] {line.text}  {line.note}".rstrip())
            answer = self._typed(
                f"Numbers to use, separated by spaces (Enter: as ticked{', b back' if back else ''}): "
            )
            if back and answer == "b":
                raise Back
            if answer:
                chosen = {int(word) for word in answer.replace(",", " ").split() if word.isdigit()}
                marks = [i in chosen for i in range(1, len(lines) + 1)]
            return marks
        at, done = 0, len(lines)  # the row after the lines is Continue
        hint = self.style(f"↑↓ move · Space tick · Enter continue{' · Esc back' if back else ''} · Ctrl-C quits", "2")
        with self._drawing() as draw, self._raw() as key:
            while True:
                pointer = self.style(">", "pink") if at == done else " "
                go = self.style("Continue", "1", "mint") if at == done else self.style("Continue", "2")
                rows = self._rows(lines, at if at < done else -1, marks, 0, len(lines))
                draw([f"{self.style(title, '1')}   {hint}", "", *rows, "", f"  {pointer} {go}"])
                pressed = key()
                if pressed == UP:
                    at = (at - 1) % (done + 1)
                elif pressed == DOWN:
                    at = (at + 1) % (done + 1)
                elif pressed == SPACE and at < done:
                    marks[at] = not marks[at]
                elif pressed == BACK and back:
                    draw([])
                    raise Back
                elif pressed == ENTER:
                    ticked_text = ", ".join(line.text for line, on in zip(lines, marks, strict=True) if on) or "nothing"
                    self.shown += self._answered(title, ticked_text)
                    draw(self.shown[-1:])
                    return marks

    def downloading(self) -> Callable[[Any], None]:
        """A download's progress (pull's events: done, total, bytes_per_second, seconds_left) as a bar on one line."""

        def show(event: Any) -> None:
            share = event.done / event.total if event.total else 1.0
            bar = self.style("█" * round(24 * share), "mint") + self.style("░" * (24 - round(24 * share)), "2")
            rate = f"{event.bytes_per_second / 1e6:.0f} MB/s" if event.bytes_per_second else ""
            left = f"{event.seconds_left:.0f} s left" if event.seconds_left else ""
            tail = " · ".join(t for t in (f"{event.done / 1e9:.1f} / {event.total / 1e9:.1f} GB", rate, left) if t)
            ending = "\n" if event.phase in ("complete", "failed", "cancelled") else ""
            self.out.write(f"\r\x1b[2K  {bar}  {tail}{ending}" if self.moves else f"  {tail}\n" if ending else "")
            self.out.flush()

        return show

    def choose(self, title: str, options: Sequence[Line], default: int, back: bool = False) -> int:
        """One of a few options (a pick without details or extra keys)."""
        chosen = self.pick(title, options, default, back=back)
        assert isinstance(chosen, int)
        return chosen

    def confirm(self, question: str, default: bool) -> bool:
        while True:
            answer = self._typed(f"{question} [{'Y/n' if default else 'y/N'}] ").lower()
            if not answer:
                return default
            if answer in ("y", "yes", "n", "no"):
                return answer.startswith("y")

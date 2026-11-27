"""Inky's sticker for the terminal, src/ippon/inky.png: assets/inky-sticker.svg (made by scripts/banner.py) drawn by a
browser 640 pixels wide, still, trimmed to the sticker's edge and shadow, and stored in 256 colours. ippon sends the
whole picture each time it draws it, and 256 colours keep the shadow's soft edge at a fraction of the size.

    uv run --with pillow python scripts/sticker.py [CHROME]

CHROME is the browser that draws it: Google Chrome's own path on a Mac unless given.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHROME = sys.argv[1] if len(sys.argv) > 1 else "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WIDTH = 640  # 2 pixels to a point on a Retina screen, for a sticker up to 44 columns of a 13-point font

with tempfile.TemporaryDirectory() as work:
    folder = Path(work)
    svg = (ROOT / "assets" / "inky-sticker.svg").read_text()
    # held still: the drawing's blink and its typing arm would otherwise be caught wherever the browser draws them
    still = svg.replace("</style>", ".octo-inky * { animation: none !important; }</style>", 1)
    (folder / "sticker.svg").write_text(still)
    page = f'<img src="sticker.svg" style="width:{WIDTH}px">'
    (folder / "page.html").write_text(f'<!doctype html><body style="margin:0;background:transparent">{page}</body>')
    shot = folder / "shot.png"
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
         "--default-background-color=00000000", f"--window-size={WIDTH},{WIDTH}", f"--screenshot={shot}",
         (folder / "page.html").as_uri()],
        check=True,
        capture_output=True,
    )  # fmt: skip
    picture = Image.open(shot).convert("RGBA")
    picture = picture.crop(picture.getchannel("A").getbbox())
    picture.quantize(256, Image.Quantize.FASTOCTREE).save(ROOT / "src" / "ippon" / "inky.png", optimize=True)
print("wrote", ROOT / "src" / "ippon" / "inky.png", picture.size)

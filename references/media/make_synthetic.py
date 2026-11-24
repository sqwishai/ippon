"""Synthetic images for the media references: python -I make_synthetic.py [--out images]

Deterministic (seed 20261023) and drawn with Pillow alone, text in its embedded Aileron Regular (dotcolon.net, CC0), so
the files are this repository's own work, dedicated CC0 1.0 (LICENSES.md). Each covers an edge case of the creators'
image pipelines; the d1-omni column is what Liquid's vision.layout makes of it (P = projected tokens):

    chart-bars-800x600.png          text and bars, RGB                        one 416x576 crop, P 234
    table-invoice-1024x512.png      a table screenshot, area exactly 2 x 262,144: not tiled (`>`), P 242
    solid-red-1x1.png               one pixel: below the 64x1024 floor, upscaled to 256x256, P 64
    solid-blue-24x24.png            below the floor, P 64
    small-circle-100x80.jpg         below the floor, upscaled to 256x288, P 72
    gradient-512x384.png            multiples of 32 inside the budget: resized to its own size, P 192
    squares-512x512.png             exactly the 262,144 px budget, P 256
    circles-720x720.png             720/32 = 22.5 rounds to even (22): not tiled, P 256
    circles-725x725.png             just over the tiling line: 2x2 tiles and a thumbnail, P 1,280
    dashboard-1000x800.jpg          aspect 5:4, a tie between grids 2x2 and 3x2 (the larger wins here), P 1,774
    shapes-4000x3000.jpg            4,000x3,000: 3x2 tiles and a 416x576 thumbnail, P 1,770
    receipt-400x3200.png            aspect 1:8: 1x8 tiles and a 1440x160 thumbnail, P 2,273
    banner-3000x60.png              aspect 50:1: one 64x3008 crop, P 188
    gray-gradient-640x480.png       mode L (grayscale)
    palette-stripes-600x400.png     mode P with a transparent index (its stored colour shows after convert("RGB"))
    rgba-circle-512x384.png         RGBA whose transparent pixels hold magenta (alpha is dropped, not composited)
    arrow-exif6.jpg                 stored turned 90 degrees, EXIF orientation 6: upright only once the EXIF is applied
    cmyk-blocks-640x480.jpg         a CMYK JPEG
    gray16-gradient-512x256.png     a 16-bit grayscale PNG (convert("RGB") clips at 255: nearly white)

Byte-identical output needs the Pillow, zlib and libjpeg-turbo the committed files were made with (Pillow 12.3.0);
the committed files, with their sha256 in requests/media-d1.json, are what the references are made on.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SEED = 20261023
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
RED, GREEN, BLUE = (204, 32, 32), (40, 150, 60), (40, 80, 200)
PALETTE = [RED, (240, 170, 20), GREEN, BLUE, (130, 60, 160), (20, 160, 170)]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def column_gradient(width: int, height: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    """A vertical gradient, each row one colour (exact: a 1-pixel column widened with NEAREST)."""
    rows = bytearray()
    for y in range(height):
        t = y / max(1, height - 1)
        rows += bytes(round(a + (b - a) * t) for a, b in zip(top, bottom, strict=True))
    return Image.frombytes("RGB", (1, height), bytes(rows)).resize((width, height), Image.Resampling.NEAREST)


def scatter(rng: random.Random, n: int, size: tuple[int, int], radius: int, margin: int) -> list[tuple[int, int]]:
    """n centres at least 2.5 radii apart, inside the margin."""
    out: list[tuple[int, int]] = []
    while len(out) < n:
        x, y = rng.randint(margin, size[0] - margin), rng.randint(margin, size[1] - margin)
        if all((x - a) ** 2 + (y - b) ** 2 >= (2.5 * radius) ** 2 for a, b in out):
            out.append((x, y))
    return out


def chart(rng: random.Random) -> Image.Image:
    regions, values = ["North", "South", "East", "West", "Central"], [42, 77, 35, 58, 64]
    im = Image.new("RGB", (800, 600), WHITE)
    d = ImageDraw.Draw(im)
    d.text((400, 28), "Units sold by region, 2025", fill=BLACK, font=font(30), anchor="mt")
    x0, y0, x1, y1 = 90, 520, 760, 100
    d.line([(x0, y0), (x1, y0)], fill=BLACK, width=3)
    d.line([(x0, y0), (x0, y1)], fill=BLACK, width=3)
    for tick in range(0, 101, 20):
        y = y0 - (y0 - y1) * tick / 100
        d.line([(x0 - 8, y), (x0, y)], fill=BLACK, width=2)
        d.text((x0 - 14, y), str(tick), fill=BLACK, font=font(18), anchor="rm")
    width = 90
    gap = (x1 - x0 - 5 * width) / 6
    colours = rng.sample(PALETTE, 5)
    for i, (region, value, colour) in enumerate(zip(regions, values, colours, strict=True)):
        left = x0 + gap * (i + 1) + width * i
        top = y0 - (y0 - y1) * value / 100
        d.rectangle([left, top, left + width, y0 - 2], fill=colour)
        d.text((left + width / 2, top - 8), str(value), fill=BLACK, font=font(20), anchor="mb")
        d.text((left + width / 2, y0 + 12), region, fill=BLACK, font=font(20), anchor="mt")
    return im


def table(rng: random.Random) -> Image.Image:
    rows = [
        ("Printer paper, 5 reams", 2, 24.50),
        ("Toner cartridge", 1, 89.00),
        ("Stapler", 3, 7.25),
        ("Desk lamp", 1, 34.99),
        ("USB-C cable", 4, 9.80),
        ("Notebook, A5", 6, 3.15),
    ]
    im = Image.new("RGB", (1024, 512), (246, 247, 249))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1023, 36], fill=(225, 227, 231))
    for i, colour in enumerate([(236, 95, 87), (245, 190, 60), (98, 197, 84)]):
        d.ellipse([16 + 24 * i, 11, 30 + 24 * i, 25], fill=colour)
    d.text((512, 18), "Invoice 2025-0417 - Office Supplies Ltd", fill=(60, 60, 60), font=font(17), anchor="mm")
    columns = [(40, "Item", "la"), (560, "Qty", "ra"), (760, "Unit price", "ra"), (980, "Amount", "ra")]
    top, height = 60, 50
    d.rectangle([24, top, 1000, top + height], fill=(214, 228, 245))
    for x, title, anchor in columns:
        d.text((x, top + height / 2), title, fill=BLACK, font=font(22), anchor=anchor[0] + "m")
    total = 0.0
    for r, (item, qty, price) in enumerate(rows, start=1):
        y = top + r * height
        d.rectangle([24, y, 1000, y + height], fill=WHITE if r % 2 else (240, 242, 245))
        amount = qty * price
        total += amount
        cells = [item, str(qty), f"${price:.2f}", f"${amount:.2f}"]
        for (x, _, anchor), text in zip(columns, cells, strict=True):
            d.text((x, y + height / 2), text, fill=(30, 30, 30), font=font(21), anchor=anchor[0] + "m")
    y = top + (len(rows) + 1) * height
    d.text((760, y + 22), "Total", fill=BLACK, font=font(23), anchor="rm")
    d.text((980, y + 22), f"${total:.2f}", fill=BLACK, font=font(23), anchor="rm")
    for r in range(len(rows) + 2):
        d.line([(24, top + r * height), (1000, top + r * height)], fill=(180, 185, 192), width=1)
    return im


def solid(size: tuple[int, int], colour: tuple[int, int, int]) -> Callable[[random.Random], Image.Image]:
    return lambda rng: Image.new("RGB", size, colour)


def small_circle(rng: random.Random) -> Image.Image:
    im = Image.new("RGB", (100, 80), (128, 128, 128))
    ImageDraw.Draw(im).ellipse([25, 15, 75, 65], fill=(250, 220, 40))
    return im


def gradient(rng: random.Random) -> Image.Image:
    """Red to blue, left to right; exact, one column per x."""
    w, h = 512, 384
    row = b"".join(bytes((round(255 * (1 - x / (w - 1))), 0, round(255 * x / (w - 1)))) for x in range(w))
    return Image.frombytes("RGB", (w, h), row * h)


def squares(rng: random.Random) -> Image.Image:
    im = Image.new("RGB", (512, 512), WHITE)
    d = ImageDraw.Draw(im)
    for x, y in scatter(rng, 4, (512, 512), 40, 70):
        d.rectangle([x - 40, y - 40, x + 40, y + 40], fill=GREEN)
    return im


def circles(side: int, n: int) -> Callable[[random.Random], Image.Image]:
    def draw(rng: random.Random) -> Image.Image:
        im = Image.new("RGB", (side, side), (250, 248, 240))
        d = ImageDraw.Draw(im)
        for x, y in scatter(rng, n, (side, side), 60, 90):
            d.ellipse([x - 60, y - 60, x + 60, y + 60], fill=rng.choice(PALETTE), outline=BLACK, width=3)
        return im

    return draw


def dashboard(rng: random.Random) -> Image.Image:
    im = Image.new("RGB", (1000, 800), (24, 28, 36))
    d = ImageDraw.Draw(im)
    d.text((500, 90), "SYSTEM STATUS", fill=(200, 205, 215), font=font(56), anchor="mm")
    d.rounded_rectangle([200, 190, 800, 420], radius=30, fill=(30, 120, 60))
    d.text((500, 305), "ONLINE", fill=WHITE, font=font(110), anchor="mm")
    for i, (label, value) in enumerate([("Uptime", "99.2%"), ("Errors today", "3"), ("Queue", "12 jobs")]):
        y = 500 + 90 * i
        d.text((220, y), label, fill=(170, 175, 185), font=font(40), anchor="lm")
        d.text((780, y), value, fill=WHITE, font=font(40), anchor="rm")
    return im


def shapes(rng: random.Random) -> Image.Image:
    im = column_gradient(4000, 3000, (150, 200, 240), (250, 250, 245))
    d = ImageDraw.Draw(im)
    centres = scatter(rng, 6, (4000, 3000), 260, 400)
    for x, y in centres[:4]:
        r = rng.randint(180, 260)
        d.ellipse([x - r, y - r, x + r, y + r], fill=RED)
    for x, y in centres[4:]:
        r = rng.randint(160, 220)
        d.rectangle([x - r, y - r, x + r, y + r], fill=BLUE)
    d.text((3900, 2920), "LOT 7", fill=BLACK, font=font(72), anchor="rs")
    return im


def receipt(rng: random.Random) -> Image.Image:
    items = ["Milk", "Bread", "Eggs", "Apples", "Rice", "Coffee", "Tea", "Butter", "Cheese", "Pasta"]
    items += ["Tomatoes", "Onions", "Yogurt", "Honey", "Oats", "Bananas", "Carrots", "Soap", "Salt", "Beans"]
    im = Image.new("RGB", (400, 3200), (252, 252, 248))
    d = ImageDraw.Draw(im)
    d.text((200, 70), "CORNER MARKET", fill=BLACK, font=font(34), anchor="mm")
    d.text((200, 120), "12 Elm Street  -  14 Oct 2025", fill=(60, 60, 60), font=font(18), anchor="mm")
    y, total = 200, 0
    for i in range(40):
        cents = rng.randint(89, 1299)
        total += cents
        d.text((30, y), f"{items[i % len(items)]} x{1 + i // len(items)}", fill=BLACK, font=font(22), anchor="lm")
        d.text((370, y), f"{cents / 100:.2f}", fill=BLACK, font=font(22), anchor="rm")
        y += 66
    d.line([(30, y), (370, y)], fill=BLACK, width=2)
    d.text((30, y + 50), "TOTAL", fill=BLACK, font=font(34), anchor="lm")
    d.text((370, y + 50), f"{total / 100:.2f}", fill=BLACK, font=font(34), anchor="rm")
    d.text((200, y + 160), "THANK YOU", fill=BLACK, font=font(26), anchor="mm")
    return im


def banner(rng: random.Random) -> Image.Image:
    im = Image.new("RGB", (3000, 60), (200, 20, 40))
    text = "SUMMER SALE  -  50% OFF EVERYTHING  -  ENDS SUNDAY"
    ImageDraw.Draw(im).text((1500, 30), text, fill=WHITE, font=font(40), anchor="mm")
    return im


def gray(rng: random.Random) -> Image.Image:
    im = column_gradient(640, 480, (40, 40, 40), (230, 230, 230)).convert("L")
    ImageDraw.Draw(im).text((320, 240), "GRAYSCALE", fill=0, font=font(64), anchor="mm")
    return im


def palette(rng: random.Random) -> Image.Image:
    """Index 0 is transparent and stored as magenta; 1 to 3 are three vertical stripes."""
    im = Image.new("P", (600, 400), 0)
    im.putpalette([255, 0, 255, *GREEN, *WHITE, *BLUE])
    d = ImageDraw.Draw(im)
    for i in range(3):
        d.rectangle([40 + 174 * i, 40, 40 + 174 * (i + 1) - 1, 359], fill=i + 1)
    im.info["transparency"] = 0
    return im


def rgba(rng: random.Random) -> Image.Image:
    im = Image.new("RGBA", (512, 384), (255, 0, 255, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([136, 72, 376, 312], fill=(*GREEN, 255))
    d.text((256, 192), "LOGO", fill=(0, 0, 0, 128), font=font(56), anchor="mm")
    return im


def arrow_upright() -> Image.Image:
    im = Image.new("RGB", (480, 640), WHITE)
    d = ImageDraw.Draw(im)
    d.polygon([(240, 60), (400, 260), (300, 260), (300, 520), (180, 520), (180, 260), (80, 260)], fill=BLACK)
    d.text((240, 590), "UP", fill=BLACK, font=font(48), anchor="mm")
    return im


def cmyk(rng: random.Random) -> Image.Image:
    im = Image.new("CMYK", (640, 480), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, ink in enumerate([(255, 0, 0, 0), (0, 255, 0, 0), (0, 0, 255, 0), (0, 0, 0, 255)]):
        x = 40 + 150 * i
        d.rectangle([x, 140, x + 110, 340], fill=ink)
    return im


def gray16(rng: random.Random) -> Image.Image:
    """0 to 65,535 left to right, little-endian 16-bit samples."""
    w, h = 512, 256
    row = b"".join(round(65535 * x / (w - 1)).to_bytes(2, "little") for x in range(w))
    return Image.frombytes("I;16", (w, h), row * h)


IMAGES: list[tuple[str, Callable[[random.Random], Image.Image]]] = [
    ("chart-bars-800x600.png", chart),
    ("table-invoice-1024x512.png", table),
    ("solid-red-1x1.png", solid((1, 1), (220, 30, 30))),
    ("solid-blue-24x24.png", solid((24, 24), (30, 60, 200))),
    ("small-circle-100x80.jpg", small_circle),
    ("gradient-512x384.png", gradient),
    ("squares-512x512.png", squares),
    ("circles-720x720.png", circles(720, 3)),
    ("circles-725x725.png", circles(725, 5)),
    ("dashboard-1000x800.jpg", dashboard),
    ("shapes-4000x3000.jpg", shapes),
    ("receipt-400x3200.png", receipt),
    ("banner-3000x60.png", banner),
    ("gray-gradient-640x480.png", gray),
    ("palette-stripes-600x400.png", palette),
    ("rgba-circle-512x384.png", rgba),
    ("cmyk-blocks-640x480.jpg", cmyk),
    ("gray16-gradient-512x256.png", gray16),
]


def save(im: Image.Image, path: Path) -> None:
    if path.suffix == ".jpg":
        im.save(path, "JPEG", quality=90)
    else:
        im.save(path, "PNG", **({"transparency": im.info["transparency"]} if im.mode == "P" else {}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "images")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for name, make in IMAGES:
        save(make(rng), args.out / name)
    # stored turned a quarter (counter-clockwise), with orientation 6 ("turn 90 degrees clockwise to view")
    exif = Image.Exif()
    exif[0x0112] = 6
    arrow_upright().transpose(Image.Transpose.ROTATE_90).save(
        args.out / "arrow-exif6.jpg", "JPEG", quality=90, exif=exif.tobytes()
    )
    for name in [n for n, _ in IMAGES] + ["arrow-exif6.jpg"]:
        path = args.out / name
        with Image.open(path) as im:
            print(
                f"{name}: {im.size[0]}x{im.size[1]} {im.mode}, {path.stat().st_size:,} bytes, sha256 "
                f"{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
            )


if __name__ == "__main__":
    main()

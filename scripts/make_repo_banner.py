"""Compose thematic open-deepthink repo banner with exact typography."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "static" / "readme-current-banner.png"
OUT = ROOT / "static" / "open-deepthink-banner.png"
OUT_README = ROOT / "static" / "open-deepthink-banner-readme.png"

W, H = 1280, 720


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates: list[str] = []
    if bold:
        candidates += [
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
        ]
    candidates += [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\consola.ttf",
    ]
    for p in candidates:
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def cover_crop(src: Image.Image, w: int, h: int) -> Image.Image:
    sw, sh = src.size
    scale = max(w / sw, h / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    bg = src.resize((nw, nh), Image.Resampling.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return bg.crop((left, top, left + w, top + h))


def radial_alpha(w: int, h: int, cx: float, cy: float, rx: float, ry: float, power: float) -> Image.Image:
    mask = Image.new("L", (w, h), 0)
    px = mask.load()
    for y in range(h):
        for x in range(w):
            dx = (x - cx) / rx
            dy = (y - cy) / ry
            d = math.sqrt(dx * dx + dy * dy)
            a = int(max(0, min(255, 255 * max(0.0, 1.0 - d) ** power)))
            px[x, y] = a
    return mask


def main() -> None:
    src = Image.open(SRC).convert("RGB")
    canvas = cover_crop(src, W, H)
    cx, cy = W / 2, H / 2

    # Soft center plate for legibility (keep nebula edges)
    plate = Image.new("RGB", (W, H), (6, 3, 18))
    plate_mask = radial_alpha(W, H, cx, cy - 10, W * 0.42, H * 0.36, 1.45)
    # strengthen center a bit
    plate_mask = ImageEnhance_Brightness(plate_mask, 0.78)
    canvas = Image.composite(plate, canvas, plate_mask)

    # Outer vignette
    black = Image.new("RGB", (W, H), (0, 0, 0))
    vig = Image.new("L", (W, H), 0)
    vpx = vig.load()
    for y in range(H):
        for x in range(W):
            dx = (x - cx) / (W * 0.62)
            dy = (y - cy) / (H * 0.62)
            d = math.sqrt(dx * dx + dy * dy)
            a = int(max(0, min(170, 170 * max(0.0, d - 0.72) / 0.55)))
            vpx[x, y] = a
    canvas = Image.composite(black, canvas, vig)

    glow = canvas.filter(ImageFilter.GaussianBlur(14))
    canvas = Image.blend(canvas, glow, 0.16)

    draw = ImageDraw.Draw(canvas)
    title = "open-deepthink"
    subtitle = "Qualitative Neural Networks  ·  Diffusion  ·  Distillation"
    tag = "depth through structured iteration"

    title_font = load_font(92, bold=True)
    sub_font = load_font(28)
    tag_font = load_font(22)

    def measure(font: ImageFont.ImageFont, text: str) -> tuple[int, int]:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    tw, th = measure(title_font, title)
    sw, sh = measure(sub_font, subtitle)
    tgw, tgh = measure(tag_font, tag)

    title_y = H // 2 - th // 2 - 40
    sub_y = title_y + th + 30
    tag_y = sub_y + sh + 16
    tx = (W - tw) // 2

    # Neon underline
    line_w = int(tw * 0.52)
    line_x0 = (W - line_w) // 2
    line_y = title_y + th + 14
    for i in range(line_w):
        t = i / max(1, line_w - 1)
        r = int(70 + 170 * t)
        g = int(245 - 150 * t)
        b = int(255 - 30 * t)
        for dy in (0, 1, 2):
            draw.point((line_x0 + i, line_y + dy), fill=(r, g, b))

    # Title shadows + cyan/magenta fringe + white core
    for ox, oy in ((0, 3), (2, 2), (-2, 2), (0, 0)):
        draw.text((tx + ox, title_y + oy), title, font=title_font, fill=(0, 0, 0))
    draw.text((tx - 1, title_y), title, font=title_font, fill=(80, 255, 245))
    draw.text((tx + 1, title_y), title, font=title_font, fill=(230, 120, 255))
    draw.text((tx, title_y), title, font=title_font, fill=(250, 252, 255))

    sx = (W - sw) // 2
    draw.text((sx + 1, sub_y + 1), subtitle, font=sub_font, fill=(0, 0, 0))
    draw.text((sx, sub_y), subtitle, font=sub_font, fill=(198, 208, 238))

    tgx = (W - tgw) // 2
    draw.text((tgx + 1, tag_y + 1), tag, font=tag_font, fill=(0, 0, 0))
    draw.text((tgx, tag_y), tag, font=tag_font, fill=(148, 172, 210))

    # Accent nodes
    def node(x: int, y: int, r: int, color: tuple[int, int, int]) -> None:
        for rr in range(r + 7, r, -1):
            draw.ellipse([x - rr, y - rr, x + rr, y + rr], outline=color, width=1)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    ny = title_y + th // 2
    node(tx - 52, ny, 5, (90, 255, 245))
    node(tx + tw + 52, ny, 5, (230, 130, 255))
    draw.line([(tx - 44, ny), (tx - 10, ny)], fill=(90, 255, 245), width=1)
    draw.line([(tx + tw + 10, ny), (tx + tw + 44, ny)], fill=(230, 130, 255), width=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, "PNG", optimize=True)
    # README display size closer to original proportions
    readme = canvas.resize((1248, 702), Image.Resampling.LANCZOS)
    readme.save(OUT_README, "PNG", optimize=True)
    print(f"wrote {OUT} {canvas.size}")
    print(f"wrote {OUT_README} {readme.size}")


def ImageEnhance_Brightness(img: Image.Image, factor: float) -> Image.Image:
    """Scale grayscale mask brightness without importing ImageEnhance for masks."""
    return img.point(lambda p: int(max(0, min(255, p * factor))))


if __name__ == "__main__":
    main()

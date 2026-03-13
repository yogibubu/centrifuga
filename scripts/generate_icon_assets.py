#!/usr/bin/env python3
"""Generate PNG and ICNS assets for the macOS app icon from the SVG source."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = ROOT / "assets" / "icons"
SVG_PATH = ICON_DIR / "app_icon.svg"
PNG_PATH = ICON_DIR / "app_icon_256.png"
ICNS_PATH = ICON_DIR / "app_icon.icns"
ICONSET_DIR = ICON_DIR / "app_icon.iconset"

ICON_SIZES = {
    "icon_16x16.png": 16,
    "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32,
    "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128,
    "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256,
    "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512,
    "icon_512x512@2x.png": 1024,
}


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def render_png(size: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run("qlmanage", "-t", "-s", str(size), "-o", str(destination.parent), str(SVG_PATH))
    generated = destination.parent / f"{SVG_PATH.name}.png"
    generated.replace(destination)


def main() -> None:
    if not SVG_PATH.exists():
        raise FileNotFoundError(f"Missing icon source: {SVG_PATH}")

    if ICONSET_DIR.exists():
        shutil.rmtree(ICONSET_DIR)
    ICONSET_DIR.mkdir(parents=True)

    for filename, size in ICON_SIZES.items():
        render_png(size, ICONSET_DIR / filename)

    shutil.copy2(ICONSET_DIR / "icon_128x128@2x.png", PNG_PATH)
    run("iconutil", "-c", "icns", str(ICONSET_DIR), "-o", str(ICNS_PATH))


if __name__ == "__main__":
    main()

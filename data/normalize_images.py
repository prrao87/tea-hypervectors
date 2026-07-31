"""Normalize downloaded product images to validated JPEG files."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from tea_data import DEFAULT_IMAGE_DIR

JPEG_MAGIC = b"\xff\xd8\xff"
SOURCE_EXTENSIONS = {".heic", ".heif", ".jpeg", ".jpg", ".png", ".webp"}


def is_jpeg(path: Path) -> bool:
    with path.open("rb") as image_file:
        return image_file.read(len(JPEG_MAGIC)) == JPEG_MAGIC


def verify_jpeg(path: Path) -> None:
    if not is_jpeg(path):
        raise ValueError(f"{path} does not contain JPEG data")
    with Image.open(path) as image:
        image.verify()


def save_as_jpeg(source: Path, destination: Path, quality: int) -> None:
    register_heif_opener()
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", image.size, "white")
            alpha = image.getchannel("A")
            background.paste(image.convert("RGB"), mask=alpha)
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        exif = image.getexif()
        output = BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=quality,
            optimize=True,
            exif=exif.tobytes(),
        )
        temporary = destination.with_name(f"{destination.name}.tmp")
        temporary.write_bytes(output.getvalue())
        verify_jpeg(temporary)
        temporary.replace(destination)


def normalize_image(source: Path, destination: Path, quality: int) -> str:
    if source == destination:
        verify_jpeg(source)
        return "verified"

    if destination.exists():
        verify_jpeg(destination)
        source.unlink()
        return "resumed"

    if is_jpeg(source):
        verify_jpeg(source)
        source.replace(destination)
        return "renamed"

    save_as_jpeg(source, destination, quality)
    source.unlink()
    return "converted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality for images that require transcoding; default: 95.",
    )
    args = parser.parse_args()
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    return args


def main() -> int:
    args = parse_args()
    sources = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS
    )
    if not sources:
        raise ValueError(f"No supported images found in {args.image_dir}")

    counts = {"verified": 0, "renamed": 0, "converted": 0, "resumed": 0}
    total = len(sources)
    for index, source in enumerate(sources, start=1):
        destination = source.with_suffix(".jpg")
        status = normalize_image(source, destination, args.quality)
        counts[status] += 1
        print(f"[{index:3d}/{total}] {status:9s} {destination.name}", flush=True)

    remaining = sorted(
        path.name
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS - {".jpg"}
    )
    if remaining:
        raise RuntimeError(f"Image normalization left non-JPEG files: {remaining}")

    summary = {
        "image_dir": str(args.image_dir.resolve()),
        "total": total,
        **counts,
        "remaining_non_jpg": len(remaining),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

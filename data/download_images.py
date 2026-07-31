"""
Download final-dataset images from the dataset's CDN source, verifying and normalizing them to JPEG format,
Runs sequentially with resumable progress.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from normalize_images import normalize_image, verify_jpeg
from tea_data import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_INPUT,
    downloaded_image_path,
    normalized_image_path,
    read_ingestion_metadata,
)

DEFAULT_DELAY_SECONDS = 1.0
PROGRESS_FILENAME = "download_progress.json"
USER_AGENT = (
    "ChaYiTeaResearchDataset/1.0 "
    "(non-commercial research; respectful 1-second image request interval)"
)


class RespectfulImageDownloader:
    """Download images with at least the configured delay between requests."""

    def __init__(self, delay_seconds: float = DEFAULT_DELAY_SECONDS) -> None:
        self.delay_seconds = delay_seconds
        self.remote_requests = 0
        self.last_request_started: float | None = None

    def _wait_for_request_slot(self) -> None:
        if self.last_request_started is not None:
            elapsed = time.monotonic() - self.last_request_started
            if elapsed < self.delay_seconds:
                time.sleep(self.delay_seconds - elapsed)
        self.last_request_started = time.monotonic()
        self.remote_requests += 1

    def fetch(self, url: str, destination: Path, *, retries: int = 3) -> None:
        for attempt in range(retries):
            self._wait_for_request_slot()
            request = Request(
                url,
                headers={
                    "Accept": "image/*,*/*;q=0.8",
                    "User-Agent": USER_AGENT,
                },
            )
            try:
                with urlopen(request, timeout=30) as response:
                    image = response.read()
                if not image:
                    raise RuntimeError(f"Image response was empty: {url}")
                temporary = destination.with_name(f"{destination.name}.tmp")
                temporary.write_bytes(image)
                temporary.replace(destination)
                return
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                    raise RuntimeError(f"HTTP {error.code} while downloading {url}") from error
                retry_after = error.headers.get("Retry-After")
                wait_seconds = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else 30.0 * (attempt + 1)
                )
                print(
                    f"HTTP {error.code}; waiting {wait_seconds:.0f}s before retry.",
                    file=sys.stderr,
                )
                time.sleep(wait_seconds)
            except URLError as error:
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"Network error while downloading {url}: {error.reason}"
                    ) from error
                time.sleep(30.0 * (attempt + 1))

        raise AssertionError("Unreachable")


def write_progress(path: Path, progress: dict[str, object]) -> None:
    progress["updated_at"] = datetime.now(UTC).isoformat()
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(progress, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N products; zero means all products.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Seconds between remote image requests; must be at least 1.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality when the CDN returns a non-JPEG image; default: 95.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or positive")
    if args.delay < 1:
        parser.error("--delay must be at least 1 second")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    return args


def main() -> int:
    args = parse_args()
    products = read_ingestion_metadata(args.input, args.limit).select("id", "title", "image_url")
    if products.is_empty():
        raise ValueError("No products selected for image download")

    args.image_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.image_dir / PROGRESS_FILENAME
    downloader = RespectfulImageDownloader(args.delay)
    total = products.height
    completed = 0
    cache_hits = 0
    reused_duplicate_urls = 0
    paths_by_url: dict[str, Path] = {}

    for index, row in enumerate(products.iter_rows(named=True), start=1):
        product_id = int(row["id"])
        title = str(row["title"])
        image_url = str(row["image_url"])
        downloaded_path = downloaded_image_path(args.image_dir, product_id, image_url)
        destination = normalized_image_path(args.image_dir, product_id)

        try:
            if destination.exists() and destination.stat().st_size:
                verify_jpeg(destination)
                if downloaded_path != destination and downloaded_path.exists():
                    normalize_image(downloaded_path, destination, args.quality)
                cache_hits += 1
                status = "cached"
            elif downloaded_path.exists() and downloaded_path.stat().st_size:
                status = normalize_image(downloaded_path, destination, args.quality)
            elif (
                image_url in paths_by_url
                and paths_by_url[image_url].exists()
                and paths_by_url[image_url].stat().st_size
            ):
                shutil.copyfile(paths_by_url[image_url], destination)
                verify_jpeg(destination)
                reused_duplicate_urls += 1
                status = "reused"
            else:
                downloader.fetch(image_url, downloaded_path)
                normalization = normalize_image(downloaded_path, destination, args.quality)
                status = "downloaded" if normalization == "verified" else normalization
        except Exception as error:
            write_progress(
                progress_path,
                {
                    "input": str(args.input.resolve()),
                    "image_dir": str(args.image_dir.resolve()),
                    "completed": completed,
                    "total": total,
                    "next_index": index,
                    "last_error": str(error),
                    "remote_requests": downloader.remote_requests,
                    "cache_hits": cache_hits,
                    "reused_duplicate_urls": reused_duplicate_urls,
                    "jpeg_quality": args.quality,
                },
            )
            raise

        paths_by_url[image_url] = destination
        completed = index
        write_progress(
            progress_path,
            {
                "input": str(args.input.resolve()),
                "image_dir": str(args.image_dir.resolve()),
                "completed": completed,
                "total": total,
                "next_index": index + 1 if index < total else None,
                "last_completed_id": product_id,
                "last_completed_title": title,
                "remote_requests": downloader.remote_requests,
                "cache_hits": cache_hits,
                "reused_duplicate_urls": reused_duplicate_urls,
                "jpeg_quality": args.quality,
            },
        )
        print(
            f"[{completed:3d}/{total}] {status:10s} {product_id} {title}",
            flush=True,
        )

    print(json.dumps(json.loads(progress_path.read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

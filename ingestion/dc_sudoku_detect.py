from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests


BASE_URL = "http://103.241.136.50/epaper/DC/HYD/510X798"

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "templates" / "sudoku_reference.jpg"
HEADING1 = ROOT / "templates" / "heading_sudoku1.png"
HEADING2 = ROOT / "templates" / "heading_sudoku2.png"

REFERENCE_W = 732
REFERENCE_H = 606

DEFAULT_PAGE_MIN = 1
DEFAULT_PAGE_MAX = 12
DEFAULT_VARIANT_MIN = 1
DEFAULT_VARIANT_MAX = 10

MAX_WORKERS = 12
TIMEOUT = 20

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140 Safari/537.36"
)


@dataclass
class Candidate:
    page: int
    variant: int
    url: str
    path: str
    width: int
    height: int
    bytes: int

    dimension_score: float = 0.0
    heading1_score: float = 0.0
    heading2_score: float = 0.0
    layout_score: float = 0.0
    final_score: float = 0.0


def url_for(date: str, page: int, variant: int) -> str:
    return (
        f"{BASE_URL}/{date}/b_images/"
        f"HYD_{date}_tabp{page}_{variant}.jpg"
    )


def download(
    date: str,
    page: int,
    variant: int,
    out: Path,
) -> Optional[Candidate]:

    url = url_for(date, page, variant)

    try:
        s = requests.Session()
        s.headers.update({"User-Agent": UA})

        r = s.get(
            url,
            timeout=TIMEOUT,
        )

        if r.status_code != 200:
            return None

        data = r.content
        ctype = r.headers.get("Content-Type", "").lower()

        if "image" not in ctype and not data.startswith(b"\xff\xd8"):
            return None

        if len(data) < 1000:
            return None

        path = out / f"tabp{page}_{variant}.jpg"
        path.write_bytes(data)

        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if image is None:
            path.unlink(missing_ok=True)
            return None

        h, w = image.shape[:2]

        return Candidate(
            page,
            variant,
            url,
            str(path),
            w,
            h,
            len(data),
        )

    except Exception:
        return None


def load(path: Path) -> np.ndarray:
    data = np.fromfile(
        str(path),
        dtype=np.uint8,
    )
    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read {path}"
        )

    return image


def edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    return cv2.Canny(
        gray,
        50,
        150,
    )


def score_dimensions(
    width: int,
    height: int,
) -> float:

    # Main heuristic. Tolerates moderate resolution variation.
    dw = abs(width - REFERENCE_W) / REFERENCE_W
    dh = abs(height - REFERENCE_H) / REFERENCE_H

    return max(
        0.0,
        min(
            100.0,
            100.0 * math.exp(-4.0 * (dw + dh)),
        ),
    )


def match_template(
    image: np.ndarray,
    template: np.ndarray,
) -> float:

    # Search using a gray-edge representation so OCR digits and mild
    # compression differences matter less.
    source = edges(image)
    tmpl = edges(template)

    # If source is smaller than the template, resize it proportionally.
    if (
        source.shape[0] < tmpl.shape[0]
        or source.shape[1] < tmpl.shape[1]
    ):
        scale = max(
            tmpl.shape[0] / source.shape[0],
            tmpl.shape[1] / source.shape[1],
        )

        source = cv2.resize(
            source,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    result = cv2.matchTemplate(
        source,
        tmpl,
        cv2.TM_CCOEFF_NORMED,
    )

    _, max_value, _, _ = cv2.minMaxLoc(
        result
    )

    return max(
        0.0,
        min(
            100.0,
            ((float(max_value) + 1.0) / 2.0) * 100.0,
        ),
    )


def layout_similarity(
    image: np.ndarray,
    reference: np.ndarray,
) -> float:

    normalized = cv2.resize(
        image,
        (
            REFERENCE_W,
            REFERENCE_H,
        ),
        interpolation=cv2.INTER_AREA,
    )

    ref_edges = edges(reference)
    img_edges = edges(normalized)

    # Reduce to emphasize stable large-scale geometry rather than
    # individual puzzle digits.
    ref_small = cv2.resize(
        ref_edges,
        (320, 260),
        interpolation=cv2.INTER_AREA,
    )

    img_small = cv2.resize(
        img_edges,
        (320, 260),
        interpolation=cv2.INTER_AREA,
    )

    value = cv2.matchTemplate(
        img_small,
        ref_small,
        cv2.TM_CCOEFF_NORMED,
    )[0, 0]

    return max(
        0.0,
        min(
            100.0,
            ((float(value) + 1.0) / 2.0) * 100.0,
        ),
    )


def score_candidate(
    candidate: Candidate,
    reference: np.ndarray,
    heading1: np.ndarray,
    heading2: np.ndarray,
) -> dict:

    image = load(
        Path(candidate.path)
    )

    candidate.dimension_score = score_dimensions(
        candidate.width,
        candidate.height,
    )

    candidate.heading1_score = match_template(
        image,
        heading1,
    )

    candidate.heading2_score = match_template(
        image,
        heading2,
    )

    candidate.layout_score = layout_similarity(
        image,
        reference,
    )

    # Dimensions are intentionally the strongest signal.
    candidate.final_score = (
        0.50 * candidate.dimension_score
        + 0.18 * candidate.heading1_score
        + 0.18 * candidate.heading2_score
        + 0.14 * candidate.layout_score
    )

    return asdict(candidate)


def main():

    parser = argparse.ArgumentParser(
        description="Headless DC Sudoku image detector."
    )

    parser.add_argument(
        "--date",
        required=True,
    )

    parser.add_argument(
        "--page-min",
        type=int,
        default=DEFAULT_PAGE_MIN,
    )

    parser.add_argument(
        "--page-max",
        type=int,
        default=DEFAULT_PAGE_MAX,
    )

    parser.add_argument(
        "--variant-min",
        type=int,
        default=DEFAULT_VARIANT_MIN,
    )

    parser.add_argument(
        "--variant-max",
        type=int,
        default=DEFAULT_VARIANT_MAX,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
    )

    parser.add_argument(
        "--keep-all",
        action="store_true",
    )

    args = parser.parse_args()

    root = Path("dc_test") / args.date
    temp = root / "temp_images"
    sudoku = root / "sudoku"
    ranked = sudoku / "ranked"
    other = root / "other_images"

    if root.exists():
        shutil.rmtree(root)

    temp.mkdir(parents=True)
    sudoku.mkdir(parents=True)
    ranked.mkdir(parents=True)

    reference = load(REFERENCE)
    heading1 = load(HEADING1)
    heading2 = load(HEADING2)

    print("=" * 72)
    print("DC SUDOKU IMAGE DISCOVERY v3")
    print("=" * 72)
    print(
        f"Reference: "
        f"{reference.shape[1]}x{reference.shape[0]}"
    )
    print(
        f"Scanning: pages {args.page_min}-{args.page_max}, "
        f"variants {args.variant_min}-{args.variant_max}"
    )

    tasks = [
        (page, variant)
        for page in range(
            args.page_min,
            args.page_max + 1,
        )
        for variant in range(
            args.variant_min,
            args.variant_max + 1,
        )
    ]

    downloaded = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = [
            executor.submit(
                download,
                args.date,
                page,
                variant,
                temp,
            )
            for page, variant in tasks
        ]

        for future in concurrent.futures.as_completed(futures):

            result = future.result()

            if result:
                downloaded.append(result)

    print(
        f"Downloaded {len(downloaded)} valid images."
    )

    ranked_items = []

    for item in downloaded:

        scored = score_candidate(
            item,
            reference,
            heading1,
            heading2,
        )

        ranked_items.append(scored)

    ranked_items.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    print()
    print("=" * 72)
    print("TOP CANDIDATES")
    print("=" * 72)

    for rank, item in enumerate(
        ranked_items[:15],
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"tabp{item['page']}_{item['variant']}.jpg "
            f"{item['width']}x{item['height']} "
            f"final={item['final_score']:.1f} "
            f"dim={item['dimension_score']:.1f} "
            f"h1={item['heading1_score']:.1f} "
            f"h2={item['heading2_score']:.1f} "
            f"layout={item['layout_score']:.1f}"
        )

    if not ranked_items:
        print("No images found.")
        return

    best = ranked_items[0]

    best_path = Path(
        best["path"]
    )

    selected = (
        sudoku
        / "sudoku_source.jpg"
    )

    shutil.copy2(
        best_path,
        selected,
    )

    for rank, item in enumerate(
        ranked_items[:10],
        start=1,
    ):
        src = Path(item["path"])
        dst = (
            ranked
            / f"rank_{rank:02d}_{src.name}"
        )
        shutil.copy2(
            src,
            dst,
        )

    if not args.keep_all:

        other.mkdir(
            parents=True,
            exist_ok=True,
        )

        for item in downloaded:

            src = Path(item.path)

            if src.resolve() == best_path.resolve():
                continue

            shutil.move(
                str(src),
                str(other / src.name),
            )

    diagnostics = {
        "date": args.date,
        "reference_dimensions": {
            "width": REFERENCE_W,
            "height": REFERENCE_H,
        },
        "weights": {
            "dimensions": 0.50,
            "heading1": 0.18,
            "heading2": 0.18,
            "layout": 0.14,
        },
        "downloaded": len(downloaded),
        "best": best,
        "top50": ranked_items[:50],
    }

    (root / "detection.json").write_text(
        json.dumps(
            diagnostics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("SELECTED")
    print("=" * 72)

    print(
        f"tabp{best['page']}_{best['variant']}.jpg"
    )

    print(
        "Size:",
        f"{best['width']}x{best['height']}",
    )

    print(
        "Final:",
        f"{best['final_score']:.2f}",
    )

    print(
        "Dimensions:",
        f"{best['dimension_score']:.2f}",
    )

    print(
        "Heading 1:",
        f"{best['heading1_score']:.2f}",
    )

    print(
        "Heading 2:",
        f"{best['heading2_score']:.2f}",
    )

    print(
        "Layout:",
        f"{best['layout_score']:.2f}",
    )

    print()
    print(
        "Sudoku source:",
        selected,
    )

    print(
        "Diagnostics:",
        root / "detection.json",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rapidocr import RapidOCR

# These are retained as approximate source-image regions. The source image is
# deliberately kept at its original dimensions; no global resize/normalization
# is performed before cropping.
REGIONS = {
    "dc-1": (40, 75, 341, 339),
    "dc-2": (383, 70, 689, 338),
}

MIN_DIGIT_SCORE = 0.35


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def crop_region(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid crop {box} for source {w}x{h}")
    return image[y1:y2, x1:x2]


def extract_cell(crop: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = crop.shape[:2]
    y1 = round(r * h / 9)
    y2 = round((r + 1) * h / 9)
    x1 = round(c * w / 9)
    x2 = round((c + 1) * w / 9)
    # Small inset prevents the outer grid line from being passed to OCR,
    # without altering the source image itself.
    mx = max(2, round((x2 - x1) * 0.08))
    my = max(2, round((y2 - y1) * 0.08))
    cell = crop[y1 + my:y2 - my, x1 + mx:x2 - mx]
    if cell.size == 0:
        raise RuntimeError(f"Empty cell R{r + 1}C{c + 1}")
    return cell


def recognize_digit(engine: RapidOCR, cell: np.ndarray) -> tuple[int, float, str]:
    """Recognize a single Sudoku cell using RapidOCR recognition only.

    Detection is disabled because each input is already a single cell. The
    recognized text is accepted only when it is exactly one digit 1..9.
    """
    try:
        result = engine(cell, use_det=False, use_cls=True, use_rec=True)
    except Exception:
        return 0, 0.0, ""

    texts: list[str] = []
    scores: list[float] = []

    # RapidOCR's current TextRecOutput exposes txts and scores. Keep a small
    # compatibility fallback for older result representations.
    raw_txts = getattr(result, "txts", None)
    raw_scores = getattr(result, "scores", None)
    if raw_txts is not None and raw_scores is not None:
        texts = list(raw_txts)
        scores = [float(x) for x in raw_scores]
    else:
        data = getattr(result, "res", None)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    texts.append(str(item[0]))
                    try:
                        scores.append(float(item[1]))
                    except (TypeError, ValueError):
                        scores.append(0.0)

    best_digit, best_score, best_text = 0, 0.0, ""
    for text, score in zip(texts, scores):
        clean = str(text).strip()
        if len(clean) != 1 or clean not in "123456789":
            continue
        if score > best_score:
            best_digit, best_score, best_text = int(clean), float(score), clean

    if best_score < MIN_DIGIT_SCORE:
        return 0, round(best_score, 4), best_text
    return best_digit, round(best_score, 4), best_text


def read_grid(crop: np.ndarray, engine: RapidOCR) -> tuple[list[list[int]], list[list[float]], list[list[str]]]:
    grid = [[0] * 9 for _ in range(9)]
    confidence = [[0.0] * 9 for _ in range(9)]
    raw = [[""] * 9 for _ in range(9)]

    for r in range(9):
        for c in range(9):
            cell = extract_cell(crop, r, c)
            digit, score, text = recognize_digit(engine, cell)
            grid[r][c] = digit
            confidence[r][c] = score
            raw[r][c] = text

    return grid, confidence, raw


def solution_count(grid: list[list[int]], limit: int = 2) -> int:
    board = [row[:] for row in grid]

    def valid(r: int, c: int, n: int) -> bool:
        if n in board[r]:
            return False
        if any(board[i][c] == n for i in range(9)):
            return False
        br, bc = (r // 3) * 3, (c // 3) * 3
        return all(board[i][j] != n for i in range(br, br + 3) for j in range(bc, bc + 3))

    def search(count: list[int]) -> None:
        if count[0] >= limit:
            return
        best = None
        options = None
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    opts = [n for n in range(1, 10) if valid(r, c, n)]
                    if not opts:
                        return
                    if options is None or len(opts) < len(options):
                        best, options = (r, c), opts
        if best is None:
            count[0] += 1
            return
        r, c = best
        for n in options:
            board[r][c] = n
            search(count)
            board[r][c] = 0
            if count[0] >= limit:
                return

    count = [0]
    search(count)
    return count[0]


def validate(grid: list[list[int]]) -> dict[str, Any]:
    errors: list[str] = []
    for r in range(9):
        vals = [n for n in grid[r] if n]
        if len(vals) != len(set(vals)):
            errors.append(f"row {r + 1} conflict")
    for c in range(9):
        vals = [grid[r][c] for r in range(9) if grid[r][c]]
        if len(vals) != len(set(vals)):
            errors.append(f"column {c + 1} conflict")
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            vals = [grid[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3) if grid[r][c]]
            if len(vals) != len(set(vals)):
                errors.append(f"box {br // 3 + 1},{bc // 3 + 1} conflict")
    count = solution_count(grid)
    return {
        "valid": not errors and count > 0,
        "unique": count == 1,
        "solution_count": count,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless DC Sudoku crop/OCR pipeline using RapidOCR.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    root = Path("dc_test") / args.date / "sudoku"
    root.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source)
    image = load_image(source_path)
    source_h, source_w = image.shape[:2]

    # Do not normalize/resize the source image.
    print(f"Source dimensions: {source_w}x{source_h}")
    print("Source normalization: DISABLED")

    engine = RapidOCR()
    puzzles = []

    for puzzle_id, ref_box in REGIONS.items():
        crop = crop_region(image, ref_box)
        crop_path = root / f"{puzzle_id}.png"
        cv2.imwrite(str(crop_path), crop)

        grid, confidence, raw = read_grid(crop, engine)
        checks = validate(grid)
        puzzles.append({
            "id": puzzle_id,
            "title": "Sudoku 1" if puzzle_id == "dc-1" else "Sudoku 2",
            "verified": bool(checks["valid"] and checks["unique"]),
            "grid": grid,
            "ocr_confidence": confidence,
            "ocr_raw": raw,
            "validation": checks,
            "crop": str(crop_path),
        })

    result = {
        "date": args.date,
        "edition": "Hyderabad",
        "source": "Deccan Chronicle",
        "source_image": str(source_path),
        "source_dimensions": {"width": source_w, "height": source_h},
        "normalization": {"enabled": False},
        "ocr_engine": {
            "name": "RapidOCR",
            "engine": "ONNX Runtime (default)",
            "recognition_only": True,
            "minimum_digit_score": MIN_DIGIT_SCORE,
        },
        "puzzles": puzzles,
    }
    (root / "ocr_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    canonical = {
        "date": result["date"],
        "edition": result["edition"],
        "source": result["source"],
        "puzzles": [
            {
                "id": p["id"],
                "title": p["title"],
                "verified": bool(p.get("verified", False)),
                "grid": p["grid"],
            }
            for p in result["puzzles"]
        ],
    }
    (root / "today.json").write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")

    print("PIPELINE STATUS: COMPLETE")
    for p in puzzles:
        print(p["title"], p["validation"])


if __name__ == "__main__":
    main()

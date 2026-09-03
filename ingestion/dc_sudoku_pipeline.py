from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract

# The DC Coffee-Break image has a stable layout.  Instead of asking a generic
# OCR model to interpret the whole newspaper image, we explicitly isolate the
# two Sudoku boards first and discard everything else.
# Coordinates below are for the canonical 732x606 source image used by the
# current DC assets.  They are scaled only when the downloaded asset has a
# slightly different pixel size; the source image itself is never resized.
CANONICAL_SIZE = (732, 606)  # width, height

BOARD_BOXES = {
    "dc-1": (40, 74, 342, 340),
    "dc-2": (383, 72, 691, 340),
}

# Detected grid-line centers for the canonical source.  Removing these lines
# leaves almost pure digit components inside the 81 cells.
GRID_LINES = {
    "dc-1": {
        "x": [1, 35, 68, 101, 135, 167, 201, 234, 267, 300],
        "y": [2, 31, 60, 89, 118, 148, 177, 206, 235, 264],
    },
    "dc-2": {
        "x": [1, 35, 68, 102, 136, 169, 203, 236, 270, 304],
        "y": [0, 29, 58, 88, 117, 147, 176, 206, 235, 265],
    },
}


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def scaled_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    sx = width / CANONICAL_SIZE[0]
    sy = height / CANONICAL_SIZE[1]
    x1, y1, x2, y2 = box
    return (
        round(x1 * sx),
        round(y1 * sy),
        round(x2 * sx),
        round(y2 * sy),
    )


def scaled_lines(lines: list[int], scale: float) -> list[int]:
    return [round(v * scale) for v in lines]


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


def build_digit_mask(crop: np.ndarray, x_lines: list[int], y_lines: list[int]) -> np.ndarray:
    """Return a binary mask containing only ink inside the Sudoku cells."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    keep = np.full_like(ink, 255)
    line_half = max(2, round(min(crop.shape[:2]) / 140))

    # Erase every known vertical/horizontal grid line, including the heavy 3x3
    # box boundaries.  This is intentionally deterministic for the DC layout.
    for x in x_lines:
        cv2.line(keep, (x, 0), (x, keep.shape[0] - 1), 0, line_half * 2 + 1)
    for y in y_lines:
        cv2.line(keep, (0, y), (keep.shape[1] - 1, y), 0, line_half * 2 + 1)

    cleaned = cv2.bitwise_and(ink, keep)
    return cleaned


def tesseract_digit(component: np.ndarray) -> tuple[int, list[str]]:
    """Recognize one already-isolated printed digit.

    Multiple Tesseract segmentation modes are used because the component is
    tiny.  We accept the most frequent valid 1..9 result.
    """
    white_on_black = component
    black_on_white = cv2.bitwise_not(white_on_black)
    variants = [
        cv2.resize(black_on_white, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC),
        cv2.resize(
            cv2.morphologyEx(black_on_white, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)),
            None,
            fx=5,
            fy=5,
            interpolation=cv2.INTER_CUBIC,
        ),
    ]

    votes: list[str] = []
    for image in variants:
        for psm in (10, 8, 13):
            text = pytesseract.image_to_string(
                image,
                config=f"--psm {psm} -c tessedit_char_whitelist=123456789",
            ).strip()
            if len(text) == 1 and text in "123456789":
                votes.append(text)

    if not votes:
        return 0, []

    # Deterministic majority vote; ties are resolved by the first observed vote.
    counts: dict[str, int] = {}
    for vote in votes:
        counts[vote] = counts.get(vote, 0) + 1
    best = max(votes, key=lambda value: counts[value])
    return int(best), votes


def ocr_grid(crop: np.ndarray, x_lines: list[int], y_lines: list[int]) -> tuple[list[list[int]], dict[str, Any], np.ndarray]:
    """OCR a Sudoku crop after grid-line removal and component detection."""
    cleaned = build_digit_mask(crop, x_lines, y_lines)
    grid = [[0] * 9 for _ in range(9)]
    occupied = 0
    details: list[dict[str, Any]] = []

    count, labels, stats, centers = cv2.connectedComponentsWithStats(cleaned, 8)
    for idx in range(1, count):
        x, y, width, height, area = stats[idx]
        cx, cy = centers[idx]

        # Printed Sudoku digits in these crops are small, isolated components.
        # These guards also prevent tiny anti-aliasing specks from becoming clues.
        if area < 20 or width < 3 or height < 8 or width > 25 or height > 25:
            continue

        col = int(np.searchsorted(x_lines, cx, side="right") - 1)
        row = int(np.searchsorted(y_lines, cy, side="right") - 1)
        if not (0 <= row < 9 and 0 <= col < 9):
            continue

        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(cleaned.shape[1], x + width + pad)
        y1 = min(cleaned.shape[0], y + height + pad)
        component = (labels[y0:y1, x0:x1] == idx).astype(np.uint8) * 255

        digit, votes = tesseract_digit(component)
        if digit:
            grid[row][col] = digit
            occupied += 1

        details.append({
            "row": row + 1,
            "column": col + 1,
            "bbox": [int(x), int(y), int(width), int(height)],
            "area": int(area),
            "digit": digit,
            "votes": votes,
        })

    return grid, {
        "engine": "Tesseract OCR on isolated digit components",
        "layout": "fixed Deccan Chronicle Sudoku board mask",
        "occupied_cells": occupied,
        "components": details,
        "note": "Only the two Sudoku board regions are OCR'd. Grid lines are masked before connected-component OCR; the original source image is never globally resized or normalized.",
    }, cleaned


def draw_grid_debug(crop: np.ndarray, x_lines: list[int], y_lines: list[int], grid: list[list[int]]) -> np.ndarray:
    debug = crop.copy()
    for x in x_lines:
        cv2.line(debug, (x, 0), (x, debug.shape[0] - 1), (0, 0, 255), 1)
    for y in y_lines:
        cv2.line(debug, (0, y), (debug.shape[1] - 1, y), (0, 0, 255), 1)

    for r in range(9):
        for c in range(9):
            value = grid[r][c]
            if not value:
                continue
            x1, x2 = x_lines[c], x_lines[c + 1]
            y1, y2 = y_lines[r], y_lines[r + 1]
            cv2.putText(
                debug,
                str(value),
                (int((x1 + x2) / 2 - 7), int((y1 + y2) / 2 + 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return debug


def solution_count(grid: list[list[int]], limit: int = 2) -> int:
    board = [row[:] for row in grid]

    def valid(r: int, c: int, n: int) -> bool:
        if n in board[r]:
            return False
        if any(board[i][c] == n for i in range(9)):
            return False
        br, bc = (r // 3) * 3, (c // 3) * 3
        return all(board[i][j] != n for i in range(br, br + 3) for j in range(bc, bc + 3))

    def search(found: list[int]) -> None:
        if found[0] >= limit:
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
            found[0] += 1
            return
        r, c = best
        for n in options:
            board[r][c] = n
            search(found)
            board[r][c] = 0
            if found[0] >= limit:
                return

    found = [0]
    search(found)
    return found[0]


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
    parser = argparse.ArgumentParser(description="Headless DC Sudoku OCR using a fixed board mask and Tesseract.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    root = Path("dc_test") / args.date / "sudoku"
    root.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source)
    image = load_image(source_path)
    source_h, source_w = image.shape[:2]

    print(f"Source dimensions: {source_w}x{source_h}")
    print("Source normalization: DISABLED")
    print("OCR strategy: fixed Sudoku board masks -> grid-line removal -> connected components -> Tesseract")

    sx = source_w / CANONICAL_SIZE[0]
    sy = source_h / CANONICAL_SIZE[1]
    puzzles = []

    for puzzle_id, box in BOARD_BOXES.items():
        scaled = scaled_box(box, source_w, source_h)
        crop = crop_region(image, scaled)
        crop_path = root / f"{puzzle_id}-raw-crop.png"
        cv2.imwrite(str(crop_path), crop)

        canonical_lines = GRID_LINES[puzzle_id]
        x_lines = scaled_lines(canonical_lines["x"], sx)
        y_lines = scaled_lines(canonical_lines["y"], sy)
        # Convert canonical board-local coordinates into the current crop size.
        board_sx = crop.shape[1] / (box[2] - box[0])
        board_sy = crop.shape[0] / (box[3] - box[1])
        x_lines = [round(v * board_sx) for v in canonical_lines["x"]]
        y_lines = [round(v * board_sy) for v in canonical_lines["y"]]

        print(f"{puzzle_id}: crop={crop.shape[1]}x{crop.shape[0]}")

        try:
            grid, ocr_meta, cleaned = ocr_grid(crop, x_lines, y_lines)
            checks = validate(grid)
            error = None
        except Exception as exc:
            grid = [[0] * 9 for _ in range(9)]
            checks = validate(grid)
            ocr_meta = {
                "engine": "Tesseract OCR",
                "error": f"{type(exc).__name__}: {exc}",
            }
            cleaned = build_digit_mask(crop, x_lines, y_lines)
            error = str(exc)

        cv2.imwrite(str(root / f"{puzzle_id}-masked.png"), cleaned)
        cv2.imwrite(str(root / f"{puzzle_id}-grid-debug.png"), draw_grid_debug(crop, x_lines, y_lines, grid))

        print(f"{puzzle_id} grid: {grid}")
        print(f"{puzzle_id} validation: {checks}")

        puzzles.append({
            "id": puzzle_id,
            "title": "Sudoku 1" if puzzle_id == "dc-1" else "Sudoku 2",
            "verified": bool(checks["valid"] and checks["unique"]),
            "grid": grid,
            "ocr": ocr_meta,
            "validation": checks,
            "crop": str(crop_path),
            "masked": str(root / f"{puzzle_id}-masked.png"),
            "grid_debug": str(root / f"{puzzle_id}-grid-debug.png"),
            "error": error,
        })

    result = {
        "date": args.date,
        "edition": "Hyderabad",
        "source": "Deccan Chronicle",
        "source_image": str(source_path),
        "source_dimensions": {"width": source_w, "height": source_h},
        "normalization": {"enabled": False},
        "ocr_engine": "Tesseract on fixed Sudoku masks",
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
    for puzzle in puzzles:
        print(puzzle["title"], puzzle["validation"])


if __name__ == "__main__":
    main()

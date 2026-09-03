from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract

# IMPORTANT: dc_sudoku_detect.py owns source discovery and is intentionally
# untouched. This OCR stage receives the selected 732x606 sudoku_source.jpg.
# The DC Coffee-Break page has a stable layout, so we use fixed board/cell
# geometry and never ask OCR to read the surrounding newspaper text.
CANONICAL_SIZE = (732, 606)  # width, height

BOARD_BOXES = {
    "dc-1": (40, 74, 342, 340),
    "dc-2": (383, 72, 691, 340),
}

# Board-local grid-line coordinates measured on the 732x606 source.
# These are the 10 vertical and 10 horizontal boundaries of each 9x9 board.
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

DIGITS = "123456789"


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def crop_region(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid crop {box} for source {w}x{h}")
    return image[y1:y2, x1:x2]


def scaled_lines(lines: list[int], scale: float) -> list[int]:
    return [round(v * scale) for v in lines]


def isolate_digit(cell: np.ndarray) -> tuple[np.ndarray | None, float, int]:
    """Isolate only the digit ink from one already-separated Sudoku cell."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell

    variants = [
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        ),
    ]

    best: np.ndarray | None = None
    best_area = 0
    h, w = gray.shape[:2]

    for binary in variants:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for idx in range(1, count):
            x, y, cw, ch, area = [int(v) for v in stats[idx]]
            if area < 5:
                continue
            if x <= 0 or y <= 0 or x + cw >= w - 1 or y + ch >= h - 1:
                continue
            # A real printed digit fits comfortably inside our cell.
            if cw < 2 or ch < 5 or cw > int(w * 0.90) or ch > int(h * 0.95):
                continue
            if area > best_area:
                best_area = area
                best = (labels == idx).astype(np.uint8) * 255

    if best is None or best_area < 10:
        return None, 0.0, 0

    return best, best_area / float(max(1, w * h)), best_area


def normalize_digit(mask: np.ndarray, size: int = 40) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    roi = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = max(roi.shape)
    canvas = np.zeros((side, side), dtype=np.uint8)
    oy = (side - roi.shape[0]) // 2
    ox = (side - roi.shape[1]) // 2
    canvas[oy:oy + roi.shape[0], ox:ox + roi.shape[1]] = roi
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_CUBIC)


def ocr_digit(sample: np.ndarray) -> tuple[int, list[str]]:
    """Recognize one isolated digit with two independent Tesseract segmenters."""
    white_background = cv2.bitwise_not(sample)
    rendered = cv2.copyMakeBorder(
        white_background, 14, 14, 14, 14, cv2.BORDER_CONSTANT, value=255
    )
    rendered = cv2.resize(rendered, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    votes: list[str] = []
    for psm in (10, 13):
        text = pytesseract.image_to_string(
            rendered,
            config=f"--psm {psm} -c tessedit_char_whitelist={DIGITS}",
        ).strip()
        if len(text) == 1 and text in DIGITS:
            votes.append(text)

    if not votes:
        return 0, []

    # Prefer agreement. If the two segmenters disagree, retain the first result
    # and mark the cell as lower confidence in metadata.
    if len(votes) == 2 and votes[0] == votes[1]:
        return int(votes[0]), votes
    return int(votes[0]), votes


def ocr_grid(
    crop: np.ndarray,
    x_lines: list[int],
    y_lines: list[int],
) -> tuple[list[list[int]], dict[str, Any], np.ndarray, np.ndarray]:
    grid = [[0] * 9 for _ in range(9)]
    confidence = [[0.0] * 9 for _ in range(9)]
    details: list[dict[str, Any]] = []

    # White canvas containing only the isolated digit masks. This is the actual
    # OCR debug mask; the original source/crop are never modified.
    masked = np.full(crop.shape[:2], 255, dtype=np.uint8)
    debug = crop.copy()

    for r in range(9):
        for c in range(9):
            # Keep a 3px moat around every known grid line. This is the critical
            # newspaper-specific isolation step.
            x1 = x_lines[c] + 3
            x2 = x_lines[c + 1] - 3
            y1 = y_lines[r] + 3
            y2 = y_lines[r + 1] - 3
            if x2 <= x1 or y2 <= y1:
                details.append({"row": r + 1, "column": c + 1, "digit": 0, "reason": "invalid_cell"})
                continue

            cell = crop[y1:y2, x1:x2]
            mask, ink_ratio, area = isolate_digit(cell)

            if mask is None:
                details.append({
                    "row": r + 1,
                    "column": c + 1,
                    "digit": 0,
                    "ink_ratio": round(float(ink_ratio), 5),
                    "area": area,
                    "reason": "empty",
                })
                continue

            sample = normalize_digit(mask)
            digit, votes = ocr_digit(sample)
            agreed = len(votes) == 2 and votes[0] == votes[1]
            conf = 0.95 if agreed else (0.55 if len(votes) == 2 else 0.45)

            grid[r][c] = digit
            confidence[r][c] = conf if digit else 0.0

            # Place the isolated digit mask back into its original cell location.
            local = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            masked[y1:y2, x1:x2] = np.minimum(masked[y1:y2, x1:x2], 255 - local)

            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 1)
            if digit:
                cv2.putText(
                    debug,
                    str(digit),
                    (int((x1 + x2) / 2 - 5), int((y1 + y2) / 2 + 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            details.append({
                "row": r + 1,
                "column": c + 1,
                "digit": digit,
                "votes": votes,
                "agreed": agreed,
                "confidence": round(float(conf), 4),
                "ink_ratio": round(float(ink_ratio), 5),
                "area": area,
            })

    return grid, {
        "engine": "Tesseract OCR on isolated Sudoku cells",
        "occupied_cells": sum(v != 0 for row in grid for v in row),
        "confidence": confidence,
        "components": details,
        "note": (
            "The selected DC image is processed at its original resolution. "
            "Each of the two fixed Sudoku boards is divided into 81 known cells; "
            "a 3px moat removes grid boundaries, then only the digit component "
            "inside that cell is sent to Tesseract."
        ),
    }, masked, debug


def solution_count(grid: list[list[int]], limit: int = 2) -> int:
    board = [row[:] for row in grid]

    def valid(r: int, c: int, n: int) -> bool:
        if n in board[r]:
            return False
        if any(board[i][c] == n for i in range(9)):
            return False
        br, bc = (r // 3) * 3, (c // 3) * 3
        return all(board[i][j] != n for i in range(br, br + 3) for j in range(bc, bc + 3))

    found = [0]

    def search() -> None:
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
            search()
            board[r][c] = 0
            if found[0] >= limit:
                return

    search()
    return found[0]


def validate(grid: list[list[int]]) -> dict[str, Any]:
    errors: list[str] = []
    for r in range(9):
        values = [n for n in grid[r] if n]
        if len(values) != len(set(values)):
            errors.append(f"row {r + 1} conflict")
    for c in range(9):
        values = [grid[r][c] for r in range(9) if grid[r][c]]
        if len(values) != len(set(values)):
            errors.append(f"column {c + 1} conflict")
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            values = [
                grid[r][c]
                for r in range(br, br + 3)
                for c in range(bc, bc + 3)
                if grid[r][c]
            ]
            if len(values) != len(set(values)):
                errors.append(f"box {br // 3 + 1},{bc // 3 + 1} conflict")

    count = solution_count(grid)
    return {
        "valid": not errors and count > 0,
        "unique": count == 1,
        "solution_count": count,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DC Sudoku OCR using fixed per-cell geometry.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    root = Path("dc_test") / args.date / "sudoku"
    root.mkdir(parents=True, exist_ok=True)

    image = load_image(Path(args.source))
    source_h, source_w = image.shape[:2]
    print(f"Source dimensions: {source_w}x{source_h}")
    if (source_w, source_h) != CANONICAL_SIZE:
        print(f"WARNING: expected {CANONICAL_SIZE}, received {(source_w, source_h)}")
    print("Source normalization: DISABLED")
    print("Detection stage: UNCHANGED")
    print("OCR: fixed board -> fixed cells -> isolated digit -> Tesseract")

    puzzles = []
    for puzzle_id, box in BOARD_BOXES.items():
        crop = crop_region(image, box)
        raw_path = root / f"{puzzle_id}-raw-crop.png"
        cv2.imwrite(str(raw_path), crop)

        canonical = GRID_LINES[puzzle_id]
        sx = crop.shape[1] / float(box[2] - box[0])
        sy = crop.shape[0] / float(box[3] - box[1])
        x_lines = scaled_lines(canonical["x"], sx)
        y_lines = scaled_lines(canonical["y"], sy)

        print(f"{puzzle_id}: crop={crop.shape[1]}x{crop.shape[0]}")
        grid, ocr_meta, masked, debug = ocr_grid(crop, x_lines, y_lines)
        checks = validate(grid)

        masked_path = root / f"{puzzle_id}-masked.png"
        debug_path = root / f"{puzzle_id}-grid-debug.png"
        cv2.imwrite(str(masked_path), masked)
        cv2.imwrite(str(debug_path), debug)

        print(f"{puzzle_id} grid: {grid}")
        print(f"{puzzle_id} validation: {checks}")

        puzzles.append({
            "id": puzzle_id,
            "title": "Sudoku 1" if puzzle_id == "dc-1" else "Sudoku 2",
            "verified": bool(checks["valid"] and checks["unique"]),
            "grid": grid,
            "ocr": ocr_meta,
            "validation": checks,
            "crop": str(raw_path),
            "masked": str(masked_path),
            "grid_debug": str(debug_path),
            "error": None,
        })

    result = {
        "date": args.date,
        "edition": "Hyderabad",
        "source": "Deccan Chronicle",
        "source_image": str(args.source),
        "source_dimensions": {"width": source_w, "height": source_h},
        "normalization": {"enabled": False},
        "ocr_engine": "cell-isolated Tesseract",
        "puzzles": puzzles,
    }
    (root / "ocr_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    canonical = {
        "date": args.date,
        "edition": "Hyderabad",
        "source": "Deccan Chronicle",
        "puzzles": [
            {
                "id": p["id"],
                "title": p["title"],
                "verified": bool(p["verified"]),
                "grid": p["grid"],
            }
            for p in puzzles
        ],
    }
    (root / "today.json").write_text(
        json.dumps(canonical, indent=2) + "\n",
        encoding="utf-8",
    )

    print("PIPELINE STATUS: COMPLETE")


if __name__ == "__main__":
    main()

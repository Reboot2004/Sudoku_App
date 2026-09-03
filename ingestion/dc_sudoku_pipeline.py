from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract

# IMPORTANT: dc_sudoku_detect.py owns source-image discovery.  This file only
# performs OCR after detection has produced sudoku_source.jpg.
# The DC Coffee-Break page has a stable 732x606 layout, so OCR can use fixed
# board geometry rather than a generic Sudoku detector.
CANONICAL_SIZE = (732, 606)  # width, height

# These are BOARD-LOCAL coordinates for the two Sudoku grids.
BOARD_BOXES = {
    "dc-1": (40, 74, 342, 340),
    "dc-2": (383, 72, 691, 340),
}

# Exact grid-line locations measured on the 732x606 DC asset.  We use them only
# to define cell interiors; OCR never sees the grid lines.
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

# The page also contains yesterday's two completed 9x9 answer grids.  They are
# printed in the same typeface and resolution as today's puzzle digits.  We use
# them as an image-specific font/template bank, so recognition is adapted to
# the exact newspaper asset instead of relying on a generic CNN/font.
ANSWER_BOXES = {
    "dc-1": (131, 380, 343, 570),
    "dc-2": (496, 382, 706, 570),
}

TEMPLATE_SIZE = 32
DIGITS = "123456789"


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def scaled_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    sx = width / CANONICAL_SIZE[0]
    sy = height / CANONICAL_SIZE[1]
    x1, y1, x2, y2 = box
    return round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)


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


def cell_boxes_from_lines(
    x_lines: list[int], y_lines: list[int], inset: int = 3
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for r in range(9):
        for c in range(9):
            x1 = x_lines[c] + inset
            x2 = x_lines[c + 1] - inset
            y1 = y_lines[r] + inset
            y2 = y_lines[r + 1] - inset
            boxes.append((x1, y1, x2, y2))
    return boxes


def equal_cell_boxes(width: int, height: int, inset: int = 2) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for r in range(9):
        for c in range(9):
            x1 = round(c * width / 9) + inset
            x2 = round((c + 1) * width / 9) - inset
            y1 = round(r * height / 9) + inset
            y2 = round((r + 1) * height / 9) - inset
            boxes.append((x1, y1, x2, y2))
    return boxes


def threshold_variants(cell: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    variants: list[np.ndarray] = []
    variants.append(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1])
    variants.append(
        cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
    )
    return variants


def clean_digit_mask(mask: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Return the largest plausible ink component and its fill ratio."""
    cleaned = mask.copy()
    h, w = cleaned.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    candidates: list[tuple[int, int]] = []

    for idx in range(1, count):
        x, y, cw, ch, area = [int(v) for v in stats[idx]]
        if area < 4:
            continue
        if x <= 0 or y <= 0 or x + cw >= w or y + ch >= h:
            continue
        if cw < 2 or ch < 5:
            continue
        if cw > int(w * 0.90) or ch > int(h * 0.95):
            continue
        candidates.append((area, idx))

    if not candidates:
        return None, 0.0

    area, idx = max(candidates)
    component = np.zeros_like(cleaned)
    component[labels == idx] = 255
    return component, area / float(max(1, w * h))


def normalize_digit(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.uint8)

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    roi = mask[y1:y2, x1:x2]

    side = max(roi.shape)
    canvas = np.zeros((side, side), dtype=np.uint8)
    ox = (side - roi.shape[1]) // 2
    oy = (side - roi.shape[0]) // 2
    canvas[oy : oy + roi.shape[0], ox : ox + roi.shape[1]] = roi
    return cv2.resize(canvas, (TEMPLATE_SIZE, TEMPLATE_SIZE), interpolation=cv2.INTER_AREA)


def cell_digit_image(cell: np.ndarray) -> tuple[np.ndarray | None, float, int]:
    """Build a canonical digit bitmap from one isolated cell."""
    best_mask: np.ndarray | None = None
    best_ratio = 0.0
    best_area = 0

    for variant in threshold_variants(cell):
        mask, ratio = clean_digit_mask(variant)
        if mask is None:
            continue
        area = int(np.count_nonzero(mask))
        if area > best_area:
            best_mask = mask
            best_ratio = ratio
            best_area = area

    if best_mask is None:
        return None, 0.0, 0

    # Typical DC printed digits occupy a modest fraction of their cell.  Very
    # tiny components are newspaper noise, anti-aliasing or dust.
    if best_area < 12 or best_ratio < 0.015:
        return None, best_ratio, best_area

    return normalize_digit(best_mask), best_ratio, best_area


def row_ocr(answer_crop: np.ndarray) -> tuple[list[list[int]], dict[str, Any]]:
    """OCR a completed 9x9 answer grid row-by-row.

    Each row is already isolated from the rest of the newspaper. Tesseract is
    therefore asked to read nine digits rather than arbitrary page text.
    """
    grid = [[0] * 9 for _ in range(9)]
    row_details: list[dict[str, Any]] = []
    boxes = equal_cell_boxes(answer_crop.shape[1], answer_crop.shape[0], inset=1)

    for r in range(9):
        row_cells = []
        for c in range(9):
            x1, y1, x2, y2 = boxes[r * 9 + c]
            row_cells.append(answer_crop[y1:y2, x1:x2])
        row_image = np.concatenate(row_cells, axis=1)

        best_text = ""
        best_score = -1.0
        for variant in threshold_variants(row_image):
            rendered = cv2.bitwise_not(variant)
            rendered = cv2.copyMakeBorder(
                rendered, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255
            )
            rendered = cv2.resize(
                rendered, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
            )
            data = pytesseract.image_to_data(
                rendered,
                config="--psm 7 -c tessedit_char_whitelist=123456789",
                output_type=pytesseract.Output.DICT,
            )
            pieces: list[str] = []
            confs: list[float] = []
            for text, conf in zip(data["text"], data["conf"]):
                cleaned = "".join(ch for ch in text if ch in DIGITS)
                if cleaned:
                    pieces.append(cleaned)
                    try:
                        confs.append(float(conf))
                    except ValueError:
                        pass
            text = "".join(pieces)
            if len(text) == 9:
                score = float(np.mean(confs)) if confs else 0.0
                if score > best_score:
                    best_text = text
                    best_score = score

        if len(best_text) == 9:
            grid[r] = [int(ch) for ch in best_text]
        row_details.append({"row": r + 1, "text": best_text, "confidence": best_score})

    return grid, {"rows": row_details}


def build_template_bank(
    image: np.ndarray,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    """Learn the printed 1..9 shapes from yesterday's completed answers."""
    templates: dict[str, list[np.ndarray]] = defaultdict(list)
    sources: list[dict[str, Any]] = []

    for name, box in ANSWER_BOXES.items():
        crop = crop_region(image, box)
        answer_grid, meta = row_ocr(crop)
        sources.append({"id": name, "ocr": answer_grid, "meta": meta})

        boxes = equal_cell_boxes(crop.shape[1], crop.shape[0], inset=2)
        for r in range(9):
            for c in range(9):
                digit = answer_grid[r][c]
                if digit not in range(1, 10):
                    continue
                x1, y1, x2, y2 = boxes[r * 9 + c]
                sample, _, _ = cell_digit_image(crop[y1:y2, x1:x2])
                if sample is not None:
                    templates[str(digit)].append(sample)

    summary = {digit: len(values) for digit, values in templates.items()}
    if len(templates) < 9:
        missing = [digit for digit in DIGITS if digit not in templates]
        raise RuntimeError(
            "Could not build a complete newspaper digit template bank; "
            f"missing digits: {missing}"
        )

    return dict(templates), {"template_counts": summary, "answer_sources": sources}


def template_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric bitmap distance; lower is better."""
    best = 1.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
            if dy < 0:
                shifted[dy:, :] = 0
            elif dy > 0:
                shifted[:dy, :] = 0
            if dx < 0:
                shifted[:, dx:] = 0
            elif dx > 0:
                shifted[:, :dx] = 0
            score = float(np.mean(cv2.absdiff(shifted, b))) / 255.0
            best = min(best, score)
    return best


def classify_template(
    sample: np.ndarray, templates: dict[str, list[np.ndarray]]
) -> tuple[int, float, list[dict[str, Any]]]:
    ranked: list[tuple[float, str]] = []
    for digit, samples in templates.items():
        distance = min(template_distance(sample, ref) for ref in samples)
        ranked.append((distance, digit))
    ranked.sort()
    best_distance, best_digit = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 1.0
    margin = max(0.0, runner_up - best_distance)
    confidence = max(0.0, min(1.0, 1.0 - best_distance))
    if margin < 0.015:
        confidence *= 0.75

    top = [
        {"digit": int(d), "distance": round(float(dist), 5)}
        for dist, d in ranked[:3]
    ]
    return int(best_digit), float(confidence), top


def ocr_puzzle(
    crop: np.ndarray,
    x_lines: list[int],
    y_lines: list[int],
    templates: dict[str, list[np.ndarray]],
) -> tuple[list[list[int]], dict[str, Any], np.ndarray]:
    grid = [[0] * 9 for _ in range(9)]
    confidence = [[0.0] * 9 for _ in range(9)]
    best_candidates = [[[] for _ in range(9)] for _ in range(9)]
    occupied = 0
    debug = crop.copy()
    details: list[dict[str, Any]] = []

    for r in range(9):
        for c in range(9):
            x1 = x_lines[c] + 3
            x2 = x_lines[c + 1] - 3
            y1 = y_lines[r] + 3
            y2 = y_lines[r + 1] - 3
            if x2 <= x1 or y2 <= y1:
                continue

            cell = crop[y1:y2, x1:x2]
            sample, ink_ratio, area = cell_digit_image(cell)
            if sample is None:
                details.append(
                    {
                        "row": r + 1,
                        "column": c + 1,
                        "digit": 0,
                        "confidence": 0.0,
                        "ink_ratio": round(float(ink_ratio), 5),
                        "area": area,
                    }
                )
                continue

            digit, conf, ranked = classify_template(sample, templates)
            grid[r][c] = digit
            confidence[r][c] = round(conf, 4)
            best_candidates[r][c] = ranked
            occupied += 1

            # Debug image: red cell rectangle + recognized digit in blue.
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 1)
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

            details.append(
                {
                    "row": r + 1,
                    "column": c + 1,
                    "digit": digit,
                    "confidence": round(conf, 4),
                    "ink_ratio": round(float(ink_ratio), 5),
                    "area": area,
                    "candidates": ranked,
                }
            )

    return grid, {
        "engine": "image-specific template matching",
        "template_source": "Deccan Chronicle yesterday-answer grids",
        "occupied_cells": occupied,
        "confidence": confidence,
        "candidates": best_candidates,
        "components": details,
        "note": (
            "The Sudoku boards are isolated using fixed DC geometry. Each cell is "
            "OCR'd independently against digit templates learned from the two "
            "completed answer grids in the same 732x606 source image. No global "
            "resize, normalization, perspective correction or generic Sudoku "
            "detection is performed."
        ),
    }, debug


def solution_count(grid: list[list[int]], limit: int = 2) -> int:
    board = [row[:] for row in grid]

    def valid(r: int, c: int, n: int) -> bool:
        if n in board[r]:
            return False
        if any(board[i][c] == n for i in range(9)):
            return False
        br, bc = (r // 3) * 3, (c // 3) * 3
        return all(
            board[i][j] != n
            for i in range(br, br + 3)
            for j in range(bc, bc + 3)
        )

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
            vals = [
                grid[r][c]
                for r in range(br, br + 3)
                for c in range(bc, bc + 3)
                if grid[r][c]
            ]
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
    parser = argparse.ArgumentParser(
        description="DC Sudoku OCR using fixed board geometry and image-specific digit templates."
    )
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
    print("Detection stage: UNCHANGED")
    print("OCR: fixed board cells + same-image digit templates")

    templates, template_meta = build_template_bank(image)
    print(f"Template counts: {template_meta['template_counts']}")
    (root / "template_meta.json").write_text(
        json.dumps(template_meta, indent=2), encoding="utf-8"
    )

    puzzles = []
    for puzzle_id, box in BOARD_BOXES.items():
        scaled = scaled_box(box, source_w, source_h)
        crop = crop_region(image, scaled)
        raw_path = root / f"{puzzle_id}-raw-crop.png"
        cv2.imwrite(str(raw_path), crop)

        # Keep the canonical DC cell geometry. Scaling is only applied to the
        # coordinates if an asset has a different total pixel size.
        canonical_lines = GRID_LINES[puzzle_id]
        sx = crop.shape[1] / float(box[2] - box[0])
        sy = crop.shape[0] / float(box[3] - box[1])
        x_lines = [round(v * sx) for v in canonical_lines["x"]]
        y_lines = [round(v * sy) for v in canonical_lines["y"]]

        print(f"{puzzle_id}: crop={crop.shape[1]}x{crop.shape[0]}")
        grid, ocr_meta, debug = ocr_puzzle(
            crop, x_lines, y_lines, templates
        )
        checks = validate(grid)

        masked_path = root / f"{puzzle_id}-masked.png"
        debug_path = root / f"{puzzle_id}-grid-debug.png"
        cv2.imwrite(str(masked_path), cv2.cvtColor(debug, cv2.COLOR_BGR2GRAY))
        cv2.imwrite(str(debug_path), debug)

        print(f"{puzzle_id} grid: {grid}")
        print(f"{puzzle_id} validation: {checks}")

        puzzles.append(
            {
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
            }
        )

    result = {
        "date": args.date,
        "edition": "Hyderabad",
        "source": "Deccan Chronicle",
        "source_image": str(source_path),
        "source_dimensions": {"width": source_w, "height": source_h},
        "normalization": {"enabled": False},
        "ocr_engine": "same-image digit template matching",
        "template_meta": template_meta,
        "puzzles": puzzles,
    }
    (root / "ocr_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    canonical = {
        "date": result["date"],
        "edition": result["edition"],
        "source": result["source"],
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
        json.dumps(canonical, indent=2) + "\n", encoding="utf-8"
    )

    print("PIPELINE STATUS: COMPLETE")
    for puzzle in puzzles:
        print(puzzle["title"], puzzle["validation"])


if __name__ == "__main__":
    main()

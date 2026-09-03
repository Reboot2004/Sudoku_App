from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract

# Detection owns discovery of sudoku_source.jpg. OCR only consumes that image.
CANONICAL_SIZE = (732, 606)  # width, height
OCR_SCALE = 4
BOARD_BOXES = {
    "dc-1": (40, 74, 342, 340),
    "dc-2": (383, 72, 691, 340),
}
GRID_LINES = {
    "dc-1": {"x": [1, 35, 68, 101, 135, 167, 201, 234, 267, 300], "y": [2, 31, 60, 89, 118, 148, 177, 206, 235, 264]},
    "dc-2": {"x": [1, 35, 68, 102, 136, 169, 203, 236, 270, 304], "y": [0, 29, 58, 88, 117, 147, 176, 206, 235, 265]},
}
DIGITS = "123456789"


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def scaled_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    sx = width / CANONICAL_SIZE[0]
    sy = height / CANONICAL_SIZE[1]
    x1, y1, x2, y2 = box
    return round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)


def crop_region(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid crop {box} for source {w}x{h}")
    return image[y1:y2, x1:x2]


def cell_boxes(x_lines: list[int], y_lines: list[int], inset: int = 3):
    for r in range(9):
        for c in range(9):
            yield r, c, (x_lines[c] + inset, y_lines[r] + inset, x_lines[c + 1] - inset, y_lines[r + 1] - inset)


def preprocess_cell(cell: np.ndarray) -> list[np.ndarray]:
    """Upscale only for OCR; the saved/display image is never resized."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=OCR_SCALE, fy=OCR_SCALE, interpolation=cv2.INTER_CUBIC)
    enlarged = cv2.copyMakeBorder(enlarged, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)

    variants = [enlarged]
    variants.append(cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
    variants.append(cv2.adaptiveThreshold(enlarged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7))
    return variants


def ocr_cell(cell: np.ndarray) -> tuple[int, float, list[str]]:
    votes: list[str] = []
    confs: list[float] = []
    for image in preprocess_cell(cell):
        for psm in (10, 13):
            data = pytesseract.image_to_data(
                image,
                config=f"--psm {psm} -c tessedit_char_whitelist={DIGITS}",
                output_type=pytesseract.Output.DICT,
            )
            for text, conf in zip(data["text"], data["conf"]):
                text = "".join(ch for ch in text if ch in DIGITS)
                if len(text) == 1:
                    votes.append(text)
                    try:
                        confs.append(float(conf))
                    except ValueError:
                        pass
    if not votes:
        return 0, 0.0, []
    counts: dict[str, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts, key=counts.get)
    vote_score = counts[best] / len(votes)
    mean_conf = (float(np.mean(confs)) / 100.0) if confs else 0.0
    return int(best), round(0.5 * vote_score + 0.5 * max(0.0, min(1.0, mean_conf)), 4), votes


def ocr_grid(crop: np.ndarray, x_lines: list[int], y_lines: list[int]) -> tuple[list[list[int]], dict[str, Any], np.ndarray]:
    grid = [[0] * 9 for _ in range(9)]
    confidence = [[0.0] * 9 for _ in range(9)]
    cells_meta: list[dict[str, Any]] = []
    debug = crop.copy()
    occupied = 0

    for r, c, (x1, y1, x2, y2) in cell_boxes(x_lines, y_lines):
        cell = crop[y1:y2, x1:x2]
        digit, conf, votes = ocr_cell(cell)
        grid[r][c] = digit
        confidence[r][c] = conf
        if digit:
            occupied += 1
            cv2.putText(debug, str(digit), (int((x1 + x2) / 2 - 5), int((y1 + y2) / 2 + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
        cells_meta.append({"row": r + 1, "column": c + 1, "digit": digit, "confidence": conf, "votes": votes, "cell": [x1, y1, x2, y2]})

    for x in x_lines:
        cv2.line(debug, (x, 0), (x, debug.shape[0] - 1), (0, 0, 255), 1)
    for y in y_lines:
        cv2.line(debug, (0, y), (debug.shape[1] - 1, y), (0, 0, 255), 1)

    meta = {
        "engine": "Tesseract per-cell OCR",
        "ocr_scale": OCR_SCALE,
        "occupied_cells": occupied,
        "confidence": confidence,
        "cells": cells_meta,
        "note": "Original cropped Sudoku image is retained unchanged for app display. Upscaling is applied only to OCR working cells.",
    }
    return grid, meta, debug


def solution_count(grid: list[list[int]], limit: int = 2) -> int:
    board = [row[:] for row in grid]
    def valid(r: int, c: int, n: int) -> bool:
        if n in board[r] or any(board[i][c] == n for i in range(9)):
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
    return {"valid": not errors and count > 0, "unique": count == 1, "solution_count": count, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="DC Sudoku OCR: fixed cells with OCR-only enlargement.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    root = Path("dc_test") / args.date / "sudoku"
    root.mkdir(parents=True, exist_ok=True)
    image = load_image(Path(args.source))
    source_h, source_w = image.shape[:2]

    print(f"Source dimensions: {source_w}x{source_h}")
    print("Source normalization: DISABLED")
    print("Detection stage: UNCHANGED")
    print(f"OCR: isolated cells + {OCR_SCALE}x OCR-only enlargement")

    puzzles = []
    for puzzle_id, box in BOARD_BOXES.items():
        crop = crop_region(image, scaled_box(box, source_w, source_h))
        image_path = root / f"{puzzle_id}.jpg"
        raw_path = root / f"{puzzle_id}-raw-crop.png"
        debug_path = root / f"{puzzle_id}-ocr-debug.jpg"
        cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(raw_path), crop)

        geometry = GRID_LINES[puzzle_id]
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        sx = crop.shape[1] / float(box_w)
        sy = crop.shape[0] / float(box_h)
        x_lines = [round(v * sx) for v in geometry["x"]]
        y_lines = [round(v * sy) for v in geometry["y"]]

        grid, ocr_meta, debug = ocr_grid(crop, x_lines, y_lines)
        checks = validate(grid)
        cv2.imwrite(str(debug_path), debug, [cv2.IMWRITE_JPEG_QUALITY, 92])

        print(f"{puzzle_id} grid: {grid}")
        print(f"{puzzle_id} validation: {checks}")

        puzzles.append({
            "id": puzzle_id,
            "title": "Sudoku 1" if puzzle_id == "dc-1" else "Sudoku 2",
            "verified": bool(checks["valid"] and checks["unique"]),
            "grid": grid,
            "ocr": ocr_meta,
            "validation": checks,
            "image": f"/data/dc/{args.date}/{puzzle_id}.jpg",
            "crop": str(raw_path),
            "debug": str(debug_path),
        })

    result = {
        "date": args.date,
        "edition": "Hyderabad",
        "source": "Deccan Chronicle",
        "source_image": str(args.source),
        "source_dimensions": {"width": source_w, "height": source_h},
        "normalization": {"enabled": False},
        "ocr_engine": "Tesseract per-cell OCR with OCR-only enlargement",
        "ocr_scale": OCR_SCALE,
        "puzzles": puzzles,
    }
    (root / "ocr_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    canonical = {
        "date": result["date"],
        "edition": result["edition"],
        "source": result["source"],
        "puzzles": [
            {"id": p["id"], "title": p["title"], "verified": p["verified"], "image": p["image"], "grid": p["grid"]}
            for p in puzzles
        ],
    }
    (root / "today.json").write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
    print("PIPELINE STATUS: COMPLETE")


if __name__ == "__main__":
    main()

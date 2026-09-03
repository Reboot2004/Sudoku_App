from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Approximate source-image regions for the two DC Sudoku boards.
# The source image is deliberately kept at its original dimensions; no
# global resize/normalization is performed before cropping.
REGIONS = {
    "dc-1": (40, 75, 341, 339),
    "dc-2": (383, 70, 689, 338),
}

MODEL_LABELS = "0123456789"


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


def extract_digit(cell: np.ndarray) -> np.ndarray | None:
    """Extract the dominant digit contour from one original-resolution cell."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Remove border-touching grid fragments. This only changes the cell passed
    # to the classifier; the original source image remains untouched.
    h, w = threshold.shape
    mask = np.zeros_like(threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            continue
        if area < (w * h) * 0.015:
            continue
        candidates.append((area, contour))

    if not candidates:
        return None

    _, contour = max(candidates, key=lambda item: item[0])
    cv2.drawContours(mask, [contour], -1, 255, -1)
    digit = cv2.bitwise_and(threshold, threshold, mask=mask)
    return digit


def prepare_digit(digit: np.ndarray) -> np.ndarray:
    """Prepare one extracted digit for the MNIST-trained 28x28 CNN."""
    ys, xs = np.where(digit > 0)
    if len(xs) == 0:
        raise ValueError("Empty digit mask")

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    roi = digit[y1:y2, x1:x2]

    side = max(roi.shape)
    canvas = np.zeros((side, side), dtype=np.uint8)
    oy = (side - roi.shape[0]) // 2
    ox = (side - roi.shape[1]) // 2
    canvas[oy:oy + roi.shape[0], ox:ox + roi.shape[1]] = roi
    resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def load_cnn(model_path: Path):
    if not model_path.exists():
        raise RuntimeError(f"CNN model not found: {model_path}")
    model = load_model(str(model_path), compile=False)
    print(f"CNN model loaded: {model_path}")
    print(f"CNN input shape: {model.input_shape}; output shape: {model.output_shape}")
    return model


def ocr_grid(crop: np.ndarray, model) -> tuple[list[list[int]], dict[str, Any]]:
    grid = [[0] * 9 for _ in range(9)]
    confidence = [[0.0] * 9 for _ in range(9)]
    recognized = [[""] * 9 for _ in range(9)]
    occupied = 0

    for r in range(9):
        for c in range(9):
            y1, y2 = round(r * crop.shape[0] / 9), round((r + 1) * crop.shape[0] / 9)
            x1, x2 = round(c * crop.shape[1] / 9), round((c + 1) * crop.shape[1] / 9)
            cell = crop[y1:y2, x1:x2]
            digit = extract_digit(cell)
            if digit is None:
                continue

            sample = prepare_digit(digit)
            prediction = model.predict(sample.reshape(1, 28, 28, 1), verbose=0)[0]
            cls = int(np.argmax(prediction))
            conf = float(prediction[cls])

            # The source model was trained on MNIST classes 0..9. Sudoku has
            # no zero-valued clue, so class 0 is interpreted as empty/noise.
            if cls == 0:
                continue

            grid[r][c] = cls
            confidence[r][c] = round(conf, 4)
            recognized[r][c] = MODEL_LABELS[cls]
            occupied += 1

    return grid, {
        "engine": "Sotejaswini Sudoku-Solver-OCR CNN",
        "model": "model-4.h5",
        "model_source": "https://github.com/Sotejaswini/Sudoku-Solver-OCR",
        "architecture_source": "MNIST-trained Keras CNN",
        "occupied_cells": occupied,
        "confidence": confidence,
        "recognized": recognized,
        "note": "Per-cell resize to 28x28 is required by the MNIST model; the full DC source image is never globally resized.",
    }


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
    parser = argparse.ArgumentParser(description="Headless DC Sudoku OCR using the Sotejaswini CNN digit model.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="dc_test/models/model-4.h5")
    args = parser.parse_args()

    root = Path("dc_test") / args.date / "sudoku"
    root.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source)
    image = load_image(source_path)
    source_h, source_w = image.shape[:2]

    print(f"Source dimensions: {source_w}x{source_h}")
    print("Source normalization: DISABLED")
    print("OCR engine: Sotejaswini CNN (model-4.h5)")

    model = load_cnn(Path(args.model))
    puzzles = []

    for puzzle_id, ref_box in REGIONS.items():
        crop = crop_region(image, ref_box)
        crop_path = root / f"{puzzle_id}.png"
        cv2.imwrite(str(crop_path), crop)
        print(f"{puzzle_id}: crop={crop.shape[1]}x{crop.shape[0]}")

        try:
            grid, ocr_meta = ocr_grid(crop, model)
            checks = validate(grid)
            error = None
        except Exception as exc:
            grid = [[0] * 9 for _ in range(9)]
            checks = validate(grid)
            ocr_meta = {
                "engine": "Sotejaswini CNN",
                "model": "model-4.h5",
                "error": f"{type(exc).__name__}: {exc}",
            }
            error = str(exc)

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
            "error": error,
        })

    result = {
        "date": args.date,
        "edition": "Hyderabad",
        "source": "Deccan Chronicle",
        "source_image": str(source_path),
        "source_dimensions": {"width": source_w, "height": source_h},
        "normalization": {"enabled": False},
        "ocr_engine": "Sotejaswini CNN model-4.h5",
        "model_source": "https://github.com/Sotejaswini/Sudoku-Solver-OCR",
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

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sudoku_ocr import Board

# Approximate source-image regions for the two DC Sudoku boards.
# The source image is deliberately kept at its original dimensions; no
# global resize/normalization is performed before cropping.
REGIONS = {
    "dc-1": (40, 75, 341, 339),
    "dc-2": (383, 70, 689, 338),
}


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


def board_to_grid(value: Any) -> list[list[int]]:
    """Convert sudoku-ocr's board_value into a plain 9x9 integer list."""
    grid = np.asarray(value)
    if grid.shape != (9, 9):
        raise RuntimeError(f"sudoku-ocr returned unexpected board shape: {grid.shape}")
    return [[int(cell) for cell in row] for row in grid.tolist()]


def ocr_grid(crop_path: Path) -> tuple[list[list[int]], dict[str, Any]]:
    """Run the sudoku-ocr PyPI package directly on the original Sudoku crop."""
    board = Board()
    board.prepare_img(str(crop_path))
    board.ocr_sudoku()

    grid = board_to_grid(board.board_value)
    return grid, {
        "engine": "sudoku-ocr",
        "package": "sudoku-ocr",
        "api": ["Board.prepare_img", "Board.ocr_sudoku", "Board.board_value"],
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
    parser = argparse.ArgumentParser(description="Headless DC Sudoku crop/OCR pipeline using sudoku-ocr.")
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
    print("OCR engine: sudoku-ocr (PyPI)")

    puzzles = []
    for puzzle_id, ref_box in REGIONS.items():
        crop = crop_region(image, ref_box)
        crop_path = root / f"{puzzle_id}.png"
        cv2.imwrite(str(crop_path), crop)
        print(f"{puzzle_id}: crop={crop.shape[1]}x{crop.shape[0]}")

        try:
            grid, ocr_meta = ocr_grid(crop_path)
            checks = validate(grid)
            error = None
        except Exception as exc:
            grid = [[0] * 9 for _ in range(9)]
            checks = validate(grid)
            ocr_meta = {
                "engine": "sudoku-ocr",
                "package": "sudoku-ocr",
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
        "ocr_engine": "sudoku-ocr",
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

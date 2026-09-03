from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pytesseract

REFERENCE_W = 732
REFERENCE_H = 606
REGIONS = {
    "dc-1": (40, 75, 341, 339),
    "dc-2": (383, 70, 689, 338),
}
OCR_TIMEOUT = 3


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def normalize_source(image: np.ndarray) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    scale_x = REFERENCE_W / w
    scale_y = REFERENCE_H / h
    normalized = cv2.resize(image, (REFERENCE_W, REFERENCE_H), interpolation=cv2.INTER_AREA)
    return normalized, {
        "source_width": w,
        "source_height": h,
        "baseline_width": REFERENCE_W,
        "baseline_height": REFERENCE_H,
        "scale_x": round(scale_x, 8),
        "scale_y": round(scale_y, 8),
        "normalization": "resize_to_baseline",
    }


def remove_grid_lines(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8
    )
    h, w = inv.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 12), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 12)))
    horizontal = cv2.morphologyEx(inv, cv2.MORPH_OPEN, hk)
    vertical = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vk)
    lines = cv2.bitwise_or(horizontal, vertical)
    clean = cv2.bitwise_and(inv, cv2.bitwise_not(lines))
    return clean


def extract_cell(clean: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = clean.shape[:2]
    y1, y2 = round(r * h / 9), round((r + 1) * h / 9)
    x1, x2 = round(c * w / 9), round((c + 1) * w / 9)
    mx = max(3, round((x2 - x1) * 0.16))
    my = max(3, round((y2 - y1) * 0.16))
    return clean[y1 + my:y2 - my, x1 + mx:x2 - mx]


def has_digit(cell: np.ndarray) -> bool:
    """Reject empty cells and residual grid artifacts before OCR."""
    if cell.size == 0:
        return False
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cell, 8)
    cell_area = cell.shape[0] * cell.shape[1]
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < max(10, cell_area * 0.015):
            continue
        if w <= 2 or h <= 3:
            continue
        if area > cell_area * 0.45:
            continue
        # Digits have a meaningful 2-D bounding box; long thin remnants are lines.
        if w / max(h, 1) < 0.15 or w / max(h, 1) > 2.5:
            continue
        return True
    return False


def ocr_cell(cell: np.ndarray) -> tuple[int, float]:
    if not has_digit(cell):
        return 0, 0.0

    image = cv2.resize(cell, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    config = "--psm 10 -c tessedit_char_whitelist=123456789"
    try:
        data = pytesseract.image_to_data(
            image,
            config=config,
            output_type=pytesseract.Output.DICT,
            timeout=OCR_TIMEOUT,
        )
    except (RuntimeError, pytesseract.TesseractError):
        return 0, 0.0

    best_digit, best_conf = 0, 0.0
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        try:
            score = float(conf)
        except (TypeError, ValueError):
            continue
        if len(text) == 1 and text in "123456789" and score > best_conf:
            best_digit, best_conf = int(text), score
    return best_digit, round(best_conf, 1)


def read_grid(crop: np.ndarray) -> tuple[list[list[int]], list[list[float]]]:
    clean = remove_grid_lines(crop)
    grid = [[0] * 9 for _ in range(9)]
    confidence = [[0.0] * 9 for _ in range(9)]

    tasks = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for r in range(9):
            for c in range(9):
                cell = extract_cell(clean, r, c)
                tasks[pool.submit(ocr_cell, cell)] = (r, c)
        for future in as_completed(tasks):
            r, c = tasks[future]
            try:
                digit, conf = future.result()
            except Exception:
                digit, conf = 0, 0.0
            grid[r][c] = digit
            confidence[r][c] = conf
    return grid, confidence


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


def validate(grid: list[list[int]]) -> dict:
    errors = []
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    root = Path("dc_test") / args.date / "sudoku"
    root.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source)
    image = load_image(source_path)
    normalized, normalization = normalize_source(image)
    normalized_path = root / "sudoku_source_normalized.jpg"
    cv2.imwrite(str(normalized_path), normalized)

    puzzles = []
    for puzzle_id, ref_box in REGIONS.items():
        x1, y1, x2, y2 = ref_box
        crop = normalized[y1:y2, x1:x2]
        crop_path = root / f"{puzzle_id}.png"
        cv2.imwrite(str(crop_path), crop)
        grid, confidence = read_grid(crop)
        checks = validate(grid)
        puzzles.append({
            "id": puzzle_id,
            "title": "Sudoku 1" if puzzle_id == "dc-1" else "Sudoku 2",
            "verified": bool(checks["valid"] and checks["unique"]),
            "grid": grid,
            "ocr_confidence": confidence,
            "validation": checks,
            "crop": str(crop_path),
        })

    result = {
        "date": args.date,
        "edition": "Hyderabad",
        "source": "Deccan Chronicle",
        "source_image": str(source_path),
        "source_dimensions": {"width": normalization["source_width"], "height": normalization["source_height"]},
        "normalization": normalization,
        "normalized_source": str(normalized_path),
        "puzzles": puzzles,
    }
    (root / "ocr_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    canonical = {
        "date": result["date"],
        "edition": result["edition"],
        "source": result["source"],
        "puzzles": [
            {"id": p["id"], "title": p["title"], "verified": bool(p.get("verified", False)), "grid": p["grid"]}
            for p in result["puzzles"]
        ],
    }
    (root / "today.json").write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
    print("PIPELINE STATUS: COMPLETE")
    print(f"Source dimensions: {normalization['source_width']}x{normalization['source_height']}")
    print(f"Normalized to: {REFERENCE_W}x{REFERENCE_H}")
    for p in puzzles:
        print(p["title"], p["validation"])


if __name__ == "__main__":
    main()

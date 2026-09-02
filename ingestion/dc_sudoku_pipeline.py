from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytesseract


# Baseline source layout used by the tested crop coordinates.
REFERENCE_W = 732
REFERENCE_H = 606
REGIONS = {
    "dc-1": (40, 75, 341, 339),
    "dc-2": (383, 70, 689, 338),
}


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def normalize_source(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Normalize the selected source to the tested 732x606 baseline."""
    h, w = image.shape[:2]
    scale_x = REFERENCE_W / w
    scale_y = REFERENCE_H / h
    normalized = cv2.resize(
        image,
        (REFERENCE_W, REFERENCE_H),
        interpolation=cv2.INTER_AREA,
    )
    return normalized, {
        "source_width": w,
        "source_height": h,
        "baseline_width": REFERENCE_W,
        "baseline_height": REFERENCE_H,
        "scale_x": round(scale_x, 8),
        "scale_y": round(scale_y, 8),
        "normalization": "resize_to_baseline",
    }


def preprocess(cell: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=7, fy=7, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def ocr_digit(cell: np.ndarray) -> dict:
    image = preprocess(cell)
    ink_ratio = float((image < 128).mean())
    if ink_ratio < 0.012:
        return {"digit": 0, "confidence": 0.0}

    votes: list[str] = []
    confidences: list[float] = []
    for psm in (10, 8, 13):
        data = pytesseract.image_to_data(
            image,
            config=f"--psm {psm} -c tessedit_char_whitelist=123456789",
            output_type=pytesseract.Output.DICT,
        )
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            try:
                score = float(conf)
            except ValueError:
                score = -1.0
            if text and text[0] in "123456789" and score >= 0:
                votes.append(text[0])
                confidences.append(score)
                break

    if not votes:
        return {"digit": 0, "confidence": 0.0}

    counts = {d: votes.count(d) for d in set(votes)}
    digit = max(counts, key=counts.get)
    selected = [c for d, c in zip(votes, confidences) if d == digit]
    return {"digit": int(digit), "confidence": round(float(np.mean(selected)), 1)}


def read_grid(crop: np.ndarray) -> tuple[list[list[int]], list[list[float]]]:
    h, w = crop.shape[:2]
    grid: list[list[int]] = []
    confidence: list[list[float]] = []
    for r in range(9):
        row, conf_row = [], []
        y1, y2 = round(r * h / 9), round((r + 1) * h / 9)
        for c in range(9):
            x1, x2 = round(c * w / 9), round((c + 1) * w / 9)
            margin_x = max(2, round((x2 - x1) * 0.10))
            margin_y = max(2, round((y2 - y1) * 0.10))
            cell = crop[y1 + margin_y:y2 - margin_y, x1 + margin_x:x2 - margin_x]
            result = ocr_digit(cell)
            row.append(result["digit"])
            conf_row.append(result["confidence"])
        grid.append(row)
        confidence.append(conf_row)
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
        "source_dimensions": {
            "width": normalization["source_width"],
            "height": normalization["source_height"],
        },
        "normalization": normalization,
        "normalized_source": str(normalized_path),
        "puzzles": puzzles,
    }
    (root / "ocr_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Verification remains metadata only. The Action publishes the OCR result
    # regardless of the verification flag.
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
    print(f"Source dimensions: {normalization['source_width']}x{normalization['source_height']}")
    print(f"Normalized to: {REFERENCE_W}x{REFERENCE_H}")
    for p in puzzles:
        print(p["title"], p["validation"])


if __name__ == "__main__":
    main()

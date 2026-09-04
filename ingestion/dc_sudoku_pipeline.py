from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract import TesseractNotFoundError
except Exception:  # pytesseract missing -> OCR degrades to empty, pipeline still runs
    pytesseract = None
    class TesseractNotFoundError(Exception):  # type: ignore
        pass

# ---------------------------------------------------------------------------
# Detection owns discovery of sudoku_source.jpg. OCR only consumes that image.
# BOARD_BOXES is kept as an ROI hint only; per-crop grid lines are detected
# dynamically (the old fixed GRID_LINES were off by ~5px on X and varied by
# date, leaking grid borders into cells -> Tesseract read lines as "1").
# Detection / download code (dc_sudoku_detect.py) is intentionally untouched.
# ---------------------------------------------------------------------------
CANONICAL_SIZE = (732, 606)  # width, height
BOARD_BOXES = {
    "dc-1": (40, 74, 342, 340),
    "dc-2": (383, 72, 691, 340),
}
DIGITS = "123456789"

# OCR tuning
OCR_UPSCALE = 3          # working upscale for a single cell (was 4x, amplified JPEG noise)
LINE_INSET_FRAC = 0.14   # inset from detected line centres (excludes 2-3px borders)
EMPTY_INK_FRAC = 0.055   # below this (after cleaning) -> empty without calling Tesseract
MIN_CONF = 30.0          # minimum Tesseract confidence to accept a vote
SINGLE_CONF = 40.0       # single (unconfirmed) vote accepted at/above this;
                         # CI 2026-09-04: true 9s often yield exactly one vote


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


# ---------------------------------------------------------------------------
# Dynamic grid-line detection (per crop, replaces fixed GRID_LINES)
# ---------------------------------------------------------------------------
def _pick_lines(proj: np.ndarray, n: int = 10) -> list[int]:
    L = len(proj)
    min_dist = max(4, L // 13)
    k = max(3, L // 200)
    if k % 2 == 0:
        k += 1
    sm = cv2.GaussianBlur(proj.reshape(-1, 1).astype(np.float32), (1, k), 0).ravel()
    order = np.argsort(sm)[::-1]
    picked: list[int] = []
    taken = np.zeros(L, dtype=bool)
    for i in order:
        i = int(i)
        if taken[i]:
            continue
        picked.append(i)
        taken[max(0, i - min_dist):min(L, i + min_dist)] = True
        if len(picked) >= n:
            break
    return sorted(picked)


def _lines_ok(xs: list[float], ys: list[float], w: int, h: int) -> bool:
    if len(xs) != 10 or len(ys) != 10:
        return False
    # monotonic, inside image, roughly uniform spacing (last gap may be short at edge)
    if not all(b > a for a, b in zip(xs, xs[1:])) or not all(b > a for a, b in zip(ys, ys[1:])):
        return False
    if xs[0] < -4 or ys[0] < -4 or xs[-1] > w + 4 or ys[-1] > h + 4:
        return False
    dx = np.diff(np.array(xs, dtype=float))
    dy = np.diff(np.array(ys, dtype=float))
    # median cell size sanity: expect 20..60px at native crop resolution
    if not (15 < float(np.median(dx)) < 80 and 15 < float(np.median(dy)) < 80):
        return False
    # at least 7 of 9 gaps within 25% of median
    medx, medy = float(np.median(dx)), float(np.median(dy))
    if int((np.abs(dx - medx) < 0.30 * medx).sum()) < 7:
        return False
    if int((np.abs(dy - medy) < 0.30 * medy).sum()) < 7:
        return False
    return True


def _snap_edge(lines: list[float], edge: float) -> list[float]:
    """Snap a trailing/leading grid line to the image edge when it is close to
    the edge but leaves an anomalously short end cell (CI 2026-09-04 dc-1:
    right border found 4px inside, last column squeezed 25.5px vs 33 median,
    clipping R4C9's 9). Conservative: only fires within half a median cell of
    the edge with a <85%-of-median end gap."""
    out = list(lines)
    if len(out) != 10:
        return out
    gaps = [b - a for a, b in zip(out, out[1:])]
    med = float(np.median(gaps))
    if med <= 0:
        return out
    if out[-1] < edge - 1 and (edge - out[-1]) < 0.5 * med and gaps[-1] < 0.85 * med:
        out[-1] = edge
    if out[0] > 1 and out[0] < 0.5 * med and gaps[0] < 0.85 * med:
        out[0] = 0.0
    return out


def detect_grid_lines(crop_bgr: np.ndarray) -> tuple[list[float], list[float], str]:
    """Return (x_lines, y_lines) in crop pixels + method name."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 2
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(big, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    bh, bw = th.shape
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, bh // 9)))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, bw // 9), 1))
    vert = cv2.morphologyEx(th, cv2.MORPH_OPEN, vk)
    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, hk)
    vproj = vert.sum(axis=0).astype(np.float64)
    hproj = horiz.sum(axis=1).astype(np.float64)
    vx = _pick_lines(vproj)
    hy = _pick_lines(hproj)
    xs = [v / scale for v in vx]
    ys = [v / scale for v in hy]
    if _lines_ok(xs, ys, w, h):
        # clamp into image
        xs = [min(max(0.0, x), float(w - 1)) for x in xs]
        ys = [min(max(0.0, y), float(h - 1)) for y in ys]
        xs = _snap_edge(xs, float(w - 1))
        ys = _snap_edge(ys, float(h - 1))
        return xs, ys, "morphology"
    # fallback: uniform split (never crash the daily job on detection)
    xs = [i * (w - 1) / 9.0 for i in range(10)]
    ys = [i * (h - 1) / 9.0 for i in range(10)]
    return xs, ys, "uniform-fallback"


def extract_cell(gray: np.ndarray, xs: list[float], ys: list[float],
                 r: int, c: int, inset_frac: float = LINE_INSET_FRAC) -> np.ndarray:
    x1, x2 = xs[c], xs[c + 1]
    y1, y2 = ys[r], ys[r + 1]
    cw, ch = x2 - x1, y2 - y1
    ix1 = int(round(x1 + cw * inset_frac))
    iy1 = int(round(y1 + ch * inset_frac))
    ix2 = int(round(x2 - cw * inset_frac))
    iy2 = int(round(y2 - ch * inset_frac))
    ix1, iy1 = max(0, ix1), max(0, iy1)
    ix2, iy2 = min(gray.shape[1], ix2), min(gray.shape[0], iy2)
    if ix2 <= ix1 + 4 or iy2 <= iy1 + 4:  # degenerate -> fall back to centre crop
        cx1, cy1 = int(x1), int(y1)
        return gray[cy1:cy1 + 8, cx1:cx1 + 8]
    return gray[iy1:iy2, ix1:ix2]


def clean_cell(cell: np.ndarray) -> tuple[np.ndarray, float]:
    """Denoise + adaptive threshold (inverted: ink=255). Returns (binary, ink_frac)."""
    if cell.size == 0:
        return np.zeros((8, 8), dtype=np.uint8), 0.0
    blur = cv2.GaussianBlur(cell, (3, 3), 0)
    # block size relative to cell, must be odd >= 11
    bs = max(11, (min(cell.shape) // 2) * 2 + 1)
    if bs % 2 == 0:
        bs += 1
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, bs, 7)
    th = cv2.medianBlur(th, 3)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # ignore 1px rim (residual border) for the ink estimate
    inner = th[1:-1, 1:-1] if min(th.shape) > 4 else th
    ink = float((inner > 0).mean()) if inner.size else 0.0
    return th, ink


def largest_digit_blob(binary: np.ndarray) -> tuple[float, tuple[int, int, int, int] | None]:
    """Area fraction + bbox of the largest plausible digit contour."""
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, None
    h, w = binary.shape
    cell_area = float(h * w)
    best = None
    best_area = 0.0
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < 0.003 * cell_area:      # speckle
            continue
        if area > 0.45 * cell_area:       # smudge / border remnant
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 0.12 * w or bh < 0.25 * h:  # too thin to be a digit
            continue
        if area > best_area:
            best_area, best = area, (x, y, bw, bh)
    if best is None:
        return 0.0, None
    return best_area / cell_area, best


def ocr_cell(cell_gray: np.ndarray) -> tuple[int, float, list[str], float, str]:
    """Returns (digit, conf01, votes, ink_frac, decision). digit=0 means empty."""
    binary, ink = clean_cell(cell_gray)
    if ink < EMPTY_INK_FRAC:
        return 0, 0.0, [], ink, "empty-ink"
    area_frac, bbox = largest_digit_blob(binary)
    if bbox is None:
        return 0, 0.0, [], ink, "empty-nocontour"
    if pytesseract is None:
        return 0, 0.0, [], ink, "no-tesseract"
    # isolate the digit blob, pad square, upscale for Tesseract
    x, y, bw, bh = bbox
    pad = 3
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(binary.shape[1], x + bw + pad), min(binary.shape[0], y + bh + pad)
    digit_img = binary[y1:y2, x1:x2]
    digit_inv = 255 - digit_img  # Tesseract wants black text on white
    side = max(digit_img.shape[0], digit_img.shape[1])
    canvas = np.full((side + 8, side + 8), 255, dtype=np.uint8)
    canvas[4:4 + digit_inv.shape[0], 4:4 + digit_inv.shape[1]] = digit_inv
    target = 96
    scale = target / max(1, max(canvas.shape))
    ocr_img = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ocr_img = cv2.copyMakeBorder(ocr_img, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    # second variant: plain Otsu-upscaled cell (helps thin newsprint digits)
    gray_big = cv2.resize(cell_gray, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE,
                          interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(gray_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.copyMakeBorder(otsu, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    votes: list[str] = []
    confs: list[float] = []
    try:
        for image, psms in ((ocr_img, (10, 8)), (otsu, (10,))):
            for psm in psms:
                try:
                    data = pytesseract.image_to_data(
                        image,
                        config=f"--oem 1 --psm {psm} -c tessedit_char_whitelist={DIGITS}",
                        output_type=pytesseract.Output.DICT,
                    )
                except TesseractNotFoundError:
                    return 0, 0.0, [], ink, "no-tesseract-binary"
                except Exception:
                    continue
                for text, conf in zip(data.get("text", []), data.get("conf", [])):
                    t = "".join(ch for ch in str(text) if ch in DIGITS)
                    try:
                        cf = float(conf)
                    except (ValueError, TypeError):
                        continue
                    if len(t) == 1 and cf >= MIN_CONF:
                        votes.append(t)
                        confs.append(cf)
    except Exception:
        pass
    if not votes:
        return 0, 0.0, [], ink, "empty-lowconf"
    counts: dict[str, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts, key=lambda k: (counts[k], np.mean([c for v, c in zip(votes, confs) if v == k])))
    best_confs = [c for v, c in zip(votes, confs) if v == best]
    # require agreement: >=2 votes, a single vote at/above SINGLE_CONF, or a
    # single vote backed by a strong digit-sized blob (area>=5% of the cell).
    # Cells reaching this point already passed ink + blob gates (CI empties
    # measure ink 0.000), so specks cannot sneak in as singles; true 9s often
    # yield exactly one Tesseract vote.
    n_best = len([v for v in votes if v == best])
    if n_best < 2 and max(best_confs) < SINGLE_CONF and area_frac < 0.05:
        return 0, 0.0, votes, ink, "empty-noagreement"
    conf01 = round(float(np.mean(best_confs)) / 100.0, 4)
    return int(best), conf01, votes, ink, "ocr"


def ocr_grid(crop_bgr: np.ndarray) -> tuple[list[list[int]], dict[str, Any], np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    xs, ys, method = detect_grid_lines(crop_bgr)
    grid = [[0] * 9 for _ in range(9)]
    conf = [[0.0] * 9 for _ in range(9)]
    cells_meta: list[dict[str, Any]] = []
    occupied = 0
    debug = crop_bgr.copy()
    # montage of what was actually sent to OCR (white bg)
    thumbs: list[np.ndarray] = []
    for r in range(9):
        for c in range(9):
            cell = extract_cell(gray, xs, ys, r, c)
            digit, cf, votes, ink, decision = ocr_cell(cell)
            grid[r][c] = digit
            conf[r][c] = cf
            if digit:
                occupied += 1
            cells_meta.append({"row": r + 1, "column": c + 1, "digit": digit,
                               "confidence": cf, "votes": votes,
                               "ink_frac": round(float(ink), 4), "decision": decision})
            small = cv2.resize(cell, (48, 48), interpolation=cv2.INTER_AREA)
            thumbs.append(small)
    for x in [int(round(v)) for v in xs]:
        cv2.line(debug, (x, 0), (x, debug.shape[0] - 1), (0, 0, 255), 1)
    for y in [int(round(v)) for v in ys]:
        cv2.line(debug, (0, y), (debug.shape[1] - 1, y), (0, 0, 255), 1)
    montage = np.vstack([np.hstack(thumbs[i * 9:(i + 1) * 9]) for i in range(9)])
    meta = {"engine": "Tesseract per-cell OCR (dynamic grid lines + empty gating)",
            "line_method": method,
            "x_lines": [round(float(v), 2) for v in xs],
            "y_lines": [round(float(v), 2) for v in ys],
            "occupied_cells": occupied,
            "confidence": conf, "cells": cells_meta}
    return grid, meta, debug, montage


# ---------------------------------------------------------------------------
# Sudoku validation + confidence-guided repair (never publish line-ghosts)
# ---------------------------------------------------------------------------
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
        for n in options or []:
            board[r][c] = n
            search(found)
            board[r][c] = 0
            if found[0] >= limit:
                return

    found = [0]
    search(found)
    return found[0]


def find_conflicts(grid: list[list[int]]) -> set[tuple[int, int]]:
    bad: set[tuple[int, int]] = set()
    for r in range(9):
        seen: dict[int, list[int]] = {}
        for c in range(9):
            v = grid[r][c]
            if v:
                seen.setdefault(v, []).append(c)
        for v, cols in seen.items():
            if len(cols) > 1:
                bad.update((r, c) for c in cols)
    for c in range(9):
        seen = {}
        for r in range(9):
            v = grid[r][c]
            if v:
                seen.setdefault(v, []).append(r)
        for v, rows in seen.items():
            if len(rows) > 1:
                bad.update((r, c) for r in rows)
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            seen = {}
            for r in range(br, br + 3):
                for c in range(bc, bc + 3):
                    v = grid[r][c]
                    if v:
                        seen.setdefault(v, []).append((r, c))
            for v, cells in seen.items():
                if len(cells) > 1:
                    bad.update(cells)
    return bad


def validate(grid: list[list[int]]) -> dict[str, Any]:
    errors: list[str] = []
    bad = find_conflicts(grid)
    if bad:
        errors.append(f"{len(bad)} conflicting cells")
    count = solution_count(grid)
    return {"valid": not bad and count > 0, "unique": count == 1,
            "solution_count": count, "errors": errors}


def repair_grid(grid: list[list[int]], conf: list[list[float]]) -> tuple[list[list[int]], list[str]]:
    """Drop lowest-confidence conflicting/unsolvable digits so the published
    grid is at worst incomplete, never contradictory. Returns (grid, notes)."""
    notes: list[str] = []
    grid = [row[:] for row in grid]
    for _ in range(12):
        bad = find_conflicts(grid)
        if not bad:
            break
        r, c = min(bad, key=lambda rc: conf[rc[0]][rc[1]])
        notes.append(f"cleared conflict R{r + 1}C{c + 1}={grid[r][c]} conf={conf[r][c]}")
        grid[r][c] = 0
        conf[r][c] = 0.0
    for _ in range(10):
        if solution_count(grid) > 0 or not any(any(row) for row in grid):
            break
        # unsolvable but conflict-free -> drop globally weakest digit
        filled = [(conf[r][c], r, c) for r in range(9) for c in range(9) if grid[r][c]]
        if not filled:
            break
        _, r, c = min(filled)
        notes.append(f"cleared unsolvable R{r + 1}C{c + 1}={grid[r][c]} conf={conf[r][c]}")
        grid[r][c] = 0
        conf[r][c] = 0.0
    return grid, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="DC Sudoku OCR: dynamic grid lines + empty gating + repair.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    root = Path("dc_test") / args.date / "sudoku"
    root.mkdir(parents=True, exist_ok=True)
    image = load_image(Path(args.source))
    source_h, source_w = image.shape[:2]

    print(f"Source dimensions: {source_w}x{source_h}")
    print("Detection stage: UNCHANGED (sudoku_source.jpg from dc_sudoku_detect.py)")
    print("OCR: dynamic per-crop grid lines + adaptive empty gating + Tesseract + repair")

    puzzles = []
    for puzzle_id, box in BOARD_BOXES.items():
        crop = crop_region(image, scaled_box(box, source_w, source_h))
        image_path = root / f"{puzzle_id}.jpg"
        debug_path = root / f"{puzzle_id}-ocr-debug.jpg"
        montage_path = root / f"{puzzle_id}-cells.jpg"
        cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        grid, ocr_meta, debug, montage = ocr_grid(crop)
        conf = ocr_meta["confidence"]
        grid, repair_notes = repair_grid(grid, conf)
        ocr_meta["repair"] = repair_notes
        checks = validate(grid)
        cv2.imwrite(str(debug_path), debug, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(montage_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 90])

        clues = sum(1 for row in grid for v in row if v)
        print(f"{puzzle_id} lines({ocr_meta['line_method']}): x={ocr_meta['x_lines']}")
        print(f"{puzzle_id} grid: {grid}")
        print(f"{puzzle_id} clues={clues} occupied_raw={ocr_meta['occupied_cells']} "
              f"repair={repair_notes or 'none'} validation={checks}")

        puzzles.append({
            "id": puzzle_id,
            "title": "Sudoku 1" if puzzle_id == "dc-1" else "Sudoku 2",
            "verified": bool(checks["valid"] and checks["unique"] and 17 <= clues <= 40),
            "grid": grid,
            "ocr": ocr_meta,
            "validation": checks,
            "image": f"/data/dc/{args.date}/{puzzle_id}.jpg",
            "debug": str(debug_path),
            "cells_montage": str(montage_path),
        })

    result = {
        "date": args.date,
        "edition": "Hyderabad",
        "source": "Deccan Chronicle",
        "source_image": str(args.source),
        "source_dimensions": {"width": source_w, "height": source_h},
        "normalization": {"enabled": False},
        "ocr_engine": "Tesseract per-cell OCR (dynamic lines + empty gating + repair)",
        "puzzles": puzzles,
    }
    (root / "ocr_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    canonical = {
        "date": result["date"],
        "edition": result["edition"],
        "source": result["source"],
        "puzzles": [
            {"id": p["id"], "title": p["title"], "verified": p["verified"],
             "image": p["image"], "grid": p["grid"]}
            for p in puzzles
        ],
    }
    (root / "today.json").write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
    print("PIPELINE STATUS: COMPLETE")


if __name__ == "__main__":
    main()

# DC Sudoku ingestion

Current tested stage: direct-image discovery and Sudoku-source detection.

## Structure

    ingestion/
    ├── dc_sudoku_detect.py
    └── templates/
        ├── sudoku_reference.jpg
        ├── heading_sudoku1.png
        └── heading_sudoku2.png

## What it does

For a supplied date, the detector enumerates the Deccan Chronicle Hyderabad
`tabpX_Y.jpg` image space, downloads valid image assets, and ranks them using
the known Sudoku reference.

The current scoring uses:
- dimension similarity as the strongest heuristic
- Sudoku 1 heading visual matching
- Sudoku 2 heading visual matching
- overall layout similarity

The article link is not used.

## Run locally

    pip install requests opencv-python numpy

    python dc_sudoku_detect.py --date 2026-08-31

For development, retain all downloaded candidates:

    python dc_sudoku_detect.py --date 2026-08-31 --keep-all

## Next stage

The next production steps are separate and should be added only after testing:
1. precise crop of Sudoku 1 and Sudoku 2
2. digit recognition
3. Sudoku validation / uniqueness check
4. generation of the canonical daily `today.json`

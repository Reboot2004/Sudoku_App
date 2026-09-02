# DC Sudoku Mobile v0.4

Two source modes:
- **My Sudoku**: enter TOI/printed puzzles that are not online.
- **Deccan Chronicle**: daily Hyderabad puzzles from the automated image ingestion pipeline.

## Personal puzzle flow
1. Add the already-present printed numbers.
2. Confirm.
3. The app checks conflicts and solvability.
4. Solvable -> silently enters the solve state.
5. Invalid/unsolvable -> asks for recheck.

## Solve behavior
- Clear Cell is always cell-only.
- Solve obeys the selected Cell/Row/Column/Box/Entire Grid scope.
- Givens are dark navy; later entries are blue.
- Candidates live in fixed 3x3 cell slots.
- No Save button.

## Run
Frontend:
    cd frontend
    npm install
    npm run dev

Backend:
    cd backend
    python -m venv .venv
    .venv\\Scripts\\activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

## APK without Android Studio
Use `scripts\\build-apk.bat` with Node.js, JDK, Android SDK/build-tools and Gradle available. Capacitor creates the Android project and Gradle produces the debug APK.

## GitHub Actions
`daily-dc-ingestion.yml` is Monday-Saturday only. It currently executes the proven DC direct-image detector and uploads its scan. OCR/canonical `today.json` publishing comes next.

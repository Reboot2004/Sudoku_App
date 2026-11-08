# Sudoku Solver — DC Sudoku Mobile v0.4

Mobile-first Sudoku app with daily newspaper puzzles and a personal puzzle solver. Two source modes:

- **My Sudoku**: enter TOI/printed puzzles that are not online.
- **Deccan Chronicle**: daily Hyderabad puzzles from the automated image ingestion pipeline.

## Features

- Personal puzzle flow: add printed givens -> confirm -> app checks conflicts and solvability -> solvable enters solve state, invalid asks for recheck.
- Solve behavior: Clear Cell is always cell-only. Solve obeys selected Cell / Row / Column / Box / Entire Grid scope.
- Givens dark navy, later entries blue. Candidates live in fixed 3x3 cell slots. No Save button.
- Deccan Chronicle daily ingestion: image detection + OCR + canonical `today.json` with publish gate refusing unverified output.
- OTA frontend updates: push to `main` -> `deploy-web.yml` publishes `frontend/dist` to `gh-pages`, installed APKs fetch new `app-manifest.json` offline-fallback to bundled copy.

## Stack

Frontend: TypeScript + Vite + Capacitor (Android APK)
Backend: FastAPI + Python (ingestion/OCR)
Infra: GitHub Actions (`build-mobile.yml`, `deploy-web.yml`, `daily-dc-ingestion.yml`)

## Run

Frontend:
```bash
cd frontend
npm install
npm run dev
```

Backend:
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## APK without Android Studio

Use `scripts\build-apk.bat` with Node.js, JDK, Android SDK/build-tools and Gradle available. Capacitor creates the Android project and Gradle produces the debug APK.

## Over-the-air frontend updates (no repeated sideloads)

- Push a frontend change to `main` -> `deploy-web.yml` publishes `frontend/dist` to the `gh-pages` branch.
- On next launch, installed apps fetch the new `app-manifest.json` and load updated modules (offline falls back to bundled copy).
- One-time repo setup: Settings -> Pages -> Deploy from branch -> `gh-pages`.
- Sideload just once to get the loader; afterwards UI fixes ship via `git push`.
- The `vX.Y.Z · local/remote` badge (bottom-right) opens an Updates menu: check now, pin offline, reload.

## GitHub Actions

`daily-dc-ingestion.yml` is Monday-Saturday only. It runs detection + OCR + canonical `today.json` publishing, with a publish gate that refuses unverified output.

## Links

- GitHub: https://github.com/Reboot2004/Sudoku_App

## Portfolio note

Small but complete product: ingestion automation + mobile app + CI/CD + OTA distribution. Demo-ready for portfolio.

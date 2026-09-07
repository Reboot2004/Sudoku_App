# Architecture
Mobile solver/candidates run locally. Backend supplies daily DC data and provides optional personal-puzzle API persistence.

DC ingestion uses direct image assets (`tabpX_Y`) and the proven reference/dimension/template detector; article links are not required.
The daily GitHub workflow runs Monday-Saturday at 08:00 IST and skips Sundays.

## OTA frontend updates (no reinstall)
Like the puzzle data, the frontend code is now dynamic too:

- `frontend/src/loader.js` is a tiny stable bootstrap baked into the APK. On launch it fetches
  `app-manifest.json` from the `gh-pages` branch (GitHub Pages, then jsDelivr, then raw.githubusercontent)
  and `import()`s the newest app bundle (`src/main.js` build output). Everything still executes in the
  local origin, so `localStorage` saves carry over between bundled and remote code.
- Remote wins only when it is newer (higher `version`, or same version with a later `builtAt`).
  Offline, pinned (`?app=local`, or the Updates badge menu), or failed checks fall back to the bundled code.
- `frontend/src/update-policy.js` holds the pure version/channel decision logic (unit-testable, no DOM).
- `frontend/scripts/gen-manifest.mjs` runs as part of `npm run build` and writes `dist/app-manifest.json`
  plus fallback `<meta name="ota-app-js/css">` tags into `dist/index.html`.
- `.github/workflows/deploy-web.yml` publishes `frontend/dist` to the `gh-pages` branch on every
  frontend change, so a plain `git push` updates all installed apps on next launch.
- Remote hosts are listed in `frontend/public/app-config.json` (forks: edit the `repo` there) and
  mirrored in `frontend/capacitor.config.ts` `allowNavigation`.

One-time setup: repository Settings -> Pages -> Deploy from branch -> `gh-pages`. The jsDelivr/raw
fallbacks work even before Pages is enabled. After installing one APK that contains the loader,
further frontend changes need no sideload.

Note: over-the-air code updates are fine for sideloaded/GitHub-release APKs, but Google Play policy
forbids downloading executable code into Play-distributed apps. If this ever ships on Play, keep the
loader pinned to `local` (or remove the remote channel) and update via Play releases instead.

// Reads Vite's build manifest and publishes dist/app-manifest.json for the
// OTA loader, plus fallback <meta> tags in dist/index.html.
// Run via `npm run build` (cwd = frontend/).
import { readFileSync, writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const frontend = resolve(here, '..');
const dist = resolve(frontend, 'dist');

const pkg = JSON.parse(readFileSync(resolve(frontend, 'package.json'), 'utf8'));
const viteManifest = JSON.parse(readFileSync(resolve(dist, '.vite', 'manifest.json'), 'utf8'));

const appEntry = viteManifest['src/main.js'];
if (!appEntry || !appEntry.file) {
  throw new Error('src/main.js missing from Vite manifest; check rollupOptions.input');
}

let sha = 'local';
try {
  sha = execSync('git rev-parse --short HEAD', { cwd: resolve(frontend, '..') })
    .toString()
    .trim();
} catch {
  /* not a git checkout */
}

const manifest = {
  schema: 1,
  name: pkg.name,
  version: pkg.version,
  build: sha,
  builtAt: new Date().toISOString(),
  js: appEntry.file,
  css: appEntry.css && appEntry.css[0] ? appEntry.css[0] : null,
};

writeFileSync(resolve(dist, 'app-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');

// Bundled fallback: lets the loader find the local app bundle even if
// app-manifest.json is ever missing from dist/.
const indexPath = resolve(dist, 'index.html');
let html = readFileSync(indexPath, 'utf8');
const meta =
  `<meta name="ota-app-js" content="./${manifest.js}">` +
  (manifest.css ? `\n  <meta name="ota-app-css" content="./${manifest.css}">` : '');
if (!html.includes('<!-- OTA-APP-META -->')) {
  throw new Error('dist/index.html missing <!-- OTA-APP-META --> placeholder');
}
html = html.replace('<!-- OTA-APP-META -->', meta);
writeFileSync(indexPath, html);

console.log(`[gen-manifest] ${manifest.name} v${manifest.version} (${manifest.build})`);
console.log(`[gen-manifest] js=${manifest.js} css=${manifest.css || '(none)'}`);

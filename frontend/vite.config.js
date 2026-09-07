import { defineConfig } from 'vite';
import { resolve, dirname } from 'path';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';

const here = dirname(fileURLToPath(import.meta.url));
const pkg = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8'));

export default defineConfig({
  // Relative asset URLs: the same dist/ runs from Capacitor (http://localhost),
  // from a plain file host, and from a GitHub Pages project sub-path.
  base: './',
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    // Emits .vite/manifest.json so scripts/gen-manifest.mjs can publish the
    // exact hashed app bundle name for the OTA loader.
    manifest: true,
    rollupOptions: {
      // index -> stable loader shell (src/loader.js); app -> dynamically
      // loaded frontend modules (src/main.js). The loader stays small and
      // rarely changes; the app chunk updates over the air.
      input: {
        index: resolve(here, 'index.html'),
        app: resolve(here, 'src/main.js'),
      },
    },
  },
});

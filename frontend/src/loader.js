// Stable OTA bootstrap. This file must change rarely: installed APKs keep
// their bundled copy, and it loads newer app modules (src/main.js builds)
// from the web when available. Same-origin execution, so localStorage saves
// carry over between bundled and remote code.
import { pickChannel, resolveAsset, isValidManifest, isNewer } from './update-policy.js';

const TIMEOUT_MS = 6000;
const PIN_KEY = 'dcsudoku.appPin'; // 'local' | 'remote' | null
const META_KEY = 'dcsudoku.appMeta'; // last running { version, channel }
const LOCAL_VERSION = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : 'dev';

const DEFAULT_CONFIG = {
  repo: 'Reboot2004/Sudoku_App',
  manifestPath: 'app-manifest.json',
  deployBranch: 'gh-pages',
  remoteBases: [],
};

function withTimeout(ms) {
  const c = new AbortController();
  const t = setTimeout(() => c.abort(), ms);
  return { signal: c.signal, done: () => clearTimeout(t) };
}

async function fetchJson(url) {
  const { signal, done } = withTimeout(TIMEOUT_MS);
  try {
    const r = await fetch(url + (url.includes('?') ? '&' : '?') + '_=' + Date.now(), {
      cache: 'no-store',
      signal,
    });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  } finally {
    done();
  }
}

function defaultRemoteBases(cfg) {
  if (Array.isArray(cfg.remoteBases) && cfg.remoteBases.length) return cfg.remoteBases;
  const [owner, repo] = String(cfg.repo || '').split('/');
  if (!owner || !repo) return [];
  const branch = cfg.deployBranch || 'gh-pages';
  return [
    `https://${owner.toLowerCase()}.github.io/${repo}`,
    `https://cdn.jsdelivr.net/gh/${cfg.repo}@${branch}`,
    `https://raw.githubusercontent.com/${cfg.repo}/${branch}`,
  ];
}

function splash(msg) {
  const el = document.querySelector('#app');
  if (el && !el.dataset.booted) {
    el.innerHTML =
      `<div style="min-height:60vh;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;` +
      `font-family:system-ui,sans-serif;color:#143a52">` +
      `<div style="font-size:13px;font-weight:800;letter-spacing:.08em">DC SUDOKU</div>` +
      `<div style="font-size:14px;color:#5d707d">${msg}</div></div>`;
  }
}

function loadCss(href) {
  return new Promise((resolve) => {
    if (!href) return resolve();
    if (document.querySelector(`link[data-ota][href="${href}"]`)) return resolve();
    const l = document.createElement('link');
    l.rel = 'stylesheet';
    l.href = href;
    l.dataset.ota = '1';
    l.onload = () => resolve();
    l.onerror = () => resolve(); // never block boot on CSS
    document.head.appendChild(l);
  });
}

function badge(version, channel, onAction) {
  let el = document.querySelector('#ota-badge');
  if (!el) {
    el = document.createElement('div');
    el.id = 'ota-badge';
    el.style.cssText =
      'position:fixed;right:8px;bottom:8px;z-index:9999;display:flex;gap:6px;align-items:center;' +
      'font-family:system-ui,sans-serif;font-size:11px;font-weight:700;color:#fff;' +
      'background:rgba(20,58,82,.82);border-radius:999px;padding:5px 6px 5px 10px;backdrop-filter:blur(4px);';
    document.body.appendChild(el);
  }
  el.innerHTML = '';
  const label = document.createElement('span');
  label.textContent = `v${version} · ${channel}`;
  el.appendChild(label);
  const btn = document.createElement('button');
  btn.textContent = 'Updates';
  btn.style.cssText =
    'border:0;border-radius:999px;background:#f6c453;color:#143a52;font-weight:800;font-size:11px;padding:4px 9px;cursor:pointer;';
  btn.addEventListener('click', onAction);
  el.appendChild(btn);
}

function toast(msg) {
  const old = document.querySelector('#ota-toast');
  if (old) old.remove();
  const t = document.createElement('div');
  t.id = 'ota-toast';
  t.textContent = msg;
  t.style.cssText =
    'position:fixed;left:50%;bottom:44px;transform:translateX(-50%);z-index:10001;' +
    'background:#143a52;color:#fff;font-family:system-ui,sans-serif;font-size:13px;font-weight:700;' +
    'border-radius:999px;padding:9px 16px;box-shadow:0 8px 24px rgba(0,0,0,.25);';
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2200);
}

function updatesMenu(current, actions) {
  const old = document.querySelector('#ota-menu');
  if (old) old.remove();
  const m = document.createElement('div');
  m.id = 'ota-menu';
  m.style.cssText =
    'position:fixed;right:8px;bottom:44px;z-index:10000;background:#fff;color:#14232d;border:1px solid #c9d6dd;' +
    'border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.2);padding:8px;min-width:210px;' +
    'font-family:system-ui,sans-serif;font-size:13px;';
  const pin = current.pin || 'auto';
  const head = document.createElement('div');
  head.style.cssText = 'font-weight:800;margin:2px 4px 8px';
  head.textContent = `App updates · ${current.version} (${current.channel})`;
  m.appendChild(head);
  const sub = document.createElement('div');
  sub.style.cssText = 'font-size:11px;color:#5d707d;margin:0 4px 8px';
  sub.textContent = `Channel: ${pin}. New builds publish here automatically; no reinstall needed.`;
  m.appendChild(sub);
  const mk = (text, fn) => {
    const b = document.createElement('button');
    b.textContent = text;
    b.style.cssText =
      'display:block;width:100%;text-align:left;border:1px solid #c8d5dc;background:#fff;border-radius:8px;' +
      'padding:8px;margin-top:6px;font-weight:700;cursor:pointer;';
    b.addEventListener('click', () => {
      m.remove();
      fn();
    });
    m.appendChild(b);
  };
  mk('Check for update now', actions.check);
  mk(pin === 'local' ? 'Follow remote updates (auto)' : 'Pin to bundled version (offline)', actions.togglePin);
  mk('Reload app', actions.reload);
  const close = document.createElement('button');
  close.textContent = 'Close';
  close.style.cssText =
    'display:block;width:100%;border:0;background:transparent;color:#5d707d;padding:8px;margin-top:4px;cursor:pointer;';
  close.addEventListener('click', () => m.remove());
  m.appendChild(close);
  document.body.appendChild(m);
}

async function boot() {
  splash('Loading…');
  // Dev server: run sources directly for HMR.
  if (import.meta.env && import.meta.env.DEV) {
    await import('/src/main.js');
    return;
  }

  const params = new URLSearchParams(location.search);
  const queryForce = params.get('app') === 'local' || params.get('app') === 'remote' ? params.get('app') : null;
  let pin = null;
  try {
    pin = localStorage.getItem(PIN_KEY);
    if (pin !== 'local' && pin !== 'remote') pin = null;
  } catch {
    pin = null;
  }
  const force = queryForce || pin;

  let cfg = { ...DEFAULT_CONFIG };
  try {
    const remote_cfg = await fetchJson('./app-config.json');
    if (remote_cfg && typeof remote_cfg === 'object') cfg = { ...DEFAULT_CONFIG, ...remote_cfg };
  } catch {
    /* keep defaults */
  }

  const manifestPath = cfg.manifestPath || 'app-manifest.json';
  const localUrl = new URL('./' + manifestPath, location.href).toString();
  const local = await fetchJson(localUrl);
  const localManifest = isValidManifest(local)
    ? local
    : { schema: 1, name: 'dc-sudoku-mobile', version: LOCAL_VERSION, js: null, css: null };

  const remotes = [];
  if (force !== 'local') {
    const bases = defaultRemoteBases(cfg);
    const results = await Promise.all(
      bases.map(async (base) => {
        const url = base.replace(/\/$/, '') + '/' + manifestPath;
        const manifest = await fetchJson(url);
        return manifest ? { manifest, base, url } : null;
      }),
    );
    for (const r of results) if (r) remotes.push(r);
  }

  const choice = pickChannel({ local: localManifest, remotes, force });
  let jsUrl = null;
  let cssUrl = null;
  let channel = choice.channel;
  let version = (choice.manifest && choice.manifest.version) || LOCAL_VERSION;
  let runningManifest = choice.manifest;

  try {
    if (choice.channel === 'remote') {
      const remoteUrl = remotes.find((r) => r.manifest === choice.manifest)?.url;
      const assets = resolveAsset(choice.manifest, remoteUrl);
      jsUrl = assets.js;
      cssUrl = assets.css;
    } else if (localManifest.js) {
      const assets = resolveAsset(localManifest, localUrl);
      jsUrl = assets.js;
      cssUrl = assets.css;
    }
  } catch {
    jsUrl = null;
  }

  const runLocalBundleFallback = async () => {
    // Last resort for builds without app-manifest.json: meta tags injected
    // into dist/index.html by scripts/gen-manifest.mjs.
    const js = document.querySelector('meta[name="ota-app-js"]')?.content;
    const css = document.querySelector('meta[name="ota-app-css"]')?.content;
    if (js) {
      await loadCss(css ? new URL(css, location.href).toString() : null);
      await import(/* @vite-ignore */ new URL(js, location.href).toString());
      return true;
    }
    return false;
  };

  const markBooted = () => {
    const el = document.querySelector('#app');
    if (el) el.dataset.booted = '1';
    try {
      window.__DC_APP__ = { version, channel, pin: pin || 'auto' };
      localStorage.setItem(META_KEY, JSON.stringify({ version, channel }));
    } catch {
      /* ignore */
    }
    badge(version, channel, () =>
      updatesMenu({ version, channel, pin: pin || 'auto' }, menuActions()),
    );
  };

  const menuActions = () => ({
    check: async () => {
      toast('Checking for update…');
      const fresh = [];
      for (const base of defaultRemoteBases(cfg)) {
        const m = await fetchJson(base.replace(/\/$/, '') + '/' + manifestPath);
        if (m) fresh.push({ manifest: m, base });
      }
      const next = pickChannel({ local: localManifest, remotes: fresh, force: pin });
      if (next.channel === 'remote' && isNewer(next.manifest, runningManifest)) {
        toast(`Update v${next.manifest.version} found — reloading…`);
        setTimeout(() => location.reload(), 900);
      } else {
        toast('Already up to date.');
      }
    },
    togglePin: () => {
      try {
        if (pin === 'local') localStorage.removeItem(PIN_KEY);
        else localStorage.setItem(PIN_KEY, 'local');
      } catch {
        /* ignore */
      }
      location.reload();
    },
    reload: () => location.reload(),
  });

  try {
    if (jsUrl) {
      splash(channel === 'remote' ? `Updating to v${version}…` : 'Loading…');
      await loadCss(cssUrl);
      await import(/* @vite-ignore */ jsUrl);
      markBooted();
      return;
    }
    if (await runLocalBundleFallback()) {
      channel = 'local';
      version = LOCAL_VERSION;
      runningManifest = localManifest;
      markBooted();
      return;
    }
    throw new Error('no bundle found');
  } catch (e) {
    // Remote failed after being selected -> fall back to bundled code.
    if (channel === 'remote') {
      try {
        if (localManifest.js) {
          const assets = resolveAsset(localManifest, localUrl);
          await loadCss(assets.css);
          await import(/* @vite-ignore */ assets.js);
          channel = 'local';
          version = localManifest.version;
          runningManifest = localManifest;
          markBooted();
          return;
        }
        if (await runLocalBundleFallback()) {
          channel = 'local';
          version = LOCAL_VERSION;
          runningManifest = localManifest;
          markBooted();
          return;
        }
      } catch {
        /* fall through to error */
      }
    }
    splash('Could not start the app. Check your connection and reload.');
    console.error('[ota-loader] boot failed', e);
  }
}

boot();

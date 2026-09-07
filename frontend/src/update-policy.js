// Pure OTA update policy: no DOM, no network. Safe to unit-test in Node.
// Manifest schema v1: { schema:1, name, version, build, builtAt, js, css }
// `js`/`css` are URLs relative to the manifest URL that served them.

export function parseVersion(v) {
  if (typeof v !== 'string') return [];
  const m = v.trim().match(/^v?(\d+(?:\.\d+){0,3})(?:[-+].*)?$/);
  if (!m) return [];
  return m[1].split('.').map((n) => parseInt(n, 10));
}

// Returns 1 if a > b, -1 if a < b, 0 if equal/undecidable.
export function compareVersions(a, b) {
  const pa = parseVersion(a);
  const pb = parseVersion(b);
  if (!pa.length || !pb.length) {
    if (a === b) return 0;
    return 0; // undecidable -> treat as equal (stay on local)
  }
  const n = Math.max(pa.length, pb.length);
  for (let i = 0; i < n; i++) {
    const x = pa[i] || 0;
    const y = pb[i] || 0;
    if (x > y) return 1;
    if (x < y) return -1;
  }
  return 0;
}

export function isValidManifest(m) {
  return (
    !!m &&
    typeof m === 'object' &&
    m.schema === 1 &&
    typeof m.version === 'string' &&
    typeof m.js === 'string' &&
    m.js.length > 0
  );
}

// A remote bundle is "newer" when its version is strictly higher, or when
// versions match but it was built later (same-version redeploy / hotfix).
export function isNewer(remote, local) {
  if (!isValidManifest(remote)) return false;
  if (!isValidManifest(local)) return true;
  const vc = compareVersions(remote.version, local.version);
  if (vc !== 0) return vc > 0;
  if (typeof remote.builtAt === 'string' && typeof local.builtAt === 'string') {
    return remote.builtAt > local.builtAt;
  }
  return false;
}

// Decide which bundle to run. Returns { channel:'remote'|'local', manifest, base }.
// - `force` is 'local' | 'remote' | null (query-string / pinned override).
// - Remote wins only when it is newer (see isNewer). Otherwise -> local.
export function pickChannel({ local, remotes, force }) {
  if (force === 'local') return { channel: 'local', manifest: local, base: null };
  let best = null;
  for (const r of remotes || []) {
    if (!isValidManifest(r.manifest)) continue;
    if (!best || compareVersions(r.manifest.version, best.manifest.version) > 0) best = r;
  }
  if (force === 'remote' && best) return { channel: 'remote', manifest: best.manifest, base: best.base };
  if (best && isNewer(best.manifest, local)) {
    return { channel: 'remote', manifest: best.manifest, base: best.base };
  }
  return { channel: 'local', manifest: local, base: null };
}

// Resolve manifest-relative asset paths against the manifest URL.
// Local manifests resolve against the page URL (base === null).
export function resolveAsset(manifest, manifestUrl) {
  const js = new URL(manifest.js, manifestUrl).toString();
  const css = manifest.css ? new URL(manifest.css, manifestUrl).toString() : null;
  return { js, css };
}

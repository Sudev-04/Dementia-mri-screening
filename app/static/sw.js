// Service worker: makes the app open and run with no network.
// Bump VERSION whenever any file in SHELL changes so phones pick up the update.
const VERSION = "v4";
const CACHE = `alzscreen-${VERSION}`;
const SHELL = [
  "/",
  "/static/styles.css",
  "/static/js/app.js",
  "/static/js/scoring.js",
  "/static/js/store.js",
  "/static/js/scanlocal.js",
  "/static/manifest.webmanifest",
  "/static/icon.svg",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE)
      .then((c) => c.addAll([...SHELL, "/api/bundle"]))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== location.origin) return;

  // Screening questions and rules: fresh copy when online, saved copy when not.
  if (url.pathname === "/api/bundle") {
    event.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
          return res;
        })
        .catch(() => caches.match(req)),
    );
    return;
  }
  if (url.pathname.startsWith("/api/")) return; // personal data is never cached here

  // App shell: saved copy first, so the app opens instantly without signal.
  event.respondWith(caches.match(req, { ignoreSearch: true }).then((hit) => hit || fetch(req)));
});

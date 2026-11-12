// On-device storage. Screenings live in IndexedDB so they survive the browser
// being closed with no signal; small settings live in localStorage.

const DB_NAME = "alzscreen";
const STORE = "screenings";
let dbPromise = null;

function open() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "id" });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}

async function tx(mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const req = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(req ? req.result : undefined);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

export const screenings = {
  put: (rec) => tx("readwrite", (s) => s.put(rec)),
  get: (id) => tx("readonly", (s) => s.get(id)),
  all: () => tx("readonly", (s) => s.getAll()),
  remove: (id) => tx("readwrite", (s) => s.delete(id)),
};

export const settings = {
  get(key, fallback = null) {
    try {
      const raw = localStorage.getItem(`alz.${key}`);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      if (value === null || value === undefined) localStorage.removeItem(`alz.${key}`);
      else localStorage.setItem(`alz.${key}`, JSON.stringify(value));
    } catch {
      /* storage full or blocked: the app still works for this session */
    }
  },
};

// Personal details should not sit on a field phone longer than needed.
// Uploaded screenings are removed from the device after this many days.
export const KEEP_SYNCED_DAYS = 7;

export async function purgeOld(now = Date.now()) {
  const limit = now - KEEP_SYNCED_DAYS * 24 * 3600 * 1000;
  let removed = 0;
  for (const rec of await screenings.all()) {
    if (rec.synced && rec.synced_at && Date.parse(rec.synced_at) < limit) {
      await screenings.remove(rec.id);
      removed += 1;
    }
  }
  return removed;
}

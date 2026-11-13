// Scan measurement in the browser, for the server-less demo.
// Mirrors common/preprocess.py (input checks) and common/measure.py (brain fraction);
// tests/js/scan_parity.mjs checks it against the Python result on the same file.

const TYPES = { 2: [Uint8Array, 1], 4: [Int16Array, 2], 8: [Int32Array, 4], 16: [Float32Array, 4], 64: [Float64Array, 8], 256: [Int8Array, 1], 512: [Uint16Array, 2], 768: [Uint32Array, 4] };
const GET = { 2: "getUint8", 4: "getInt16", 8: "getInt32", 16: "getFloat32", 64: "getFloat64", 256: "getInt8", 512: "getUint16", 768: "getUint32" };
export const EXPECTED_SHAPE = [176, 208, 176];

export class ScanRejected extends Error {}

/** Parse an Analyze 7.5 pair (hdr + img) or a single-file NIfTI-1 buffer into {data, shape, zooms}. */
export function parseVolume(hdrBuf, imgBuf) {
  const h = new DataView(hdrBuf);
  if (hdrBuf.byteLength < 348) throw new ScanRejected("The file is too small to be an MRI header.");
  let le = true;
  if (h.getInt32(0, true) !== 348) {
    if (h.getInt32(0, false) !== 348) throw new ScanRejected("The file could not be read as an MRI volume (not Analyze or NIfTI-1).");
    le = false;
  }
  const ndim = h.getInt16(40, le);
  const shape = [h.getInt16(42, le), h.getInt16(44, le), h.getInt16(46, le)];
  const extra = [4, 5, 6, 7].slice(0, Math.max(0, ndim - 3)).map((i) => h.getInt16(40 + 2 * i, le)).filter((n) => n > 1);
  if (ndim < 3 || extra.length || shape.some((n) => n < 1)) throw new ScanRejected(`Expected a 3D volume, got dimensions ${[...shape, ...extra].join(" x ")}.`);
  const datatype = h.getInt16(70, le);
  if (!TYPES[datatype]) throw new ScanRejected(`Unsupported voxel data type (${datatype}).`);
  const zooms = [h.getFloat32(80, le), h.getFloat32(84, le), h.getFloat32(88, le)];
  const magic = String.fromCharCode(h.getUint8(344), h.getUint8(345), h.getUint8(346));
  const single = magic === "n+1";
  const body = single ? hdrBuf : imgBuf;
  if (!body) throw new ScanRejected("Upload the .img file together with the .hdr file.");
  const offset = single ? Math.round(h.getFloat32(108, le)) : 0;
  const n = shape[0] * shape[1] * shape[2];
  const size = TYPES[datatype][1];
  if (body.byteLength < offset + n * size) throw new ScanRejected("The image file is shorter than its header says. The download may be incomplete.");
  const view = new DataView(body, offset, n * size);
  const data = new Float32Array(n);
  const get = GET[datatype];
  for (let i = 0; i < n; i++) data[i] = view[get](i * size, le);
  if (magic.startsWith("n")) {
    const slope = h.getFloat32(112, le), inter = h.getFloat32(116, le);
    if (slope && Number.isFinite(slope) && (slope !== 1 || inter !== 0)) for (let i = 0; i < n; i++) data[i] = data[i] * slope + inter;
  }
  return { data, shape, zooms };
}

export function checkInput({ data, shape, zooms }) {
  const problems = [];
  if (shape.join() !== EXPECTED_SHAPE.join()) problems.push(`shape is (${shape.join(", ")}), expected (${EXPECTED_SHAPE.join(", ")}) (atlas-registered 1 mm volume)`);
  if (zooms.some((z) => Math.abs(z - 1) > 0.05)) problems.push(`voxel size is (${zooms.map((z) => z.toFixed(3)).join(", ")}) mm, expected 1 mm isotropic`);
  let background = 0, bad = false;
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i])) bad = true;
    else if (data[i] <= 0) background++;
  }
  if (bad) problems.push("volume contains NaN or infinite values");
  const frac = background / data.length;
  if (!problems.length && !(frac >= 0.45 && frac <= 0.92)) problems.push(`${Math.round(frac * 100)}% of voxels are background; a skull-stripped brain in atlas space has between 45% and 92%`);
  if (problems.length) {
    throw new ScanRejected(`This scan does not match what the measurement was built for: ${problems.join("; ")}. Only T1 volumes that are bias-corrected, skull-stripped and registered to the T88 atlas, as in OASIS-1, are accepted.`);
  }
  return { background_fraction: Math.round(frac * 1e4) / 1e4 };
}

/** numpy.percentile (linear interpolation) of the positive voxels. */
export function robustMax(data, q = 99.5) {
  let n = 0;
  for (let i = 0; i < data.length; i++) if (data[i] > 0) n++;
  if (!n) throw new ScanRejected("volume has no brain voxels");
  const pos = new Float32Array(n);
  for (let i = 0, k = 0; i < data.length; i++) if (data[i] > 0) pos[k++] = data[i];
  pos.sort();
  const idx = (q / 100) * (n - 1), lo = Math.floor(idx), hi = Math.min(n - 1, lo + 1);
  return { scale: pos[lo] + (pos[hi] - pos[lo]) * (idx - lo), brainVoxels: n };
}

export function brainFraction(data, threshold) {
  const { scale, brainVoxels } = robustMax(data);
  const cut = Math.fround(threshold * scale);
  let tissue = 0;
  for (let i = 0; i < data.length; i++) if (data[i] > cut) tissue++;
  return { fraction: tissue / brainVoxels, scale };
}

function erf(x) {
  const s = x < 0 ? -1 : 1, a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a));
}

export function assess(data, age, ref) {
  const { fraction, scale } = brainFraction(data, ref.threshold);
  const [a, b, c] = ref.age_poly_coef;
  const expected = a * age * age + b * age + c;
  const z = (fraction - expected) / (age >= 60 ? ref.resid_sd_60plus : ref.resid_sd);
  const r4 = (v) => Math.round(v * 1e4) / 1e4;
  return {
    measure: { brain_fraction: r4(fraction), expected_for_age: r4(expected), z: Math.round(z * 100) / 100,
      percentile: Math.round(1000 * 0.5 * (1 + erf(z / Math.SQRT2))) / 10, band: z >= -1 ? "usual" : z >= -2 ? "lower" : "well_below",
      age, age_in_reference_range: age >= ref.age_range[0] && age <= ref.age_range[1], reference_n: ref.reference_n },
    scale,
  };
}

/** A synthetic stand-in for a scan: noise-textured ellipsoid with a dark centre. Not anatomy. */
export function sampleVolume() {
  const [nx, ny, nz] = EXPECTED_SHAPE, data = new Float32Array(nx * ny * nz);
  const r = 26 + Math.random() * 10;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const dx = x - 88, dy = y - 104, dz = z - 88;
    if ((dx / 66) ** 2 + (dy / 84) ** 2 + (dz / 62) ** 2 > 1) continue;
    let v = 900 + 60 * (Math.random() + Math.random() + Math.random() - 1.5) * 2;
    if ((dx / r) ** 2 + (dy / (r * 1.5)) ** 2 + (dz / r) ** 2 <= 1) v *= 0.2;
    data[x + nx * (y + ny * z)] = v;
  }
  return { data, shape: EXPECTED_SHAPE, zooms: [1, 1, 1] };
}

/** Three orthogonal mid-slices as a PNG data URL (browser only). */
export function renderSlices({ data, shape }, scale) {
  const [nx, ny, nz] = shape, at = (x, y, z) => Math.max(0, Math.min(255, (255 * data[x + nx * (y + ny * z)]) / scale));
  const views = [[ny, nz, (i, j) => at(nx >> 1, i, j)], [nx, nz, (i, j) => at(i, ny >> 1, j)], [nx, ny, (i, j) => at(i, j, nz >> 1)]];
  const gap = 8, height = Math.max(...views.map((v) => v[1])), width = views.reduce((s, v) => s + v[0], 0) + gap * 2;
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(width, height);
  for (let k = 3; k < img.data.length; k += 4) img.data[k] = 255;
  let x0 = 0;
  for (const [w, h, f] of views) {
    const y0 = (height - h) >> 1;
    for (let row = 0; row < h; row++) for (let col = 0; col < w; col++) {
      const g = f(col, h - 1 - row), p = 4 * ((y0 + row) * width + x0 + col);
      img.data[p] = img.data[p + 1] = img.data[p + 2] = g;
    }
    x0 += w + gap;
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

export async function readFiles(files) {
  const byExt = {};
  for (const f of files) byExt[f.name.toLowerCase().match(/\.nii\.gz$|\.[a-z]+$/)?.[0] || ""] = f;
  if (byExt[".hdr"] && byExt[".img"] && files.length === 2) return parseVolume(await byExt[".hdr"].arrayBuffer(), await byExt[".img"].arrayBuffer());
  if (byExt[".nii"] && files.length === 1) return parseVolume(await byExt[".nii"].arrayBuffer());
  if (byExt[".nii.gz"] && files.length === 1) {
    const stream = byExt[".nii.gz"].stream().pipeThrough(new DecompressionStream("gzip"));
    return parseVolume(await new Response(stream).arrayBuffer());
  }
  throw new ScanRejected("Upload one NIfTI file, or exactly one .hdr and one .img that belong together.");
}

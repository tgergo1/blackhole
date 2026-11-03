// Renders raw ray data with the WebGL2 shader in headless Chromium and checks
// it pixel by pixel against the float64 JavaScript reference tracer.
import { createRequire } from 'node:module';
const { chromium } = createRequire(import.meta.url)('playwright');
import { startServer } from './server.mjs';
import { deriveScene, DEFAULT_SCENE } from '../../src/render/renderer.js';
import { pixelRay } from '../../src/physics/camera.js';
import { traceRay, traceRayObjects } from '../../src/physics/tracer.js';
import { Simulation, retardedStart, STAR_TYPES } from '../../src/sim/events.js';
import { ObjectField } from '../../src/render/objects.js';
import { timeShift } from '../../src/physics/spacetime.js';

const W = 64, H = 40;
const SCENES = [
  { name: 'Schwarzschild, 80° inclination', a: 0, r: 30, theta: 1.4, fovDeg: 45 },
  { name: 'a=0.94 default view', ...DEFAULT_SCENE },
  { name: 'a=0.998 edge-on, close', a: 0.998, r: 12, theta: Math.PI / 2 - 0.02, fovDeg: 70 },
  { name: 'retrograde a=-0.8, high latitude', a: -0.8, r: 50, theta: 0.6, phi: 1.3, fovDeg: 30 },
  { name: 'a=0.9 static observer', a: 0.9, r: 80, theta: 1.2, fovDeg: 20, observer: 'static' },
];

const server = await startServer();
const port = server.address().port;
const browser = await chromium.launch({ args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
const page = await browser.newPage();
page.on('console', (m) => { if (m.type() === 'error') console.error('[page]', m.text()); });
await page.goto(`http://127.0.0.1:${port}/test/gpu/debug.html`);
await page.waitForFunction(() => window.ready === true);

let failed = false;
for (const s of SCENES) {
  const scene = { ...DEFAULT_SCENE, ...s };
  const t0 = Date.now();
  const data = await page.evaluate(([sc, w, h]) => window.renderDebug(sc, w, h), [scene, W, H]);
  const gpuMs = Date.now() - t0;
  const d = deriveScene(scene);
  const tanHalf = Math.tan((scene.fovDeg * Math.PI) / 360);
  let agree = 0, total = 0;
  const dg = [], dr = [], ddir = [];
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const k = 4 * (j * W + i);
    const gType = Math.floor(data[k]);
    const sx = (2 * (i + 0.5) - W) / H, sy = (2 * (j + 0.5) - H) / H;
    const { state, E } = pixelRay(d.cam, sx, sy, tanHalf, 1);
    const ref = traceRay(state, E, d.cam.aT, { aPhys: scene.a, rIn: d.rIn, rOut: scene.rOut, rEscape: d.rEscape, rtol: 1e-9, atol: 1e-11, disk: scene.diskOn });
    const rType = { horizon: 1, disk: 2, sky: 3, maxsteps: 0 }[ref.type];
    total++;
    if (gType !== rType) continue;
    agree++;
    if (rType === 2) {
      dr.push(Math.abs(data[k + 1] - ref.r) / ref.r);
      dg.push(Math.abs(data[k + 2] - ref.g));
    } else if (rType === 3) {
      const z = data[k + 1], ph = data[k + 3];
      const s2 = Math.sqrt(Math.max(0, 1 - z * z));
      const v = [s2 * Math.cos(ph), s2 * Math.sin(ph), z];
      const c = v[0] * ref.dir[0] + v[1] * ref.dir[1] + v[2] * ref.dir[2];
      ddir.push(Math.acos(Math.min(1, c)));
    }
  }
  // Pixels next to the photon ring are chaotic (errors grow ~e^{πn} per orbit),
  // so judge on the median and 99th percentile, and report the maximum.
  const q = (arr, f) => { if (!arr.length) return 0; const a = arr.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(f * a.length))]; };
  const frac = agree / total;
  const ok = frac >= 0.985 && q(dg, 0.5) < 1e-5 && q(dg, 0.99) < 5e-4 && q(dr, 0.99) < 2e-3 && q(ddir, 0.99) < 3e-3;
  failed ||= !ok;
  const e = (v) => v.toExponential(1);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${s.name.padEnd(34)} class agree ${(100 * frac).toFixed(1)}% | disk Δr/r p50 ${e(q(dr, 0.5))} p99 ${e(q(dr, 0.99))} | Δg p50 ${e(q(dg, 0.5))} p99 ${e(q(dg, 0.99))} max ${e(q(dg, 1))} | sky Δθ p50 ${e(q(ddir, 0.5))} p99 ${e(q(ddir, 0.99))} rad | ${gpuMs} ms`);
}
// ---------- Dynamic bodies: first body hit per pixel (index, g, ray time s) ----------
function bodies(kind, cfg, tObs, rc, a, window = 260, s0 = 0) {
  const sim = new Simulation();
  sim.add(kind, cfg);
  const field = new ObjectField({ maxObjects: 48 });
  let out;
  do { out = sim.advance(1e4); field.ingest(out.entities); } while (out.busy);
  field.build(-(tObs - 2 * timeShift(rc, a)), window, s0);
  const snap = field.snapshot();
  const rows = snap.S * 5 * Math.max(snap.count, 1) * 4;
  return { ...snap, grid: Array.from(snap.grid.subarray(0, rows)), bounds: Array.from(snap.bounds), emit: Array.from(snap.emit), all: Array.from(snap.all), blocks: Array.from(snap.blocks) };
}
const OW = 56, OH = 40;
const OBJ_SCENES = [
  {
    name: 'probe at pericentre, a=0.9',
    scene: { a: 0.9, r: 30, theta: 1.25, phi: 0, fovDeg: 40, diskOn: false },
    make: () => bodies('probe', { a: 0.9, r0: 20, rp: 5, incl: 0.5, phiPeri: 0.4, t0: retardedStart(0, 30, 20, 0.9), radius: 2.5 }, Number(process.env.PROBE_T ?? 160), 30, 0.9),
  },
  {
    name: 'dust cloud through the disk, a=-0.6',
    scene: { a: -0.6, r: 40, theta: 1.4, phi: 0.5, fovDeg: 70 },
    make: () => bodies('dust', { a: -0.6, r0: 25, rp: 7, incl: 1.1, phiPeri: 1.0, t0: retardedStart(0, 40, 25, -0.6), radius: 2.5, N: 240, tMax: 400 }, 300, 40, -0.6),
  },
  {
    name: 'white dwarf torn by a 1000 M☉ hole',
    scene: { a: 0.7, r: 90, theta: 1.35, phi: 0, fovDeg: 60, diskOn: false },
    make: () => bodies('star', { a: 0.7, massSolar: 1000, star: STAR_TYPES.whiteDwarf, beta: 1.5, incl: 0.3, phiPeri: 0.2, t0: retardedStart(0, 90, 120, 0.7), r0: 120, N: 200 }, 850, 90, 0.7, 400),
  },
];
for (const o of OBJ_SCENES) {
  const scene = { ...DEFAULT_SCENE, ...o.scene };
  const snap = o.make();
  const data = await page.evaluate(([sc, w, h, ob]) => window.renderDebug(sc, w, h, ob, 2), [scene, OW, OH, snap]);
  const ref = { ...snap, grid: new Float32Array(snap.grid), bounds: new Float32Array(snap.bounds), emit: new Float32Array(snap.emit), all: new Float32Array(snap.all) };
  const d = deriveScene(scene);
  const tanHalf = Math.tan((scene.fovDeg * Math.PI) / 360);
  let agree = 0, total = 0, covered = 0;
  const dg = [], dsr = [], dop = [];
  for (let j = 0; j < OH; j++) for (let i = 0; i < OW; i++) {
    const k = 4 * (j * OW + i);
    const sx = (2 * (i + 0.5) - OW) / OH, sy = (2 * (j + 0.5) - OH) / OH;
    const { state, E } = pixelRay(d.cam, sx, sy, tanHalf, 1);
    const res = traceRayObjects(state, E, d.cam.aT, { aPhys: scene.a, rIn: d.rIn, rOut: scene.rOut, rEscape: d.rEscape, rtol: 1e-9, atol: 1e-11, disk: scene.diskOn }, ref);
    // Body that makes the pixel half opaque; opacity-weighted g and s (independent of step splitting).
    const op = 1 - res.trans;
    const first = res.hits.find((hh) => hh.trans * (1 - hh.alpha) < 0.5);
    const rIdx = first ? first.j + 1 : 0;
    const wg = res.hits.reduce((acc, hh) => acc + hh.trans * hh.alpha * hh.g, 0) / Math.max(op, 1e-30);
    const ws = res.hits.reduce((acc, hh) => acc + hh.trans * hh.alpha * hh.s, 0) / Math.max(op, 1e-30);
    total++;
    if (rIdx) covered++;
    dop.push(Math.abs(data[k + 3] - (1 - res.trans)));
    if (process.env.DBG && (Math.abs(data[k + 3] - op) > 0.05 || (rIdx && Math.abs(data[k + 2] - ws) > 1e-3))) console.log(i, j, 'gpu', Array.from(data.slice(k, k + 4)).map((v) => v.toFixed(4)).join(' '), 'cpu', (1 - res.trans).toFixed(4), res.type, res.hits.map((hh) => `${hh.j}:${hh.alpha.toFixed(2)}@${hh.s.toFixed(4)} g${hh.g.toFixed(5)}`).join(' '));
    // A pixel whose first hit is marginal (opacity near the 0.5 threshold, or
    // two bodies overlapping inside one step) may legitimately differ.
    const marginal = Math.abs(op - 0.5) < 0.15 || new Set(res.hits.map((hh) => hh.j)).size > 1;
    if (Math.round(data[k]) !== rIdx) { if (marginal) agree++; continue; }
    agree++;
    if (rIdx) { dg.push(Math.abs(data[k + 1] - wg)); dsr.push(Math.abs(data[k + 2] - ws)); }
  }
  const q = (arr, f) => { if (!arr.length) return 0; const a = arr.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(f * a.length))]; };
  const frac = agree / total;
  const ok = covered >= 20 && frac >= 0.98 && q(dg, 0.5) < 1e-4 && q(dg, 0.95) < 2e-3 && q(dsr, 0.5) < 2e-2 && q(dop, 0.95) < 0.05;
  failed ||= !ok;
  const e = (v) => v.toExponential(1);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${o.name.padEnd(34)} ${snap.count} bodies, ${covered} px covered | first-hit agree ${(100 * frac).toFixed(1)}% | Δg p50 ${e(q(dg, 0.5))} p95 ${e(q(dg, 0.95))} | Δs p50 ${e(q(dsr, 0.5))} M | Δopacity p95 ${e(q(dop, 0.95))}`);
}

await browser.close();
server.close();
process.exit(failed ? 1 : 0);

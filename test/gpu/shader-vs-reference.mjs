// Renders raw ray data with the WebGL2 shader in headless Chromium and checks
// it pixel by pixel against the float64 JavaScript reference tracer.
import { createRequire } from 'node:module';
const { chromium } = createRequire(import.meta.url)('playwright');
import { startServer } from './server.mjs';
import { deriveScene, DEFAULT_SCENE } from '../../src/render/renderer.js';
import { pixelRay } from '../../src/physics/camera.js';
import { traceRay } from '../../src/physics/tracer.js';

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
await browser.close();
server.close();
process.exit(failed ? 1 : 0);

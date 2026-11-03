import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  iscoRadius, photonOrbitRadii, radiativeEfficiency, fluxShape, fluxPeak, keplerOmega,
  circularOrbitEnergy, diskFourVelocity, peakTemperature, limbDarkening,
} from '../src/physics/disk.js';
import { metricLower } from '../src/physics/kerr.js';
import { blackbodyChromaticity, buildBlackbodyLUT } from '../src/physics/color.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} |${a} - ${b}| = ${Math.abs(a - b)} > ${tol}`);

test('ISCO and photon orbit radii (Bardeen, Press & Teukolsky 1972)', () => {
  close(iscoRadius(0), 6, 1e-12);
  close(iscoRadius(1), 1, 1e-6);
  close(iscoRadius(-1), 9, 1e-12);
  close(iscoRadius(0.998), 1.2369, 1e-4);
  close(photonOrbitRadii(0).pro, 3, 1e-12);
  close(photonOrbitRadii(1).pro, 1, 1e-12);
  close(photonOrbitRadii(1).retro, 4, 1e-12);
});

test('radiative efficiency 1 − E_isco', () => {
  close(radiativeEfficiency(0), 1 - Math.sqrt(8 / 9), 1e-12);
  close(radiativeEfficiency(0.998), 0.3208, 5e-4); // Thorne (1974) limit
});

test('circular orbit four-velocity is normalised and matches the analytic energy', () => {
  for (const a of [0, 0.9, -0.7]) {
    for (const r of [iscoRadius(a) + 0.01, 8, 25]) {
      // Physical (non-mirrored) spacetime, gas at (r', 0, 0) with r'² = r² + a².
      const x = Math.sqrt(r * r + a * a);
      const u = diskFourVelocity(x, 0, keplerOmega(r, a), a);
      const g = metricLower([x, 0, 0], a);
      let n = 0;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) n += g[i][j] * u[i] * u[j];
      close(n, -1, 1e-12);
      const ut = [0, 1, 2, 3].reduce((s, j) => s + g[0][j] * u[j], 0);
      close(-ut, circularOrbitEnergy(r, a), 1e-12, `E(r=${r}, a=${a})`);
    }
  }
});

// Direct numerical evaluation of the Page–Thorne integral
//   F = (Ṁ / 4π r) · (−Ω,r) / (E − ΩL)² · ∫_{r_isco}^{r} (E − ΩL) L,r dr
function pageThorneNumeric(r, a) {
  const E = (x) => circularOrbitEnergy(x, a);
  const L = (x) => (x * x - 2 * a * Math.sqrt(x) + a * a) / (x ** 0.75 * Math.sqrt(x ** 1.5 - 3 * Math.sqrt(x) + 2 * a));
  const W = (x) => keplerOmega(x, a);
  const d = (fn, x) => (fn(x + 1e-6 * x) - fn(x - 1e-6 * x)) / (2e-6 * x);
  const r0 = iscoRadius(a);
  const N = 4000;
  let I = 0;
  for (let i = 0; i < N; i++) {
    // midpoint rule in a stretched variable to resolve the ISCO end
    const t0 = i / N, t1 = (i + 1) / N, tm = (t0 + t1) / 2;
    const map = (t) => r0 + (r - r0) * t * t;
    const x = map(tm), dx = map(t1) - map(t0);
    I += (E(x) - W(x) * L(x)) * d(L, x) * dx;
  }
  const F = (1 / (4 * Math.PI * r)) * (-d(W, r)) / (E(r) - W(r) * L(r)) ** 2 * I;
  return F / (3 / (8 * Math.PI)); // in units of 3Ṁ/(8π M²)
}

test('closed-form Page–Thorne flux matches direct numerical integration', () => {
  for (const a of [0, 0.5, 0.9, -0.5]) {
    for (const r of [iscoRadius(a) * 1.1, iscoRadius(a) * 2, 30]) {
      const ref = pageThorneNumeric(r, a);
      close(fluxShape(r, a), ref, 2e-5 * Math.abs(ref) + 1e-12, `a=${a} r=${r}`);
    }
  }
  assert.equal(fluxShape(iscoRadius(0.3), 0.3), 0);
  // Newtonian limit F → 3ṀM/(8π r³) up to O(√(r_in/r)) corrections
  close(fluxShape(1e6, 0.5) * 1e18, 1, 1e-2);
});

test('peak temperature scales as (ṁ / M)^{1/4} and has a sane magnitude', () => {
  const t1 = peakTemperature(10, 0.1, 0);
  const t2 = peakTemperature(1e8, 0.1, 0);
  close(t1 / t2, Math.pow(1e7, 0.25), 1e-9 * t1 / t2);
  assert.ok(t1 > 3e6 && t1 < 3e7, `10 Msun X-ray binary disk ~10^7 K, got ${t1}`);
  assert.ok(fluxPeak(0).r > 6 && fluxPeak(0).r < 12);
});

test('limb darkening conserves flux', () => {
  let s = 0;
  const N = 10000;
  for (let i = 0; i < N; i++) { const mu = (i + 0.5) / N; s += limbDarkening(mu) * mu / N; }
  close(2 * s, 1, 1e-6);
});

test('blackbody chromaticities match CIE illuminants', () => {
  const [x65, y65] = blackbodyChromaticity(6504);
  close(x65, 0.3135, 0.003); close(y65, 0.3237, 0.004); // Planckian locus at 6504 K
  const [xa, ya] = blackbodyChromaticity(2856);
  close(xa, 0.4476, 0.003); close(ya, 0.4074, 0.003); // CIE illuminant A
  const lut = buildBlackbodyLUT(64);
  assert.equal(lut.data.length, 256);
});

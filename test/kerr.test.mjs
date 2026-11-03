import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  geodesicRHS, hamiltonian, metricLower, metricUpper, ksRadius, carterQ,
  angularMomentum, fromSpheroidal, horizons,
} from '../src/physics/kerr.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} |${a} - ${b}| > ${tol}`);

test('upper and lower metrics are inverse', () => {
  for (const a of [0, 0.5, -0.9, 0.999]) {
    for (const p of [[3, 1, 2], [0.3, -0.2, 1.5], [10, -4, -0.01], [1.2, 0.4, 0.0]]) {
      const g = metricLower(p, a), gi = metricUpper(p, a);
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += g[i][k] * gi[k][j];
        close(s, i === j ? 1 : 0, 1e-12, `a=${a} p=${p} (${i},${j})`);
      }
    }
  }
});

test('Kerr–Schild radius satisfies its defining spheroid equation', () => {
  for (const a of [0, 0.7, 0.998]) {
    for (const [r, th, ph] of [[1.5, 0.3, 1], [8, 1.57, -2], [2.2, 2.9, 0.4], [50, 0.01, 3]]) {
      const p = fromSpheroidal(r, th, ph, a);
      close(ksRadius(...p, a), r, 1e-12 * r);
    }
  }
});

test('analytic Hamiltonian gradient matches central finite differences', () => {
  const E = 1.0;
  for (const a of [0, 0.6, -0.95]) {
    for (const s of [[4, 2, 1.5, 0.3, -0.7, 0.2], [1.1, -2.3, 0.4, -0.5, 0.1, 0.9], [6, 0.5, -3, 0.2, 0.2, -0.4]]) {
      const d = geodesicRHS(s, E, a);
      for (let i = 0; i < 6; i++) {
        const h = 1e-6;
        const sp = s.slice(), sm = s.slice();
        sp[i] += h; sm[i] -= h;
        const fd = (hamiltonian(sp, E, a) - hamiltonian(sm, E, a)) / (2 * h);
        // dx/dλ = +∂H/∂p ; dp/dλ = −∂H/∂x
        const expect = i < 3 ? -fd : fd;
        const got = i < 3 ? d[i + 3] : d[i - 3];
        close(got, expect, 1e-7 * Math.max(1, Math.abs(expect)), `a=${a} comp=${i}`);
      }
    }
  }
});

test('spin sign convention: a > 0 drags frames toward +φ', () => {
  const g = metricLower([3, 0, 0], 0.9);
  // ∂_φ = (0, −y, x, 0) = (0, 0, 3, 0) here; ω = −g_tφ / g_φφ
  const gtphi = 3 * g[0][2], gphiphi = 9 * g[2][2];
  assert.ok(-gtphi / gphiphi > 0);
});

test('horizon radii', () => {
  close(horizons(0).rPlus, 2, 1e-15);
  close(horizons(1).rPlus, 1, 1e-15);
});

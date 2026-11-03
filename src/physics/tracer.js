// Adaptive Dormand–Prince 5(4) integration of Kerr null geodesics with
// thin-disk intersection. Float64 reference for the GPU shader.

import { geodesicRHS, ksRadius, horizons, hamiltonian, angularMomentum } from './kerr.js';
import { iscoRadius, keplerOmega, diskFourVelocity } from './disk.js';

// Dormand & Prince (1980) tableau.
const A = [
  [],
  [1 / 5],
  [3 / 40, 9 / 40],
  [44 / 45, -56 / 15, 32 / 9],
  [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
  [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
  [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
];
// Error coefficients b5 − b4.
const EC = [71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40];

/**
 * One DOPRI5 step from s with derivative k1 = f(s). Returns
 * { y, k7 (= f(y), FSAL), err (scaled RMS error) }.
 */
export function dopriStep(s, k1, h, E, a, rtol, atol) {
  const k = [k1];
  const tmp = new Array(6);
  for (let st = 1; st < 7; st++) {
    for (let i = 0; i < 6; i++) {
      let acc = s[i];
      for (let j = 0; j < st; j++) acc += h * A[st][j] * k[j][i];
      tmp[i] = acc;
    }
    k.push(geodesicRHS(tmp, E, a, new Array(6)));
  }
  const y = tmp.slice(); // stage 7 point is the 5th-order solution
  let errSq = 0;
  const r = Math.max(1, Math.abs(ksRadius(s[0], s[1], s[2], a)));
  for (let i = 0; i < 6; i++) {
    let e = 0;
    for (let j = 0; j < 7; j++) e += EC[j] * k[j][i];
    e *= h;
    // Positions scale with r, momenta are O(E).
    const sc = atol + rtol * (i < 3 ? r : Math.max(Math.abs(s[i]), Math.abs(y[i]), E));
    errSq += (e / sc) ** 2;
  }
  return { y, k7: k[6], err: Math.sqrt(errSq / 6) };
}

/** Classical RK4 step (used to land exactly on the disk plane). */
function rk4(s, h, E, a) {
  const f = (y) => geodesicRHS(y, E, a, new Array(6));
  const k1 = f(s);
  const k2 = f(s.map((v, i) => v + 0.5 * h * k1[i]));
  const k3 = f(s.map((v, i) => v + 0.5 * h * k2[i]));
  const k4 = f(s.map((v, i) => v + h * k3[i]));
  return s.map((v, i) => v + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

/** Newton iteration on λ to land exactly on z = 0 starting from state s. */
function landOnPlane(s, E, a) {
  let y = s.slice();
  for (let it = 0; it < 6; it++) {
    const d = geodesicRHS(y, E, a, new Array(6));
    if (Math.abs(d[2]) < 1e-300) break;
    const h = -y[2] / d[2];
    y = rk4(y, h, E, a);
    if (Math.abs(y[2]) < 1e-12) break;
  }
  return y;
}

/**
 * Trace one ray in the mirror spacetime (spin aT).
 * opts: { aPhys, rIn, rOut (disk extent), rEscape, rtol, atol, maxSteps, disk }
 */
export function traceRay(state, E, aT, opts = {}) {
  const aPhys = opts.aPhys ?? -aT;
  const rtol = opts.rtol ?? 1e-10;
  const atol = opts.atol ?? 1e-12;
  const maxSteps = opts.maxSteps ?? 20000;
  const rEsc = opts.rEscape ?? 1e4;
  const rH = horizons(aT).rPlus;
  const diskOn = opts.disk !== false;
  const rIn = opts.rIn ?? iscoRadius(aPhys);
  const rOut = opts.rOut ?? 40;

  let s = state.slice();
  let k1 = geodesicRHS(s, E, aT, new Array(6));
  let r0 = ksRadius(s[0], s[1], s[2], aT);
  let h = opts.h0 ?? 0.01 * Math.max(1, r0);
  const out = { steps: 0, rejected: 0, E, H0: hamiltonian(s, E, aT), L0: angularMomentum(s) };
  for (let n = 0; n < maxSteps; n++) {
    const r = ksRadius(s[0], s[1], s[2], aT);
    // Cap the step so a single step cannot cross the disk plane twice.
    const hMax = 0.5 * Math.max(r, 0.1);
    if (h > hMax) h = hMax;
    const st = dopriStep(s, k1, h, E, aT, rtol, atol);
    out.steps++;
    if (st.err > 1 || !Number.isFinite(st.err)) {
      out.rejected++;
      h *= Math.max(0.1, 0.9 * Math.pow(st.err || 1e10, -0.2));
      continue;
    }
    const y = st.y;
    // Disk crossing (z changes sign).
    if (diskOn && s[2] * y[2] <= 0 && s[2] !== y[2]) {
      const hit = landOnPlane(Math.abs(s[2]) < Math.abs(y[2]) ? s : y, E, aT);
      const rho2 = hit[0] * hit[0] + hit[1] * hit[1] - aT * aT;
      const rd = Math.sqrt(Math.max(rho2, 0));
      if (rd >= rIn && rd <= rOut) {
        return { ...out, type: 'disk', state: hit, r: rd, ...diskShading(hit, E, aT, aPhys) };
      }
    }
    s = y;
    k1 = st.k7;
    const rn = ksRadius(s[0], s[1], s[2], aT);
    if (rn < rH * (1 - 1e-6)) return { ...out, type: 'horizon', state: s, r: rn };
    const vr = s[0] * k1[0] + s[1] * k1[1] + s[2] * k1[2];
    if (rn > rEsc && vr > 0) {
      const v = Math.hypot(k1[0], k1[1], k1[2]);
      return { ...out, type: 'sky', state: s, r: rn, dir: [k1[0] / v, k1[1] / v, k1[2] / v], g: 1 / E };
    }
    h *= Math.min(5, Math.max(0.2, 0.9 * Math.pow(st.err || 1e-10, -0.2)));
  }
  return { ...out, type: 'maxsteps', state: s };
}

/**
 * Redshift g = ν_obs/ν_emit and emission-angle cosine μ at a disk hit.
 * Ray normalisation p·u_obs = −1 is assumed (see camera.rayFromDirection).
 */
export function diskShading(hit, E, aT, aPhys) {
  const rd = Math.sqrt(Math.max(hit[0] * hit[0] + hit[1] * hit[1] - aT * aT, 0));
  // Disk gas rotates prograde in the physical spacetime; under time reversal
  // its angular velocity flips sign.
  const omega = -keplerOmega(rd, aPhys);
  const u = diskFourVelocity(hit[0], hit[1], omega, aT);
  const pDotU = -E * u[0] + hit[3] * u[1] + hit[4] * u[2] + hit[5] * u[3];
  const g = -1 / pDotU;
  const mu = Math.min(1, Math.abs(hit[5]) / Math.abs(pDotU));
  return { g, mu, omega };
}

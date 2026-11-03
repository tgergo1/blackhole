// Adaptive Dormand–Prince 5(4) integration of Kerr null geodesics with
// thin-disk intersection. Float64 reference for the GPU shader.

import { geodesicRHS, ksRadius, horizons, hamiltonian, angularMomentum, dtdl } from './kerr.js';
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
export function dopriStep(s, k1, h, E, a, rtol, atol, stages = null) {
  const k = [k1];
  const tmp = new Array(6);
  if (stages) stages.push(s.slice());
  for (let st = 1; st < 7; st++) {
    for (let i = 0; i < 6; i++) {
      let acc = s[i];
      for (let j = 0; j < st; j++) acc += h * A[st][j] * k[j][i];
      tmp[i] = acc;
    }
    if (stages) stages.push(tmp.slice());
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

// ---------- Rays with dynamic bodies (float64 reference for the shader) ----------
//
// Same algorithm as objPos/objStepLimit/objSegment in shaders.js, reading the
// same Float32 grid (src/render/objects.js), but in double precision.

const B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84];

function objPos(snap, j, s) {
  const u = (s - (snap.s0 ?? 0)) / snap.ds;
  if (u < 0 || u > snap.S - 1) return null;
  const k = Math.min(Math.floor(u), snap.S - 2);
  const f = u - k;
  const G = snap.grid, row = j * 5, stride = snap.S * 4;
  const tx = (kk, r) => (row + r) * stride + kk * 4;
  if (G[tx(k, 2) + 3] < 0.5 || G[tx(k + 1, 2) + 3] < 0.5) return null;
  const a0 = tx(k, 0), b0 = tx(k + 1, 0), a1 = tx(k, 1), b1 = tx(k + 1, 1);
  const h = snap.ds, f2 = f * f, f3 = f2 * f;
  const C = [0, 1, 2].map((i) => (2 * f3 - 3 * f2 + 1) * G[a0 + i] + (f3 - 2 * f2 + f) * h * (G[a1 + 1 + i] / G[a1])
    + (-2 * f3 + 3 * f2) * G[b0 + i] + (f3 - f2) * h * (G[b1 + 1 + i] / G[b1]));
  const U = [0, 1, 2, 3].map((i) => G[a1 + i] + f * (G[b1 + i] - G[a1 + i]));
  const Amax = Math.max(G[tx(k, 3) + 3], G[tx(k + 1, 3) + 3]);
  const Amin = Math.min(G[tx(k, 4) + 3], G[tx(k + 1, 4) + 3]);
  const lerp3 = (r) => [0, 1, 2].map((i) => G[tx(k, r) + i] + f * (G[tx(k + 1, r) + i] - G[tx(k, r) + i]));
  const tau = G[a0 + 3] + f * (G[b0 + 3] - G[a0 + 3]);
  return { C, U, Amax, Amin, w: [lerp3(2), lerp3(3), lerp3(4)], tau };
}

function segSphereDist(a, b, c) {
  const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const L2 = Math.max(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2, 1e-300);
  const t = Math.min(1, Math.max(0, ((c[0] - a[0]) * ab[0] + (c[1] - a[1]) * ab[1] + (c[2] - a[2]) * ab[2]) / L2));
  return Math.hypot(a[0] + t * ab[0] - c[0], a[1] + t * ab[1] - c[1], a[2] + t * ab[2] - c[2]);
}

const dot3 = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];

function objStepLimit(snap, x, s, h, speed) {
  const reach = h * speed;
  const all = snap.all;
  if (Math.hypot(x[0] - all[0], x[1] - all[1], x[2] - all[2]) > all[3] + reach) return h;
  for (let j = 0; j < snap.count; j++) {
    const B = snap.bounds.subarray(4 * j, 4 * j + 4);
    if (B[3] < 0 || Math.hypot(x[0] - B[0], x[1] - B[1], x[2] - B[2]) > B[3] + reach) continue;
    const o = objPos(snap, j, s);
    if (!o) continue;
    const d = Math.hypot(x[0] - o.C[0], x[1] - o.C[1], x[2] - o.C[2]) - o.Amax;
    h = Math.min(h, (0.5 * Math.max(d, o.Amin)) / speed);
    if (d < o.Amax) h = Math.min(h, (0.04 * ksRadius(x[0], x[1], x[2], snap.aT ?? 0)) / speed);
  }
  return h;
}

/** Bodies crossed by the segment; returns hits [{ j, g, s, mu, alpha }] in order. */
function objSegment(snap, x0, x1, s0, s1, E, p, acc) {
  const all = snap.all;
  if (segSphereDist(x0, x1, all) > all[3]) return;
  const dseg = [x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]];
  const dsr = s1 - s0, sm = 0.5 * (s0 + s1);
  for (let j = 0; j < snap.count; j++) {
    const B = snap.bounds.subarray(4 * j, 4 * j + 4);
    if (B[3] < 0 || segSphereDist(x0, x1, B) > B[3]) continue;
    const o = objPos(snap, j, sm);
    if (!o) continue;
    const V = [o.U[1] / o.U[0], o.U[2] / o.U[0], o.U[3] / o.U[0]];
    const D0 = [0, 1, 2].map((i) => x0[i] - o.C[i] - V[i] * (s0 - sm));
    const dD = [0, 1, 2].map((i) => dseg[i] - V[i] * dsr);
    const P = o.w.map((w) => dot3(w, D0)), Q = o.w.map((w) => dot3(w, dD));
    const qa = dot3(Q, Q), qb = dot3(P, Q), qc = dot3(P, P) - 1;
    const disc = qb * qb - qa * qc;
    if (qa <= 0 || disc <= 0) continue;
    const sq = Math.sqrt(disc);
    const t1 = Math.max((-qb - sq) / qa, 0), t2 = Math.min((-qb + sq) / qa, 1);
    if (t2 <= t1) continue;
    const em = snap.emit.subarray(4 * j, 4 * j + 4);
    const chord = (t2 - t1) * Math.sqrt(qa);
    const alpha = 1 - Math.exp(-em[1] * chord);
    const g = -1 / (-E * o.U[0] + p[0] * o.U[1] + p[1] * o.U[2] + p[2] * o.U[3]);
    let mu = 1;
    if (t1 > 0) {
      const y = P.map((c, i) => c + t1 * Q[i]);
      mu = Math.min(1, Math.max(0, -dot3(y, Q) / (Math.hypot(...y) * Math.sqrt(qa))));
    }
    acc.hits.push({ j, g, s: s0 + t1 * dsr, mu, alpha, trans: acc.trans, tau: o.tau });
    acc.trans *= 1 - alpha;
  }
}

/**
 * traceRay with dynamic bodies: also integrates the mirror time s along the
 * ray (ds/dλ = E + f(E + k·p)) and records every body crossed.
 * Returns the usual result plus { hits, trans, sEnd }.
 */
export function traceRayObjects(state, E, aT, opts, snap) {
  const aPhys = opts.aPhys ?? -aT;
  const rtol = opts.rtol ?? 1e-10;
  const atol = opts.atol ?? 1e-12;
  const maxSteps = opts.maxSteps ?? 50000;
  const rEsc = opts.rEscape ?? 1e4;
  const rH = horizons(aT).rPlus;
  const diskOn = opts.disk !== false;
  const rIn = opts.rIn ?? iscoRadius(aPhys);
  const rOut = opts.rOut ?? 40;
  let s = state.slice();
  let k1 = geodesicRHS(s, E, aT, new Array(6));
  let sRay = 0;
  let h = 0.01 * Math.max(1, ksRadius(s[0], s[1], s[2], aT));
  const acc = { hits: [], trans: 1 };
  snap = { ...snap, aT };
  const done = (o) => ({ ...o, hits: acc.hits, trans: acc.trans, sEnd: sRay });
  for (let n = 0; n < maxSteps; n++) {
    const r = ksRadius(s[0], s[1], s[2], aT);
    h = Math.min(h, 0.5 * Math.max(r, 0.1));
    if (snap.count) {
      const speed = Math.hypot(k1[0], k1[1], k1[2]) + dtdl(s, E, aT);
      h = objStepLimit(snap, s, sRay, h, speed);
    }
    const stages = [];
    const st = dopriStep(s, k1, h, E, aT, rtol, atol, stages);
    if (st.err > 1 || !Number.isFinite(st.err)) {
      h *= Math.max(0.1, 0.9 * Math.pow(st.err || 1e10, -0.2));
      continue;
    }
    let ds = 0;
    for (let j = 0; j < 6; j++) if (B5[j]) ds += B5[j] * dtdl(stages[j], E, aT);
    const sNew = sRay + h * ds;
    const y = st.y;
    if (snap.count && acc.trans > 1e-4) objSegment(snap, s, y, sRay, sNew, E, [s[3], s[4], s[5]], acc);
    if (diskOn && s[2] * y[2] <= 0 && s[2] !== y[2]) {
      const hit = landOnPlane(Math.abs(s[2]) < Math.abs(y[2]) ? s : y, E, aT);
      const rho2 = hit[0] * hit[0] + hit[1] * hit[1] - aT * aT;
      const rd = Math.sqrt(Math.max(rho2, 0));
      if (rd >= rIn && rd <= rOut) {
        sRay = sNew;
        return done({ type: 'disk', state: hit, r: rd, ...diskShading(hit, E, aT, aPhys) });
      }
    }
    s = y;
    sRay = sNew;
    k1 = st.k7;
    const rn = ksRadius(s[0], s[1], s[2], aT);
    if (rn < rH * (1 - 1e-6)) return done({ type: 'horizon', state: s, r: rn });
    const vr = s[0] * k1[0] + s[1] * k1[1] + s[2] * k1[2];
    if (rn > rEsc && vr > 0) {
      const v = Math.hypot(k1[0], k1[1], k1[2]);
      return done({ type: 'sky', state: s, r: rn, dir: [k1[0] / v, k1[1] / v, k1[2] / v], g: 1 / E });
    }
    h *= Math.min(5, Math.max(0.2, 0.9 * Math.pow(st.err || 1e-10, -0.2)));
  }
  return done({ type: 'maxsteps', state: s });
}

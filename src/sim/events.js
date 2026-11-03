// Dynamic objects around the hole: test-body probes, self-gravitating stars
// (tidal disruption) and pressureless dust clouds.
//
// Everything is integrated in the *physical* spacetime (ingoing Kerr–Schild
// coordinates of Kerr(a), G = c = M = 1) and converted to the ray tracer's
// mirror spacetime only when a sample is emitted (spacetime.js: mirrorPoint).
//
// Each rendered body is an "entity": a sequence of samples along its world
// line. A sample describes an ellipsoid in the body's rest frame:
//   [t', C'x, C'y, C'z, Ũt, Ũx, Ũy, Ũz, w1(3), w2(3), w3(3), τ, Amax]
// with t' the mirror time, C' the mirror position, Ũ the (future-directed)
// mirror four-velocity and w_k the spatial parts of the covectors
// g'(ê_k)/A_k, so that a mirror displacement D at equal t' lies inside the
// body iff Σ_k (w_k·D)² ≤ 1 (exact for D ⟂ Ũ because ê_k ⟂ Ũ).
//
// The work is written as generators so it can run incrementally on the main
// thread or flat out in a Web Worker.

import { ksRadius, horizons, metricLower, lower, geodesicRHS, dtdl, hamiltonian } from '../physics/kerr.js';
import {
  christoffel, tidalTensor, symEig3, fourVelocity, triadFor, timelikeIntegrator, unpackTriad, reorthonormalise,
  stateFromConstants, orbitConstants, mirrorPoint, mirrorVector, gradR, physicalPoint, timeShift,
} from '../physics/spacetime.js';
import { Dopri5 } from '../physics/ode.js';
import { SPHStar } from '../physics/star.js';

export const SAMPLE = 19;

// Physical constants (SI, CODATA 2018 / IAU 2015).
export const UNITS = {
  GMsun_c2: 1476.6250614046494, // m
  GMsun_c3: 4.925490947641267e-6, // s
  Rsun: 6.957e8, // m
  yr: 3.15576e7, // s
};

/** Star catalogue (mass in M☉, radius in R☉, polytropic index, effective temperature). */
export const STAR_TYPES = {
  sun: { label: 'Sun-like star (1 M☉, 1 R☉)', mass: 1, radius: 1, n: 3, Teff: 5772 },
  redDwarf: { label: 'Red dwarf (0.3 M☉, 0.3 R☉)', mass: 0.3, radius: 0.3, n: 1.5, Teff: 3400 },
  blueGiant: { label: 'Hot massive star (10 M☉, 5 R☉)', mass: 10, radius: 5, n: 3, Teff: 25000 },
  redGiant: { label: 'Red giant envelope (1.4 M☉, 10 R☉)', mass: 1.4, radius: 10, n: 1.5, Teff: 4300 },
  whiteDwarf: { label: 'White dwarf (0.6 M☉, 0.012 R☉)', mass: 0.6, radius: 0.012, n: 1.5, Teff: 20000 },
};

/** Star size and tidal scales in units of the hole mass M. */
export function starScales(star, massSolar) {
  const R = (star.radius * UNITS.Rsun) / (UNITS.GMsun_c2 * massSolar); // R*/M
  const m = star.mass / massSolar; // M*/M
  return { R, m, tStar: Math.sqrt((R * R * R) / m), rt: R * Math.cbrt(1 / m) };
}

// ---------- Sample construction ----------

/**
 * Mirror-space ellipsoid sample from a physical event.
 * axes: three orthonormal spatial vectors e_k (⟂ u) with semi-axes A_k (M).
 */
export function makeSample(a, t, pos, u, axes, A, tau) {
  const m = mirrorPoint(t, pos[0], pos[1], pos[2], a);
  const U = mirrorVector(pos[0], pos[1], pos[2], a, u).map((c) => -c);
  const gm = metricLower([m.x, m.y, m.z], -a);
  const out = new Float64Array(SAMPLE);
  out[0] = m.tp; out[1] = m.x; out[2] = m.y; out[3] = m.z;
  out.set(U, 4);
  for (let k = 0; k < 3; k++) {
    const e = mirrorVector(pos[0], pos[1], pos[2], a, axes[k]);
    const w = lower(gm, e);
    out[8 + 3 * k] = w[1] / A[k];
    out[9 + 3 * k] = w[2] / A[k];
    out[10 + 3 * k] = w[3] / A[k];
  }
  out[17] = tau;
  out[18] = Math.max(A[0], A[1], A[2]);
  return out;
}

/** Growable sample buffer for one entity. */
export class Entity {
  constructor(id, kind, meta) {
    this.id = id;
    this.kind = kind;       // 'probe' | 'star' | 'core' | 'chunk'
    this.meta = meta;       // { Trest, opacity, blink, limb, label, massFraction }
    this.data = new Float64Array(SAMPLE * 64);
    this.count = 0;
    this.sent = 0;
    this.ended = false;
    this.info = [];         // HUD rows (physical quantities), sparse
  }
  push(sample) {
    // Keep t' strictly decreasing (physical time increasing); drop duplicates.
    if (this.count && !(sample[0] < this.data[(this.count - 1) * SAMPLE])) return;
    if ((this.count + 1) * SAMPLE > this.data.length) {
      const d = new Float64Array(this.data.length * 2);
      d.set(this.data);
      this.data = d;
    }
    this.data.set(sample, this.count * SAMPLE);
    this.count++;
  }
}

// ---------- Helpers ----------

/** dt'/dτ along a physical world line (t' = −t + 2∫2r/Δ dr). */
function dtpDtau(pos, u, a) {
  const { r, g } = gradR(pos[0], pos[1], pos[2], a);
  const D = r * r - 2 * r + a * a;
  const drdt = g[0] * u[1] + g[1] * u[2] + g[2] * u[3];
  return -u[0] + ((4 * r) / D) * drdt;
}

/** Step cap for a world line: resolves the mirror time and the orbit. */
function stepCap(pos, u, a, dtpMax = 0.5) {
  const r = ksRadius(pos[0], pos[1], pos[2], a);
  const rate = Math.abs(dtpDtau(pos, u, a));
  return Math.min(dtpMax / Math.max(rate, 1e-12), 0.05 * Math.pow(Math.max(r, 1), 1.5), 0.5 * r);
}

function ySplit(Y) {
  return { t: Y[0], s: [Y[1], Y[2], Y[3], Y[4], Y[5], Y[6]], pos: [Y[1], Y[2], Y[3]] };
}

/** Start time so that the first light from r0 reaches the camera now. */
export function retardedStart(tObs, rCam, r0, a) {
  return tObs - 2 * timeShift(rCam, a) + 2 * timeShift(r0, a) - (rCam - r0);
}

/** Initial physical state of an orbit with pericentre azimuth phiPeri. */
export function launchState({ a, r0, rp, E = 1, incl = 0, phiPeri = 0, plunge = false }) {
  const C = plunge ? { E, Lz: 0, Q: 0 } : orbitConstants({ E, rp, incl, a, rStart: r0 });
  // Start on the equator moving inwards; find where the pericentre lands and rotate.
  const trial = stateFromConstants(r0, Math.PI / 2, 0, C.E, C.Lz, C.Q, a, -1, -1);
  let phiAtPeri = 0;
  if (!plunge && C.Lz !== 0) {
    const I = timelikeIntegrator(0, trial.s, C.E, a, { rtol: 1e-9, atol: 1e-11 });
    let prevR = r0;
    for (let k = 0; k < 20000; k++) {
      I.step(stepCap([I.y[1], I.y[2], I.y[3]], fourVelocity(ySplit(I.y).s, C.E, a), a, 5));
      const r = ksRadius(I.y[1], I.y[2], I.y[3], a);
      if (r > prevR || r < horizons(a).rPlus * 1.01) { phiAtPeri = Math.atan2(I.y[2], I.y[1]); break; }
      prevR = r;
    }
  }
  const st = stateFromConstants(r0, Math.PI / 2, phiPeri - phiAtPeri, C.E, C.Lz, C.Q, a, -1, -1);
  return { ...st, C };
}

// ---------- Probe: a single timelike geodesic with a transported frame ----------

/**
 * A small glowing test body (no self-gravity, negligible size) on a Kerr
 * geodesic, carrying a clock (blinks every `blink` units of proper time) and
 * a parallel-transported frame in which the tidal tensor is evaluated.
 */
export function* probeEvent(cfg, emit) {
  const { a, t0, radius = 0.25, tMax = 4000 } = cfg;
  const L = launchState(cfg);
  const E = L.E;
  const u0 = fourVelocity(L.s, E, a);
  const pos0 = L.s.slice(0, 3);
  const I = timelikeIntegrator(t0, L.s, E, a, { triad: triadFor(pos0, a, u0), rtol: 1e-10, atol: 1e-12 });
  const ent = emit.entity('probe', { Trest: cfg.Trest ?? 9000, opacity: 40, blink: cfg.blink ?? 10, limb: 0, label: cfg.label ?? 'Probe' });
  const rH = horizons(a).rPlus;
  let lastInfo = -Infinity;
  for (let k = 0; ; k++) {
    const { t, s, pos } = ySplit(I.y);
    const u = fourVelocity(s, E, a);
    const triad = unpackTriad(I.y);
    const r = ksRadius(pos[0], pos[1], pos[2], a);
    if (r <= rH * (1 + 1e-7) || I.lambda > tMax || (r > cfg.r0 * 1.6)) break;
    ent.push(makeSample(a, t, pos, u, triad, [radius, radius, radius], I.lambda));
    if (I.lambda - lastInfo > 0.5) {
      const T = tidalTensor(pos, a, u, reorthonormalise(pos, a, u, triad));
      const ev = symEig3(T).values;
      ent.info.push({ t, tau: I.lambda, r, dtdtau: u[0], tidal: ev, E, Lz: L.C.Lz, Q: L.C.Q });
      lastInfo = I.lambda;
    }
    I.step(stepCap(pos, u, a));
    if (k % 20 === 19) yield;
  }
  ent.ended = true;
}

// ---------- Ballistic world line with frozen shape (debris plunge) ----------

function* ballistic(a, t0, s, E, axes, A, tau0, ent, tEnd) {
  const rH = horizons(a).rPlus;
  const pos = s.slice(0, 3);
  const u = fourVelocity(s, E, a);
  const I = timelikeIntegrator(t0, s, E, a, { triad: reorthonormalise(pos, a, u, axes), rtol: 1e-9, atol: 1e-11 });
  for (let k = 0; ; k++) {
    const { t, s: st, pos: p } = ySplit(I.y);
    const uu = fourVelocity(st, E, a);
    const r = ksRadius(p[0], p[1], p[2], a);
    if (r <= rH * (1 + 1e-7) || t > tEnd) break;
    ent.push(makeSample(a, t, p, uu, unpackTriad(I.y), A, tau0 + I.lambda));
    I.step(stepCap(p, uu, a));
    if (k % 50 === 49) yield;
  }
  ent.ended = true;
}

// ---------- Fermi normal coordinates ↔ global coordinates ----------

/**
 * Global (x^μ, p_μ) of a point at Fermi position ξ (M units) and velocity v
 * (units of c) relative to the centre of mass at X with four-velocity u and
 * triad e_i. Second-order exponential map: x = X + ξ − ½Γξξ.
 */
export function fermiToGlobal(a, X, u, triad, xi, v) {
  const G = christoffel(X.slice(1), a);
  const d = [0, 1, 2, 3].map((m) => xi[0] * triad[0][m] + xi[1] * triad[1][m] + xi[2] * triad[2][m]);
  const x = [0, 1, 2, 3].map((m) => {
    let q = 0;
    for (let al = 0; al < 4; al++) for (let be = 0; be < 4; be++) q += G[m][al][be] * d[al] * d[be];
    return X[m] + d[m] - 0.5 * q;
  });
  const v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const gam = 1 / Math.sqrt(1 - v2);
  let U = [0, 1, 2, 3].map((m) => gam * (u[m] + v[0] * triad[0][m] + v[1] * triad[1][m] + v[2] * triad[2][m]));
  // Parallel-transport the velocity from X to x along ξ.
  U = U.map((c, m) => {
    let q = 0;
    for (let al = 0; al < 4; al++) for (let be = 0; be < 4; be++) q += G[m][al][be] * d[al] * U[be];
    return c - q;
  });
  const g = metricLower(x.slice(1), a);
  let n = 0;
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) n += g[i][j] * U[i] * U[j];
  U = U.map((c) => c / Math.sqrt(-n));
  const p = lower(g, U);
  return { t: x[0], s: [x[1], x[2], x[3], p[1], p[2], p[3]], E: -p[0] };
}

// ---------- Debris: many geodesics sampled at common coordinate times ----------

class DebrisParticle {
  constructor(a, t, s, E) {
    this.E = E;
    this.alive = true;
    const f = (Y, out) => {
      const st = [Y[1], Y[2], Y[3], Y[4], Y[5], Y[6]];
      const d = geodesicRHS(st, E, a, new Array(6));
      out[0] = dtdl(st, E, a);
      for (let i = 0; i < 6; i++) out[i + 1] = d[i];
    };
    this.I = new Dopri5(f, [t, ...s], { rtol: 1e-9, atol: 1e-11, h0: 0.1 });
  }
  /** Advance to coordinate time T and return [x,y,z,p_x,p_y,p_z] there. */
  advanceTo(T, a, rH) {
    const I = this.I;
    while (I.y[0] < T) {
      const r = ksRadius(I.y[1], I.y[2], I.y[3], a);
      if (r < rH * 0.98) { this.alive = false; return null; }
      I.step(Math.max(0.05, 0.1 * r));
    }
    // Invert t(θ) within the last step (monotonic): secant/bisection.
    let lo = 0, hi = 1;
    if (I.prevY) {
      for (let k = 0; k < 40; k++) {
        const m = 0.5 * (lo + hi);
        if (I.interpolate(m)[0] < T) lo = m; else hi = m;
      }
      const y = I.interpolate(0.5 * (lo + hi));
      return [y[1], y[2], y[3], y[4], y[5], y[6]];
    }
    return [I.y[1], I.y[2], I.y[3], I.y[4], I.y[5], I.y[6]];
  }
}

/** Ellipsoid of a group of simultaneous particles: centre, mean u, axes. */
function groupEllipsoid(a, states, Es, minAxis, scale) {
  const n = states.length;
  const C = [0, 0, 0];
  let U = [0, 0, 0, 0];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) C[k] += states[i][k] / n;
    const u = fourVelocity(states[i], Es[i], a);
    U = U.map((c, k) => c + u[k] / n);
  }
  const g = metricLower(C, a);
  let nn = 0;
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) nn += g[i][j] * U[i] * U[j];
  U = U.map((c) => c / Math.sqrt(-nn));
  const tri = triadFor(C, a, U);
  const lu = lower(g, U);
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const lt = tri.map((e) => lower(g, e));
  for (let i = 0; i < n; i++) {
    const D = [0, states[i][0] - C[0], states[i][1] - C[1], states[i][2] - C[2]];
    const du = lu[1] * D[1] + lu[2] * D[2] + lu[3] * D[3];
    const xi = D.map((c, k) => c + du * U[k]); // project ⟂ U
    const comp = lt.map((w) => w[0] * xi[0] + w[1] * xi[1] + w[2] * xi[2] + w[3] * xi[3]);
    for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) M[p][q] += (comp[p] * comp[q]) / n;
  }
  const { values, vectors } = symEig3(M);
  const axes = vectors.map((v) => [0, 1, 2, 3].map((m) => v[0] * tri[0][m] + v[1] * tri[1][m] + v[2] * tri[2][m]));
  const A = values.map((l) => Math.max(scale * Math.sqrt(Math.max(l, 0)), minAxis));
  return { C, U, axes, A };
}

/**
 * Integrate free debris particles, emitting one ellipsoid entity per group
 * of particles with similar orbital energy (they form contiguous pieces of
 * the stream). Groups that plunge are continued as ballistic bodies down to
 * the horizon so their fading and freezing are resolved.
 */
function* debrisPhase(a, particles, opts, emit, stats) {
  const { tStart, tEnd, nGroups, minAxis, Trest, opacity, massPerParticle, label } = opts;
  const rH = horizons(a).rPlus;
  const order = particles.map((p, i) => i).sort((i, j) => particles[i].E - particles[j].E);
  const groups = [];
  const per = Math.ceil(order.length / nGroups);
  for (let gI = 0; gI * per < order.length; gI++) {
    const members = order.slice(gI * per, (gI + 1) * per);
    groups.push({
      members,
      ent: emit.entity('chunk', { Trest, opacity, blink: 0, limb: 0, label, massFraction: members.length * massPerParticle }),
      done: false,
    });
  }
  const subs = [];
  let T = tStart;
  let rMed = 10;
  while (T <= tEnd) {
    const states = new Map();
    for (const i of order) {
      const p = particles[i];
      if (!p.alive) continue;
      const s = p.advanceTo(T, a, rH);
      if (!s) { stats.swallowed += massPerParticle; continue; }
      if (ksRadius(s[0], s[1], s[2], a) < rH) { p.alive = false; stats.swallowed += massPerParticle; continue; }
      states.set(i, s);
    }
    const radii = [];
    for (const g of groups) {
      if (g.done) continue;
      const st = [], Es = [];
      for (const i of g.members) if (states.has(i)) { st.push(states.get(i)); Es.push(particles[i].E); }
      if (st.length < 3) { g.done = true; g.ent.ended = true; continue; }
      const ell = groupEllipsoid(a, st, Es, minAxis, Math.sqrt(5));
      const r = ksRadius(ell.C[0], ell.C[1], ell.C[2], a);
      radii.push(r);
      const vr = gradR(ell.C[0], ell.C[1], ell.C[2], a).g.reduce((acc, c, k) => acc + c * ell.U[k + 1], 0);
      if (r < 2.5 * rH && vr < 0) {
        // Plunging piece: continue its centroid as a ballistic body with frozen shape.
        const g4 = metricLower(ell.C, a);
        const pl = lower(g4, ell.U);
        const s0 = [ell.C[0], ell.C[1], ell.C[2], pl[1], pl[2], pl[3]];
        subs.push(ballistic(a, T, s0, -pl[0], ell.axes, ell.A, 0, g.ent, tEnd));
        g.done = true;
        continue;
      }
      g.ent.push(makeSample(a, T, ell.C, ell.U, ell.axes, ell.A, 0));
    }
    for (const sub of subs) sub.next();
    if (radii.length) { radii.sort((x, y) => x - y); rMed = radii[radii.length >> 1]; }
    if (!groups.some((g) => !g.done)) break;
    T += Math.min(8, Math.max(0.4, 0.02 * rMed));
    yield;
  }
  // Finish plunging pieces.
  for (const sub of subs) for (const _ of sub) yield;
  for (const g of groups) g.ent.ended = true;
  // Particles still alive at the end; those inside 3 r₊ moving inwards are counted as swallowed.
  stats.survivorEps = [];
  for (const p of particles) {
    if (!p.alive) continue;
    const r = ksRadius(p.I.y[1], p.I.y[2], p.I.y[3], a);
    if (r < 3 * rH) { stats.swallowed += massPerParticle; continue; }
    stats.survivorEps.push(p.E - 1);
  }
  if (opts.finish) opts.finish(stats);
}

// ---------- Fallback rate of bound debris ----------

/**
 * Mass return rate from the specific orbital energies ε = E − 1 of the
 * debris (frozen after the encounter): a particle with ε < 0 returns to
 * pericentre after one Keplerian period 2π (−2ε)^{−3/2} M (its apocentre is
 * ≳ 10³ M, where the orbit is Newtonian to high accuracy).
 */
export function fallbackRate(eps, massPerParticle, starMass, massSolar, nBins = 40) {
  const bound = eps.filter((e) => e < 0).map((e) => 2 * Math.PI * Math.pow(-2 * e, -1.5));
  if (bound.length < 5) return null;
  const tSec = bound.map((t) => t * UNITS.GMsun_c3 * massSolar);
  const lo = Math.log10(Math.min(...tSec)), hi = Math.log10(Math.max(...tSec)) + 1e-9;
  const edges = Array.from({ length: nBins + 1 }, (_, i) => Math.pow(10, lo + ((hi - lo) * i) / nBins));
  const counts = new Array(nBins).fill(0);
  for (const t of tSec) counts[Math.min(nBins - 1, Math.floor(((Math.log10(t) - lo) / (hi - lo)) * nBins))]++;
  const MsunPerYr = (n, dt) => (n * massPerParticle * starMass) / (dt / UNITS.yr);
  const t = [], mdot = [];
  for (let i = 0; i < nBins; i++) {
    if (!counts[i]) continue;
    t.push(Math.sqrt(edges[i] * edges[i + 1]));
    mdot.push(MsunPerYr(counts[i], edges[i + 1] - edges[i]));
  }
  // Eddington accretion rate for η = 0.1 (L_Edd = 1.257e31 W per M☉).
  const mdotEdd = (1.2572e31 * massSolar) / (0.1 * 299792458 ** 2) / 1.98847e30 * UNITS.yr;
  return { t, mdot, mdotEdd, tMin: Math.min(...tSec) };
}

// ---------- Star (SPH in Fermi normal coordinates) ----------

const relaxedCache = new Map();
/** Relaxed polytrope (cached per index and resolution; stars are self-similar). */
export function relaxedStar(n, N) {
  const key = `${n}:${N}`;
  if (!relaxedCache.has(key)) {
    const s = SPHStar.polytrope({ N, n });
    s.relax(3, 0.05);
    let r2 = 0;
    for (let i = 0; i < N; i++) r2 += s.pos[3 * i] ** 2 + s.pos[3 * i + 1] ** 2 + s.pos[3 * i + 2] ** 2;
    s.rms = Math.sqrt(r2 / N);
    relaxedCache.set(key, s);
  }
  const src = relaxedCache.get(key);
  const s = new SPHStar({ N, gamma: src.gamma });
  for (const k of ['pos', 'vel', 'acc', 'A', 'rho', 'h', 'divv', 'curlv', 'phi']) s[k].set(src[k]);
  s.cs = src.cs.slice();
  s.vsig = src.vsig.slice();
  s.rms = src.rms;
  return s;
}

/**
 * Tidal encounter of a star with the hole.
 * cfg: { a, massSolar, star (STAR_TYPES entry), beta | rp, incl, E, phiPeri, t0, N, r0 }
 * The star's centre of mass follows a Kerr geodesic carrying a parallel-
 * transported tetrad; the gas is evolved with SPH in the Fermi normal
 * coordinates of that geodesic, with the exact Kerr tidal tensor
 * C_ij = R_{0i0j} (quadrupole order: valid while the star is small compared
 * with r). After pericentre, unbound gas is converted to independent
 * geodesics (the "frozen-in" ballistic phase of a tidal disruption event)
 * while any self-bound core continues in the Fermi frame.
 */
export function* starEvent(cfg, emit) {
  const { a, massSolar, star, N = 800, t0 } = cfg;
  const sc = starScales(star, massSolar);
  const rH = horizons(a).rPlus;
  const rp = cfg.rp ?? sc.rt / (cfg.beta ?? 1);
  const r0 = cfg.r0 ?? Math.max(3 * sc.rt, 3 * rp, 40);
  const stats = {
    rt: sc.rt, rp, r0, Rstar: sc.R, mStar: sc.m, tStar: sc.tStar, beta: sc.rt / rp, swallowed: 0, bound: 0, unbound: 0,
    coreFraction: 0, outcome: 'running', maxStretch: 1,
  };
  emit.stats(stats);
  const sph = relaxedStar(star.n, N);
  yield;
  const L = launchState({ a, r0, rp, E: cfg.E ?? 1, incl: cfg.incl ?? 0, phiPeri: cfg.phiPeri ?? 0, plunge: cfg.plunge });
  const E = L.E;
  stats.plunge = !!(cfg.plunge || L.C.plunge);
  const pos0 = L.s.slice(0, 3);
  const u0 = fourVelocity(L.s, E, a);
  const cm = timelikeIntegrator(t0, L.s, E, a, { triad: triadFor(pos0, a, u0), rtol: 1e-10, atol: 1e-12 });
  const tS = sc.tStar;
  const Rm = sc.R; // star radius in M
  const vScale = Math.sqrt(sc.m / sc.R); // star velocity unit in c
  const scaleCal = 1 / (sph.rms / Math.sqrt(3)); // ellipsoid calibration: initial star → R*
  const ent = emit.entity('star', { Trest: star.Teff, opacity: 40, blink: 0, limb: 1, label: star.label });
  const cmState = () => {
    const { t, s, pos } = ySplit(cm.y);
    const u = fourVelocity(s, E, a);
    return { t, s, pos, u, triad: reorthonormalise(pos, a, u, unpackTriad(cm.y)) };
  };
  const tidalStar = () => {
    const c = cmState();
    return tidalTensor(c.pos, a, c.u, c.triad).map((row) => row.map((v) => v * tS * tS));
  };
  const shapeSample = (c, mask) => {
    const mo = sph.moments(mask);
    const { values, vectors } = symEig3(mo.M);
    const A = values.map((l) => Math.max(scaleCal * Math.sqrt(Math.max(l, 0)), 0.02) * Rm);
    stats.maxStretch = Math.max(stats.maxStretch, Math.sqrt(values[2] / Math.max(values[0], 1e-12)));
    // Principal axes as components in the Fermi triad (rebuilt on the current triad when sampled).
    return { comps: vectors, A, offset: mo.centre };
  };
  const inFrame = (comps, triad) => comps.map((v) => [0, 1, 2, 3].map((m) => v[0] * triad[0][m] + v[1] * triad[1][m] + v[2] * triad[2][m]));
  sph.tidal = tidalStar();
  sph.computeForces();

  let passedPeri = false, prevR = r0, minR = r0, mask = sph.alive, phase = 'fnc', coreSettle = Infinity;
  const rSwitch = Math.max(2 * rp, 4 * sc.rt);
  let lastInfo = -Infinity;
  for (let iter = 0; ; iter++) {
    const c = cmState();
    const r = ksRadius(c.pos[0], c.pos[1], c.pos[2], a);
    if (r <= rH * (1 + 1e-7)) {
      if (phase === 'fnc' || stats.coreFraction > 0.97) stats.outcome = 'swallowed whole';
      else stats.outcome = 'partial disruption, core swallowed';
      stats.swallowed = phase === 'fnc' ? 1 : stats.swallowed + stats.coreFraction;
      break;
    }
    // Outbound after pericentre (the star must first have moved inwards).
    minR = Math.min(minR, r);
    passedPeri = minR < 0.98 * r0 && r > minR * (1 + 1e-6) && r >= prevR;
    prevR = r;
    // Validity of the quadrupole Fermi expansion: star extent ≪ r.
    let ext = 0;
    for (let i = 0; i < N; i++) if (mask[i]) ext = Math.max(ext, Math.hypot(sph.pos[3 * i], sph.pos[3 * i + 1], sph.pos[3 * i + 2]));
    const extentM = ext * Rm;
        const tooBig = extentM > 0.2 * r;
    if (phase === 'fnc' && ((passedPeri && r > rSwitch) || tooBig)) {
      // ----- Switch: bound core stays in the Fermi frame, the rest becomes debris.
      // If the gas outgrows the Fermi patch before it has left pericentre, the
      // whole star is released ballistically ("frozen-in" approximation). -----
      const core = tooBig && !passedPeri ? { fraction: 0, mask: new Uint8Array(N) } : sph.boundCore();
      const coreMask = core.fraction > 0.03 ? core.mask : new Uint8Array(N);
      const parts = [];
      const eps = [];
      for (let i = 0; i < N; i++) {
        if (!sph.alive[i] || coreMask[i]) continue;
        const xi = [sph.pos[3 * i] * Rm, sph.pos[3 * i + 1] * Rm, sph.pos[3 * i + 2] * Rm];
        const v = [sph.vel[3 * i] * vScale, sph.vel[3 * i + 1] * vScale, sph.vel[3 * i + 2] * vScale];
        const X = [c.t, ...c.pos];
        const gs = fermiToGlobal(a, X, c.u, c.triad, xi, v);
        parts.push(new DebrisParticle(a, gs.t, gs.s, gs.E));
        eps.push(gs.E - 1);
        sph.alive[i] = 0;
      }
      stats.coreFraction = core.fraction > 0.03 ? core.fraction : 0;
      stats.bound = eps.filter((e) => e < 0).length / N;
      stats.unbound = eps.filter((e) => e >= 0).length / N;
      stats.outcome = stats.coreFraction > 0 ? (stats.coreFraction > 0.97 ? 'survived (no mass loss)' : 'partial disruption') : 'full disruption';
      stats.fallback = fallbackRate(eps, 1 / N, star.mass, massSolar);
      stats.epsSpread = eps.length ? spread(eps) : null;
      stats.deltaEps = (sc.R / (sc.rt * sc.rt)); // frozen-in estimate GM R*/r_t² (in c²)
      emit.stats(stats);
      if (parts.length) {
        const tEnd = c.t + Math.min(Math.max(3 * (c.t - t0), 1500), 12000);
        emit.spawn(debrisPhase(a, parts, {
          tStart: Math.max(...parts.map((p) => p.I.y[0])), tEnd, nGroups: Math.min(40, Math.max(1, Math.floor(parts.length / 10))),
          minAxis: 0.15 * Rm, Trest: star.Teff, opacity: 6, massPerParticle: 1 / N, label: 'Debris stream',
          finish: (st) => {
            const surv = st.survivorEps;
            st.bound = surv.filter((e) => e < 0).length / N;
            st.unbound = surv.filter((e) => e >= 0).length / N;
            st.fallback = fallbackRate(surv, 1 / N, star.mass, massSolar);
            if (st.swallowed > 0.95) st.outcome = 'torn apart inside the capture radius and swallowed';
            delete st.survivorEps;
            emit.stats(st);
          },
        }, emit, stats));
      }
      if (!stats.coreFraction) break;
      phase = 'core';
      mask = coreMask;
      ent.meta.label = 'Surviving core';
      sph.computeDensity();
      sph.computeForces();
      cfg.tEndCore = c.t + Math.min(Math.max(3 * (c.t - t0), 1500), 12000);
      coreSettle = sph.t + 5;
    }
    if (phase === 'core' && c.t > cfg.tEndCore) break;

    // ----- One SPH step in the Fermi frame, with the centre of mass advanced alongside -----
    const Cnow = sph.tidal;
    let cmax = 0;
    for (const row of Cnow) for (const v of row) cmax = Math.max(cmax, Math.abs(v));
    // While the tide is weak (|C| ≪ GM*/R*³) the star is in hydrostatic
    // equilibrium and responds quasi-statically (δR/R ~ |C|): hold the gas
    // and move only the centre of mass. The surviving core is also held
    // after it has had 5 t* to settle.
    const quiet = (cmax < 2e-3 && phase === 'fnc' && !passedPeri) || (phase === 'core' && sph.t > coreSettle);
    const dtOrbit = (0.02 * Math.pow(r, 1.5)) / tS;
    const dtS = quiet ? dtOrbit : Math.min(sph.timestep(), 0.1 / Math.sqrt(cmax + 1e-30), dtOrbit);
    const tauTarget = cm.lambda + dtS * tS;
    // Advance the centre of mass, emitting samples with the current shape.
    const shape = shapeSample(c, mask);
    while (cm.lambda < tauTarget) {
      const cc = cmState();
      const rr = ksRadius(cc.pos[0], cc.pos[1], cc.pos[2], a);
      if (rr <= rH * (1 + 1e-7)) break;
      const off = shape.offset.map((v) => v * Rm);
      const P = [0, 1, 2].map((k) => cc.pos[k] + off[0] * cc.triad[0][k + 1] + off[1] * cc.triad[1][k + 1] + off[2] * cc.triad[2][k + 1]);
      ent.push(makeSample(a, cc.t, P, cc.u, inFrame(shape.comps, cc.triad), shape.A, cm.lambda));
      if (cm.lambda - lastInfo > 1) {
        const T = tidalTensor(cc.pos, a, cc.u, cc.triad);
        ent.info.push({ t: cc.t, tau: cm.lambda, r: rr, dtdtau: cc.u[0], tidal: symEig3(T).values, stretch: stats.maxStretch });
        lastInfo = cm.lambda;
      }
      cm.step(Math.min(stepCap(cc.pos, cc.u, a), tauTarget - cm.lambda + 1e-12));
    }
    if (quiet) { sph.t += dtS; sph.tidal = tidalStar(); }
    else sph.step(dtS, () => tidalStar());
    if (iter % 2 === 1) yield;
  }
  ent.ended = true;
  if (stats.outcome === 'running') stats.outcome = 'survived';
  emit.stats(stats);
}

function spread(arr) {
  const s = arr.slice().sort((x, y) => x - y);
  const q = (f) => s[Math.min(s.length - 1, Math.floor(f * s.length))];
  return { p05: q(0.05), p50: q(0.5), p95: q(0.95) };
}

/**
 * A pressureless, non-self-gravitating dust cloud (radius R in M): every
 * grain is an independent Kerr geodesic, so its deformation is the exact
 * (all-order) tidal field, with no hydrodynamic approximation at all.
 */
export function* dustEvent(cfg, emit) {
  const { a, t0, radius = 1.5, N = 400, Trest = 7000 } = cfg;
  const L = launchState(cfg);
  const pos0 = L.s.slice(0, 3);
  const u0 = fourVelocity(L.s, L.E, a);
  const triad = triadFor(pos0, a, u0);
  const rand = mulberry(11);
  const parts = [];
  while (parts.length < N) {
    const xi = [2 * rand() - 1, 2 * rand() - 1, 2 * rand() - 1];
    if (xi[0] ** 2 + xi[1] ** 2 + xi[2] ** 2 > 1) continue;
    const gs = fermiToGlobal(a, [t0, ...pos0], u0, triad, xi.map((v) => v * radius), [0, 0, 0]);
    parts.push(new DebrisParticle(a, gs.t, gs.s, gs.E));
  }
  const stats = { outcome: 'running', swallowed: 0 };
  // Grains start at slightly different coordinate times (Fermi simultaneity); begin sampling once all exist.
  const tStart = Math.max(...parts.map((p) => p.I.y[0]));
  yield* debrisPhase(a, parts, {
    tStart, tEnd: t0 + (cfg.tMax ?? 4000), nGroups: Math.min(24, Math.floor(N / 12)), minAxis: 0.08 * radius,
    Trest, opacity: 4, massPerParticle: 1 / N, label: 'Dust cloud',
  }, emit, stats);
  stats.outcome = 'done';
  emit.stats(stats);
}

function mulberry(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- Scheduler ----------

/**
 * Runs event generators cooperatively. `advance(ms)` does up to `ms` of work
 * and returns the new samples per entity since the last call.
 */
export class Simulation {
  constructor() {
    this.entities = [];
    this.tasks = [];
    this.events = [];
    this.nextId = 1;
    this.errors = [];
  }

  add(kind, cfg) {
    const ev = { id: this.events.length + 1, kind, cfg, stats: {}, entities: [] };
    const emit = {
      entity: (k, meta) => {
        const e = new Entity(this.nextId++, k, { ...meta, event: ev.id });
        this.entities.push(e);
        ev.entities.push(e.id);
        return e;
      },
      stats: (s) => { ev.stats = { ...s }; },
      spawn: (gen) => this.tasks.push(gen),
    };
    const gen = kind === 'star' ? starEvent(cfg, emit) : kind === 'dust' ? dustEvent(cfg, emit) : probeEvent(cfg, emit);
    this.tasks.push(gen);
    this.events.push(ev);
    return ev.id;
  }

  get busy() {
    return this.tasks.length > 0;
  }

  advance(ms = 8) {
    const t0 = performance.now();
    while (this.tasks.length && performance.now() - t0 < ms) {
      for (let i = 0; i < this.tasks.length; i++) {
        let r;
        try {
          r = this.tasks[i].next();
        } catch (err) {
          // A launch that has no valid orbit (e.g. a bound orbit starting beyond apocentre) ends that task only.
          this.errors.push(err.message ?? String(err));
          r = { done: true };
        }
        if (r.done) { this.tasks.splice(i, 1); i--; }
        if (performance.now() - t0 >= ms) break;
      }
    }
    return this.drain();
  }

  /** New samples since the last drain: [{ id, kind, meta, samples, ended, info }]. */
  drain() {
    const out = [];
    for (const e of this.entities) {
      if (e.count === e.sent && !(e.ended && !e.reportedEnd)) continue;
      out.push({ id: e.id, kind: e.kind, meta: e.meta, samples: e.data.slice(e.sent * SAMPLE, e.count * SAMPLE), ended: e.ended, info: e.info.splice(0) });
      e.sent = e.count;
      if (e.ended) e.reportedEnd = true;
    }
    return { entities: out, events: this.events.map((ev) => ({ id: ev.id, kind: ev.kind, stats: ev.stats })), busy: this.busy, errors: this.errors.splice(0) };
  }
}

export { physicalPoint, hamiltonian };

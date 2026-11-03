// Timelike geodesics, parallel transport, curvature and the mirror map,
// checked against analytic results and against independent numerics.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { metricLower, metricUpper, ksRadius, hamiltonian, angularMomentum, horizons, dot, lower, geodesicRHS, fromSpheroidal } from '../src/physics/kerr.js';
import {
  christoffel, riemannCoord, tidalTensor, symEig3, canonicalFrame, fourVelocity, triadFor, timelikeIntegrator,
  unpackTriad, stateFromConstants, carterQTimelike, pericentre, orbitConstants, radialPotential, polarPotential,
  mirrorPoint, mirrorVector, physicalPoint, timeShift,
} from '../src/physics/spacetime.js';
import { circularOrbitEnergy, keplerOmega, iscoRadius } from '../src/physics/disk.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} |${a} - ${b}| = ${Math.abs(a - b)} > ${tol}`);
const POINTS = [[3.1, -1.7, 2.2], [7, 2, -0.5], [1.2, 0.9, 0.6], [0.4, 1.6, -0.3], [12, -9, 4]];
const SPINS = [0, 0.7, -0.95];

test('analytic Christoffel symbols match finite differences of the metric', () => {
  const h = 1e-5;
  for (const a of SPINS) for (const pos of POINTS) {
    const G = christoffel(pos, a), gi = metricUpper(pos, a);
    const dg = [null];
    for (let j = 0; j < 3; j++) {
      const p1 = pos.slice(), p2 = pos.slice();
      p1[j] += h; p2[j] -= h;
      const g1 = metricLower(p1, a), g2 = metricLower(p2, a);
      dg.push(g1.map((r, i) => r.map((c, k) => (c - g2[i][k]) / (2 * h))));
    }
    const d = (al, m, n) => (al === 0 ? 0 : dg[al][m][n]);
    for (let mu = 0; mu < 4; mu++) for (let al = 0; al < 4; al++) for (let be = 0; be < 4; be++) {
      let s = 0;
      for (let la = 0; la < 4; la++) s += 0.5 * gi[mu][la] * (d(al, la, be) + d(be, la, al) - d(la, al, be));
      close(G[mu][al][be], s, 1e-8 * (1 + Math.abs(s)), `a=${a} pos=${pos} Γ^${mu}_${al}${be}`);
    }
  }
});

test('type-D Riemann tensor (from Ψ₂) equals the Riemann tensor of the Christoffel symbols, also inside the horizon', () => {
  const h = 1e-5;
  for (const a of SPINS) for (const pos of POINTS) {
    const G = christoffel(pos, a), g = metricLower(pos, a);
    const DG = [null];
    for (let j = 0; j < 3; j++) {
      const p1 = pos.slice(), p2 = pos.slice();
      p1[j] += h; p2[j] -= h;
      const A = christoffel(p1, a), B = christoffel(p2, a);
      DG.push(A.map((m, i) => m.map((r, k) => r.map((c, l) => (c - B[i][k][l]) / (2 * h)))));
    }
    const Rup = (rho, sig, mu, nu) => {
      let v = (mu ? DG[mu][rho][nu][sig] : 0) - (nu ? DG[nu][rho][mu][sig] : 0);
      for (let la = 0; la < 4; la++) v += G[rho][mu][la] * G[la][nu][sig] - G[rho][nu][la] * G[la][mu][sig];
      return v;
    };
    const R = riemannCoord(pos, a);
    let scale = 0;
    for (const v of R) scale = Math.max(scale, Math.abs(v));
    for (let al = 0; al < 4; al++) for (let sg = 0; sg < 4; sg++) for (let mu = 0; mu < 4; mu++) for (let nu = 0; nu < 4; nu++) {
      let v = 0;
      for (let rho = 0; rho < 4; rho++) v += g[al][rho] * Rup(rho, sg, mu, nu);
      close(R[((al * 4 + sg) * 4 + mu) * 4 + nu], v, 1e-7 * scale, `a=${a} pos=${pos} R_${al}${sg}${mu}${nu}`);
    }
  }
});

test('canonical frame is orthonormal and regular across the horizon', () => {
  for (const a of SPINS) for (const pos of [...POINTS, fromSpheroidal(horizons(a).rPlus, 1.1, 0.3, a)]) {
    const { frame, g } = canonicalFrame(pos, a);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) close(dot(g, frame[i], frame[j]), i === j ? (i ? 1 : -1) : 0, 1e-10);
  }
});

test('tidal tensor: radial infall in Schwarzschild has eigenvalues (−2, 1, 1) M/r³ at any speed', () => {
  for (const r0 of [20, 6, 2.5, 1.2]) {
    // Fall from rest at infinity (E = 1) along a radial line, also inside r = 2M.
    const { s, E } = stateFromConstants(r0, 1.0, 0.4, 1, 0, 0, 0, -1, 1);
    const u = fourVelocity(s, E, 0);
    const T = tidalTensor(s.slice(0, 3), 0, u, triadFor(s.slice(0, 3), 0, u));
    const ev = symEig3(T).values;
    [-2, 1, 1].forEach((k, i) => close(ev[i], k / r0 ** 3, 1e-10 / r0 ** 3, `r=${r0}`));
  }
});

test('tidal tensor on Kerr equatorial circular orbits matches Marck (1983)', () => {
  for (const a of [0, 0.9, -0.9, 0.998]) for (const r of [iscoRadius(a) * 1.01, 8, 30]) {
    // Prograde (a > 0) or retrograde (a < 0) circular orbit constants (Bardeen et al. 1972).
    const sq = Math.sqrt(r ** 1.5 - 3 * Math.sqrt(r) + 2 * a);
    const E = circularOrbitEnergy(r, a);
    const Lz = (r * r - 2 * a * Math.sqrt(r) + a * a) / (r ** 0.75 * sq);
    const { s } = stateFromConstants(r, Math.PI / 2, 0.2, E, Lz, 0, a, -1, 1);
    const u = fourVelocity(s, E, a);
    const T = tidalTensor(s.slice(0, 3), a, u, triadFor(s.slice(0, 3), a, u));
    const K = (Lz - a * E) ** 2;
    const expect = [-(2 + (3 * K) / r ** 2), 1, 1 + (3 * K) / r ** 2].map((v) => v / r ** 3).sort((x, y) => x - y);
    const ev = symEig3(T).values;
    for (let i = 0; i < 3; i++) close(ev[i], expect[i], 1e-9 / r ** 3, `a=${a} r=${r}`);
  }
});

test('initial data from (E, Lz, Q) is a unit timelike state with the requested constants', () => {
  for (const a of SPINS) for (const [r, th] of [[12, 1.3], [25, 1.8], [18, 1.45]]) {
    const E = 0.97, Lz = 3.1, Q = 4.2;
    assert.ok(radialPotential(r, E, Lz, Q, a) > 0 && polarPotential(th, E, Lz, Q, a) > 0);
    const { s } = stateFromConstants(r, th, 0.4, E, Lz, Q, a, -1, 1);
    close(hamiltonian(s, E, a), -0.5, 1e-13, 'u·u = −1');
    close(angularMomentum(s), Lz, 1e-12);
    close(carterQTimelike(s, E, a), Q, 1e-11);
  }
});

function integrate(s0, E, a, tauMax, opts = {}, onStep) {
  const I = timelikeIntegrator(0, s0, E, a, opts);
  while (I.lambda < tauMax) {
    I.step(Math.min(opts.hMax ?? 1, tauMax - I.lambda));
    if (onStep && onStep(I) === false) break;
  }
  return I;
}

test('generic Kerr orbit conserves u·u, E, Lz and Q and turns at the analytic pericentre', () => {
  for (const a of [0.9, -0.6]) {
    const C = orbitConstants({ E: 0.99, rp: 7, incl: 1.0, a, rStart: 30 });
    const { s } = stateFromConstants(30, Math.PI / 2, 0, C.E, C.Lz, C.Q, a, -1, 1);
    let minR = Infinity, maxdH = 0, maxdQ = 0, maxdL = 0;
    integrate(s, C.E, a, 600, { hMax: 2 }, (I) => {
      const st = Array.from(I.y.slice(1, 7));
      // Minimum radius inside the step from the dense output.
      for (let k = 0; k <= 64; k++) {
        const y = I.interpolate(k / 64);
        minR = Math.min(minR, ksRadius(y[1], y[2], y[3], a));
      }
      maxdH = Math.max(maxdH, Math.abs(hamiltonian(st, C.E, a) + 0.5));
      maxdQ = Math.max(maxdQ, Math.abs(carterQTimelike(st, C.E, a) - C.Q));
      maxdL = Math.max(maxdL, Math.abs(angularMomentum(st) - C.Lz));
    });
    assert.ok(maxdH < 1e-10 && maxdQ < 1e-8 && maxdL < 1e-10, `drift H ${maxdH} Q ${maxdQ} L ${maxdL}`);
    close(minR, 7, 1e-8, `a=${a} pericentre`);
    close(pericentre(C.E, C.Lz, C.Q, a, 30), 7, 1e-9);
  }
});

test('radial free fall from rest: proper time follows the Schwarzschild cycloid, smoothly through r = 2M', () => {
  const r0 = 10;
  const E = Math.sqrt(1 - 2 / r0);
  const rs = r0 * (1 - 1e-9); // start just below the turning point
  const { s } = stateFromConstants(rs, 0.9, 0.3, E, 0, 0, 0, -1, 1);
  const cyc = (r) => { const eta = Math.acos((2 * r) / r0 - 1); return Math.sqrt(r0 ** 3 / 8) * (eta + Math.sin(eta)); };
  for (const rT of [6, 2, 0.6]) {
    let tau = null;
    integrate(s, E, 0, 200, { hMax: 0.05 }, (I) => {
      const r = ksRadius(I.y[1], I.y[2], I.y[3], 0);
      if (r <= rT) {
        // refine on the last step
        let lo = 0, hi = 1;
        for (let k = 0; k < 60; k++) {
          const m = 0.5 * (lo + hi), y = I.interpolate(m);
          if (ksRadius(y[1], y[2], y[3], 0) > rT) lo = m; else hi = m;
        }
        tau = I.lambda - I.lastH * (1 - hi);
        return false;
      }
    });
    const expected = cyc(rT) - cyc(rs);
    close(tau, expected, 2e-8 * expected, `τ to r=${rT}`);
  }
});

test('circular equatorial orbits keep r and the Kepler frequency Ω = 1/(r^{3/2} + a)', () => {
  for (const a of [0, 0.9, -0.9]) {
    const r = 12;
    const sq = Math.sqrt(r ** 1.5 - 3 * Math.sqrt(r) + 2 * a);
    const E = circularOrbitEnergy(r, a);
    const Lz = (r * r - 2 * a * Math.sqrt(r) + a * a) / (r ** 0.75 * sq);
    const { s } = stateFromConstants(r, Math.PI / 2, 0, E, Lz, 0, a, -1, 1);
    let maxDr = 0;
    const I = integrate(s, E, a, 2000, { hMax: 2 }, (J) => {
      maxDr = Math.max(maxDr, Math.abs(ksRadius(J.y[1], J.y[2], J.y[3], a) - r));
    });
    // Floor: R(r) = 0 is met to rounding, which leaves p_r ~ √ε ≈ 1e-7.
    assert.ok(maxDr < 3e-6, `radius drift ${maxDr}`);
    // Total azimuth swept (unwrap) over coordinate time t.
    const phi = Math.atan2(I.y[2], I.y[1]);
    const t = I.y[0];
    const turns = Math.floor((keplerOmega(r, a) * t) / (2 * Math.PI));
    const expectedPhi = keplerOmega(r, a) * t - 2 * Math.PI * turns;
    let d = phi - expectedPhi;
    d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
    close(d, 0, 2e-6, `a=${a} phase after ${turns} orbits`);
  }
});

test('equatorial pericentre advance equals the Kerr quadrature 2∫ dφ/dr dr', () => {
  for (const a of [0, 0.7, -0.7]) {
    const E = 0.97, rp = 9;
    const C = orbitConstants({ E, rp, incl: 0, a, rStart: rp * 1.02 });
    const Lz = C.Lz;
    // Apocentre: largest root of R(r).
    let ra = rp * 1.5;
    while (radialPotential(ra, E, Lz, 0, a) > 0) ra *= 1.2;
    let lo = ra / 1.2, hi = ra;
    for (let k = 0; k < 200; k++) { const m = 0.5 * (lo + hi); if (radialPotential(m, E, Lz, 0, a) > 0) lo = m; else hi = m; }
    ra = lo;
    // Δφ = 2∫ [(Lz − aE) + aP/Δ]/√R dr, with r = (rp + ra)/2 − (ra − rp)/2 cos χ (removes endpoint singularities).
    const N = 20000;
    let sum = 0;
    for (let i = 0; i < N; i++) {
      const chi = (Math.PI * (i + 0.5)) / N;
      const r = 0.5 * (rp + ra) - 0.5 * (ra - rp) * Math.cos(chi);
      const drdchi = 0.5 * (ra - rp) * Math.sin(chi);
      const D = r * r - 2 * r + a * a;
      const P = E * (r * r + a * a) - a * Lz;
      sum += (((Lz - a * E) + (a * P) / D) / Math.sqrt(Math.max(radialPotential(r, E, Lz, 0, a), 1e-300))) * drdchi;
    }
    const dphiExpected = 2 * sum * (Math.PI / N);
    // Integrate from pericentre to the next pericentre and measure the BL azimuth change.
    const { s } = stateFromConstants(rp, Math.PI / 2, 0, E, Lz, 0, a, 1, 1);
    let phi = 0, last = Math.atan2(s[1], s[0]), passedApo = false, done = false;
    const I = timelikeIntegrator(0, s, E, a, { rtol: 1e-12, atol: 1e-14 });
    const vr = (Y, dY) => Y[1] * dY[1] + Y[2] * dY[2]; // ∝ dr/dτ on the equator
    while (!done) {
      I.step(1);
      const p = Math.atan2(I.y[2], I.y[1]);
      let d = p - last; d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
      phi += d; last = p;
      const v = vr(I.y, I.k1);
      if (v < 0) passedApo = true;
      if (passedApo && v > 0) {
        // Pericentre inside this step: bisect dr/dτ = 0 using the dense output.
        const dr = (th) => {
          const e = 1e-7, y1 = I.interpolate(Math.min(1, th + e)), y0 = I.interpolate(Math.max(0, th - e));
          return ksRadius(y1[1], y1[2], y1[3], a) - ksRadius(y0[1], y0[2], y0[3], a);
        };
        let lo2 = 0, hi2 = 1;
        for (let k = 0; k < 60; k++) { const m = 0.5 * (lo2 + hi2); if (dr(m) < 0) lo2 = m; else hi2 = m; }
        const y = I.interpolate(0.5 * (lo2 + hi2));
        let d2 = Math.atan2(y[2], y[1]) - p; d2 -= 2 * Math.PI * Math.round(d2 / (2 * Math.PI));
        phi += d2;
        done = true;
      }
    }
    // KS and BL azimuths differ by a function of r only, equal at both pericentres.
    close(phi, dphiExpected, 2e-7 * dphiExpected, `a=${a}`);
    assert.ok(phi > 2 * Math.PI);
  }
});

test('parallel transport: geodetic (de Sitter) precession on circular Schwarzschild orbits', () => {
  for (const r of [8, 20]) {
    const E = circularOrbitEnergy(r, 0);
    const Lz = Math.sqrt(r) / Math.sqrt(1 - 3 / r);
    const { s } = stateFromConstants(r, Math.PI / 2, 0, E, Lz, 0, 0, -1, 1);
    const pos = s.slice(0, 3), u = fourVelocity(s, E, 0);
    // Spin initially along the comoving radial direction.
    const g = metricLower(pos, 0);
    const rad = (p, uu) => {
      // unit vector along ∂_r in the comoving frame: x̂ direction projected ⟂ u
      const gg = metricLower(p, 0);
      const n = [0, p[0], p[1], p[2]].map((c, i) => (i ? c / Math.hypot(p[0], p[1], p[2]) : 0));
      const k = dot(gg, n, uu);
      const w = n.map((c, i) => c + k * uu[i]);
      const nn = Math.sqrt(dot(gg, w, w));
      return w.map((c) => c / nn);
    };
    const e = [rad(pos, u)];
    const T = (2 * Math.PI) / keplerOmega(r, 0);
    const I = timelikeIntegrator(0, s, E, 0, { triad: e, rtol: 1e-12, atol: 1e-14 });
    while (I.y[0] < T) I.step(Math.max(1e-6, (T - I.y[0]) / 1.5));
    // Land on t = T by bisection.
    let lo = 0, hi = 1;
    for (let k = 0; k < 80; k++) { const m = 0.5 * (lo + hi); if (I.interpolate(m)[0] < T) lo = m; else hi = m; }
    const Y = I.interpolate(hi);
    const p2 = [Y[1], Y[2], Y[3]], u2 = fourVelocity(Array.from(Y.slice(1, 7)), E, 0);
    const er = rad(p2, u2);
    const g2 = metricLower(p2, 0);
    const spin = unpackTriad(Y)[0];
    // Azimuthal comoving direction completes the frame.
    let ephi = [0, -p2[1], p2[0], 0];
    const k1 = dot(g2, ephi, u2), k2 = dot(g2, ephi, er);
    ephi = ephi.map((c, i) => c + k1 * u2[i] - k2 * er[i]);
    ephi = ephi.map((c) => c / Math.sqrt(dot(g2, ephi, ephi)));
    const alpha = Math.atan2(dot(g2, spin, ephi), dot(g2, spin, er));
    // Relative to the rotating radial direction the spin turns by −2π√(1 − 3M/r) per orbit.
    let d = alpha + 2 * Math.PI * Math.sqrt(1 - 3 / r);
    d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
    close(d, 0, 1e-6, `r=${r}`);
    close(dot(g2, spin, spin), 1, 1e-10, 'transport preserves the norm');
    close(dot(g2, spin, u2), 0, 1e-10, 'and orthogonality to u');
    void g;
  }
});

test('mirror map: physical geodesics become geodesics of Kerr(−a) with the same energy', () => {
  for (const a of [0, 0.6, -0.9]) {
    const E = 0.97, Lz = 3.1, Q = 4.2;
    const { s } = stateFromConstants(12, 1.1, 0.4, E, Lz, Q, a, -1, 1);
    const I = timelikeIntegrator(0, s, E, a);
    let checked = 0;
    for (let k = 0; k < 400; k++) {
      I.step(0.5);
      const Y = I.y;
      const r = ksRadius(Y[1], Y[2], Y[3], a);
      if (r < horizons(a).rPlus * 1.3) break;
      const st = Array.from(Y.slice(1, 7));
      const u = fourVelocity(st, E, a);
      const m = mirrorPoint(Y[0], Y[1], Y[2], Y[3], a);
      close(ksRadius(m.x, m.y, m.z, -a), r, 1e-12 * r, 'radius preserved');
      const um = mirrorVector(Y[1], Y[2], Y[3], a, u).map((c) => -c);
      const gm = metricLower([m.x, m.y, m.z], -a);
      close(dot(gm, um, um), -1, 1e-9, 'unit timelike in the mirror metric');
      assert.ok(um[0] > 0, 'future-directed after reversal');
      const pl = lower(gm, um);
      close(-pl[0], E, 1e-9, 'energy preserved');
      close(m.x * pl[2] - m.y * pl[1], -Lz, 1e-9, 'L flips sign');
      // Tangent of the mapped curve equals −ũ (finite difference along the orbit).
      const eps = 1e-5;
      const Y2 = I.interpolate(1 - eps / I.lastH);
      const m2 = mirrorPoint(Y2[0], Y2[1], Y2[2], Y2[3], a);
      const fd = [m.tp - m2.tp, m.x - m2.x, m.y - m2.y, m.z - m2.z].map((c) => c / eps);
      fd.forEach((c, i) => close(c, -um[i], 1e-4 * (1 + Math.abs(um[i])), `tangent ${i}`));
      const back = physicalPoint(m.x, m.y, m.z, a);
      back.forEach((c, i) => close(c, Y[i + 1], 1e-11 * r));
      checked++;
    }
    assert.ok(checked > 20);
  }
  // Schwarzschild: t' = −(t − 4 ln|r − 2|).
  close(mirrorPoint(5, 7, 0, 0, 0).tp, -(5 - 4 * Math.log(5)), 1e-12);
  close(timeShift(10, 0), 2 * Math.log(8), 1e-12);
  void geodesicRHS;
});

// Stars, tides and tidal disruption: SPH star, Fermi-frame dynamics and the
// full encounter pipeline, against analytic and literature results.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { laneEmden, SPHStar, gravForce, gravPotential } from '../src/physics/star.js';
import { ksRadius, horizons } from '../src/physics/kerr.js';
import {
  fourVelocity, triadFor, timelikeIntegrator, unpackTriad, tidalTensor, reorthonormalise,
} from '../src/physics/spacetime.js';
import { Simulation, STAR_TYPES, starScales, fermiToGlobal, launchState } from '../src/sim/events.js';
import { Dopri5 } from '../src/physics/ode.js';
import { geodesicRHS, dtdl } from '../src/physics/kerr.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} |${a} - ${b}| = ${Math.abs(a - b)} > ${tol}`);

test('Lane–Emden surfaces and masses match the tabulated values (Chandrasekhar 1939)', () => {
  for (const [n, xi1, m] of [[1, Math.PI, Math.PI], [1.5, 3.65375, 2.71406], [3, 6.89685, 2.01824]]) {
    const le = laneEmden(n);
    close(le.xi1, xi1, 2e-5, `ξ₁ for n=${n}`);
    close(-le.xi1 * le.xi1 * le.dtheta1, m, 2e-5, `−ξ₁²θ'(ξ₁) for n=${n}`);
  }
});

test('softened gravity is continuous and exactly Newtonian beyond two smoothing lengths', () => {
  for (const h of [0.1, 0.37]) {
    for (const q of [1, 2]) {
      close(gravForce(q * h * (1 - 1e-9), h), gravForce(q * h * (1 + 1e-9), h), 1e-6 / (h * h));
      close(gravPotential(q * h * (1 - 1e-9), h), gravPotential(q * h * (1 + 1e-9), h), 1e-6 / h);
    }
    close(gravForce(2.5 * h, h), 1 / (2.5 * h) ** 2, 1e-15);
    // Force is −dΦ/dr.
    const r = 0.8 * h, e = 1e-7;
    close(gravForce(r, h), (gravPotential(r + e, h) - gravPotential(r - e, h)) / (2 * e), 1e-5 / (h * h));
  }
});

test('an isolated SPH polytrope is in virial and hydrostatic equilibrium and conserves energy', () => {
  const N = 500;
  const s = SPHStar.polytrope({ N, n: 1.5 });
  s.relax(3, 0.05);
  const e0 = s.energies();
  // Virial theorem for γ = 5/3 gas: 2K + W + 3(γ − 1)U = 0 → W + 2U ≈ 0 at rest.
  close((e0.W + 2 * e0.U) / Math.abs(e0.W), 0, 0.03, 'virial');
  // Binding energy of an n = 1.5 polytrope: W = −3/(5 − n) GM²/R (softening makes it slightly shallower).
  close(e0.W, -3 / 3.5, 0.06, 'gravitational energy');
  const half = () => {
    const r = [];
    for (let i = 0; i < N; i++) r.push(Math.hypot(s.pos[3 * i], s.pos[3 * i + 1], s.pos[3 * i + 2]));
    r.sort((x, y) => x - y);
    return r[N >> 1];
  };
  const r50 = half();
  while (s.t < 3) s.step(s.timestep());
  close(half(), r50, 0.03, 'half-mass radius stays put over 3 dynamical times');
  close((s.energies().total - e0.total) / Math.abs(e0.total), 0, 2e-3, 'total energy');
});

function fermiVsExact(scale) {
  // A small cloud of free particles on a deep Kerr flyby, evolved two ways:
  // (1) geodesic deviation in the Fermi frame of the centre of mass,
  // (2) each particle as its own exact Kerr geodesic.
  const a = 0.9;
  const L = launchState({ a, r0: 40, rp: 7, incl: 0.6, phiPeri: 0.3 });
  const E = L.E;
  const pos0 = L.s.slice(0, 3), u0 = fourVelocity(L.s, E, a);
  const cm = timelikeIntegrator(0, L.s, E, a, { triad: triadFor(pos0, a, u0), rtol: 1e-12, atol: 1e-14 });
  const xi0 = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.6, -0.5, 0.4]].map((x) => x.map((c) => c * scale));
  const X0 = [0, ...pos0];
  const tri0 = unpackTriad(cm.y);
  const exact = xi0.map((x) => {
    const g = fermiToGlobal(a, X0, u0, tri0, x, [0, 0, 0]);
    const f = (Y, out) => {
      const st = [Y[1], Y[2], Y[3], Y[4], Y[5], Y[6]];
      const d = geodesicRHS(st, g.E, a, new Array(6));
      out[0] = dtdl(st, g.E, a);
      for (let i = 0; i < 6; i++) out[i + 1] = d[i];
    };
    return { I: new Dopri5(f, [g.t, ...g.s], { rtol: 1e-12, atol: 1e-14 }), E: g.E };
  });
  // Fermi-frame state: ξ and dξ/dτ, advanced with RK4 in τ using C along the CM.
  let xi = xi0.map((x) => x.slice()), v = xi0.map(() => [0, 0, 0]);
  const tidal = () => {
    const pos = [cm.y[1], cm.y[2], cm.y[3]];
    const u = fourVelocity(Array.from(cm.y.slice(1, 7)), E, a);
    return { C: tidalTensor(pos, a, u, reorthonormalise(pos, a, u, unpackTriad(cm.y))), pos, u };
  };
  let C = tidal().C;
  const tauEnd = 260; // well past pericentre
  const dtau = 0.25;
  let tau = 0;
  const acc = (x, CC) => [0, 1, 2].map((i) => -(CC[i][0] * x[0] + CC[i][1] * x[1] + CC[i][2] * x[2]));
  while (tau < tauEnd) {
    // Velocity Verlet with C sampled at the start and end of the step.
    const a0 = xi.map((x) => acc(x, C));
    xi = xi.map((x, k) => x.map((c, i) => c + dtau * v[k][i] + 0.5 * dtau * dtau * a0[k][i]));
    while (cm.lambda < tau + dtau - 1e-12) cm.step(tau + dtau - cm.lambda);
    tau += dtau;
    C = tidal().C;
    const a1 = xi.map((x) => acc(x, C));
    v = v.map((vv, k) => vv.map((c, i) => c + 0.5 * dtau * (a0[k][i] + a1[k][i])));
  }
  const r = ksRadius(cm.y[1], cm.y[2], cm.y[3], a);
  assert.ok(r > 20, `centre of mass is outbound again (r=${r})`);
  const T = tidal();
  const X = [cm.y[0], ...T.pos];
  const tri = reorthonormalise(T.pos, a, T.u, unpackTriad(cm.y));
  let worst = 0, growth = 0;
  xi.forEach((x, k) => {
    const g = fermiToGlobal(a, X, T.u, tri, x, v[k]);
    const { I } = exact[k];
    while (I.y[0] < g.t) I.step(Math.max(1e-3, g.t - I.y[0]));
    let lo = 0, hi = 1;
    for (let it = 0; it < 60; it++) { const m = 0.5 * (lo + hi); if (I.interpolate(m)[0] < g.t) lo = m; else hi = m; }
    const y = I.interpolate(hi);
    const err = Math.hypot(y[1] - g.s[0], y[2] - g.s[1], y[3] - g.s[2]);
    const sep = Math.hypot(...x);
    growth = Math.max(growth, sep / Math.hypot(...xi0[k]));
    worst = Math.max(worst, err / sep);
  });
  return { worst, growth };
}

test('Fermi-frame tidal dynamics (−C ξ along a transported tetrad) converges to exact neighbouring geodesics', () => {
  // The quadrupole (geodesic-deviation) truncation error is O(ξ/r): it must
  // shrink in proportion to the cloud size.
  const big = fermiVsExact(0.02), small = fermiVsExact(0.005);
  assert.ok(big.growth > 3, `tides stretched the cloud (×${big.growth.toFixed(1)})`);
  assert.ok(small.worst < 1.2e-3, `relative error ${small.worst} for ξ = 0.005 M`);
  const order = Math.log(big.worst / small.worst) / Math.log(4);
  assert.ok(order > 0.9, `error ∝ ξ^${order.toFixed(2)} (expected ≥ 1)`);
});

async function runEncounter(cfg) {
  const sim = new Simulation();
  sim.add('star', cfg);
  let out;
  do out = sim.advance(1e4); while (out.busy);
  return out.events[0].stats;
}

test('tidal disruption: debris energy spread matches GM R*/r_t² and splits evenly into bound and unbound', async () => {
  const star = STAR_TYPES.redDwarf;
  const st = await runEncounter({ a: 0.5, massSolar: 1e6, star, beta: 2, incl: 0.4, t0: 0, N: 400 });
  assert.equal(st.outcome, 'full disruption');
  const de = starScales(star, 1e6).R / starScales(star, 1e6).rt ** 2;
  assert.ok(st.epsSpread.p95 > 0.7 * de && st.epsSpread.p95 < 1.6 * de, `p95 ${st.epsSpread.p95 / de} Δε`);
  assert.ok(st.epsSpread.p05 < -0.7 * de && st.epsSpread.p05 > -1.6 * de, `p05 ${st.epsSpread.p05 / de} Δε`);
  const ratio = st.bound / st.unbound;
  assert.ok(ratio > 0.75 && ratio < 1.33, `bound/unbound ${ratio}`);
  // Fallback: first return near 2π(M/2Δε)^{3/2}·(GM/c³), a few weeks for 10⁶ M☉.
  const tmin = 2 * Math.PI * Math.pow(2 * de, -1.5) * 4.925490947641267e-6 * 1e6;
  assert.ok(st.fallback.tMin > 0.3 * tmin && st.fallback.tMin < 1.5 * tmin, `t_min ${st.fallback.tMin / 86400} d vs ${tmin / 86400} d`);
});

test('tidal disruption outcomes: grazing encounters strip little, deep ones disrupt, heavy holes swallow whole', async () => {
  const graze = await runEncounter({ a: 0, massSolar: 1e6, star: STAR_TYPES.redDwarf, beta: 0.5, t0: 0, N: 400 });
  assert.ok(graze.coreFraction > 0.85, `β = 0.5 keeps the star (core ${graze.coreFraction})`);
  const whole = await runEncounter({ a: 0, massSolar: 3e8, star: STAR_TYPES.sun, beta: 1, t0: 0, N: 300 });
  assert.ok(whole.swallowed > 0.95, `r_t < capture radius above the Hills mass: swallowed ${whole.swallowed}`);
  assert.ok(starScales(STAR_TYPES.sun, 3e8).rt < 4 && horizons(0).rPlus === 2);
});

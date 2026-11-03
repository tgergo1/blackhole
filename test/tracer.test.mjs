import { test } from 'node:test';
import assert from 'node:assert/strict';
import { hamiltonian, angularMomentum, carterQ, geodesicRHS, ksRadius, horizons } from '../src/physics/kerr.js';
import { makeCamera, pixelRay, rayFromDirection, OBSERVERS } from '../src/physics/camera.js';
import { traceRay, dopriStep } from '../src/physics/tracer.js';
import { photonOrbitRadii, iscoRadius } from '../src/physics/disk.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} |${a} - ${b}| = ${Math.abs(a - b)} > ${tol}`);

// Integrate a ray step by step, recording invariants.
function integrateRecording(state, E, a, nSteps, h0 = 0.05) {
  let s = state.slice();
  let k1 = geodesicRHS(s, E, a, new Array(6));
  let h = h0;
  const H0 = hamiltonian(s, E, a), L0 = angularMomentum(s), Q0 = carterQ(s, E, a);
  let maxdH = 0, maxdL = 0, maxdQ = 0, minR = Infinity;
  for (let i = 0; i < nSteps; i++) {
    const st = dopriStep(s, k1, h, E, a, 1e-11, 1e-13);
    if (st.err > 1) { h *= 0.5; continue; }
    s = st.y; k1 = st.k7;
    const r = ksRadius(s[0], s[1], s[2], a);
    minR = Math.min(minR, r);
    if (r < horizons(a).rPlus || r > 500) break;
    maxdH = Math.max(maxdH, Math.abs(hamiltonian(s, E, a) - H0));
    maxdL = Math.max(maxdL, Math.abs(angularMomentum(s) - L0));
    maxdQ = Math.max(maxdQ, Math.abs(carterQ(s, E, a) - Q0) / Math.max(1, Math.abs(Q0)));
    h = Math.min(h * Math.min(5, 0.9 * Math.pow(st.err || 1e-10, -0.2)), 0.5 * r);
  }
  return { maxdH, maxdL, maxdQ, minR };
}

test('null constraint, L and Carter constant are conserved through strong-field passes', () => {
  for (const a of [0.0, 0.7, -0.998]) {
    const cam = makeCamera({ a, r: 30, theta: 1.1, phi: 0.3, observer: OBSERVERS.ZAMO });
    for (const [sx, sy] of [[0.18, 0.05], [-0.2, 0.12], [0.02, 0.21]]) {
      const { state, E } = pixelRay(cam, sx, sy, Math.tan(0.5), 1);
      close(hamiltonian(state, E, cam.aT), 0, 1e-13, 'initial ray is null');
      const res = integrateRecording(state, E, cam.aT, 4000);
      assert.ok(res.minR < 6, `ray should pass close to the hole (minR=${res.minR})`);
      assert.ok(res.maxdH < 1e-9, `H drift ${res.maxdH}`);
      assert.ok(res.maxdL < 1e-9, `L drift ${res.maxdL}`);
      assert.ok(res.maxdQ < 1e-8, `Q drift ${res.maxdQ}`);
    }
  }
});

// Bisect the horizontal screen coordinate where capture switches to escape.
function shadowEdge(cam, sLo, sHi, tanHalf, opts) {
  const captured = (sx) => traceRay(...Object.values(pixelRay(cam, sx, 0, tanHalf, 1)), cam.aT, opts).type === 'horizon';
  const cLo = captured(sLo);
  assert.notEqual(cLo, captured(sHi));
  for (let i = 0; i < 40; i++) {
    const m = 0.5 * (sLo + sHi);
    if (captured(m) === cLo) sLo = m; else sHi = m;
  }
  const { state, E } = pixelRay(cam, 0.5 * (sLo + sHi), 0, tanHalf, 1);
  return { sx: 0.5 * (sLo + sHi), xi: angularMomentum(state) / E };
}

test('equatorial shadow edges match the circular photon orbits, with correct handedness', () => {
  const rObs = 1e4;
  const tanHalf = 10 / rObs;
  const opts = { disk: false, rEscape: 2e4, rtol: 1e-11, atol: 1e-13 };
  for (const a of [0, 0.5, 0.9, 0.998]) {
    const cam = makeCamera({ a, r: rObs, theta: Math.PI / 2, phi: 0, observer: OBSERVERS.STATIC });
    // Analytic critical impact parameters (Bardeen 1973):
    const bPro = -a + 6 * Math.cos(Math.acos(-a) / 3);
    const bRetro = a + 6 * Math.cos(Math.acos(a) / 3);
    const left = shadowEdge(cam, -0.01, -0.9, tanHalf, opts);
    const right = shadowEdge(cam, 0.01, 0.9, tanHalf, opts);
    // In the mirror spacetime L flips sign: physical ξ = −ξ_mirror.
    close(-left.xi, bPro, 2e-6 * bRetro, `a=${a} prograde edge (screen left)`);
    close(-right.xi, -bRetro, 2e-6 * bRetro, `a=${a} retrograde edge (screen right)`);
    // Screen positions: the flattened prograde edge is on the left (approaching side).
    close(left.sx * tanHalf * rObs, -bPro, 3e-3 * bPro, 'left edge position');
    close(right.sx * tanHalf * rObs, bRetro, 3e-3 * bRetro, 'right edge position');
    // sanity of the photon-orbit helper
    const ph = photonOrbitRadii(a);
    assert.ok(ph.pro <= ph.retro);
  }
});

// Analytic capture criterion: a photon from infinity with constants (ξ, η)
// falls in iff the radial potential R(r) stays positive down to r₊.
function analyticallyCaptured(xi, eta, a) {
  const rp = horizons(a).rPlus;
  const R = (r) => (r * r + a * a - a * xi) ** 2 - (r * r - 2 * r + a * a) * (eta + (xi - a) ** 2);
  for (let i = 0; i <= 20000; i++) {
    const r = rp + (i / 20000) ** 2 * 200;
    if (R(r) < 0) return false;
  }
  return true;
}

test('capture vs escape agrees with the analytic Kerr shadow (Bardeen curve) for inclined views', () => {
  const rObs = 5e3;
  for (const [a, th] of [[0.9, 1.0], [0.998, 1.4], [-0.6, 0.5]]) {
    const cam = makeCamera({ a, r: rObs, theta: th, phi: 0.7, observer: OBSERVERS.STATIC });
    const tanHalf = 9 / rObs;
    let n = 0, ambiguous = 0;
    for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) {
      const sx = -1 + (2 * i + 0.5) / 9, sy = -1 + (2 * j + 0.5) / 9;
      const { state, E } = pixelRay(cam, sx, sy, tanHalf, 1);
      const xi = angularMomentum(state) / E;
      const eta = carterQ(state, E, cam.aT) / (E * E);
      // Skip rays within a hair of the critical curve.
      const near = [0.995, 1.005].map((f) => analyticallyCaptured(xi * f, eta * f, cam.aT));
      if (near[0] !== near[1]) { ambiguous++; continue; }
      const res = traceRay(state, E, cam.aT, { disk: false, rEscape: 1e4 });
      assert.equal(res.type === 'horizon', analyticallyCaptured(xi, eta, cam.aT), `a=${a} pixel ${sx},${sy}`);
      n++;
    }
    assert.ok(n > 60, `enough rays tested (${n}, ambiguous ${ambiguous})`);
  }
});

test('face-on disk redshift equals the analytic transverse-Doppler + gravitational factor', () => {
  const rObs = 2000;
  for (const a of [0, 0.7, -0.5]) {
    const cam = makeCamera({ a, r: rObs, theta: 0, phi: 0, observer: OBSERVERS.STATIC });
    const tanHalf = 25 / rObs;
    for (const sx of [0.4, 0.6, 0.9]) {
      const { state, E } = pixelRay(cam, sx, 0, tanHalf, 1);
      const res = traceRay(state, E, cam.aT, { rOut: 60, rEscape: 3 * rObs });
      assert.equal(res.type, 'disk');
      const r = res.r;
      // On-axis observer: L = 0, so ν_emit = u^t E; static observer at rObs measures E/√(−g_tt), with −g_tt = 1 − 2r/(r² + a²) on the axis.
      const ut = (r ** 1.5 + a) / (r ** 0.75 * Math.sqrt(r ** 1.5 - 3 * Math.sqrt(r) + 2 * a));
      const expected = 1 / ut / Math.sqrt(1 - (2 * rObs) / (rObs * rObs + a * a));
      close(res.g, expected, 1e-8, `a=${a} r=${r}`);
      assert.ok(r > iscoRadius(a));
    }
  }
});

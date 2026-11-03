// Rendering of dynamic bodies: light-travel time along rays, redshift of a
// body falling into the hole, and the world-line resampling used by the GPU.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeCamera, pixelRay, OBSERVERS } from '../src/physics/camera.js';
import { traceRayObjects } from '../src/physics/tracer.js';
import { ksRadius } from '../src/physics/kerr.js';
import { timeShift } from '../src/physics/spacetime.js';
import { Simulation, SAMPLE, retardedStart } from '../src/sim/events.js';
import { ObjectField } from '../src/render/objects.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} |${a} - ${b}| = ${Math.abs(a - b)} > ${tol}`);
const EMPTY = { count: 0 };

test('mirror time along a radial ray obeys t′ + r = const (ingoing null ray of Kerr–Schild)', () => {
  for (const rc of [30, 300]) {
    const cam = makeCamera({ a: 0, r: rc, theta: Math.PI / 2, phi: 0, observer: OBSERVERS.STATIC });
    const { state, E } = pixelRay(cam, 0, 0, 0.1, 1);
    const res = traceRayObjects(state, E, cam.aT, { disk: false, rtol: 1e-11, atol: 1e-13 }, EMPTY);
    assert.equal(res.type, 'horizon');
    close(res.sEnd, rc - res.r, 1e-8 * rc, `r_cam = ${rc}`);
  }
});

function simulate(kind, cfg) {
  const sim = new Simulation();
  sim.add(kind, cfg);
  const field = new ObjectField({ maxObjects: 8 });
  let out;
  do { out = sim.advance(1e4); field.ingest(out.entities); } while (out.busy);
  return { field, stats: out.events[0].stats };
}

test('a body falling straight in is seen with g = (1 − √(2M/r)) / √(1 − 2M/r_obs), at the retarded time, and freezes', () => {
  const rc = 60, r0 = 40;
  const t0 = retardedStart(0, rc, r0, 0);
  // Probe on the camera's line of sight, falling radially from rest at infinity.
  const { field } = simulate('probe', { a: 0, r0, plunge: true, phiPeri: 0, t0, radius: 0.001 });
  const cam = makeCamera({ a: 0, r: rc, theta: Math.PI / 2, phi: 0, observer: OBSERVERS.STATIC });
  const { state, E } = pixelRay(cam, 0, 0, 0.1, 1);
  const window = 200;
  let prevG = Infinity, seen = 0;
  for (const tObs of [0, 40, 80, 120, 150, 170, 190, 210, 240, 280]) {
    const tpCam = -(tObs - 2 * timeShift(rc, 0));
    field.build(tpCam, window);
    const snap = field.snapshot();
    const res = traceRayObjects(state, E, cam.aT, { disk: false, rtol: 1e-11, atol: 1e-13 }, snap);
    if (!res.hits.length) continue;
    seen++;
    const hit = res.hits[0];
    const rEmit = rc - hit.s; // radial ray: t′ + r = const
    const expected = (1 - Math.sqrt(2 / rEmit)) / Math.sqrt(1 - 2 / rc);
    close(hit.g, expected, 1e-4, `t_obs = ${tObs}, r_emit = ${rEmit.toFixed(3)}`);
    assert.ok(hit.g < prevG, 'redshift grows monotonically');
    prevG = hit.g;
  }
  assert.ok(seen >= 6, `body seen at ${seen} observer times`);
  assert.ok(prevG < 0.05, `image fades towards the horizon (g = ${prevG})`);
});

test('world-line resampling reproduces the simulated samples at the nodes', () => {
  const { field } = simulate('probe', { a: 0.7, r0: 30, rp: 8, incl: 0.4, t0: 0, radius: 0.1 });
  const e = [...field.entities.values()][0];
  const out = new Float64Array(SAMPLE);
  for (let i = 1; i < e.count - 1; i += Math.floor(e.count / 25)) {
    const tp = e.data[i * SAMPLE];
    ObjectField.stateAt(e, tp, out);
    for (let k = 1; k < SAMPLE; k++) close(out[k], e.data[i * SAMPLE + k], 1e-9 * (1 + Math.abs(out[k])));
    // Midpoints are smooth (Hermite in t′): position error small compared with the step.
    const tm = 0.5 * (tp + e.data[(i + 1) * SAMPLE]);
    ObjectField.stateAt(e, tm, out);
    const r = ksRadius(out[1], out[2], out[3], -0.7);
    assert.ok(r > 1 && r < 1.6 * 30 + 1);
  }
});

// Observer and local tetrad.
//
// Backward ray tracing follows *past-directed* null geodesics from the camera.
// Ingoing Kerr–Schild coordinates are regular on the future horizon but not on
// the past horizon, which past-directed rays asymptote to. We therefore use the
// time-reversal isometry  T: Kerr(a) → Kerr(−a),  t ↦ −t:  a past-directed ray
// in Kerr(a) is a future-directed ray in Kerr(−a) with the same spatial path.
// All tracing happens in that "mirror" spacetime with spin aT = −a, where the
// rays cross the (future) horizon smoothly. Scalars such as the redshift
// g = (p·u_obs)/(p·u_emit) are invariant under T, and spatial coordinates are
// unchanged, so the image is identical. Under T every observer's spatial
// velocity flips sign; for static observers and ZAMOs this is automatic
// because they are defined geometrically from the (mirror) metric.

import { metricLower, metricUpper, lower, dot, ksRadius, fromSpheroidal } from './kerr.js';

export const OBSERVERS = { STATIC: 'static', ZAMO: 'zamo' };

/** Internal (mirror) spin used for tracing. */
export function tracingSpin(a) {
  return -a;
}

function scale(v, s) {
  return v.map((c) => c * s);
}
function add(u, v, s = 1) {
  return u.map((c, i) => c + s * v[i]);
}

/**
 * Observer four-velocity u^μ in the (mirror) Kerr–Schild metric of spin aT.
 *  - static: u ∝ ∂_t, exists only outside the ergosphere.
 *  - zamo:   zero-angular-momentum observer, u_μ ∝ −∇t_BL with
 *            t_BL = t_KS − ∫ 2r/Δ dr; exists everywhere outside r₊.
 */
export function observerVelocity(type, pos, aT) {
  const g = metricLower(pos, aT);
  const gi = metricUpper(pos, aT);
  let u;
  if (type === OBSERVERS.STATIC) {
    if (g[0][0] >= 0) throw new Error('Static observers cannot exist inside the ergosphere');
    u = [1 / Math.sqrt(-g[0][0]), 0, 0, 0];
  } else {
    const r = ksRadius(pos[0], pos[1], pos[2], aT);
    const delta = r * r - 2 * r + aT * aT;
    if (delta <= 0) throw new Error('ZAMO undefined inside the horizon');
    const a2 = aT * aT;
    const D = r * r + (a2 * pos[2] * pos[2]) / (r * r);
    const gradR = [(pos[0] * r) / D, (pos[1] * r) / D, (pos[2] * (r * r + a2)) / (r * D)];
    const s = (2 * r) / delta;
    const cov = [-1, s * gradR[0], s * gradR[1], s * gradR[2]]; // −∇t_BL
    u = [0, 1, 2, 3].map((m) => gi[m][0] * cov[0] + gi[m][1] * cov[1] + gi[m][2] * cov[2] + gi[m][3] * cov[3]);
    const n = dot(g, u, u);
    u = scale(u, 1 / Math.sqrt(-n));
    if (u[0] < 0) u = scale(u, -1);
  }
  return u;
}

/** Gram–Schmidt a coordinate vector against an orthonormal set (u timelike). */
function orthonormalise(g, v, basis) {
  let w = v.slice();
  for (const { e, norm } of basis) w = add(w, e, -dot(g, w, e) / norm);
  const n = dot(g, w, w);
  return scale(w, 1 / Math.sqrt(Math.abs(n)));
}

/**
 * Camera on an orbit around the hole looking at the centre, optionally
 * rotated by yaw/pitch in its own rest frame.
 *
 * params: { a, r, theta, phi, observer, yaw, pitch }   (angles in radians)
 * returns { pos, aT, u, forward, right, up } with vectors in mirror KS
 * components. A pixel direction d = cosα forward + … gives the ray tangent
 * p^μ = u^μ + d^μ (unit energy in the observer frame).
 */
export function makeCamera({ a, r, theta, phi, observer = OBSERVERS.ZAMO, yaw = 0, pitch = 0 }) {
  const aT = tracingSpin(a);
  const th = Math.min(Math.max(theta, 0), Math.PI);
  const pos = fromSpheroidal(r, th, phi, aT);
  const g = metricLower(pos, aT);
  const u = observerVelocity(observer, pos, aT);
  // Coordinate guesses: inward radial, "north" (−θ̂) and east (φ̂).
  const rhat = [Math.sin(th) * Math.cos(phi), Math.sin(th) * Math.sin(phi), Math.cos(th)];
  const north = [-Math.cos(th) * Math.cos(phi), -Math.cos(th) * Math.sin(phi), Math.sin(th)];
  const B = [{ e: u, norm: -1 }];
  const fwd0 = orthonormalise(g, [0, -rhat[0], -rhat[1], -rhat[2]], B);
  B.push({ e: fwd0, norm: 1 });
  const up0 = orthonormalise(g, [0, north[0], north[1], north[2]], B);
  B.push({ e: up0, norm: 1 });
  // right = forward × up (right-handed screen: x right, y up, looking forward).
  const east = [-Math.sin(phi), Math.cos(phi), 0];
  const cross = [
    -rhat[1] * north[2] + rhat[2] * north[1],
    -rhat[2] * north[0] + rhat[0] * north[2],
    -rhat[0] * north[1] + rhat[1] * north[0],
  ];
  // (−r̂) × north = east·sign; use it to pick the orientation robustly.
  const sgn = Math.sign(cross[0] * east[0] + cross[1] * east[1] + cross[2] * east[2]) || 1;
  const right0 = orthonormalise(g, [0, sgn * east[0], sgn * east[1], sgn * east[2]], B);
  // Apply yaw (about up) then pitch (about right) in the local frame.
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const f1 = add(scale(fwd0, cy), right0, sy);
  const r1 = add(scale(right0, cy), fwd0, -sy);
  const forward = add(scale(f1, cp), up0, sp);
  const up = add(scale(up0, cp), f1, -sp);
  return { pos, aT, a, u, forward, right: r1, up, g };
}

/**
 * Initial state for the pixel at normalised screen coords (sx, sy) ∈ [−1, 1]
 * (sy up). tanHalfFov is the vertical half-angle tangent.
 * Returns { state: [x,y,z,p_x,p_y,p_z], E } in the mirror spacetime.
 */
export function pixelRay(cam, sx, sy, tanHalfFov, aspect) {
  const dx = sx * tanHalfFov * aspect;
  const dy = sy * tanHalfFov;
  const n = 1 / Math.sqrt(1 + dx * dx + dy * dy);
  const d = add(add(scale(cam.forward, n), cam.right, dx * n), cam.up, dy * n);
  return rayFromDirection(cam, d);
}

/** Ray tangent p^μ = u^μ + d^μ for a unit spatial direction d ⟂ u. */
export function rayFromDirection(cam, d) {
  const pUp = add(cam.u, d);
  const p = lower(cam.g, pUp);
  return { state: [cam.pos[0], cam.pos[1], cam.pos[2], p[1], p[2], p[3]], E: -p[0] };
}

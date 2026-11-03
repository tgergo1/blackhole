// Kerr spacetime in Cartesian Kerr–Schild coordinates (G = c = M = 1).
//
//   g_{μν} = η_{μν} + f l_μ l_ν,          g^{μν} = η^{μν} − f l^μ l^ν
//   l_μ    = (1, (r x + a y)/(r²+a²), (r y − a x)/(r²+a²), z/r)
//   f      = 2 r³ / (r⁴ + a² z²)
//
// where r is the Boyer–Lindquist-like radius defined implicitly by
//   (x² + y²)/(r² + a²) + z²/r² = 1.
//
// With this sign convention a > 0 means the hole spins about +z
// (frame dragging ω = −g_tφ/g_φφ > 0). These are the *ingoing* coordinates:
// regular across the future horizon, so future-directed rays fall through
// r = r₊ without the step size collapsing.
//
// Null geodesics follow from the super-Hamiltonian
//   H = ½ g^{μν} p_μ p_ν = ½ ( −E² + |p|² − f (E + k·p)² )
// with p_t = −E conserved and k = spatial part of l. Hamilton's equations are
// evaluated with fully analytic gradients (no finite differences), so the
// GPU port has the same accuracy characteristics as this float64 reference.
//
// This file is the single source of truth for the equations; the GLSL shader
// in src/render/shaders.js is a line-by-line port and is checked against it.

/** Kerr–Schild radius r(x, y, z) for spin a. Stable in both limits. */
export function ksRadius(x, y, z, a) {
  const a2 = a * a;
  const w = x * x + y * y + z * z - a2;
  const s = Math.sqrt(w * w + 4 * a2 * z * z);
  // r² = (w + s)/2, rewritten to avoid cancellation when w < 0.
  const r2 = w >= 0 ? 0.5 * (w + s) : (2 * a2 * z * z) / (s - w || 1e-300);
  return Math.sqrt(r2);
}

/** Outer and inner horizon radii. */
export function horizons(a) {
  const d = Math.sqrt(Math.max(0, 1 - a * a));
  return { rPlus: 1 + d, rMinus: 1 - d };
}

/** Kerr–Schild scalar f and covector l_μ = (1, kx, ky, kz) at a point. */
export function ksFields(x, y, z, a) {
  const r = ksRadius(x, y, z, a);
  const r2 = r * r;
  const q = r2 + a * a;
  const f = (2 * r2 * r) / (r2 * r2 + a * a * z * z);
  return { r, f, k: [(r * x + a * y) / q, (r * y - a * x) / q, z / r] };
}

/** Covariant metric g_{μν} as a 4×4 array (index 0 = t). */
export function metricLower(pos, a) {
  const { f, k } = ksFields(pos[0], pos[1], pos[2], a);
  const l = [1, k[0], k[1], k[2]];
  const g = [];
  for (let m = 0; m < 4; m++) {
    g.push([]);
    for (let n = 0; n < 4; n++) {
      const eta = m === n ? (m === 0 ? -1 : 1) : 0;
      g[m].push(eta + f * l[m] * l[n]);
    }
  }
  return g;
}

/** Contravariant metric g^{μν}. */
export function metricUpper(pos, a) {
  const { f, k } = ksFields(pos[0], pos[1], pos[2], a);
  const l = [-1, k[0], k[1], k[2]]; // l^μ = η^{μν} l_ν
  const g = [];
  for (let m = 0; m < 4; m++) {
    g.push([]);
    for (let n = 0; n < 4; n++) {
      const eta = m === n ? (m === 0 ? -1 : 1) : 0;
      g[m].push(eta - f * l[m] * l[n]);
    }
  }
  return g;
}

export function lower(g, v) {
  return [0, 1, 2, 3].map((m) => g[m][0] * v[0] + g[m][1] * v[1] + g[m][2] * v[2] + g[m][3] * v[3]);
}

export function dot(g, u, v) {
  let s = 0;
  for (let m = 0; m < 4; m++) for (let n = 0; n < 4; n++) s += g[m][n] * u[m] * v[n];
  return s;
}

/** H = ½ g^{μν} p_μ p_ν for state s = [x, y, z, p_x, p_y, p_z] and p_t = −E. */
export function hamiltonian(s, E, a) {
  const { f, k } = ksFields(s[0], s[1], s[2], a);
  const K = k[0] * s[3] + k[1] * s[4] + k[2] * s[5];
  const kp = E + K;
  return 0.5 * (-E * E + s[3] * s[3] + s[4] * s[4] + s[5] * s[5] - f * kp * kp);
}

/**
 * Hamilton's equations dx^i/dλ = ∂H/∂p_i, dp_i/dλ = −∂H/∂x^i.
 * Writes 6 derivatives into `out` and returns it.
 */
export function geodesicRHS(s, E, a, out = new Array(6)) {
  const x = s[0], y = s[1], z = s[2], px = s[3], py = s[4], pz = s[5];
  const a2 = a * a;
  const r = ksRadius(x, y, z, a);
  const r2 = r * r;
  const z2 = z * z;
  const q = r2 + a2;
  // ∇r from implicit differentiation of r⁴ − (ρ² − a²) r² − a² z² = 0.
  const D = r2 + (a2 * z2) / r2;
  const drdx = (x * r) / D;
  const drdy = (y * r) / D;
  const drdz = (z * q) / (r * D);
  // f = 2r³/(r⁴ + a²z²) and its partials (∂/∂r at fixed z, ∂/∂z at fixed r).
  const den = r2 * r2 + a2 * z2;
  const den2 = den * den;
  const f = (2 * r2 * r) / den;
  const dfdr = (2 * r2 * (3 * a2 * z2 - r2 * r2)) / den2;
  const dfdzExplicit = (-4 * r2 * r * a2 * z) / den2;
  // k and K = k·p.
  const kx = (r * x + a * y) / q;
  const ky = (r * y - a * x) / q;
  const kz = z / r;
  const S = x * px + y * py;
  const A = y * px - x * py;
  const K = kx * px + ky * py + kz * pz;
  const kp = E + K;
  // ∇K: implicit dependence through r plus explicit dependence on x, y, z.
  const dKdr = S / q - (2 * r * (r * S + a * A)) / (q * q) - (z * pz) / r2;
  const dKx = dKdr * drdx + (r * px - a * py) / q;
  const dKy = dKdr * drdy + (r * py + a * px) / q;
  const dKz = dKdr * drdz + pz / r;
  const c = f * kp;
  const h = 0.5 * kp * kp;
  out[0] = px - c * kx;
  out[1] = py - c * ky;
  out[2] = pz - c * kz;
  out[3] = h * dfdr * drdx + c * dKx;
  out[4] = h * dfdr * drdy + c * dKy;
  out[5] = h * (dfdr * drdz + dfdzExplicit) + c * dKz;
  return out;
}

/** dt/dλ = ∂H/∂p_t (positive for future-directed rays). */
export function dtdl(s, E, a) {
  const { f, k } = ksFields(s[0], s[1], s[2], a);
  const kp = E + k[0] * s[3] + k[1] * s[4] + k[2] * s[5];
  return E + f * kp;
}

/** Axial angular momentum L = p_φ = x p_y − y p_x (exactly conserved). */
export function angularMomentum(s) {
  return s[0] * s[4] - s[1] * s[3];
}

/**
 * Carter constant Q = p_θ² + cos²θ (L²/sin²θ − a² E²).
 * θ is the spheroidal angle with cos θ = z/r, and p_θ = ∂x^i/∂θ p_i at fixed
 * (r, φ); p_θ is identical in Kerr–Schild and Boyer–Lindquist coordinates.
 * Not built into the integrator, so its conservation is a genuine test.
 */
export function carterQ(s, E, a) {
  const r = ksRadius(s[0], s[1], s[2], a);
  const c = s[2] / r;
  const sn2 = Math.max(1e-300, 1 - c * c);
  const sn = Math.sqrt(sn2);
  const pth = ((s[0] * s[3] + s[1] * s[4]) * c) / sn - r * sn * s[5];
  const L = angularMomentum(s);
  return pth * pth + c * c * ((L * L) / sn2 - a * a * E * E);
}

/** Boyer–Lindquist-style radius, polar angle and KS azimuth of a point. */
export function spheroidal(x, y, z, a) {
  const r = ksRadius(x, y, z, a);
  return { r, theta: Math.acos(Math.max(-1, Math.min(1, z / r))), phi: Math.atan2(y, x) };
}

/** Cartesian KS point from spheroidal (r, θ, φ). */
export function fromSpheroidal(r, theta, phi, a) {
  const R = Math.sqrt(r * r + a * a) * Math.sin(theta);
  return [R * Math.cos(phi), R * Math.sin(phi), r * Math.cos(theta)];
}

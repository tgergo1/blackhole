// Timelike geodesics, parallel transport and curvature of Kerr in Cartesian
// Kerr–Schild coordinates (G = c = M = 1). Companion to kerr.js, which holds
// the metric and the Hamiltonian geodesic equations used here unchanged:
// H = ½ g^{μν} p_μ p_ν is the same function for photons (H = 0) and for
// massive bodies (H = −½ with unit rest mass, so the affine parameter is the
// proper time τ).
//
// Contents
//  - christoffel(): analytic Γ^μ_{αβ} from g = η + f l l (no finite differences)
//  - Weyl/Riemann tensor from the Petrov type-D structure of Kerr: in the
//    canonical frame the only curvature is Ψ₂ = −M/(r − i a cosθ)³
//  - tidalTensor(): E_ij = R_{μανβ} e_i^μ u^α e_j^ν u^β for any observer
//  - timelike geodesic + parallel-transported tetrad integrator
//  - the time-reversal map from physical (ingoing) coordinates to the mirror
//    spacetime used by the ray tracer (see camera.js)
//  - Boyer–Lindquist constants of motion → Kerr–Schild initial data

import { ksRadius, ksFields, metricLower, metricUpper, geodesicRHS, dtdl, horizons, lower, dot } from './kerr.js';
import { Dopri5 } from './ode.js';

const ETA = [-1, 1, 1, 1];

/** ∇r (spatial gradient of the Kerr–Schild radius). */
export function gradR(x, y, z, a) {
  const r = ksRadius(x, y, z, a);
  const a2 = a * a;
  const D = r * r + (a2 * z * z) / (r * r);
  return { r, g: [(x * r) / D, (y * r) / D, (z * (r * r + a2)) / (r * D)] };
}

/**
 * Christoffel symbols Γ^μ_{αβ} at a point (index 0 = t). Analytic: with
 * h_{μν} = f l_μ l_ν and ∂_t h = 0, Γ_{λμν} = ½(∂_μ h_{λν} + ∂_ν h_{λμ} − ∂_λ h_{μν}).
 */
export function christoffel(pos, a) {
  const [x, y, z] = pos;
  const a2 = a * a;
  const { r, g: dr } = gradR(x, y, z, a);
  const r2 = r * r, q = r2 + a2, z2 = z * z;
  const den = r2 * r2 + a2 * z2;
  const f = (2 * r2 * r) / den;
  const dfdr = (2 * r2 * (3 * a2 * z2 - r2 * r2)) / (den * den);
  const dfdz = (-4 * r2 * r * a2 * z) / (den * den);
  const l = [1, (r * x + a * y) / q, (r * y - a * x) / q, z / r];
  // dl[j][μ] = ∂_j l_μ for spatial j = 1..3 (index 0 unused: ∂_t = 0).
  const df = [0, 0, 0, 0];
  const dl = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let j = 1; j <= 3; j++) {
    const drj = dr[j - 1];
    df[j] = dfdr * drj + (j === 3 ? dfdz : 0);
    const dx = j === 1 ? 1 : 0, dy = j === 2 ? 1 : 0, dz = j === 3 ? 1 : 0;
    dl[j][1] = (drj * x + r * dx + a * dy) / q - ((r * x + a * y) * 2 * r * drj) / (q * q);
    dl[j][2] = (drj * y + r * dy - a * dx) / q - ((r * y - a * x) * 2 * r * drj) / (q * q);
    dl[j][3] = dz / r - (z * drj) / r2;
  }
  // dh[α][μ][ν] = ∂_α h_{μν}
  const dh = [];
  for (let al = 0; al < 4; al++) {
    const m = [];
    for (let mu = 0; mu < 4; mu++) {
      const row = [];
      for (let nu = 0; nu < 4; nu++) {
        row.push(al === 0 ? 0 : df[al] * l[mu] * l[nu] + f * (dl[al][mu] * l[nu] + l[mu] * dl[al][nu]));
      }
      m.push(row);
    }
    dh.push(m);
  }
  const gi = metricUpper(pos, a);
  const G = [];
  for (let mu = 0; mu < 4; mu++) {
    const M = [];
    for (let al = 0; al < 4; al++) {
      const row = [];
      for (let be = 0; be < 4; be++) {
        let s = 0;
        for (let la = 0; la < 4; la++) {
          s += gi[mu][la] * 0.5 * (dh[al][la][be] + dh[be][la][al] - dh[la][al][be]);
        }
        row.push(s);
      }
      M.push(row);
    }
    G.push(M);
  }
  return G;
}

// ---------- Curvature ----------

function normalizeVec(g, v) {
  const n = dot(g, v, v);
  return v.map((c) => c / Math.sqrt(Math.abs(n)));
}
function sub(u, v, s) {
  return u.map((c, i) => c - s * v[i]);
}
function det4(m) {
  // m[i] are column vectors; determinant by cofactor expansion.
  const M = (i, j) => m[j][i];
  let d = 0;
  for (let c = 0; c < 4; c++) {
    const cols = [0, 1, 2, 3].filter((k) => k !== c);
    const minor =
      M(1, cols[0]) * (M(2, cols[1]) * M(3, cols[2]) - M(2, cols[2]) * M(3, cols[1])) -
      M(1, cols[1]) * (M(2, cols[0]) * M(3, cols[2]) - M(2, cols[2]) * M(3, cols[0])) +
      M(1, cols[2]) * (M(2, cols[0]) * M(3, cols[1]) - M(2, cols[1]) * M(3, cols[0]));
    d += (c % 2 ? -1 : 1) * M(0, c) * minor;
  }
  return d;
}

/**
 * Orthonormal frame adapted to Kerr's principal (canonical) 2-planes,
 * built only from horizon-regular objects so it stays valid on and inside
 * r₊ (away from the ring singularity and the axis):
 *  - timelike plane span(V, l), V = (r²+a²)∂_t + a∂_φ, l^μ = g^{μν} l_ν the
 *    Kerr–Schild principal null direction;
 *  - spacelike plane span(∇θ, W), W = a sin²θ ∂_t + ∂_φ.
 * Weyl is invariant under boosts in the first plane and rotations in the
 * second, so any orthonormal basis of each plane works; the orientation is
 * fixed so that the frame reduces to the Carter frame (t, r, θ, φ) outside.
 * `uHint` (a timelike vector) selects e0 within the timelike plane.
 */
export function canonicalFrame(pos, a, uHint) {
  const [x, y, z] = pos;
  const g = metricLower(pos, a);
  const gi = metricUpper(pos, a);
  const { r, g: dr } = gradR(x, y, z, a);
  const cth = z / r;
  const sth2 = Math.max(1e-300, 1 - cth * cth);
  const { k } = ksFields(x, y, z, a);
  const lLow = [1, k[0], k[1], k[2]];
  const lUp = [0, 1, 2, 3].map((m) => gi[m][0] * lLow[0] + gi[m][1] * lLow[1] + gi[m][2] * lLow[2] + gi[m][3] * lLow[3]);
  const V = [r * r + a * a, -a * y, a * x, 0];
  const W = [a * sth2, -y, x, 0];
  // ∇θ with θ = acos(z/r).
  const sth = Math.sqrt(sth2);
  const dth = [0, 0, 0, 0];
  for (let j = 0; j < 3; j++) dth[j + 1] = (-1 / sth) * ((j === 2 ? 1 / r : 0) - (z * dr[j]) / (r * r));
  const thUp = [0, 1, 2, 3].map((m) => gi[m][0] * dth[0] + gi[m][1] * dth[1] + gi[m][2] * dth[2] + gi[m][3] * dth[3]);

  // Timelike plane span(V, l). Its second null direction is n = V − (V·V)/(2V·l) l,
  // so n − sign(V·l) l is timelike everywhere (also inside the horizon).
  const gVV = dot(g, V, V), gVl = dot(g, V, lUp), gll = dot(g, lUp, lUp);
  const n = V.map((c, i) => c - (gVV / (2 * gVl)) * lUp[i]);
  let e0 = n.map((c, i) => c - Math.sign(gVl) * lUp[i]);
  if (uHint) {
    // Project the hint onto the plane; use it if the projection is timelike.
    const bu = [dot(g, uHint, V), dot(g, uHint, lUp)];
    const detG = gVV * gll - gVl * gVl;
    const cV = (bu[0] * gll - bu[1] * gVl) / detG;
    const cl = (bu[1] * gVV - bu[0] * gVl) / detG;
    const p = V.map((c, i) => cV * c + cl * lUp[i]);
    if (dot(g, p, p) < 0) e0 = p;
  }
  e0 = normalizeVec(g, e0);
  // Future-directed: l^μ = (−1, k) is past-directed null, so e0·l > 0.
  if (dot(g, e0, lUp) < 0) e0 = e0.map((c) => -c);
  let e1 = normalizeVec(g, sub(lUp, e0, -dot(g, lUp, e0)));
  // Orientation: l^μ = (−1, k) is past-directed, ∝ e1 − e0 in the Carter
  // frame with e1 outward, so e0∧e1 ∝ +V∧l, i.e. (e0∧e1):(V∧l) < 0.
  const or = dot(g, e0, V) * dot(g, e1, lUp) - dot(g, e0, lUp) * dot(g, e1, V);
  if (or > 0) e1 = e1.map((c) => -c);
  // Spacelike plane.
  let e2 = thUp.slice();
  for (const e of [e0, e1]) e2 = sub(e2, e, dot(g, e2, e) * (e === e0 ? -1 : 1));
  e2 = normalizeVec(g, e2);
  let e3 = W.slice();
  for (const [e, s] of [[e0, -1], [e1, 1], [e2, 1]]) e3 = sub(e3, e, dot(g, e3, e) * s);
  e3 = normalizeVec(g, e3);
  if (det4([e0, e1, e2, e3]) < 0) e3 = e3.map((c) => -c);
  // Ψ₂ = −M/(r − i a cosθ)³
  const re = r, im = -a * cth;
  const m2 = re * re + im * im;
  // 1/(re + i im)³ = conj(w)³/|w|⁶
  const cr = re, ci = -im;
  const c3r = cr * cr * cr - 3 * cr * ci * ci;
  const c3i = 3 * cr * cr * ci - ci * ci * ci;
  const psi2 = [-c3r / (m2 * m2 * m2), -c3i / (m2 * m2 * m2)];
  return { frame: [e0, e1, e2, e3], psi2, r, g };
}

const LEVI = (() => {
  const e = [];
  for (let i = 0; i < 3; i++) {
    e.push([]);
    for (let j = 0; j < 3; j++) {
      e[i].push([]);
      for (let k = 0; k < 3; k++) e[i][j].push(i === j || j === k || i === k ? 0 : ((j - i + 3) % 3 === 1 ? 1 : -1));
    }
  }
  return e;
})();

/**
 * Vacuum Riemann tensor R_{abcd} in an orthonormal frame from its electric
 * (E_ij = R_{0i0j}) and magnetic (B_ij) parts:
 *   R_{0ijk} = ε_{jkm} B_{mi},   R_{ijkl} = −ε_{ijm} ε_{kln} E_{mn}.
 */
export function riemannFromEB(E, B) {
  const R = new Float64Array(256);
  const pair = (a, b) => (a === 0 ? [1, b - 1] : b === 0 ? [-1, a - 1] : null);
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) for (let c = 0; c < 4; c++) for (let d = 0; d < 4; d++) {
    if (a === b || c === d) continue;
    const p1 = pair(a, b), p2 = pair(c, d);
    let v = 0;
    if (p1 && p2) v = p1[0] * p2[0] * E[p1[1]][p2[1]];
    else if (p1) {
      const j = c - 1, k = d - 1;
      for (let m = 0; m < 3; m++) v += LEVI[j][k][m] * B[m][p1[1]];
      v *= p1[0];
    } else if (p2) {
      const j = a - 1, k = b - 1;
      for (let m = 0; m < 3; m++) v += LEVI[j][k][m] * B[m][p2[1]];
      v *= p2[0];
    } else {
      const i = a - 1, j = b - 1, k = c - 1, l = d - 1;
      for (let m = 0; m < 3; m++) for (let n = 0; n < 3; n++) v -= LEVI[i][j][m] * LEVI[k][l][n] * E[m][n];
    }
    R[((a * 4 + b) * 4 + c) * 4 + d] = v;
  }
  return R;
}

/** Riemann tensor R_{abcd} in the canonical frame (type D: E + iB ∝ Ψ₂ diag(2, −1, −1)). */
export function canonicalRiemann(psi2) {
  const [A, Bi] = psi2;
  const E = [[2 * A, 0, 0], [0, -A, 0], [0, 0, -A]];
  // Magnetic part sign fixed by the orientation conventions of riemannFromEB
  // and canonicalFrame; verified against finite differences of Γ in the tests.
  const B = [[-2 * Bi, 0, 0], [0, Bi, 0], [0, 0, Bi]];
  return riemannFromEB(E, B);
}

/** Frame components v^a = η^{aa} g(v, e_a) of a coordinate vector. */
export function frameComponents(g, frame, v) {
  const lv = lower(g, v);
  return frame.map((e, a) => ETA[a] * (lv[0] * e[0] + lv[1] * e[1] + lv[2] * e[2] + lv[3] * e[3]));
}

/** Covariant Riemann tensor in coordinates, R_{μνρσ} (flat Float64Array, 256). */
export function riemannCoord(pos, a) {
  const { frame, psi2, g } = canonicalFrame(pos, a);
  const Rf = canonicalRiemann(psi2);
  // ω^a_μ = η^{aa} e_a^ν g_{νμ}
  const om = frame.map((e, i) => lower(g, e).map((c) => ETA[i] * c));
  const out = new Float64Array(256);
  // Transform one index at a time (4 × 4⁵ operations).
  let T = Rf;
  for (let slot = 0; slot < 4; slot++) {
    const N = new Float64Array(256);
    const stride = [64, 16, 4, 1][slot];
    for (let idx = 0; idx < 256; idx++) {
      const mu = Math.floor(idx / stride) % 4;
      const base = idx - mu * stride;
      let s = 0;
      for (let aI = 0; aI < 4; aI++) s += om[aI][mu] * T[base + aI * stride];
      N[idx] = s;
    }
    T = N;
  }
  out.set(T);
  return out;
}

/**
 * Tidal tensor E_ij = R_{μανβ} e_i^μ u^α e_j^ν u^β seen by an observer with
 * four-velocity u and orthonormal spatial triad e_1..e_3 (coordinate
 * components). Geodesic deviation: d²ξ^i/dτ² = −E_ij ξ^j.
 */
export function tidalTensor(pos, a, u, triad) {
  const { frame, psi2, g } = canonicalFrame(pos, a, u);
  const R = canonicalRiemann(psi2);
  const uc = frameComponents(g, frame, u);
  const ec = triad.map((e) => frameComponents(g, frame, e));
  // M_{b d} contracted with u on slots 2 and 4: K_{ac} = R_{abcd} u^b u^d.
  const K = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let a1 = 0; a1 < 4; a1++) for (let c = 0; c < 4; c++) {
    let s = 0;
    for (let b = 0; b < 4; b++) for (let d = 0; d < 4; d++) s += R[((a1 * 4 + b) * 4 + c) * 4 + d] * uc[b] * uc[d];
    K[a1][c] = s;
  }
  const out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0;
    for (let a1 = 0; a1 < 4; a1++) for (let c = 0; c < 4; c++) s += K[a1][c] * ec[i][a1] * ec[j][c];
    out[i][j] = s;
  }
  return out;
}

/** Eigenvalues/vectors of a symmetric 3×3 matrix (cyclic Jacobi). Ascending. */
export function symEig3(M) {
  const A = M.map((r) => r.slice());
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = A[0][1] ** 2 + A[0][2] ** 2 + A[1][2] ** 2;
    if (off < 1e-30 * (A[0][0] ** 2 + A[1][1] ** 2 + A[2][2] ** 2 + 1e-300)) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = A[k][p], akq = A[k][q];
        A[k][p] = c * akp - s * akq;
        A[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = A[p][k], aqk = A[q][k];
        A[p][k] = c * apk - s * aqk;
        A[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq;
        V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const idx = [0, 1, 2].sort((i, j) => A[i][i] - A[j][j]);
  return { values: idx.map((i) => A[i][i]), vectors: idx.map((i) => [V[0][i], V[1][i], V[2][i]]) };
}

// ---------- Timelike geodesics with a parallel-transported tetrad ----------

/** Four-velocity u^μ = g^{μν} p_ν from KS state [x,y,z,p_x,p_y,p_z] and E = −p_t. */
export function fourVelocity(s, E, a) {
  const d = geodesicRHS(s, E, a, new Array(6));
  return [dtdl(s, E, a), d[0], d[1], d[2]];
}

/**
 * Right-hand side for the extended state
 *   Y = [t, x, y, z, p_x, p_y, p_z, e1^μ (4), e2^μ (4), e3^μ (4)]
 * with respect to proper time. The momentum is per unit rest mass (u·u = −1).
 * nTriad = 0 integrates the bare geodesic.
 */
export function makeTimelikeRHS(E, a, nTriad = 3) {
  const s = new Array(6), d = new Array(6);
  return (Y, out) => {
    s[0] = Y[1]; s[1] = Y[2]; s[2] = Y[3]; s[3] = Y[4]; s[4] = Y[5]; s[5] = Y[6];
    geodesicRHS(s, E, a, d);
    out[0] = dtdl(s, E, a);
    for (let i = 0; i < 6; i++) out[i + 1] = d[i];
    if (nTriad) {
      const G = christoffel(s, a);
      const u = [out[0], d[0], d[1], d[2]];
      for (let k = 0; k < nTriad; k++) {
        const o = 7 + 4 * k;
        for (let mu = 0; mu < 4; mu++) {
          let acc = 0;
          for (let al = 0; al < 4; al++) {
            const ua = u[al];
            if (ua === 0) continue;
            const Gm = G[mu][al];
            acc += ua * (Gm[0] * Y[o] + Gm[1] * Y[o + 1] + Gm[2] * Y[o + 2] + Gm[3] * Y[o + 3]);
          }
          out[o + mu] = -acc;
        }
      }
    }
  };
}

/** Error scale: positions by r, momenta and tetrads O(1), time by |t| + r. */
export function timelikeScale(a) {
  return (i, y0, y1) => {
    if (i === 0) return Math.max(1, Math.abs(y0[0]));
    if (i <= 3) return Math.max(1, Math.hypot(y0[1], y0[2], y0[3]));
    return Math.max(1, Math.abs(y0[i]), Math.abs(y1[i]));
  };
}

/** Build an orthonormal triad orthogonal to u from coordinate axes. */
export function triadFor(pos, a, u) {
  const g = metricLower(pos, a);
  const basis = [{ e: u, n: -1 }];
  const out = [];
  for (const axis of [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) {
    let w = axis.slice();
    for (const { e, n } of basis) w = sub(w, e, dot(g, w, e) / n);
    w = normalizeVec(g, w);
    basis.push({ e: w, n: 1 });
    out.push(w);
  }
  return out;
}

/** Re-orthonormalise a drifting tetrad (Gram–Schmidt against u, in order). */
export function reorthonormalise(pos, a, u, triad) {
  const g = metricLower(pos, a);
  const basis = [{ e: u, n: -1 }];
  return triad.map((v) => {
    let w = v.slice();
    for (const { e, n } of basis) w = sub(w, e, dot(g, w, e) / n);
    w = normalizeVec(g, w);
    basis.push({ e: w, n: 1 });
    return w;
  });
}

/** Pack a timelike state and triad into the extended vector Y. */
export function packState(t, s, triad) {
  const Y = new Float64Array(7 + 4 * (triad ? triad.length : 0));
  Y[0] = t;
  for (let i = 0; i < 6; i++) Y[i + 1] = s[i];
  if (triad) triad.forEach((e, k) => { for (let m = 0; m < 4; m++) Y[7 + 4 * k + m] = e[m]; });
  return Y;
}
export function unpackTriad(Y) {
  const n = (Y.length - 7) / 4;
  const out = [];
  for (let k = 0; k < n; k++) out.push([Y[7 + 4 * k], Y[8 + 4 * k], Y[9 + 4 * k], Y[10 + 4 * k]]);
  return out;
}

/** Integrator for a timelike geodesic (optionally with a transported triad). */
export function timelikeIntegrator(t0, s0, E, a, { triad = null, rtol = 1e-11, atol = 1e-13, h0 = 0.05 } = {}) {
  const Y = packState(t0, s0, triad);
  return new Dopri5(makeTimelikeRHS(E, a, triad ? triad.length : 0), Y, { rtol, atol, h0, scaleFn: timelikeScale(a) });
}

// ---------- Boyer–Lindquist constants of motion ----------

/** Radial potential R(r) for unit rest mass (μ = 1; μ = 0 for photons). */
export function radialPotential(r, E, Lz, Q, a, mu2 = 1) {
  const D = r * r - 2 * r + a * a;
  return (E * (r * r + a * a) - a * Lz) ** 2 - D * (mu2 * r * r + (Lz - a * E) ** 2 + Q);
}
/** Polar potential Θ(θ). */
export function polarPotential(theta, E, Lz, Q, a, mu2 = 1) {
  const c = Math.cos(theta), s2 = Math.max(1e-300, Math.sin(theta) ** 2);
  return Q - c * c * (a * a * (mu2 - E * E) + (Lz * Lz) / s2);
}

/** Carter constant for a timelike state (unit rest mass). */
export function carterQTimelike(s, E, a, mu2 = 1) {
  const r = ksRadius(s[0], s[1], s[2], a);
  const c = s[2] / r;
  const sn2 = Math.max(1e-300, 1 - c * c);
  const sn = Math.sqrt(sn2);
  const pth = ((s[0] * s[3] + s[1] * s[4]) * c) / sn - r * sn * s[5];
  const L = s[0] * s[4] - s[1] * s[3];
  return pth * pth + c * c * (a * a * (mu2 - E * E) + (L * L) / sn2);
}

/** ∫ 2r/Δ dr (ingoing − BL time offset), with a fixed integration constant. */
export function timeShift(r, a) {
  const { rPlus: rp, rMinus: rm } = horizons(a);
  const d = rp - rm;
  if (d < 1e-9) throw new Error('timeShift: extremal spin not supported');
  return ((2 * rp) / d) * Math.log(Math.abs(r - rp)) - ((2 * rm) / d) * (rm > 0 ? Math.log(Math.abs(r - rm)) : 0);
}
/** ∫ a/Δ dr (ingoing − BL azimuth offset). */
export function phiShift(r, a) {
  const { rPlus: rp, rMinus: rm } = horizons(a);
  const d = rp - rm;
  return (a / d) * Math.log(Math.abs((r - rp) / (r - rm)));
}

/**
 * Kerr–Schild state at spheroidal (r, θ, φ) from BL constants (E, Lz, Q)
 * and the signs of the radial and polar motion (sr = −1 ingoing).
 * Returns { s: [x,y,z,p_x,p_y,p_z], E }.
 */
export function stateFromConstants(r, theta, phi, E, Lz, Q, a, sr = -1, sth = 1, mu2 = 1) {
  const D = r * r - 2 * r + a * a;
  const R = Math.max(0, radialPotential(r, E, Lz, Q, a, mu2));
  const Th = Math.max(0, polarPotential(theta, E, Lz, Q, a, mu2));
  const prBL = (sr * Math.sqrt(R)) / D;
  const pth = sth * Math.sqrt(Th);
  // KS radial covector: p_r(KS) = p_r(BL) + (2rE − a Lz)/Δ (t_in = t_BL + ∫2r/Δ, φ_in = φ_BL + ∫a/Δ).
  const pr = prBL + (2 * r * E - a * Lz) / D;
  const R0 = Math.sqrt(r * r + a * a) * Math.sin(theta);
  const pos = [R0 * Math.cos(phi), R0 * Math.sin(phi), r * Math.cos(theta)];
  const { g: dr } = gradR(pos[0], pos[1], pos[2], a);
  const sn = Math.sin(theta);
  const dth = [0, 1, 2].map((j) => (-1 / sn) * ((j === 2 ? 1 / r : 0) - (pos[2] * dr[j]) / (r * r)));
  // φ_KS = atan2(y, x) − atan(a/r), i.e. x + iy = (r + ia) e^{iφ} sinθ for this l_μ.
  const rho2 = pos[0] * pos[0] + pos[1] * pos[1];
  const dph = [0, 1, 2].map((j) => (j === 0 ? -pos[1] / rho2 : j === 1 ? pos[0] / rho2 : 0) + (a / (r * r + a * a)) * dr[j]);
  const p = [0, 1, 2].map((j) => pr * dr[j] + pth * dth[j] + Lz * dph[j]);
  return { s: [pos[0], pos[1], pos[2], p[0], p[1], p[2]], E };
}

/**
 * Largest turning point of the radial motion below r0 (pericentre), or 0 if
 * the orbit plunges. Scans R(r) inwards and refines by bisection.
 */
export function pericentre(E, Lz, Q, a, r0) {
  const rp = horizons(a).rPlus;
  if (radialPotential(r0, E, Lz, Q, a) < 0) return NaN; // r0 is not on the orbit
  const n = 4000;
  let prev = r0;
  for (let i = 1; i <= n; i++) {
    const r = rp + (r0 - rp) * (1 - i / n) ** 2;
    if (radialPotential(r, E, Lz, Q, a) < 0) {
      let lo = r, hi = prev;
      for (let k = 0; k < 100; k++) {
        const m = 0.5 * (lo + hi);
        if (radialPotential(m, E, Lz, Q, a) < 0) lo = m; else hi = m;
      }
      return hi;
    }
    prev = r;
  }
  return 0;
}

/**
 * Constants (E, Lz, Q) of an orbit with energy E, pericentre rp and
 * inclination ι (cos ι = Lz/√(Lz² + Q); ι > 90° is retrograde).
 */
export function orbitConstants({ E = 1, rp, incl = 0, a, rStart = 1e3 }) {
  const ci = Math.cos(incl), si2 = Math.sin(incl) ** 2;
  const rH = horizons(a).rPlus;
  const R = (r, L) => radialPotential(r, E, L * ci, L * L * si2, a);
  // Does an orbit with this L travel freely from rStart down to r1?
  const clear = (L, r1) => {
    for (let i = 0; i <= 2000; i++) {
      const r = r1 + ((rStart - r1) * i) / 2000;
      if (R(r, L) <= 0) return false;
    }
    return true;
  };
  // Capture: the largest L that still plunges (whirls near the unstable orbits).
  const capture = () => {
    let l0 = 0, l1 = 1;
    while (clear(l1, rH) && l1 < 1e4) l1 *= 2;
    for (let k = 0; k < 60; k++) { const m = 0.5 * (l0 + l1); if (clear(m, rH)) l0 = m; else l1 = m; }
    const Lp = 0.995 * l0;
    return { E, Lz: Lp * ci, Q: Lp * Lp * si2, L: Lp, plunge: true };
  };
  if (rp <= rH * 1.0001) return capture();
  if (R(rp, 0) <= 0) return { E, Lz: 0, Q: 0, L: 0, plunge: true };
  // As L grows from 0 the turning point first reaches rp at the smallest
  // root of R(rp; L): scan upwards, then bisect.
  let lo = 0, hi = 0;
  const dL = 0.01 * Math.sqrt(rp);
  while (R(rp, hi) > 0) { lo = hi; hi += dL; if (hi > 1e4) return capture(); }
  for (let k = 0; k < 200; k++) {
    const m = 0.5 * (lo + hi);
    if (R(rp, m) > 0) lo = m; else hi = m;
  }
  const L = lo;
  if (rStart > rp && R(rStart, L) < 0) throw new Error('orbitConstants: rStart lies beyond the apocentre');
  // A potential barrier between rStart and rp means rp is inside the capture radius.
  if (!clear(L, rp * (1 + 1e-9))) return capture();
  return { E, Lz: L * ci, Q: L * L * si2, L, plunge: false };
}

// ---------- Time-reversal map into the ray tracer's mirror spacetime ----------
//
// Physical spacetime: ingoing Kerr–Schild coordinates of Kerr(a). The ray
// tracer works in ingoing Kerr–Schild coordinates of Kerr(−a), obtained by
// t → −t; there the image of the physical point has
//   t' = −(t − 2∫2r/Δ dr),  x' + iy' = (x + iy) (r − ia)/(r + ia) e^{−2iΦ(r)},  z' = z
// (Φ = ∫ a/Δ dr): ingoing coordinates of Kerr(−a) are the time-reversed
// *outgoing* coordinates of Kerr(a). The map is an isometry of the exterior.
// Future-directed physical vectors become past-directed; we return −T_*(u)
// so that emitter four-velocities are future-directed like the observer's.

export function mirrorPoint(t, x, y, z, a) {
  const r = ksRadius(x, y, z, a);
  const P = phiShift(r, a);
  // (r − ia)/(r + ia) = e^{−2i atan(a/r)}
  const ang = -2 * Math.atan2(a, r) - 2 * P;
  const c = Math.cos(ang), s = Math.sin(ang);
  return { tp: -(t - 2 * timeShift(r, a)), x: c * x - s * y, y: s * x + c * y, z, r };
}

/** Push a vector v^μ at physical point (x, y, z) forward to the mirror. */
export function mirrorVector(x, y, z, a, v) {
  const { r, g: dr } = gradR(x, y, z, a);
  const D = r * r - 2 * r + a * a;
  const P = phiShift(r, a);
  const ang = -2 * Math.atan2(a, r) - 2 * P;
  const c = Math.cos(ang), s = Math.sin(ang);
  const vr = dr[0] * v[1] + dr[1] * v[2] + dr[2] * v[3];
  // d ang/dr = 2a/(r²+a²) − 2a/Δ
  const dang = ((2 * a) / (r * r + a * a) - (2 * a) / D) * vr;
  // d(rotated point) = R dζ + i ζ' dang, ζ' = rotated point
  const xr = c * x - s * y, yr = s * x + c * y;
  return [
    -v[0] + ((4 * r) / D) * vr,
    c * v[1] - s * v[2] - yr * dang,
    s * v[1] + c * v[2] + xr * dang,
    v[3],
  ];
}

/** Inverse of mirrorPoint (mirror spatial point → physical spatial point). */
export function physicalPoint(xm, ym, zm, a) {
  const r = ksRadius(xm, ym, zm, a);
  const ang = 2 * Math.atan2(a, r) + 2 * phiShift(r, a);
  const c = Math.cos(ang), s = Math.sin(ang);
  return [c * xm - s * ym, s * xm + c * ym, zm];
}

// Self-gravitating stars as smoothed-particle hydrodynamics (SPH).
//
// Units inside this file: G = 1, stellar mass M* = 1, stellar radius R* = 1,
// time unit t* = √(R*³/GM*). The star lives in the Fermi normal coordinates
// of its centre of mass (see tde.js), where the black hole enters only
// through the tidal acceleration −C ξ.
//
// Hydrodynamics follows the standard formulation (Monaghan 1992; Springel &
// Hernquist 2002 entropy variable; Price 2012 review):
//  - M4 cubic-spline kernel, h = η (m/ρ)^{1/3} with η = 1.2
//  - adiabatic ideal gas P = A ρ^γ, entropy A only raised by shocks
//  - Monaghan artificial viscosity (α = 1, β = 2) with the Balsara switch
//  - self-gravity by direct summation, softened with the same spline kernel
//    (Hernquist & Katz 1989), so it is exact Newtonian gravity beyond 2h
//  - kick–drift–kick leapfrog with Courant and acceleration limits

/** Lane–Emden solution for polytropic index n (RK4 in ξ). */
export function laneEmden(n, dxi = 2e-4) {
  // θ'' = −θⁿ − 2θ'/ξ, series start θ = 1 − ξ²/6 + nξ⁴/120.
  let xi = 1e-3;
  let th = 1 - (xi * xi) / 6 + (n * xi ** 4) / 120;
  let dth = -xi / 3 + (n * xi ** 3) / 30;
  const xs = [0], ths = [1], ms = [0];
  const f = (x, t, d) => [d, -Math.pow(Math.max(t, 0), n) - (2 * d) / x];
  while (th > 0) {
    const k1 = f(xi, th, dth);
    const k2 = f(xi + dxi / 2, th + (dxi / 2) * k1[0], dth + (dxi / 2) * k1[1]);
    const k3 = f(xi + dxi / 2, th + (dxi / 2) * k2[0], dth + (dxi / 2) * k2[1]);
    const k4 = f(xi + dxi, th + dxi * k3[0], dth + dxi * k3[1]);
    const thN = th + (dxi / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    const dthN = dth + (dxi / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
    if (thN <= 0) {
      // Linear interpolation of the surface.
      const fr = th / (th - thN);
      const xi1 = xi + fr * dxi;
      const d1 = dth + fr * (dthN - dth);
      xs.push(xi1); ths.push(0); ms.push(-xi1 * xi1 * d1);
      return { n, xi1, dtheta1: d1, xi: xs, theta: ths, mass: ms };
    }
    xi += dxi; th = thN; dth = dthN;
    xs.push(xi); ths.push(th); ms.push(-xi * xi * dth);
  }
  throw new Error('laneEmden: no surface');
}

/** Normalised polytrope (R = M = G = 1): density, pressure and enclosed mass vs r. */
export function polytrope(n) {
  const le = laneEmden(n);
  const Mtot = le.mass[le.mass.length - 1];
  const rhoC = le.xi1 / (3 * -le.dtheta1) * (3 / (4 * Math.PI)); // ρ_c/ρ̄ × ρ̄ with ρ̄ = 3/4π
  // K from the central condition: R = ξ₁ √((n+1)K ρ_c^{1/n−1} / 4π)
  const K = (4 * Math.PI * Math.pow(rhoC, 1 - 1 / n)) / ((n + 1) * le.xi1 * le.xi1);
  const N = le.xi.length;
  const interp = (arr, r) => {
    const x = r * le.xi1;
    if (x >= le.xi1) return arr[N - 1];
    let lo = 0, hi = N - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (le.xi[m] <= x) lo = m; else hi = m; }
    const t = (x - le.xi[lo]) / (le.xi[hi] - le.xi[lo]);
    return arr[lo] + t * (arr[hi] - arr[lo]);
  };
  return {
    n, K, rhoC, xi1: le.xi1,
    rho: (r) => rhoC * Math.pow(Math.max(interp(le.theta, r), 0), n),
    pressure: (r) => K * Math.pow(rhoC * Math.pow(Math.max(interp(le.theta, r), 0), n), 1 + 1 / n),
    mass: (r) => interp(le.mass, r) / Mtot,
    /** Radius enclosing mass fraction q (bisection). */
    radiusOfMass(q) {
      let lo = 0, hi = 1;
      for (let k = 0; k < 60; k++) { const m = 0.5 * (lo + hi); if (this.mass(m) < q) lo = m; else hi = m; }
      return 0.5 * (lo + hi);
    },
  };
}

// Small deterministic PRNG (mulberry32) so runs are reproducible.
export function rng(seed = 1) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SIGMA = 1 / Math.PI;
function kernelW(r, h) {
  const q = r / h, s = SIGMA / (h * h * h);
  if (q < 1) return s * (1 - 1.5 * q * q + 0.75 * q * q * q);
  if (q < 2) { const t = 2 - q; return s * 0.25 * t * t * t; }
  return 0;
}
/** dW/dr. */
function kernelDW(r, h) {
  const q = r / h, s = SIGMA / (h * h * h * h);
  if (q < 1) return s * (-3 * q + 2.25 * q * q);
  if (q < 2) { const t = 2 - q; return -s * 0.75 * t * t; }
  return 0;
}
/** Softened gravity: force per unit mass² at separation r with softening h (→ 1/r² beyond 2h). */
export function gravForce(r, h) {
  const q = r / h;
  if (q >= 2) return 1 / (r * r);
  const q2 = q * q, q3 = q2 * q, q4 = q3 * q;
  const g = q < 1 ? (4 / 3) * q - 1.2 * q3 + 0.5 * q4 : (8 / 3) * q - 3 * q2 + 1.2 * q3 - q4 / 6 - 1 / (15 * q2);
  return g / (h * h);
}
/** Softened potential (negative). */
export function gravPotential(r, h) {
  const q = r / h;
  if (q >= 2) return -1 / r;
  const q2 = q * q, q3 = q2 * q, q4 = q3 * q, q5 = q4 * q;
  const p = q < 1
    ? (2 / 3) * q2 - 0.3 * q4 + 0.1 * q5 - 1.4
    : (4 / 3) * q2 - q3 + 0.3 * q4 - q5 / 30 - 1.6 + 1 / (15 * q);
  return p / h;
}

/**
 * Particle star. Positions/velocities are Float64Array(3N); all particles
 * have equal mass 1/N. `alive` lets particles leave (to the debris phase).
 */
export class SPHStar {
  constructor({ N, gamma = 5 / 3, eta = 1.2, alpha = 1, beta = 2 }) {
    this.N = N;
    this.gamma = gamma;
    this.eta = eta;
    this.alpha = alpha;
    this.beta = beta;
    this.m = 1 / N;
    this.pos = new Float64Array(3 * N);
    this.vel = new Float64Array(3 * N);
    this.acc = new Float64Array(3 * N);
    this.A = new Float64Array(N);   // entropy P/ρ^γ
    this.dA = new Float64Array(N);
    this.rho = new Float64Array(N);
    this.h = new Float64Array(N).fill(0.1);
    this.divv = new Float64Array(N);
    this.curlv = new Float64Array(N);
    this.phi = new Float64Array(N);
    this.alive = new Uint8Array(N).fill(1);
    this.t = 0;
    this.tidal = null; // 3×3 tidal tensor C (star units) or null
  }

  /**
   * Build a polytrope of index n: a cubic lattice in the unit ball is
   * remapped radially so that enclosed mass follows the Lane–Emden profile,
   * and each particle gets the polytrope's entropy at its radius. A short
   * damped relaxation (relax()) then removes lattice noise.
   */
  static polytrope({ N = 1000, n = 1.5, gamma = 5 / 3, seed = 7 } = {}) {
    const poly = polytrope(n);
    const rand = rng(seed);
    // Lattice spacing for ~N points in the unit ball, with a random rotation.
    const pts = [];
    let d = Math.cbrt((4 / 3) * Math.PI / N);
    for (let attempt = 0; attempt < 40; attempt++) {
      pts.length = 0;
      const k = Math.ceil(1 / d) + 1;
      const off = [rand() - 0.5, rand() - 0.5, rand() - 0.5].map((v) => v * d);
      for (let i = -k; i <= k; i++) for (let j = -k; j <= k; j++) for (let l = -k; l <= k; l++) {
        const x = i * d + off[0], y = j * d + off[1], z = l * d + off[2];
        if (x * x + y * y + z * z < 1) pts.push([x, y, z]);
      }
      if (pts.length >= N) break;
      d *= 0.99;
    }
    // Keep the N innermost points, re-scaled to fill the ball uniformly.
    pts.sort((p, q) => p[0] ** 2 + p[1] ** 2 + p[2] ** 2 - (q[0] ** 2 + q[1] ** 2 + q[2] ** 2));
    pts.length = N;
    const s = new SPHStar({ N, gamma });
    for (let i = 0; i < N; i++) {
      const [x, y, z] = pts[i];
      const ru = Math.hypot(x, y, z);
      const q = (i + 0.5) / N; // enclosed mass fraction of this shell
      const r = poly.radiusOfMass(q);
      const sc = ru > 0 ? r / ru : 0;
      s.pos[3 * i] = x * sc; s.pos[3 * i + 1] = y * sc; s.pos[3 * i + 2] = z * sc;
      const rho = Math.max(poly.rho(r), 1e-6 * poly.rhoC);
      s.A[i] = poly.pressure(r) / Math.pow(rho, gamma);
      s.h[i] = 1.2 * Math.cbrt(s.m / rho);
    }
    s.poly = poly;
    s.computeDensity();
    s.computeDensity();
    return s;
  }

  /** Density, smoothing lengths, ∇·v and |∇×v| (gather, O(N²)). */
  computeDensity() {
    const { N, pos, vel, h, rho, m, alive } = this;
    for (let it = 0; it < 2; it++) {
      rho.fill(0);
      this.divv.fill(0);
      this.curlv.fill(0);
      const cx = new Float64Array(N), cy = new Float64Array(N), cz = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        if (!alive[i]) continue;
        rho[i] += m * kernelW(0, h[i]);
        const xi = pos[3 * i], yi = pos[3 * i + 1], zi = pos[3 * i + 2];
        const hi2 = 4 * h[i] * h[i];
        for (let j = i + 1; j < N; j++) {
          if (!alive[j]) continue;
          const dx = xi - pos[3 * j], dy = yi - pos[3 * j + 1], dz = zi - pos[3 * j + 2];
          const r2 = dx * dx + dy * dy + dz * dz;
          const hj2 = 4 * h[j] * h[j];
          if (r2 >= hi2 && r2 >= hj2) continue;
          const r = Math.sqrt(r2);
          const dvx = vel[3 * i] - vel[3 * j], dvy = vel[3 * i + 1] - vel[3 * j + 1], dvz = vel[3 * i + 2] - vel[3 * j + 2];
          const vdr = dvx * dx + dvy * dy + dvz * dz;
          const crx = dvy * dz - dvz * dy, cry = dvz * dx - dvx * dz, crz = dvx * dy - dvy * dx;
          if (r2 < hi2) {
            rho[i] += m * kernelW(r, h[i]);
            if (r > 0) {
              const f = (m * kernelDW(r, h[i])) / r;
              this.divv[i] -= f * vdr;
              cx[i] -= f * crx; cy[i] -= f * cry; cz[i] -= f * crz;
            }
          }
          if (r2 < hj2) {
            rho[j] += m * kernelW(r, h[j]);
            if (r > 0) {
              const f = (m * kernelDW(r, h[j])) / r;
              this.divv[j] -= f * vdr;
              cx[j] -= f * crx; cy[j] -= f * cry; cz[j] -= f * crz;
            }
          }
        }
      }
      for (let i = 0; i < N; i++) {
        if (!alive[i]) continue;
        this.divv[i] /= rho[i];
        this.curlv[i] = Math.hypot(cx[i], cy[i], cz[i]) / rho[i];
      }
      // Relax h toward η (m/ρ)^{1/3}; a second pass makes h and ρ consistent.
      let maxChange = 0;
      for (let i = 0; i < N; i++) {
        if (!alive[i]) continue;
        const target = this.eta * Math.cbrt(m / rho[i]);
        const hn = Math.min(Math.max(target, 0.5 * h[i]), 2 * h[i]);
        maxChange = Math.max(maxChange, Math.abs(hn - h[i]) / h[i]);
        h[i] = hn;
      }
      if (maxChange < 0.03) break;
    }
  }

  /** Accelerations from pressure, viscosity, self-gravity and tides; entropy rates. */
  computeForces() {
    const { N, pos, vel, h, rho, A, m, gamma, alive, acc, dA, phi } = this;
    acc.fill(0);
    dA.fill(0);
    phi.fill(0);
    const P = new Float64Array(N), c = new Float64Array(N), bal = new Float64Array(N);
    let maxMu = 0;
    for (let i = 0; i < N; i++) {
      if (!alive[i]) continue;
      P[i] = A[i] * Math.pow(rho[i], gamma);
      c[i] = Math.sqrt((gamma * P[i]) / rho[i]);
      const dv = Math.abs(this.divv[i]);
      bal[i] = dv / (dv + this.curlv[i] + (1e-4 * c[i]) / h[i]);
    }
    this.vsig = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      if (!alive[i]) continue;
      const xi = pos[3 * i], yi = pos[3 * i + 1], zi = pos[3 * i + 2];
      const Pi = P[i] / (rho[i] * rho[i]);
      for (let j = i + 1; j < N; j++) {
        if (!alive[j]) continue;
        const dx = xi - pos[3 * j], dy = yi - pos[3 * j + 1], dz = zi - pos[3 * j + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        const r = Math.sqrt(r2);
        const hb = 0.5 * (h[i] + h[j]);
        // Self-gravity (softened with the mean smoothing length).
        const gf = (m * gravForce(r, hb)) / r;
        acc[3 * i] -= gf * dx; acc[3 * i + 1] -= gf * dy; acc[3 * i + 2] -= gf * dz;
        acc[3 * j] += gf * dx; acc[3 * j + 1] += gf * dy; acc[3 * j + 2] += gf * dz;
        const pot = m * gravPotential(r, hb);
        phi[i] += pot; phi[j] += pot;
        // Hydrodynamics within the kernel support.
        if (r >= 2 * h[i] && r >= 2 * h[j]) continue;
        const dW = 0.5 * (kernelDW(r, h[i]) + kernelDW(r, h[j]));
        const dvx = vel[3 * i] - vel[3 * j], dvy = vel[3 * i + 1] - vel[3 * j + 1], dvz = vel[3 * i + 2] - vel[3 * j + 2];
        const vdr = dvx * dx + dvy * dy + dvz * dz;
        let Pi_ij = 0;
        if (vdr < 0) {
          const mu = (hb * vdr) / (r2 + 0.01 * hb * hb);
          maxMu = Math.max(maxMu, -mu);
          const cb = 0.5 * (c[i] + c[j]), rb = 0.5 * (rho[i] + rho[j]);
          Pi_ij = (0.5 * (bal[i] + bal[j]) * (-this.alpha * cb * mu + this.beta * mu * mu)) / rb;
          const vs = cb - 2 * mu;
          if (vs > this.vsig[i]) this.vsig[i] = vs;
          if (vs > this.vsig[j]) this.vsig[j] = vs;
        }
        const coef = (m * (Pi + P[j] / (rho[j] * rho[j]) + Pi_ij) * dW) / (r || 1);
        acc[3 * i] -= coef * dx; acc[3 * i + 1] -= coef * dy; acc[3 * i + 2] -= coef * dz;
        acc[3 * j] += coef * dx; acc[3 * j + 1] += coef * dy; acc[3 * j + 2] += coef * dz;
        if (Pi_ij > 0) {
          const heat = (0.5 * m * Pi_ij * vdr * dW) / (r || 1);
          dA[i] += heat; dA[j] += heat;
        }
      }
    }
    for (let i = 0; i < N; i++) {
      if (!alive[i]) continue;
      dA[i] *= (gamma - 1) / Math.pow(rho[i], gamma - 1);
    }
    if (this.tidal) {
      const C = this.tidal;
      for (let i = 0; i < N; i++) {
        if (!alive[i]) continue;
        const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
        acc[3 * i] -= C[0][0] * x + C[0][1] * y + C[0][2] * z;
        acc[3 * i + 1] -= C[1][0] * x + C[1][1] * y + C[1][2] * z;
        acc[3 * i + 2] -= C[2][0] * x + C[2][1] * y + C[2][2] * z;
      }
    }
    this.cs = c;
    this.maxMu = maxMu;
  }

  /** Stable global time step (Courant + acceleration criteria). */
  timestep(cfl = 0.3) {
    let dt = Infinity;
    for (let i = 0; i < this.N; i++) {
      if (!this.alive[i]) continue;
      const vs = Math.max(this.vsig?.[i] ?? 0, this.cs[i]) + 1e-12;
      dt = Math.min(dt, (cfl * this.h[i]) / vs);
      const a = Math.hypot(this.acc[3 * i], this.acc[3 * i + 1], this.acc[3 * i + 2]);
      if (a > 0) dt = Math.min(dt, cfl * Math.sqrt(this.h[i] / a));
    }
    return dt;
  }

  /** One KDK leapfrog step. tidalAt(t) returns the tidal tensor at star time t (or null). */
  step(dt, tidalAt = null, damping = 0) {
    const { N, pos, vel, acc, A, dA, alive } = this;
    const half = 0.5 * dt;
    for (let i = 0; i < N; i++) {
      if (!alive[i]) continue;
      for (let k = 0; k < 3; k++) vel[3 * i + k] += half * acc[3 * i + k];
      A[i] = Math.max(A[i] + half * dA[i], 1e-12);
    }
    for (let i = 0; i < 3 * N; i++) pos[i] += dt * vel[i];
    this.t += dt;
    this.tidal = tidalAt ? tidalAt(this.t) : null;
    // Predict velocities for the viscosity/Balsara terms.
    const vHalf = vel.slice();
    for (let i = 0; i < 3 * N; i++) vel[i] += half * acc[i];
    this.computeDensity();
    this.computeForces();
    vel.set(vHalf);
    for (let i = 0; i < N; i++) {
      if (!alive[i]) continue;
      for (let k = 0; k < 3; k++) vel[3 * i + k] += half * acc[3 * i + k];
      A[i] = Math.max(A[i] + half * dA[i], 1e-12);
      if (damping) for (let k = 0; k < 3; k++) vel[3 * i + k] *= 1 - damping;
    }
  }

  /** Damped relaxation to hydrostatic equilibrium (no tides), then recentre. */
  relax(tEnd = 3, damping = 0.05) {
    this.computeDensity();
    this.computeForces();
    while (this.t < tEnd) this.step(Math.min(this.timestep(), 0.05), null, damping);
    this.recentre();
    this.vel.fill(0);
    this.t = 0;
    this.computeDensity();
    this.computeForces();
  }

  /** Remove the centre-of-mass position and velocity of the live particles. */
  recentre() {
    const c = this.centreOfMass();
    for (let i = 0; i < this.N; i++) {
      if (!this.alive[i]) continue;
      for (let k = 0; k < 3; k++) { this.pos[3 * i + k] -= c.x[k]; this.vel[3 * i + k] -= c.v[k]; }
    }
  }

  centreOfMass(mask = this.alive) {
    const x = [0, 0, 0], v = [0, 0, 0];
    let n = 0;
    for (let i = 0; i < this.N; i++) {
      if (!mask[i]) continue;
      n++;
      for (let k = 0; k < 3; k++) { x[k] += this.pos[3 * i + k]; v[k] += this.vel[3 * i + k]; }
    }
    return { x: x.map((c) => c / (n || 1)), v: v.map((c) => c / (n || 1)), n };
  }

  /** Energies (kinetic, internal, gravitational) of the live particles. */
  energies() {
    const { N, m, vel, A, rho, gamma, phi, alive } = this;
    let K = 0, U = 0, W = 0;
    for (let i = 0; i < N; i++) {
      if (!alive[i]) continue;
      K += 0.5 * m * (vel[3 * i] ** 2 + vel[3 * i + 1] ** 2 + vel[3 * i + 2] ** 2);
      U += (m * A[i] * Math.pow(rho[i], gamma - 1)) / (gamma - 1);
      W += 0.5 * m * phi[i];
    }
    return { K, U, W, total: K + U + W };
  }

  /** Mass-weighted second moments ⟨ξξᵀ⟩ about the centre of mass of `mask`. */
  moments(mask = this.alive) {
    const c = this.centreOfMass(mask);
    const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < this.N; i++) {
      if (!mask[i]) continue;
      const d = [0, 1, 2].map((k) => this.pos[3 * i + k] - c.x[k]);
      for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) M[p][q] += d[p] * d[q];
    }
    for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) M[p][q] /= c.n || 1;
    return { centre: c.x, M, n: c.n };
  }

  /**
   * Self-bound core: iteratively keep particles whose energy relative to the
   * core (kinetic + self-gravity of the core) is negative.
   */
  boundCore() {
    const { N, pos, vel, m, h, alive } = this;
    let mask = Uint8Array.from(alive);
    for (let it = 0; it < 8; it++) {
      const c = this.centreOfMass(mask);
      const next = new Uint8Array(N);
      let changed = 0;
      for (let i = 0; i < N; i++) {
        if (!alive[i]) continue;
        let pot = 0;
        for (let j = 0; j < N; j++) {
          if (j === i || !mask[j]) continue;
          const r = Math.hypot(pos[3 * i] - pos[3 * j], pos[3 * i + 1] - pos[3 * j + 1], pos[3 * i + 2] - pos[3 * j + 2]);
          pot += m * gravPotential(r, 0.5 * (h[i] + h[j]));
        }
        const v2 = (vel[3 * i] - c.v[0]) ** 2 + (vel[3 * i + 1] - c.v[1]) ** 2 + (vel[3 * i + 2] - c.v[2]) ** 2;
        const internal = (this.A[i] * Math.pow(this.rho[i], this.gamma - 1)) / (this.gamma - 1);
        next[i] = 0.5 * v2 + pot + internal < 0 ? 1 : 0;
        if (next[i] !== mask[i]) changed++;
      }
      mask = next;
      if (!changed) break;
    }
    let n = 0;
    for (let i = 0; i < N; i++) n += mask[i];
    return { mask, fraction: n / N };
  }
}

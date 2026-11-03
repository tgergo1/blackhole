// Relativistic thin accretion disk (Novikov & Thorne 1973; Page & Thorne 1974).
// Units G = c = M = 1 unless stated. Spin a may be negative (retrograde disk).

import { metricLower } from './kerr.js';

/** Prograde ISCO radius (Bardeen, Press & Teukolsky 1972). a < 0 = retrograde. */
export function iscoRadius(a) {
  const z1 = 1 + Math.cbrt(1 - a * a) * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
  const z2 = Math.sqrt(3 * a * a + z1 * z1);
  return 3 + z2 - Math.sign(a || 1) * Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

/** Equatorial circular photon orbit radii {pro, retro} (Bardeen et al. 1972). */
export function photonOrbitRadii(a) {
  const s = Math.abs(a);
  const pro = 2 * (1 + Math.cos((2 / 3) * Math.acos(-s)));
  const retro = 2 * (1 + Math.cos((2 / 3) * Math.acos(s)));
  return { pro, retro };
}

/** Keplerian angular velocity of a prograde equatorial circular orbit. */
export function keplerOmega(r, a) {
  return 1 / (Math.pow(r, 1.5) + a);
}

/** Specific energy of a circular orbit; radiative efficiency is 1 − E(r_isco). */
export function circularOrbitEnergy(r, a) {
  const v = Math.pow(r, 1.5);
  return (v - 2 * Math.sqrt(r) + a) / (Math.pow(r, 0.75) * Math.sqrt(v - 3 * Math.sqrt(r) + 2 * a));
}

export function radiativeEfficiency(a) {
  return 1 - circularOrbitEnergy(iscoRadius(a), a);
}

/**
 * Four-velocity u^μ of disk gas at Cartesian point (x, y, 0) of a Kerr–Schild
 * metric with spin aMetric, rotating at angular velocity omega about z.
 * Normalised numerically against the metric, so no closed form is assumed.
 */
export function diskFourVelocity(x, y, omega, aMetric) {
  const g = metricLower([x, y, 0], aMetric);
  const v = [1, -omega * y, omega * x, 0];
  let n = 0;
  for (let m = 0; m < 4; m++) for (let k = 0; k < 4; k++) n += g[m][k] * v[m] * v[k];
  const ut = 1 / Math.sqrt(-n);
  return v.map((c) => c * ut);
}

/**
 * Dimensionless Page–Thorne flux shape: F(r) = (3 Ṁ / 8π M²) · fluxShape(r, a).
 * Zero at the ISCO (zero-torque inner boundary); → r^{-3} at large r.
 */
export function fluxShape(r, a) {
  const x = Math.sqrt(r);
  const x0 = Math.sqrt(iscoRadius(a));
  if (x <= x0) return 0;
  const th = Math.acos(a);
  const x1 = 2 * Math.cos((th - Math.PI) / 3);
  const x2 = 2 * Math.cos((th + Math.PI) / 3);
  const x3 = -2 * Math.cos(th / 3);
  const term = (xi, xj, xk) =>
    ((3 * (xi - a) ** 2) / (xi * (xi - xj) * (xi - xk))) * Math.log((x - xi) / (x0 - xi));
  const bracket =
    x - x0 - 1.5 * a * Math.log(x / x0) - term(x1, x2, x3) - term(x2, x1, x3) - term(x3, x1, x2);
  return bracket / (x ** 4 * (x ** 3 - 3 * x + 2 * a));
}

/** Radius (in M) and value of the flux maximum, found by golden-section search. */
export function fluxPeak(a, rOut = 1000) {
  let lo = iscoRadius(a);
  let hi = Math.max(lo * 6, 30);
  const gr = (Math.sqrt(5) - 1) / 2;
  let c = hi - gr * (hi - lo);
  let d = lo + gr * (hi - lo);
  for (let i = 0; i < 200; i++) {
    if (fluxShape(c, a) > fluxShape(d, a)) hi = d;
    else lo = c;
    c = hi - gr * (hi - lo);
    d = lo + gr * (hi - lo);
  }
  const r = 0.5 * (lo + hi);
  return { r: Math.min(r, rOut), F: fluxShape(r, a) };
}

// SI constants (CODATA 2018).
export const SI = {
  G: 6.6743e-11,
  c: 299792458,
  sigma: 5.670374419e-8,
  Msun: 1.98847e30,
  // Eddington luminosity per solar mass for ionised hydrogen (Thomson opacity).
  LeddPerMsun: 1.2572e31,
};

/**
 * Peak effective temperature (K, local rest frame) of a Novikov–Thorne disk
 * around a hole of `massSolar` solar masses accreting at `eddRatio` × the
 * Eddington luminosity (with Ṁ = L / (η c²) and η from the spin).
 */
export function peakTemperature(massSolar, eddRatio, a) {
  const M = massSolar * SI.Msun;
  const eta = radiativeEfficiency(a);
  const Mdot = (eddRatio * SI.LeddPerMsun * massSolar) / (eta * SI.c ** 2);
  const pref = (3 * Mdot * SI.c ** 6) / (8 * Math.PI * SI.G ** 2 * M * M);
  const F = pref * fluxPeak(a).F;
  return Math.pow(F / SI.sigma, 0.25);
}

/** Local effective temperature at radius r given the peak temperature. */
export function diskTemperature(r, a, tPeak, Fpeak = fluxPeak(a).F) {
  const F = fluxShape(r, a);
  return F > 0 ? tPeak * Math.pow(F / Fpeak, 0.25) : 0;
}

/**
 * Electron-scattering limb darkening (Chandrasekhar 1960), normalised so the
 * emergent flux equals that of an isotropic emitter.
 */
export function limbDarkening(mu) {
  return (1 + 2.06 * mu) / (1 + (2.06 * 2) / 3);
}

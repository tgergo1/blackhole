// Generic adaptive Dormand–Prince 5(4) integrator for systems of any size,
// with continuous (dense) output. Used for timelike geodesics, parallel
// transport and the debris particles. The photon tracer keeps its own
// specialised 6-component version in tracer.js (mirrored by the shader).

const A = [
  [],
  [1 / 5],
  [3 / 40, 9 / 40],
  [44 / 45, -56 / 15, 32 / 9],
  [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
  [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
  [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
];
const EC = [71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40];

/**
 * One DOPRI5 step. f(y, out) writes dy/dλ into out.
 * Returns { y, k (7 stage derivatives), err } where err is the scaled RMS
 * error with per-component scale atol + rtol·scale[i] (scale defaults to |y|).
 */
export function dopri5Step(f, y, k1, h, rtol, atol, scaleFn) {
  const n = y.length;
  const k = [k1];
  const tmp = new Float64Array(n);
  for (let st = 1; st < 7; st++) {
    for (let i = 0; i < n; i++) {
      let acc = y[i];
      for (let j = 0; j < st; j++) acc += h * A[st][j] * k[j][i];
      tmp[i] = acc;
    }
    const d = new Float64Array(n);
    f(tmp, d);
    k.push(d);
  }
  let errSq = 0;
  for (let i = 0; i < n; i++) {
    let e = 0;
    for (let j = 0; j < 7; j++) e += EC[j] * k[j][i];
    e *= h;
    const sc = atol + rtol * (scaleFn ? scaleFn(i, y, tmp) : Math.max(Math.abs(y[i]), Math.abs(tmp[i])));
    errSq += (e / sc) ** 2;
  }
  return { y: tmp, k, err: Math.sqrt(errSq / n) };
}

/**
 * Cubic Hermite interpolation between (y0, f0) and (y1, f1) over a step h,
 * at fraction θ ∈ [0, 1]. Accurate to O(h⁴), enough for sampling outputs.
 */
export function hermite(y0, f0, y1, f1, h, th, out = new Float64Array(y0.length)) {
  const t2 = th * th, t3 = t2 * th;
  const h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + th, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
  for (let i = 0; i < y0.length; i++) out[i] = h00 * y0[i] + h10 * h * f0[i] + h01 * y1[i] + h11 * h * f1[i];
  return out;
}

/**
 * Adaptive integrator object. step() advances by one accepted step, honouring
 * an optional per-step cap hMax(y). Exposes the previous and current state and
 * derivatives so callers can interpolate within the last step.
 */
export class Dopri5 {
  constructor(f, y0, { rtol = 1e-10, atol = 1e-12, h0 = 1e-2, scaleFn = null } = {}) {
    this.f = f;
    this.y = Float64Array.from(y0);
    this.k1 = new Float64Array(this.y.length);
    f(this.y, this.k1);
    this.h = h0;
    this.rtol = rtol;
    this.atol = atol;
    this.scaleFn = scaleFn;
    this.lambda = 0;
    this.accepted = 0;
    this.rejected = 0;
  }

  /** Advance one accepted step (never longer than hCap). Returns the step taken. */
  step(hCap = Infinity) {
    for (let tries = 0; tries < 60; tries++) {
      const h = Math.min(this.h, hCap);
      const st = dopri5Step(this.f, this.y, this.k1, h, this.rtol, this.atol, this.scaleFn);
      if (!(st.err <= 1)) {
        this.rejected++;
        this.h = h * Math.max(0.1, 0.9 * Math.pow(st.err || 1e10, -0.2));
        if (!Number.isFinite(st.err)) this.h = h * 0.1;
        continue;
      }
      this.prevY = this.y;
      this.prevK = this.k1;
      this.y = st.y;
      this.k1 = st.k[6];
      this.lastH = h;
      this.lambda += h;
      this.accepted++;
      this.h = h * Math.min(5, Math.max(0.2, 0.9 * Math.pow(st.err || 1e-10, -0.2)));
      return h;
    }
    throw new Error('DOPRI5: step size underflow');
  }

  /** State at fraction θ of the last step. */
  interpolate(th, out) {
    return hermite(this.prevY, this.prevK, this.y, this.k1, this.lastH, th, out);
  }
}

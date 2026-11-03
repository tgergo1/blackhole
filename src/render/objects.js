// Bridges the simulated world lines (src/sim/events.js) and the ray tracer.
//
// Each frame the observer sits at mirror time t'_cam. A backward ray leaves
// the camera at s = t' − t'_cam = 0 and s grows along it (ds/dλ = dt'/dλ),
// so the ray meets every body at the moment its light had to be emitted:
// light-travel time, the different ages of the primary and higher-order
// images, and the freezing at the horizon all follow from this.
//
// For the GPU we resample every body's world line on a uniform grid
// s_k = s0 + k Δs (k < S) and pack it into an RGBA32F texture, 5 texels per sample:
//   row 0: C' (mirror position), τ (proper time)
//   row 1: Ũ (mirror four-velocity, future-directed)
//   row 2: w1, valid flag        row 3: w2, A_max        row 4: w3, A_min
// Samples with Ũ^t > 1e3 are dropped: the body is then within a hair of the
// horizon, its light is redshifted by g ≲ 10⁻³ (intensity ≲ 10⁻¹²), and the
// outgoing mirror coordinates there exceed float32 precision.
// The CPU reference tracer reads exactly the same Float32 data.

import { SAMPLE } from '../sim/events.js';

export const MAX_OBJECTS = 48;
export const GRID_S = 1024;
export const TEXELS = 5;
// Per-body bounding spheres of blocks of BLOCK consecutive grid samples, used
// by the shader to skip bodies a ray step cannot reach (pure culling: a body
// is skipped only when it could not have changed the step or the colour).
export const BLOCK = 16;

const KIND_PRIORITY = { probe: 0, star: 0, core: 0, chunk: 1 };

export class ObjectField {
  constructor({ maxObjects = MAX_OBJECTS, S = GRID_S } = {}) {
    this.maxObjects = maxObjects;
    this.S = S;
    this.grid = new Float32Array(S * TEXELS * maxObjects * 4);
    this.bounds = new Float32Array(4 * maxObjects);
    this.NB = Math.ceil(S / BLOCK);
    this.blocks = new Float32Array(4 * this.NB * maxObjects);
    this.emit = new Float32Array(4 * maxObjects);
    this.count = 0;
    this.ds = 1;
    this.s0 = 0;
    this.all = new Float32Array(4);
    this.entities = new Map();
    this.slots = [];
  }

  /** Append samples from the simulation (see Simulation.drain()). */
  ingest(list) {
    for (const e of list) {
      let ent = this.entities.get(e.id);
      if (!ent) {
        ent = { id: e.id, kind: e.kind, meta: e.meta, data: new Float64Array(SAMPLE * 256), count: 0, ended: false, info: [], maxR: 0 };
        this.entities.set(e.id, ent);
      }
      ent.meta = e.meta;
      const n = e.samples.length / SAMPLE;
      if ((ent.count + n) * SAMPLE > ent.data.length) {
        const d = new Float64Array(Math.max(ent.data.length * 2, (ent.count + n) * SAMPLE));
        d.set(ent.data);
        ent.data = d;
      }
      ent.data.set(e.samples, ent.count * SAMPLE);
      for (let k = 0; k < n; k++) {
        const o = k * SAMPLE;
        if (e.samples[o + 4] < 1e3) ent.maxR = Math.max(ent.maxR, Math.hypot(e.samples[o + 1], e.samples[o + 2], e.samples[o + 3]) + e.samples[o + 18]);
      }
      ent.count += n;
      ent.ended = e.ended;
      if (e.info?.length) ent.info.push(...e.info);
    }
  }

  /** Largest distance from the hole (mirror Kerr–Schild norm + body size) of any stored sample. */
  extent() {
    let m = 0;
    for (const e of this.entities.values()) m = Math.max(m, e.maxR);
    return m;
  }

  /**
   * Smallest mirror time simulated so far over bodies still being simulated
   * (Infinity if none): the observer must not see past it.
   */
  frontier() {
    let m = Infinity;
    for (const e of this.entities.values()) if (!e.ended && e.count) m = Math.min(m, e.data[(e.count - 1) * SAMPLE]);
    return m;
  }

  clear() {
    this.entities.clear();
    this.count = 0;
  }

  /** Drop entities whose whole world line is older than the window (t' > tpMax). */
  prune(tpCam, window) {
    for (const [id, e] of this.entities) {
      if (e.ended && e.count && e.data[(e.count - 1) * SAMPLE] > tpCam + window) this.entities.delete(id);
    }
  }

  /**
   * State of entity e at mirror time tp (Hermite in t' for the position,
   * linear for the rest). Returns null outside the simulated range.
   */
  static stateAt(e, tp, out) {
    const d = e.data, n = e.count;
    if (n < 2) return null;
    if (tp > d[0] || tp < d[(n - 1) * SAMPLE]) return null;
    // Samples have decreasing t'.
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (d[m * SAMPLE] >= tp) lo = m; else hi = m;
    }
    const a = lo * SAMPLE, b = hi * SAMPLE;
    const h = d[b] - d[a]; // negative
    const f = h !== 0 ? (tp - d[a]) / h : 0;
    const f2 = f * f, f3 = f2 * f;
    const h00 = 2 * f3 - 3 * f2 + 1, h10 = f3 - 2 * f2 + f, h01 = -2 * f3 + 3 * f2, h11 = f3 - f2;
    for (let k = 0; k < 3; k++) {
      const va = d[a + 5 + k] / d[a + 4], vb = d[b + 5 + k] / d[b + 4]; // dC'/dt' = Ũ^i/Ũ^t
      out[1 + k] = h00 * d[a + 1 + k] + h10 * h * va + h01 * d[b + 1 + k] + h11 * h * vb;
    }
    out[0] = tp;
    for (let k = 4; k < SAMPLE; k++) out[k] = d[a + k] + f * (d[b + k] - d[a + k]);
    return out;
  }

  /**
   * Resample all entities onto the grid for an observer at mirror time
   * tpCam, covering ray times s ∈ [s0, s0 + window].
   */
  build(tpCam, window, s0 = 0) {
    this.ds = window / (this.S - 1);
    this.s0 = s0;
    const cand = [];
    for (const e of this.entities.values()) {
      if (e.count < 2) continue;
      const first = e.data[0], last = e.data[(e.count - 1) * SAMPLE];
      if (last > tpCam + s0 + window || first < tpCam + s0) continue; // no overlap with the window
      cand.push(e);
    }
    cand.sort((x, y) => (KIND_PRIORITY[x.kind] ?? 2) - (KIND_PRIORITY[y.kind] ?? 2) || x.id - y.id);
    this.slots = cand.slice(0, this.maxObjects);
    this.count = this.slots.length;
    const st = new Float64Array(SAMPLE);
    const g = this.grid;
    const rowStride = this.S * 4;
    let amin = [Infinity, Infinity, Infinity], amax = [-Infinity, -Infinity, -Infinity];
    this.slots.forEach((e, j) => {
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      let Amax = 0, any = false;
      const base = j * TEXELS * rowStride;
      const okArr = this.okArr ?? (this.okArr = new Uint8Array(this.S));
      const P = this.posArr ?? (this.posArr = new Float64Array(4 * this.S));
      for (let k = 0; k < this.S; k++) {
        const s = s0 + k * this.ds;
        const ok = ObjectField.stateAt(e, tpCam + s, st) && st[4] < 1e3;
        const o = base + k * 4;
        if (!ok) {
          for (let r = 0; r < TEXELS; r++) g.fill(0, o + r * rowStride, o + r * rowStride + 4);
          okArr[k] = 0;
          continue;
        }
        any = true;
        okArr[k] = 1;
        P[4 * k] = st[1]; P[4 * k + 1] = st[2]; P[4 * k + 2] = st[3]; P[4 * k + 3] = st[18];
        g[o] = st[1]; g[o + 1] = st[2]; g[o + 2] = st[3]; g[o + 3] = st[17];
        g[o + rowStride] = st[4]; g[o + rowStride + 1] = st[5]; g[o + rowStride + 2] = st[6]; g[o + rowStride + 3] = st[7];
        g[o + 2 * rowStride] = st[8]; g[o + 2 * rowStride + 1] = st[9]; g[o + 2 * rowStride + 2] = st[10]; g[o + 2 * rowStride + 3] = 1;
        g[o + 3 * rowStride] = st[11]; g[o + 3 * rowStride + 1] = st[12]; g[o + 3 * rowStride + 2] = st[13]; g[o + 3 * rowStride + 3] = st[18];
        const wmax = Math.max(Math.hypot(st[8], st[9], st[10]), Math.hypot(st[11], st[12], st[13]), Math.hypot(st[14], st[15], st[16]));
        g[o + 4 * rowStride] = st[14]; g[o + 4 * rowStride + 1] = st[15]; g[o + 4 * rowStride + 2] = st[16]; g[o + 4 * rowStride + 3] = 1 / wmax;
        for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], st[1 + c]); hi[c] = Math.max(hi[c], st[1 + c]); }
        Amax = Math.max(Amax, st[18]);
      }
      // Block b covers the interpolation intervals starting at samples b·BLOCK … b·BLOCK + BLOCK − 1,
      // i.e. samples b·BLOCK … b·BLOCK + BLOCK.
      for (let bk = 0; bk < this.NB; bk++) {
        const lo2 = [Infinity, Infinity, Infinity], hi2 = [-Infinity, -Infinity, -Infinity];
        let bA = 0, bAny = false;
        for (let k = bk * BLOCK; k <= Math.min(bk * BLOCK + BLOCK, this.S - 1); k++) {
          if (!okArr[k]) continue;
          bAny = true;
          for (let c = 0; c < 3; c++) { lo2[c] = Math.min(lo2[c], P[4 * k + c]); hi2[c] = Math.max(hi2[c], P[4 * k + c]); }
          bA = Math.max(bA, P[4 * k + 3]);
        }
        const o = 4 * (j * this.NB + bk);
        if (!bAny) { this.blocks.set([0, 0, 0, -1], o); continue; }
        const ext = Math.hypot(hi2[0] - lo2[0], hi2[1] - lo2[1], hi2[2] - lo2[2]);
        // Margin for the Hermite interpolant bulging past the sample hull.
        this.blocks.set([0.5 * (lo2[0] + hi2[0]), 0.5 * (lo2[1] + hi2[1]), 0.5 * (lo2[2] + hi2[2]), 0.51 * ext + 1.05 * bA + 0.05 * this.ds * BLOCK], o);
      }
      if (!any) { this.bounds.fill(0, 4 * j, 4 * j + 4); this.bounds[4 * j + 3] = -1; return; }
      const c = [0, 1, 2].map((i) => 0.5 * (lo[i] + hi[i]));
      const rad = 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) + 1.05 * Amax;
      this.bounds.set([c[0], c[1], c[2], rad], 4 * j);
      const m = e.meta;
      this.emit.set([m.Trest ?? 6000, m.opacity ?? 10, m.blink ?? 0, m.limb ?? 0], 4 * j);
      for (let i = 0; i < 3; i++) { amin[i] = Math.min(amin[i], c[i] - rad); amax[i] = Math.max(amax[i], c[i] + rad); }
    });
    if (this.count) {
      const c = [0, 1, 2].map((i) => 0.5 * (amin[i] + amax[i]));
      this.all.set([c[0], c[1], c[2], 0.5 * Math.hypot(amax[0] - amin[0], amax[1] - amin[1], amax[2] - amin[2])]);
    } else this.all.set([0, 0, 0, -1]);
    return this.count;
  }

  /** Grid data in the layout used by the shader: { grid, bounds, emit, count, ds, S, all }. */
  snapshot() {
    return { grid: this.grid, bounds: this.bounds, emit: this.emit, blocks: this.blocks, NB: this.NB, count: this.count, ds: this.ds, s0: this.s0, S: this.S, all: this.all, maxObjects: this.maxObjects };
  }
}

/** Texture height for a given capacity. */
export function gridRows(maxObjects = MAX_OBJECTS) {
  return maxObjects * TEXELS;
}

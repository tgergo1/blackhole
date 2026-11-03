// "Throw something in": launches bodies into the simulation, runs the
// observer's clock and feeds the ray tracer with their resampled world lines.
import { SimClient } from '../sim/client.js';
import { STAR_TYPES, starScales, retardedStart, launchState, SAMPLE, UNITS } from '../sim/events.js';
import { ObjectField } from '../render/objects.js';
import { timeShift, physicalPoint, mirrorPoint } from '../physics/spacetime.js';

const DEG = Math.PI / 180;
const V_INF_HYPERBOLIC = 0.2;

export const OBJECT_DEFAULTS = {
  objType: 'probe', objOrbit: 'parabolic', beta: 1.5, rp: 6, objIncl: 30, objDir: 150, objN: 600, speed: 30,
};

const fmtNum = (v, d = 3) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(1) : v.toPrecision(d));

export function formatDuration(sec) {
  const a = Math.abs(sec);
  if (a < 1e-3) return `${(sec * 1e6).toPrecision(3)} µs`;
  if (a < 1) return `${(sec * 1e3).toPrecision(3)} ms`;
  if (a < 120) return `${sec.toPrecision(3)} s`;
  if (a < 7200) return `${(sec / 60).toPrecision(3)} min`;
  if (a < 2 * 86400) return `${(sec / 3600).toPrecision(3)} h`;
  if (a < UNITS.yr) return `${(sec / 86400).toPrecision(3)} days`;
  return `${(sec / UNITS.yr).toPrecision(3)} yr`;
}

export class ObjectsController {
  /**
   * renderer: Renderer; statusEl, chart: DOM nodes;
   * moveCamera(patch): ask the UI to move the camera (used when a launch point is behind it).
   */
  constructor({ renderer, statusEl, chart, moveCamera }) {
    this.renderer = renderer;
    this.statusEl = statusEl;
    this.chart = chart;
    this.moveCamera = moveCamera;
    this.cfg = { ...OBJECT_DEFAULTS };
    this.field = new ObjectField();
    this.tObs = 0;
    this.paused = false;
    this.launches = []; // per simulation event: { kind, label, cfg, stats }
    this.errors = [];
    this.dirty = false;
    this.lastKey = '';
    this.waiting = false;
    this.lastStatus = 0;
    this.sim = new SimClient((out) => this.ingest(out));
  }

  ingest(out) {
    this.field.ingest(out.entities);
    for (const ev of out.events) if (this.launches[ev.id - 1]) this.launches[ev.id - 1].stats = ev.stats;
    if (out.errors?.length) this.errors.push(...out.errors);
    if (out.entities.length) this.dirty = true;
  }

  get active() {
    return this.field.entities.size > 0 || this.sim.busy;
  }

  clear() {
    this.sim.clear();
    this.field.clear();
    this.launches = [];
    this.errors = [];
    this.dirty = true;
  }

  /** Launch the configured object so that it appears now. */
  throwObject() {
    const scene = this.renderer.scene;
    const a = scene.a, rc = scene.r;
    const cam = this.renderer.derived.cam;
    // Pericentre azimuth relative to the camera's (physical) azimuth.
    const P = physicalPoint(cam.pos[0], cam.pos[1], cam.pos[2], a);
    const phiPeri = Math.atan2(P[1], P[0]) + this.cfg.objDir * DEG;
    const plunge = this.cfg.objOrbit === 'plunge';
    const E = this.cfg.objOrbit === 'hyperbolic' ? 1 / Math.sqrt(1 - V_INF_HYPERBOLIC ** 2) : 1;
    const incl = this.cfg.objIncl * DEG;
    const type = this.cfg.objType;
    let kind, cfg, label;
    if (type === 'probe' || type === 'dust') {
      kind = type;
      const rp = this.cfg.rp;
      const r0 = Math.max(40, 3 * rp);
      cfg = { a, rp, r0, E, incl, phiPeri, plunge };
      if (type === 'probe') cfg.radius = Math.min(Math.max(0.015 * rc, 0.2), 4); // display size only (a test body)
      else { cfg.radius = Math.min(Math.max(0.012 * rc, 1), 6); cfg.N = 400; }
      label = type === 'probe' ? 'Probe' : 'Dust cloud';
    } else {
      kind = 'star';
      const star = STAR_TYPES[type];
      const sc = starScales(star, scene.massSolar);
      const rp = sc.rt / this.cfg.beta;
      const r0 = Math.max(3 * sc.rt, 3 * rp, 40);
      cfg = { a, massSolar: scene.massSolar, star, beta: this.cfg.beta, rp, E, incl, phiPeri, plunge, N: this.cfg.objN, r0 };
      label = star.label;
    }
    // Make sure the launch point is in front of the camera's distance.
    if (cfg.r0 * 1.2 > rc) {
      this.moveCamera({ r: cfg.r0 * 1.6, fovDeg: Math.max(scene.fovDeg, 50) });
      this.noteTime = performance.now();
      this.note = `Camera moved out to r = ${(cfg.r0 * 1.6).toFixed(0)} M so the launch point (r = ${cfg.r0.toFixed(0)} M) is in view.`;
    } else this.note = '';
    const rCam = this.renderer.scene.r;
    // Start so that light from the launch point reaches the camera now:
    // retardedStart is exact for a point on the camera's radial line; for the
    // actual launch point add the extra straight-line path in mirror coordinates.
    let extra = 0;
    try {
      const L = launchState(cfg);
      const m = mirrorPoint(0, L.s[0], L.s[1], L.s[2], a);
      const c = this.renderer.derived.cam.pos;
      extra = Math.max(0, Math.hypot(m.x - c[0], m.y - c[1], m.z - c[2]) - (rCam - cfg.r0));
    } catch { /* no valid orbit: the simulation reports the error */ }
    cfg.t0 = retardedStart(this.tObs, rCam, cfg.r0, a) - extra;
    this.sim.add(kind, cfg);
    const Trest = kind === 'star' ? cfg.star.Teff : kind === 'probe' ? 9000 : 7000;
    this.renderer.objRefT = Math.max(this.launches.length ? this.renderer.objRefT : 0, Trest);
    this.renderer.sample = 0;
    this.launches.push({ kind, label, cfg, stats: {} });
    this.paused = false;
  }

  /** Advance the observer clock and upload the bodies. Returns true while the image is changing. */
  frame(dtMs) {
    this.sim.tick(8);
    const scene = this.renderer.scene;
    const a = scene.a, rc = scene.r;
    if (!this.active) {
      if (this.field.count) { this.field.count = 0; this.renderer.setObjects({ count: 0 }); }
      if (!this.paused) this.tObs += (this.cfg.speed * dtMs) / 1000;
      this.updateStatus();
      return false;
    }
    const TS2 = 2 * timeShift(rc, a);
    const rFar = this.field.extent();
    const s0 = Math.max(0, rc - rFar - 5);
    const window = Math.min(Math.max(rc + rFar + 60 - s0, 50), 2500);
    // The observer may not see past what has been simulated.
    let tMax = Infinity;
    const front = this.field.frontier();
    if (Number.isFinite(front)) tMax = TS2 - front + s0;
    else if (this.sim.busy) tMax = this.tObs; // between phases: hold
    this.waiting = false;
    let advanced = false;
    if (!this.paused) {
      const next = this.tObs + (this.cfg.speed * dtMs) / 1000;
      const t = Math.min(next, Math.max(this.tObs, tMax));
      this.waiting = t < next;
      advanced = t !== this.tObs;
      this.tObs = t;
    }
    const tpCam = -(this.tObs - TS2);
    const key = `${rc}|${a}|${this.tObs}`;
    if (advanced || this.dirty || key !== this.lastKey) {
      this.field.prune(tpCam, s0 + window);
      this.field.build(tpCam, window, s0);
      this.renderer.setObjects(this.field.snapshot());
      this.dirty = false;
      this.lastKey = key;
      this.tpCam = tpCam;
    }
    this.updateStatus();
    return advanced;
  }

  /**
   * The sample of entity e seen now along (roughly) the direct line of sight:
   * solves t′ − t′_cam = |C′(t′) − X′_cam| (straight-line light delay in mirror
   * Kerr–Schild coordinates). Returns its proper time τ or null.
   */
  seenTau(e) {
    const cam = this.renderer.derived.cam.pos;
    const d = e.data;
    const f = (i) => {
      const o = i * SAMPLE;
      return d[o] - this.tpCam - Math.hypot(d[o + 1] - cam[0], d[o + 2] - cam[1], d[o + 3] - cam[2]);
    };
    if (e.count < 2 || f(0) < 0) return null; // not yet visible
    if (f(e.count - 1) > 0) return e.ended ? { tau: d[(e.count - 1) * SAMPLE + 17], last: true } : null;
    let lo = 0, hi = e.count - 1; // f(lo) ≥ 0 > f(hi)
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (f(m) >= 0) lo = m; else hi = m;
    }
    return { tau: d[lo * SAMPLE + 17], last: false };
  }

  infoAt(e, tau) {
    const info = e.info;
    if (!info.length) return null;
    let lo = 0, hi = info.length - 1;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (info[m].tau <= tau) lo = m; else hi = m;
    }
    return Math.abs(info[hi].tau - tau) < Math.abs(info[lo].tau - tau) ? info[hi] : info[lo];
  }

  updateStatus(force = false) {
    const now = performance.now();
    if (!force && now - this.lastStatus < 200) return;
    this.lastStatus = now;
    const scene = this.renderer.scene;
    const Msec = UNITS.GMsun_c3 * scene.massSolar;
    const lines = [];
    lines.push(`Observer time <b>${fmtNum(this.tObs, 4)} M</b> = ${formatDuration(this.tObs * Msec)}${this.paused ? ' · paused' : ''}${this.waiting ? ' · <span class="warn">waiting for the simulation…</span>' : ''}`);
    if (this.note && now - this.noteTime > 8000) this.note = '';
    if (this.note) lines.push(`<span class="warn">${this.note}</span>`);
    for (const err of this.errors.slice(-2)) lines.push(`<span class="warn">Launch failed: ${err}</span>`);
    const L = this.launches[this.launches.length - 1];
    if (L) {
      const id = this.launches.length;
      const st = L.stats ?? {};
      lines.push(`<b>${L.label}</b>${this.launches.length > 1 ? ` (${this.launches.length} launched)` : ''}`);
      // Primary entity of the latest event (probe / star / its surviving core).
      let ent = null;
      for (const e of this.field.entities.values()) if (e.meta?.event === id && (e.kind === 'probe' || e.kind === 'star')) ent = e;
      if (ent) {
        const seen = this.seenTau(ent);
        const inf = seen && this.infoAt(ent, seen.tau);
        if (!seen) lines.push('its light has not reached you yet');
        else if (inf) {
          const lam = inf.tidal[0] / (Msec * Msec); // s⁻² (most stretching eigenvalue)
          const acc = Math.abs(lam) * 2;
          lines.push(`as seen now: r = ${inf.r.toFixed(2)} M · own clock τ = ${inf.tau.toFixed(1)} M · dτ/dt = ${(1 / inf.dtdtau).toFixed(3)}`);
          if (L.kind === 'probe') lines.push(`tidal stretch over a 2 m body: ${fmtNum(acc)} m/s² (${fmtNum(acc / 9.81)} g)`);
          if (seen.last && ent.ended) lines.push(inf.r < 1.5 * this.renderer.derived.rH ? 'image frozen at the horizon, fading as its light is redshifted away' : 'left the simulated region');
        }
      }
      if (L.kind === 'star') {
        if (st.rt) lines.push(`β = ${st.beta.toFixed(2)} · tidal radius r<sub>t</sub> = ${fmtNum(st.rt)} M · pericentre ${fmtNum(st.rp)} M · star radius ${fmtNum(st.Rstar)} M`);
        if (st.outcome === 'running' || !st.outcome) lines.push('outcome: simulating the encounter…');
        else {
          lines.push(`simulated outcome: <b>${st.outcome}</b>${st.maxStretch > 1.05 ? ` · stretched up to ×${st.maxStretch.toFixed(1)}` : ''}`);
          const pct = (v) => `${Math.round(100 * (v ?? 0))}%`;
          if (st.bound || st.unbound || st.swallowed) lines.push(`gas: bound ${pct(st.bound)} · unbound ${pct(st.unbound)} · swallowed ${pct(st.swallowed)}${st.coreFraction ? ` · surviving core ${pct(st.coreFraction)}` : ''}`);
          if (st.fallback) {
            const peak = Math.max(...st.fallback.mdot);
            lines.push(`first debris returns after ${formatDuration(st.fallback.tMin)}; peak fallback ${fmtNum(peak)} M☉/yr = ${fmtNum(peak / st.fallback.mdotEdd)} × Eddington`);
          }
        }
      } else if (L.kind === 'dust' && st.outcome === 'done') {
        lines.push(`swallowed ${Math.round(100 * (st.swallowed ?? 0))}% of the grains`);
      }
    } else {
      lines.push('Pick an object and press Throw.');
    }
    this.statusEl.innerHTML = lines.join('<br>');
    this.drawFallback(L?.kind === 'star' ? L.stats?.fallback : null);
  }

  drawFallback(fb) {
    const cv = this.chart;
    if (!fb || fb.t.length < 3) { cv.hidden = true; this.chartFor = null; return; }
    cv.hidden = false;
    if (this.chartFor === fb) return;
    this.chartFor = fb;
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height, pad = { l: 64, r: 14, t: 26, b: 46 };
    ctx.clearRect(0, 0, W, H);
    const unit = fb.tMin > 2 * 86400 ? [86400, 'days'] : fb.tMin > 7200 ? [3600, 'hours'] : [60, 'minutes'];
    const xs = fb.t.map((t) => Math.log10(t / unit[0]));
    const ys = fb.mdot.map((m) => Math.log10(m / fb.mdotEdd));
    const x0 = Math.min(...xs) - 0.1, x1 = Math.max(...xs) + 0.1;
    const y0 = Math.min(Math.min(...ys), -0.5) - 0.3, y1 = Math.max(Math.max(...ys), 0.5) + 0.3;
    const X = (x) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
    const Y = (y) => H - pad.b - ((y - y0) / (y1 - y0)) * (H - pad.t - pad.b);
    ctx.font = '20px system-ui, sans-serif';
    ctx.fillStyle = '#9aa1b2';
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    for (let k = Math.ceil(y0); k <= y1; k++) {
      ctx.beginPath(); ctx.moveTo(pad.l, Y(k)); ctx.lineTo(W - pad.r, Y(k)); ctx.stroke();
      ctx.fillText(`10${supStr(k)}`, 8, Y(k) + 7);
    }
    for (let k = Math.ceil(x0); k <= x1; k++) {
      ctx.beginPath(); ctx.moveTo(X(k), pad.t); ctx.lineTo(X(k), H - pad.b); ctx.stroke();
      ctx.fillText(`10${supStr(k)}`, X(k) - 14, H - pad.b + 24);
    }
    ctx.fillText(`time since disruption (${unit[1]})`, pad.l + 60, H - 6);
    ctx.fillText('fallback rate / Eddington', pad.l, 20);
    // Eddington level.
    ctx.strokeStyle = '#6a7287';
    ctx.setLineDash([6, 6]);
    ctx.beginPath(); ctx.moveTo(pad.l, Y(0)); ctx.lineTo(W - pad.r, Y(0)); ctx.stroke();
    // t^(−5/3) through the peak.
    const ip = ys.indexOf(Math.max(...ys));
    ctx.strokeStyle = '#7fb0ff';
    ctx.beginPath();
    ctx.moveTo(X(xs[ip]), Y(ys[ip]));
    ctx.lineTo(X(x1), Y(ys[ip] - (5 / 3) * (x1 - xs[ip])));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#7fb0ff';
    ctx.fillText('t^−5/3', X(x1) - 80, Math.min(Y(ys[ip] - (5 / 3) * (x1 - xs[ip])) - 8, H - pad.b - 8));
    // Simulated histogram.
    ctx.strokeStyle = '#ffb35c';
    ctx.lineWidth = 3;
    ctx.beginPath();
    xs.forEach((x, i) => (i ? ctx.lineTo(X(x), Y(ys[i])) : ctx.moveTo(X(x), Y(ys[i]))));
    ctx.stroke();
  }
}

function supStr(k) {
  return String(k).replace('-', '⁻').replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]);
}

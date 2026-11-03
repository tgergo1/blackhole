// Front end for the object simulation: a module Web Worker when available,
// otherwise the same Simulation stepped on the main thread a few ms per frame.
import { Simulation } from './events.js';

export class SimClient {
  constructor(onData) {
    this.onData = onData;
    this.gen = 0;
    this.busy = false;
    this.log = [];    // adds since the last clear, replayed if the worker fails
    this.local = null;
    try {
      this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
      this.worker.onmessage = (e) => {
        const d = e.data;
        if (d.gen !== this.gen) return;
        this.busy = d.busy || d.nAdd < this.log.length;
        this.onData(d);
      };
      this.worker.onerror = (e) => {
        e.preventDefault?.();
        this.useLocal();
      };
    } catch {
      this.useLocal();
    }
  }

  get mode() {
    return this.worker ? 'worker' : 'main thread';
  }

  useLocal() {
    this.worker?.terminate();
    this.worker = null;
    this.local = new Simulation();
    for (const [kind, cfg] of this.log) this.local.add(kind, cfg);
    this.busy = this.local.busy;
  }

  add(kind, cfg) {
    this.log.push([kind, cfg]);
    this.busy = true;
    if (this.worker) { this.worker.postMessage({ cmd: 'add', kind, cfg }); }
    else this.local.add(kind, cfg);
  }

  clear() {
    this.gen++;
    this.log = [];
    this.busy = false;
    if (this.worker) this.worker.postMessage({ cmd: 'clear', gen: this.gen });
    else this.local = new Simulation();
  }

  /** Call once per frame (does the work itself only in main-thread mode). */
  tick(ms = 8) {
    if (!this.local || !this.local.busy) {
      if (this.local) this.busy = false;
      return;
    }
    const out = this.local.advance(ms);
    this.busy = out.busy;
    this.onData(out);
  }
}

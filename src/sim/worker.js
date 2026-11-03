// Runs the object simulation (src/sim/events.js) off the main thread and
// streams new world-line samples back to the renderer.
import { Simulation } from './events.js';

let sim = new Simulation();
let gen = 0;
let nAdd = 0; // adds received since the last clear
let timer = null;

function pump() {
  timer = null;
  const out = sim.advance(40);
  const transfer = out.entities.map((e) => e.samples.buffer);
  postMessage({ gen, nAdd, ...out }, transfer);
  if (sim.busy) timer = setTimeout(pump, 0);
}

onmessage = (e) => {
  const m = e.data;
  if (m.cmd === 'clear') {
    sim = new Simulation();
    gen = m.gen;
    nAdd = 0;
  } else if (m.cmd === 'add') {
    sim.add(m.kind, m.cfg);
    nAdd++;
  }
  if (!timer) timer = setTimeout(pump, 0);
};

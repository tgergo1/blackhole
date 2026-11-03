import { Renderer, DEFAULT_SCENE } from '../render/renderer.js';
import { horizons } from '../physics/kerr.js';
import { photonOrbitRadii, radiativeEfficiency } from '../physics/disk.js';

const DEG = Math.PI / 180;

const PRESETS = {
  'Ultramassive hole, faint warm disk': {},
  'Schwarzschild (non-spinning)': { a: 0, theta: 80 * DEG },
  'Near-extremal, edge-on': { a: 0.998, theta: 89.5 * DEG },
  'Face-on': { theta: 2 * DEG },
  'Retrograde disk': { a: -0.9, theta: 80 * DEG },
  'Quasar (10⁹ M☉, 0.3 Eddington)': { massSolar: 1e9, eddRatio: 0.3, fCol: 1.7, rOut: 60, exposureEV: -1.5 },
  'X-ray binary (10 M☉)': { massSolar: 10, eddRatio: 0.1, fCol: 1.7, exposureEV: 0 },
  'Lensing grid, no disk': { diskOn: false, sky: 2, skyBrightness: 3, theta: 85 * DEG, a: 0.99 },
  'Close fly-by': { r: 6, theta: 84 * DEG, fovDeg: 100, a: 0.9 },
};

const canvas = document.getElementById('view');
const hud = document.getElementById('hud');
const errorBox = document.getElementById('error');

function showError(msg) {
  errorBox.hidden = false;
  errorBox.textContent = msg;
}

let renderer;
try {
  renderer = new Renderer(canvas, { preserveDrawingBuffer: true });
} catch (e) {
  showError(`${e.message} This simulator needs a browser with WebGL2 and floating-point render targets.`);
  throw e;
}

// ---------- Scene state & validation ----------
function clampScene(s) {
  const out = { ...s };
  const rH = horizons(out.a).rPlus;
  // Static observers only exist outside the ergosphere: r > 1 + √(1 − a² cos²θ).
  if (out.observer === 'static') {
    const rErgo = 1 + Math.sqrt(1 - out.a * out.a * Math.cos(out.theta) ** 2);
    if (out.r < rErgo * 1.02) out.observer = 'zamo';
  }
  out.r = Math.max(out.r, rH * 1.03);
  out.theta = Math.min(Math.max(out.theta, 0), Math.PI);
  out.pitch = Math.min(Math.max(out.pitch, -1.5), 1.5);
  return out;
}

let target = { ...DEFAULT_SCENE };
let interacting = false;
let lastInteraction = 0;

function apply(patch, { exposureOnly = false } = {}) {
  target = clampScene({ ...target, ...patch });
  if (exposureOnly) renderer.setExposure(target.exposureEV);
  else {
    try {
      const rs = interacting ? Math.min(target.resolutionScale, 0.5) : target.resolutionScale;
      renderer.setScene({ ...target, resolutionScale: rs });
    } catch (e) {
      showError(e.message);
      return;
    }
  }
  errorBox.hidden = true;
  syncControls();
}

// ---------- Controls ----------
const el = (id) => document.getElementById(id);
const logRange = (input) => input.dataset.log?.split(',').map(Number);

const fmt = {
  a: (v) => v.toFixed(3),
  r: (v) => (v < 10 ? v.toFixed(2) : v.toFixed(1)),
  incl: (v) => v.toFixed(1),
  fovDeg: (v) => v.toFixed(1),
  massSolar: (v) => v.toExponential(1).replace('e+', '×10^').replace(/\^(\d+)/, (m, d) => sup(d)),
  eddRatio: (v) => (v < 0.01 ? v.toExponential(1) : v.toFixed(3)),
  rOut: (v) => v.toFixed(0),
  fCol: (v) => v.toFixed(2),
  skyBrightness: (v) => v.toFixed(2),
  exposureEV: (v) => (v >= 0 ? '+' : '') + v.toFixed(1),
  rtol: (v) => v.toExponential(0),
  resolutionScale: (v) => `${Math.round(v * 100)}%`,
};
function sup(d) {
  return d.replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]);
}

function readControl(id) {
  const input = el(id);
  const v = Number(input.value);
  const lr = logRange(input);
  if (lr) return lr[0] * Math.pow(lr[1] / lr[0], v);
  if ('exp10' in input.dataset) return Math.pow(10, v);
  return v;
}
function writeControl(id, value) {
  const input = el(id);
  const lr = logRange(input);
  if (lr) input.value = Math.log(value / lr[0]) / Math.log(lr[1] / lr[0]);
  else if ('exp10' in input.dataset) input.value = Math.log10(value);
  else input.value = value;
  const out = document.querySelector(`output[data-for="${id}"]`);
  if (out) out.textContent = fmt[id] ? fmt[id](value) : value;
}

const RANGE_IDS = ['a', 'r', 'fovDeg', 'massSolar', 'eddRatio', 'rOut', 'fCol', 'skyBrightness', 'exposureEV', 'rtol', 'resolutionScale'];

function syncControls() {
  for (const id of RANGE_IDS) writeControl(id, target[id]);
  writeControl('incl', target.theta / DEG);
  el('observer').value = target.observer;
  el('sky').value = String(target.sky);
  el('diskOn').checked = target.diskOn;
  el('limb').checked = target.limb;
}

for (const id of RANGE_IDS) {
  el(id).addEventListener('input', () => apply({ [id]: readControl(id) }, { exposureOnly: id === 'exposureEV' }));
}
el('incl').addEventListener('input', () => apply({ theta: readControl('incl') * DEG }));
el('observer').addEventListener('change', (e) => apply({ observer: e.target.value }));
el('sky').addEventListener('change', (e) => apply({ sky: Number(e.target.value) }));
el('diskOn').addEventListener('change', (e) => apply({ diskOn: e.target.checked }));
el('limb').addEventListener('change', (e) => apply({ limb: e.target.checked }));

const presetSel = el('preset');
for (const name of Object.keys(PRESETS)) presetSel.add(new Option(name, name));
presetSel.addEventListener('change', () => {
  target = { ...DEFAULT_SCENE };
  apply(PRESETS[presetSel.value]);
});
el('reset').addEventListener('click', () => apply({ phi: 0, yaw: 0, pitch: 0, theta: DEFAULT_SCENE.theta, r: DEFAULT_SCENE.r, fovDeg: DEFAULT_SCENE.fovDeg }));
el('snapshot').addEventListener('click', () => {
  const a = document.createElement('a');
  a.download = `kerr-a${target.a.toFixed(3)}-i${(target.theta / DEG).toFixed(0)}.png`;
  a.href = canvas.toDataURL('image/png');
  a.click();
});

const toggle = el('toggle');
toggle.addEventListener('click', () => {
  const hidden = document.body.classList.toggle('panel-hidden');
  toggle.setAttribute('aria-expanded', String(!hidden));
});
if (window.matchMedia('(max-width: 700px)').matches) {
  document.body.classList.add('panel-hidden');
  toggle.setAttribute('aria-expanded', 'false');
}

// ---------- Mouse / touch navigation ----------
let drag = null;
canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, look: e.shiftKey || e.button === 2 };
});
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX;
  drag.y = e.clientY;
  const k = (target.fovDeg * DEG) / canvas.clientHeight;
  beginInteraction();
  if (drag.look) apply({ yaw: target.yaw + dx * k, pitch: target.pitch - dy * k });
  else apply({ phi: target.phi - dx * 0.006, theta: target.theta - dy * 0.006 });
});
const endDrag = () => { drag = null; };
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  beginInteraction();
  apply({ r: target.r * Math.exp(e.deltaY * 0.001) });
}, { passive: false });

window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  const step = 0.05;
  const moves = {
    ArrowLeft: { phi: target.phi + step }, ArrowRight: { phi: target.phi - step },
    ArrowUp: { theta: target.theta - step }, ArrowDown: { theta: target.theta + step },
    '+': { r: target.r / 1.05 }, '=': { r: target.r / 1.05 }, '-': { r: target.r * 1.05 },
  };
  if (moves[e.key]) { e.preventDefault(); beginInteraction(); apply(moves[e.key]); }
});

// While the view moves, render at reduced resolution; refine when idle.
function beginInteraction() {
  lastInteraction = performance.now();
  if (!interacting) {
    interacting = true;
    renderer.scene.resolutionScale = Math.min(target.resolutionScale, 0.5);
  }
}

// ---------- HUD ----------
let fps = 0, lastT = performance.now();
function updateHud(samples) {
  const s = renderer.scene, d = renderer.derived;
  const ph = photonOrbitRadii(s.a);
  hud.innerHTML = [
    `a = ${s.a.toFixed(3)}  ·  r<sub>+</sub> = ${d.rH.toFixed(3)} M  ·  r<sub>ISCO</sub> = ${d.rIn.toFixed(3)} M`,
    `photon orbits ${ph.pro.toFixed(3)} / ${ph.retro.toFixed(3)} M  ·  efficiency ${(100 * radiativeEfficiency(s.a)).toFixed(1)}%`,
    `disk peak T = ${Math.round(d.tPeak).toLocaleString()} K at r = ${d.rPeak.toFixed(2)} M`,
    `camera r = ${s.r.toFixed(2)} M, i = ${(s.theta / DEG).toFixed(1)}°  ·  ${samples} spp  ·  ${fps.toFixed(0)} fps`,
  ].join('<br>');
}

function loop(t) {
  const dt = t - lastT;
  lastT = t;
  fps = 0.9 * fps + 0.1 * (1000 / Math.max(dt, 1));
  if (interacting && t - lastInteraction > 200) {
    interacting = false;
    renderer.scene.resolutionScale = target.resolutionScale;
    renderer.sample = 0;
  }
  const samples = renderer.frame(interacting ? 1 : (window.__maxSamples ?? 64));
  updateHud(samples);
  requestAnimationFrame(loop);
}

apply({});
requestAnimationFrame(loop);
window.__renderer = renderer;

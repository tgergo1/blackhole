// WebGL2 renderer: progressive (jittered, accumulated) geodesic ray tracing.

import { FULLSCREEN_VS, TRACE_FS, DISPLAY_FS } from './shaders.js';
import { makeCamera, OBSERVERS } from '../physics/camera.js';
import { horizons } from '../physics/kerr.js';
import { iscoRadius, fluxPeak, peakTemperature } from '../physics/disk.js';
import { buildBlackbodyLUT, LUT_LOG_T_MIN, LUT_LOG_T_MAX, relativeLuminance } from '../physics/color.js';

export const DEFAULT_SCENE = {
  a: 0.94,              // dimensionless spin a/M (negative = retrograde disk)
  r: 70,                // camera radius (M)
  theta: (76 * Math.PI) / 180, // camera polar angle (inclination)
  phi: 0,
  yaw: 0,
  pitch: 0,
  fovDeg: 28,
  observer: OBSERVERS.ZAMO,
  massSolar: 2e10,      // hole mass, sets disk temperature together with eddRatio
  eddRatio: 5e-4,
  fCol: 1.0,            // spectral hardening (colour-correction) factor
  limb: true,
  diskOn: true,
  rOut: 30,
  sky: 1,               // 0 none, 1 stars, 2 checkerboard
  skyBrightness: 3,
  exposureEV: 0.5,
  rtol: 2e-5,
  maxSteps: 1200,
  resolutionScale: 1,
};

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s);
    gl.deleteShader(s);
    throw new Error('Shader compile failed: ' + log);
  }
  return s;
}

function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, 'aPos');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('Link failed: ' + gl.getProgramInfoLog(p));
  const uniforms = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(p, i);
    uniforms[info.name] = gl.getUniformLocation(p, info.name);
  }
  return { p, u: uniforms };
}

// Halton sequence for sub-pixel jitter.
function halton(i, b) {
  let f = 1, r = 0;
  while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); }
  return r;
}

/** Roots of x³ − 3x + 2a = 0 used by the Page–Thorne flux. */
export function pageThorneRoots(a) {
  const th = Math.acos(a);
  return [2 * Math.cos((th - Math.PI) / 3), 2 * Math.cos((th + Math.PI) / 3), -2 * Math.cos(th / 3)];
}

/** Everything derived from a scene that the shader needs (pure, testable). */
export function deriveScene(scene) {
  const cam = makeCamera(scene);
  const rIn = iscoRadius(scene.a);
  const peak = fluxPeak(scene.a);
  const tPeak = peakTemperature(scene.massSolar, scene.eddRatio, scene.a);
  const rCam = scene.r;
  // Base exposure: map the brightest local-frame disk emission to ~1.
  const yPeak = relativeLuminance(scene.fCol * tPeak) / scene.fCol ** 4;
  return {
    cam,
    rIn,
    fPeak: peak.F,
    rPeak: peak.r,
    tPeak,
    rH: horizons(scene.a).rPlus,
    rEscape: Math.max(5000, 20 * rCam),
    roots: pageThorneRoots(scene.a),
    baseExposure: 0.6 / yPeak,
  };
}

export class Renderer {
  constructor(canvas, { preserveDrawingBuffer = false } = {}) {
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, preserveDrawingBuffer });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;
    this.canvas = canvas;
    this.floatRT = !!gl.getExtension('EXT_color_buffer_float');
    this.halfRT = !this.floatRT && !!gl.getExtension('EXT_color_buffer_half_float');
    this.floatLinear = !!gl.getExtension('OES_texture_float_linear');
    if (!this.floatRT && !this.halfRT) throw new Error('This GPU cannot render to floating-point textures.');
    this.trace = program(gl, FULLSCREEN_VS, TRACE_FS);
    this.display = program(gl, FULLSCREEN_VS, DISPLAY_FS);

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    const lut = buildBlackbodyLUT(1024);
    this.lutSize = lut.size;
    this.lut = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.lut);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, lut.size, 1, 0, gl.RGBA, gl.FLOAT, lut.data);
    this.setFilter(gl.NEAREST);

    this.targets = [];
    this.sample = 0;
    this.scene = { ...DEFAULT_SCENE };
    this.dirty = true;
  }

  setFilter(mode) {
    const gl = this.gl;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mode);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, mode);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  makeTarget(w, h, internal) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const fmt = internal ?? (this.floatRT ? gl.RGBA32F : gl.RGBA16F);
    gl.texImage2D(gl.TEXTURE_2D, 0, fmt, w, h, 0, gl.RGBA, this.floatRT ? gl.FLOAT : gl.HALF_FLOAT, null);
    this.setFilter(this.floatLinear || !this.floatRT ? gl.LINEAR : gl.NEAREST);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { tex, fb, w, h };
  }

  ensureTargets(w, h) {
    if (this.targets.length && this.targets[0].w === w && this.targets[0].h === h) return;
    const gl = this.gl;
    for (const t of this.targets) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); }
    this.targets = [this.makeTarget(w, h), this.makeTarget(w, h)];
    for (const t of this.targets) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    this.sample = 0;
  }

  setScene(patch) {
    const next = { ...this.scene, ...patch };
    const derived = deriveScene(next); // throws for invalid observers
    this.scene = next;
    this.derived = derived;
    this.sample = 0;
  }

  /** Change only post-processing (does not restart accumulation). */
  setExposure(ev) {
    this.scene.exposureEV = ev;
  }

  bindTraceUniforms(w, h, jitter, debug) {
    const gl = this.gl;
    const { u } = this.trace;
    const s = this.scene;
    const d = this.derived ?? (this.derived = deriveScene(s));
    const c = d.cam;
    gl.uniform3f(u.uCamPos, c.pos[0], c.pos[1], c.pos[2]);
    gl.uniform4fv(u.uCamU, c.u);
    gl.uniform4fv(u.uCamF, c.forward);
    gl.uniform4fv(u.uCamR, c.right);
    gl.uniform4fv(u.uCamUp, c.up);
    gl.uniform1f(u.uTanHalfFov, Math.tan((s.fovDeg * Math.PI) / 360));
    gl.uniform1f(u.uPixelAngle, (2 * Math.tan((s.fovDeg * Math.PI) / 360)) / h);
    gl.uniform2f(u.uResolution, w, h);
    gl.uniform2f(u.uJitter, jitter[0], jitter[1]);
    gl.uniform1f(u.uSpinT, c.aT);
    gl.uniform1f(u.uSpinPhys, s.a);
    gl.uniform1f(u.uRHorizon, d.rH);
    gl.uniform1f(u.uREscape, d.rEscape);
    gl.uniform1f(u.uRtol, s.rtol);
    gl.uniform1i(u.uMaxSteps, s.maxSteps);
    gl.uniform1i(u.uDiskOn, s.diskOn ? 1 : 0);
    gl.uniform1f(u.uRIn, d.rIn);
    gl.uniform1f(u.uROut, s.rOut);
    gl.uniform1f(u.uX0, Math.sqrt(d.rIn));
    gl.uniform3fv(u.uXRoots, d.roots);
    gl.uniform1f(u.uFPeak, d.fPeak);
    gl.uniform1f(u.uTPeak, d.tPeak);
    gl.uniform1f(u.uFCol, s.fCol);
    gl.uniform1i(u.uLimb, s.limb ? 1 : 0);
    gl.uniform1i(u.uSkyMode, s.sky);
    gl.uniform1f(u.uSkyBrightness, s.skyBrightness);
    gl.uniform2f(u.uLUTRange, LUT_LOG_T_MIN, LUT_LOG_T_MAX);
    gl.uniform1f(u.uLUTSize, this.lutSize);
    gl.uniform1i(u.uDebug, debug ? 1 : 0);
    gl.uniform1f(u.uRadianceScale, d.baseExposure);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.lut);
    gl.uniform1i(u.uLUT, 0);
    gl.uniform1i(u.uPrev, 1);
  }

  /** Render one more accumulated sample and present. Returns samples so far. */
  frame(maxSamples = 64) {
    const gl = this.gl;
    const s = this.scene;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const ch = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== cw || this.canvas.height !== ch) {
      this.canvas.width = cw;
      this.canvas.height = ch;
    }
    const w = Math.max(1, Math.round(cw * s.resolutionScale));
    const h = Math.max(1, Math.round(ch * s.resolutionScale));
    this.ensureTargets(w, h);
    gl.bindVertexArray(this.vao);

    if (this.sample < maxSamples) {
      const src = this.targets[this.sample % 2];
      const dst = this.targets[(this.sample + 1) % 2];
      gl.useProgram(this.trace.p);
      const n = this.sample;
      const jitter = n === 0 ? [0, 0] : [halton(n, 2) - 0.5, halton(n, 3) - 0.5];
      this.bindTraceUniforms(w, h, jitter, false);
      gl.uniform1f(this.trace.u.uSampleIndex, n);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, src.tex);
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb);
      gl.viewport(0, 0, w, h);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.sample++;
    }
    const img = this.targets[this.sample % 2];
    gl.useProgram(this.display.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, cw, ch);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, img.tex);
    gl.uniform1i(this.display.u.uImage, 0);
    gl.uniform1f(this.display.u.uExposure, Math.pow(2, s.exposureEV));
    gl.uniform2f(this.display.u.uDisplaySize, cw, ch);
    gl.uniform2f(this.display.u.uImageSize, w, h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    return this.sample;
  }

  /** Raw per-pixel ray data (result type, r, g, extra) for validation. */
  renderDebug(w, h) {
    const gl = this.gl;
    if (!this.floatRT) throw new Error('Debug readback needs EXT_color_buffer_float');
    const t = this.makeTarget(w, h, gl.RGBA32F);
    gl.bindVertexArray(this.vao);
    gl.useProgram(this.trace.p);
    this.bindTraceUniforms(w, h, [0, 0], true);
    gl.uniform1f(this.trace.u.uSampleIndex, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.lut); // anything but the render target
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
    gl.viewport(0, 0, w, h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const out = new Float32Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, out);
    gl.deleteTexture(t.tex);
    gl.deleteFramebuffer(t.fb);
    return out;
  }
}

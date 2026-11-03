// GLSL ES 3.00 shaders. The geodesic code is a direct port of
// src/physics/kerr.js and src/physics/tracer.js; see those files for the
// derivations. test/gpu/shader-vs-reference.mjs checks the two agree.

export const FULLSCREEN_VS = /* glsl */ `#version 300 es
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
`;

export const TRACE_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;

// Camera (mirror-spacetime Kerr–Schild components, computed in float64 on the CPU).
uniform vec3 uCamPos;
uniform vec4 uCamU, uCamF, uCamR, uCamUp;
uniform float uTanHalfFov, uAspect;
uniform vec2 uResolution, uJitter;

// Spacetime and integrator.
uniform float uSpinT;      // tracing spin (= −a, see camera.js)
uniform float uSpinPhys;   // physical spin a
uniform float uRHorizon, uREscape, uRtol;
uniform int uMaxSteps;

// Disk.
uniform bool uDiskOn;
uniform float uRIn, uROut;
uniform float uX0;         // √r_isco
uniform vec3 uXRoots;      // roots of x³ − 3x + 2a = 0
uniform float uFPeak, uTPeak, uFCol;
uniform bool uLimb;

// Sky.
uniform int uSkyMode;      // 0 none, 1 stars, 2 checker
uniform float uSkyBrightness;

uniform sampler2D uLUT;    // blackbody linear sRGB, log-spaced in T
uniform vec2 uLUTRange;    // log10 T min/max
uniform float uLUTSize;

uniform sampler2D uPrev;   // running mean of previous samples
uniform float uSampleIndex;
uniform int uDebug;        // 1: output raw ray data instead of colour
uniform float uRadianceScale; // base exposure, applied before storage so half-float targets cannot overflow

out vec4 outColor;

// ---------- Kerr–Schild geometry (see kerr.js) ----------
float ksRadius(vec3 x) {
  float a2 = uSpinT * uSpinT;
  float w = dot(x, x) - a2;
  float s = sqrt(w * w + 4.0 * a2 * x.z * x.z);
  float r2 = w >= 0.0 ? 0.5 * (w + s) : (2.0 * a2 * x.z * x.z) / max(s - w, 1e-30);
  return sqrt(max(r2, 1e-30));
}

void geodesicRHS(vec3 x, vec3 p, float E, out vec3 dx, out vec3 dp) {
  float a = uSpinT, a2 = a * a;
  float r = ksRadius(x);
  float r2 = r * r, z2 = x.z * x.z;
  float q = r2 + a2;
  float D = r2 + a2 * z2 / r2;
  vec3 gradR = vec3(x.x * r / D, x.y * r / D, x.z * q / (r * D));
  float den = r2 * r2 + a2 * z2;
  float f = 2.0 * r2 * r / den;
  float dfdr = 2.0 * r2 * (3.0 * a2 * z2 - r2 * r2) / (den * den);
  float dfdz = -4.0 * r2 * r * a2 * x.z / (den * den);
  vec3 k = vec3((r * x.x + a * x.y) / q, (r * x.y - a * x.x) / q, x.z / r);
  float S = x.x * p.x + x.y * p.y;
  float A = x.y * p.x - x.x * p.y;
  float kp = E + dot(k, p);
  float dKdr = S / q - 2.0 * r * (r * S + a * A) / (q * q) - x.z * p.z / r2;
  vec3 gradK = dKdr * gradR + vec3((r * p.x - a * p.y) / q, (r * p.y + a * p.x) / q, p.z / r);
  float c = f * kp;
  dx = p - c * k;
  dp = 0.5 * kp * kp * (dfdr * gradR + vec3(0.0, 0.0, dfdz)) + c * gradK;
}

// Project p back onto the null cone H = 0 (removes float32 drift).
vec3 projectNull(vec3 x, vec3 p, float E) {
  float a = uSpinT;
  float r = ksRadius(x);
  float q = r * r + a * a;
  float f = 2.0 * r * r * r / (r * r * r * r + a * a * x.z * x.z);
  vec3 k = vec3((r * x.x + a * x.y) / q, (r * x.y - a * x.x) / q, x.z / r);
  float K = dot(k, p);
  float A2 = dot(p, p) - f * K * K;
  float B2 = -2.0 * f * E * K;
  float C2 = -E * E * (1.0 + f);
  float disc = B2 * B2 - 4.0 * A2 * C2;
  if (A2 <= 0.0 || disc < 0.0) return p;
  float s = (-B2 + sqrt(disc)) / (2.0 * A2);
  return abs(s - 1.0) < 1e-2 ? p * s : p;
}

// ---------- Dormand–Prince 5(4) ----------
void rk4(inout vec3 x, inout vec3 p, float h, float E) {
  vec3 k1x, k1p, k2x, k2p, k3x, k3p, k4x, k4p;
  geodesicRHS(x, p, E, k1x, k1p);
  geodesicRHS(x + 0.5 * h * k1x, p + 0.5 * h * k1p, E, k2x, k2p);
  geodesicRHS(x + 0.5 * h * k2x, p + 0.5 * h * k2p, E, k3x, k3p);
  geodesicRHS(x + h * k3x, p + h * k3p, E, k4x, k4p);
  x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
  p += h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
}

// ---------- Disk physics (see disk.js) ----------
float fluxShape(float r) {
  float a = uSpinPhys;
  float x = sqrt(r);
  float x0 = uX0;
  if (x <= x0) return 0.0;
  float x1 = uXRoots.x, x2 = uXRoots.y, x3 = uXRoots.z;
  float t1 = 3.0 * (x1 - a) * (x1 - a) / (x1 * (x1 - x2) * (x1 - x3)) * log((x - x1) / (x0 - x1));
  float t2 = 3.0 * (x2 - a) * (x2 - a) / (x2 * (x2 - x1) * (x2 - x3)) * log((x - x2) / (x0 - x2));
  float t3 = 3.0 * (x3 - a) * (x3 - a) / (x3 * (x3 - x1) * (x3 - x2)) * log((x - x3) / (x0 - x3));
  float br = x - x0 - 1.5 * a * log(x / x0) - t1 - t2 - t3;
  return max(br, 0.0) / (x * x * x * x * (x * x * x - 3.0 * x + 2.0 * a));
}

vec3 blackbody(float T) {
  // Manual linear interpolation in log T of the float LUT.
  float lt = log(max(T, 1.0)) * 0.43429448190325176;
  float u = (lt - uLUTRange.x) / (uLUTRange.y - uLUTRange.x) * (uLUTSize - 1.0);
  if (u <= 0.0) return vec3(0.0);
  u = min(u, uLUTSize - 1.001);
  int i = int(floor(u));
  float t = u - float(i);
  vec3 c0 = texelFetch(uLUT, ivec2(i, 0), 0).rgb;
  vec3 c1 = texelFetch(uLUT, ivec2(i + 1, 0), 0).rgb;
  return mix(c0, c1, t);
}
float blackbodyY(float T) {
  float lt = log(max(T, 1.0)) * 0.43429448190325176;
  float u = clamp((lt - uLUTRange.x) / (uLUTRange.y - uLUTRange.x) * (uLUTSize - 1.0), 0.0, uLUTSize - 1.001);
  int i = int(floor(u));
  return mix(texelFetch(uLUT, ivec2(i, 0), 0).a, texelFetch(uLUT, ivec2(i + 1, 0), 0).a, u - float(i));
}

// ---------- Sky ----------
// Integer hash (PCG3D, Jarzynski & Olano 2020): no float precision artefacts.
uvec3 pcg3d(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
vec3 hash33(ivec3 p) { return vec3(pcg3d(uvec3(p))) * (1.0 / 4294967296.0); }
float hash13(vec3 p) { return hash33(ivec3(floor(p))).x; }
float valueNoise(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), f.x),
                 mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x),
                 mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}

uniform float uPixelAngle; // angular size of one pixel at the screen centre

// Stars are small blobs of fixed angular size on the celestial sphere. Lensing
// conserves surface brightness, so magnified images get brighter in total flux
// automatically. Each star is a blackbody seen with frequency ratio g.
vec3 starLayer(vec3 d, float cells, float density, float g, int salt) {
  vec3 c = d * cells;
  ivec3 cell = ivec3(floor(c));
  vec3 col = vec3(0.0);
  float sigma = max(0.7 * uPixelAngle, 2e-5);
  for (int ix = -1; ix <= 1; ix++)
  for (int iy = -1; iy <= 1; iy++)
  for (int iz = -1; iz <= 1; iz++) {
    ivec3 id = cell + ivec3(ix, iy, iz);
    vec3 cid = vec3(id) + 0.5;
    // Only cells straddling the sphere own a star (avoids radial duplicates).
    if (abs(length(cid) - cells) > 0.5) continue;
    vec3 h = hash33(id + ivec3(salt, 7 * salt, 13 * salt));
    if (h.x > density) continue;
    vec3 sp = normalize(cid + 0.8 * (hash33(id + ivec3(91, 17, salt)) - 0.5));
    if (dot(d, sp) < 0.0) continue;
    vec3 diff = d - sp;
    float w = exp(-0.5 * dot(diff, diff) / (sigma * sigma));
    if (w < 1e-4) continue;
    float T = 2600.0 + 28000.0 * pow(h.y, 4.0);
    // Flux per star follows a steep power law; peak ∝ flux / σ².
    float flux = 1.5e-10 * pow(1.0 - h.z, -2.0) * (h.z < 0.999 ? 1.0 : 0.0);
    col += w * flux / (sigma * sigma) * blackbody(g * T) / blackbodyY(T);
  }
  return col;
}

vec3 sky(vec3 d, float g) {
  if (uSkyMode == 0) return vec3(0.0);
  if (uSkyMode == 2) {
    float th = acos(clamp(d.z, -1.0, 1.0));
    float ph = atan(d.y, d.x);
    float c = mod(floor(th / 3.14159265 * 18.0) + floor((ph + 3.14159265) / 6.2831853 * 36.0), 2.0);
    vec3 base = mix(vec3(0.05, 0.08, 0.12), vec3(0.6, 0.62, 0.66), c);
    if (abs(d.z) < 0.02) base = vec3(0.8, 0.2, 0.1);
    return base * uSkyBrightness * 0.2;
  }
  vec3 col = starLayer(d, 80.0, 0.3, g, 1) + starLayer(d, 220.0, 0.12, g, 2);
  // Faint galactic band along a tilted great circle.
  vec3 nrm = normalize(vec3(0.3, -0.5, 0.81));
  float lat = dot(d, nrm);
  float band = exp(-lat * lat / 0.015);
  float n = valueNoise(d * 6.0) * 0.6 + valueNoise(d * 14.0) * 0.3 + valueNoise(d * 30.0) * 0.1;
  col += band * n * n * 0.0025 * blackbody(g * 5200.0) / blackbodyY(5200.0);
  return col * uSkyBrightness;
}

// ---------- Main ----------
void main() {
  vec2 frag = gl_FragCoord.xy + uJitter;
  vec2 s = (2.0 * frag - uResolution) / uResolution.y; // y in [−1, 1]
  float dxs = s.x * uTanHalfFov, dys = s.y * uTanHalfFov;
  vec4 dir = (uCamF + dxs * uCamR + dys * uCamUp) / sqrt(1.0 + dxs * dxs + dys * dys);
  vec4 pu = uCamU + dir; // p^μ, normalised so that p·u_obs = −1

  // Lower the index with the Kerr–Schild metric at the camera.
  vec3 x = uCamPos;
  float a = uSpinT;
  float r = ksRadius(x);
  float q = r * r + a * a;
  float f = 2.0 * r * r * r / (r * r * r * r + a * a * x.z * x.z);
  vec3 k = vec3((r * x.x + a * x.y) / q, (r * x.y - a * x.x) / q, x.z / r);
  float lp = pu.x + dot(k, pu.yzw); // l_μ p^μ with l = (1, k)
  float E = pu.x - f * lp;          // E = −p_t = p^t − f (l·p)
  vec3 p = pu.yzw + f * lp * k;

  vec3 kx, kp;
  geodesicRHS(x, p, E, kx, kp);
  float h = 0.01 * max(1.0, r);
  int result = 0; // 0 undetermined, 1 horizon, 2 disk, 3 sky
  float gOut = 0.0, rOut = 0.0, muOut = 0.0;
  vec3 skyDir = vec3(0.0);
  int steps = 0;
  const float EPS = 1e-10;

  for (int n = 0; n < 4096; n++) {
    if (n >= uMaxSteps) break;
    steps = n;
    float rc = ksRadius(x);
    h = min(h, 0.5 * max(rc, 0.1));

    // --- DOPRI5 stages ---
    vec3 k1x = kx, k1p = kp, k2x, k2p, k3x, k3p, k4x, k4p, k5x, k5p, k6x, k6p, k7x, k7p;
    geodesicRHS(x + h * (0.2 * k1x), p + h * (0.2 * k1p), E, k2x, k2p);
    geodesicRHS(x + h * (3.0/40.0 * k1x + 9.0/40.0 * k2x),
                p + h * (3.0/40.0 * k1p + 9.0/40.0 * k2p), E, k3x, k3p);
    geodesicRHS(x + h * (44.0/45.0 * k1x - 56.0/15.0 * k2x + 32.0/9.0 * k3x),
                p + h * (44.0/45.0 * k1p - 56.0/15.0 * k2p + 32.0/9.0 * k3p), E, k4x, k4p);
    geodesicRHS(x + h * (19372.0/6561.0 * k1x - 25360.0/2187.0 * k2x + 64448.0/6561.0 * k3x - 212.0/729.0 * k4x),
                p + h * (19372.0/6561.0 * k1p - 25360.0/2187.0 * k2p + 64448.0/6561.0 * k3p - 212.0/729.0 * k4p), E, k5x, k5p);
    geodesicRHS(x + h * (9017.0/3168.0 * k1x - 355.0/33.0 * k2x + 46732.0/5247.0 * k3x + 49.0/176.0 * k4x - 5103.0/18656.0 * k5x),
                p + h * (9017.0/3168.0 * k1p - 355.0/33.0 * k2p + 46732.0/5247.0 * k3p + 49.0/176.0 * k4p - 5103.0/18656.0 * k5p), E, k6x, k6p);
    vec3 yx = x + h * (35.0/384.0 * k1x + 500.0/1113.0 * k3x + 125.0/192.0 * k4x - 2187.0/6784.0 * k5x + 11.0/84.0 * k6x);
    vec3 yp = p + h * (35.0/384.0 * k1p + 500.0/1113.0 * k3p + 125.0/192.0 * k4p - 2187.0/6784.0 * k5p + 11.0/84.0 * k6p);
    geodesicRHS(yx, yp, E, k7x, k7p);
    vec3 ex = h * (71.0/57600.0 * k1x - 71.0/16695.0 * k3x + 71.0/1920.0 * k4x - 17253.0/339200.0 * k5x + 22.0/525.0 * k6x - 0.025 * k7x);
    vec3 ep = h * (71.0/57600.0 * k1p - 71.0/16695.0 * k3p + 71.0/1920.0 * k4p - 17253.0/339200.0 * k5p + 22.0/525.0 * k6p - 0.025 * k7p);
    float scx = uRtol * max(rc, 1.0);
    vec3 scp = uRtol * max(max(abs(p), abs(yp)), vec3(E));
    float err = sqrt((dot(ex, ex) / (scx * scx) + dot(ep / scp, ep / scp)) / 6.0);

    if (err > 1.0 || isnan(err)) {
      h *= max(0.1, 0.9 * pow(max(err, EPS), -0.2));
      if (isnan(err)) h *= 0.1;
      continue;
    }

    // --- Disk crossing ---
    if (uDiskOn && x.z * yx.z <= 0.0 && x.z != yx.z) {
      vec3 hx = abs(x.z) < abs(yx.z) ? x : yx;
      vec3 hp = abs(x.z) < abs(yx.z) ? p : yp;
      for (int it = 0; it < 3; it++) {
        vec3 dx0, dp0;
        geodesicRHS(hx, hp, E, dx0, dp0);
        if (abs(dx0.z) < 1e-20) break;
        rk4(hx, hp, -hx.z / dx0.z, E);
      }
      float rd = sqrt(max(hx.x * hx.x + hx.y * hx.y - a * a, 0.0));
      if (rd >= uRIn && rd <= uROut) {
        // Disk gas: prograde Kepler in the physical spacetime → Ω flips under T.
        float om = -1.0 / (pow(rd, 1.5) + uSpinPhys);
        // u^μ = u^t (1, −Ω y, Ω x, 0), normalised with the mirror metric at z = 0.
        float qd = rd * rd + a * a;
        vec3 kd = vec3((rd * hx.x + a * hx.y) / qd, (rd * hx.y - a * hx.x) / qd, 0.0);
        float fd = 2.0 / rd;
        vec4 v = vec4(1.0, -om * hx.y, om * hx.x, 0.0);
        float lv = v.x + dot(kd, v.yzw);
        float nv = -v.x * v.x + dot(v.yzw, v.yzw) + fd * lv * lv;
        float ut = 1.0 / sqrt(max(-nv, 1e-30));
        float pDotU = ut * (-E + hp.x * v.y + hp.y * v.z);
        gOut = -1.0 / pDotU;
        muOut = min(1.0, abs(hp.z) / abs(pDotU));
        rOut = rd;
        result = 2;
        x = hx; p = hp;
        break;
      }
    }

    x = yx;
    p = projectNull(yx, yp, E);
    geodesicRHS(x, p, E, kx, kp);
    float rn = ksRadius(x);
    if (rn < uRHorizon * (1.0 - 1e-4)) { result = 1; rOut = rn; break; }
    if (rn > uREscape && dot(x, kx) > 0.0) {
      result = 3;
      skyDir = normalize(kx);
      gOut = 1.0 / E;
      rOut = rn;
      break;
    }
    h *= min(5.0, max(0.2, 0.9 * pow(max(err, EPS), -0.2)));
  }

  if (uDebug == 1) {
    // Raw data for validation against the float64 reference.
    float extra = result == 3 ? atan(skyDir.y, skyDir.x) : muOut;
    outColor = vec4(float(result) + float(steps) / 8192.0, rOut, gOut, extra);
    if (result == 3) outColor.y = skyDir.z;
    return;
  }

  vec3 col = vec3(0.0);
  if (result == 2) {
    float F = fluxShape(rOut);
    float T = uTPeak * pow(max(F / uFPeak, 0.0), 0.25);
    // Colour-corrected blackbody: I_ν = f⁻⁴ B_ν(f g T).
    col = blackbody(uFCol * gOut * T) / pow(uFCol, 4.0);
    if (uLimb) col *= (1.0 + 2.06 * muOut) / (1.0 + 2.06 * 2.0 / 3.0);
  } else if (result == 3) {
    col = sky(skyDir, gOut);
  }

  col = min(col * uRadianceScale, vec3(6e4)); // keep within half-float range
  if (uSampleIndex < 0.5) { outColor = vec4(col, 1.0); return; }
  vec3 prev = texelFetch(uPrev, ivec2(gl_FragCoord.xy), 0).rgb;
  outColor = vec4(mix(prev, col, 1.0 / (uSampleIndex + 1.0)), 1.0);
}
`;

export const DISPLAY_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uImage;
uniform float uExposure;
uniform vec2 uDisplaySize, uImageSize;
out vec4 outColor;

// ACES filmic approximation (Narkowicz 2015).
vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
vec3 srgbOETF(vec3 c) {
  return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
void main() {
  vec2 uv = gl_FragCoord.xy / uDisplaySize;
  vec3 c = texture(uImage, uv).rgb * uExposure;
  // Out-of-gamut blackbody colours (very hot/cool) have negative components:
  // desaturate toward their luminance instead of clipping hue.
  float Y = dot(c, vec3(0.2126, 0.7152, 0.0722));
  float m = min(c.r, min(c.g, c.b));
  if (m < 0.0) c = mix(c, vec3(Y), clamp(-m / max(Y - m, 1e-9), 0.0, 1.0));
  c = max(c, 0.0);
  outColor = vec4(srgbOETF(aces(c)), 1.0);
}
`;

// Blackbody colourimetry.
//
// A blackbody of temperature T seen with frequency ratio g = ν_obs/ν_emit is
// again an exact blackbody of temperature gT, because I_ν/ν³ is a Lorentz and
// gravitational invariant. So every pixel's spectrum is B_ν(gT) and its colour
// is obtained by integrating Planck's law against the CIE 1931 2° observer.
// This automatically includes relativistic beaming (bolometric I ∝ g⁴).

const H = 6.62607015e-34;
const C = 299792458;
const KB = 1.380649e-23;

// Multi-lobe Gaussian fit to the CIE 1931 colour-matching functions
// (Wyman, Sloan & Shirley 2013, JCGT 2(2)). λ in nm.
function lobe(l, mu, s1, s2) {
  const t = (l - mu) / (l < mu ? s1 : s2);
  return Math.exp(-0.5 * t * t);
}
export function cieXYZ(l) {
  return [
    1.056 * lobe(l, 599.8, 37.9, 31.0) + 0.362 * lobe(l, 442.0, 16.0, 26.7) - 0.065 * lobe(l, 501.1, 20.4, 26.2),
    0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1),
    1.217 * lobe(l, 437.0, 11.8, 36.0) + 0.681 * lobe(l, 459.0, 26.0, 13.8),
  ];
}

/** Planck spectral radiance B_λ(T) in W·sr⁻¹·m⁻³ (λ in nm). */
export function planckLambda(lnm, T) {
  const l = lnm * 1e-9;
  const x = (H * C) / (l * KB * T);
  if (x > 700) return 0;
  return (2 * H * C * C) / (l ** 5 * Math.expm1(x));
}

/** CIE XYZ of a blackbody (absolute, W·sr⁻¹·m⁻², integrated 360–830 nm). */
export function blackbodyXYZ(T) {
  const out = [0, 0, 0];
  if (!(T > 0)) return out;
  const dl = 1;
  for (let l = 360; l <= 830; l += dl) {
    const b = planckLambda(l, T) * dl * 1e-9;
    const c = cieXYZ(l);
    out[0] += b * c[0];
    out[1] += b * c[1];
    out[2] += b * c[2];
  }
  return out;
}

// XYZ → linear sRGB (D65).
export const XYZ_TO_SRGB = [
  [3.2404542, -1.5371385, -0.4985314],
  [-0.969266, 1.8760108, 0.041556],
  [0.0556434, -0.2040259, 1.0572252],
];

export function xyzToLinearSRGB(v) {
  return XYZ_TO_SRGB.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
}

/** Chromaticity (x, y) of a blackbody, for testing against the Planckian locus. */
export function blackbodyChromaticity(T) {
  const [X, Y, Z] = blackbodyXYZ(T);
  const s = X + Y + Z;
  return [X / s, Y / s];
}

// Lookup table range: log10(T / K).
export const LUT_LOG_T_MIN = 2.5;
export const LUT_LOG_T_MAX = 9.0;

/**
 * Build a table of linear-sRGB radiance for log-spaced temperatures.
 * Values are divided by `norm` (luminance of a reference temperature) so the
 * table stays in a comfortable float range; returns Float32Array RGBA.
 */
export function buildBlackbodyLUT(size = 1024, refT = 6500) {
  const norm = blackbodyXYZ(refT)[1];
  const data = new Float32Array(size * 4);
  for (let i = 0; i < size; i++) {
    const logT = LUT_LOG_T_MIN + ((LUT_LOG_T_MAX - LUT_LOG_T_MIN) * i) / (size - 1);
    const xyz = blackbodyXYZ(10 ** logT);
    const rgb = xyzToLinearSRGB(xyz);
    data[4 * i] = rgb[0] / norm;
    data[4 * i + 1] = rgb[1] / norm;
    data[4 * i + 2] = rgb[2] / norm;
    data[4 * i + 3] = xyz[1] / norm;
  }
  return { data, size, norm };
}

/** Luminance Y of a blackbody relative to `refT` (same normalisation as the LUT). */
export function relativeLuminance(T, refT = 6500) {
  return blackbodyXYZ(T)[1] / blackbodyXYZ(refT)[1];
}

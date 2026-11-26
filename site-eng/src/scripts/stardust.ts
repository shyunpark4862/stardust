/*
 * The hero backdrop: a barred spiral galaxy seen at an angle, with a sparse
 * field of stars behind it.
 *
 * Every particle is one GL point on a circular orbit. Its orbit (radius,
 * starting angle, height) is fixed at start-up; the vertex shader advances
 * the angle from the clock, so the CPU does no per-frame work beyond setting
 * a few uniforms.
 *
 * The arms are a density wave, as in a real disc: a spiral pattern that turns
 * at its own, slower speed while the stars pass through it. A particle is lit
 * by where it stands in the pattern, so the arms keep their shape instead of
 * winding up. Young blue stars and pink gas clouds show only inside the arms;
 * the old, yellower disc dims between them; dark dust lines the inner edge.
 *
 * Two thin jets leave the centre along the axis of the disc, as they do from
 * the black hole of an active galaxy: narrow at the base, widening and
 * fading as the matter streams outward.
 */

/**
 * What a galaxy looks like, whichever way its particles are moved: the
 * spiral pattern, the jets and the colours.
 */
export const GALAXY_GLSL = /* glsl */ `
const float CAMERA = 3.4;
const float BAR = 0.26;      // inside this radius the arms straighten into a bar
const float WIND = 3.1;      // cot(pitch angle): how tightly the arms wrap
const float PATTERN = 0.03;  // angular speed of the spiral pattern
const float JET = 1.25;      // length of each jet

// How strongly the spiral pattern lights a point, 0 (between arms) to 1.
float arms(float r, float ang, float sharp, float offset) {
  float lr = log(max(r, BAR) / BAR);
  float psi = 2.0 * (ang + WIND * lr) + offset;
  // Real arms wander: bend the ridge a little along its length.
  psi += 0.4 * sin(3.7 * lr + 1.3) + 0.22 * sin(3.0 * ang + 4.0 * r);
  float main = pow(0.5 + 0.5 * cos(psi), sharp);
  // Fainter spurs branch off in the outer disc.
  float psi2 = 2.0 * (ang + WIND * 1.35 * lr) + offset + 2.2;
  float spur = 0.55 * smoothstep(0.6, 1.2, r) * pow(0.5 + 0.5 * cos(psi2), sharp * 1.4);
  // Brighter and fainter stretches along an arm.
  float knots = 0.72 + 0.28 * sin(13.0 * lr + 4.0 * ang) * sin(7.0 * lr - 5.0 * ang + 1.0);
  return max(main, spur) * knots;
}

// A jet particle at a moment in time. "along" holds its side (sign) and
// where along the jet it starts (0..1): it streams outward and starts over
// at the base. "across" is its place across the beam, which opens into a
// narrow cone. "grow" (0..1) switches the jet on. Sets how strongly it is lit.
vec3 jetPoint(float across, float along, float turn, float time, float grow, out float lit) {
  float side = sign(along);
  float s = fract(abs(along) + time * 0.085);
  float len = JET * s * grow;
  // Needle-thin at the base; further out the beam opens and frays.
  float wide = across * (0.004 + 0.045 * len + 0.05 * len * len * len);
  float a = turn + 7.0 * len;
  // The beam is unstable: kinks grow with distance and travel outward.
  vec2 kink = 0.055 * len * len * vec2(sin(9.0 * len - time * 1.7 + side), cos(7.0 * len - time * 1.3));
  // Bright near the base, gone at the tip. Shocks run up the beam as
  // bright knots, and the source itself flickers.
  float shock = 0.5 + 0.5 * sin(17.0 * len - time * 2.4 + side * 1.3);
  float knots = 0.55 + 1.1 * shock * shock * shock;
  float flicker = 0.88 + 0.12 * sin(time * 3.1 + side);
  lit = pow(1.0 - s, 1.5) * smoothstep(0.0, 0.03, s) * knots * flicker * smoothstep(0.55, 1.0, grow);
  return vec3(cos(a) * wide + kink.x, side * (0.02 + len), sin(a) * wide + kink.y);
}

// Colour runs from young blue through white to the warm light of old
// stars; 2 is the pink of glowing gas, below 0 is dust (no light at all).
vec3 tint(float t) {
  vec3 cool = vec3(0.30, 0.50, 1.00);
  vec3 pale = vec3(0.80, 0.88, 1.00);
  vec3 cream = vec3(1.00, 0.93, 0.82);
  vec3 warm = vec3(1.00, 0.76, 0.52);
  if (t > 1.5) return vec3(1.00, 0.40, 0.56);
  if (t < 0.0) return vec3(0.0);
  return t < 0.5 ? mix(cool, pale, t / 0.5)
       : t < 0.8 ? mix(pale, cream, (t - 0.5) / 0.3)
       : mix(cream, warm, min((t - 0.8) / 0.2, 1.0));
}
`;

const VERT = /* glsl */ `
attribute vec4 aOrbit; // radius, start angle, height, kind (0 disc, 1 field, 2 jet)
attribute vec4 aLook;  // size, brightness, twinkle phase, colour
attribute vec4 aWave;  // arm contrast, arm sharpness, phase offset, twinkle amount

uniform float uTime;
uniform float uIntro;  // 0 -> 1 while the disc unfolds
uniform float uFade;   // 1 -> 0 as the hero scrolls away
uniform float uDpr;
uniform float uScale;
uniform vec2 uRes;
uniform vec2 uCenter;
uniform vec2 uTilt;    // pitch, roll

varying float vAlpha;
varying vec3 vColor;
${GALAXY_GLSL}
void main() {
  float r = aOrbit.x;
  float kind = aOrbit.w;

  // A flat rotation curve (omega ~ 1/r), as measured in real discs: inner
  // orbits lap the outer ones. The field barely drifts.
  float omega = mix(0.055 / (0.09 + r), 0.006, kind);
  float orbit = aOrbit.y + uTime * omega;
  float wind = (1.0 - uIntro) * 1.6 / (0.35 + r);
  float theta = orbit - wind * (1.0 - kind);

  float grow = mix(mix(0.18, 1.0, uIntro), 1.0, kind);
  vec3 p = vec3(cos(theta) * r * grow, aOrbit.z * grow, sin(theta) * r * grow);
  float lit = mix(1.0, arms(r, orbit - uTime * PATTERN, aWave.y, aWave.z), aWave.x);
  if (kind > 1.5) p = jetPoint(r, aOrbit.z, aOrbit.y, uTime, uIntro, lit);

  float cp = cos(uTilt.x), sp = sin(uTilt.x);
  p = vec3(p.x, p.y * cp - p.z * sp, p.y * sp + p.z * cp);
  float cr = cos(uTilt.y), sr = sin(uTilt.y);
  p = vec3(p.x * cr - p.y * sr, p.x * sr + p.y * cr, p.z);

  float persp = CAMERA / max(CAMERA + p.z, 0.5);
  vec2 xy = p.xy * persp * uScale;
  xy.x *= uRes.y / uRes.x;
  gl_Position = vec4(uCenter + xy, 0.0, 1.0);

  float size = aLook.x * persp * uDpr * clamp(uRes.y / (uDpr * 820.0), 0.8, 1.5);
  gl_PointSize = max(size, 1.0);

  float twinkle = 1.0 - aWave.w + aWave.w * sin(uTime * (0.6 + aLook.z) + aLook.z * 40.0);
  // Points smaller than a pixel keep their energy by dimming instead.
  float sub = min(size, 1.0);
  vAlpha = 1.7 * aLook.y * lit * twinkle * sub * uFade * mix(uIntro, 1.0, min(kind, 1.0) * 0.6);
  vColor = tint(aLook.w);
}
`;

const FRAG = /* glsl */ `
precision mediump float;
varying float vAlpha;
varying vec3 vColor;

void main() {
  float d = length(gl_PointCoord - 0.5) * 2.0;
  float a = smoothstep(1.0, 0.0, d);
  a *= a * vAlpha;
  gl_FragColor = vec4(vColor * a, a);
}
`;

/** Deterministic PRNG (mulberry32): the sky looks the same on every visit. */
export function prng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface Particles {
  orbit: Float32Array;
  look: Float32Array;
  wave: Float32Array;
  /** Draw order: [0, back) lights, [back, back + dust) dust, the rest lights in front. */
  back: number;
  dust: number;
  count: number;
}

export function buildParticles(total: number, jets = true): Particles {
  const rand = prng(20261115);
  const gauss = () => {
    const u = Math.max(rand(), 1e-9);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  };
  const clamp = (x: number) => Math.min(Math.max(x, 0), 1);
  // Radius in an exponential disc of scale length h (surface density ~ exp(-r/h)).
  const disc = (h: number, min: number, max: number) => {
    let r: number;
    do r = -h * Math.log(Math.max(rand() * rand(), 1e-12));
    while (r < min || r > max);
    return r;
  };

  const orbit: number[] = [];
  const look: number[] = [];
  const wave: number[] = [];
  const add = (
    r: number, h: number, kind: number,
    size: number, bright: number, colour: number,
    contrast = 0, sharp = 1, offset = 0, twinkle = 0.22,
  ) => {
    orbit.push(r, rand() * Math.PI * 2, h, kind);
    look.push(size, bright, rand(), colour);
    wave.push(contrast, sharp, offset, twinkle);
  };
  const n = (share: number) => Math.round(total * share);

  // Background field: a wide, slow shell of faint stars.
  for (let i = n(0.06); i > 0; i--) {
    add(1.2 + rand() * 1.9, (rand() * 2 - 1) * 2.6, 1, 1.0 + Math.pow(rand(), 6) * 2.6, 0.25 + rand() * 0.55, rand());
  }
  // Unresolved starlight: large, very faint sprites that add up to the glow
  // of the disc, warm at the centre and blue outside.
  for (let i = n(0.08); i > 0; i--) {
    const r = disc(0.36, 0.04, 1.9);
    add(r, gauss() * 0.01, 0, 18 + rand() * 22, 0.017 + rand() * 0.014, clamp(0.76 - 0.85 * r), 0.9, 2.2, gauss() * 0.25, 0);
  }
  for (let i = n(0.006); i > 0; i--) {
    const r3 = Math.abs(gauss()) * 0.12;
    const c = rand() * 2 - 1;
    add(r3 * Math.sqrt(1 - c * c), r3 * c * 0.65, 0, 20 + rand() * 20, 0.008 + rand() * 0.008, 0.86 + rand() * 0.1, 0, 1, 0, 0);
  }
  // Bulge: old, warm stars in a flattened ball, dense at the centre with
  // long faint wings.
  for (let i = n(0.1); i > 0; i--) {
    const q = Math.sqrt(rand());
    const r3 = Math.min((0.07 * q) / (1 - q + 1e-3), 0.55);
    const c = rand() * 2 - 1;
    add(r3 * Math.sqrt(1 - c * c), r3 * c * 0.65, 0, 0.9 + rand() * 1.0, 0.1 + rand() * 0.16, 0.8 + rand() * 0.2);
  }
  // Old disc: dims between the arms but never goes dark.
  for (let i = n(0.32); i > 0; i--) {
    const r = disc(0.32, 0.06, 2.0);
    const colour = clamp(0.84 - r * 0.3 + (rand() - 0.5) * 0.3);
    add(r, gauss() * 0.014 * (0.6 + r), 0, 0.8 + Math.pow(rand(), 5) * 1.4, 0.18 + rand() * 0.3, colour, 0.66, 1.8);
  }
  // Young stars: blue, bright, and seen only inside the arms.
  const young = () => {
    const r = disc(0.44, 0.2, 2.0);
    let colour = 0.05 + rand() * 0.4;
    if (rand() < 0.012) colour = 0.92 + rand() * 0.08; // a rare red giant
    add(r, gauss() * 0.009 * (0.6 + r), 0, 0.9 + Math.pow(rand(), 6) * 2.8, 0.45 + rand() * 0.55, colour, 1, 3.2, gauss() * 0.3);
  };
  for (let i = n(0.27); i > 0; i--) young();
  // Jets: fine grains and a little soft light around them. The side that
  // points away from the viewer is drawn here, behind the dust.
  const jet = (side: number) => {
    for (let i = n(0.022); i > 0; i--) {
      const at = side * (0.001 + rand() * 0.998);
      // Mostly small soft sprites that overlap into one continuous beam;
      // the rest are grains that show the flow.
      if (rand() < 0.65) add(Math.abs(gauss()) * 0.55, at, 2, 4 + rand() * 6, 0.04 + rand() * 0.035, 0.2 + rand() * 0.25, 0, 1, 0, 0);
      else add(Math.abs(gauss()), at, 2, 0.9 + Math.pow(rand(), 4) * 1.6, 0.25 + rand() * 0.4, 0.25 + rand() * 0.3, 0, 1, 0, 0.2);
    }
  };
  if (jets) jet(1);
  const back = orbit.length / 4;

  // Dust: dark lanes along the inner edge of each arm.
  for (let i = n(0.045); i > 0; i--) {
    const r = disc(0.42, 0.16, 1.7);
    add(r, gauss() * 0.005, 0, 9 + rand() * 16, 0.06 + rand() * 0.06, -1, 1, 4, 0.75 + gauss() * 0.15, 0);
  }
  const dust = orbit.length / 4 - back;

  // In front of the dust: more young stars, and pink clouds of glowing gas
  // strung along the ridge of the arms.
  if (jets) jet(-1);
  for (let i = n(0.13); i > 0; i--) young();
  for (let i = n(0.008); i > 0; i--) {
    const r = disc(0.44, 0.3, 1.7);
    add(r, gauss() * 0.006, 0, 3 + rand() * 3.5, 0.2 + rand() * 0.22, 2, 1, 7, 0.15 + gauss() * 0.2, 0.08);
  }

  return {
    orbit: new Float32Array(orbit),
    look: new Float32Array(look),
    wave: new Float32Array(wave),
    back,
    dust,
    count: orbit.length / 4,
  };
}

function compile(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) ?? 'shader compile failed');
  }
  return shader;
}

export function mountStardust(canvas: HTMLCanvasElement, host: HTMLElement, jets = true): void {
  const gl = canvas.getContext('webgl', { alpha: true, antialias: false, premultipliedAlpha: true });
  if (!gl) return;

  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const small = matchMedia('(max-width: 800px)').matches;
  const total = small ? 30000 : 90000;

  const program = gl.createProgram()!;
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(program);
  gl.useProgram(program);

  const { orbit, look, wave, back, dust, count } = buildParticles(total, jets);
  for (const [name, data] of [['aOrbit', orbit], ['aLook', look], ['aWave', wave]] as const) {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(program, name);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 0, 0);
  }

  const uni = (name: string) => gl.getUniformLocation(program, name);
  const u = {
    time: uni('uTime'), intro: uni('uIntro'), fade: uni('uFade'), dpr: uni('uDpr'),
    scale: uni('uScale'), res: uni('uRes'), center: uni('uCenter'), tilt: uni('uTilt'),
  };

  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.clearColor(0, 0, 0, 0);

  let dpr = 1;
  let center: [number, number] = [0, 0];
  const resize = () => {
    dpr = Math.min(devicePixelRatio || 1, 2);
    const w = host.clientWidth;
    const h = host.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    const portrait = w / h < 0.9;
    center = portrait ? [0.1, -0.42] : [0.42, -0.08];
    gl.uniform1f(u.dpr, dpr);
    gl.uniform2f(u.res, canvas.width, canvas.height);
    gl.uniform2f(u.center, center[0], center[1]);
    gl.uniform1f(u.scale, portrait ? 0.6 : 0.95);
    // The CSS glow behind the core follows the same centre.
    host.style.setProperty('--core-x', `${(center[0] * 0.5 + 0.5) * 100}%`);
    host.style.setProperty('--core-y', `${(0.5 - center[1] * 0.5) * 100}%`);
  };

  // Pointer parallax, eased so the disc feels heavy.
  const pointer = { x: 0, y: 0, tx: 0, ty: 0 };
  addEventListener('pointermove', (e) => {
    pointer.tx = (e.clientX / innerWidth) * 2 - 1;
    pointer.ty = (e.clientY / innerHeight) * 2 - 1;
  }, { passive: true });

  const INTRO_MS = 3200;
  const start = performance.now();
  // Draw only while the hero is on screen and the tab is showing. The two
  // are tracked apart: coming back to the tab must not restart the loop
  // when the hero has been scrolled away.
  let onScreen = true;
  let pageShown = !document.hidden;
  const visible = () => onScreen && pageShown;
  let frame = 0;

  const draw = (now: number) => {
    frame = 0;
    const elapsed = now - start;
    const k = reduced ? 1 : Math.min(elapsed / INTRO_MS, 1);
    const intro = 1 - Math.pow(1 - k, 4);

    pointer.x += (pointer.tx - pointer.x) * 0.04;
    pointer.y += (pointer.ty - pointer.y) * 0.04;

    const fade = Math.max(0, 1 - scrollY / (host.clientHeight * 0.9));
    gl.uniform1f(u.time, reduced ? 40 : 40 + elapsed / 1000);
    gl.uniform1f(u.intro, intro);
    gl.uniform1f(u.fade, fade);
    gl.uniform2f(u.tilt, 0.62 + pointer.y * 0.05 - (1 - intro) * 0.22, -0.3 + pointer.x * 0.035);
    gl.clear(gl.COLOR_BUFFER_BIT);
    // Light adds up; dust takes light away, from the stars drawn so far and
    // from the page glow behind the canvas.
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArrays(gl.POINTS, 0, back);
    gl.blendFuncSeparate(gl.ZERO, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.POINTS, back, dust);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArrays(gl.POINTS, back + dust, count - back - dust);

    if (!reduced && visible()) frame = requestAnimationFrame(draw);
  };
  const kick = () => {
    if (!frame) frame = requestAnimationFrame(draw);
  };

  resize();
  new ResizeObserver(() => { resize(); kick(); }).observe(host);
  new IntersectionObserver(([entry]) => {
    onScreen = entry.isIntersecting;
    if (visible()) kick();
  }).observe(host);
  document.addEventListener('visibilitychange', () => {
    pageShown = !document.hidden;
    if (visible()) kick();
  });
  kick();
  canvas.classList.add('is-ready');
}

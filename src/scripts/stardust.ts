/*
 * The hero backdrop: a disc of dust in differential rotation, seen at a low
 * angle, with a sparse field of stars behind it.
 *
 * Every particle is one GL point. Its orbit (radius, starting angle, height)
 * is fixed at start-up; the vertex shader advances the angle from the clock,
 * so the CPU does no per-frame work beyond setting a few uniforms.
 */

const VERT = /* glsl */ `
attribute vec4 aOrbit; // radius, start angle, height, kind (0 disc, 1 field)
attribute vec4 aLook;  // size, brightness, twinkle phase, temperature

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

const float CAMERA = 3.4;

void main() {
  float r = aOrbit.x;
  float kind = aOrbit.w;

  // Inner orbits run faster (Kepler: omega ~ r^-1.5); the field barely drifts.
  float omega = mix(0.055 / (0.12 + r * sqrt(r)), 0.006, kind);
  float wind = (1.0 - uIntro) * 1.6 / (0.35 + r);
  float theta = aOrbit.y + uTime * omega - wind * (1.0 - kind);

  float grow = mix(mix(0.18, 1.0, uIntro), 1.0, kind);
  vec3 p = vec3(cos(theta) * r * grow, aOrbit.z * grow, sin(theta) * r * grow);

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

  float twinkle = 0.78 + 0.22 * sin(uTime * (0.6 + aLook.z) + aLook.z * 40.0);
  // Points smaller than a pixel keep their energy by dimming instead.
  float sub = min(size, 1.0);
  vAlpha = 1.7 * aLook.y * twinkle * sub * uFade * mix(uIntro, 1.0, kind * 0.6);

  vec3 cool = vec3(0.30, 0.50, 1.00);
  vec3 pale = vec3(0.80, 0.88, 1.00);
  vec3 warm = vec3(1.00, 0.86, 0.68);
  float t = aLook.w;
  vColor = t < 0.9 ? mix(cool, pale, t / 0.9) : mix(pale, warm, (t - 0.9) / 0.1);
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
function prng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Ring centres, widths and weights: dense bands separated by faint gaps.
const RINGS: [number, number, number][] = [
  [0.34, 0.05, 0.1],
  [0.52, 0.035, 0.12],
  [0.7, 0.05, 0.17],
  [0.93, 0.04, 0.16],
  [1.16, 0.07, 0.19],
  [1.48, 0.09, 0.16],
  [1.86, 0.13, 0.1],
];

function buildParticles(count: number): { orbit: Float32Array; look: Float32Array } {
  const rand = prng(20261115);
  const gauss = () => {
    const u = Math.max(rand(), 1e-9);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  };
  const orbit = new Float32Array(count * 4);
  const look = new Float32Array(count * 4);

  for (let i = 0; i < count; i++) {
    const pick = rand();
    let r: number, h: number, kind = 0, size: number, bright: number, temp: number;

    if (pick < 0.07) {
      // Background field: a wide, slow shell of faint stars.
      kind = 1;
      r = 1.2 + rand() * 1.9;
      h = (rand() * 2 - 1) * 2.6;
      size = 1.0 + Math.pow(rand(), 6) * 2.6;
      bright = 0.25 + rand() * 0.55;
      temp = rand();
    } else if (pick < 0.11) {
      // Core: a small bright bulge.
      r = Math.abs(gauss()) * 0.13;
      h = gauss() * 0.03;
      size = 0.9 + rand() * 1.0;
      bright = 0.3 + rand() * 0.3;
      temp = 0.75 + rand() * 0.25;
    } else if (pick < 0.26) {
      // Diffuse dust across the whole disc.
      r = 0.2 + Math.sqrt(rand()) * 1.95;
      h = gauss() * 0.02 * (0.6 + r);
      size = 0.9 + rand() * 1.1;
      bright = 0.16 + rand() * 0.3;
      temp = Math.max(0, 0.75 - r * 0.32 + (rand() - 0.5) * 0.4);
    } else {
      // Rings.
      let w = rand();
      let ring = RINGS[RINGS.length - 1];
      for (const candidate of RINGS) {
        if (w < candidate[2]) {
          ring = candidate;
          break;
        }
        w -= candidate[2];
      }
      r = Math.max(0.05, ring[0] + gauss() * ring[1]);
      h = gauss() * 0.009 * (0.5 + r);
      size = 1.0 + Math.pow(rand(), 5) * 2.6;
      bright = 0.45 + rand() * 0.55;
      temp = Math.max(0, 0.85 - r * 0.36 + (rand() - 0.5) * 0.35);
      if (rand() < 0.012) temp = 0.93 + rand() * 0.07; // a rare warm grain
    }

    orbit.set([r, rand() * Math.PI * 2, h, kind], i * 4);
    look.set([size, bright, rand(), Math.min(temp, 1)], i * 4);
  }
  return { orbit, look };
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

export function mountStardust(canvas: HTMLCanvasElement, host: HTMLElement): void {
  const gl = canvas.getContext('webgl', { alpha: true, antialias: false, premultipliedAlpha: true });
  if (!gl) return;

  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const small = matchMedia('(max-width: 800px)').matches;
  const count = small ? 16000 : 42000;

  const program = gl.createProgram()!;
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(program);
  gl.useProgram(program);

  const { orbit, look } = buildParticles(count);
  for (const [name, data] of [['aOrbit', orbit], ['aLook', look]] as const) {
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
  gl.blendFunc(gl.ONE, gl.ONE); // additive: overlapping dust adds up to light
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
    gl.uniform1f(u.scale, portrait ? 0.46 : 0.86);
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
    gl.uniform2f(u.tilt, 0.4 + pointer.y * 0.05 - (1 - intro) * 0.22, -0.3 + pointer.x * 0.035);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.POINTS, 0, count);

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

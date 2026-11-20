/*
 * The hero backdrop, second version: two spiral galaxies fall together,
 * pass through each other, throw out tidal tails, merge, and settle into the
 * one galaxy that stardust.ts draws. After that it turns as that one does.
 *
 * The collision is a restricted N-body run, the method of Toomre & Toomre
 * (1972): the stars do not pull on each other. Each one feels only the two
 * galaxy cores, and the cores follow one orbit worked out ahead of time. That
 * is enough to raise bridges and tails, and it costs one force evaluation
 * per star instead of one per pair.
 *
 * Keeping it light:
 * - The stars are moved by the GPU. The vertex shader that draws a star also
 *   advances it, and transform feedback writes the new position and velocity
 *   into a second buffer that is the input of the next frame. No star data
 *   crosses from the CPU after start-up; a frame sets a few uniforms.
 * - The CPU integrates two bodies once, at start-up, and afterwards only
 *   looks the core positions up.
 * - Nothing runs while the hero is off screen or the tab is hidden.
 * - If frames come slowly the canvas resolution steps down.
 * - Without WebGL 2, or when the reader asked for less motion, the finished
 *   galaxy of stardust.ts is shown instead, standing still.
 *
 * Time runs fast during the collision (a real one takes a billion years) and
 * eases to the calm pace of the finished galaxy as it settles. A readout on
 * the page shows the time elapsed and the pace.
 *
 * The camera travels with the story: it closes in for each passage, rises
 * to show the tails, circles the two cores as they merge, and draws back to
 * the resting view of the finished galaxy.
 */

import { GALAXY_GLSL, buildParticles, mountStardust, prng } from './stardust';

// ---- The two galaxies and their orbit (units of the finished galaxy) ----

/** Circular speed far out in the finished galaxy: v(r) = SPEED r / (CORE + r). */
const SPEED = 0.055;
const CORE = 0.09;
/** Share of the mass and of the stars, and size relative to the finished galaxy. */
const SHARE = [0.6, 0.4];
const SIZE = [0.6, 0.48];
/** Tilt of each disc away from the plane of the orbit, and the way it leans. */
const LEAN: [number, number][] = [[0.35, Math.PI / 2], [0.5, Math.PI]];
/** Where the second galaxy starts as seen from the first, and how it moves. */
const START = { x: 1.125, z: -2.375, inward: 0.03, sideways: 0.02 };
/** Drag between the two when they overlap: what makes them merge. */
const DRAG = { strength: 0.035, reach: 0.7 };
/** How much faster than the finished galaxy time runs during the collision. */
const WARP = 18;
/** Seconds over which the merged galaxy settles. */
const SETTLE_S = 5;
const WARP_EASE_S = 2.2;
/** Years in one unit of model time. A turn at mid-disc then takes some 600 million years. */
const MYR = 5;

/**
 * The camera's path. "at" runs 0 -> 1 through the collision (in model time)
 * and 1 -> 2 over the seconds of settling. Yaw circles the scene, pitch is
 * the height above the plane of the orbit, dist the distance to the centre.
 * The last key is the resting view of the finished galaxy.
 */
const CAMERA_KEYS = [
  { at: 0.0, yaw: 0.0, pitch: 0.62, roll: -0.3, dist: 3.6 },
  { at: 0.2, yaw: 0.2, pitch: 0.6, roll: -0.28, dist: 3.1 },
  { at: 0.33, yaw: 0.55, pitch: 0.5, roll: -0.22, dist: 2.2 }, // first passage, close
  { at: 0.5, yaw: 1.05, pitch: 0.8, roll: -0.18, dist: 3.0 }, // the tails, from above
  { at: 0.68, yaw: 1.55, pitch: 0.5, roll: -0.15, dist: 2.0 }, // second passage
  { at: 0.86, yaw: 2.05, pitch: 0.42, roll: -0.12, dist: 1.65 }, // circling the two cores
  { at: 1.0, yaw: 2.45, pitch: 0.46, roll: -0.15, dist: 1.9 },
  { at: 1.45, yaw: 3.0, pitch: 0.56, roll: -0.25, dist: 2.8 },
  { at: 2.0, yaw: 3.3, pitch: 0.62, roll: -0.3, dist: 3.4 },
] as const;
const CAMERA_SETTLE_S = 9;
type View = { yaw: number; pitch: number; roll: number; dist: number };

/** The camera at a point of the story: a smooth curve through the keys, at rest at both ends. */
function cameraAt(at: number, out: View): void {
  const keys = CAMERA_KEYS;
  const last = keys.length - 1;
  const t = Math.min(Math.max(at, keys[0].at), keys[last].at);
  let i = 0;
  while (i < last - 1 && t > keys[i + 1].at) i++;
  const a = keys[i], b = keys[i + 1];
  const span = b.at - a.at;
  const f = (t - a.at) / span;
  const f2 = f * f, f3 = f2 * f;
  for (const key of ['yaw', 'pitch', 'roll', 'dist'] as const) {
    // Cubic Hermite; the slope at a key is that of the chord across it.
    const ma = i === 0 ? 0 : (b[key] - keys[i - 1][key]) / (b.at - keys[i - 1].at);
    const mb = i + 1 === last ? 0 : (keys[i + 2][key] - a[key]) / (keys[i + 2].at - a.at);
    out[key] =
      (2 * f3 - 3 * f2 + 1) * a[key] + (f3 - 2 * f2 + f) * span * ma +
      (-2 * f3 + 3 * f2) * b[key] + (f3 - f2) * span * mb;
  }
}

interface Galaxy {
  speed2: number; // SPEED^2 scaled by the mass share
  core: number;
  edge: number; // past this radius it pulls like a point mass
  size: number;
  e1: [number, number, number]; // the disc plane: stars turn from e1 towards e2
  e2: [number, number, number];
  normal: [number, number, number];
  pattern: number; // angular speed of its spiral pattern
}

function galaxy(i: number): Galaxy {
  const [tilt, lean] = LEAN[i];
  const n: [number, number, number] = [Math.sin(tilt) * Math.cos(lean), Math.cos(tilt), Math.sin(tilt) * Math.sin(lean)];
  // e1 = y x n, e2 = e1 x n: with n along y this is (x, z), the finished disc.
  let e1: [number, number, number] = [n[2], 0, -n[0]];
  const len = Math.hypot(e1[0], e1[2]);
  e1 = len > 1e-6 ? [e1[0] / len, 0, e1[2] / len] : [1, 0, 0];
  const e2: [number, number, number] = [
    e1[1] * n[2] - e1[2] * n[1],
    e1[2] * n[0] - e1[0] * n[2],
    e1[0] * n[1] - e1[1] * n[0],
  ];
  const size = SIZE[i];
  const speed = SPEED * Math.sqrt(SHARE[i]);
  return {
    speed2: speed * speed,
    core: CORE * size,
    edge: 2.2 * size,
    size,
    e1,
    e2,
    normal: n,
    pattern: speed / (CORE * size + 1.74 * size),
  };
}

/** Pull towards a core at distance d, per unit of speed2: flat rotation curve inside the edge. */
function pull(d: number, g: Galaxy): number {
  const { core: a, edge: R } = g;
  return d < R ? d / ((a + d) * (a + d)) : (R * R * R) / ((a + R) * (a + R) * d * d);
}

const ORBIT_DT = 0.05;

/**
 * The path of one core as seen from the other, in the plane of the orbit:
 * (x, z) every ORBIT_DT. Returns also the moment from which the two stay
 * together, when settling begins.
 */
function coreOrbit(a: Galaxy, b: Galaxy): { path: Float32Array; settleAt: number; turns: number[] } {
  let x = START.x, z = START.z;
  const d0 = Math.hypot(x, z);
  let vx = (-START.inward * x - START.sideways * z) / d0;
  let vz = (-START.inward * z + START.sideways * x) / d0;
  const path: number[] = [];
  const turns: number[] = []; // model times of closest and farthest points, in turn
  let lastApart = 0;
  let before = Infinity, falling = true;
  for (let step = 0; step < 12000; step++) {
    const d = Math.hypot(x, z);
    path.push(x, z);
    if (d < before !== falling) {
      falling = !falling;
      turns.push(step * ORBIT_DT);
    }
    before = d;
    if (d > 0.12) lastApart = step;
    if (step - lastApart > 400 && d < 0.01) break;
    const g = a.speed2 * pull(d, a) + b.speed2 * pull(d, b);
    const drag = DRAG.strength / Math.pow(1 + (d / DRAG.reach) ** 2, 1.5);
    vx += ((-g * x) / Math.max(d, 1e-9) - drag * vx) * ORBIT_DT;
    vz += ((-g * z) / Math.max(d, 1e-9) - drag * vz) * ORBIT_DT;
    x += vx * ORBIT_DT;
    z += vz * ORBIT_DT;
  }
  return { path: new Float32Array(path), settleAt: lastApart * ORBIT_DT, turns };
}

// ---- Shaders ----

const VERT = /* glsl */ `#version 300 es
precision highp float;

in vec4 aPos;   // position, how strongly the spiral pattern lights it
in vec4 aVel;   // velocity
in vec4 aHome;  // orbit in the finished galaxy: radius, height; kind (0 star, 1 field); galaxy it starts in
in vec4 aLook;  // size, brightness, twinkle phase, colour
in vec4 aWave;  // arm contrast, arm sharpness, phase offset, twinkle amount

uniform float uDt;       // model time this frame
uniform float uWarp;     // model time per second
uniform float uClock;    // seconds, for twinkle
uniform float uSettle;   // 0 -> 1 as the merged galaxy settles
uniform vec2 uLight;     // this frame's pull of the lighting towards: its own galaxy's pattern, the finished one
uniform vec3 uPhase;     // angle of the spiral pattern: first galaxy, second, finished
uniform vec3 uCore[4];   // core of the first galaxy at the start and end of the frame, then the second
uniform vec4 uGal[2];    // speed^2, core radius, edge, size
uniform vec3 uE1[2];
uniform vec3 uE2[2];

uniform float uFade;     // 1 -> 0 as the hero scrolls away
uniform float uDpr;
uniform float uScale;
uniform vec2 uRes;
uniform vec2 uCenter;
uniform vec4 uView;      // camera: yaw, pitch, roll, distance to the centre

out vec4 oPos;
out vec4 oVel;
out float vAlpha;
out vec3 vColor;
${GALAXY_GLSL}
const float SPEED = ${SPEED.toFixed(4)};
const float CORE = ${CORE.toFixed(4)};

// Pull of one galaxy on a point at offset d from its core: a flat rotation
// curve inside the edge, a point mass beyond it.
vec3 pull(vec3 d, vec4 g) {
  float r = max(length(d), 1e-5);
  float a = g.y + r, e = g.y + g.z;
  float q = r < g.z ? r / (a * a) : g.z * g.z * g.z / (e * e * r * r);
  return -g.x * q * d / r;
}

void main() {
  vec3 x = aPos.xyz;
  vec3 v = aVel.xyz;
  float lit = aPos.w;
  float kind = aHome.z;
  float shown = 1.0;

  if (kind < 0.5) {
    // ---- Move the star: two half steps, the cores moving meanwhile ----
    float h = uDt * 0.5;
    for (int i = 0; i < 2; i++) {
      float f = (float(i) + 0.5) * 0.5;
      vec3 acc = pull(x - mix(uCore[0], uCore[1], f), uGal[0]) + pull(x - mix(uCore[2], uCore[3], f), uGal[1]);
      if (uSettle > 0.0) {
        // Settling. The pull turns into that of the finished galaxy, and
        // friction (as gas has, and stars borrow here) brings the star onto
        // its circular orbit there: radius aHome.x, height aHome.y. The
        // strengths are per second of the clock, whatever the pace of time.
        float r = max(length(x.xz), 1e-5);
        vec2 er = x.xz / r;
        vec2 et = vec2(-er.y, er.x);
        float a = CORE + r;
        acc = mix(acc, vec3(er.x, 0.0, er.y) * (-SPEED * SPEED * r / (a * a)), uSettle);
        // Each star is first aimed a little off its speed, so that clumps
        // smear around the disc instead of surviving as rings.
        float stray = 1.0 + 0.8 * (aLook.z - 0.5) * (1.0 - uSettle);
        float circular = SPEED * aHome.x / (CORE + aHome.x) * stray;
        float ar = -2.5 / (uWarp * uWarp) * (r - aHome.x) - 3.2 / uWarp * dot(v.xz, er);
        float at = -1.5 / uWarp * (dot(v.xz, et) - circular);
        acc.xz += uSettle * (ar * er + at * et);
        acc.y += uSettle * (-2.5 / (uWarp * uWarp) * (x.y - aHome.y) - 3.2 / uWarp * v.y);
      }
      v += acc * h;
      x += v * h;
    }

    // ---- Light it ----
    // While a galaxy is whole its own spiral pattern lights its stars. In
    // the collision each star keeps the light it had, so the arms are drawn
    // out into the tails. As the remnant settles, the pattern of the
    // finished galaxy takes over.
    if (aWave.x > 0.0) {
      if (uLight.x > 0.0) {
        int k = int(aHome.w + 0.5);
        vec3 d = x - uCore[2 * k + 1];
        vec2 q = vec2(dot(d, uE1[k]), dot(d, uE2[k]));
        float own = arms(length(q) / uGal[k].w, atan(q.y, q.x) - (k == 0 ? uPhase.x : uPhase.y), aWave.y, aWave.z);
        lit += (own - lit) * uLight.x;
      }
      if (uLight.y > 0.0) {
        float fin = arms(length(x.xz), atan(x.z, x.x) - uPhase.z, aWave.y, aWave.z);
        lit += (fin - lit) * uLight.y;
      }
    }
  }
  oPos = vec4(x, lit);
  oVel = vec4(v, aVel.w);

  // ---- Draw it ----
  vec3 p = x;
  float glow = mix(1.0, lit, aWave.x);
  if (kind > 0.5) {
    float turn = 0.006 * uClock;
    p = vec3(x.x * cos(turn) - x.z * sin(turn), x.y, x.x * sin(turn) + x.z * cos(turn));
    shown = 0.6;
  }

  // The camera circles the centre (yaw), looks down on the plane (pitch),
  // and stands at a distance from it. Its lens is fixed, so that coming
  // closer both enlarges the scene and deepens the perspective.
  float cy = cos(uView.x), sy = sin(uView.x);
  p = vec3(p.x * cy - p.z * sy, p.y, p.x * sy + p.z * cy);
  float cp = cos(uView.y), sp = sin(uView.y);
  p = vec3(p.x, p.y * cp - p.z * sp, p.y * sp + p.z * cp);
  float cr = cos(uView.z), sr = sin(uView.z);
  p = vec3(p.x * cr - p.y * sr, p.x * sr + p.y * cr, p.z);

  float depth = uView.w + p.z;
  float persp = CAMERA / max(depth, 0.2);
  // A star about to pass the camera fades out instead of filling the view.
  float near = smoothstep(0.2, 0.85, depth);
  vec2 xy = p.xy * persp * uScale;
  xy.x *= uRes.y / uRes.x;
  gl_Position = near > 0.0 ? vec4(uCenter + xy, 0.0, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);

  // The soft sprites of unresolved light are sized for the finished galaxy;
  // on the smaller galaxies they start smaller.
  float sprite = aLook.x > 6.0 && kind < 0.5 ? mix(0.62, 1.0, uSettle) : 1.0;
  float size = min(aLook.x * sprite * persp * uDpr * clamp(uRes.y / (uDpr * 820.0), 0.8, 1.5), 110.0 * uDpr);
  gl_PointSize = max(size, 1.0);

  float twinkle = 1.0 - aWave.w + aWave.w * sin(uClock * (0.6 + aLook.z) + aLook.z * 40.0);
  // Points smaller than a pixel keep their energy by dimming instead.
  float sub = min(size, 1.0);
  vAlpha = 1.7 * aLook.y * glow * twinkle * sub * uFade * shown * near;
  vColor = tint(aLook.w);
}
`;

const FRAG = /* glsl */ `#version 300 es
precision mediump float;
in float vAlpha;
in vec3 vColor;
out vec4 color;

void main() {
  float d = length(gl_PointCoord - 0.5) * 2.0;
  float a = smoothstep(1.0, 0.0, d);
  a *= a * vAlpha;
  color = vec4(vColor * a, a);
}
`;

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) ?? 'shader compile failed');
  }
  return shader;
}

const smooth = (x: number) => {
  const k = Math.min(Math.max(x, 0), 1);
  return k * k * (3 - 2 * k);
};

export function mountMerger(canvas: HTMLCanvasElement, host: HTMLElement): void {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const gl = reduced
    ? null
    : canvas.getContext('webgl2', { alpha: true, antialias: false, premultipliedAlpha: true });
  if (!gl) {
    mountStardust(canvas, host, false);
    return;
  }

  const small = matchMedia('(max-width: 800px)').matches;
  const total = small ? 28000 : 80000;

  const program = gl.createProgram()!;
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAG));
  gl.transformFeedbackVaryings(program, ['oPos', 'oVel'], gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? 'link failed');
  gl.useProgram(program);

  // ---- Start-up: the orbit of the cores and where every star begins ----
  const gal = [galaxy(0), galaxy(1)];
  const { path, settleAt, turns } = coreOrbit(gal[0], gal[1]);
  const steps = path.length / 2;
  // The heavier galaxy stays nearer the common centre, which is where the
  // finished galaxy will sit.
  const cores = (t: number, out: Float32Array, at: number) => {
    const s = Math.min(t / ORBIT_DT, steps - 1);
    const i = Math.min(Math.floor(s), steps - 2);
    const f = s - i;
    const x = path[2 * i] + (path[2 * i + 2] - path[2 * i]) * f;
    const z = path[2 * i + 1] + (path[2 * i + 3] - path[2 * i + 1]) * f;
    out[at] = x * SHARE[1];
    out[at + 1] = 0;
    out[at + 2] = z * SHARE[1];
    out[at + 6] = -x * SHARE[0];
    out[at + 7] = 0;
    out[at + 8] = -z * SHARE[0];
  };

  const { orbit, look, wave, back, dust, count } = buildParticles(total, false);
  const state = new Float32Array(count * 8);
  const home = new Float32Array(count * 4);
  {
    const rand = prng(20261120);
    const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-9))) * Math.cos(2 * Math.PI * rand());
    const d0 = Math.hypot(START.x, START.z);
    const vx = (-START.inward * START.x - START.sideways * START.z) / d0;
    const vz = (-START.inward * START.z + START.sideways * START.x) / d0;
    const centre = [[START.x * SHARE[1], START.z * SHARE[1]], [-START.x * SHARE[0], -START.z * SHARE[0]]];
    const drift = [[vx * SHARE[1], vz * SHARE[1]], [-vx * SHARE[0], -vz * SHARE[0]]];
    for (let i = 0; i < count; i++) {
      const [R, angle, H, kind] = orbit.subarray(i * 4, i * 4 + 4);
      const s = i * 8;
      if (kind === 1) {
        home.set([R, H, 1, 0], i * 4);
        state.set([Math.cos(angle) * R, H, Math.sin(angle) * R], s);
      } else {
        // A star of one of the two galaxies: the same orbit, scaled down,
        // in that galaxy's plane. Its radius is blurred a little against its
        // place in the finished disc so that the merger mixes well, and it
        // has a small random motion of its own, as real disc stars do.
        const k = rand() < SHARE[0] ? 0 : 1;
        const g = gal[k];
        const rho = R * g.size * Math.exp(0.3 * gauss());
        const h = H * g.size;
        const c = Math.cos(angle), sn = Math.sin(angle);
        const far = Math.hypot(rho, h);
        const speed = (Math.sqrt(g.speed2) * far) / (g.core + far);
        const jitter = 0.07 * Math.sqrt(g.speed2);
        for (let a = 0; a < 3; a++) {
          const centreA = a === 0 ? centre[k][0] : a === 2 ? centre[k][1] : 0;
          const driftA = a === 0 ? drift[k][0] : a === 2 ? drift[k][1] : 0;
          state[s + a] = centreA + rho * (c * g.e1[a] + sn * g.e2[a]) + h * g.normal[a];
          state[s + 4 + a] = driftA + speed * (-sn * g.e1[a] + c * g.e2[a]) + jitter * gauss();
        }
        state[s + 3] = 0.35;
        home.set([R, H, 0, k], i * 4);
      }
    }
  }

  // Two copies of the moving state: one is read while the other is written.
  const stateBuf = [gl.createBuffer()!, gl.createBuffer()!];
  for (const b of stateBuf) {
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, state, gl.DYNAMIC_COPY);
  }
  const fixed = ([['aHome', home], ['aLook', look], ['aWave', wave]] as const).map(([name, data]) => {
    const buffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return [name, buffer] as const;
  });
  const vaos = stateBuf.map((b) => {
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    for (const [name, offset] of [['aPos', 0], ['aVel', 16]] as const) {
      const loc = gl.getAttribLocation(program, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 32, offset);
    }
    for (const [name, buffer] of fixed) {
      const loc = gl.getAttribLocation(program, name);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 0, 0);
    }
    return vao;
  });
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  // Reading copy i writes copy 1 - i.
  const feedback = stateBuf.map((_, i) => {
    const tf = gl.createTransformFeedback()!;
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, tf);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, stateBuf[1 - i]);
    return tf;
  });
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);

  const uni = (name: string) => gl.getUniformLocation(program, name);
  const u = {
    dt: uni('uDt'), warp: uni('uWarp'), clock: uni('uClock'), settle: uni('uSettle'),
    light: uni('uLight'), phase: uni('uPhase'), core: uni('uCore'),
    fade: uni('uFade'), dpr: uni('uDpr'), scale: uni('uScale'), res: uni('uRes'),
    center: uni('uCenter'), view: uni('uView'),
  };
  gl.uniform4fv(uni('uGal'), gal.flatMap((g) => [g.speed2, g.core, g.edge, g.size]));
  gl.uniform3fv(uni('uE1'), gal.flatMap((g) => g.e1));
  gl.uniform3fv(uni('uE2'), gal.flatMap((g) => g.e2));

  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.clearColor(0, 0, 0, 0);

  let cap = 2; // most device pixels per CSS pixel; steps down if frames come slowly
  const resize = () => {
    const dpr = Math.min(devicePixelRatio || 1, cap);
    const w = host.clientWidth;
    const h = host.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    const portrait = w / h < 0.9;
    const center = portrait ? [0.1, -0.42] : [0.42, -0.08];
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

  // Draw only while the hero is on screen and the tab is showing; the run
  // waits, it does not skip ahead.
  let onScreen = true;
  let pageShown = !document.hidden;
  const visible = () => onScreen && pageShown;
  let frame = 0;

  let last = 0;
  let clock = 0; // seconds drawn so far
  let tau = 0; // model time
  let settling = -1; // seconds since settling began
  let reading = 0;
  const core = new Float32Array(12);
  const view: View = { yaw: 0, pitch: 0, roll: 0, dist: 0 };

  // The readout: time elapsed, the pace, and where the story stands. The
  // text is rewritten a few times a second, and only when it has changed.
  const timeEl = host.querySelector<HTMLElement>('[data-sim-time]');
  const rateEl = host.querySelector<HTMLElement>('[data-sim-rate]');
  const phaseEl = host.querySelector<HTMLElement>('[data-sim-phase]');
  const [near1 = 0, far1 = 0, near2 = 0] = turns;
  const phase = () => {
    if (settling >= 8) return 'Steady rotation';
    if (settling >= 0) return 'Relaxation';
    if (tau < near1 * 0.72) return 'Approach';
    if (tau < (near1 + far1) / 2) return 'First passage';
    if (tau < (far1 + near2) / 2) return 'Separation';
    if (tau < near2 + (near2 - far1) * 0.4) return 'Second passage';
    return 'Coalescence';
  };
  const shownText = ['', '', ''];
  let sinceReadout = 1;
  const readout = (dtReal: number, warp: number) => {
    sinceReadout += dtReal;
    if (sinceReadout < 0.08) return;
    sinceReadout = 0;
    // Elapsed in units of 100 million years (억 년), the pace in units of
    // 10 thousand years (만 년) a second: the units a Korean reader counts in.
    const text = [((tau * MYR) / 100).toFixed(2), (Math.round(warp * MYR * 10) * 10).toLocaleString('en-US'), phase()];
    [timeEl, rateEl, phaseEl].forEach((el, i) => {
      if (el && shownText[i] !== text[i]) el.textContent = shownText[i] = text[i];
    });
  };
  host.dataset.sim = '';
  // Slow frames: a running mean of the frame time, checked now and then.
  let mean = 16;
  let sinceCheck = 0;

  // The core glow of the page waits for the merger.
  host.dataset.merging = '';

  const draw = (now: number) => {
    frame = 0;
    const gap = last ? now - last : 16.7;
    last = now;
    // A long gap is a pause (hidden tab, scrolled away), not time to make up.
    const dtReal = Math.min(Math.max(gap, 0), 34) / 1000;
    clock += dtReal;

    if (gap > 0 && gap < 200) {
      mean += (gap - mean) * 0.05;
      if (++sinceCheck > 120 && mean > 26 && cap > 1.25) {
        cap = Math.max(1.25, cap - 0.5);
        sinceCheck = 0;
        mean = 16;
        resize();
      }
    }

    // The pace of time: fast through the collision, easing to 1 as it settles.
    if (settling < 0 && tau >= settleAt) {
      settling = 0;
      delete host.dataset.merging;
    }
    if (settling >= 0) settling += dtReal;
    const warp = settling < 0 ? WARP * (0.35 + 0.65 * smooth(clock / 1.5)) : 1 + (WARP - 1) * Math.exp(-settling / WARP_EASE_S);
    const settle = settling < 0 ? 0 : smooth(settling / SETTLE_S);
    const dt = warp * dtReal;

    cores(tau, core, 0);
    cores(tau + dt, core, 3);
    const apart = Math.hypot(core[3] - core[9], core[5] - core[11]);
    tau += dt;

    pointer.x += (pointer.tx - pointer.x) * 0.04;
    pointer.y += (pointer.ty - pointer.y) * 0.04;

    gl.uniform1f(u.dt, dt);
    gl.uniform1f(u.warp, warp);
    gl.uniform1f(u.clock, 40 + clock);
    gl.uniform1f(u.settle, settle);
    gl.uniform2f(
      u.light,
      (1 - Math.exp(-6 * dtReal)) * smooth((apart - 1.3) / 0.6),
      1 - Math.exp(-1.5 * dtReal * settle * settle),
    );
    gl.uniform3f(u.phase, gal[0].pattern * tau, gal[1].pattern * tau, 0.03 * tau);
    gl.uniform3fv(u.core, core);
    gl.uniform1f(u.fade, Math.max(0, 1 - scrollY / (host.clientHeight * 0.9)));
    cameraAt(settling < 0 ? tau / settleAt : 1 + settling / CAMERA_SETTLE_S, view);
    gl.uniform4f(u.view, view.yaw, view.pitch + pointer.y * 0.05, view.roll + pointer.x * 0.035, view.dist);
    readout(dtReal, warp);

    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindVertexArray(vaos[reading]);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, feedback[reading]);
    gl.beginTransformFeedback(gl.POINTS);
    // Light adds up; dust takes light away, from the stars drawn so far and
    // from the page glow behind the canvas. The three draws together write
    // every particle's new state, in order.
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArrays(gl.POINTS, 0, back);
    gl.blendFuncSeparate(gl.ZERO, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.POINTS, back, dust);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArrays(gl.POINTS, back + dust, count - back - dust);
    gl.endTransformFeedback();
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    reading = 1 - reading;

    if (visible()) frame = requestAnimationFrame(draw);
  };
  const kick = () => {
    if (!frame) {
      last = 0;
      frame = requestAnimationFrame(draw);
    }
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

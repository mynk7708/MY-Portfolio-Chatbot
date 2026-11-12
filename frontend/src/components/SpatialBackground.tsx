/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * SpatialBackground.tsx
 *
 * A REAL mouse-reactive 2D WebGL fluid simulation (Navier-Stokes on the GPU),
 * written in raw WebGL. No Three.js, no libraries.
 *
 * Behaviour
 *  - The page starts PURE WHITE. Nothing is drawn until the pointer moves.
 *  - Pointer / touch movement injects velocity + pastel dye ("splats").
 *  - The velocity field curls, advects and projects the dye into smoke / ink /
 *    jelly-like tendrils and vortices.
 *  - When the pointer stops, the fluid keeps moving, then dissipates to white.
 *  - After the dye has fully faded the simulation goes to sleep (zero GPU cost)
 *    and wakes up again on the next movement.
 *
 * Pipeline per frame (ping-pong framebuffers)
 *   splat -> curl -> vorticity -> divergence -> pressure (Jacobi) ->
 *   gradient subtract -> advect velocity -> advect dye -> display on white
 *
 * Initialisation order (so splat() can never touch undefined buffers):
 *   canvas -> WebGL context -> canvas size -> texture formats -> shaders ->
 *   framebuffers -> verify -> ready = true -> listeners -> animation loop
 */
import { useEffect, useRef } from "react";

interface SpatialBackgroundProps {
  className?: string;
}

/* -------------------------------------------------------------------------- */
/*  Tunable configuration                                                     */
/* -------------------------------------------------------------------------- */

const CONFIG = {
  SIM_RESOLUTION: 128,
  DYE_RESOLUTION: 1024,
  DYE_RESOLUTION_MOBILE: 512,
  DENSITY_DISSIPATION: 1.1, // how fast dye fades (1.0 - 1.5 feels right)
  VELOCITY_DISSIPATION: 0.2,
  PRESSURE: 0.8,
  PRESSURE_ITERATIONS: 20,
  CURL: 30, // vorticity confinement -> swirls & vortices
  SPLAT_RADIUS: 0.25,
  SPLAT_FORCE: 6000,
  DYE_INTENSITY: 0.45, // dye amount added per splat
  DENSITY_GAIN: 1.8, // how quickly dye becomes opaque / saturated
  SHADING: 14, // strength of the glassy / jelly shading
  COLOR_SPEED: 3.0, // palette travel per unit of pointer distance
  IDLE_SLEEP_MS: 9000, // stop simulating this long after the last movement
};

/** Soft iridescent pastel palette (cyan -> mint -> blue -> violet -> pink). */
const PALETTE_HEX = [
  "#73E8FF",
  "#79F2C0",
  "#8FA8FF",
  "#B58CFF",
  "#E8A0FF",
  "#FF9DD6",
];

type RGB = [number, number, number];

const hexToRgb = (hex: string): RGB => {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
};

const PALETTE: RGB[] = PALETTE_HEX.map(hexToRgb);

/** Smoothly ping-pongs through the palette so colours never jump abruptly. */
const paletteColor = (phase: number): RGB => {
  const last = PALETTE.length - 1;
  const period = last * 2;
  let p = ((phase % period) + period) % period;
  if (p > last) p = period - p;
  const i = Math.min(Math.floor(p), last - 1);
  let f = p - i;
  f = f * f * (3 - 2 * f);
  const a = PALETTE[i];
  const b = PALETTE[i + 1];
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];
};

/* -------------------------------------------------------------------------- */
/*  GLSL (ES 1.00 - works in both WebGL1 and WebGL2 contexts)                 */
/* -------------------------------------------------------------------------- */

const VERTEX_SHADER = `
  precision highp float;
  attribute vec2 aPosition;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform vec2 texelSize;
  void main () {
    vUv = aPosition * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(aPosition, 0.0, 1.0);
  }
`;

const FRAG_HEADER = `
  #ifdef GL_FRAGMENT_PRECISION_HIGH
    precision highp float;
    precision highp sampler2D;
  #else
    precision mediump float;
    precision mediump sampler2D;
  #endif
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
`;

/** Multiplies a texture by a constant (used to damp the pressure field). */
const CLEAR_SHADER = `
  uniform sampler2D uTexture;
  uniform float value;
  void main () {
    gl_FragColor = value * texture2D(uTexture, vUv);
  }
`;

/** Gaussian splat: adds velocity or dye around a point. */
const SPLAT_SHADER = `
  uniform sampler2D uTarget;
  uniform float aspectRatio;
  uniform vec3 color;
  uniform vec2 point;
  uniform float radius;
  void main () {
    vec2 p = vUv - point.xy;
    p.x *= aspectRatio;
    vec3 splat = exp(-dot(p, p) / radius) * color;
    vec3 base = texture2D(uTarget, vUv).xyz;
    gl_FragColor = vec4(base + splat, 1.0);
  }
`;

/** Semi-Lagrangian advection (with manual bilinear filtering fallback). */
const ADVECTION_SHADER = `
  uniform sampler2D uVelocity;
  uniform sampler2D uSource;
  uniform vec2 texelSize;
  uniform vec2 dyeTexelSize;
  uniform float dt;
  uniform float dissipation;

  vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
    vec2 st = uv / tsize - 0.5;
    vec2 iuv = floor(st);
    vec2 fuv = fract(st);
    vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
    vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
    vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
    vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);
    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
  }

  void main () {
  #ifdef MANUAL_FILTERING
    vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
    vec4 result = bilerp(uSource, coord, dyeTexelSize);
  #else
    vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
    vec4 result = texture2D(uSource, coord);
  #endif
    float decay = 1.0 + dissipation * dt;
    gl_FragColor = result / decay;
  }
`;

const DIVERGENCE_SHADER = `
  uniform sampler2D uVelocity;
  void main () {
    float L = texture2D(uVelocity, vL).x;
    float R = texture2D(uVelocity, vR).x;
    float T = texture2D(uVelocity, vT).y;
    float B = texture2D(uVelocity, vB).y;
    vec2 C = texture2D(uVelocity, vUv).xy;
    if (vL.x < 0.0) { L = -C.x; }
    if (vR.x > 1.0) { R = -C.x; }
    if (vT.y > 1.0) { T = -C.y; }
    if (vB.y < 0.0) { B = -C.y; }
    float div = 0.5 * (R - L + T - B);
    gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
  }
`;

const CURL_SHADER = `
  uniform sampler2D uVelocity;
  void main () {
    float L = texture2D(uVelocity, vL).y;
    float R = texture2D(uVelocity, vR).y;
    float T = texture2D(uVelocity, vT).x;
    float B = texture2D(uVelocity, vB).x;
    float vorticity = R - L - T + B;
    gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
  }
`;

/** Vorticity confinement: pushes velocity along the curl gradient -> swirls. */
const VORTICITY_SHADER = `
  uniform sampler2D uVelocity;
  uniform sampler2D uCurl;
  uniform float curl;
  uniform float dt;
  void main () {
    float L = texture2D(uCurl, vL).x;
    float R = texture2D(uCurl, vR).x;
    float T = texture2D(uCurl, vT).x;
    float B = texture2D(uCurl, vB).x;
    float C = texture2D(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity += force * dt;
    velocity = min(max(velocity, -1000.0), 1000.0);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
  }
`;

/** One Jacobi iteration of the pressure Poisson equation. */
const PRESSURE_SHADER = `
  uniform sampler2D uPressure;
  uniform sampler2D uDivergence;
  void main () {
    float L = texture2D(uPressure, vL).x;
    float R = texture2D(uPressure, vR).x;
    float T = texture2D(uPressure, vT).x;
    float B = texture2D(uPressure, vB).x;
    float divergence = texture2D(uDivergence, vUv).x;
    float pressure = (L + R + B + T - divergence) * 0.25;
    gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
  }
`;

/** Makes the velocity field divergence-free. */
const GRADIENT_SUBTRACT_SHADER = `
  uniform sampler2D uPressure;
  uniform sampler2D uVelocity;
  void main () {
    float L = texture2D(uPressure, vL).x;
    float R = texture2D(uPressure, vR).x;
    float T = texture2D(uPressure, vT).x;
    float B = texture2D(uPressure, vB).x;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity.xy -= vec2(R - L, T - B);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
  }
`;

/**
 * Display: composites the dye over PURE WHITE.
 * The dye texture stores colour * density. We split it into
 *   density  = max channel
 *   hue      = dye / density
 * then fade the hue in over white (so low density = pale / transparent),
 * deepen the colour in dense cores, and add soft glassy shading + glints.
 */
const DISPLAY_SHADER = `
  uniform sampler2D uDye;
  uniform vec2 uStep;
  uniform float uGain;
  uniform float uShading;

  float density (vec3 c) {
    return max(c.r, max(c.g, c.b));
  }

  void main () {
    vec3 A = max(texture2D(uDye, vUv).rgb, vec3(0.0));
    float m = density(A);
    vec3 hue = A / max(m, 0.0001);

    float d = m * uGain;
    float alpha = 1.0 - exp(-d);

    // dense cores become deeper / more saturated, thin edges stay pastel
    float core = smoothstep(0.6, 3.0, d);
    vec3 body = mix(hue, hue * hue, 0.75 * core);

    // height-field style shading from the density gradient
    float hL = density(texture2D(uDye, vUv - vec2(uStep.x, 0.0)).rgb);
    float hR = density(texture2D(uDye, vUv + vec2(uStep.x, 0.0)).rgb);
    float hB = density(texture2D(uDye, vUv - vec2(0.0, uStep.y)).rgb);
    float hT = density(texture2D(uDye, vUv + vec2(0.0, uStep.y)).rgb);
    vec2 grad = vec2(hR - hL, hT - hB) * uShading;

    vec3 n = normalize(vec3(-grad, 1.0));
    vec3 l = normalize(vec3(-0.45, 0.65, 0.62));
    float diff = clamp(dot(n, l), 0.0, 1.0);
    vec3 hv = normalize(l + vec3(0.0, 0.0, 1.0));
    float spec = pow(clamp(dot(n, hv), 0.0, 1.0), 60.0);
    float rim = clamp(length(grad) * 1.2, 0.0, 1.0);

    body *= mix(1.0, 0.88 + 0.26 * diff, alpha);
    body = mix(body, vec3(1.0), spec * 0.6 * alpha);
    body = mix(body, vec3(1.0), rim * 0.28 * alpha);

    vec3 col = mix(vec3(1.0), body, alpha * 0.96);
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
  }
`;

/* -------------------------------------------------------------------------- */
/*  Types                                                                     */
/* -------------------------------------------------------------------------- */

interface ProgramObj {
  program: WebGLProgram;
  uniforms: Record<string, WebGLUniformLocation | null>;
  bind: () => void;
}

interface Programs {
  clear: ProgramObj;
  splat: ProgramObj;
  advection: ProgramObj;
  divergence: ProgramObj;
  curl: ProgramObj;
  vorticity: ProgramObj;
  pressure: ProgramObj;
  gradientSubtract: ProgramObj;
  display: ProgramObj;
}

interface FBO {
  texture: WebGLTexture;
  fbo: WebGLFramebuffer;
  width: number;
  height: number;
  texelSizeX: number;
  texelSizeY: number;
  attach: (unit: number) => number;
}

interface DoubleFBO {
  width: number;
  height: number;
  texelSizeX: number;
  texelSizeY: number;
  read: FBO;
  write: FBO;
  swap: () => void;
}

interface Sim {
  dye: DoubleFBO;
  velocity: DoubleFBO;
  divergence: FBO;
  curl: FBO;
  pressure: DoubleFBO;
}

interface TexFormat {
  internalFormat: number;
  format: number;
}

interface QueuedMove {
  px: number; // previous position (uv)
  py: number;
  x: number; // current position (uv)
  y: number;
  dx: number; // aspect-corrected delta
  dy: number;
}

/* -------------------------------------------------------------------------- */
/*  Component                                                                 */
/* -------------------------------------------------------------------------- */

export default function SpatialBackground({
  className,
}: SpatialBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    /* ---------------------------- 1. canvas ------------------------------ */
    const canvas = canvasRef.current;
    if (!canvas) return;

    /* ------------------------- 2. WebGL context -------------------------- */
    const ctxParams = {
      alpha: false,
      depth: false,
      stencil: false,
      antialias: false,
      preserveDrawingBuffer: false,
    };

    let isWebGL2 = true;
    let ctx = canvas.getContext("webgl2", ctxParams) as
      | WebGLRenderingContext
      | WebGL2RenderingContext
      | null;
    if (!ctx) {
      isWebGL2 = false;
      ctx = (canvas.getContext("webgl", ctxParams) ||
        canvas.getContext("experimental-webgl", ctxParams)) as
        | WebGLRenderingContext
        | null;
    }
    if (!ctx) {
      console.warn("[SpatialBackground] WebGL is not available.");
      return;
    }
    // WebGL1 shares the subset of the API used here; typed as WebGL2 for convenience.
    const gl = ctx as WebGL2RenderingContext;
    const glAny = gl as any;

    /* --------------------------- runtime state --------------------------- */
    const caps = {
      texType: 0,
      linear: false,
      rgba: null as TexFormat | null,
      rg: null as TexFormat | null,
      r: null as TexFormat | null,
    };

    let progs: Programs | null = null;
    let sim: Sim | null = null;
    let quadBuffer: WebGLBuffer | null = null;

    let ready = false;
    let sleeping = true;
    let running = false;
    let rafId = 0;
    let lastTime = 0;
    let lastActivity = 0;
    let colorPhase = Math.random() * 10;

    const pointer = { x: 0, y: 0, has: false };
    const queue: QueuedMove[] = [];

    /* ----------------------------- 3. sizing ----------------------------- */
    const syncSize = (): boolean => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        return true;
      }
      return false;
    };

    const getResolution = (resolution: number) => {
      let aspect = gl.drawingBufferWidth / gl.drawingBufferHeight;
      if (!isFinite(aspect) || aspect <= 0) aspect = 1;
      if (aspect < 1) aspect = 1 / aspect;
      const min = Math.round(resolution);
      const max = Math.round(resolution * aspect);
      if (gl.drawingBufferWidth > gl.drawingBufferHeight) {
        return { width: max, height: min };
      }
      return { width: min, height: max };
    };

    const dyeResolutionSetting = () =>
      window.innerWidth < 768
        ? CONFIG.DYE_RESOLUTION_MOBILE
        : CONFIG.DYE_RESOLUTION;

    /* ----------------------- 4. texture format support ------------------- */
    const supportRenderTextureFormat = (
      internalFormat: number,
      format: number,
      type: number
    ): boolean => {
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        internalFormat,
        4,
        4,
        0,
        format,
        type,
        null
      );
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        texture,
        0
      );
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.deleteFramebuffer(fbo);
      gl.deleteTexture(texture);
      return status === gl.FRAMEBUFFER_COMPLETE;
    };

    const getSupportedFormat = (
      internalFormat: number,
      format: number,
      type: number
    ): TexFormat | null => {
      if (!supportRenderTextureFormat(internalFormat, format, type)) {
        if (!isWebGL2) return null;
        if (internalFormat === gl.R16F) {
          return getSupportedFormat(gl.RG16F, gl.RG, type);
        }
        if (internalFormat === gl.RG16F) {
          return getSupportedFormat(gl.RGBA16F, gl.RGBA, type);
        }
        return null;
      }
      return { internalFormat, format };
    };

    /* ---------------------------- 5. shaders ----------------------------- */
    const compile = (type: number, source: string): WebGLShader | null => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error(
          "[SpatialBackground] shader compile error:",
          gl.getShaderInfoLog(shader)
        );
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const makeProgram = (
      vertex: WebGLShader,
      fragmentBody: string
    ): ProgramObj | null => {
      const defines = caps.linear ? "" : "#define MANUAL_FILTERING\n";
      const fragment = compile(
        gl.FRAGMENT_SHADER,
        defines + FRAG_HEADER + fragmentBody
      );
      if (!fragment) return null;

      const program = gl.createProgram();
      if (!program) return null;
      gl.attachShader(program, vertex);
      gl.attachShader(program, fragment);
      gl.bindAttribLocation(program, 0, "aPosition");
      gl.linkProgram(program);
      gl.deleteShader(fragment);

      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error(
          "[SpatialBackground] program link error:",
          gl.getProgramInfoLog(program)
        );
        gl.deleteProgram(program);
        return null;
      }

      const uniforms: Record<string, WebGLUniformLocation | null> = {};
      const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
      for (let i = 0; i < count; i++) {
        const info = gl.getActiveUniform(program, i);
        if (info) uniforms[info.name] = gl.getUniformLocation(program, info.name);
      }

      return { program, uniforms, bind: () => gl.useProgram(program) };
    };

    const createPrograms = (): Programs | null => {
      const vertex = compile(gl.VERTEX_SHADER, VERTEX_SHADER);
      if (!vertex) return null;

      const clear = makeProgram(vertex, CLEAR_SHADER);
      const splat = makeProgram(vertex, SPLAT_SHADER);
      const advection = makeProgram(vertex, ADVECTION_SHADER);
      const divergence = makeProgram(vertex, DIVERGENCE_SHADER);
      const curl = makeProgram(vertex, CURL_SHADER);
      const vorticity = makeProgram(vertex, VORTICITY_SHADER);
      const pressure = makeProgram(vertex, PRESSURE_SHADER);
      const gradientSubtract = makeProgram(vertex, GRADIENT_SUBTRACT_SHADER);
      const display = makeProgram(vertex, DISPLAY_SHADER);
      gl.deleteShader(vertex);

      if (
        !clear ||
        !splat ||
        !advection ||
        !divergence ||
        !curl ||
        !vorticity ||
        !pressure ||
        !gradientSubtract ||
        !display
      ) {
        return null;
      }
      return {
        clear,
        splat,
        advection,
        divergence,
        curl,
        vorticity,
        pressure,
        gradientSubtract,
        display,
      };
    };

    /* -------------------------- 6. framebuffers -------------------------- */
    const createFBO = (
      w: number,
      h: number,
      fmt: TexFormat,
      filter: number
    ): FBO => {
      gl.activeTexture(gl.TEXTURE0);
      const texture = gl.createTexture() as WebGLTexture;
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        fmt.internalFormat,
        w,
        h,
        0,
        fmt.format,
        caps.texType,
        null
      );

      const fbo = gl.createFramebuffer() as WebGLFramebuffer;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        texture,
        0
      );
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error("Incomplete framebuffer");
      }
      gl.viewport(0, 0, w, h);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);

      return {
        texture,
        fbo,
        width: w,
        height: h,
        texelSizeX: 1 / w,
        texelSizeY: 1 / h,
        attach: (unit: number) => {
          gl.activeTexture(gl.TEXTURE0 + unit);
          gl.bindTexture(gl.TEXTURE_2D, texture);
          return unit;
        },
      };
    };

    const createDoubleFBO = (
      w: number,
      h: number,
      fmt: TexFormat,
      filter: number
    ): DoubleFBO => {
      const d: DoubleFBO = {
        width: w,
        height: h,
        texelSizeX: 1 / w,
        texelSizeY: 1 / h,
        read: createFBO(w, h, fmt, filter),
        write: createFBO(w, h, fmt, filter),
        swap: () => {
          const t = d.read;
          d.read = d.write;
          d.write = t;
        },
      };
      return d;
    };

    const deleteFBO = (f: FBO) => {
      gl.deleteTexture(f.texture);
      gl.deleteFramebuffer(f.fbo);
    };

    const disposeSim = () => {
      const s = sim;
      sim = null;
      if (!s) return;
      deleteFBO(s.dye.read);
      deleteFBO(s.dye.write);
      deleteFBO(s.velocity.read);
      deleteFBO(s.velocity.write);
      deleteFBO(s.pressure.read);
      deleteFBO(s.pressure.write);
      deleteFBO(s.divergence);
      deleteFBO(s.curl);
    };

    const initFramebuffers = (): boolean => {
      const { rgba, rg, r } = caps;
      if (!rgba || !rg || !r) return false;

      const simRes = getResolution(CONFIG.SIM_RESOLUTION);
      const dyeRes = getResolution(dyeResolutionSetting());
      const filtering = caps.linear ? gl.LINEAR : gl.NEAREST;

      gl.disable(gl.BLEND);
      disposeSim();

      const next: Sim = {
        dye: createDoubleFBO(dyeRes.width, dyeRes.height, rgba, filtering),
        velocity: createDoubleFBO(simRes.width, simRes.height, rg, filtering),
        divergence: createFBO(simRes.width, simRes.height, r, gl.NEAREST),
        curl: createFBO(simRes.width, simRes.height, r, gl.NEAREST),
        pressure: createDoubleFBO(simRes.width, simRes.height, r, gl.NEAREST),
      };
      sim = next;
      return true;
    };

    const clearAll = () => {
      const s = sim;
      if (!s) return;
      gl.clearColor(0, 0, 0, 1);
      const targets: FBO[] = [
        s.dye.read,
        s.dye.write,
        s.velocity.read,
        s.velocity.write,
        s.pressure.read,
        s.pressure.write,
      ];
      for (const t of targets) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
        gl.viewport(0, 0, t.width, t.height);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }
    };

    /* ------------------------------ drawing ------------------------------ */
    const blit = (target: FBO | null) => {
      if (target) {
        gl.viewport(0, 0, target.width, target.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      } else {
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      }
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };

    const correctRadius = (radius: number): number => {
      const aspect = canvas.width / canvas.height;
      return aspect > 1 ? radius * aspect : radius;
    };

    /** Injects velocity + dye. Guarded: never runs before buffers exist. */
    const splat = (
      x: number,
      y: number,
      dx: number,
      dy: number,
      color: RGB
    ) => {
      const s = sim;
      const P = progs;
      if (!ready || !s || !P) return;

      const p = P.splat;
      p.bind();
      gl.uniform1f(p.uniforms.aspectRatio, canvas.width / canvas.height);
      gl.uniform2f(p.uniforms.point, x, y);
      gl.uniform1f(p.uniforms.radius, correctRadius(CONFIG.SPLAT_RADIUS / 100));

      // velocity
      gl.uniform1i(p.uniforms.uTarget, s.velocity.read.attach(0));
      gl.uniform3f(p.uniforms.color, dx, dy, 0.0);
      blit(s.velocity.write);
      s.velocity.swap();

      // dye
      gl.uniform1i(p.uniforms.uTarget, s.dye.read.attach(0));
      gl.uniform3f(p.uniforms.color, color[0], color[1], color[2]);
      blit(s.dye.write);
      s.dye.swap();
    };

    /** One fluid step: curl -> vorticity -> divergence -> pressure -> advect. */
    const step = (dt: number) => {
      const s = sim;
      const P = progs;
      if (!ready || !s || !P) return;

      gl.disable(gl.BLEND);

      // curl
      P.curl.bind();
      gl.uniform2f(P.curl.uniforms.texelSize, s.velocity.texelSizeX, s.velocity.texelSizeY);
      gl.uniform1i(P.curl.uniforms.uVelocity, s.velocity.read.attach(0));
      blit(s.curl);

      // vorticity confinement
      P.vorticity.bind();
      gl.uniform2f(P.vorticity.uniforms.texelSize, s.velocity.texelSizeX, s.velocity.texelSizeY);
      gl.uniform1i(P.vorticity.uniforms.uVelocity, s.velocity.read.attach(0));
      gl.uniform1i(P.vorticity.uniforms.uCurl, s.curl.attach(1));
      gl.uniform1f(P.vorticity.uniforms.curl, CONFIG.CURL);
      gl.uniform1f(P.vorticity.uniforms.dt, dt);
      blit(s.velocity.write);
      s.velocity.swap();

      // divergence
      P.divergence.bind();
      gl.uniform2f(P.divergence.uniforms.texelSize, s.velocity.texelSizeX, s.velocity.texelSizeY);
      gl.uniform1i(P.divergence.uniforms.uVelocity, s.velocity.read.attach(0));
      blit(s.divergence);

      // damp previous pressure
      P.clear.bind();
      gl.uniform1i(P.clear.uniforms.uTexture, s.pressure.read.attach(0));
      gl.uniform1f(P.clear.uniforms.value, CONFIG.PRESSURE);
      blit(s.pressure.write);
      s.pressure.swap();

      // pressure solve (Jacobi)
      P.pressure.bind();
      gl.uniform2f(P.pressure.uniforms.texelSize, s.velocity.texelSizeX, s.velocity.texelSizeY);
      gl.uniform1i(P.pressure.uniforms.uDivergence, s.divergence.attach(0));
      for (let i = 0; i < CONFIG.PRESSURE_ITERATIONS; i++) {
        gl.uniform1i(P.pressure.uniforms.uPressure, s.pressure.read.attach(1));
        blit(s.pressure.write);
        s.pressure.swap();
      }

      // gradient subtract
      P.gradientSubtract.bind();
      gl.uniform2f(P.gradientSubtract.uniforms.texelSize, s.velocity.texelSizeX, s.velocity.texelSizeY);
      gl.uniform1i(P.gradientSubtract.uniforms.uPressure, s.pressure.read.attach(0));
      gl.uniform1i(P.gradientSubtract.uniforms.uVelocity, s.velocity.read.attach(1));
      blit(s.velocity.write);
      s.velocity.swap();

      // advect velocity
      P.advection.bind();
      gl.uniform2f(P.advection.uniforms.texelSize, s.velocity.texelSizeX, s.velocity.texelSizeY);
      gl.uniform2f(P.advection.uniforms.dyeTexelSize, s.velocity.texelSizeX, s.velocity.texelSizeY);
      const velocityId = s.velocity.read.attach(0);
      gl.uniform1i(P.advection.uniforms.uVelocity, velocityId);
      gl.uniform1i(P.advection.uniforms.uSource, velocityId);
      gl.uniform1f(P.advection.uniforms.dt, dt);
      gl.uniform1f(P.advection.uniforms.dissipation, CONFIG.VELOCITY_DISSIPATION);
      blit(s.velocity.write);
      s.velocity.swap();

      // advect dye
      gl.uniform2f(P.advection.uniforms.dyeTexelSize, s.dye.texelSizeX, s.dye.texelSizeY);
      gl.uniform1i(P.advection.uniforms.uVelocity, s.velocity.read.attach(0));
      gl.uniform1i(P.advection.uniforms.uSource, s.dye.read.attach(1));
      gl.uniform1f(P.advection.uniforms.dissipation, CONFIG.DENSITY_DISSIPATION);
      blit(s.dye.write);
      s.dye.swap();
    };

    /** Composites the dye over pure white. */
    const render = () => {
      const s = sim;
      const P = progs;
      if (!ready || !s || !P) return;

      gl.disable(gl.BLEND);
      P.display.bind();
      gl.uniform1i(P.display.uniforms.uDye, s.dye.read.attach(0));
      gl.uniform2f(
        P.display.uniforms.uStep,
        s.dye.texelSizeX * 1.5,
        s.dye.texelSizeY * 1.5
      );
      gl.uniform1f(P.display.uniforms.uGain, CONFIG.DENSITY_GAIN);
      gl.uniform1f(P.display.uniforms.uShading, CONFIG.SHADING);
      blit(null);
    };

    /* --------------------------- input -> splats ------------------------- */
    const processQueue = (now: number) => {
      if (queue.length === 0) return;
      const items = queue.splice(0, queue.length);
      lastActivity = now;

      for (const m of items) {
        const dist = Math.hypot(m.dx, m.dy);
        colorPhase += dist * CONFIG.COLOR_SPEED + 0.004;
        const rgb = paletteColor(colorPhase);

        // subdivide long segments so fast strokes stay continuous
        const n = Math.min(10, Math.max(1, Math.ceil(dist / 0.015)));
        const dyeScale = CONFIG.DYE_INTENSITY * Math.pow(n, -0.6);
        const color: RGB = [rgb[0] * dyeScale, rgb[1] * dyeScale, rgb[2] * dyeScale];
        const fx = (m.dx * CONFIG.SPLAT_FORCE) / n;
        const fy = (m.dy * CONFIG.SPLAT_FORCE) / n;

        for (let i = 1; i <= n; i++) {
          const t = i / n;
          splat(m.px + (m.x - m.px) * t, m.py + (m.y - m.py) * t, fx, fy, color);
        }
      }
    };

    const tick = () => {
      if (!ready) {
        running = false;
        return;
      }
      const now = performance.now();
      let dt = (now - lastTime) / 1000;
      lastTime = now;
      dt = Math.min(Math.max(dt, 0.001), 0.016666);

      if (syncSize()) handleBufferResize();
      processQueue(now);
      step(dt);
      render();

      if (now - lastActivity > CONFIG.IDLE_SLEEP_MS) {
        // everything has faded: wipe to guaranteed pure white and sleep
        clearAll();
        render();
        sleeping = true;
        running = false;
        return;
      }
      rafId = requestAnimationFrame(tick);
    };

    const wake = () => {
      const now = performance.now();
      sleeping = false;
      lastActivity = now;
      if (!running && ready) {
        running = true;
        lastTime = now;
        rafId = requestAnimationFrame(tick);
      }
    };

    /** Called whenever the canvas pixel size changed. */
    function handleBufferResize() {
      const target = getResolution(dyeResolutionSetting());
      const s = sim;
      if (!s || s.dye.width !== target.width || s.dye.height !== target.height) {
        try {
          initFramebuffers();
        } catch (err) {
          console.error("[SpatialBackground] resize failed:", err);
          ready = false;
        }
      }
    }

    const pushMove = (clientX: number, clientY: number) => {
      if (!ready) return;
      const w = window.innerWidth || 1;
      const h = window.innerHeight || 1;
      const x = clientX / w;
      const y = 1 - clientY / h;

      if (!pointer.has) {
        pointer.x = x;
        pointer.y = y;
        pointer.has = true;
        return;
      }

      let dx = x - pointer.x;
      let dy = y - pointer.y;
      if (dx === 0 && dy === 0) return;

      const aspect = canvas.width / canvas.height;
      if (aspect < 1) dx *= aspect;
      if (aspect > 1) dy /= aspect;

      queue.push({ px: pointer.x, py: pointer.y, x, y, dx, dy });
      if (queue.length > 90) queue.shift();
      pointer.x = x;
      pointer.y = y;
      wake();
    };

    /* ------------------------------ listeners ---------------------------- */
    const onPointerMove = (e: PointerEvent) => {
      if (e.pointerType === "touch") return; // handled by touch events below
      pushMove(e.clientX, e.clientY);
    };
    const onTouchStart = (e: TouchEvent) => {
      const t = e.touches[0];
      if (!t) return;
      pointer.x = t.clientX / (window.innerWidth || 1);
      pointer.y = 1 - t.clientY / (window.innerHeight || 1);
      pointer.has = true;
    };
    const onTouchMove = (e: TouchEvent) => {
      const t = e.touches[0];
      if (t) pushMove(t.clientX, t.clientY);
    };
    const resetPointer = () => {
      pointer.has = false;
    };
    const onResize = () => {
      if (!ready) return;
      if (syncSize()) handleBufferResize();
      if (sleeping) render();
    };
    const onContextLost = (e: Event) => {
      e.preventDefault();
      ready = false;
      running = false;
      cancelAnimationFrame(rafId);
    };
    const onContextRestored = () => {
      if (initGL()) {
        sleeping = true;
        render();
      }
    };

    /* -------------------- 7. full GL initialisation ---------------------- */
    function initGL(): boolean {
      ready = false;
      try {
        // texture type + extensions
        if (isWebGL2) {
          glAny.getExtension("EXT_color_buffer_float");
          glAny.getExtension("EXT_color_buffer_half_float");
          caps.texType = gl.HALF_FLOAT;
          caps.linear = true; // half-float filtering is core in WebGL2
        } else {
          const hf = glAny.getExtension("OES_texture_half_float");
          if (hf) {
            caps.texType = hf.HALF_FLOAT_OES;
            caps.linear = !!glAny.getExtension("OES_texture_half_float_linear");
            glAny.getExtension("EXT_color_buffer_half_float");
          } else if (glAny.getExtension("OES_texture_float")) {
            glAny.getExtension("WEBGL_color_buffer_float");
            caps.texType = gl.FLOAT;
            caps.linear = !!glAny.getExtension("OES_texture_float_linear");
          } else {
            console.warn("[SpatialBackground] float textures unsupported.");
            return false;
          }
        }

        // renderable formats
        if (isWebGL2) {
          caps.rgba = getSupportedFormat(gl.RGBA16F, gl.RGBA, caps.texType);
          caps.rg = getSupportedFormat(gl.RG16F, gl.RG, caps.texType);
          caps.r = getSupportedFormat(gl.R16F, gl.RED, caps.texType);
        } else {
          const f = getSupportedFormat(gl.RGBA, gl.RGBA, caps.texType);
          caps.rgba = f;
          caps.rg = f;
          caps.r = f;
        }
        if (!caps.rgba || !caps.rg || !caps.r) {
          console.warn("[SpatialBackground] no renderable float format.");
          return false;
        }

        // full-screen quad
        quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
        gl.bufferData(
          gl.ARRAY_BUFFER,
          new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
          gl.STATIC_DRAW
        );
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(0);

        // shaders
        progs = createPrograms();
        if (!progs) return false;

        // framebuffers
        if (!initFramebuffers()) return false;

        // verify everything exists before allowing any splat()
        const s = sim as Sim | null;
        if (!s || !s.dye || !s.velocity || !s.pressure || !s.divergence || !s.curl) {
          return false;
        }

        ready = true;
        return true;
      } catch (err) {
        console.error("[SpatialBackground] init failed:", err);
        ready = false;
        return false;
      }
    }

    const disposeGL = () => {
      disposeSim();
      const P = progs;
      progs = null;
      if (P) {
        (Object.keys(P) as (keyof Programs)[]).forEach((k) =>
          gl.deleteProgram(P[k].program)
        );
      }
      if (quadBuffer) {
        gl.deleteBuffer(quadBuffer);
        quadBuffer = null;
      }
    };

    /* ------------------------------- start ------------------------------- */
    syncSize();
    if (!initGL()) {
      // WebGL fluid unavailable: the page simply stays white.
      disposeGL();
      return;
    }
    render(); // paint white immediately (no black flash)

    canvas.addEventListener("webglcontextlost", onContextLost, false);
    canvas.addEventListener("webglcontextrestored", onContextRestored, false);
    window.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchmove", onTouchMove, { passive: true });
    window.addEventListener("touchend", resetPointer, { passive: true });
    window.addEventListener("touchcancel", resetPointer, { passive: true });
    window.addEventListener("blur", resetPointer);
    window.addEventListener("resize", onResize);
    document.documentElement.addEventListener("mouseleave", resetPointer);

    /* ------------------------------ cleanup ------------------------------ */
    return () => {
      ready = false;
      running = false;
      cancelAnimationFrame(rafId);
      canvas.removeEventListener("webglcontextlost", onContextLost, false);
      canvas.removeEventListener("webglcontextrestored", onContextRestored, false);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", resetPointer);
      window.removeEventListener("touchcancel", resetPointer);
      window.removeEventListener("blur", resetPointer);
      window.removeEventListener("resize", onResize);
      document.documentElement.removeEventListener("mouseleave", resetPointer);
      queue.length = 0;
      // NOTE: the GL context itself is intentionally NOT force-lost, so React
      // StrictMode's mount -> unmount -> mount cycle keeps working.
      disposeGL();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={
        className ? `spatial-background ${className}` : "spatial-background"
      }
      aria-hidden="true"
      style={{
        position: "fixed",
        inset: 0,
        width: "100%",
        height: "100%",
        zIndex: 0,
        pointerEvents: "none",
        display: "block",
        background: "#ffffff",
      }}
    />
  );
}
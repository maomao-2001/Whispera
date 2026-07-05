import * as THREE from "../node_modules/three/build/three.module.js";

const mount = document.getElementById("sceneBackdrop");

const vertexShader = `
uniform float u_time;
uniform float u_audio_intensity;

varying float vDisplacementMix;
varying vec3 vViewNormal;
varying vec3 vViewPosition;

vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
  return mod289(((x * 34.0) + 10.0) * x);
}

vec4 taylorInvSqrt(vec4 r) {
  return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v) {
  const vec2 c = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 d = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i = floor(v + dot(v, c.yyy));
  vec3 x0 = v - i + dot(i, c.xxx);

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + c.xxx;
  vec3 x2 = x0 - i2 + c.yyy;
  vec3 x3 = x0 - d.yyy;

  i = mod289(i);
  vec4 p = permute(
    permute(
      permute(i.z + vec4(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4(0.0, i1.y, i2.y, 1.0)
    )
    + i.x + vec4(0.0, i1.x, i2.x, 1.0)
  );

  float n_ = 1.0 / 7.0;
  vec3 ns = n_ * d.wyz - d.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);

  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;

  return 42.0 * dot(
    m * m,
    vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3))
  );
}

void main() {
  vec3 noiseSample = position * 1.1 + vec3(0.0, 0.0, u_time * 0.35);
  float noise = snoise(noiseSample);
  float amplitude = 0.08 + u_audio_intensity * 0.4;
  float displacement = noise * amplitude;
  vec3 displacedPosition = position + normal * displacement;
  vec3 displacedNormal = normalize(normal + normal * noise * 0.2);
  vec4 modelViewPosition = modelViewMatrix * vec4(displacedPosition, 1.0);

  vDisplacementMix = clamp(noise * 0.5 + 0.5, 0.0, 1.0);
  vViewNormal = normalize(normalMatrix * displacedNormal);
  vViewPosition = modelViewPosition.xyz;

  gl_Position = projectionMatrix * modelViewPosition;
}
`;

const fragmentShader = `
varying float vDisplacementMix;
varying vec3 vViewNormal;
varying vec3 vViewPosition;

void main() {
  vec3 viewNormal = normalize(vViewNormal);
  vec3 viewDirection = normalize(-vViewPosition);

  float fresnel = pow(1.0 - max(dot(viewNormal, viewDirection), 0.0), 2.8);
  vec3 lowColor = vec3(0.03, 0.08, 0.28);
  vec3 highColor = vec3(0.82, 0.42, 1.0);
  vec3 baseColor = mix(lowColor, highColor, vDisplacementMix);

  vec3 coreColor = baseColor * 0.5;
  vec3 glowColor = vec3(0.58, 0.88, 1.0) * fresnel * 1.45;
  vec3 color = coreColor + baseColor * 0.35 + glowColor;

  gl_FragColor = vec4(color, 0.95);
}
`;

let analyserData = null;
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function readAudioIntensity() {
  const analyser = window.__ttsAudioAnalyser;
  if (!analyser) {
    return 0;
  }

  const sampleCount = analyser.fftSize;
  if (!analyserData || analyserData.length !== sampleCount) {
    analyserData = new Uint8Array(sampleCount);
  }

  analyser.getByteTimeDomainData(analyserData);

  let total = 0;
  for (let i = 0; i < analyserData.length; i += 1) {
    total += Math.abs((analyserData[i] - 128) / 128);
  }

  return Math.min(Math.max((total / analyserData.length) * 4.0, 0), 1);
}

if (mount) {
  const maxPixelRatio = prefersReducedMotion.matches ? 1 : 1.35;
  const targetFrameMs = prefersReducedMotion.matches ? 1000 / 20 : 1000 / 30;
  const geometryDetail = prefersReducedMotion.matches ? 24 : 40;
  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.z = 5.6;

  const renderer = new THREE.WebGLRenderer({
    antialias: false,
    alpha: true,
    powerPreference: "low-power"
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, maxPixelRatio));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x000000, 0);
  mount.appendChild(renderer.domElement);

  const geometry = new THREE.IcosahedronGeometry(1, geometryDetail);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      u_time: { value: 0 },
      u_audio_intensity: { value: 0 }
    },
    vertexShader,
    fragmentShader
  });
  const globe = new THREE.Mesh(geometry, material);
  scene.add(globe);

  let animationFrameId = 0;
  let lastFrameTime = 0;
  const clock = new THREE.Clock();

  const renderFrame = (now = 0) => {
    animationFrameId = window.requestAnimationFrame(renderFrame);
    if (document.hidden) {
      return;
    }
    if (lastFrameTime && now - lastFrameTime < targetFrameMs) {
      return;
    }
    const deltaSeconds = lastFrameTime
      ? Math.min((now - lastFrameTime) / 1000, 0.08)
      : targetFrameMs / 1000;
    lastFrameTime = now;
    material.uniforms.u_time.value = clock.getElapsedTime();
    material.uniforms.u_audio_intensity.value = readAudioIntensity();
    globe.rotation.x += deltaSeconds * 0.072;
    globe.rotation.y += deltaSeconds * 0.144;
    renderer.render(scene, camera);
  };

  const handleResize = () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, maxPixelRatio));
  };

  const handleVisibilityChange = () => {
    if (!document.hidden) {
      lastFrameTime = 0;
    }
  };

  const cleanup = () => {
    window.cancelAnimationFrame(animationFrameId);
    window.removeEventListener("resize", handleResize);
    window.removeEventListener("beforeunload", cleanup);
    document.removeEventListener("visibilitychange", handleVisibilityChange);
    geometry.dispose();
    material.dispose();
    renderer.dispose();
  };

  window.addEventListener("resize", handleResize);
  window.addEventListener("beforeunload", cleanup);
  document.addEventListener("visibilitychange", handleVisibilityChange);

  renderFrame();
}

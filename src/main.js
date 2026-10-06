import './style.css';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import Lenis from 'lenis';

gsap.registerPlugin(ScrollTrigger);

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const finePointer = matchMedia('(hover: hover) and (pointer: fine)').matches;

/* ------------------------------------------------------------------ *
 *  Per-section scene state + colour themes (tweened by scroll)
 * ------------------------------------------------------------------ */
// c1/c2/c3 = accent trio, bg = deep background tint
const KEYS = [
  { x: 0.5, s: 1.0, str: 0.45, hue: 0.0, camZ: 6.0, az: 0.0, morph: 0, c1: '#7c5cff', c2: '#00e0ff', c3: '#ff4fd8', bg: '#05050a' },   // hero
  { x: -0.5, s: 0.8, str: 0.8, hue: 0.22, camZ: 6.3, az: -0.28, morph: 0, c1: '#ff7a1a', c2: '#ff2d8a', c3: '#7a3cff', bg: '#0c0508' }, // about
  { x: 0.72, s: 0.66, str: 0.55, hue: 0.48, camZ: 5.8, az: 0.26, morph: 0, c1: '#00f5c4', c2: '#2d6bff', c3: '#c6ff3d', bg: '#030b0c' }, // features
  { x: 0.52, s: 0.9, str: 0.25, hue: 0.7, camZ: 6.2, az: 0.0, morph: 0.6, c1: '#ff3cac', c2: '#8a4bff', c3: '#2bd2ff', bg: '#09041a' },   // process
  { x: 0.55, s: 0.0001, str: 0.4, hue: 0.8, camZ: 6.6, az: -0.18, morph: 0, c1: '#ffb800', c2: '#ff3d5a', c3: '#8a2bff', bg: '#0b0610' }, // showcase
  { x: 0.0, s: 1.0, str: 0.5, hue: 0.92, camZ: 6.0, az: 0.0, morph: 0, c1: '#00e0ff', c2: '#7c5cff', c3: '#ff4fd8', bg: '#05050a' },   // contact
];
const C = (hex) => new THREE.Color(hex);
const state = {
  ...KEYS[0], assemble: 0, scroll: 0, pulse: 0, vel: 0, hueAdd: 0, morphAdd: 0,
  c1: C(KEYS[0].c1), c2: C(KEYS[0].c2), c3: C(KEYS[0].c3), bg: C(KEYS[0].bg),
};
const mouse = { x: 0, y: 0, sx: 0, sy: 0, speed: 0 };

/* ------------------------------------------------------------------ *
 *  WebGL
 * ------------------------------------------------------------------ */
const canvas = $('#gl');
let gl = null;

const NOISE = /* glsl */ `
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+10.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0);const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy));vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz);vec3 l=1.0-g;vec3 i1=min(g.xyz,l.zxy);vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx;vec3 x2=x0-i2+C.yyy;vec3 x3=x0-D.yyy;
  i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857;vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z);vec4 x_=floor(j*ns.z);vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy;vec4 y=y_*ns.x+ns.yyyy;vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy);vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0;vec4 s1=floor(b1)*2.0+1.0;vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x);vec3 p1=vec3(a0.zw,h.y);vec3 p2=vec3(a1.xy,h.z);vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
  vec4 m=max(0.5-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0);m=m*m;
  return 105.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}`;

function initGL() {
  const isMobile = innerWidth < 800;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance', alpha: false });
  renderer.setClearColor(0x05050a, 1);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;
  const pr = Math.min(devicePixelRatio || 1, isMobile ? 1.5 : 1.75);
  renderer.setPixelRatio(pr);
  renderer.setSize(innerWidth, innerHeight, false);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x05050a, 0.024);
  const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 120);
  camera.position.set(0, 0, 6);

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  /* ---- aurora backdrop (fullscreen, colours follow the section theme) ---- */
  const auroraU = {
    uTime: { value: 0 }, uAspect: { value: innerWidth / innerHeight }, uMouse: { value: new THREE.Vector2() },
    uBg: { value: state.bg }, uC1: { value: state.c1 }, uC2: { value: state.c2 }, uC3: { value: state.c3 }, uScroll: { value: 0 },
  };
  const aurora = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.ShaderMaterial({
      uniforms: auroraU, depthTest: false, depthWrite: false, fog: false,
      vertexShader: `varying vec2 vUv; void main(){ vUv = position.xy*0.5+0.5; gl_Position = vec4(position.xy, 1.0, 1.0); }`,
      fragmentShader: /* glsl */ `
        ${NOISE}
        uniform float uTime, uAspect, uScroll; uniform vec2 uMouse; uniform vec3 uBg, uC1, uC2, uC3;
        varying vec2 vUv;
        void main(){
          vec2 uv = (vUv - 0.5) * vec2(uAspect, 1.0);
          float t = uTime * 0.07;
          vec2 m = uMouse * 0.15;
          float n1 = snoise(vec3(uv*0.9 + m + vec2(0.0, uScroll*2.0), t))*0.5+0.5;
          float n2 = snoise(vec3(uv*1.4 - m*1.5 + 4.0, t*1.3 + 10.0))*0.5+0.5;
          float n3 = snoise(vec3(uv*0.6 + 9.0, t*0.8 - 5.0))*0.5+0.5;
          vec3 col = uBg;
          col += uC1 * pow(n1, 2.6) * 0.34;
          col += uC2 * pow(n2, 3.0) * 0.28;
          col += uC3 * pow(n3, 3.4) * 0.30;
          float v = smoothstep(1.25, 0.15, length(uv*vec2(0.85,1.0)));
          col *= mix(0.55, 1.0, v);
          gl_FragColor = vec4(col, 1.0);
        }`,
    }),
  );
  aurora.frustumCulled = false;
  aurora.renderOrder = -10;
  scene.add(aurora);

  /* ---- hero blob ---- */
  const blobU = {
    uTime: { value: 0 }, uStrength: { value: 0.45 }, uFreq: { value: 1.15 }, uMorph: { value: 0 },
    uHue: { value: 0 }, uMouse: { value: new THREE.Vector2() },
    uC1: { value: state.c1 }, uC2: { value: state.c2 }, uC3: { value: state.c3 },
  };
  const blobMat = new THREE.ShaderMaterial({
    uniforms: blobU,
    vertexShader: /* glsl */ `
      ${NOISE}
      uniform float uTime, uStrength, uFreq, uMorph; uniform vec2 uMouse;
      varying vec3 vNormalW; varying vec3 vPosW; varying float vDisp;
      vec3 shape(vec3 p){
        vec3 n = normalize(p);
        vec3 a = pow(abs(n), vec3(8.0));
        vec3 cube = n / pow(a.x + a.y + a.z, 0.125) * length(p) * 0.86;
        return mix(p, cube, uMorph);
      }
      vec3 displace(vec3 p){
        vec3 n = normalize(p);
        float f = snoise(p*uFreq + vec3(0.0,0.0,uTime*0.35))*uStrength*0.5
                + snoise(p*uFreq*2.3 - vec3(uTime*0.5))*uStrength*0.14;
        float bump = pow(max(dot(n, normalize(vec3(uMouse, 1.0))), 0.0), 7.0) * 0.28;
        return shape(p) + n*(f + bump);
      }
      void main(){
        vec3 n = normalize(position);
        vec3 t = abs(n.y) > 0.99 ? vec3(1.0,0.0,0.0) : normalize(cross(n, vec3(0.0,1.0,0.0)));
        vec3 b = cross(n, t);
        float e = 0.012; float r = length(position);
        vec3 d0 = displace(position);
        vec3 d1 = displace(normalize(position + t*e)*r);
        vec3 d2 = displace(normalize(position + b*e)*r);
        vec3 nn = normalize(cross(d1-d0, d2-d0));
        vDisp = length(d0) - r;
        vNormalW = normalize(mat3(modelMatrix) * nn);
        vec4 w = modelMatrix * vec4(d0, 1.0);
        vPosW = w.xyz;
        gl_Position = projectionMatrix * viewMatrix * w;
      }`,
    fragmentShader: /* glsl */ `
      uniform float uTime, uHue; uniform vec3 uC1, uC2, uC3;
      varying vec3 vNormalW; varying vec3 vPosW; varying float vDisp;
      vec3 pal(float t){ return 0.5 + 0.5*cos(6.28318*(vec3(1.0,0.7,0.4)*t + vec3(0.0,0.15,0.2))); }
      void main(){
        vec3 N = normalize(vNormalW);
        vec3 V = normalize(cameraPosition - vPosW);
        float fres = pow(1.0 - max(dot(N,V),0.0), 2.4);
        float t = dot(N, vec3(0.35,0.55,0.75))*0.5 + 0.5;
        float k = clamp(t + vDisp*1.6 + sin(uTime*0.4)*0.12, 0.0, 1.0);
        vec3 grad = mix(mix(uC1, uC2, smoothstep(0.15,0.65,k)), uC3, smoothstep(0.55,1.0,k));
        vec3 iri = mix(grad, pal(t*0.8 + uHue + uTime*0.04 + fres*0.4), 0.28);
        vec3 L = normalize(vec3(-0.5,0.8,0.9));
        float diff = max(dot(N,L),0.0);
        float spec = pow(max(dot(reflect(-L,N),V),0.0), 48.0);
        vec3 base = vec3(0.02,0.02,0.06) + iri*(0.16 + diff*0.42);
        vec3 col = mix(base, iri*0.95, smoothstep(0.0,1.0,fres+0.08));
        col += spec*0.55 + fres*0.28*uC2;
        gl_FragColor = vec4(col, 1.0);
      }`,
  });
  const blob = new THREE.Mesh(new THREE.IcosahedronGeometry(1.4, isMobile ? 48 : 72), blobMat);
  const blobGroup = new THREE.Group();
  blobGroup.add(blob);
  scene.add(blobGroup);

  /* ---- orbit group: glowing rings + iridescent floating shapes ---- */
  const orbit = new THREE.Group();
  scene.add(orbit);
  const ringMats = [];
  [[2.25, 0.5, 0.3], [2.75, -0.9, 0.2], [3.4, 1.3, -0.4]].forEach(([r, rx, rz], i) => {
    const m = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 - i * 0.12, blending: THREE.AdditiveBlending, depthWrite: false });
    const ring = new THREE.Mesh(new THREE.TorusGeometry(r, 0.007 + i * 0.002, 8, 220), m);
    ring.rotation.set(rx, 0, rz);
    ring.userData = { sx: 0.12 + i * 0.07, sy: 0.18 - i * 0.06 };
    orbit.add(ring);
    ringMats.push(m);
  });
  const satGeos = [
    new THREE.IcosahedronGeometry(0.24, 0), new THREE.OctahedronGeometry(0.28), new THREE.TorusKnotGeometry(0.17, 0.055, 80, 10),
    new THREE.DodecahedronGeometry(0.23), new THREE.TorusGeometry(0.21, 0.075, 14, 36), new THREE.TetrahedronGeometry(0.3),
  ];
  const SAT = isMobile ? 9 : 18;
  const sats = [];
  for (let i = 0; i < SAT; i++) {
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0xffffff, metalness: 0.45, roughness: 0.14, iridescence: 1, iridescenceIOR: 1.7,
      iridescenceThicknessRange: [180, 720], clearcoat: 1, clearcoatRoughness: 0.1, envMapIntensity: 1.1,
    });
    const mesh = new THREE.Mesh(satGeos[i % satGeos.length], mat);
    const d = {
      mesh, mat, r: 2.6 + Math.random() * 3.2, speed: (0.12 + Math.random() * 0.22) * (Math.random() < 0.5 ? -1 : 1),
      phase: Math.random() * 6.28, y: (Math.random() - 0.5) * 2.4, depth: 0.3 + Math.random(), k: Math.random(),
      spin: new THREE.Vector3(Math.random(), Math.random(), Math.random()).multiplyScalar(0.8), size: 0.55 + Math.random() * 0.7,
    };
    sats.push(d);
    orbit.add(mesh);
  }

  /* ---- star dust ---- */
  const PCOUNT = isMobile ? 1800 : 4200;
  const pPos = new Float32Array(PCOUNT * 3), pRand = new Float32Array(PCOUNT);
  for (let i = 0; i < PCOUNT; i++) {
    const r = 3 + Math.pow(Math.random(), 0.7) * 22;
    const th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
    pPos[i * 3] = r * Math.sin(ph) * Math.cos(th);
    pPos[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th) * 0.8;
    pPos[i * 3 + 2] = r * Math.cos(ph);
    pRand[i] = Math.random();
  }
  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
  pGeo.setAttribute('aRand', new THREE.BufferAttribute(pRand, 1));
  const pU = { uTime: { value: 0 }, uPR: { value: pr }, uC1: { value: state.c1 }, uC2: { value: state.c2 }, uC3: { value: state.c3 } };
  const points = new THREE.Points(pGeo, new THREE.ShaderMaterial({
    uniforms: pU, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      uniform float uTime, uPR; attribute float aRand; varying float vR; varying float vTw;
      void main(){
        vR = aRand;
        vTw = 0.55 + 0.45*sin(uTime*(0.6+aRand*1.6) + aRand*40.0);
        vec4 mv = modelViewMatrix * vec4(position,1.0);
        gl_PointSize = (1.2 + aRand*3.4) * uPR * (22.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uC1, uC2, uC3; varying float vR; varying float vTw;
      void main(){
        float d = length(gl_PointCoord - 0.5);
        float a = smoothstep(0.5, 0.0, d); a *= a;
        vec3 c = vR < 0.33 ? uC1 : (vR < 0.66 ? uC2 : uC3);
        c = mix(c, vec3(1.0), 0.35);
        gl_FragColor = vec4(c, a * vTw * 0.95);
      }`,
  }));
  scene.add(points);

  /* ---- warp streaks: appear & stretch with scroll velocity ---- */
  const SN = isMobile ? 260 : 620;
  const sPos = new Float32Array(SN * 6), sFlag = new Float32Array(SN * 2), sR = new Float32Array(SN * 2);
  for (let i = 0; i < SN; i++) {
    let x, y;
    do { x = (Math.random() - 0.5) * 36; y = (Math.random() - 0.5) * 22; } while (Math.hypot(x, y) < 3.2);
    const z = Math.random() * 50 - 40, rr = Math.random();
    sPos.set([x, y, z, x, y, z], i * 6);
    sFlag[i * 2] = 0; sFlag[i * 2 + 1] = 1; sR[i * 2] = sR[i * 2 + 1] = rr;
  }
  const sGeo = new THREE.BufferGeometry();
  sGeo.setAttribute('position', new THREE.BufferAttribute(sPos, 3));
  sGeo.setAttribute('aFlag', new THREE.BufferAttribute(sFlag, 1));
  sGeo.setAttribute('aRand', new THREE.BufferAttribute(sR, 1));
  const sU = { uTravel: { value: 0 }, uLen: { value: 0 }, uAlpha: { value: 0 }, uC1: { value: state.c1 }, uC2: { value: state.c2 }, uC3: { value: state.c3 } };
  const streaks = new THREE.LineSegments(sGeo, new THREE.ShaderMaterial({
    uniforms: sU, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    vertexShader: /* glsl */ `
      uniform float uTravel, uLen; attribute float aFlag, aRand; varying float vF; varying float vR;
      void main(){
        vec3 p = position;
        p.z = mod(p.z + uTravel + 40.0, 50.0) - 40.0;
        p.z -= aFlag * uLen;
        vF = aFlag; vR = aRand;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      uniform float uAlpha; uniform vec3 uC1, uC2, uC3; varying float vF; varying float vR;
      void main(){
        vec3 c = vR < 0.33 ? uC1 : (vR < 0.66 ? uC2 : uC3);
        gl_FragColor = vec4(mix(c, vec3(1.0), 0.4), uAlpha * (1.0 - vF*0.85));
      }`,
  }));
  streaks.frustumCulled = false;
  scene.add(streaks);

  /* ---- shard swarm -> torus knot ---- */
  const N = isMobile ? 260 : 460;
  const shards = new THREE.InstancedMesh(
    new THREE.BoxGeometry(0.16, 0.16, 0.16),
    new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.9, roughness: 0.3, envMapIntensity: 0.55 }), N);
  shards.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  const target = new Float32Array(N * 3), scatter = new Float32Array(N * 3), delay = new Float32Array(N), seed = new Float32Array(N * 3);
  const tmpC = new THREE.Color();
  for (let i = 0; i < N; i++) {
    const u = (i / N) * Math.PI * 2, k = 0.62, j = 0.16;
    target[i * 3] = (2 + Math.cos(3 * u)) * Math.cos(2 * u) * k + (Math.random() - 0.5) * j;
    target[i * 3 + 1] = (2 + Math.cos(3 * u)) * Math.sin(2 * u) * k + (Math.random() - 0.5) * j;
    target[i * 3 + 2] = Math.sin(3 * u) * k + (Math.random() - 0.5) * j;
    const r = 5 + Math.random() * 6, th = Math.random() * 6.283, ph = Math.acos(2 * Math.random() - 1);
    scatter.set([r * Math.sin(ph) * Math.cos(th), r * Math.sin(ph) * Math.sin(th), r * Math.cos(ph)], i * 3);
    delay[i] = Math.random() * 0.45;
    seed.set([Math.random() * 6.28, Math.random() * 6.28, 0.6 + Math.random() * 0.9], i * 3);
    tmpC.setHSL((i / N) * 1.0 + (Math.random() - 0.5) * 0.04, 0.9, 0.52);
    shards.setColorAt(i, tmpC);
  }
  shards.instanceColor.needsUpdate = true;
  shards.frustumCulled = false;
  const shardGroup = new THREE.Group();
  shardGroup.add(shards);
  shardGroup.visible = false;
  scene.add(shardGroup);

  const l1 = new THREE.PointLight(0xff4fd8, 14, 20, 2);
  const l2 = new THREE.PointLight(0x00e0ff, 14, 20, 2);
  const l3 = new THREE.PointLight(0xffffff, 9, 14, 2); // follows the cursor
  scene.add(l1, l2, l3, new THREE.AmbientLight(0xffffff, 0.2));

  /* ---- post: bloom + velocity chromatic aberration + vignette ---- */
  const composer = new EffectComposer(renderer);
  composer.setPixelRatio(pr);
  composer.setSize(innerWidth, innerHeight);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), isMobile ? 0.35 : 0.5, 0.6, 0.78);
  composer.addPass(bloom);
  const caPass = new ShaderPass({
    uniforms: { tDiffuse: { value: null }, uAmount: { value: 0.002 } },
    vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D tDiffuse; uniform float uAmount; varying vec2 vUv;
      void main(){
        vec2 d = vUv - 0.5; float r = dot(d,d);
        vec2 off = d * (0.4 + r*3.0) * uAmount * 6.0;
        vec3 c = vec3(texture2D(tDiffuse, vUv + off).r, texture2D(tDiffuse, vUv).g, texture2D(tDiffuse, vUv - off).b);
        c *= 1.0 - r*0.55;
        gl_FragColor = vec4(c, 1.0);
      }`,
  });
  composer.addPass(caPass);
  composer.addPass(new OutputPass());

  const dummy = new THREE.Object3D();
  const resize = () => {
    const w = innerWidth, h = innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    bloom.resolution.set(w, h);
    auroraU.uAspect.value = w / h;
  };
  addEventListener('resize', resize);

  let last = performance.now(), t = 0, rotY = 0, travel = 0;
  const smooth = (x) => x * x * (3 - 2 * x);
  const clamp01 = (x) => Math.min(1, Math.max(0, x));
  const tmpV = new THREE.Vector3();

  function render() {
    const now = performance.now();
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now; t += dt;
    const narrow = camera.aspect < 1;
    const k3 = Math.min(1, dt * 3), k5 = Math.min(1, dt * 5);

    mouse.sx += (mouse.x - mouse.sx) * Math.min(1, dt * 4);
    mouse.sy += (mouse.y - mouse.sy) * Math.min(1, dt * 4);
    mouse.speed *= 0.92;

    // camera orbits the scene as you scroll
    const camZ = state.camZ + (narrow ? 2.4 : 0);
    const az = state.az + mouse.sx * 0.08;
    const camX = Math.sin(az) * camZ + mouse.sx * 0.35;
    camera.position.x += (camX - camera.position.x) * k3;
    camera.position.y += (mouse.sy * 0.3 - camera.position.y) * k3;
    camera.position.z += (Math.cos(az) * camZ - camera.position.z) * k3;
    camera.lookAt(0, 0, 0);

    const vh = 2 * camZ * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const halfW = (vh * camera.aspect) / 2;
    const px = narrow ? 0 : state.x * halfW;
    const sc = state.s * (narrow ? 0.7 : 1);

    // theme colours -> scene
    renderer.setClearColor(state.bg, 1);
    scene.fog.color.copy(state.bg);
    auroraU.uTime.value = t; auroraU.uScroll.value = state.scroll; auroraU.uMouse.value.set(mouse.sx, mouse.sy);

    // blob
    state.pulse *= 0.94;
    blobU.uTime.value = t;
    blobU.uStrength.value = state.str + mouse.speed * 0.25 + state.pulse * 0.7 + Math.min(0.25, Math.abs(state.vel) * 0.01);
    blobU.uHue.value = state.hue + state.hueAdd;
    blobU.uMorph.value = clamp01(state.morph + state.morphAdd);
    blobU.uMouse.value.set(mouse.sx, mouse.sy);
    blob.visible = sc > 0.02;
    blobGroup.position.x += (px - blobGroup.position.x) * k5;
    blobGroup.position.y = Math.sin(t * 0.6) * 0.08;
    blobGroup.scale.setScalar(sc * (1 + state.pulse * 0.08));
    rotY += dt * 0.18;
    blobGroup.rotation.y = rotY + mouse.sx * 0.5 + state.scroll * 3.2;
    blobGroup.rotation.x = Math.sin(t * 0.3) * 0.15 - mouse.sy * 0.3 + state.scroll * 1.2;

    // orbiting rings + satellites follow the blob; fade away during the shard section
    const vis = clamp01(Math.max(sc, 0.0) * 1.4) * (1 - clamp01(state.assemble * 2));
    orbit.visible = vis > 0.01;
    if (orbit.visible) {
      orbit.position.x += (px - orbit.position.x) * k5;
      orbit.rotation.y = mouse.sx * 0.35 + state.scroll * 1.5;
      orbit.rotation.x = -mouse.sy * 0.2;
      orbit.scale.setScalar(narrow ? 0.7 : 1);
      ringMats.forEach((m, i) => { m.color.copy(i % 2 ? state.c2 : state.c1); m.opacity = (0.55 - i * 0.12) * vis; });
      orbit.children.forEach((c) => { if (c.userData.sx) { c.rotation.x += dt * c.userData.sx; c.rotation.y += dt * c.userData.sy; } });
      sats.forEach((d) => {
        const a = d.phase + t * d.speed + state.scroll * 5 * Math.sign(d.speed);
        d.mesh.position.set(Math.cos(a) * d.r, d.y + Math.sin(a * 1.7 + d.phase) * 0.6, Math.sin(a) * d.r * 0.4);
        d.mesh.rotation.x += dt * d.spin.x; d.mesh.rotation.y += dt * d.spin.y; d.mesh.rotation.z += dt * d.spin.z;
        d.mesh.scale.setScalar(d.size * vis * (1 + Math.sin(t * 2 + d.phase) * 0.06));
        d.mat.color.copy(state.c1).lerp(state.c3, d.k).lerp(state.c2, 0.25 + 0.25 * Math.sin(t * 0.5 + d.phase));
      });
    }

    // dust
    pU.uTime.value = t;
    points.rotation.y = t * 0.02 + state.scroll * 2.5;
    points.rotation.x = state.scroll * 0.8;
    points.position.y = state.scroll * 3;

    // warp streaks
    const v = Math.abs(state.vel);
    travel += dt * (2 + v * 1.4);
    sU.uTravel.value = travel;
    sU.uLen.value = Math.min(7, v * 0.3);
    sU.uAlpha.value = clamp01(v * 0.02) * 0.45;
    streaks.visible = sU.uAlpha.value > 0.01;

    caPass.uniforms.uAmount.value = 0.0016 + Math.min(0.005, v * 0.0002);

    // shards
    const a = state.assemble;
    shardGroup.visible = a > 0.003;
    if (shardGroup.visible) {
      shardGroup.position.x += (px - shardGroup.position.x) * k5;
      shardGroup.rotation.y += (mouse.sx * 1.2 + t * 0.15 - shardGroup.rotation.y) * Math.min(1, dt * 1.5);
      shardGroup.rotation.x += (-mouse.sy * 0.6 + 0.3 - shardGroup.rotation.x) * Math.min(1, dt * 2);
      shardGroup.scale.setScalar(narrow ? 0.75 : 1);
      for (let i = 0; i < N; i++) {
        const local = clamp01((a * 1.5 - delay[i]) / 1.0);
        const e = 1 - Math.pow(1 - local, 3);
        const wob = Math.sin(t * seed[i * 3 + 2] + seed[i * 3]) * 0.035 * e;
        dummy.position.set(
          scatter[i * 3] + (target[i * 3] - scatter[i * 3]) * e + wob,
          scatter[i * 3 + 1] + (target[i * 3 + 1] - scatter[i * 3 + 1]) * e + wob,
          scatter[i * 3 + 2] + (target[i * 3 + 2] - scatter[i * 3 + 2]) * e - wob);
        dummy.rotation.set(seed[i * 3] + t * 0.6 * (1 - e) + e * 0.4, seed[i * 3 + 1] + t * 0.5 * (1 - e), 0);
        dummy.scale.setScalar(smooth(local) * (0.7 + seed[i * 3 + 2] * 0.5));
        dummy.updateMatrix();
        shards.setMatrixAt(i, dummy.matrix);
      }
      shards.instanceMatrix.needsUpdate = true;
    }

    // coloured lights (theme) + cursor light
    l1.color.copy(state.c3); l2.color.copy(state.c1);
    l1.position.set(Math.cos(t * 0.7) * 4, Math.sin(t * 0.5) * 3, 3).add(shardGroup.position);
    l2.position.set(Math.cos(t * 0.6 + 3) * 4, Math.sin(t * 0.4 + 2) * 3, 3).add(shardGroup.position);
    tmpV.set(mouse.sx * halfW, mouse.sy * vh * 0.5, 2.6);
    l3.position.lerp(tmpV, k5);
    l3.color.copy(state.c2).lerp(new THREE.Color(1, 1, 1), 0.4);

    composer.render();
  }

  return { render };
}

try {
  gl = initGL();
} catch (err) {
  console.warn('WebGL unavailable, falling back to static background.', err);
  canvas.style.display = 'none';
  document.body.style.background = 'radial-gradient(circle at 70% 30%, #2a1a66 0%, #05050a 55%)';
}

/* ------------------------------------------------------------------ *
 *  Smooth scroll + master ticker
 * ------------------------------------------------------------------ */
const progressBar = $('#progress i');
const lenis = new Lenis({ lerp: reduceMotion ? 1 : 0.085, wheelMultiplier: 0.9 });
lenis.stop();
lenis.on('scroll', (e) => {
  ScrollTrigger.update();
  const max = document.documentElement.scrollHeight - innerHeight;
  state.scroll = max > 0 ? Math.min(1, Math.max(0, e.scroll / max)) : 0;
  progressBar.style.transform = `scaleY(${state.scroll})`;
});

// marquee driven by scroll velocity
const marqueeTrack = $('.marquee__track');
let marqueeX = 0, marqueeDir = -1;
gsap.ticker.add((time) => {
  lenis.raf(time * 1000);
  state.vel += (lenis.velocity - state.vel) * 0.12;
  if (Math.abs(state.vel) > 0.5) marqueeDir = state.vel > 0 ? -1 : 1;
  const half = marqueeTrack.scrollWidth / 2;
  marqueeX += marqueeDir * (reduceMotion ? 0 : 1.2 + Math.abs(state.vel) * 0.9);
  if (marqueeX <= -half) marqueeX += half;
  if (marqueeX > 0) marqueeX -= half;
  marqueeTrack.style.transform = `translate3d(${marqueeX}px,0,0)`;
  if (gl) gl.render();
});
gsap.ticker.lagSmoothing(0);

$$('a[href^="#"]').forEach((a) =>
  a.addEventListener('click', (e) => {
    const id = a.getAttribute('href');
    const el = id === '#top' ? 0 : $(id);
    if (el === null) return;
    e.preventDefault();
    lenis.scrollTo(el, { offset: 0, duration: 1.8 });
  }),
);

/* ------------------------------------------------------------------ *
 *  Pointer interactions
 * ------------------------------------------------------------------ */
addEventListener('pointermove', (e) => {
  const nx = (e.clientX / innerWidth) * 2 - 1;
  const ny = -((e.clientY / innerHeight) * 2 - 1);
  mouse.speed = Math.min(1, mouse.speed + Math.hypot(nx - mouse.x, ny - mouse.y) * 2);
  mouse.x = nx; mouse.y = ny;
});
addEventListener('pointerdown', () => { state.pulse = 1; });

if (finePointer) {
  const cur = $('#cursor');
  const ring = $('b', cur);
  const qx = gsap.quickTo(cur, 'x', { duration: 0.12, ease: 'power3' });
  const qy = gsap.quickTo(cur, 'y', { duration: 0.12, ease: 'power3' });
  let lx = 0, ly = 0, cx = 0, cy = 0;
  addEventListener('pointermove', (e) => { cx = e.clientX; cy = e.clientY; qx(cx); qy(cy); });
  gsap.ticker.add(() => {
    lx += (cx - lx) * 0.16; ly += (cy - ly) * 0.16;
    gsap.set(ring, { x: lx - cx, y: ly - cy });
  });
  $$('[data-hover]').forEach((el) => {
    el.addEventListener('pointerenter', () => cur.classList.add('is-hover'));
    el.addEventListener('pointerleave', () => cur.classList.remove('is-hover'));
  });
  $$('[data-magnetic]').forEach((el) => {
    const mx = gsap.quickTo(el, 'x', { duration: 0.5, ease: 'power3' });
    const my = gsap.quickTo(el, 'y', { duration: 0.5, ease: 'power3' });
    el.addEventListener('pointermove', (e) => {
      const r = el.getBoundingClientRect();
      mx((e.clientX - (r.left + r.width / 2)) * 0.35);
      my((e.clientY - (r.top + r.height / 2)) * 0.5);
    });
    el.addEventListener('pointerleave', () => { mx(0); my(0); });
  });
  $$('[data-tilt]').forEach((el) => {
    const rx = gsap.quickTo(el, 'rotationX', { duration: 0.5, ease: 'power3' });
    const ry = gsap.quickTo(el, 'rotationY', { duration: 0.5, ease: 'power3' });
    el.addEventListener('pointermove', (e) => {
      const r = el.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
      ry((px - 0.5) * 16); rx(-(py - 0.5) * 16);
      el.style.setProperty('--mx', `${px * 100}%`);
      el.style.setProperty('--my', `${py * 100}%`);
    });
    el.addEventListener('pointerleave', () => { rx(0); ry(0); });
  });
}

/* ------------------------------------------------------------------ *
 *  Text prep
 * ------------------------------------------------------------------ */
$$('#heroTitle .line').forEach((line) => {
  const walk = (node) => {
    [...node.childNodes].forEach((n) => {
      if (n.nodeType === 3) {
        const frag = document.createDocumentFragment();
        [...n.textContent].forEach((ch) => {
          const s = document.createElement('span');
          s.className = 'char';
          s.innerHTML = ch === ' ' ? '&nbsp;' : ch;
          frag.appendChild(s);
        });
        n.replaceWith(frag);
      } else if (n.nodeType === 1) walk(n);
    });
  };
  walk(line);
});
const chars = $$('#heroTitle .char');
gsap.set(chars, { yPercent: 115, rotate: 8 });
gsap.set('.reveal', { opacity: 0, y: 30 });

const manifesto = $('#manifesto');
manifesto.innerHTML = manifesto.textContent.trim().split(/\s+/).map((w) => `<span class="w">${w}</span>`).join(' ');

/* ------------------------------------------------------------------ *
 *  Scroll choreography (created in document order)
 * ------------------------------------------------------------------ */
const root = document.documentElement;
const rgb = (c) => `rgb(${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)})`;
const hexRgb = (h) => rgb(C(h));

function themeTween(i, trigger, start, end) {
  const from = KEYS[i - 1], to = KEYS[i];
  const st = { trigger, start, end, scrub: 1 };
  gsap.fromTo(state, { x: from.x, s: from.s, str: from.str, hue: from.hue, camZ: from.camZ, az: from.az, morph: from.morph },
    { x: to.x, s: to.s, str: to.str, hue: to.hue, camZ: to.camZ, az: to.az, morph: to.morph, ease: 'none', immediateRender: false, scrollTrigger: st });
  ['c1', 'c2', 'c3', 'bg'].forEach((k) => {
    const f = C(from[k]), t = C(to[k]);
    gsap.fromTo(state[k], { r: f.r, g: f.g, b: f.b }, { r: t.r, g: t.g, b: t.b, ease: 'none', immediateRender: false, scrollTrigger: st });
  });
  gsap.fromTo(root, { '--a': hexRgb(from.c1), '--b': hexRgb(from.c2), '--c': hexRgb(from.c3) },
    { '--a': hexRgb(to.c1), '--b': hexRgb(to.c2), '--c': hexRgb(to.c3), ease: 'none', immediateRender: false, scrollTrigger: st });
}

function buildScroll() {
  const S = (sel, from = 'top 85%', to = 'top 25%') => [sel, from, to];

  themeTween(1, ...S('#about'));

  gsap.to($$('.w', manifesto), {
    opacity: 1, ease: 'none', stagger: 0.12,
    scrollTrigger: { trigger: manifesto, start: 'top 78%', end: 'bottom 50%', scrub: true },
  });

  themeTween(2, ...S('#features'));

  gsap.from('.card', { y: 90, opacity: 0, rotationX: -25, duration: 1.1, ease: 'power3.out', stagger: 0.14, scrollTrigger: { trigger: '.cards', start: 'top 82%' } });
  gsap.from('.features__head > *', { y: 40, opacity: 0, duration: 1, ease: 'power3.out', stagger: 0.12, scrollTrigger: { trigger: '.features__head', start: 'top 85%' } });

  // process: theme in, then pinned horizontal scroll
  themeTween(3, ...S('#process', 'top 85%', 'top 15%'));
  const track = $('#track');
  const dots = $$('#dots i');
  const dist = () => track.scrollWidth - innerWidth;
  gsap.to(track, {
    x: () => -dist(), ease: 'none',
    scrollTrigger: {
      trigger: '#process', pin: true, scrub: 1, start: 'top top', end: () => '+=' + dist() * 1.1, invalidateOnRefresh: true,
      onUpdate: (self) => {
        const p = self.progress;
        state.morphAdd = (0.5 - 0.5 * Math.cos(p * Math.PI * 3)) * 0.5;
        state.hueAdd = p * 0.5;
        const idx = Math.min(dots.length - 1, Math.round(p * (dots.length - 1)));
        dots.forEach((d, i) => d.classList.toggle('on', i === idx));
      },
      onLeave: () => { state.morphAdd = 0; state.hueAdd = 0; },
      onLeaveBack: () => { state.morphAdd = 0; state.hueAdd = 0; },
    },
  });
  dots[0].classList.add('on');

  themeTween(4, ...S('#showcase', 'top 90%', 'top 40%'));
  gsap.fromTo(state, { assemble: 0 }, { assemble: 1, ease: 'none', immediateRender: false, scrollTrigger: { trigger: '#showcase', start: 'top 65%', end: 'center 40%', scrub: 1 } });
  gsap.fromTo(state, { assemble: 1 }, { assemble: 0, ease: 'none', immediateRender: false, scrollTrigger: { trigger: '.marquee', start: 'top 95%', end: 'top 25%', scrub: 1 } });

  gsap.from('.showcase__inner > *', { y: 40, opacity: 0, duration: 1, ease: 'power3.out', stagger: 0.1, scrollTrigger: { trigger: '.showcase__inner', start: 'top 80%' } });
  $$('[data-count]').forEach((el) => {
    const o = { v: 0 };
    ScrollTrigger.create({ trigger: el, start: 'top 90%', once: true, onEnter: () => gsap.to(o, { v: +el.dataset.count, duration: 2, ease: 'power2.out', onUpdate: () => (el.textContent = Math.round(o.v)) }) });
  });

  themeTween(5, ...S('#contact', 'top 95%', 'top 30%'));
  gsap.from('.contact__title', { y: 60, opacity: 0, duration: 1.1, ease: 'power3.out', scrollTrigger: { trigger: '.contact', start: 'top 60%' } });
  gsap.from('.btn--xl', { scale: 0.85, opacity: 0, duration: 1, ease: 'back.out(1.6)', scrollTrigger: { trigger: '.contact', start: 'top 50%' } });

  // heading parallax
  $$('.h2').forEach((h) => {
    gsap.fromTo(h, { yPercent: 8 }, { yPercent: -8, ease: 'none', scrollTrigger: { trigger: h, start: 'top bottom', end: 'bottom top', scrub: true } });
  });

  // nav hide / show
  const nav = $('.nav');
  let last = 0;
  lenis.on('scroll', ({ scroll, direction }) => {
    if (scroll < 120) gsap.to(nav, { yPercent: 0, duration: 0.4, overwrite: true });
    else if (direction > 0 && scroll > last) gsap.to(nav, { yPercent: -110, duration: 0.4, overwrite: true });
    else if (direction < 0) gsap.to(nav, { yPercent: 0, duration: 0.4, overwrite: true });
    last = scroll;
  });
}

/* ------------------------------------------------------------------ *
 *  Loader -> intro
 * ------------------------------------------------------------------ */
document.body.classList.add('is-loading');
const numEl = $('#loaderNum');
const barEl = $('#loaderBar');
const prog = { v: 0 };
const fontsReady = Promise.race([document.fonts ? document.fonts.ready : Promise.resolve(), new Promise((r) => setTimeout(r, 2500))]);

gsap.to(prog, {
  v: 100, duration: reduceMotion ? 0.2 : 1.8, ease: 'power2.inOut',
  onUpdate: () => { numEl.textContent = Math.round(prog.v); barEl.style.transform = `scaleX(${prog.v / 100})`; },
  onComplete: async () => { await fontsReady; startIntro(); },
});

function startIntro() {
  buildScroll();
  const tl = gsap.timeline();
  tl.to('#loader', { yPercent: -100, duration: 1, ease: 'power4.inOut' })
    .add(() => { document.body.classList.remove('is-loading'); lenis.start(); ScrollTrigger.refresh(); }, '-=0.5')
    .to(chars, { yPercent: 0, rotate: 0, duration: 1.1, ease: 'power4.out', stagger: 0.025 }, '-=0.5')
    .to('.reveal', { opacity: 1, y: 0, duration: 1, ease: 'power3.out', stagger: 0.12 }, '-=0.7')
    .set('#loader', { display: 'none' });
}

addEventListener('load', () => ScrollTrigger.refresh());

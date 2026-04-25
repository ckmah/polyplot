import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "https://esm.sh/react@18.3.1";
import { createRoot } from "https://esm.sh/react-dom@18.3.1/client";
import { Canvas, useFrame, useThree } from "https://esm.sh/@react-three/fiber@8.17.10?deps=react@18.3.1,react-dom@18.3.1,three@0.160.1";
import {
  AdaptiveDpr,
  Bounds,
  Environment,
  GizmoHelper,
  GizmoViewport,
  Grid,
  OrbitControls,
  useBounds,
} from "https://esm.sh/@react-three/drei@9.105.0?deps=react@18.3.1,react-dom@18.3.1,three@0.160.1,@react-three/fiber@8.17.10";
import * as THREE from "https://esm.sh/three@0.160.1";
import { GLTFLoader } from "https://esm.sh/three@0.160.1/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "https://esm.sh/three@0.160.1/examples/jsm/libs/meshopt_decoder.module.js";

// Shared loader wired up for meshopt-compressed GLBs (gltfpack -cc output).
const _gltfLoader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);

const h = React.createElement;

const INITIAL_VIEW_DIR = new THREE.Vector3(0.52, 0.46, 0.72).normalize();

// Orbit zoom vs framing distance (initial camera sits ~fitDistance from target).
const ZOOM_MIN_FACTOR = 0.045;
const ZOOM_MAX_FACTOR = 0.3;
// Stream tiles by distance to camera (radii derived from camera.far + tile grid size).
const TILE_LOAD_FAR_MUL = 1.06;
const TILE_UNLOAD_MUL = 1.38;
const TILE_PAD_XY = 0.45;
const TILE_STALE_LOAD_SLACK = 1.06;
const TILE_FADE_MS = 280;
const CAMERA_RESET_MS = 360;
// Grazing XY views (|view·Z| low): shrink far as a fraction of depth-to-bbox along
// look direction so the frustum does not include the full lateral extent at once.
const FAR_EDGE_FRAC = 0.5;
const FAR_TOP_FRAC = 1.06;
const FAR_WZ_EDGE = 0.1;
const FAR_WZ_TOP = 0.72;

/** Closest-point distance from p to axis-aligned bbox (tile.bbox). */
function tileDistanceToAABB(px, py, pz, bbox) {
  const x = Math.min(Math.max(px, bbox[0]), bbox[3]);
  const y = Math.min(Math.max(py, bbox[1]), bbox[4]);
  const z = Math.min(Math.max(pz, bbox[2]), bbox[5]);
  const dx = px - x;
  const dy = py - y;
  const dz = pz - z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function tileStreamingRadii(camFar, tileSizeXy) {
  const ts = Number(tileSizeXy) || 0;
  const rLoad = camFar * TILE_LOAD_FAR_MUL + ts * TILE_PAD_XY;
  const rUnload = rLoad * TILE_UNLOAD_MUL;
  return { rLoad, rUnload };
}

function tileEntryForKey(tiles, key) {
  for (let i = 0; i < tiles.length; i++) {
    const t = tiles[i];
    if (`${t.col}_${t.row}` === key) return t;
  }
  return null;
}

/** Furthest bbox corner along camera look direction (positive = in front). */
function maxBboxDepthAlongDir(camPos, dir, minX, minY, minZ, maxX, maxY, maxZ) {
  let maxD = 0;
  for (let ix = 0; ix < 2; ix++) {
    const x = ix ? maxX : minX;
    for (let iy = 0; iy < 2; iy++) {
      const y = iy ? maxY : minY;
      for (let iz = 0; iz < 2; iz++) {
        const z = iz ? maxZ : minZ;
        const d =
          (x - camPos.x) * dir.x + (y - camPos.y) * dir.y + (z - camPos.z) * dir.z;
        if (d > maxD) maxD = d;
      }
    }
  }
  return maxD;
}

function useTrait(model, key) {
  const [val, setVal] = useState(() => model.get(key));
  useEffect(() => {
    const fn = () => setVal(model.get(key));
    model.on(`change:${key}`, fn);
    return () => {
      if (typeof model.off === "function") model.off(`change:${key}`, fn);
    };
  }, [model, key]);
  return val;
}

function framingFromBBox(bbox) {
  if (!bbox || bbox.length !== 6) {
    return { center: [0, 0, 0], distance: 5 };
  }
  const [minX, minY, minZ, maxX, maxY, maxZ] = bbox;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  const dx = maxX - minX;
  const dy = maxY - minY;
  const dz = maxZ - minZ;
  const diag = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
  return { center: [cx, cy, cz], distance: diag };
}

function fitCameraDistanceFromBBox(bbox, fovDeg, aspect, margin = 1.32) {
  const [minX, minY, minZ, maxX, maxY, maxZ] = bbox;
  const dx = Math.max(maxX - minX, 1e-9);
  const dy = Math.max(maxY - minY, 1e-9);
  const dz = Math.max(maxZ - minZ, 1e-9);
  const a = Math.max(aspect || 1, 0.05);
  const vFov = (fovDeg * Math.PI) / 180;
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * a);
  // Half the AABB space diagonal = bounding-sphere radius. Using max(dx,dy,dz)/2
  // is too small for angled views (typical flat XY slabs look "extremely zoomed in").
  const radius = 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz);
  const distV = radius / Math.tan(vFov / 2);
  const distH = radius / Math.tan(hFov / 2);
  return margin * Math.max(distV, distH);
}

function ReferenceGrid({ bbox }) {
  const { center, distance: diag } = framingFromBBox(bbox);
  const minZ = bbox[2];
  const floorZ = minZ - Math.max(0.02, diag * 0.03);
  const zGrid = floorZ + Math.max(0.0005, diag * 5e-5);
  const span = Math.max(diag * 14, 12);
  const cell = Math.max(0.04, diag / 64);
  const section = Math.max(0.25, diag / 16);
  return h(Grid, {
    position: [center[0], center[1], zGrid],
    rotation: [Math.PI / 2, 0, 0],
    args: [span, span],
    infiniteGrid: false,
    followCamera: false,
    cellSize: cell,
    sectionSize: section,
    cellColor: "#5c6470",
    sectionColor: "#5b8fd9",
    fadeDistance: Math.max(48, diag * 4.5),
    fadeStrength: 2.05,
    cellThickness: 1.1,
    sectionThickness: 1.1,
  });
}

function ZUpOrbitControls() {
  const orbitRef = useRef(null);
  return h(OrbitControls, {
    ref: orbitRef,
    makeDefault: true,
    enableDamping: true,
    dampingFactor: 0.08,
    enablePan: true,
    enableZoom: true,
    enableRotate: true,
    minDistance: 0.01,
    maxDistance: 1e6,
    screenSpacePanning: false,
  });
}

function BoundsRefit({ frameTick, bboxKey, bbox }) {
  const api = useBounds();
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);
  const controls = useThree((s) => s.controls);
  const fitDistanceRef = useRef(1);
  const sceneExtentRef = useRef(1);
  const resetAnimRef = useRef(null);
  const bbox6Ref = useRef({
    minX: 0,
    minY: 0,
    minZ: 0,
    maxX: 1,
    maxY: 1,
    maxZ: 1,
  });

  const applyRenderFar = useCallback(() => {
    const fd = fitDistanceRef.current;
    const extent = sceneExtentRef.current;
    if (!(fd > 0) || !(extent > 0)) return;
    const { minX, minY, minZ, maxX, maxY, maxZ } = bbox6Ref.current;
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    const wz = Math.abs(dir.z);
    const t = THREE.MathUtils.smoothstep(wz, FAR_WZ_EDGE, FAR_WZ_TOP);
    const maxD = maxBboxDepthAlongDir(
      camera.position,
      dir,
      minX,
      minY,
      minZ,
      maxX,
      maxY,
      maxZ,
    );
    const depth = Math.max(maxD, extent * 0.02, camera.near * 10);
    const farFull = depth * FAR_TOP_FRAC + camera.near * 2;
    const frac = THREE.MathUtils.lerp(FAR_EDGE_FRAC, 1, t);
    let far = farFull * frac;
    far = Math.max(far, camera.near * 50);
    far = Math.min(far, farFull);
    camera.far = far;
    camera.updateProjectionMatrix();
  }, [camera]);

  useLayoutEffect(() => {
    if (!bbox || bbox.length !== 6) return;
    if (!size.width || !size.height) return;
    const aspect = size.width / size.height;
    const [minX, minY, minZ, maxX, maxY, maxZ] = bbox;
    const sceneBox = new THREE.Box3(
      new THREE.Vector3(minX, minY, minZ),
      new THREE.Vector3(maxX, maxY, maxZ),
    );
    const { center: c } = framingFromBBox(bbox);
    const center = new THREE.Vector3(c[0], c[1], c[2]);
    const distance = fitCameraDistanceFromBBox(bbox, camera.fov, aspect, 1.32);
    const pos = center.clone().addScaledVector(INITIAL_VIEW_DIR, distance);
    queueMicrotask(() => {
      api.refresh(sceneBox);
      bbox6Ref.current = { minX, minY, minZ, maxX, maxY, maxZ };
      const extent = sceneBox.getSize(new THREE.Vector3()).length();
      sceneExtentRef.current = Math.max(extent, 1e-9);
      fitDistanceRef.current = Math.max(distance, 1e-9);
      camera.near = Math.max(0.01, extent * 1e-4);
      camera.up.set(0, 0, 1);
      const canAnimate =
        frameTick > 0 && controls && controls.target;
      if (controls && controls.target) {
        controls.minDistance = Math.max(extent * 1e-5, distance * ZOOM_MIN_FACTOR);
        controls.maxDistance = distance * ZOOM_MAX_FACTOR;
      }
      if (!canAnimate) {
        resetAnimRef.current = null;
        camera.position.copy(pos);
        camera.lookAt(center);
        if (controls && controls.target) {
          controls.target.copy(center);
          controls.update();
        }
        applyRenderFar();
        camera.updateMatrixWorld();
        invalidate();
        return;
      }
      resetAnimRef.current = {
        t0: performance.now(),
        dur: CAMERA_RESET_MS,
        fromPos: camera.position.clone(),
        fromTgt: controls.target.clone(),
        toPos: pos.clone(),
        toTgt: center.clone(),
      };
      controls.update();
      applyRenderFar();
      camera.updateMatrixWorld();
      invalidate();
    });
  }, [api, frameTick, bboxKey, size.width, size.height, bbox, camera, controls, invalidate, applyRenderFar]);

  useFrame(() => {
    const anim = resetAnimRef.current;
    if (!anim || !controls || !controls.target) return;
    const t = Math.min(1, (performance.now() - anim.t0) / anim.dur);
    const k = t * t * (3 - 2 * t);
    camera.position.lerpVectors(anim.fromPos, anim.toPos, k);
    controls.target.lerpVectors(anim.fromTgt, anim.toTgt, k);
    controls.update();
    applyRenderFar();
    invalidate();
    if (t >= 1) {
      camera.position.copy(anim.toPos);
      controls.target.copy(anim.toTgt);
      controls.update();
      resetAnimRef.current = null;
      applyRenderFar();
      invalidate();
    }
  });

  useEffect(() => {
    if (!controls) return;
    const onChange = () => {
      applyRenderFar();
      invalidate();
    };
    controls.addEventListener("change", onChange);
    return () => controls.removeEventListener("change", onChange);
  }, [controls, applyRenderFar, invalidate]);

  useEffect(() => {
    if (!controls || !controls.domElement) return;
    const el = controls.domElement;
    const cancel = () => {
      resetAnimRef.current = null;
    };
    el.addEventListener("pointerdown", cancel);
    return () => el.removeEventListener("pointerdown", cancel);
  }, [controls]);

  return null;
}

function ShadowFloor({ bbox }) {
  if (!bbox || bbox.length !== 6) return null;
  const { center, distance: diag } = framingFromBBox(bbox);
  const minZ = bbox[2];
  const floorZ = minZ - Math.max(0.02, diag * 0.03);
  const span = Math.max(diag * 4, 4);
  const geom = useMemo(() => new THREE.PlaneGeometry(span, span), [span]);
  const mat = useMemo(() => new THREE.ShadowMaterial({ opacity: 0.5, transparent: true }), []);
  useEffect(() => () => { geom.dispose(); mat.dispose(); }, [geom, mat]);
  return h("mesh", {
    geometry: geom,
    material: mat,
    position: [center[0], center[1], floorZ],
    receiveShadow: true,
  });
}

function KeyLight({ bbox }) {
  const ref = useRef(null);
  const { center, distance: diag } = useMemo(() => framingFromBBox(bbox || []), [bbox]);

  useLayoutEffect(() => {
    const light = ref.current;
    if (!light || !bbox || bbox.length !== 6) return;
    const [cx, cy, cz] = center;
    light.target.position.set(cx, cy, cz);
    light.target.updateMatrixWorld();

    const pad = Math.max(diag * 2.5, 2);
    const cam = light.shadow.camera;
    cam.left = -pad; cam.right = pad; cam.top = pad; cam.bottom = -pad;
    cam.near = 0.05;
    cam.far = Math.max(diag * 40, 200);
    cam.updateProjectionMatrix();

    light.shadow.mapSize.width = 2048;
    light.shadow.mapSize.height = 2048;
    light.shadow.bias = -0.00025;
    light.shadow.normalBias = 0.035;
  }, [bbox, center, diag]);

  const [cx, cy, cz] = center;
  const d = Math.max(diag, 0.5);
  return h("directionalLight", {
    ref,
    position: [cx + d * 1.4, cy + d * 1.1, cz + d * 1.8],
    intensity: 1.2,
    castShadow: true,
  });
}

function DemandInvalidate({ opacity }) {
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => { invalidate(); }, [opacity, invalidate]);
  return null;
}

// ---------------------------------------------------------------------------
// Tile management
// ---------------------------------------------------------------------------

function _makeMaterial(wireframe, opacity) {
  const a = opacity == null ? 1 : Math.min(1, Math.max(0, opacity));
  // MeshPhysicalMaterial adds a thin clearcoat + sheen for depth in shading.
  // Clearcoat/sheen amplify any per-face normal difference, which otherwise
  // makes earcut sliver tris on flat caps visible under PBR — kept subtle.
  // Meshes are watertight, so FrontSide is safe and halves shading work.
  return new THREE.MeshPhysicalMaterial({
    vertexColors: true,
    wireframe: !!wireframe,
    side: THREE.FrontSide,
    metalness: 0.05,
    roughness: 0.55,
    clearcoat: 0.12,
    clearcoatRoughness: 0.7,
    sheen: 0.08,
    sheenRoughness: 0.85,
    envMapIntensity: 0.7,
    flatShading: false,
    transparent: a < 1,
    opacity: a,
    depthWrite: a >= 0.99,
  });
}

function TileManager({ tileServerUrl, tilesJsonPath, wireframe, opacity, maxFetches }) {
  const { invalidate } = useThree();
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls);
  const invalidateRef = useRef(invalidate);
  invalidateRef.current = invalidate;
  const tileServerUrlRef = useRef(tileServerUrl);
  tileServerUrlRef.current = tileServerUrl;
  const reconcileRef = useRef(() => {});
  const tileFadeStartRef = useRef(new Map());

  const tilesRef     = useRef(null);          // parsed tiles.json
  const loadedRef    = useRef(new Map());     // key -> THREE.Group
  const pendingRef   = useRef(new Set());     // keys currently fetching
  const propsRef     = useRef({ wireframe, opacity });
  const fetchGenRef  = useRef(0);             // bump when tile server / index changes
  const [version, setVersion] = useState(0);

  useEffect(() => { propsRef.current = { wireframe, opacity }; }, [wireframe, opacity]);

  useEffect(() => {
    loadedRef.current.forEach((group, key) => {
      const fading = tileFadeStartRef.current.has(key);
      group.traverse((obj) => {
        if (!obj.isMesh || !obj.material) return;
        obj.material.wireframe = !!wireframe;
        if (!fading) {
          const a = opacity == null ? 1 : Math.min(1, Math.max(0, opacity));
          obj.material.transparent = a < 1;
          obj.material.opacity = a;
          obj.material.depthWrite = a >= 0.99;
        }
        obj.material.needsUpdate = true;
      });
    });
    invalidateRef.current();
  }, [wireframe, opacity]);

  function _unloadTile(key, bump = true) {
    tileFadeStartRef.current.delete(key);
    const group = loadedRef.current.get(key);
    if (!group) return;
    group.traverse((obj) => {
      if (obj.isMesh) {
        obj.geometry.dispose();
        if (obj.material) obj.material.dispose();
      }
    });
    loadedRef.current.delete(key);
    if (bump) { setVersion((v) => v + 1); invalidateRef.current(); }
  }

  function _disposeAllLoaded() {
    tileFadeStartRef.current.clear();
    for (const key of [...loadedRef.current.keys()]) {
      _unloadTile(key, false);
    }
    pendingRef.current.clear();
    setVersion((v) => v + 1);
  }

  const cap = maxFetches == null || maxFetches < 1 ? 4 : maxFetches;

  function _pumpLoads() {
    const idx = tilesRef.current;
    const base = tileServerUrlRef.current;
    if (!idx || !idx.tiles || !base) return;
    const px = camera.position.x;
    const py = camera.position.y;
    const pz = camera.position.z;
    const { rLoad } = tileStreamingRadii(camera.far, idx.tile_size_xy);
    const candidates = [];
    for (let i = 0; i < idx.tiles.length; i++) {
      const tile = idx.tiles[i];
      const key = `${tile.col}_${tile.row}`;
      if (loadedRef.current.has(key) || pendingRef.current.has(key)) continue;
      const d = tileDistanceToAABB(px, py, pz, tile.bbox);
      if (d <= rLoad) candidates.push({ key, d, tile });
    }
    candidates.sort((a, b) => a.d - b.d);
    for (let j = 0; j < candidates.length; j++) {
      if (pendingRef.current.size >= cap) break;
      const { key, tile } = candidates[j];
      _loadTile(key, `${base}/${tile.glb}`, null);
    }
  }

  function _reconcileStreaming() {
    const idx = tilesRef.current;
    if (!idx || !idx.tiles) return;
    const px = camera.position.x;
    const py = camera.position.y;
    const pz = camera.position.z;
    const { rUnload } = tileStreamingRadii(camera.far, idx.tile_size_xy);
    for (const key of [...loadedRef.current.keys()]) {
      const tile = tileEntryForKey(idx.tiles, key);
      if (!tile) continue;
      if (tileDistanceToAABB(px, py, pz, tile.bbox) > rUnload) {
        _unloadTile(key, true);
      }
    }
    _pumpLoads();
  }

  reconcileRef.current = _reconcileStreaming;

  // Load tile; on success swap out the old LOD (crossfade: old stays visible
  // until new has fully arrived, eliminating the blank-frame gap).
  function _loadTile(key, url, swapOutKey) {
    const gen = fetchGenRef.current;
    pendingRef.current.add(key);
    _gltfLoader.load(
      url,
      (gltf) => {
        pendingRef.current.delete(key);
        if (gen !== fetchGenRef.current) return;
        const idx0 = tilesRef.current;
        const tile0 = idx0 && idx0.tiles ? tileEntryForKey(idx0.tiles, key) : null;
        if (tile0) {
          const { rLoad } = tileStreamingRadii(camera.far, idx0.tile_size_xy);
          const d0 = tileDistanceToAABB(
            camera.position.x,
            camera.position.y,
            camera.position.z,
            tile0.bbox,
          );
          if (d0 > rLoad * TILE_STALE_LOAD_SLACK) {
            gltf.scene.traverse((obj) => {
              if (!obj.isMesh) return;
              obj.geometry.dispose();
              if (obj.material) obj.material.dispose();
            });
            _pumpLoads();
            return;
          }
        }
        const { wireframe: wf, opacity: op } = propsRef.current;
        gltf.scene.traverse((obj) => {
          if (!obj.isMesh) return;
          // GLBs ship explicit NORMALs (cap = ±Z, side = area-weighted) so the
          // renderer does not need to recompute. Meshopt position quantization
          // would otherwise leak into face normals and produce faint radial
          // streaks on flat caps. Fall back only if normals are missing.
          if (!obj.geometry.attributes.normal) {
            obj.geometry.computeVertexNormals();
          }
          obj.material = _makeMaterial(wf, op);
          obj.material.transparent = true;
          obj.material.opacity = 0;
          obj.material.depthWrite = false;
          obj.castShadow = true;
        });
        tileFadeStartRef.current.set(key, performance.now());
        loadedRef.current.set(key, gltf.scene);
        // Only now discard the old LOD — avoids the blank-frame flicker.
        if (swapOutKey && swapOutKey !== key) _unloadTile(swapOutKey, false);
        setVersion((v) => v + 1);
        invalidateRef.current();
        _pumpLoads();
      },
      undefined,
      (err) => {
        pendingRef.current.delete(key);
        console.error("[TileManager] tile load error", key, err);
        invalidateRef.current();
        _pumpLoads();
      },
    );
  }

  useEffect(() => {
    if (!tileServerUrl) return;
    const ac = new AbortController();
    let alive = true;
    fetchGenRef.current += 1;
    const gen = fetchGenRef.current;
    tilesRef.current = null;
    _disposeAllLoaded();

    const url = `${tileServerUrl}/${tilesJsonPath}`;
    fetch(url, { signal: ac.signal })
      .then((r) => {
        if (!alive || gen !== fetchGenRef.current) return Promise.reject(new Error("stale"));
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((data) => {
        if (!alive || gen !== fetchGenRef.current) return;
        tilesRef.current = data;
        setVersion((v) => v + 1);
        invalidateRef.current();
        _pumpLoads();
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            reconcileRef.current();
            invalidateRef.current();
          });
        });
      })
      .catch((err) => {
        if (!alive) return;
        if (err && (err.name === "AbortError" || err.message === "stale")) return;
        console.error("[TileManager] tiles.json fetch error:", err);
      });

    return () => {
      alive = false;
      ac.abort();
      fetchGenRef.current += 1;
      tilesRef.current = null;
      _disposeAllLoaded();
    };
  }, [tileServerUrl, tilesJsonPath]);

  useEffect(() => {
    if (!controls) return;
    const onCam = () => {
      reconcileRef.current();
      invalidateRef.current();
    };
    controls.addEventListener("change", onCam);
    onCam();
    return () => controls.removeEventListener("change", onCam);
  }, [controls]);

  useFrame(() => {
    const fades = tileFadeStartRef.current;
    if (fades.size === 0) return;
    const op = propsRef.current.opacity;
    const aFull = op == null ? 1 : Math.min(1, Math.max(0, op));
    const now = performance.now();
    let needInv = false;
    for (const key of [...fades.keys()]) {
      const group = loadedRef.current.get(key);
      if (!group) {
        fades.delete(key);
        continue;
      }
      const t0 = fades.get(key);
      const u = Math.min(1, (now - t0) / TILE_FADE_MS);
      const sm = u * u * (3 - 2 * u);
      const a = aFull * sm;
      group.traverse((obj) => {
        if (!obj.isMesh || !obj.material) return;
        obj.material.opacity = a;
        obj.material.transparent = a < 0.99 || aFull < 0.99;
        obj.material.depthWrite = a >= 0.99 && aFull >= 0.99;
        obj.material.needsUpdate = true;
      });
      needInv = true;
      if (u >= 1) {
        fades.delete(key);
        group.traverse((obj) => {
          if (!obj.isMesh || !obj.material) return;
          obj.material.opacity = aFull;
          obj.material.transparent = aFull < 1;
          obj.material.depthWrite = aFull >= 0.99;
          obj.material.needsUpdate = true;
        });
      }
    }
    if (needInv) invalidateRef.current();
  });

  // Render loaded tile scenes as primitives
  return h(
    React.Fragment,
    null,
    ...[...loadedRef.current.entries()].map(([key, scene]) =>
      h("primitive", { key, object: scene }),
    ),
  );
}

// ---------------------------------------------------------------------------
// Scene root
// ---------------------------------------------------------------------------

function Scene({ model }) {
  const tileServerUrl      = useTrait(model, "tile_server_url");
  const tilesJsonPath      = useTrait(model, "tiles_json_path");
  const bbox               = useTrait(model, "bbox");
  const maxFetches         = useTrait(model, "max_concurrent_fetches");

  const [opacity, setOpacity] = useState(1.0);
  const [wireframe, setWireframe] = useState(false);
  const [background, setBackground] = useState("#111418");

  const [frameTick, setFrameTick] = useState(0);
  const resetCamera = useCallback(() => setFrameTick((t) => t + 1), []);

  const hasBbox  = bbox && bbox.length === 6;
  const bboxKey  = hasBbox ? bbox.join(",") : "";
  const hasServer = !!tileServerUrl;

  return h(
    React.Fragment,
    null,
    h(
      Canvas,
      {
        frameloop: "demand",
        camera: { fov: 45, near: 0.01, far: 10000, position: [1, 1, 2] },
        style: { background: background || "#111418" },
        gl: { antialias: true, logarithmicDepthBuffer: true },
        shadows: true,
        dpr: [1, 2],
        onCreated: ({ gl, camera }) => {
          camera.up.set(0, 0, 1);
          gl.shadowMap.enabled = true;
          gl.shadowMap.type = THREE.PCFSoftShadowMap;
        },
      },
      h(AdaptiveDpr),
      h(DemandInvalidate, { opacity }),
      h("ambientLight", { intensity: 0.32 }),
      h("hemisphereLight", {
        args: ["#bcd6f5", "#1b2128", 0.45],
        position: [0, 0, 1],
      }),
      hasBbox && h(KeyLight, { bbox }),
      h("directionalLight", { position: [-3, -2, -4], intensity: 0.22 }),
      // Neutral preset → cheap HDR-ish environment without shipping an asset;
      // gives MeshPhysicalMaterial fresnel/clearcoat something meaningful to
      // reflect. `background:false` so it doesn't override the canvas color.
      h(Environment, { preset: "city", background: false }),
      h(ZUpOrbitControls),
      hasServer && hasBbox &&
        h(
          Bounds,
          { fit: false, clip: false, observe: false, margin: 1.35 },
          h(TileManager, { tileServerUrl, tilesJsonPath, wireframe, opacity, maxFetches }),
          h(BoundsRefit, { frameTick, bboxKey, bbox }),
        ),
      hasBbox && h(ShadowFloor, { bbox }),
      hasBbox && h(ReferenceGrid, { bbox }),
      // Screen-space ambient occlusion — grounds cells and darkens crevices.
      // Runs after the scene renders, last in the Canvas children.
      h(GizmoHelper, { alignment: "bottom-right", margin: [80, 72] }, h(GizmoViewport, { labels: ["X", "Y", "Z"] })),
    ),
    h(
      "div",
      { className: "polyfiber-r3f-cambar" },
      h(
        "button",
        { type: "button", className: "polyfiber-r3f-cambar-btn", onClick: resetCamera },
        "Reset view",
      ),
      h(
        "label",
        { className: "polyfiber-r3f-cambar-ctl" },
        h("input", {
          type: "checkbox",
          checked: wireframe,
          "aria-label": "Wireframe",
          onChange: (e) => setWireframe(!!e.target.checked),
        }),
        " Wireframe",
      ),
      h(
        "label",
        { className: "polyfiber-r3f-cambar-ctl" },
        "Opacity ",
        h("input", {
          type: "range",
          min: 0,
          max: 1,
          step: 0.02,
          value: opacity,
          onChange: (e) => setOpacity(parseFloat(e.target.value) || 0),
        }),
      ),
      h(
        "label",
        { className: "polyfiber-r3f-cambar-ctl" },
        "BG ",
        h("input", {
          type: "color",
          value: background.length === 7 ? background : "#111418",
          onChange: (e) => setBackground(e.target.value),
        }),
      ),
      h(
        "span",
        { className: "polyfiber-r3f-cambar-hint" },
        "Left drag: orbit · Scroll: zoom · Right drag: pan",
      ),
    ),
    h("div", { className: "polyfiber-r3f-hud" }, tileServerUrl
      ? `streaming from ${tileServerUrl}`
      : "waiting for tile server…"
    ),
  );
}

function render({ model, el }) {
  el.classList.add("polyfiber-r3f-host");
  const root = createRoot(el);
  root.render(h(Scene, { model }));
  return () => {
    root.unmount();
    el.classList.remove("polyfiber-r3f-host");
  };
}

export default { render };

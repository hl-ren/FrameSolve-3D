import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { BoxSelect, ChevronDown, ChevronRight, CircleDot, CircleHelp, Eraser, Eye, EyeOff, FilePlus2, FolderOpen, Hammer, Link2, Play, Plus, RefreshCcw, Save, Sigma, Trash2, Undo2, X } from "lucide-react";
import { solveModalAnalysis, solveStructure } from "./core/fem";
import { type BoundaryCondition, type DofKey, dofKeys, type ElementForce, type ElementLoad, type ElementType, type LoadCoordinate, type Material, type ModalResult, type Section, type SolveResult, type StructureModel, type StructureNode, type Vec3 } from "./core/types";

type Tool = "select" | "member" | "support" | "load" | "delete";
type UnitKey = "m" | "cm" | "mm";
type LanguageKey = "zh" | "en";
type ExampleKey = "cube" | "bridge" | "tower" | "cantilever" | "orientalPearl" | "tongjiCivil";

const unitScale: Record<UnitKey, number> = {
  m: 1,
  cm: 0.01,
  mm: 0.001,
};

const unitLabels: Record<UnitKey, string> = {
  m: "m",
  cm: "cm",
  mm: "mm",
};

const defaultGridStepByUnit: Record<UnitKey, number> = {
  m: 1,
  cm: 5,
  mm: 5,
};

type HoverInfo = {
  elementId: string;
  x: number;
  y: number;
  lines: string[];
};

type GridBounds = {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  zMin: number;
  zMax: number;
};

type ElementLoadDraft = {
  type: ElementLoad["type"];
  coordinate: LoadCoordinate;
  position: number;
  x: number;
  y: number;
  z: number;
};

type ElementVisualKind = "bar" | "beam" | "mixed";

type ExampleDefinition = {
  key: ExampleKey;
  name: string;
  nameEn: string;
  unit: UnitKey;
  gridStep?: number;
  selectNodeId: string;
  model: () => StructureModel;
};

type ProjectFile = {
  app: "FrameSolve 3D" | "BeamBar3D";
  version: "1.0";
  savedAt: string;
  unit: UnitKey;
  gridStep: number;
  gridVisible: boolean;
  loadZ: number;
  offset: { dx: number; dy: number; dz: number };
  coordinateNode: Vec3;
  defaultMaterialId?: string;
  defaultSectionId?: string;
  language?: LanguageKey;
  model: StructureModel;
};

type SolveKind = "static" | "modal";

const appName = "FrameSolve 3D";

const solveScaleLimits: Record<SolveKind, { activeNodes: number; elements: number; weightedDofs: number }> = {
  static: { activeNodes: 850, elements: 6500, weightedDofs: 2800 },
  modal: { activeNodes: 550, elements: 4200, weightedDofs: 1800 },
};

const defaultMaterials: Material[] = [
  { id: "wood", name: "Wood", E: 1.1e10, G: 6.9e8, density: 500 },
  { id: "glulam", name: "Glulam", E: 1.25e10, G: 7.8e8, density: 540 },
  { id: "steel", name: "Steel", E: 2.06e11, G: 7.9e10, density: 7850 },
  { id: "concrete", name: "Concrete", E: 3.0e10, G: 1.25e10, density: 2500 },
];

const defaultSections: Section[] = [
  { id: "timber100", name: "Timber 100x100", A: 0.01, Iy: 8.333e-6, Iz: 8.333e-6, J: 1.4e-5 },
  { id: "timber50x100", name: "Timber 50x100", A: 0.005, Iy: 4.167e-6, Iz: 1.042e-6, J: 2.8e-6 },
  { id: "timber100x200", name: "Timber 100x200", A: 0.02, Iy: 6.667e-5, Iz: 1.667e-5, J: 2.1e-5 },
  { id: "steelTube80", name: "Steel Tube 80x4", A: 0.000955, Iy: 7.2e-7, Iz: 7.2e-7, J: 1.44e-6 },
];

const defaultGridBounds: GridBounds = {
  xMin: 0,
  xMax: 10,
  yMin: 0,
  yMax: 10,
  zMin: 0,
  zMax: 10,
};

const memberSnapRatios = [1 / 3, 1 / 2, 2 / 3];

function gridValues(min: number, max: number, step: number): number[] {
  const values: number[] = [];
  const start = Math.ceil(min / step) * step;
  for (let value = start; value <= max + 1e-9; value += step) {
    values.push(Number(value.toFixed(6)));
  }
  return values;
}

function snapDown(value: number, step: number): number {
  return Number((Math.floor(value / step) * step).toFixed(6));
}

function snapUp(value: number, step: number): number {
  return Number((Math.ceil(value / step) * step).toFixed(6));
}

function getGridBounds(nodes: StructureNode[], step: number): GridBounds {
  if (nodes.length === 0) return defaultGridBounds;
  const margin = step * 2;
  const xs = nodes.map((node) => node.x);
  const ys = nodes.map((node) => node.y);
  const zs = nodes.map((node) => node.z);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const zMin = Math.min(...zs);
  const zMax = Math.max(...zs);
  return {
    xMin: snapDown(xMin < defaultGridBounds.xMin ? xMin - margin : defaultGridBounds.xMin, step),
    xMax: snapUp(xMax > defaultGridBounds.xMax ? xMax + margin : defaultGridBounds.xMax, step),
    yMin: snapDown(yMin < defaultGridBounds.yMin ? yMin - margin : defaultGridBounds.yMin, step),
    yMax: snapUp(yMax > defaultGridBounds.yMax ? yMax + margin : defaultGridBounds.yMax, step),
    zMin: snapDown(zMin < defaultGridBounds.zMin ? zMin - margin : defaultGridBounds.zMin, step),
    zMax: snapUp(zMax > defaultGridBounds.zMax ? zMax + margin : defaultGridBounds.zMax, step),
  };
}

function closestGridPointToRay(ray: THREE.Ray, points: Vec3[], scale: number, pickRadius: number): Vec3 | null {
  let best: Vec3 | null = null;
  let bestDistance = pickRadius * pickRadius;
  const point = new THREE.Vector3();
  for (const candidate of points) {
    point.set(candidate.x * scale, candidate.z * scale, candidate.y * scale);
    const distance = ray.distanceSqToPoint(point);
    if (distance <= bestDistance) {
      bestDistance = distance;
      best = candidate;
    }
  }
  return best;
}

function nearestMemberSnapRatio(hitPoint: THREE.Vector3, start: StructureNode, end: StructureNode, scale: number): number | null {
  const a = new THREE.Vector3(start.x * scale, start.z * scale, start.y * scale);
  const b = new THREE.Vector3(end.x * scale, end.z * scale, end.y * scale);
  const ab = b.clone().sub(a);
  const lengthSquared = ab.lengthSq();
  if (lengthSquared < 1e-12) return null;
  const ratio = THREE.MathUtils.clamp(hitPoint.clone().sub(a).dot(ab) / lengthSquared, 0, 1);
  let bestRatio = memberSnapRatios[0];
  let bestDistance = Math.abs(ratio - bestRatio);
  for (const candidate of memberSnapRatios.slice(1)) {
    const distance = Math.abs(ratio - candidate);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestRatio = candidate;
    }
  }
  return bestDistance <= 0.12 ? bestRatio : null;
}

function memberSnapLabel(ratio: number): string {
  if (Math.abs(ratio - 0.5) < 1e-6) return "1/2";
  return ratio < 0.5 ? "1/3" : "2/3";
}

function createSpatialGridLineGeometry(xValues: number[], yValues: number[], zValues: number[], scale: number): THREE.BufferGeometry {
  const positions: number[] = [];
  const pushLine = (start: Vec3, end: Vec3) => {
    positions.push(start.x * scale, start.z * scale, start.y * scale, end.x * scale, end.z * scale, end.y * scale);
  };

  for (const y of yValues) {
    for (const z of zValues) {
      pushLine({ x: xValues[0], y, z }, { x: xValues[xValues.length - 1], y, z });
    }
  }
  for (const x of xValues) {
    for (const z of zValues) {
      pushLine({ x, y: yValues[0], z }, { x, y: yValues[yValues.length - 1], z });
    }
  }
  for (const x of xValues) {
    for (const y of yValues) {
      pushLine({ x, y, z: zValues[0] }, { x, y, z: zValues[zValues.length - 1] });
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  return geometry;
}

function createTextSprite(text: string, color: string, scale = 1): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 64;
  const context = canvas.getContext("2d")!;
  context.font = "700 30px sans-serif";
  context.fillStyle = color;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
  sprite.scale.set(0.55 * scale, 0.28 * scale, 1);
  return sprite;
}

function createCoordinateAxes(symbolScale: number): THREE.Group {
  const group = new THREE.Group();
  const origin = new THREE.Mesh(
    new THREE.SphereGeometry(0.07 * symbolScale, 18, 12),
    new THREE.MeshStandardMaterial({ color: "#111827", roughness: 0.35 }),
  );
  group.add(origin);

  const axes = [
    { label: "X", color: "#d33f32", direction: new THREE.Vector3(1, 0, 0), position: new THREE.Vector3(2.15 * symbolScale, 0, 0) },
    { label: "Y", color: "#1b8f4d", direction: new THREE.Vector3(0, 0, 1), position: new THREE.Vector3(0, 0, 2.15 * symbolScale) },
    { label: "Z", color: "#1f6feb", direction: new THREE.Vector3(0, 1, 0), position: new THREE.Vector3(0, 2.15 * symbolScale, 0) },
  ];
  for (const axis of axes) {
    group.add(new THREE.ArrowHelper(axis.direction, new THREE.Vector3(0, 0, 0), 2 * symbolScale, axis.color, 0.22 * symbolScale, 0.1 * symbolScale));
    const label = createTextSprite(axis.label, axis.color, symbolScale);
    label.position.copy(axis.position);
    group.add(label);
  }
  const originLabel = createTextSprite("O", "#111827", symbolScale);
  originLabel.position.set(-0.25 * symbolScale, 0.2 * symbolScale, -0.25 * symbolScale);
  group.add(originLabel);
  return group;
}

function elementHasExplicitRelease(element: { releaseStart?: Partial<Record<DofKey, boolean>>; releaseEnd?: Partial<Record<DofKey, boolean>> }): boolean {
  return [...Object.values(element.releaseStart ?? {}), ...Object.values(element.releaseEnd ?? {})].some(Boolean);
}

function elementVisualKind(element: { type: ElementType; startNodeId: string; endNodeId: string; releaseStart?: Partial<Record<DofKey, boolean>>; releaseEnd?: Partial<Record<DofKey, boolean>> }, nodeMap: Map<string, StructureNode>): ElementVisualKind {
  if (element.type === "bar3d") return "bar";
  const start = nodeMap.get(element.startNodeId);
  const end = nodeMap.get(element.endNodeId);
  return start?.joint === "hinged" || end?.joint === "hinged" || elementHasExplicitRelease(element) ? "mixed" : "beam";
}

function elementTypeColor(kind: ElementVisualKind): string {
  if (kind === "bar") return "#2563eb";
  if (kind === "mixed") return "#d97706";
  return "#7c3aed";
}

function elementTypeLabel(kind: ElementVisualKind, language: LanguageKey = "zh"): string {
  if (language === "en") {
    if (kind === "bar") return "Bar";
    if (kind === "mixed") return "Hinged beam / mixed";
    return "Beam";
  }
  if (kind === "bar") return "杆单元";
  if (kind === "mixed") return "铰接梁/混合";
  return "梁单元";
}

function inferElementType(element: { id: string; startNodeId: string; endNodeId: string }, model: StructureModel): ElementType {
  const hasElementLoad = (model.elementLoads ?? []).some((load) => load.elementId === element.id);
  if (hasElementLoad) return "beam3d";
  const start = model.nodes.find((node) => node.id === element.startNodeId);
  const end = model.nodes.find((node) => node.id === element.endNodeId);
  return start?.joint === "rigid" || end?.joint === "rigid" ? "beam3d" : "bar3d";
}

const emptyModel = (): StructureModel => ({
  nodes: [],
  elements: [],
  materials: defaultMaterials.map((material) => ({ ...material })),
  sections: defaultSections.map((section) => ({ ...section })),
  boundaries: [],
  loads: [],
  elementLoads: [],
  nodalMasses: [],
});

const newProjectModel = (): StructureModel => ({
  ...emptyModel(),
  nodes: [{ id: "N1", x: 0, y: 0, z: 0 }],
  boundaries: [{ nodeId: "N1", ux: true, uy: true, uz: true, rx: true, ry: true, rz: true }],
});

function barElement(id: string, startNodeId: string, endNodeId: string, sectionId = "timber100"): StructureModel["elements"][number] {
  return { id, type: "bar3d", startNodeId, endNodeId, materialId: "wood", sectionId };
}

function bars(pairs: Array<[string, string]>, sectionId = "timber100"): StructureModel["elements"] {
  return pairs.map(([startNodeId, endNodeId], index) => barElement(`E${index + 1}`, startNodeId, endNodeId, sectionId));
}

const sampleModel = (): StructureModel => ({
  ...emptyModel(),
  nodes: [
    { id: "N1", x: 0, y: 0, z: 0 },
    { id: "N2", x: 10, y: 0, z: 0 },
    { id: "N3", x: 10, y: 10, z: 0 },
    { id: "N4", x: 0, y: 10, z: 0 },
    { id: "N5", x: 0, y: 0, z: 10 },
    { id: "N6", x: 10, y: 0, z: 10 },
    { id: "N7", x: 10, y: 10, z: 10 },
    { id: "N8", x: 0, y: 10, z: 10 },
  ],
  elements: [
    { id: "E1", type: "bar3d", startNodeId: "N1", endNodeId: "N2", materialId: "wood", sectionId: "timber100" },
    { id: "E2", type: "bar3d", startNodeId: "N2", endNodeId: "N3", materialId: "wood", sectionId: "timber100" },
    { id: "E3", type: "bar3d", startNodeId: "N3", endNodeId: "N4", materialId: "wood", sectionId: "timber100" },
    { id: "E4", type: "bar3d", startNodeId: "N4", endNodeId: "N1", materialId: "wood", sectionId: "timber100" },
    { id: "E5", type: "bar3d", startNodeId: "N5", endNodeId: "N6", materialId: "wood", sectionId: "timber100" },
    { id: "E6", type: "bar3d", startNodeId: "N6", endNodeId: "N7", materialId: "wood", sectionId: "timber100" },
    { id: "E7", type: "bar3d", startNodeId: "N7", endNodeId: "N8", materialId: "wood", sectionId: "timber100" },
    { id: "E8", type: "bar3d", startNodeId: "N8", endNodeId: "N5", materialId: "wood", sectionId: "timber100" },
    { id: "E9", type: "bar3d", startNodeId: "N1", endNodeId: "N5", materialId: "wood", sectionId: "timber100" },
    { id: "E10", type: "bar3d", startNodeId: "N2", endNodeId: "N6", materialId: "wood", sectionId: "timber100" },
    { id: "E11", type: "bar3d", startNodeId: "N3", endNodeId: "N7", materialId: "wood", sectionId: "timber100" },
    { id: "E12", type: "bar3d", startNodeId: "N4", endNodeId: "N8", materialId: "wood", sectionId: "timber100" },
    { id: "E13", type: "bar3d", startNodeId: "N1", endNodeId: "N3", materialId: "wood", sectionId: "timber100" },
    { id: "E14", type: "bar3d", startNodeId: "N5", endNodeId: "N7", materialId: "wood", sectionId: "timber100" },
    { id: "E15", type: "bar3d", startNodeId: "N1", endNodeId: "N6", materialId: "wood", sectionId: "timber100" },
    { id: "E16", type: "bar3d", startNodeId: "N2", endNodeId: "N7", materialId: "wood", sectionId: "timber100" },
    { id: "E17", type: "bar3d", startNodeId: "N3", endNodeId: "N8", materialId: "wood", sectionId: "timber100" },
    { id: "E18", type: "bar3d", startNodeId: "N4", endNodeId: "N5", materialId: "wood", sectionId: "timber100" },
  ],
  boundaries: [
    { nodeId: "N1", ux: true, uy: true, uz: true },
    { nodeId: "N2", ux: true, uy: true, uz: true },
    { nodeId: "N3", ux: true, uy: true, uz: true },
    { nodeId: "N4", ux: true, uy: true, uz: true },
  ],
  loads: [
    { nodeId: "N5", fz: -15000 },
    { nodeId: "N6", fz: -15000 },
    { nodeId: "N7", fz: -15000 },
    { nodeId: "N8", fz: -15000 },
  ],
});

const bridgeTrussModel = (): StructureModel => ({
  ...emptyModel(),
  nodes: [
    { id: "N1", x: 0, y: 0, z: 0 },
    { id: "N2", x: 0, y: 10, z: 0 },
    { id: "N3", x: 10, y: 0, z: 0 },
    { id: "N4", x: 10, y: 10, z: 0 },
    { id: "N5", x: 20, y: 0, z: 0 },
    { id: "N6", x: 20, y: 10, z: 0 },
    { id: "N7", x: 30, y: 0, z: 0 },
    { id: "N8", x: 30, y: 10, z: 0 },
    { id: "N9", x: 0, y: 0, z: 8 },
    { id: "N10", x: 0, y: 10, z: 8 },
    { id: "N11", x: 10, y: 0, z: 8 },
    { id: "N12", x: 10, y: 10, z: 8 },
    { id: "N13", x: 20, y: 0, z: 8 },
    { id: "N14", x: 20, y: 10, z: 8 },
    { id: "N15", x: 30, y: 0, z: 8 },
    { id: "N16", x: 30, y: 10, z: 8 },
  ],
  elements: bars([
    ["N1", "N3"], ["N3", "N5"], ["N5", "N7"], ["N2", "N4"], ["N4", "N6"], ["N6", "N8"],
    ["N9", "N11"], ["N11", "N13"], ["N13", "N15"], ["N10", "N12"], ["N12", "N14"], ["N14", "N16"],
    ["N1", "N2"], ["N3", "N4"], ["N5", "N6"], ["N7", "N8"], ["N9", "N10"], ["N11", "N12"], ["N13", "N14"], ["N15", "N16"],
    ["N1", "N9"], ["N2", "N10"], ["N3", "N11"], ["N4", "N12"], ["N5", "N13"], ["N6", "N14"], ["N7", "N15"], ["N8", "N16"],
    ["N1", "N11"], ["N3", "N9"], ["N3", "N13"], ["N5", "N11"], ["N5", "N15"], ["N7", "N13"],
    ["N2", "N12"], ["N4", "N10"], ["N4", "N14"], ["N6", "N12"], ["N6", "N16"], ["N8", "N14"],
    ["N9", "N12"], ["N10", "N11"], ["N11", "N14"], ["N12", "N13"], ["N13", "N16"], ["N14", "N15"],
  ], "timber50x100"),
  boundaries: [
    { nodeId: "N1", ux: true, uy: true, uz: true },
    { nodeId: "N2", ux: true, uy: true, uz: true },
    { nodeId: "N3", ux: true, uy: true, uz: true },
    { nodeId: "N4", ux: true, uy: true, uz: true },
    { nodeId: "N5", ux: true, uy: true, uz: true },
    { nodeId: "N6", ux: true, uy: true, uz: true },
    { nodeId: "N7", ux: true, uy: true, uz: true },
    { nodeId: "N8", ux: true, uy: true, uz: true },
  ],
  loads: [
    { nodeId: "N3", fz: -6000 },
    { nodeId: "N4", fz: -6000 },
    { nodeId: "N5", fz: -6000 },
    { nodeId: "N6", fz: -6000 },
  ],
});

const towerTrussModel = (): StructureModel => ({
  ...emptyModel(),
  nodes: [
    { id: "N1", x: 0, y: 0, z: 0 },
    { id: "N2", x: 12, y: 0, z: 0 },
    { id: "N3", x: 12, y: 12, z: 0 },
    { id: "N4", x: 0, y: 12, z: 0 },
    { id: "N5", x: 2, y: 2, z: 10 },
    { id: "N6", x: 10, y: 2, z: 10 },
    { id: "N7", x: 10, y: 10, z: 10 },
    { id: "N8", x: 2, y: 10, z: 10 },
    { id: "N9", x: 4, y: 4, z: 20 },
    { id: "N10", x: 8, y: 4, z: 20 },
    { id: "N11", x: 8, y: 8, z: 20 },
    { id: "N12", x: 4, y: 8, z: 20 },
    { id: "N13", x: 6, y: 6, z: 30 },
  ],
  elements: bars([
    ["N1", "N2"], ["N2", "N3"], ["N3", "N4"], ["N4", "N1"],
    ["N5", "N6"], ["N6", "N7"], ["N7", "N8"], ["N8", "N5"],
    ["N9", "N10"], ["N10", "N11"], ["N11", "N12"], ["N12", "N9"],
    ["N1", "N5"], ["N2", "N6"], ["N3", "N7"], ["N4", "N8"],
    ["N5", "N9"], ["N6", "N10"], ["N7", "N11"], ["N8", "N12"],
    ["N9", "N13"], ["N10", "N13"], ["N11", "N13"], ["N12", "N13"],
    ["N1", "N6"], ["N2", "N5"], ["N2", "N7"], ["N3", "N6"], ["N3", "N8"], ["N4", "N7"], ["N4", "N5"], ["N1", "N8"],
    ["N5", "N10"], ["N6", "N9"], ["N6", "N11"], ["N7", "N10"], ["N7", "N12"], ["N8", "N11"], ["N8", "N9"], ["N5", "N12"],
  ]),
  boundaries: [
    { nodeId: "N1", ux: true, uy: true, uz: true },
    { nodeId: "N2", ux: true, uy: true, uz: true },
    { nodeId: "N3", ux: true, uy: true, uz: true },
    { nodeId: "N4", ux: true, uy: true, uz: true },
  ],
  loads: [
    { nodeId: "N13", fx: 2500, fy: 1000, fz: -8000 },
  ],
  nodalMasses: [{ nodeId: "N13", mass: 50 }],
});

const cantileverTrussModel = (): StructureModel => ({
  ...emptyModel(),
  nodes: [
    { id: "N1", x: 0, y: 0, z: 0 },
    { id: "N2", x: 0, y: 8, z: 0 },
    { id: "N3", x: 0, y: 0, z: 8 },
    { id: "N4", x: 0, y: 8, z: 8 },
    { id: "N5", x: 10, y: 0, z: 0 },
    { id: "N6", x: 10, y: 8, z: 0 },
    { id: "N7", x: 10, y: 0, z: 8 },
    { id: "N8", x: 10, y: 8, z: 8 },
    { id: "N9", x: 20, y: 0, z: 0 },
    { id: "N10", x: 20, y: 8, z: 0 },
    { id: "N11", x: 20, y: 0, z: 8 },
    { id: "N12", x: 20, y: 8, z: 8 },
    { id: "N13", x: 30, y: 4, z: 4 },
  ],
  elements: bars([
    ["N1", "N5"], ["N5", "N9"], ["N2", "N6"], ["N6", "N10"], ["N3", "N7"], ["N7", "N11"], ["N4", "N8"], ["N8", "N12"],
    ["N1", "N2"], ["N2", "N4"], ["N4", "N3"], ["N3", "N1"],
    ["N5", "N6"], ["N6", "N8"], ["N8", "N7"], ["N7", "N5"],
    ["N9", "N10"], ["N10", "N12"], ["N12", "N11"], ["N11", "N9"],
    ["N1", "N6"], ["N2", "N5"], ["N3", "N8"], ["N4", "N7"], ["N1", "N7"], ["N2", "N8"], ["N3", "N5"], ["N4", "N6"],
    ["N5", "N10"], ["N6", "N9"], ["N7", "N12"], ["N8", "N11"], ["N5", "N11"], ["N6", "N12"], ["N7", "N9"], ["N8", "N10"],
    ["N9", "N13"], ["N10", "N13"], ["N11", "N13"], ["N12", "N13"],
    ["N5", "N13"], ["N6", "N13"], ["N7", "N13"], ["N8", "N13"],
  ], "timber100x200"),
  boundaries: [
    { nodeId: "N1", ux: true, uy: true, uz: true },
    { nodeId: "N2", ux: true, uy: true, uz: true },
    { nodeId: "N3", ux: true, uy: true, uz: true },
    { nodeId: "N4", ux: true, uy: true, uz: true },
  ],
  loads: [{ nodeId: "N13", fy: 2500, fz: -12000 }],
  nodalMasses: [{ nodeId: "N13", mass: 80 }],
});

const orientalPearlTrussModel = (): StructureModel => {
  const base = emptyModel();
  const nodes: StructureNode[] = [];
  const pairs: Array<[string, string]> = [];
  const pairSet = new Set<string>();
  const center = { x: 34, y: 34 };
  const addPair = (a: string, b: string) => {
    if (a === b) return;
    const key = a < b ? `${a}|${b}` : `${b}|${a}`;
    if (pairSet.has(key)) return;
    pairSet.add(key);
    pairs.push([a, b]);
  };
  const addNode = (id: string, x: number, y: number, z: number) => {
    nodes.push({ id, x: Number(x.toFixed(3)), y: Number(y.toFixed(3)), z: Number(z.toFixed(3)) });
    return id;
  };
  const makeRing = (prefix: string, z: number, radius: number, count: number, angleOffset = Math.PI / 2) => {
    const ids: string[] = [];
    for (let i = 0; i < count; i += 1) {
      const angle = angleOffset + (Math.PI * 2 * i) / count;
      ids.push(addNode(`${prefix}_${i + 1}`, center.x + Math.cos(angle) * radius, center.y + Math.sin(angle) * radius, z));
    }
    return ids;
  };
  const connectRing = (ids: string[], skip = 1) => {
    const offset = Math.max(1, Math.round(skip));
    for (let i = 0; i < ids.length; i += 1) {
      addPair(ids[i], ids[(i + offset) % ids.length]);
    }
  };
  const connectRings = (lower: string[], upper: string[], diagonalOffset = 1) => {
    const count = Math.min(lower.length, upper.length);
    for (let i = 0; i < count; i += 1) {
      addPair(lower[i], upper[i]);
      addPair(lower[i], upper[(i + diagonalOffset) % upper.length]);
      addPair(lower[i], upper[(i - diagonalOffset + upper.length) % upper.length]);
    }
  };

  const coreZ = [0, 18, 28, 40, 52, 64, 76, 88, 100, 112, 122];
  const coreRings = new Map<number, string[]>();
  for (const z of coreZ) {
    const ring = makeRing(`CORE_${z}`, z, z < 100 ? 3.2 : 2.1, 6);
    coreRings.set(z, ring);
    connectRing(ring);
    connectRing(ring, 2);
  }
  for (let i = 0; i < coreZ.length - 1; i += 1) {
    connectRings(coreRings.get(coreZ[i])!, coreRings.get(coreZ[i + 1])!);
  }

  const sphereRings: string[][] = [];
  const addSphere = (prefix: string, profile: Array<{ z: number; r: number }>, count: number) => {
    const rings = profile.map((item) => {
      const ring = makeRing(`${prefix}_${item.z}`, item.z, item.r, count);
      connectRing(ring);
      connectRing(ring, Math.max(2, count / 4));
      const coreRing = coreRings.get(item.z);
      if (coreRing) {
        for (let i = 0; i < ring.length; i += 1) {
          const coreIndex = Math.round((i * coreRing.length) / ring.length) % coreRing.length;
          addPair(ring[i], coreRing[coreIndex]);
          addPair(ring[i], coreRing[(coreIndex + 1) % coreRing.length]);
        }
      }
      const lowerCoreZ = [...coreZ].reverse().find((z) => z < item.z);
      const upperCoreZ = coreZ.find((z) => z > item.z);
      for (const adjacentZ of [lowerCoreZ, upperCoreZ]) {
        const adjacentCoreRing = adjacentZ === undefined ? undefined : coreRings.get(adjacentZ);
        if (!adjacentCoreRing) continue;
        for (let i = 0; i < ring.length; i += 1) {
          const coreIndex = Math.round((i * adjacentCoreRing.length) / ring.length) % adjacentCoreRing.length;
          addPair(ring[i], adjacentCoreRing[coreIndex]);
        }
      }
      sphereRings.push(ring);
      return ring;
    });
    for (let i = 0; i < rings.length - 1; i += 1) connectRings(rings[i], rings[i + 1], i % 2 === 0 ? 1 : count - 1);
  };

  addSphere("LOWER_BALL", [
    { z: 28, r: 7 },
    { z: 40, r: 16.5 },
    { z: 52, r: 7 },
  ], 8);
  addSphere("UPPER_BALL", [
    { z: 64, r: 5.5 },
    { z: 76, r: 10.5 },
    { z: 88, r: 5.5 },
  ], 8);
  addSphere("SPACE_CAPSULE", [
    { z: 100, r: 5.8 },
    { z: 112, r: 4 },
  ], 6);

  const legLevels = [
    { z: 0, radius: 31, width: 2.9 },
    { z: 18, radius: 23, width: 2.3 },
    { z: 36, radius: 14, width: 1.7 },
    { z: 56, radius: 7, width: 1.2 },
  ];
  const baseSupportIds: string[] = [];
  for (let leg = 0; leg < 3; leg += 1) {
    const angle = Math.PI / 2 + (Math.PI * 2 * leg) / 3;
    const radial = { x: Math.cos(angle), y: Math.sin(angle) };
    const tangent = { x: -Math.sin(angle), y: Math.cos(angle) };
    const stations: string[][] = [];
    for (let levelIndex = 0; levelIndex < legLevels.length; levelIndex += 1) {
      const level = legLevels[levelIndex];
      const cx = center.x + radial.x * level.radius;
      const cy = center.y + radial.y * level.radius;
      const ids = [
        addNode(`LEG${leg + 1}_${levelIndex + 1}_A`, cx + tangent.x * level.width, cy + tangent.y * level.width, level.z),
        addNode(`LEG${leg + 1}_${levelIndex + 1}_B`, cx - tangent.x * level.width, cy - tangent.y * level.width, level.z),
        addNode(`LEG${leg + 1}_${levelIndex + 1}_C`, cx - radial.x * level.width * 0.8, cy - radial.y * level.width * 0.8, level.z),
      ];
      stations.push(ids);
      connectRing(ids);
      if (levelIndex === 0) baseSupportIds.push(...ids);
    }
    for (let levelIndex = 0; levelIndex < stations.length - 1; levelIndex += 1) {
      connectRings(stations[levelIndex], stations[levelIndex + 1], levelIndex % 2 === 0 ? 1 : 2);
    }
    const topStation = stations[stations.length - 1];
    const core52 = coreRings.get(52)!;
    const core64 = coreRings.get(64)!;
    const coreIndex = Math.round((leg * core52.length) / 3) % core52.length;
    for (const id of topStation) {
      addPair(id, core52[coreIndex]);
      addPair(id, core64[(coreIndex + 1) % core64.length]);
    }
  }

  const mastBase = coreRings.get(100)!;
  const mast112 = coreRings.get(112)!;
  const mast122 = coreRings.get(122)!;
  const topId = addNode("TOP", center.x, center.y, 132);
  connectRings(mastBase, mast112);
  connectRings(mast112, mast122);
  for (const id of mast122) addPair(id, topId);

  return {
    ...base,
    nodes,
    elements: bars(pairs, "timber50x100"),
    boundaries: [
      ...baseSupportIds.map((nodeId) => ({ nodeId, ux: true, uy: true, uz: true })),
      ...coreRings.get(0)!.map((nodeId) => ({ nodeId, ux: true, uy: true, uz: true })),
    ],
    loads: [
      { nodeId: "TOP", fx: 2500, fy: 1500, fz: -9000 },
      ...sphereRings.flatMap((ring) => ring.filter((_, index) => index % 4 === 0).map((nodeId) => ({ nodeId, fz: -800 }))),
    ],
    nodalMasses: [
      { nodeId: "TOP", mass: 60 },
      ...sphereRings.flatMap((ring) => ring.filter((_, index) => index % 2 === 0).map((nodeId) => ({ nodeId, mass: 4 }))),
    ],
  };
};

const tongjiCivilBuildingTrussModel = (): StructureModel => {
  const base = emptyModel();
  const nodes: StructureNode[] = [];
  const pairs: Array<[string, string]> = [];
  const xValues = [0, 8, 16, 24, 32, 40];
  const yValues = [0, 8, 16];
  const zValues = [0, 6, 12, 18, 24, 30];
  const id = (xi: number, yi: number, zi: number) => `B${xi + 1}_${yi + 1}_${zi + 1}`;
  const addPair = (a: string, b: string) => {
    if (a !== b) pairs.push([a, b]);
  };

  zValues.forEach((z, zi) => {
    yValues.forEach((y, yi) => {
      xValues.forEach((x, xi) => nodes.push({ id: id(xi, yi, zi), x, y, z }));
    });
  });
  for (let zi = 0; zi < zValues.length; zi += 1) {
    for (let yi = 0; yi < yValues.length; yi += 1) {
      for (let xi = 0; xi < xValues.length - 1; xi += 1) addPair(id(xi, yi, zi), id(xi + 1, yi, zi));
    }
    for (let xi = 0; xi < xValues.length; xi += 1) {
      for (let yi = 0; yi < yValues.length - 1; yi += 1) addPair(id(xi, yi, zi), id(xi, yi + 1, zi));
    }
    for (let xi = 0; xi < xValues.length - 1; xi += 1) {
      for (let yi = 0; yi < yValues.length - 1; yi += 1) {
        addPair(id(xi, yi, zi), id(xi + 1, yi + 1, zi));
        addPair(id(xi + 1, yi, zi), id(xi, yi + 1, zi));
      }
    }
  }
  for (let zi = 0; zi < zValues.length - 1; zi += 1) {
    for (let yi = 0; yi < yValues.length; yi += 1) {
      for (let xi = 0; xi < xValues.length; xi += 1) addPair(id(xi, yi, zi), id(xi, yi, zi + 1));
    }
    for (let xi = 0; xi < xValues.length - 1; xi += 1) {
      addPair(id(xi, 0, zi), id(xi + 1, 0, zi + 1));
      addPair(id(xi + 1, 0, zi), id(xi, 0, zi + 1));
      addPair(id(xi, 2, zi), id(xi + 1, 2, zi + 1));
      addPair(id(xi + 1, 2, zi), id(xi, 2, zi + 1));
    }
    for (let yi = 0; yi < yValues.length - 1; yi += 1) {
      addPair(id(0, yi, zi), id(0, yi + 1, zi + 1));
      addPair(id(0, yi + 1, zi), id(0, yi, zi + 1));
      addPair(id(5, yi, zi), id(5, yi + 1, zi + 1));
      addPair(id(5, yi + 1, zi), id(5, yi, zi + 1));
    }
    addPair(id(2, 1, zi), id(3, 1, zi + 1));
    addPair(id(3, 1, zi), id(2, 1, zi + 1));
  }

  return {
    ...base,
    nodes,
    elements: bars(pairs, "timber50x100"),
    boundaries: nodes.filter((node) => Math.abs(node.z) < 1e-9).map((node) => ({ nodeId: node.id, ux: true, uy: true, uz: true })),
    loads: nodes
      .filter((node) => node.z > 0 && node.z % 12 === 0)
      .map((node) => ({ nodeId: node.id, fz: -900 })),
    nodalMasses: nodes
      .filter((node) => node.z === 30)
      .map((node) => ({ nodeId: node.id, mass: 5 })),
  };
};

const exampleModels: ExampleDefinition[] = [
  { key: "cube", name: "空间立方体桁架", nameEn: "Space Cube Truss", unit: "cm", gridStep: 5, selectNodeId: "N7", model: sampleModel },
  { key: "bridge", name: "三维桥式桁架", nameEn: "3D Bridge Truss", unit: "m", gridStep: 5, selectNodeId: "N5", model: bridgeTrussModel },
  { key: "tower", name: "塔架桁架", nameEn: "Tower Truss", unit: "m", gridStep: 5, selectNodeId: "N13", model: towerTrussModel },
  { key: "cantilever", name: "悬臂空间桁架", nameEn: "Cantilever Space Truss", unit: "m", gridStep: 5, selectNodeId: "N13", model: cantileverTrussModel },
  { key: "orientalPearl", name: "东方明珠简化塔桁架", nameEn: "Simplified Oriental Pearl Tower", unit: "m", gridStep: 10, selectNodeId: "TOP", model: orientalPearlTrussModel },
  { key: "tongjiCivil", name: "多层桁架结构", nameEn: "Multi-story Truss Structure", unit: "m", gridStep: 5, selectNodeId: "B3_2_6", model: tongjiCivilBuildingTrussModel },
];

function getExample(key: ExampleKey): ExampleDefinition {
  return exampleModels.find((example) => example.key === key) ?? exampleModels[0];
}

function exampleGridStep(example: ExampleDefinition): number {
  return example.gridStep ?? defaultGridStepByUnit[example.unit];
}

function activeSolveScale(model: StructureModel) {
  const activeNodeIds = new Set<string>();
  const beamNodeIds = new Set<string>();
  for (const element of model.elements) {
    activeNodeIds.add(element.startNodeId);
    activeNodeIds.add(element.endNodeId);
    if (element.type === "beam3d") {
      beamNodeIds.add(element.startNodeId);
      beamNodeIds.add(element.endNodeId);
    }
  }
  return {
    activeNodes: activeNodeIds.size,
    elements: model.elements.length,
    beamElements: model.elements.filter((element) => element.type === "beam3d").length,
    weightedDofs: activeNodeIds.size * 3 + beamNodeIds.size * 3,
  };
}

function solveScaleError(model: StructureModel, kind: SolveKind, language: LanguageKey = "zh"): string | null {
  const scale = activeSolveScale(model);
  const limit = solveScaleLimits[kind];
  const exceeded: string[] = [];
  if (scale.activeNodes > limit.activeNodes) exceeded.push(`${language === "en" ? "active nodes" : "活跃节点"} ${scale.activeNodes}/${limit.activeNodes}`);
  if (scale.elements > limit.elements) exceeded.push(`${language === "en" ? "elements" : "单元"} ${scale.elements}/${limit.elements}`);
  if (scale.weightedDofs > limit.weightedDofs) exceeded.push(`${language === "en" ? "estimated DOFs" : "估算自由度"} ${scale.weightedDofs}/${limit.weightedDofs}`);
  if (exceeded.length === 0) return null;
  if (language === "en") {
    const title = kind === "modal" ? "Modal analysis" : "Static analysis";
    const suggestion = kind === "modal"
      ? "Reduce the requested mode count, simplify repeated grids/spherical subdivisions, or analyze the structure in smaller parts. Modal analysis is more expensive than static analysis."
      : "Remove inactive members, simplify repeated grids/spherical subdivisions, or analyze the structure in smaller parts.";
    return `${title} stopped: the current model is larger than the recommended interactive browser limit (${exceeded.join(", ")}; beam elements ${scale.beamElements}). To avoid a long browser freeze, matrix assembly was not started. ${suggestion}`;
  }
  const title = kind === "modal" ? "频率求解" : "静力求解";
  const suggestion = kind === "modal"
    ? "建议先减少模态阶数、简化重复网格/球体细分，或分段建立模型后再计算；频率分析比静力分析更消耗计算量。"
    : "建议删除不参与工作的构件、简化重复网格/球体细分，或分段建立模型后再计算。";
  return `${title}已停止：当前模型规模超过浏览器端交互求解的建议范围（${exceeded.join("，")}；梁单元 ${scale.beamElements}）。为避免页面长时间无响应，系统没有继续装配矩阵。${suggestion}`;
}

function format(value: number | undefined, digits = 3): string {
  if (value === undefined) return "-";
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.01) return value.toExponential(2);
  return value.toFixed(digits);
}

function momentUnit(unit: UnitKey): string {
  return `N·${unitLabels[unit]}`;
}

function formatDisplacement(valueMeters: number, unit: UnitKey): string {
  return `${format(valueMeters / unitScale[unit], 6)} ${unitLabels[unit]}`;
}

function formatMoment(value: number | undefined, unit: UnitKey): string {
  return format(value === undefined ? undefined : value / unitScale[unit]);
}

function formatMomentPair(start: number | undefined, end: number | undefined, unit: UnitKey): string {
  if (start === undefined && end === undefined) return "-";
  return `${formatMoment(start, unit)} / ${formatMoment(end, unit)}`;
}

function toThree(node: StructureNode, modelScale: number, displacement?: Record<DofKey, number>, displacementScale = 1): THREE.Vector3 {
  const dx = displacement?.ux ?? 0;
  const dy = displacement?.uy ?? 0;
  const dz = displacement?.uz ?? 0;
  return new THREE.Vector3(
    node.x * modelScale + dx * displacementScale,
    node.z * modelScale + dz * displacementScale,
    node.y * modelScale + dy * displacementScale,
  );
}

function createMemberMesh(start: THREE.Vector3, end: THREE.Vector3, radius: number, color: string): THREE.Mesh {
  const direction = end.clone().sub(start);
  const length = direction.length();
  const geometry = new THREE.CylinderGeometry(radius, radius, length, 10, 1);
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness: 0.05 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  return mesh;
}

function createDisplayMember(start: THREE.Vector3, end: THREE.Vector3, radius: number, color: string, opacity = 1): THREE.Mesh {
  const mesh = createMemberMesh(start, end, radius, color);
  const material = mesh.material as THREE.MeshStandardMaterial;
  material.transparent = opacity < 1;
  material.opacity = opacity;
  material.depthWrite = opacity >= 1;
  return mesh;
}

function disposeObject(object: THREE.Object3D): void {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
    mesh.geometry?.dispose();
    if (Array.isArray(material)) {
      material.forEach((item) => item.dispose());
    } else {
      material?.dispose();
    }
  });
}

function forceIsAxialOnly(force: ElementForce): boolean {
  const momentScale = Math.max(1, Math.abs(force.axial));
  const maxMoment = Math.max(
    Math.abs(force.momentYStart ?? 0),
    Math.abs(force.momentYEnd ?? 0),
    Math.abs(force.momentZStart ?? 0),
    Math.abs(force.momentZEnd ?? 0),
  );
  return maxMoment / momentScale < 1e-6;
}

function forceStatus(force: ElementForce, language: LanguageKey = "zh"): string {
  if (!forceIsAxialOnly(force)) return language === "en" ? "Beam" : "梁";
  if (force.axial > 1e-6) return language === "en" ? "Tension" : "拉杆";
  if (force.axial < -1e-6) return language === "en" ? "Compression" : "压杆";
  return language === "en" ? "Bar" : "杆";
}

function nodeDistance(a: StructureNode, b: StructureNode): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function vecSub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function vecCross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function vecLength(vector: Vec3): number {
  return Math.hypot(vector.x, vector.y, vector.z);
}

function vecNormalize(vector: Vec3): Vec3 {
  const value = vecLength(vector);
  if (value < 1e-12) return { x: 0, y: 0, z: 0 };
  return { x: vector.x / value, y: vector.y / value, z: vector.z / value };
}

function elementAxesForDisplay(start: StructureNode, end: StructureNode, preferredY?: Vec3): { ex: Vec3; ey: Vec3; ez: Vec3 } {
  const ex = vecNormalize(vecSub(end, start));
  const reference = preferredY ?? (Math.abs(ex.z) < 0.9 ? { x: 0, y: 0, z: 1 } : { x: 0, y: 1, z: 0 });
  let ez = vecCross(ex, reference);
  if (vecLength(ez) < 1e-9) ez = vecCross(ex, { x: 0, y: 1, z: 0 });
  ez = vecNormalize(ez);
  const ey = vecNormalize(vecCross(ez, ex));
  return { ex, ey, ez };
}

function localToGlobalVector(axes: { ex: Vec3; ey: Vec3; ez: Vec3 }, vector: Vec3): Vec3 {
  return {
    x: axes.ex.x * vector.x + axes.ey.x * vector.y + axes.ez.x * vector.z,
    y: axes.ex.y * vector.x + axes.ey.y * vector.y + axes.ez.y * vector.z,
    z: axes.ex.z * vector.x + axes.ey.z * vector.y + axes.ez.z * vector.z,
  };
}

function interpolateNode(start: StructureNode, end: StructureNode, ratio: number): StructureNode {
  return {
    id: "",
    x: start.x + (end.x - start.x) * ratio,
    y: start.y + (end.y - start.y) * ratio,
    z: start.z + (end.z - start.z) * ratio,
  };
}

function nextId(prefix: string, existingIds: string[]): string {
  const used = new Set(existingIds);
  let index = existingIds.length + 1;
  while (used.has(`${prefix}${index}`)) index += 1;
  return `${prefix}${index}`;
}

function slugId(text: string, fallback: string, existingIds: string[]): string {
  const base = (text.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || fallback).slice(0, 32);
  const used = new Set(existingIds);
  if (!used.has(base)) return base;
  let index = 2;
  while (used.has(`${base}-${index}`)) index += 1;
  return `${base}-${index}`;
}

function samePoint(a: Vec3, b: Vec3): boolean {
  return Math.abs(a.x - b.x) < 1e-6 && Math.abs(a.y - b.y) < 1e-6 && Math.abs(a.z - b.z) < 1e-6;
}

function pointOnSegmentRatio(point: Vec3, start: Vec3, end: Vec3): number | null {
  const ab = vecSub(end, start);
  const ap = vecSub(point, start);
  const lengthSquared = ab.x * ab.x + ab.y * ab.y + ab.z * ab.z;
  if (lengthSquared < 1e-12) return null;
  const ratio = (ap.x * ab.x + ap.y * ab.y + ap.z * ab.z) / lengthSquared;
  if (ratio <= 1e-6 || ratio >= 1 - 1e-6) return null;
  const projection = {
    x: start.x + ab.x * ratio,
    y: start.y + ab.y * ratio,
    z: start.z + ab.z * ratio,
  };
  const tolerance = Math.max(1e-6, Math.sqrt(lengthSquared) * 1e-6);
  return nodeDistance(point as StructureNode, projection as StructureNode) <= tolerance ? ratio : null;
}

function splitElementsAtNode(model: StructureModel, node: StructureNode): { model: StructureModel; splitCount: number; droppedLoadCount: number } {
  const nodeMap = new Map(model.nodes.map((item) => [item.id, item]));
  const usedElementIds = model.elements.map((element) => element.id);
  const splitElementIds = new Set<string>();
  const elements: StructureModel["elements"] = [];

  for (const element of model.elements) {
    if (element.startNodeId === node.id || element.endNodeId === node.id) {
      elements.push(element);
      continue;
    }
    const start = nodeMap.get(element.startNodeId);
    const end = nodeMap.get(element.endNodeId);
    if (!start || !end || pointOnSegmentRatio(node, start, end) === null) {
      elements.push(element);
      continue;
    }

    const nextElementId = nextId("E", usedElementIds);
    usedElementIds.push(nextElementId);
    splitElementIds.add(element.id);
    elements.push(
      { ...element, endNodeId: node.id, releaseEnd: undefined },
      { ...element, id: nextElementId, startNodeId: node.id, releaseStart: undefined },
    );
  }

  if (splitElementIds.size === 0) return { model, splitCount: 0, droppedLoadCount: 0 };
  const elementLoads = (model.elementLoads ?? []).filter((load) => !splitElementIds.has(load.elementId));
  return {
    model: { ...model, elements, elementLoads },
    splitCount: splitElementIds.size,
    droppedLoadCount: (model.elementLoads ?? []).length - elementLoads.length,
  };
}

function insertNodeAndSplit(model: StructureModel, node: StructureNode, fixedOnBasePlane: boolean): { model: StructureModel; splitCount: number; droppedLoadCount: number } {
  const exists = model.nodes.some((item) => item.id === node.id);
  const nextModel: StructureModel = {
    ...model,
    nodes: exists ? model.nodes : [...model.nodes, node],
    boundaries: !exists && fixedOnBasePlane
      ? [...model.boundaries, { nodeId: node.id, ux: true, uy: true, uz: true }]
      : model.boundaries,
  };
  return splitElementsAtNode(nextModel, node);
}

function deleteElementFromModel(model: StructureModel, elementId: string): StructureModel {
  return {
    ...model,
    elements: model.elements.filter((element) => element.id !== elementId),
    elementLoads: (model.elementLoads ?? []).filter((load) => load.elementId !== elementId),
  };
}

function deleteNodeFromModel(model: StructureModel, nodeId: string): StructureModel {
  const removedElementIds = new Set(model.elements
    .filter((element) => element.startNodeId === nodeId || element.endNodeId === nodeId)
    .map((element) => element.id));
  return {
    ...model,
    nodes: model.nodes.filter((node) => node.id !== nodeId),
    elements: model.elements.filter((element) => !removedElementIds.has(element.id)),
    boundaries: model.boundaries.filter((boundary) => boundary.nodeId !== nodeId),
    loads: model.loads.filter((load) => load.nodeId !== nodeId),
    nodalMasses: (model.nodalMasses ?? []).filter((mass) => mass.nodeId !== nodeId),
    elementLoads: (model.elementLoads ?? []).filter((load) => !removedElementIds.has(load.elementId)),
  };
}

function isUnitKey(value: unknown): value is UnitKey {
  return value === "m" || value === "cm" || value === "mm";
}

function isStructureModel(value: unknown): value is StructureModel {
  if (!value || typeof value !== "object") return false;
  const model = value as Partial<StructureModel>;
  return Array.isArray(model.nodes) &&
    Array.isArray(model.elements) &&
    Array.isArray(model.materials) &&
    Array.isArray(model.sections) &&
    Array.isArray(model.boundaries) &&
    Array.isArray(model.loads);
}

function toFiniteNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function toVec3(value: unknown, fallback: Vec3): Vec3 {
  if (!value || typeof value !== "object") return fallback;
  const vector = value as Partial<Vec3>;
  return {
    x: toFiniteNumber(vector.x, fallback.x),
    y: toFiniteNumber(vector.y, fallback.y),
    z: toFiniteNumber(vector.z, fallback.z),
  };
}

function toOffset(value: unknown, fallback: { dx: number; dy: number; dz: number }): { dx: number; dy: number; dz: number } {
  if (!value || typeof value !== "object") return fallback;
  const vector = value as Partial<{ dx: number; dy: number; dz: number }>;
  return {
    dx: toFiniteNumber(vector.dx, fallback.dx),
    dy: toFiniteNumber(vector.dy, fallback.dy),
    dz: toFiniteNumber(vector.dz, fallback.dz),
  };
}

function scaleBoundaryValues(boundary: BoundaryCondition, factor: number): BoundaryCondition {
  if (!boundary.values) return boundary;
  return {
    ...boundary,
    values: {
      ...boundary.values,
      ux: boundary.values.ux === undefined ? undefined : Number((boundary.values.ux * factor).toFixed(6)),
      uy: boundary.values.uy === undefined ? undefined : Number((boundary.values.uy * factor).toFixed(6)),
      uz: boundary.values.uz === undefined ? undefined : Number((boundary.values.uz * factor).toFixed(6)),
    },
  };
}

function isLoadCoordinate(value: unknown): value is LoadCoordinate {
  return value === "global" || value === "local";
}

function isCompleteNumberInput(value: string): boolean {
  const text = value.trim();
  if (text === "" || text === "-" || text === "+" || text === "." || text === "-." || text === "+.") return false;
  return Number.isFinite(Number(text));
}

function convertModelToMeters(model: StructureModel, scale: number): StructureModel {
  if (scale === 1) return model;
  return {
    ...model,
    nodes: model.nodes.map((node) => ({
      ...node,
      x: node.x * scale,
      y: node.y * scale,
      z: node.z * scale,
    })),
    boundaries: model.boundaries.map((boundary) => scaleBoundaryValues(boundary, scale)),
    materials: model.materials.map((material) => ({ ...material, density: material.density ?? 0 })),
    elementLoads: (model.elementLoads ?? []).map((load) => load.type === "distributed"
      ? {
          ...load,
          wx: load.wx === undefined ? undefined : load.wx / scale,
          wy: load.wy === undefined ? undefined : load.wy / scale,
          wz: load.wz === undefined ? undefined : load.wz / scale,
        }
      : load),
    nodalMasses: model.nodalMasses ?? [],
  };
}

function cloneModel(model: StructureModel): StructureModel {
  return JSON.parse(JSON.stringify(model)) as StructureModel;
}

function App() {
  const [model, setModel] = useState<StructureModel>(() => sampleModel());
  const [tool, setTool] = useState<Tool>("member");
  const [selectedExample, setSelectedExample] = useState<ExampleKey>("cube");
  const [pendingNode, setPendingNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>("N7");
  const [selectedElement, setSelectedElement] = useState<string | null>(null);
  const [loadZ, setLoadZ] = useState(-15000);
  const [loadDraft, setLoadDraft] = useState({ fx: 0, fy: 0, fz: -15000 });
  const [massDraft, setMassDraft] = useState(0);
  const [elementLoadDraft, setElementLoadDraft] = useState<ElementLoadDraft>({
    type: "point",
    coordinate: "global",
    position: 0.5,
    x: 0,
    y: 0,
    z: -1000,
  });
  const [offset, setOffset] = useState({ dx: 5, dy: 0, dz: 0 });
  const [coordinateNode, setCoordinateNode] = useState({ x: 0, y: 0, z: 0 });
  const [gridStep, setGridStep] = useState(defaultGridStepByUnit.cm);
  const [gridVisible, setGridVisible] = useState(true);
  const [unit, setUnit] = useState<UnitKey>("cm");
  const [defaultMaterialId, setDefaultMaterialId] = useState("wood");
  const [defaultSectionId, setDefaultSectionId] = useState("timber100");
  const [sectionNameDraft, setSectionNameDraft] = useState("New section");
  const [materialOpen, setMaterialOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [language, setLanguage] = useState<LanguageKey>("zh");
  const [result, setResult] = useState<SolveResult | null>(null);
  const [modalResult, setModalResult] = useState<ModalResult | null>(null);
  const [resultPopupOpen, setResultPopupOpen] = useState(false);
  const [resultPopupPosition, setResultPopupPosition] = useState({ x: 14, y: 0 });
  const [error, setError] = useState<string | null>(null);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [activeMode, setActiveMode] = useState(1);
  const [modalAnimate, setModalAnimate] = useState(false);
  const [modalColor, setModalColor] = useState("#ff2d2d");
  const [staticDeformScale, setStaticDeformScale] = useState(1);
  const [modalDeformScale, setModalDeformScale] = useState(1);
  const [modalModeCount, setModalModeCount] = useState(8);
  const mountRef = useRef<HTMLDivElement | null>(null);
  const projectInputRef = useRef<HTMLInputElement | null>(null);
  const modelRef = useRef(model);
  const toolRef = useRef(tool);
  const pendingRef = useRef(pendingNode);
  const selectedRef = useRef(selectedNode);
  const selectedElementRef = useRef(selectedElement);
  const defaultMaterialRef = useRef(defaultMaterialId);
  const defaultSectionRef = useRef(defaultSectionId);
  const unitRef = useRef(unit);
  const resultRef = useRef(result);
  const modalResultRef = useRef(modalResult);
  const activeModeRef = useRef(activeMode);
  const modalAnimateRef = useRef(modalAnimate);
  const modalColorRef = useRef(modalColor);
  const languageRef = useRef(language);
  const staticDeformScaleRef = useRef(staticDeformScale);
  const modalDeformScaleRef = useRef(modalDeformScale);
  const sceneDirtyRef = useRef(true);
  const historyRef = useRef<StructureModel[]>([]);
  const resultPopupDragRef = useRef<{ startX: number; startY: number; originX: number; originY: number } | null>(null);
  const [undoCount, setUndoCount] = useState(0);

  useEffect(() => { modelRef.current = model; }, [model]);
  useEffect(() => { toolRef.current = tool; }, [tool]);
  useEffect(() => { pendingRef.current = pendingNode; }, [pendingNode]);
  useEffect(() => { selectedRef.current = selectedNode; }, [selectedNode]);
  useEffect(() => { selectedElementRef.current = selectedElement; }, [selectedElement]);
  useEffect(() => { defaultMaterialRef.current = defaultMaterialId; }, [defaultMaterialId]);
  useEffect(() => { defaultSectionRef.current = defaultSectionId; }, [defaultSectionId]);
  useEffect(() => { unitRef.current = unit; }, [unit]);
  useEffect(() => { resultRef.current = result; }, [result]);
  useEffect(() => { modalResultRef.current = modalResult; }, [modalResult]);
  useEffect(() => { activeModeRef.current = activeMode; }, [activeMode]);
  useEffect(() => { modalAnimateRef.current = modalAnimate; }, [modalAnimate]);
  useEffect(() => { modalColorRef.current = modalColor; }, [modalColor]);
  useEffect(() => { languageRef.current = language; }, [language]);
  useEffect(() => { staticDeformScaleRef.current = staticDeformScale; }, [staticDeformScale]);
  useEffect(() => { modalDeformScaleRef.current = modalDeformScale; }, [modalDeformScale]);
  useEffect(() => { sceneDirtyRef.current = true; }, [model, result, modalResult, activeMode, modalAnimate, modalColor, staticDeformScale, modalDeformScale, selectedNode, selectedElement]);
  useEffect(() => { setModalResult(null); setActiveMode(1); setModalAnimate(false); }, [model]);

  const selectedLoad = useMemo(() => model.loads.find((load) => load.nodeId === selectedNode), [model.loads, selectedNode]);
  const selectedMass = useMemo(() => (model.nodalMasses ?? []).find((mass) => mass.nodeId === selectedNode), [model.nodalMasses, selectedNode]);
  const selectedBoundary = useMemo(() => model.boundaries.find((boundary) => boundary.nodeId === selectedNode), [model.boundaries, selectedNode]);
  const selectedElementData = useMemo(() => model.elements.find((element) => element.id === selectedElement), [model.elements, selectedElement]);
  const selectedElementLoads = useMemo(() => (model.elementLoads ?? []).filter((load) => load.elementId === selectedElement), [model.elementLoads, selectedElement]);
  const selectedElementKind = useMemo(() => {
    if (!selectedElementData) return null;
    return elementVisualKind(selectedElementData, new Map(model.nodes.map((node) => [node.id, node])));
  }, [model.nodes, selectedElementData]);
  const activeMaterialId = selectedElementData?.materialId ?? defaultMaterialId;
  const activeSectionId = selectedElementData?.sectionId ?? defaultSectionId;
  const activeMaterial = useMemo(() => model.materials.find((material) => material.id === activeMaterialId) ?? model.materials[0] ?? defaultMaterials[0], [model.materials, activeMaterialId]);
  const activeSection = useMemo(() => model.sections.find((section) => section.id === activeSectionId) ?? model.sections[0] ?? defaultSections[0], [model.sections, activeSectionId]);
  const currentUnitScale = unitScale[unit];
  const gridBounds = useMemo(() => getGridBounds(model.nodes, gridStep), [model.nodes, gridStep]);
  const en = language === "en";
  const text = {
    subtitle: en ? "3D frame/truss modeling and solving" : "3D 杆系优先 / 梁单元求解",
    saveProject: en ? "Save project" : "保存项目",
    loadProject: en ? "Load project" : "读取项目",
    undo: en ? "Undo Ctrl+Z" : "撤销 Ctrl+Z",
    loadSelectedExample: en ? "Load selected example" : "载入选中算例",
    member: en ? "Member" : "杆件",
    select: en ? "Select" : "选择",
    support: en ? "Constraints" : "约束",
    load: en ? "Loads" : "荷载",
    pickMember: en ? "Pick two points to create a bar member" : "拾取两点新建杆单元",
    selectEdit: en ? "Select and edit" : "选择与编辑",
    nodes: en ? "Nodes" : "节点",
    elements: en ? "Elements" : "单元",
    selected: en ? "Selected" : "选中",
    deleting: en ? "Deleting" : "删除模式",
    start: en ? "Start" : "起点",
    pickGrid: en ? "Pick spatial grid" : "拾取空间 grid",
    examples: en ? "Examples" : "典型模型",
    example: en ? "Example" : "算例",
    loadExample: en ? "Load example" : "加载模型",
    noSelection: en ? "No selection" : "未选择",
    node: en ? "Node" : "节点",
    bar: en ? "Bar" : "杆",
    beam: en ? "Beam" : "梁",
    hingedBeam: en ? "Hinged beam" : "铰接梁",
    barElement: en ? "Bar element" : "杆单元",
    beamElement: en ? "Beam element" : "梁单元",
    autoClassify: en ? "Auto classify elements" : "自动判定单元",
    deleteSelected: en ? "Delete" : "删除",
    deleteMode: en ? "Click nodes or members to delete; press Esc or Delete again to stop." : "点击节点或梁杆连续删除；按 Esc 或再次点击删除退出。",
    newProject: en ? "New project" : "新建项目",
    beamLoad: en ? "Beam load" : "梁荷载",
    point: en ? "Point" : "集中",
    distributed: en ? "Distributed" : "均布",
    global: en ? "Global" : "全局",
    local: en ? "Local" : "局部",
    addBeamLoad: en ? "Add beam load" : "添加梁荷载",
    beamLoadHint: en ? "Adding a beam load makes this member participate as a beam element." : "添加梁荷载后，该单元会按梁单元参与求解。",
    deleteBeamLoad: en ? "Delete beam load" : "删除梁荷载",
    unitSystem: en ? "Units" : "单位系统",
    materialSection: en ? "Materials & sections" : "材料与截面",
    material: en ? "Material" : "材料",
    sectionTag: en ? "Section tag" : "截面标签",
    sectionHint: en ? "Without a selected element, this choice becomes the default for new members." : "未选中单元时，选择会作为新建杆件默认属性",
    editing: en ? "Editing" : "正在编辑",
    newTag: en ? "New tag" : "新标签",
    addSectionTag: en ? "Add section tag" : "新增截面标签",
    spatialGrid: en ? "Spatial Grid" : "空间 Grid",
    hideGrid: en ? "Hide spatial Grid" : "隐藏空间 Grid",
    showGrid: en ? "Show spatial Grid" : "显示空间 Grid",
    spacing: en ? "Spacing" : "间距",
    numericCreate: en ? "Create by coordinates" : "数值新建",
    absoluteCoordinates: en ? "Absolute coordinates" : "绝对坐标",
    createNode: en ? "Create node" : "新建节点",
    relativeToSelected: en ? "Relative to selected node" : "相对选中节点",
    createAndConnect: en ? "Create and connect" : "新建并连接",
    nodeConditions: en ? "Node conditions" : "节点条件",
    hinged: en ? "Hinged" : "铰接",
    rigid: en ? "Rigid" : "刚接",
    confirmLoad: en ? "Confirm load" : "确认荷载",
    confirmMass: en ? "Confirm mass" : "确认质点",
    deleteMass: en ? "Delete mass" : "删除质点",
    display: en ? "Display" : "求解显示",
    staticScale: en ? "Static scale" : "静力scale",
    modalScale: en ? "Modal scale" : "模态scale",
    modalCount: en ? "Mode count" : "模态阶数",
    solve: en ? "Solve" : "求解",
    frequency: en ? "Frequency" : "频率",
    clear: en ? "Clear" : "清空",
    close: en ? "Close" : "关闭",
    resultTable: en ? "Result table" : "结果表",
    results: en ? "Results" : "结果",
    maxDisplacement: en ? "Max displacement" : "最大位移",
    forceUnit: en ? "Force" : "力",
    moment: en ? "Moment" : "弯矩",
    element: en ? "Element" : "单元",
    status: en ? "Status" : "状态",
    modal: en ? "Modes" : "模态",
    modalDescription: en ? "Thin gray lines show the original shape; colored lines show the normalized mode shape." : "灰色细线为原始构型，彩色线为归一化振型。",
    stopAnimation: en ? "Stop animation" : "停止动态",
    animate: en ? "Animate" : "动态显示",
    lineColor: en ? "Line color" : "线形颜色",
    mode: en ? "Mode" : "阶",
    helpGuide: en ? "FrameSolve 3D Product Guide" : "FrameSolve 3D 产品 Guide",
    productGuide: en ? "Product Guide" : "产品 Guide",
    authorInfo: en ? "Author" : "作者信息",
    affiliation: en ? "Affiliation" : "单位",
    title: en ? "Title" : "职称",
    associateProfessor: en ? "Associate Professor, College of Civil Engineering" : "土木学院副教授",
    author: en ? "Author" : "作者",
    email: en ? "Email" : "邮箱",
    scholar: en ? "Google Scholar" : "谷歌学术",
    language: en ? "Language" : "语言",
    chinese: "中文",
    english: "English",
  };

  useEffect(() => {
    setLoadDraft({
      fx: selectedLoad?.fx ?? 0,
      fy: selectedLoad?.fy ?? 0,
      fz: selectedLoad?.fz ?? loadZ,
    });
  }, [selectedLoad, selectedNode, loadZ]);

  useEffect(() => {
    setMassDraft(selectedMass?.mass ?? 0);
  }, [selectedMass, selectedNode]);

  const saveHistory = () => {
    historyRef.current = [...historyRef.current.slice(-49), cloneModel(modelRef.current)];
    setUndoCount(historyRef.current.length);
  };

  const undo = useCallback(() => {
    const previous = historyRef.current.pop();
    if (!previous) return;
    setModel(previous);
    setPendingNode(null);
    setSelectedElement(null);
    setResult(null);
    setError(null);
    setUndoCount(historyRef.current.length);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedNode(null);
    setSelectedElement(null);
    setPendingNode(null);
    setHoverInfo(null);
    setTool((current) => current === "delete" ? "select" : current);
  }, []);

  const startResultPopupDrag = (event: React.PointerEvent<HTMLElement>) => {
    const target = event.target as HTMLElement;
    if (target.closest("button")) return;
    const popup = event.currentTarget.closest(".resultPopup");
    const rect = popup?.getBoundingClientRect();
    resultPopupDragRef.current = {
      startX: event.clientX,
      startY: event.clientY,
      originX: rect?.left ?? resultPopupPosition.x,
      originY: rect?.top ?? resultPopupPosition.y,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const dragResultPopup = (event: React.PointerEvent<HTMLElement>) => {
    const drag = resultPopupDragRef.current;
    if (!drag) return;
    const popup = event.currentTarget.closest(".resultPopup");
    const rect = popup?.getBoundingClientRect();
    const maxX = Math.max(0, window.innerWidth - (rect?.width ?? 320));
    const maxY = Math.max(0, window.innerHeight - (rect?.height ?? 160));
    setResultPopupPosition({
      x: Math.max(0, Math.min(maxX, drag.originX + event.clientX - drag.startX)),
      y: Math.max(0, Math.min(maxY, drag.originY + event.clientY - drag.startY)),
    });
  };

  const stopResultPopupDrag = (event: React.PointerEvent<HTMLElement>) => {
    if (!resultPopupDragRef.current) return;
    resultPopupDragRef.current = null;
    event.currentTarget.releasePointerCapture(event.pointerId);
  };

  useEffect(() => {
    const move = (event: PointerEvent) => {
      const drag = resultPopupDragRef.current;
      if (!drag) return;
      const popup = document.querySelector(".resultPopup");
      const rect = popup?.getBoundingClientRect();
      const maxX = Math.max(0, window.innerWidth - (rect?.width ?? 320));
      const maxY = Math.max(0, window.innerHeight - (rect?.height ?? 160));
      setResultPopupPosition({
        x: Math.max(0, Math.min(maxX, drag.originX + event.clientX - drag.startX)),
        y: Math.max(0, Math.min(maxY, drag.originY + event.clientY - drag.startY)),
      });
    };
    const stop = () => {
      resultPopupDragRef.current = null;
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isEditable = target?.tagName === "INPUT" || target?.tagName === "TEXTAREA" || target?.tagName === "SELECT" || target?.isContentEditable;
      if (event.key === "Escape") {
        event.preventDefault();
        if (isEditable) target?.blur();
        clearSelection();
        return;
      }
      if (isEditable) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && !event.shiftKey) {
        event.preventDefault();
        undo();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [clearSelection, undo]);

  const runSolve = () => {
    const scaleError = solveScaleError(model, "static", language);
    if (scaleError) {
      setResult(null);
      setModalResult(null);
      setModalAnimate(false);
      setError(scaleError);
      return;
    }
    try {
      const solved = solveStructure(convertModelToMeters(model, currentUnitScale));
      setResult(solved);
      setResultPopupOpen(false);
      setModalResult(null);
      setActiveMode(1);
      setModalAnimate(false);
      setError(null);
    } catch (solveError) {
      setResult(null);
      setModalResult(null);
      setError(solveError instanceof Error ? solveError.message : (en ? "Solve failed." : "求解失败。"));
    }
  };

  const runModalAnalysis = () => {
    const scaleError = solveScaleError(model, "modal", language);
    if (scaleError) {
      setResult(null);
      setModalResult(null);
      setModalAnimate(false);
      setError(scaleError);
      return;
    }
    try {
      const modeCount = Math.max(1, Math.min(24, Math.round(modalModeCount)));
      const solved = solveModalAnalysis(convertModelToMeters(model, currentUnitScale), modeCount);
      setModalResult(solved);
      setResultPopupOpen(true);
      setActiveMode(solved.modes[0]?.mode ?? 1);
      setModalAnimate(false);
      setResult(null);
      setError(null);
    } catch (solveError) {
      setModalResult(null);
      setError(solveError instanceof Error ? solveError.message : (en ? "Eigenvalue solve failed." : "特征值求解失败。"));
    }
  };

  const loadExampleModel = (key = selectedExample) => {
    const example = getExample(key);
    const nextModel = example.model();
    const nextGridStep = exampleGridStep(example);
    saveHistory();
    setModel(nextModel);
    setSelectedExample(example.key);
    setSelectedNode(nextModel.nodes.some((node) => node.id === example.selectNodeId) ? example.selectNodeId : nextModel.nodes[0]?.id ?? null);
    setSelectedElement(null);
    setPendingNode(null);
    setTool("member");
    setUnit(example.unit);
    setDefaultMaterialId("wood");
    setDefaultSectionId("timber100");
    setGridStep(nextGridStep);
    setGridVisible(true);
    setOffset({ dx: nextGridStep, dy: 0, dz: 0 });
    setCoordinateNode({ x: 0, y: 0, z: 0 });
    setLoadZ(-15000);
    setLoadDraft({ fx: 0, fy: 0, fz: -15000 });
    setResult(null);
    setModalResult(null);
    setActiveMode(1);
    setModalAnimate(false);
    setError(null);
  };

  const changeUnit = (nextUnit: UnitKey) => {
    if (nextUnit === unit) return;
    const factor = unitScale[unit] / unitScale[nextUnit];
    saveHistory();
    setModel((current) => ({
      ...current,
      nodes: current.nodes.map((node) => ({
        ...node,
        x: Number((node.x * factor).toFixed(6)),
        y: Number((node.y * factor).toFixed(6)),
        z: Number((node.z * factor).toFixed(6)),
      })),
      boundaries: current.boundaries.map((boundary) => scaleBoundaryValues(boundary, factor)),
      elementLoads: (current.elementLoads ?? []).map((load) => load.type === "distributed"
        ? {
            ...load,
            wx: load.wx === undefined ? undefined : Number((load.wx / factor).toFixed(6)),
            wy: load.wy === undefined ? undefined : Number((load.wy / factor).toFixed(6)),
            wz: load.wz === undefined ? undefined : Number((load.wz / factor).toFixed(6)),
          }
        : load),
      nodalMasses: current.nodalMasses ?? [],
    }));
    setGridStep(defaultGridStepByUnit[nextUnit]);
    setOffset((value) => ({
      dx: Number((value.dx * factor).toFixed(6)),
      dy: Number((value.dy * factor).toFixed(6)),
      dz: Number((value.dz * factor).toFixed(6)),
    }));
    setCoordinateNode((value) => ({
      x: Number((value.x * factor).toFixed(6)),
      y: Number((value.y * factor).toFixed(6)),
      z: Number((value.z * factor).toFixed(6)),
    }));
    setUnit(nextUnit);
    setResult(null);
    setError(null);
  };

  useEffect(() => {
    if (!mountRef.current) return;
    const mount = mountRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#f6f8fb");
    const sceneCenter = new THREE.Vector3(
      (gridBounds.xMin + gridBounds.xMax) * currentUnitScale / 2,
      (gridBounds.zMin + gridBounds.zMax) * currentUnitScale / 2,
      (gridBounds.yMin + gridBounds.yMax) * currentUnitScale / 2,
    );
    const sceneSize = Math.max(
      (gridBounds.xMax - gridBounds.xMin) * currentUnitScale,
      (gridBounds.yMax - gridBounds.yMin) * currentUnitScale,
      (gridBounds.zMax - gridBounds.zMin) * currentUnitScale,
      gridStep * currentUnitScale * 4,
      0.001,
    );
    const symbolScale = Math.max(gridStep * currentUnitScale, 0.0001);
    const camera = new THREE.PerspectiveCamera(45, mount.clientWidth / mount.clientHeight, Math.max(sceneSize / 1000, 0.00001), Math.max(sceneSize * 200, 10));
    camera.position.copy(sceneCenter).add(new THREE.Vector3(1.45 * sceneSize, 1.2 * sceneSize, 1.55 * sceneSize));
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(sceneCenter);
    controls.enableDamping = true;

    scene.add(new THREE.HemisphereLight("#ffffff", "#b8c1d1", 2.4));
    const key = new THREE.DirectionalLight("#ffffff", 1.2);
    key.position.set(8, 10, 6);
    scene.add(key);
    const xValues = gridValues(gridBounds.xMin, gridBounds.xMax, gridStep);
    const yValues = gridValues(gridBounds.yMin, gridBounds.yMax, gridStep);
    const zValues = gridValues(gridBounds.zMin, gridBounds.zMax, gridStep);
    const gridRenderStride = Math.max(1, Math.ceil(Math.max(xValues.length, yValues.length, zValues.length) / 20));
    const visibleXValues = xValues.filter((_, index) => index % gridRenderStride === 0);
    const visibleYValues = yValues.filter((_, index) => index % gridRenderStride === 0);
    const visibleZValues = zValues.filter((_, index) => index % gridRenderStride === 0);
    const visibleGridStep = gridStep * gridRenderStride * currentUnitScale;
    const floorSize = Math.max(gridBounds.xMax - gridBounds.xMin, gridBounds.yMax - gridBounds.yMin) * currentUnitScale;
    const baseGrid = new THREE.GridHelper(floorSize, Math.max(2, Math.round(floorSize / visibleGridStep)), "#8ea0b4", "#d1d9e3");
    baseGrid.position.set((gridBounds.xMin + gridBounds.xMax) * currentUnitScale / 2, 0, (gridBounds.yMin + gridBounds.yMax) * currentUnitScale / 2);
    baseGrid.visible = true;
    scene.add(baseGrid);
    const coordinateAxes = createCoordinateAxes(symbolScale);
    scene.add(coordinateAxes);
    const spatialGridLines = new THREE.LineSegments(
      createSpatialGridLineGeometry(visibleXValues, visibleYValues, visibleZValues, currentUnitScale),
      new THREE.LineBasicMaterial({ color: "#7f91a6", transparent: true, opacity: 0.45, linewidth: 2 }),
    );
    spatialGridLines.visible = gridVisible;
    scene.add(spatialGridLines);
    const gridPoints: Vec3[] = [];
    const gridPositions: number[] = [];
    const totalGridPoints = xValues.length * yValues.length * zValues.length;
    const pointStride = Math.max(1, Math.ceil(Math.cbrt(totalGridPoints / 50000)));
    for (let xi = 0; xi < xValues.length; xi += pointStride) {
      for (let yi = 0; yi < yValues.length; yi += pointStride) {
        for (let zi = 0; zi < zValues.length; zi += pointStride) {
          const x = xValues[xi];
          const y = yValues[yi];
          const z = zValues[zi];
          gridPoints.push({ x, y, z });
          gridPositions.push(x * currentUnitScale, z * currentUnitScale, y * currentUnitScale);
        }
      }
    }
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute("position", new THREE.Float32BufferAttribute(gridPositions, 3));
    pointGeometry.userData.gridPoints = gridPoints;
    const spatialGrid = new THREE.Points(
      pointGeometry,
      new THREE.PointsMaterial({ color: "#9eb3c8", size: 0.055 * symbolScale, sizeAttenuation: true }),
    );
    spatialGrid.userData.spatialGrid = true;
    spatialGrid.visible = gridVisible;
    scene.add(spatialGrid);

    const content = new THREE.Group();
    scene.add(content);
    const modalContent = new THREE.Group();
    scene.add(modalContent);
    const raycaster = new THREE.Raycaster();
    const pickRadius = Math.max(0.18 * symbolScale, sceneSize * 0.006);
    raycaster.params.Points.threshold = pickRadius;
    const pointer = new THREE.Vector2();
    let nodePickTargets: THREE.Object3D[] = [];
    let memberPickTargets: THREE.Object3D[] = [];
    let pointerDown: { x: number; y: number } | null = null;
    let lastHoverKey = "";
    let lastModalFrameTime = 0;

    const updateModalShape = () => {
      modalContent.children.forEach(disposeObject);
      modalContent.clear();
      const current = modelRef.current;
      const modalModes = modalResultRef.current?.modes ?? [];
      const modalMode = modalModes.find((mode) => mode.mode === activeModeRef.current) ?? modalModes[0];
      if (!modalMode) return;
      const nodeMap = new Map(current.nodes.map((node) => [node.id, node]));
      const modeShapeScale = sceneSize * 0.12 * Math.max(0, modalDeformScaleRef.current);
      const modeRelativeOmega = modalModes[0] ? Math.max(0.25, Math.min(4, modalMode.omega / modalModes[0].omega)) : 1;
      const modePhase = modalAnimateRef.current ? Math.sin((performance.now() / 1000) * 2.2 * modeRelativeOmega) : 1;
      for (const element of current.elements) {
        const a = nodeMap.get(element.startNodeId);
        const b = nodeMap.get(element.endNodeId);
        if (!a || !b) continue;
        modalContent.add(createDisplayMember(
          toThree(a, currentUnitScale, modalMode.displacements[a.id], modeShapeScale * modePhase),
          toThree(b, currentUnitScale, modalMode.displacements[b.id], modeShapeScale * modePhase),
          symbolScale * 0.034,
          modalColorRef.current,
        ));
      }
    };

    const draw = () => {
      content.children.forEach(disposeObject);
      content.clear();
      nodePickTargets = [];
      memberPickTargets = [];
      const current = modelRef.current;
      const solved = resultRef.current;
      const modalModes = modalResultRef.current?.modes ?? [];
      const modalMode = modalModes.find((mode) => mode.mode === activeModeRef.current) ?? modalModes[0];
      const nodeMap = new Map(current.nodes.map((node) => [node.id, node]));
      const inferredRigidNodeIds = new Set<string>();
      for (const element of current.elements) {
        if (element.type === "beam3d") {
          inferredRigidNodeIds.add(element.startNodeId);
          inferredRigidNodeIds.add(element.endNodeId);
        }
      }
      const displacementScale = (solved?.maxDisplacement ? Math.min(200, 1.8 / solved.maxDisplacement) : 1) * Math.max(0, staticDeformScaleRef.current);
      const forceMap = new Map(solved?.elementForces.map((force) => [force.elementId, force]));
      const maxAxial = Math.max(1, ...(solved?.elementForces.map((force) => Math.abs(force.axial)) ?? [0]));
      for (const element of current.elements) {
        const a = nodeMap.get(element.startNodeId);
        const b = nodeMap.get(element.endNodeId);
        if (!a || !b) continue;
        const force = forceMap.get(element.id);
        const axialTolerance = maxAxial * 1e-6;
        const isSelectedElement = selectedElementRef.current === element.id;
        const visualKind = elementVisualKind(element, nodeMap);
        const color = isSelectedElement ? "#f2a900" : modalMode ? "#cbd5e1" : !force || Math.abs(force.axial) <= axialTolerance ? elementTypeColor(visualKind) : force.axial > 0 ? "#1b8f4d" : "#d33f32";
        const forceRatio = force ? Math.abs(force.axial) / maxAxial : 0;
        const baseRadius = visualKind === "bar" ? 0.022 : visualKind === "mixed" ? 0.03 : 0.036;
        const radius = symbolScale * (isSelectedElement ? 1.8 : 1) * (modalMode ? 0.01 : !force || Math.abs(force.axial) <= axialTolerance ? baseRadius : 0.018 + 0.11 * forceRatio);
        const memberMesh = createDisplayMember(toThree(a, currentUnitScale), toThree(b, currentUnitScale), radius, color, modalMode ? 0.62 : 1);
        memberMesh.userData.elementId = element.id;
        content.add(memberMesh);
        memberPickTargets.push(memberMesh);

        if (solved) {
          const deformed = new THREE.BufferGeometry().setFromPoints([
            toThree(a, currentUnitScale, solved.displacements[a.id], displacementScale),
            toThree(b, currentUnitScale, solved.displacements[b.id], displacementScale),
          ]);
          content.add(new THREE.Line(deformed, new THREE.LineBasicMaterial({ color: "#e4572e" })));
        }

      }

      for (const node of current.nodes) {
        const isSelectedNode = selectedRef.current === node.id;
        const isRigidJoint = node.joint ? node.joint === "rigid" : inferredRigidNodeIds.has(node.id);
        const radius = symbolScale * (isSelectedNode ? 0.12 : 0.085);
        const sphere = new THREE.Mesh(
          new THREE.SphereGeometry(radius, 24, 16),
          new THREE.MeshStandardMaterial({
            color: isSelectedNode ? "#f2a900" : "#111827",
            roughness: 0.35,
            transparent: !isRigidJoint,
            opacity: isRigidJoint ? 1 : 0.88,
            wireframe: !isRigidJoint,
          }),
        );
        sphere.position.copy(toThree(node, currentUnitScale));
        sphere.userData.nodeId = node.id;
        content.add(sphere);
        nodePickTargets.push(sphere);
      }

      for (const boundary of current.boundaries) {
        const node = nodeMap.get(boundary.nodeId);
        if (!node) continue;
        const block = new THREE.Mesh(
          new THREE.BoxGeometry(0.36 * symbolScale, 0.18 * symbolScale, 0.36 * symbolScale),
          new THREE.MeshStandardMaterial({ color: "#2d6a4f" }),
        );
        block.position.copy(toThree(node, currentUnitScale)).add(new THREE.Vector3(0, -0.16 * symbolScale, 0));
        content.add(block);
      }

      for (const mass of current.nodalMasses ?? []) {
        const node = nodeMap.get(mass.nodeId);
        if (!node || mass.mass <= 0) continue;
        const size = symbolScale * Math.min(0.42, 0.18 + 0.035 * Math.log10(mass.mass + 1));
        const block = new THREE.Mesh(
          new THREE.BoxGeometry(size, size, size),
          new THREE.MeshStandardMaterial({ color: "#7c3aed", roughness: 0.42, transparent: true, opacity: 0.82 }),
        );
        block.position.copy(toThree(node, currentUnitScale)).add(new THREE.Vector3(0, 0.24 * symbolScale, 0));
        content.add(block);
      }

      for (const load of current.loads) {
        const node = nodeMap.get(load.nodeId);
        if (!node) continue;
        const magnitude = Math.hypot(load.fx ?? 0, load.fy ?? 0, load.fz ?? 0);
        if (magnitude === 0) continue;
        const direction = new THREE.Vector3(load.fx ?? 0, load.fz ?? 0, load.fy ?? 0).normalize();
        const origin = toThree(node, currentUnitScale).clone().sub(direction.clone().multiplyScalar(0.8 * symbolScale));
        content.add(new THREE.ArrowHelper(direction, origin, 0.75 * symbolScale, "#d62828", 0.18 * symbolScale, 0.08 * symbolScale));
      }

      for (const load of current.elementLoads) {
        const element = current.elements.find((item) => item.id === load.elementId);
        if (!element) continue;
        const start = nodeMap.get(element.startNodeId);
        const end = nodeMap.get(element.endNodeId);
        if (!start || !end) continue;
        const axes = elementAxesForDisplay(start, end, element.localY);
        const rawVector = load.type === "point"
          ? { x: load.fx ?? 0, y: load.fy ?? 0, z: load.fz ?? 0 }
          : { x: load.wx ?? 0, y: load.wy ?? 0, z: load.wz ?? 0 };
        const modelVector = load.coordinate === "local" ? localToGlobalVector(axes, rawVector) : rawVector;
        if (vecLength(modelVector) < 1e-12) continue;
        const direction = new THREE.Vector3(modelVector.x, modelVector.z, modelVector.y).normalize();
        const ratios = load.type === "point" ? [Math.min(1, Math.max(0, load.position))] : [0.25, 0.5, 0.75];
        for (const ratio of ratios) {
          const point = interpolateNode(start, end, ratio);
          const origin = toThree(point, currentUnitScale).clone().sub(direction.clone().multiplyScalar(0.65 * symbolScale));
          content.add(new THREE.ArrowHelper(direction, origin, 0.55 * symbolScale, "#9d174d", 0.14 * symbolScale, 0.06 * symbolScale));
        }
      }
    };

    const findOrCreateNode = (point: Vec3): string => {
      const snap = (value: number) => Number((Math.round(value / gridStep) * gridStep).toFixed(6));
      const x = snap(point.x);
      const y = snap(point.y);
      const z = snap(point.z);
      const existing = modelRef.current.nodes.find((node) => Math.abs(node.x - x) < 1e-6 && Math.abs(node.y - y) < 1e-6 && Math.abs(node.z - z) < 1e-6);
      if (existing) return existing.id;
      const id = nextId("N", modelRef.current.nodes.map((node) => node.id));
      const next = insertNodeAndSplit(modelRef.current, { id, x, y, z }, Math.abs(z) < 1e-9);
      saveHistory();
      modelRef.current = next.model;
      setModel(next.model);
      setResult(null);
      setModalResult(null);
      if (next.droppedLoadCount > 0) {
        setError(languageRef.current === "en"
          ? "The inserted node split a member; existing beam loads on the original member were removed. Please set them again."
          : "插入节点已分割构件；原构件梁荷载已移除，请重新设置。");
      } else {
        setError(null);
      }
      return id;
    };

    const findOrCreateMemberSnapNode = (elementId: string, ratio: number): string | null => {
      const current = modelRef.current;
      const element = current.elements.find((item) => item.id === elementId);
      if (!element) return null;
      const start = current.nodes.find((node) => node.id === element.startNodeId);
      const end = current.nodes.find((node) => node.id === element.endNodeId);
      if (!start || !end) return null;
      const point = interpolateNode(start, end, ratio);
      const target = {
        id: "",
        x: Number(point.x.toFixed(6)),
        y: Number(point.y.toFixed(6)),
        z: Number(point.z.toFixed(6)),
      };
      const existing = current.nodes.find((node) => samePoint(node, target));
      if (existing) {
        const inserted = insertNodeAndSplit(current, existing, false);
        if (inserted.splitCount > 0) {
          saveHistory();
          modelRef.current = inserted.model;
          setModel(inserted.model);
          setResult(null);
          setModalResult(null);
        }
        setError(inserted.droppedLoadCount > 0
          ? (languageRef.current === "en"
              ? "The inserted node split a member; existing beam loads on the original member were removed. Please set them again."
              : "插入节点已分割构件；原构件梁荷载已移除，请重新设置。")
          : null);
        return existing.id;
      }
      const id = nextId("N", current.nodes.map((node) => node.id));
      const inserted = insertNodeAndSplit(current, { ...target, id }, Math.abs(target.z) < 1e-9);
      saveHistory();
      modelRef.current = inserted.model;
      setModel(inserted.model);
      setResult(null);
      setModalResult(null);
      setError(inserted.droppedLoadCount > 0
        ? (languageRef.current === "en"
            ? "The inserted node split a member; existing beam loads on the original member were removed. Please set them again."
            : "插入节点已分割构件；原构件梁荷载已移除，请重新设置。")
        : null);
      return id;
    };

    const onPointerDown = (event: PointerEvent) => {
      pointerDown = { x: event.clientX, y: event.clientY };
    };

    const updateHover = (next: HoverInfo | null) => {
      const key = next ? `${next.elementId}:${Math.round(next.x / 10)}:${Math.round(next.y / 10)}:${next.lines.join("|")}` : "";
      if (key === lastHoverKey) return;
      lastHoverKey = key;
      setHoverInfo(next);
    };

    const onPointerMove = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(memberPickTargets)[0];
      if (!hit?.object.userData.elementId) {
        updateHover(null);
        return;
      }
      const elementId = hit.object.userData.elementId as string;
      const current = modelRef.current;
      const element = current.elements.find((item) => item.id === elementId);
      if (!element) {
        updateHover(null);
        return;
      }
      const nodeMap = new Map(current.nodes.map((node) => [node.id, node]));
      const start = nodeMap.get(element.startNodeId);
      const end = nodeMap.get(element.endNodeId);
      const force = resultRef.current?.elementForces.find((item) => item.elementId === elementId);
      const hoverLanguage = languageRef.current;
      const snapRatio = start && end ? nearestMemberSnapRatio(hit.point, start, end, currentUnitScale) : null;
      const deleteLine = toolRef.current === "delete"
        ? (hoverLanguage === "en" ? "Click to delete" : "点击删除")
        : "";
      const snapLine = snapRatio !== null && toolRef.current !== "select" && toolRef.current !== "delete"
        ? (hoverLanguage === "en" ? `Pick ${memberSnapLabel(snapRatio)} to split member` : `拾取 ${memberSnapLabel(snapRatio)} 点分割梁杆`)
        : "";
      const lines = force
        ? [
            `${forceStatus(force, hoverLanguage)} ${elementTypeLabel(elementVisualKind(element, nodeMap), hoverLanguage)} ${elementId}`,
            `N ${format(force.axial)}`,
            force.shearY !== undefined ? `Vy ${format(force.shearY)}` : "",
            force.shearZ !== undefined ? `Vz ${format(force.shearZ)}` : "",
            force.momentYStart !== undefined ? `My ${formatMomentPair(force.momentYStart, force.momentYEnd, unitRef.current)} ${momentUnit(unitRef.current)}` : "",
            force.momentZStart !== undefined ? `Mz ${formatMomentPair(force.momentZStart, force.momentZEnd, unitRef.current)} ${momentUnit(unitRef.current)}` : "",
            deleteLine,
            snapLine,
          ].filter(Boolean)
        : [
            `${elementTypeLabel(elementVisualKind(element, nodeMap), hoverLanguage)} ${elementId}`,
            start && end ? `L ${format(nodeDistance(start, end))} ${unitLabels[unitRef.current]}` : "",
            deleteLine,
            snapLine,
            hoverLanguage === "en" ? "Internal forces appear after solving" : "求解后显示内力",
          ].filter(Boolean);
      updateHover({ elementId, x: event.clientX, y: event.clientY, lines });
    };

    const onPointerLeave = () => {
      updateHover(null);
    };

    const onPointerUp = (event: PointerEvent) => {
      if (!pointerDown) return;
      const moved = Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y);
      pointerDown = null;
      if (moved > 5) return;

      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const nodeHits = raycaster.intersectObjects(nodePickTargets).filter((hit) => hit.object.userData.nodeId);
      if (nodeHits[0]) {
        const id = nodeHits[0].object.userData.nodeId as string;
        if (toolRef.current === "delete") {
          saveHistory();
          const next = deleteNodeFromModel(modelRef.current, id);
          modelRef.current = next;
          setModel(next);
          setSelectedNode(null);
          setSelectedElement(null);
          setPendingNode(null);
          setResult(null);
          setModalResult(null);
          setError(null);
          updateHover(null);
          return;
        }
        setSelectedNode(id);
        setSelectedElement(null);
        handleNodeAction(id);
        return;
      }
      const memberHit = raycaster.intersectObjects(memberPickTargets)[0];
      if (memberHit?.object.userData.elementId && toolRef.current === "delete") {
        const elementId = memberHit.object.userData.elementId as string;
        saveHistory();
        const next = deleteElementFromModel(modelRef.current, elementId);
        modelRef.current = next;
        setModel(next);
        setSelectedNode(null);
        setSelectedElement(null);
        setPendingNode(null);
        setResult(null);
        setModalResult(null);
        setError(null);
        updateHover(null);
        return;
      }
      if (memberHit?.object.userData.elementId && toolRef.current !== "select") {
        const elementId = memberHit.object.userData.elementId as string;
        const current = modelRef.current;
        const element = current.elements.find((item) => item.id === elementId);
        const start = element ? current.nodes.find((node) => node.id === element.startNodeId) : null;
        const end = element ? current.nodes.find((node) => node.id === element.endNodeId) : null;
        const snapRatio = start && end ? nearestMemberSnapRatio(memberHit.point, start, end, currentUnitScale) : null;
        if (snapRatio !== null) {
          const id = findOrCreateMemberSnapNode(elementId, snapRatio);
          if (id) {
            setSelectedNode(id);
            setSelectedElement(null);
            handleNodeAction(id);
            return;
          }
        }
      }
      if (memberHit?.object.userData.elementId && toolRef.current === "select") {
        setSelectedElement(memberHit.object.userData.elementId as string);
        setSelectedNode(null);
        setPendingNode(null);
        return;
      }
      if (toolRef.current === "select") {
        setSelectedNode(null);
        setSelectedElement(null);
        setPendingNode(null);
        return;
      }
      if (!gridVisible) {
        setSelectedNode(null);
        setSelectedElement(null);
        setPendingNode(null);
        return;
      }
      const snappedPoint = closestGridPointToRay(raycaster.ray, gridPoints, currentUnitScale, pickRadius);
      if (!snappedPoint) {
        setSelectedNode(null);
        setSelectedElement(null);
        setPendingNode(null);
        return;
      }
      const id = findOrCreateNode(snappedPoint);
      setSelectedNode(id);
      setSelectedElement(null);
      handleNodeAction(id);
    };

    const handleNodeAction = (nodeId: string) => {
      const activeTool = toolRef.current;
      if (activeTool === "support") {
        saveHistory();
        setModel((current) => {
          const rest = current.boundaries.filter((item) => item.nodeId !== nodeId);
          return { ...current, boundaries: [...rest, { nodeId, ux: true, uy: true, uz: true, rx: true, ry: true, rz: true }] };
        });
        setResult(null);
        return;
      }
      if (activeTool === "load") {
        saveHistory();
        setModel((current) => {
          const rest = current.loads.filter((item) => item.nodeId !== nodeId);
          return { ...current, loads: [...rest, { nodeId, fz: loadZ }] };
        });
        setResult(null);
        return;
      }
      if (activeTool !== "member") return;
      if (!pendingRef.current) {
        setPendingNode(nodeId);
        return;
      }
      if (pendingRef.current === nodeId) {
        setPendingNode(null);
        return;
      }
      saveHistory();
      setModel((current) => ({
        ...current,
        elements: [
          ...current.elements,
          {
            id: nextId("E", current.elements.map((element) => element.id)),
            type: "bar3d",
            startNodeId: pendingRef.current!,
            endNodeId: nodeId,
            materialId: defaultMaterialRef.current,
            sectionId: defaultSectionRef.current,
          },
        ],
      }));
      setPendingNode(null);
      setResult(null);
    };

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerleave", onPointerLeave);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    const resize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener("resize", resize);

    let animationId = 0;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      controls.update();
      if (sceneDirtyRef.current) {
        draw();
        updateModalShape();
        sceneDirtyRef.current = false;
      } else if (modalAnimateRef.current && modalResultRef.current?.modes.length) {
        const now = performance.now();
        if (now - lastModalFrameTime > 33) {
          updateModalShape();
          lastModalFrameTime = now;
        }
      }
      renderer.render(scene, camera);
    };
    sceneDirtyRef.current = true;
    animate();

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", resize);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      controls.dispose();
      content.children.forEach(disposeObject);
      modalContent.children.forEach(disposeObject);
      disposeObject(coordinateAxes);
      spatialGridLines.geometry.dispose();
      (spatialGridLines.material as THREE.Material).dispose();
      spatialGrid.geometry.dispose();
      (spatialGrid.material as THREE.Material).dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [loadZ, gridStep, gridBounds, currentUnitScale, gridVisible]);

  const updateMaterial = (key: "E" | "G" | "density", value: number) => {
    saveHistory();
    setModel((current) => ({
      ...current,
      materials: current.materials.map((item) => item.id === activeMaterial.id ? { ...item, [key]: value } : item),
    }));
    setResult(null);
  };

  const updateSection = (key: "A" | "Iy" | "Iz" | "J", value: number) => {
    saveHistory();
    setModel((current) => ({
      ...current,
      sections: current.sections.map((item) => item.id === activeSection.id ? { ...item, [key]: value } : item),
    }));
    setResult(null);
  };

  const createSectionFromActive = () => {
    const name = sectionNameDraft.trim() || `Section ${model.sections.length + 1}`;
    const id = slugId(name, "section", model.sections.map((section) => section.id));
    saveHistory();
    setModel((current) => ({
      ...current,
      sections: [...current.sections, { ...activeSection, id, name }],
    }));
    setDefaultSectionId(id);
    setSectionNameDraft(`Section ${model.sections.length + 2}`);
    setSelectedElement(null);
    setResult(null);
    setModalResult(null);
    setError(null);
  };

  const applyMaterial = (materialId: string) => {
    if (selectedElement) {
      saveHistory();
      setModel((current) => ({
        ...current,
        elements: current.elements.map((element) => element.id === selectedElement ? { ...element, materialId } : element),
      }));
      setResult(null);
      setModalResult(null);
      return;
    }
    setDefaultMaterialId(materialId);
  };

  const applySection = (sectionId: string) => {
    if (selectedElement) {
      saveHistory();
      setModel((current) => ({
        ...current,
        elements: current.elements.map((element) => element.id === selectedElement ? { ...element, sectionId } : element),
      }));
      setResult(null);
      setModalResult(null);
      return;
    }
    setDefaultSectionId(sectionId);
  };

  const confirmSelectedLoad = () => {
    if (!selectedNode) return;
    saveHistory();
    setModel((current) => {
      const rest = current.loads.filter((item) => item.nodeId !== selectedNode);
      const next = { nodeId: selectedNode, fx: loadDraft.fx, fy: loadDraft.fy, fz: loadDraft.fz };
      return { ...current, loads: [...rest, next] };
    });
    setLoadZ(loadDraft.fz);
    setResult(null);
    setError(null);
  };

  const confirmSelectedMass = () => {
    if (!selectedNode) return;
    saveHistory();
    setModel((current) => {
      const rest = (current.nodalMasses ?? []).filter((item) => item.nodeId !== selectedNode);
      return { ...current, nodalMasses: [...rest, { nodeId: selectedNode, mass: Math.max(0, massDraft) }] };
    });
    setResult(null);
    setModalResult(null);
    setError(null);
  };

  const deleteSelectedMass = () => {
    if (!selectedNode) return;
    saveHistory();
    setModel((current) => ({
      ...current,
      nodalMasses: (current.nodalMasses ?? []).filter((item) => item.nodeId !== selectedNode),
    }));
    setResult(null);
    setModalResult(null);
    setError(null);
  };

  const toggleBoundary = (key: DofKey) => {
    if (!selectedNode) return;
    saveHistory();
    setModel((current) => {
      const found = current.boundaries.find((item) => item.nodeId === selectedNode);
      const rest = current.boundaries.filter((item) => item.nodeId !== selectedNode);
      const next = { nodeId: selectedNode, ...(found ?? {}), [key]: !(found?.[key] ?? false) };
      return { ...current, boundaries: [...rest, next] };
    });
    setResult(null);
  };

  const updateBoundaryValue = (key: DofKey, value: number) => {
    if (!selectedNode) return;
    setModel((current) => {
      const found = current.boundaries.find((item) => item.nodeId === selectedNode);
      const rest = current.boundaries.filter((item) => item.nodeId !== selectedNode);
      const next = {
        nodeId: selectedNode,
        ...(found ?? {}),
        values: { ...(found?.values ?? {}), [key]: value },
      };
      return { ...current, boundaries: [...rest, next] };
    });
    setResult(null);
    setModalResult(null);
  };

  const setSelectedJoint = (joint: "hinged" | "rigid") => {
    if (!selectedNode) return;
    saveHistory();
    setModel((current) => ({
      ...current,
      nodes: current.nodes.map((node) => node.id === selectedNode ? { ...node, joint } : node),
    }));
    setResult(null);
    setModalResult(null);
  };

  const setSelectedElementType = (type: ElementType) => {
    if (!selectedElementData || selectedElementData.type === type) return;
    saveHistory();
    setModel((current) => ({
      ...current,
      elements: current.elements.map((element) => element.id === selectedElementData.id ? { ...element, type } : element),
      elementLoads: type === "bar3d"
        ? (current.elementLoads ?? []).filter((load) => load.elementId !== selectedElementData.id)
        : (current.elementLoads ?? []),
    }));
    setResult(null);
    setModalResult(null);
    setError(null);
  };

  const autoClassifyElements = () => {
    saveHistory();
    setModel((current) => ({
      ...current,
      elements: current.elements.map((element) => ({ ...element, type: inferElementType(element, current) })),
    }));
    setResult(null);
    setModalResult(null);
    setError(null);
  };

  const clearModel = () => {
    saveHistory();
    setModel(emptyModel());
    setDefaultMaterialId("wood");
    setDefaultSectionId("timber100");
    setSelectedNode(null);
    setSelectedElement(null);
    setPendingNode(null);
    setResult(null);
    setError(null);
  };

  const createNewProject = () => {
    const nextModel = newProjectModel();
    saveHistory();
    setModel(nextModel);
    setDefaultMaterialId("wood");
    setDefaultSectionId("timber100");
    setSelectedNode("N1");
    setSelectedElement(null);
    setPendingNode(null);
    setTool("member");
    setUnit("cm");
    setGridStep(defaultGridStepByUnit.cm);
    setGridVisible(true);
    setOffset({ dx: defaultGridStepByUnit.cm, dy: 0, dz: 0 });
    setCoordinateNode({ x: 0, y: 0, z: 0 });
    setLoadZ(-15000);
    setLoadDraft({ fx: 0, fy: 0, fz: -15000 });
    setResult(null);
    setModalResult(null);
    setActiveMode(1);
    setModalAnimate(false);
    setError(null);
  };

  const deleteSelection = () => {
    if (!selectedNode && !selectedElement) return;
    const previousTool = tool;
    saveHistory();
    if (selectedElement) {
      setModel((current) => deleteElementFromModel(current, selectedElement));
      setSelectedElement(null);
      setPendingNode(null);
      setTool(previousTool === "delete" ? "select" : previousTool);
      setResult(null);
      setModalResult(null);
      setError(null);
      return;
    }
    if (!selectedNode) return;
    setModel((current) => deleteNodeFromModel(current, selectedNode));
    setSelectedNode(null);
    setSelectedElement(null);
    setPendingNode(null);
    setTool(previousTool === "delete" ? "select" : previousTool);
    setResult(null);
    setModalResult(null);
    setError(null);
  };

  const toggleDeleteMode = () => {
    if (selectedNode || selectedElement) {
      deleteSelection();
      return;
    }
    if (tool === "delete") {
      setTool("select");
      setPendingNode(null);
      return;
    }
    setTool("delete");
    setPendingNode(null);
  };

  const addElementLoad = () => {
    if (!selectedElementData) return;
    const id = nextId("L", (model.elementLoads ?? []).map((load) => load.id));
    const coordinate = elementLoadDraft.coordinate;
    const nextLoad: ElementLoad = elementLoadDraft.type === "point"
      ? {
          id,
          elementId: selectedElementData.id,
          type: "point",
          coordinate,
          position: Math.min(1, Math.max(0, elementLoadDraft.position)),
          fx: elementLoadDraft.x,
          fy: elementLoadDraft.y,
          fz: elementLoadDraft.z,
        }
      : {
          id,
          elementId: selectedElementData.id,
          type: "distributed",
          coordinate,
          wx: elementLoadDraft.x,
          wy: elementLoadDraft.y,
          wz: elementLoadDraft.z,
        };
    saveHistory();
    setModel((current) => ({
      ...current,
      elements: current.elements.map((element) => element.id === selectedElementData.id ? { ...element, type: "beam3d" } : element),
      elementLoads: [...(current.elementLoads ?? []), nextLoad],
    }));
    setResult(null);
    setError(null);
  };

  const deleteElementLoad = (loadId: string) => {
    saveHistory();
    setModel((current) => ({
      ...current,
      elementLoads: (current.elementLoads ?? []).filter((load) => load.id !== loadId),
    }));
    setResult(null);
    setError(null);
  };

  const createOffsetMember = () => {
    if (!selectedNode) return;
    const base = model.nodes.find((node) => node.id === selectedNode);
    if (!base) return;
    const target = {
      x: Number((base.x + offset.dx).toFixed(6)),
      y: Number((base.y + offset.dy).toFixed(6)),
      z: Number((base.z + offset.dz).toFixed(6)),
    };
    if (samePoint(base, target)) {
      setError(en ? "dx/dy/dz cannot all be 0." : "dx/dy/dz 不能全为 0。");
      return;
    }
    const existing = model.nodes.find((node) => samePoint(node, target));
    const targetNodeId = existing?.id ?? nextId("N", model.nodes.map((node) => node.id));
    const inserted = insertNodeAndSplit(model, { id: targetNodeId, ...target }, !existing && Math.abs(target.z) < 1e-9);
    const elementExists = inserted.model.elements.some((element) => (
      (element.startNodeId === selectedNode && element.endNodeId === targetNodeId) ||
      (element.startNodeId === targetNodeId && element.endNodeId === selectedNode)
    ));
    const nextModel = elementExists
      ? inserted.model
      : {
          ...inserted.model,
          elements: [
            ...inserted.model.elements,
            {
              id: nextId("E", inserted.model.elements.map((element) => element.id)),
              type: "bar3d" as const,
              startNodeId: selectedNode,
              endNodeId: targetNodeId,
              materialId: defaultMaterialRef.current,
              sectionId: defaultSectionRef.current,
            },
          ],
        };
    saveHistory();
    setModel(nextModel);
    setSelectedNode(targetNodeId);
    setSelectedElement(null);
    setPendingNode(null);
    setResult(null);
    setError(inserted.droppedLoadCount > 0 ? (en ? "The inserted node split a member; existing beam loads on the original member were removed. Please set them again." : "插入节点已分割构件；原构件梁荷载已移除，请重新设置。") : null);
  };

  const createCoordinateNode = () => {
    const target = {
      x: Number(coordinateNode.x.toFixed(6)),
      y: Number(coordinateNode.y.toFixed(6)),
      z: Number(coordinateNode.z.toFixed(6)),
    };
    const existing = model.nodes.find((node) => samePoint(node, target));
    if (existing) {
      const inserted = insertNodeAndSplit(model, existing, false);
      if (inserted.splitCount > 0) {
        saveHistory();
        setModel(inserted.model);
      }
      setSelectedNode(existing.id);
      setSelectedElement(null);
      setPendingNode(null);
      setResult(null);
      setError(inserted.droppedLoadCount > 0 ? (en ? "The inserted node split a member; existing beam loads on the original member were removed. Please set them again." : "插入节点已分割构件；原构件梁荷载已移除，请重新设置。") : null);
      return;
    }
    const nodeId = nextId("N", model.nodes.map((node) => node.id));
    const inserted = insertNodeAndSplit(model, { id: nodeId, ...target }, Math.abs(target.z) < 1e-9);
    saveHistory();
    setModel(inserted.model);
    setSelectedNode(nodeId);
    setSelectedElement(null);
    setPendingNode(null);
    setResult(null);
    setError(inserted.droppedLoadCount > 0 ? (en ? "The inserted node split a member; existing beam loads on the original member were removed. Please set them again." : "插入节点已分割构件；原构件梁荷载已移除，请重新设置。") : null);
  };

  const saveProject = () => {
    const project: ProjectFile = {
      app: appName,
      version: "1.0",
      savedAt: new Date().toISOString(),
      unit,
      gridStep,
      gridVisible,
      loadZ,
      offset,
      coordinateNode,
      defaultMaterialId,
      defaultSectionId,
      language,
      model,
    };
    const blob = new Blob([JSON.stringify(project, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    link.href = url;
    link.download = `FrameSolve-3D-${timestamp}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    setError(null);
  };

  const loadProject = async (file: File) => {
    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as Partial<ProjectFile>;
      if ((parsed.app !== appName && parsed.app !== "BeamBar3D") || !isStructureModel(parsed.model)) {
        throw new Error(en ? `This is not a valid ${appName} project file.` : `不是有效的 ${appName} 项目文件。`);
      }
      const nextUnit = isUnitKey(parsed.unit) ? parsed.unit : "cm";
      const nextModel = {
        ...parsed.model,
        materials: parsed.model.materials.map((material) => ({ ...material, density: material.density ?? 500 })),
        elementLoads: parsed.model.elementLoads ?? [],
        nodalMasses: parsed.model.nodalMasses ?? [],
      };
      saveHistory();
      setModel(nextModel);
      setUnit(nextUnit);
      setGridStep(toFiniteNumber(parsed.gridStep, defaultGridStepByUnit[nextUnit]));
      setGridVisible(typeof parsed.gridVisible === "boolean" ? parsed.gridVisible : true);
      if (parsed.language === "zh" || parsed.language === "en") setLanguage(parsed.language);
      setLoadZ(toFiniteNumber(parsed.loadZ, -15000));
      setOffset(toOffset(parsed.offset, { dx: 5, dy: 0, dz: 0 }));
      setCoordinateNode(toVec3(parsed.coordinateNode, { x: 0, y: 0, z: 0 }));
      setDefaultMaterialId(nextModel.materials.some((material) => material.id === parsed.defaultMaterialId) ? parsed.defaultMaterialId! : nextModel.materials[0]?.id ?? "wood");
      setDefaultSectionId(nextModel.sections.some((section) => section.id === parsed.defaultSectionId) ? parsed.defaultSectionId! : nextModel.sections[0]?.id ?? "timber100");
      setSelectedNode(nextModel.nodes[0]?.id ?? null);
      setSelectedElement(null);
      setPendingNode(null);
      setResult(null);
      setError(null);
    } catch (projectError) {
      setError(projectError instanceof Error ? projectError.message : (en ? "Failed to load project." : "读取项目失败。"));
    }
  };

  return (
    <main className="app">
      <section className="viewport" ref={mountRef} />
      {hoverInfo && (
        <div className="hoverTooltip" style={{ left: hoverInfo.x + 14, top: hoverInfo.y + 14 }}>
          {hoverInfo.lines.map((line) => <span key={line}>{line}</span>)}
        </div>
      )}
      {helpOpen && (
        <div className="modalBackdrop" onClick={() => setHelpOpen(false)}>
          <section className="helpModal" onClick={(event) => event.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby="help-title">
            <header className="helpHeader">
              <div>
                <h2 id="help-title">Help</h2>
                <span>{text.helpGuide}</span>
              </div>
              <button className="iconButton" onClick={() => setHelpOpen(false)} title={en ? "Close" : "关闭"}><X size={18} /></button>
            </header>
            <div className="helpBody">
              <section>
                <h3>{text.productGuide}</h3>
                <p>{en ? "FrameSolve 3D is a fast 3D frame/truss modeling, solving, and result visualization tool. New members are bars by default; switch to beam elements when bending, shear, member loads, or rigid-frame behavior is needed." : "FrameSolve 3D 用于三维杆系/桁架优先的快速建模、求解与结果查看。默认新建杆单元；需要弯矩、剪力、梁荷载或刚接框架时，可将单元切换为梁单元。"}</p>
                <p>{en ? "Built-in examples carry their own unit systems. When loaded, coordinates are inserted 1:1 in that unit and the grid spacing switches automatically. Materials and sections can be selected as defaults before creating members." : "典型模型带有自己的单位系统，加载时会按该单位 1:1 放入模型并自动切换 grid 间距。材料与截面可先设为默认标签，再用于后续新建杆件。"}</p>
                <p>{en ? "Supports spatial grid picking, member 1/3-midpoint-2/3 snap splitting, coordinate-based node creation, offset creation, nodal constraints and loads, point masses, project save/load, and Ctrl+Z undo. After solving, axial force tension/compression colors, deformation, omega, frequency f, modal scale, and animated mode shapes are available." : "支持空间 grid 拾取、梁杆 1/3-中点-2/3 拾取分割、坐标新建、按偏移新建、节点约束与荷载、集中质量、保存/读取项目、Ctrl+Z 撤销。求解后可显示轴力拉压颜色和变形；频率分析可选择阶数、显示 omega 与 f，调整模态 scale，并用动态振型查看不同模态。"}</p>
              </section>
              <section>
                <h3>{text.authorInfo}</h3>
                <dl>
                  <div><dt>{text.affiliation}</dt><dd>{en ? "Tongji University" : "同济大学"}</dd></div>
                  <div><dt>{text.title}</dt><dd>{text.associateProfessor}</dd></div>
                  <div><dt>{text.author}</dt><dd>{en ? "Huilong Ren" : "任辉龙"}</dd></div>
                  <div><dt>{text.email}</dt><dd><a href="mailto:hlren@tongji.edu.cn">hlren@tongji.edu.cn</a></dd></div>
                  <div><dt>{text.scholar}</dt><dd><a href="https://scholar.google.com/citations?user=Rsr_KWIAAAAJ&hl=en" target="_blank" rel="noreferrer">scholar.google.com</a></dd></div>
                  <div><dt>Version</dt><dd>1.0</dd></div>
                </dl>
              </section>
            </div>
          </section>
        </div>
      )}
      {resultPopupOpen && modalResult && (
        <section
          className="resultPopup modalResultPopup"
          style={{
            left: resultPopupPosition.x,
            bottom: resultPopupPosition.y > 0 ? "auto" : 14,
            top: resultPopupPosition.y > 0 ? resultPopupPosition.y : "auto",
          }}
          aria-label={text.resultTable}
        >
          <header
            className="resultPopupHeader"
            onPointerDown={startResultPopupDrag}
            onPointerMove={dragResultPopup}
            onPointerUp={stopResultPopupDrag}
            onPointerCancel={stopResultPopupDrag}
          >
            <strong>{text.modal}</strong>
            <button className="iconButton" onClick={() => setResultPopupOpen(false)} title={text.close}><X size={16} /></button>
          </header>
          <div className="resultPopupControls">
            <button className={modalAnimate ? "active" : ""} onClick={() => setModalAnimate((value) => !value)}>
              <Play size={16} />{modalAnimate ? text.stopAnimation : text.animate}
            </button>
            <label className="selectField colorField compactColorField">
              <span>{text.lineColor}</span>
              <input type="color" value={modalColor} onChange={(event) => setModalColor(event.target.value)} />
            </label>
          </div>
          <div className="table modalTable">
            <div className="row head"><span>{text.mode}</span><span>omega</span><span>f Hz</span></div>
            {modalResult.modes.map((mode) => (
              <div className="row" key={mode.mode}>
                <button className={activeMode === mode.mode ? "active" : ""} onClick={() => setActiveMode(mode.mode)}>
                  {mode.mode}
                </button>
                <span>{format(mode.omega)}</span>
                <span>{format(mode.frequency)}</span>
              </div>
            ))}
          </div>
        </section>
      )}
      <aside className="sidebar">
        <header className="brand">
          <div>
            <strong>FrameSolve 3D</strong>
            <span>{text.subtitle}</span>
          </div>
          <div className="headerActions">
            <button className="iconButton" onClick={() => setHelpOpen(true)} title="Help"><CircleHelp size={18} /></button>
            <button className="iconButton" onClick={createNewProject} title={text.newProject}><FilePlus2 size={18} /></button>
            <button className="iconButton" onClick={saveProject} title={text.saveProject}><Save size={18} /></button>
            <button className="iconButton" onClick={() => projectInputRef.current?.click()} title={text.loadProject}><FolderOpen size={18} /></button>
            <input
              ref={projectInputRef}
              className="hiddenFileInput"
              type="file"
              accept="application/json,.json"
              onChange={(event) => {
                const file = event.target.files?.[0];
                event.target.value = "";
                if (file) void loadProject(file);
              }}
            />
            <button className="iconButton" onClick={undo} title={text.undo} disabled={undoCount === 0}><Undo2 size={18} /></button>
            <button
              className="iconButton"
              onClick={() => loadExampleModel()}
              title={text.loadSelectedExample}
            >
              <RefreshCcw size={18} />
            </button>
          </div>
        </header>

        <div className="languageSwitch" aria-label={text.language}>
          <button className={language === "zh" ? "active" : ""} onClick={() => setLanguage("zh")}>{text.chinese}</button>
          <button className={language === "en" ? "active" : ""} onClick={() => setLanguage("en")}>{text.english}</button>
        </div>

        <div className="toolbar">
          <button className={tool === "member" ? "active" : ""} onClick={() => setTool("member")} title={text.pickMember}><Link2 size={17} />{text.member}</button>
          <button className={tool === "select" ? "active" : ""} onClick={() => setTool("select")} title={text.selectEdit}><BoxSelect size={17} />{text.select}</button>
          <button className={tool === "support" ? "active" : ""} onClick={() => setTool("support")} title={text.support}><Hammer size={17} />{text.support}</button>
          <button className={tool === "load" ? "active" : ""} onClick={() => setTool("load")} title={text.load}><CircleDot size={17} />{text.load}</button>
          <button className={tool === "delete" ? "active" : ""} onClick={toggleDeleteMode} title={text.deleteSelected} disabled={tool !== "delete" && model.nodes.length === 0 && model.elements.length === 0}><Trash2 size={17} />{text.deleteSelected}</button>
        </div>

        <div className="stats">
          <span>{text.nodes} {model.nodes.length}</span>
          <span>{text.elements} {model.elements.length}</span>
          <span>{tool === "delete" ? text.deleting : selectedElement ? `${text.selected} ${selectedElement}` : pendingNode ? `${text.start} ${pendingNode}` : text.pickGrid}</span>
        </div>

        <section className="panel compactPanel">
          <h2>{text.examples}</h2>
          <label className="selectField">
            <span>{text.example}</span>
            <select value={selectedExample} onChange={(event) => setSelectedExample(event.target.value as ExampleKey)}>
              {exampleModels.map((example) => <option key={example.key} value={example.key}>{en ? example.nameEn : example.name} ({unitLabels[example.unit]})</option>)}
            </select>
          </label>
          <button onClick={() => loadExampleModel()}><RefreshCcw size={17} />{text.loadExample}</button>
        </section>

        <section className="panel compactPanel">
          <h2>{text.select}</h2>
          <p className="selectionText">
            {selectedElementData
              ? `${selectedElementData.id}: ${selectedElementData.startNodeId} -> ${selectedElementData.endNodeId} · ${selectedElementKind ? elementTypeLabel(selectedElementKind, language) : ""}`
              : selectedNode
                ? `${text.node} ${selectedNode}`
                : text.noSelection}
          </p>
          <div className="typeLegend" aria-label={en ? "Element color legend" : "单元颜色图例"}>
            <span><i style={{ background: elementTypeColor("bar") }} />{text.bar}</span>
            <span><i style={{ background: elementTypeColor("beam") }} />{text.beam}</span>
            <span><i style={{ background: elementTypeColor("mixed") }} />{text.hingedBeam}</span>
          </div>
          {selectedElementData && (
            <div className="segmented twoSegment">
              <button className={selectedElementData.type === "bar3d" ? "active" : ""} onClick={() => setSelectedElementType("bar3d")}>
                {text.barElement}
              </button>
              <button className={selectedElementData.type === "beam3d" ? "active" : ""} onClick={() => setSelectedElementType("beam3d")}>
                {text.beamElement}
              </button>
            </div>
          )}
          <button onClick={autoClassifyElements} disabled={model.elements.length === 0}><RefreshCcw size={17} />{text.autoClassify}</button>
          {tool === "delete" && <p className="selectionText">{text.deleteMode}</p>}
        </section>

        {selectedElementData && (
          <section className="panel">
            <h2>{text.beamLoad} {selectedElementData.id}</h2>
            <div className="segmented">
              <button className={elementLoadDraft.type === "point" ? "active" : ""} onClick={() => setElementLoadDraft((current) => ({ ...current, type: "point" }))}>
                {text.point}
              </button>
              <button className={elementLoadDraft.type === "distributed" ? "active" : ""} onClick={() => setElementLoadDraft((current) => ({ ...current, type: "distributed" }))}>
                {text.distributed}
              </button>
              <button className={elementLoadDraft.coordinate === "global" ? "active" : ""} onClick={() => setElementLoadDraft((current) => ({ ...current, coordinate: current.coordinate === "global" ? "local" : "global" }))}>
                {elementLoadDraft.coordinate === "global" ? text.global : text.local}
              </button>
            </div>
            {elementLoadDraft.type === "point" && (
              <NumberField label="a/L" value={elementLoadDraft.position} onChange={(value) => setElementLoadDraft((current) => ({ ...current, position: value }))} />
            )}
            <div className="loadGrid">
              <NumberField label={elementLoadDraft.type === "point" ? "X N" : `qx N/${unitLabels[unit]}`} value={elementLoadDraft.x} onChange={(value) => setElementLoadDraft((current) => ({ ...current, x: value }))} />
              <NumberField label={elementLoadDraft.type === "point" ? "Y N" : `qy N/${unitLabels[unit]}`} value={elementLoadDraft.y} onChange={(value) => setElementLoadDraft((current) => ({ ...current, y: value }))} />
              <NumberField label={elementLoadDraft.type === "point" ? "Z N" : `qz N/${unitLabels[unit]}`} value={elementLoadDraft.z} onChange={(value) => setElementLoadDraft((current) => ({ ...current, z: value }))} />
            </div>
            <button onClick={addElementLoad}><Plus size={17} />{text.addBeamLoad}</button>
            {selectedElementData.type !== "beam3d" && <p className="selectionText">{text.beamLoadHint}</p>}
            {selectedElementLoads.length > 0 && (
              <div className="loadList">
                {selectedElementLoads.map((load) => (
                  <div className="loadItem" key={load.id}>
                    <span>
                      {load.type === "point"
                        ? `${load.id} ${text.point} ${load.coordinate === "global" ? text.global : text.local} a/L=${format(load.position, 2)} (${format(load.fx)}, ${format(load.fy)}, ${format(load.fz)})`
                        : `${load.id} ${text.distributed} ${load.coordinate === "global" ? text.global : text.local} (${format(load.wx)}, ${format(load.wy)}, ${format(load.wz)})`}
                    </span>
                    <button className="iconButton" onClick={() => deleteElementLoad(load.id)} title={text.deleteBeamLoad}><Trash2 size={15} /></button>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        <section className="panel compactPanel">
          <h2>{text.unitSystem}</h2>
          <div className="segmented">
            {(["m", "cm", "mm"] as UnitKey[]).map((item) => (
              <button key={item} className={unit === item ? "active" : ""} onClick={() => changeUnit(item)}>
                {unitLabels[item]}
              </button>
            ))}
          </div>
        </section>

        <section className="panel collapsiblePanel">
          <button className="panelToggle" onClick={() => setMaterialOpen((open) => !open)} aria-expanded={materialOpen}>
            {materialOpen ? <ChevronDown size={17} /> : <ChevronRight size={17} />}
            <span>{text.materialSection}</span>
            <small>{activeSection.name}</small>
          </button>
          {materialOpen && (
            <div className="panelBody">
              <label className="selectField">
                <span>{text.material}</span>
                <select value={activeMaterial.id} onChange={(event) => applyMaterial(event.target.value)}>
                  {model.materials.map((material) => <option key={material.id} value={material.id}>{material.name}</option>)}
                </select>
              </label>
              <label className="selectField">
                <span>{text.sectionTag}</span>
                <select value={activeSection.id} onChange={(event) => applySection(event.target.value)}>
                  {model.sections.map((section) => <option key={section.id} value={section.id}>{section.name}</option>)}
                </select>
              </label>
              <p className="selectionText">{selectedElement ? `${text.editing} ${selectedElement}` : text.sectionHint}</p>
              <label className="selectField">
                <span>{text.newTag}</span>
                <input value={sectionNameDraft} onChange={(event) => setSectionNameDraft(event.target.value)} />
              </label>
              <button onClick={createSectionFromActive}><Plus size={17} />{text.addSectionTag}</button>
              <NumberField label="E Pa" value={activeMaterial.E} onChange={(value) => updateMaterial("E", value)} />
              <NumberField label="G Pa" value={activeMaterial.G} onChange={(value) => updateMaterial("G", value)} />
              <NumberField label="rho kg/m3" value={activeMaterial.density} onChange={(value) => updateMaterial("density", value)} />
              <NumberField label="A m^2" value={activeSection.A} onChange={(value) => updateSection("A", value)} />
              <NumberField label="Iy m^4" value={activeSection.Iy} onChange={(value) => updateSection("Iy", value)} />
              <NumberField label="Iz m^4" value={activeSection.Iz} onChange={(value) => updateSection("Iz", value)} />
              <NumberField label="J m^4" value={activeSection.J} onChange={(value) => updateSection("J", value)} />
            </div>
          )}
        </section>

        <section className="panel compactPanel">
          <h2>{text.spatialGrid}</h2>
          <div className="gridInlineControls">
            <button onClick={() => setGridVisible((visible) => !visible)}>
              {gridVisible ? <EyeOff size={17} /> : <Eye size={17} />}
              {gridVisible ? text.hideGrid : text.showGrid}
            </button>
            <div className="gridStepField">
              <NumberField
                label={text.spacing}
                value={gridStep}
                onChange={(value) => {
                  if (Number.isFinite(value) && value > 0) setGridStep(value);
                }}
              />
              <span>{unitLabels[unit]}</span>
            </div>
          </div>
        </section>

        <section className="panel">
          <h2>{text.numericCreate} ({unitLabels[unit]})</h2>
          <p className="selectionText">{text.absoluteCoordinates}</p>
          <div className="offsetGrid">
            <NumberField label="x" value={coordinateNode.x} onChange={(value) => setCoordinateNode((current) => ({ ...current, x: value }))} />
            <NumberField label="y" value={coordinateNode.y} onChange={(value) => setCoordinateNode((current) => ({ ...current, y: value }))} />
            <NumberField label="z" value={coordinateNode.z} onChange={(value) => setCoordinateNode((current) => ({ ...current, z: value }))} />
          </div>
          <button onClick={createCoordinateNode}><Plus size={17} />{text.createNode}</button>
          <p className="selectionText">{text.relativeToSelected} {selectedNode ? selectedNode : ""}</p>
          <div className="offsetGrid">
            <NumberField label="dx" value={offset.dx} onChange={(value) => setOffset((current) => ({ ...current, dx: value }))} />
            <NumberField label="dy" value={offset.dy} onChange={(value) => setOffset((current) => ({ ...current, dy: value }))} />
            <NumberField label="dz" value={offset.dz} onChange={(value) => setOffset((current) => ({ ...current, dz: value }))} />
          </div>
          <button onClick={createOffsetMember} disabled={!selectedNode}><Plus size={17} />{text.createAndConnect}</button>
        </section>

        <section className="panel">
          <h2>{text.nodeConditions} {selectedNode ? selectedNode : ""}</h2>
          <div className="segmented">
            <button
              className={model.nodes.find((node) => node.id === selectedNode)?.joint === "hinged" ? "active" : ""}
              onClick={() => setSelectedJoint("hinged")}
              disabled={!selectedNode}
            >
              {text.hinged}
            </button>
            <button
              className={model.nodes.find((node) => node.id === selectedNode)?.joint === "rigid" ? "active" : ""}
              onClick={() => setSelectedJoint("rigid")}
              disabled={!selectedNode}
            >
              {text.rigid}
            </button>
          </div>
          <div className="dofGrid">
            {dofKeys.map((key) => {
              return (
                <button key={key} className={selectedBoundary?.[key] ? "active" : ""} onClick={() => toggleBoundary(key)} disabled={!selectedNode}>
                  {key}
                </button>
              );
            })}
          </div>
          <div className="boundaryValueGrid">
            {dofKeys.map((key) => (
              <NumberField
                key={key}
                label={`${key} ${key.startsWith("u") ? unitLabels[unit] : "rad"}`}
                value={selectedBoundary?.values?.[key] ?? 0}
                onChange={(value) => updateBoundaryValue(key, value)}
                disabled={!selectedNode || !selectedBoundary?.[key]}
              />
            ))}
          </div>
          <div className="loadGrid">
            <NumberField label="Fx N" value={loadDraft.fx} onChange={(value) => setLoadDraft((current) => ({ ...current, fx: value }))} disabled={!selectedNode} />
            <NumberField label="Fy N" value={loadDraft.fy} onChange={(value) => setLoadDraft((current) => ({ ...current, fy: value }))} disabled={!selectedNode} />
            <NumberField label="Fz N" value={loadDraft.fz} onChange={(value) => setLoadDraft((current) => ({ ...current, fz: value }))} disabled={!selectedNode} />
          </div>
          <button onClick={confirmSelectedLoad} disabled={!selectedNode}><CircleDot size={17} />{text.confirmLoad}</button>
          <div className="massControls">
            <NumberField label="m kg" value={massDraft} onChange={setMassDraft} disabled={!selectedNode} />
            <button onClick={confirmSelectedMass} disabled={!selectedNode}><Plus size={17} />{text.confirmMass}</button>
            <button onClick={deleteSelectedMass} disabled={!selectedNode || !selectedMass}><Trash2 size={17} />{text.deleteMass}</button>
          </div>
        </section>

        <section className="panel">
          <h2>{text.display}</h2>
          <div className="offsetGrid">
            <NumberField label={text.staticScale} value={staticDeformScale} onChange={(value) => setStaticDeformScale(Math.max(0, value))} />
            <NumberField label={text.modalScale} value={modalDeformScale} onChange={(value) => setModalDeformScale(Math.max(0, value))} />
            <NumberField label={text.modalCount} value={modalModeCount} onChange={(value) => setModalModeCount(Math.max(1, Math.min(24, Math.round(value))))} />
          </div>
        </section>

        <div className="actions">
          <button className={result ? "active" : ""} onClick={runSolve}><Play size={18} />{text.solve}</button>
          <button className={modalResult ? "active" : ""} onClick={runModalAnalysis}><Sigma size={18} />{text.frequency}</button>
          <button onClick={clearModel}><Eraser size={18} />{text.clear}</button>
        </div>

        {error && <div className="error">{error}</div>}

      </aside>
    </main>
  );
}

function NumberField({ label, value, onChange, disabled = false }: { label: string; value: number; onChange: (value: number) => void; disabled?: boolean }) {
  const [draft, setDraft] = useState(() => String(Number.isFinite(value) ? value : 0));
  const focusedRef = useRef(false);

  useEffect(() => {
    if (focusedRef.current) return;
    setDraft(String(Number.isFinite(value) ? value : 0));
  }, [value]);

  const updateDraft = (text: string) => {
    setDraft(text);
    if (isCompleteNumberInput(text)) onChange(Number(text));
  };

  const normalizeDraft = () => {
    focusedRef.current = false;
    if (!isCompleteNumberInput(draft)) {
      setDraft(String(Number.isFinite(value) ? value : 0));
      return;
    }
    const next = Number(draft);
    setDraft(String(next));
    onChange(next);
  };

  return (
    <label className="numberField">
      <span>{label}</span>
      <input
        type="text"
        inputMode="decimal"
        value={draft}
        disabled={disabled}
        onFocus={() => { focusedRef.current = true; }}
        onBlur={normalizeDraft}
        onChange={(event) => updateDraft(event.target.value)}
      />
    </label>
  );
}

export default App;

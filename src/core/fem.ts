import { dofKeys, type DofKey, type ElementForce, type ElementLoad, type ModalResult, type StructureElement, type StructureModel, type StructureNode, type Vec3, type SolveResult } from "./types";
import { choleskyFactor, multiply, multiplyVector, solveCholeskyFactor, solveSymmetricPositiveDefinite, transpose, zeros, type Matrix, solveLinearSystem } from "./math";

const dofIndex: Record<DofKey, number> = {
  ux: 0,
  uy: 1,
  uz: 2,
  rx: 3,
  ry: 4,
  rz: 5,
};

function sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function length(v: Vec3): number {
  return Math.sqrt(dot(v, v));
}

function normalize(v: Vec3): Vec3 {
  const l = length(v);
  if (l < 1e-12) throw new Error("发现零长度方向向量。");
  return { x: v.x / l, y: v.y / l, z: v.z / l };
}

function localAxes(start: StructureNode, end: StructureNode, preferredY?: Vec3): { axes: Matrix; length: number } {
  const ex = normalize(sub(end, start));
  const reference = preferredY ?? (Math.abs(ex.z) < 0.9 ? { x: 0, y: 0, z: 1 } : { x: 0, y: 1, z: 0 });
  let ez = cross(ex, reference);
  if (length(ez) < 1e-9) ez = cross(ex, { x: 0, y: 1, z: 0 });
  ez = normalize(ez);
  const ey = normalize(cross(ez, ex));
  return {
    axes: [
      [ex.x, ex.y, ex.z],
      [ey.x, ey.y, ey.z],
      [ez.x, ez.y, ez.z],
    ],
    length: length(sub(end, start)),
  };
}

function transformationMatrix(axes: Matrix): Matrix {
  const t = zeros(12, 12);
  for (let block = 0; block < 4; block += 1) {
    const offset = block * 3;
    for (let r = 0; r < 3; r += 1) {
      for (let c = 0; c < 3; c += 1) t[offset + r][offset + c] = axes[r][c];
    }
  }
  return t;
}

function beamLocalStiffness(E: number, G: number, A: number, Iy: number, Iz: number, J: number, L: number): Matrix {
  const k = zeros(12, 12);
  const set = (i: number, j: number, value: number) => {
    k[i][j] += value;
    if (i !== j) k[j][i] += value;
  };

  const axial = (E * A) / L;
  set(0, 0, axial);
  set(0, 6, -axial);
  set(6, 6, axial);

  const torsion = (G * J) / L;
  set(3, 3, torsion);
  set(3, 9, -torsion);
  set(9, 9, torsion);

  const z12 = (12 * E * Iz) / L ** 3;
  const z6 = (6 * E * Iz) / L ** 2;
  const z4 = (4 * E * Iz) / L;
  const z2 = (2 * E * Iz) / L;
  set(1, 1, z12);
  set(1, 5, z6);
  set(1, 7, -z12);
  set(1, 11, z6);
  set(5, 5, z4);
  set(5, 7, -z6);
  set(5, 11, z2);
  set(7, 7, z12);
  set(7, 11, -z6);
  set(11, 11, z4);

  const y12 = (12 * E * Iy) / L ** 3;
  const y6 = (6 * E * Iy) / L ** 2;
  const y4 = (4 * E * Iy) / L;
  const y2 = (2 * E * Iy) / L;
  set(2, 2, y12);
  set(2, 4, -y6);
  set(2, 8, -y12);
  set(2, 10, -y6);
  set(4, 4, y4);
  set(4, 8, y6);
  set(4, 10, y2);
  set(8, 8, y12);
  set(8, 10, y6);
  set(10, 10, y4);

  return k;
}

function barLocalStiffness(E: number, A: number, L: number): Matrix {
  const k = zeros(12, 12);
  const axial = (E * A) / L;
  k[0][0] = axial;
  k[0][6] = -axial;
  k[6][0] = -axial;
  k[6][6] = axial;
  return k;
}

function releaseIndices(element: StructureElement): number[] {
  const indices: number[] = [];
  for (const key of dofKeys) {
    if (element.releaseStart?.[key]) indices.push(dofIndex[key]);
    if (element.releaseEnd?.[key]) indices.push(6 + dofIndex[key]);
  }
  return indices;
}

function applyStaticCondensation(k: Matrix, released: number[]): Matrix {
  if (released.length === 0) return k;
  const releasedSet = new Set(released);
  const retained = Array.from({ length: 12 }, (_, i) => i).filter((i) => !releasedSet.has(i));
  const krr = retained.map((r) => retained.map((c) => k[r][c]));
  const krc = retained.map((r) => released.map((c) => k[r][c]));
  const kcr = released.map((r) => retained.map((c) => k[r][c]));
  const kcc = released.map((r) => released.map((c) => k[r][c]));
  const correction = zeros(retained.length, retained.length);

  for (let col = 0; col < retained.length; col += 1) {
    const rhs = kcr.map((row) => row[col]);
    const solved = solveLinearSystem(kcc, rhs);
    for (let row = 0; row < retained.length; row += 1) {
      correction[row][col] = krc[row].reduce((sum, value, i) => sum + value * solved[i], 0);
    }
  }

  const condensed = zeros(12, 12);
  for (let i = 0; i < retained.length; i += 1) {
    for (let j = 0; j < retained.length; j += 1) {
      condensed[retained[i]][retained[j]] = krr[i][j] - correction[i][j];
    }
  }
  return condensed;
}

function globalToLocalVector(axes: Matrix, vector: Vec3): Vec3 {
  return {
    x: axes[0][0] * vector.x + axes[0][1] * vector.y + axes[0][2] * vector.z,
    y: axes[1][0] * vector.x + axes[1][1] * vector.y + axes[1][2] * vector.z,
    z: axes[2][0] * vector.x + axes[2][1] * vector.y + axes[2][2] * vector.z,
  };
}

function elementLoadLocalVector(load: ElementLoad, axes: Matrix, lengthValue: number): number[] {
  const out = Array(12).fill(0);
  if (load.type === "distributed") {
    const input = { x: load.wx ?? 0, y: load.wy ?? 0, z: load.wz ?? 0 };
    const q = load.coordinate === "global" ? globalToLocalVector(axes, input) : input;
    out[0] += q.x * lengthValue / 2;
    out[6] += q.x * lengthValue / 2;
    out[1] += q.y * lengthValue / 2;
    out[5] += q.y * lengthValue ** 2 / 12;
    out[7] += q.y * lengthValue / 2;
    out[11] += -q.y * lengthValue ** 2 / 12;
    out[2] += q.z * lengthValue / 2;
    out[4] += -q.z * lengthValue ** 2 / 12;
    out[8] += q.z * lengthValue / 2;
    out[10] += q.z * lengthValue ** 2 / 12;
    return out;
  }

  const input = { x: load.fx ?? 0, y: load.fy ?? 0, z: load.fz ?? 0 };
  const point = load.coordinate === "global" ? globalToLocalVector(axes, input) : input;
  const ratio = Math.min(1, Math.max(0, load.position));
  const n1 = 1 - 3 * ratio ** 2 + 2 * ratio ** 3;
  const n2 = lengthValue * (ratio - 2 * ratio ** 2 + ratio ** 3);
  const n3 = 3 * ratio ** 2 - 2 * ratio ** 3;
  const n4 = lengthValue * (-(ratio ** 2) + ratio ** 3);

  out[0] += point.x * (1 - ratio);
  out[6] += point.x * ratio;
  out[1] += point.y * n1;
  out[5] += point.y * n2;
  out[7] += point.y * n3;
  out[11] += point.y * n4;
  out[2] += point.z * n1;
  out[4] += -point.z * n2;
  out[8] += point.z * n3;
  out[10] += -point.z * n4;
  return out;
}

function releaseLoadVector(load: number[], released: number[]): number[] {
  if (released.length === 0) return load;
  const next = [...load];
  for (const index of released) next[index] = 0;
  return next;
}

function elementDofIndices(element: StructureElement, nodeOffset: Map<string, number>): number[] {
  const start = nodeOffset.get(element.startNodeId);
  const end = nodeOffset.get(element.endNodeId);
  if (start === undefined || end === undefined) throw new Error(`单元 ${element.id} 引用了不存在的节点。`);
  return [...Array.from({ length: 6 }, (_, i) => start + i), ...Array.from({ length: 6 }, (_, i) => end + i)];
}

function createElementMatrices(model: StructureModel, element: StructureElement, nodeMap: Map<string, StructureNode>) {
  const material = model.materials.find((item) => item.id === element.materialId);
  const section = model.sections.find((item) => item.id === element.sectionId);
  const start = nodeMap.get(element.startNodeId);
  const end = nodeMap.get(element.endNodeId);
  if (!material || !section || !start || !end) throw new Error(`单元 ${element.id} 属性不完整。`);

  const { axes, length: L } = localAxes(start, end, element.localY);
  if (L < 1e-9) throw new Error(`单元 ${element.id} 长度为 0。`);
  const baseLocal = element.type === "bar3d"
    ? barLocalStiffness(material.E, section.A, L)
    : beamLocalStiffness(material.E, material.G, section.A, section.Iy, section.Iz, section.J, L);
  const local = element.type === "beam3d" ? applyStaticCondensation(baseLocal, releaseIndices(element)) : baseLocal;
  const transform = transformationMatrix(axes);
  const global = multiply(multiply(transpose(transform), local), transform);
  return { local, transform, global, axes, length: L };
}

function activeModel(inputModel: StructureModel): { model: StructureModel; inactiveNodes: StructureNode[] } {
  const connectedNodeIds = new Set<string>();
  for (const element of inputModel.elements) {
    connectedNodeIds.add(element.startNodeId);
    connectedNodeIds.add(element.endNodeId);
  }
  if (inputModel.elements.length === 0) throw new Error("模型没有单元，无法求解。");

  const activeNodes = inputModel.nodes.filter((node) => connectedNodeIds.has(node.id));
  const inactiveNodes = inputModel.nodes.filter((node) => !connectedNodeIds.has(node.id));
  const baseModel: StructureModel = inactiveNodes.length === 0
    ? { ...inputModel, elementLoads: inputModel.elementLoads ?? [], nodalMasses: inputModel.nodalMasses ?? [] }
    : {
        ...inputModel,
        nodes: activeNodes,
        boundaries: inputModel.boundaries.filter((boundary) => connectedNodeIds.has(boundary.nodeId)),
        loads: inputModel.loads.filter((load) => connectedNodeIds.has(load.nodeId)),
        elementLoads: (inputModel.elementLoads ?? []).filter((load) => inputModel.elements.some((element) => element.id === load.elementId)),
        nodalMasses: (inputModel.nodalMasses ?? []).filter((mass) => connectedNodeIds.has(mass.nodeId)),
      };
  const nodeMap = new Map(baseModel.nodes.map((node) => [node.id, node]));
  const hasExplicitBendingConnection = (release: StructureElement["releaseStart"]) => release?.ry !== undefined || release?.rz !== undefined;
  const model: StructureModel = {
    ...baseModel,
    elements: baseModel.elements.map((element) => {
      if (element.type !== "beam3d") return element;
      const startHinged = hasExplicitBendingConnection(element.releaseStart)
        ? Boolean(element.releaseStart?.ry || element.releaseStart?.rz)
        : nodeMap.get(element.startNodeId)?.joint === "hinged";
      const endHinged = hasExplicitBendingConnection(element.releaseEnd)
        ? Boolean(element.releaseEnd?.ry || element.releaseEnd?.rz)
        : nodeMap.get(element.endNodeId)?.joint === "hinged";
      if (!startHinged && !endHinged) return element;
      return {
        ...element,
        releaseStart: startHinged ? { ...element.releaseStart, ry: true, rz: true } : element.releaseStart,
        releaseEnd: endHinged ? { ...element.releaseEnd, ry: true, rz: true } : element.releaseEnd,
      };
    }),
  };
  return { model, inactiveNodes };
}

function constrainedDofs(model: StructureModel, nodeOffset: Map<string, number>, stiffness: Matrix, force?: number[]): Set<number> {
  const constrained = new Set<number>();
  for (const boundary of model.boundaries) {
    const offset = nodeOffset.get(boundary.nodeId);
    if (offset === undefined) continue;
    for (const key of dofKeys) {
      if (boundary[key]) constrained.add(offset + dofIndex[key]);
    }
  }

  for (let i = 0; i < stiffness.length; i += 1) {
    const rowNorm = stiffness[i].reduce((sum, value) => sum + Math.abs(value), 0);
    if (rowNorm < 1e-9) {
      if (force && Math.abs(force[i]) > 1e-9) {
        throw new Error("荷载施加在无刚度自由度上，请检查杆单元转角荷载或缺失连接。");
      }
      constrained.add(i);
    }
  }
  return constrained;
}

function dofLabel(model: StructureModel, index: number): string {
  const node = model.nodes[Math.floor(index / 6)];
  return `${node?.id ?? "?"}.${dofKeys[index % 6]}`;
}

function prescribedValues(model: StructureModel, nodeOffset: Map<string, number>, totalDofs: number): number[] {
  const values = Array(totalDofs).fill(0);
  for (const boundary of model.boundaries) {
    const offset = nodeOffset.get(boundary.nodeId);
    if (offset === undefined) continue;
    for (const key of dofKeys) {
      if (boundary[key]) values[offset + dofIndex[key]] = boundary.values?.[key] ?? 0;
    }
  }
  return values;
}

function jacobiEigen(input: Matrix): { values: number[]; vectors: Matrix } {
  const n = input.length;
  const a = input.map((row) => [...row]);
  const vectors = zeros(n, n);
  for (let i = 0; i < n; i += 1) vectors[i][i] = 1;
  const maxIterations = Math.max(50, n * n * 80);
  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    let p = 0;
    let q = 1;
    let max = 0;
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const value = Math.abs(a[i][j]);
        if (value > max) {
          max = value;
          p = i;
          q = j;
        }
      }
    }
    if (max < 1e-8) break;

    const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
    const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;
    const app = a[p][p];
    const aqq = a[q][q];
    const apq = a[p][q];
    a[p][p] = app - t * apq;
    a[q][q] = aqq + t * apq;
    a[p][q] = 0;
    a[q][p] = 0;
    for (let k = 0; k < n; k += 1) {
      if (k === p || k === q) continue;
      const akp = a[k][p];
      const akq = a[k][q];
      a[k][p] = c * akp - s * akq;
      a[p][k] = a[k][p];
      a[k][q] = s * akp + c * akq;
      a[q][k] = a[k][q];
    }

    for (let k = 0; k < n; k += 1) {
      const vkp = vectors[k][p];
      const vkq = vectors[k][q];
      vectors[k][p] = c * vkp - s * vkq;
      vectors[k][q] = s * vkp + c * vkq;
    }
  }
  return { values: a.map((row, index) => row[index]), vectors };
}

function vectorDot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i];
  return sum;
}

function matrixVector(a: Matrix, v: number[]): number[] {
  const out = Array(a.length).fill(0);
  for (let i = 0; i < a.length; i += 1) {
    let sum = 0;
    for (let j = 0; j < v.length; j += 1) sum += a[i][j] * v[j];
    out[i] = sum;
  }
  return out;
}

function orthonormalize(vectors: number[][]): number[][] {
  const out: number[][] = [];
  for (const input of vectors) {
    const vector = [...input];
    for (const basis of out) {
      const projection = vectorDot(vector, basis);
      for (let i = 0; i < vector.length; i += 1) vector[i] -= projection * basis[i];
    }
    const norm = Math.sqrt(Math.max(0, vectorDot(vector, vector)));
    if (norm > 1e-10) out.push(vector.map((value) => value / norm));
  }
  return out;
}

function rayleighRitz(matrix: Matrix, basis: number[][]): { values: number[]; vectors: Matrix } {
  const projected = zeros(basis.length, basis.length);
  const matrixBasis = basis.map((vector) => matrixVector(matrix, vector));
  for (let i = 0; i < basis.length; i += 1) {
    for (let j = i; j < basis.length; j += 1) {
      const value = vectorDot(basis[i], matrixBasis[j]);
      projected[i][j] = value;
      projected[j][i] = value;
    }
  }
  const small = jacobiEigen(projected);
  const vectors = zeros(matrix.length, basis.length);
  for (let mode = 0; mode < basis.length; mode += 1) {
    for (let basisIndex = 0; basisIndex < basis.length; basisIndex += 1) {
      const coefficient = small.vectors[basisIndex][mode];
      for (let row = 0; row < matrix.length; row += 1) vectors[row][mode] += basis[basisIndex][row] * coefficient;
    }
    let normSquared = 0;
    for (let row = 0; row < matrix.length; row += 1) normSquared += vectors[row][mode] * vectors[row][mode];
    const norm = Math.sqrt(Math.max(1e-20, normSquared));
    for (let row = 0; row < matrix.length; row += 1) vectors[row][mode] /= norm;
  }
  return { values: small.values, vectors };
}

function lowestEigenPairs(matrix: Matrix, modeCount: number): { values: number[]; vectors: Matrix } {
  const n = matrix.length;
  if (n <= 90 || modeCount >= n / 2) return jacobiEigen(matrix);

  const blockSize = Math.min(n, Math.max(modeCount + 4, Math.min(16, modeCount * 2)));
  let basis: number[][] = [];
  for (let column = 0; column < blockSize; column += 1) {
    const vector = Array(n).fill(0).map((_, row) => {
      const phase = (row + 1) * (column + 1);
      return Math.sin(phase * 0.731) + 0.5 * Math.cos(phase * 0.317);
    });
    basis.push(vector);
  }
  basis = orthonormalize(basis);

  let factor: Matrix;
  try {
    factor = choleskyFactor(matrix);
  } catch {
    if (n > 140) throw new Error("模态刚度矩阵不是正定矩阵，模型可能存在机构或约束不足。");
    return jacobiEigen(matrix);
  }

  const iterations = 28;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    basis = orthonormalize(basis.map((vector) => solveCholeskyFactor(factor, vector)));
    if (basis.length < blockSize) {
      for (let column = basis.length; column < blockSize; column += 1) {
        const vector = Array(n).fill(0).map((_, row) => (row === column % n ? 1 : 0));
        basis = orthonormalize([...basis, vector]);
      }
    }
  }

  return rayleighRitz(matrix, basis);
}

export function solveStructure(inputModel: StructureModel): SolveResult {
  const { model, inactiveNodes } = activeModel(inputModel);

  const nodeMap = new Map(model.nodes.map((node) => [node.id, node]));
  const nodeOffset = new Map(model.nodes.map((node, index) => [node.id, index * 6]));
  const totalDofs = model.nodes.length * 6;
  const K = zeros(totalDofs, totalDofs);
  const F = Array(totalDofs).fill(0);
  const elementMatrices = new Map<string, ReturnType<typeof createElementMatrices>>();

  for (const element of model.elements) {
    const matrices = createElementMatrices(model, element, nodeMap);
    elementMatrices.set(element.id, matrices);
    const indices = elementDofIndices(element, nodeOffset);
    for (let i = 0; i < 12; i += 1) {
      for (let j = 0; j < 12; j += 1) {
        K[indices[i]][indices[j]] += matrices.global[i][j];
      }
    }
  }

  for (const load of model.loads) {
    const offset = nodeOffset.get(load.nodeId);
    if (offset === undefined) continue;
    F[offset + 0] += load.fx ?? 0;
    F[offset + 1] += load.fy ?? 0;
    F[offset + 2] += load.fz ?? 0;
    F[offset + 3] += load.mx ?? 0;
    F[offset + 4] += load.my ?? 0;
    F[offset + 5] += load.mz ?? 0;
  }

  const elementLoadVectors = new Map<string, number[]>();
  for (const load of model.elementLoads ?? []) {
    const element = model.elements.find((item) => item.id === load.elementId);
    if (!element || element.type !== "beam3d") continue;
    const matrices = elementMatrices.get(element.id);
    if (!matrices) continue;
    const localLoad = releaseLoadVector(elementLoadLocalVector(load, matrices.axes, matrices.length), releaseIndices(element));
    const globalLoad = multiplyVector(transpose(matrices.transform), localLoad);
    const indices = elementDofIndices(element, nodeOffset);
    for (let i = 0; i < 12; i += 1) {
      F[indices[i]] += globalLoad[i];
    }
    const existing = elementLoadVectors.get(element.id) ?? Array(12).fill(0);
    for (let i = 0; i < 12; i += 1) existing[i] += localLoad[i];
    elementLoadVectors.set(element.id, existing);
  }

  const constrained = constrainedDofs(model, nodeOffset, K, F);
  const prescribed = prescribedValues(model, nodeOffset, totalDofs);

  const free = Array.from({ length: totalDofs }, (_, i) => i).filter((i) => !constrained.has(i));
  const U = [...prescribed];
  if (free.length > 0) {
    const Kr = free.map((row) => free.map((col) => K[row][col]));
    const constrainedList = Array.from(constrained);
    const nonzeroPrescribed = constrainedList.filter((index) => Math.abs(prescribed[index]) > 0);
    const Fr = free.map((row) => F[row] - nonzeroPrescribed.reduce((sum, col) => sum + K[row][col] * prescribed[col], 0));
    let Ur: number[];
    try {
      Ur = solveSymmetricPositiveDefinite(Kr, Fr);
    } catch {
      try {
        Ur = solveLinearSystem(Kr, Fr);
      } catch (error) {
        const freeLabels = free.slice(0, 12).map((index) => dofLabel(model, index)).join(", ");
        throw new Error(`全局刚度矩阵奇异，模型存在刚体运动或机构。请检查支座方向、桁架三角支撑、梁端铰接释放是否过多。当前部分自由自由度：${freeLabels}${free.length > 12 ? " ..." : ""}`);
      }
    }
    free.forEach((index, i) => {
      U[index] = Ur[i];
    });
  }

  const KU = multiplyVector(K, U);
  const R = KU.map((value, index) => value - F[index]);

  const displacements: SolveResult["displacements"] = {};
  const reactions: SolveResult["reactions"] = {};
  let maxDisplacement = 0;
  for (const node of model.nodes) {
    const offset = nodeOffset.get(node.id)!;
    displacements[node.id] = {
      ux: U[offset],
      uy: U[offset + 1],
      uz: U[offset + 2],
      rx: U[offset + 3],
      ry: U[offset + 4],
      rz: U[offset + 5],
    };
    reactions[node.id] = {
      ux: R[offset],
      uy: R[offset + 1],
      uz: R[offset + 2],
      rx: R[offset + 3],
      ry: R[offset + 4],
      rz: R[offset + 5],
    };
    maxDisplacement = Math.max(maxDisplacement, Math.hypot(U[offset], U[offset + 1], U[offset + 2]));
  }

  const elementForces: ElementForce[] = model.elements.map((element) => {
    const matrices = elementMatrices.get(element.id)!;
    const indices = elementDofIndices(element, nodeOffset);
    const globalU = indices.map((index) => U[index]);
    const localU = multiplyVector(matrices.transform, globalU);
    const fixedEndLoad = elementLoadVectors.get(element.id) ?? Array(12).fill(0);
    const localEndForces = multiplyVector(matrices.local, localU).map((value, index) => value - fixedEndLoad[index]);
    return {
      elementId: element.id,
      type: element.type,
      localEndForces,
      axial: -localEndForces[0],
      shearY: element.type === "beam3d" ? -localEndForces[1] : undefined,
      shearZ: element.type === "beam3d" ? -localEndForces[2] : undefined,
      torsion: element.type === "beam3d" ? -localEndForces[3] : undefined,
      momentYStart: element.type === "beam3d" ? -localEndForces[4] : undefined,
      momentYEnd: element.type === "beam3d" ? localEndForces[10] : undefined,
      momentZStart: element.type === "beam3d" ? -localEndForces[5] : undefined,
      momentZEnd: element.type === "beam3d" ? localEndForces[11] : undefined,
    };
  });

  for (const node of inactiveNodes) {
    displacements[node.id] = { ux: 0, uy: 0, uz: 0, rx: 0, ry: 0, rz: 0 };
    reactions[node.id] = { ux: 0, uy: 0, uz: 0, rx: 0, ry: 0, rz: 0 };
  }

  return { displacements, reactions, elementForces, maxDisplacement };
}

export function solveModalAnalysis(inputModel: StructureModel, modeCount = 8): ModalResult {
  const { model } = activeModel(inputModel);
  const nodeMap = new Map(model.nodes.map((node) => [node.id, node]));
  const nodeOffset = new Map(model.nodes.map((node, index) => [node.id, index * 6]));
  const totalDofs = model.nodes.length * 6;
  const K = zeros(totalDofs, totalDofs);
  const M = Array(totalDofs).fill(0);

  for (const element of model.elements) {
    const matrices = createElementMatrices(model, element, nodeMap);
    const indices = elementDofIndices(element, nodeOffset);
    for (let i = 0; i < 12; i += 1) {
      for (let j = 0; j < 12; j += 1) K[indices[i]][indices[j]] += matrices.global[i][j];
    }

    const material = model.materials.find((item) => item.id === element.materialId);
    const section = model.sections.find((item) => item.id === element.sectionId);
    const elementMass = Math.max(0, material?.density ?? 0) * (section?.A ?? 0) * matrices.length;
    if (elementMass > 0) {
      for (const nodeBase of [indices[0], indices[6]]) {
        M[nodeBase + 0] += elementMass / 2;
        M[nodeBase + 1] += elementMass / 2;
        M[nodeBase + 2] += elementMass / 2;
      }
    }
  }

  for (const mass of model.nodalMasses ?? []) {
    const offset = nodeOffset.get(mass.nodeId);
    if (offset === undefined || mass.mass <= 0) continue;
    M[offset + 0] += mass.mass;
    M[offset + 1] += mass.mass;
    M[offset + 2] += mass.mass;
  }

  const constrained = constrainedDofs(model, nodeOffset, K);
  const free = Array.from({ length: totalDofs }, (_, i) => i).filter((i) => !constrained.has(i));
  const massDofs = free.filter((index) => M[index] > 1e-12);
  if (massDofs.length === 0) throw new Error("模型没有有效质量，请设置材料密度或节点集中质量。");

  const masslessDofs = free.filter((index) => M[index] <= 1e-12);
  let effectiveK = massDofs.map((row) => massDofs.map((col) => K[row][col]));
  let masslessRecoveryFactor: Matrix | null = null;
  if (masslessDofs.length > 0) {
    const kmm = effectiveK;
    const kmr = massDofs.map((row) => masslessDofs.map((col) => K[row][col]));
    const krm = masslessDofs.map((row) => massDofs.map((col) => K[row][col]));
    const krr = masslessDofs.map((row) => masslessDofs.map((col) => K[row][col]));
    try {
      masslessRecoveryFactor = choleskyFactor(krr);
    } catch {
      masslessRecoveryFactor = null;
    }
    const correction = zeros(massDofs.length, massDofs.length);
    for (let col = 0; col < massDofs.length; col += 1) {
      const rhs = krm.map((row) => row[col]);
      const solved = masslessRecoveryFactor ? solveCholeskyFactor(masslessRecoveryFactor, rhs) : solveLinearSystem(krr, rhs);
      for (let row = 0; row < massDofs.length; row += 1) {
        correction[row][col] = kmr[row].reduce((sum, value, i) => sum + value * solved[i], 0);
      }
    }
    effectiveK = kmm.map((row, i) => row.map((value, j) => value - correction[i][j]));
  }

  const normalized = effectiveK.map((row, i) => row.map((value, j) => value / Math.sqrt(M[massDofs[i]] * M[massDofs[j]])));
  const eigen = lowestEigenPairs(normalized, modeCount);
  const modes = eigen.values
    .map((value, index) => ({ value, index }))
    .filter(({ value }) => value > 1e-8)
    .sort((a, b) => a.value - b.value)
    .slice(0, modeCount);

  return {
    modes: modes.map(({ value: eigenvalue, index: eigenIndex }, index) => {
      const omega = Math.sqrt(eigenvalue);
      const U = Array(totalDofs).fill(0);
      for (let i = 0; i < massDofs.length; i += 1) {
        U[massDofs[i]] = eigen.vectors[i][eigenIndex] / Math.sqrt(M[massDofs[i]]);
      }
      if (masslessDofs.length > 0) {
        const rhs = masslessDofs.map((row) => massDofs.reduce((sum, col) => sum + K[row][col] * U[col], 0));
        const krr = masslessDofs.map((row) => masslessDofs.map((col) => K[row][col]));
        const solved = masslessRecoveryFactor
          ? solveCholeskyFactor(masslessRecoveryFactor, rhs.map((value) => -value))
          : solveLinearSystem(krr, rhs.map((value) => -value));
        for (let i = 0; i < masslessDofs.length; i += 1) U[masslessDofs[i]] = solved[i];
      }
      const maxAmplitude = Math.max(1e-12, ...model.nodes.map((_, nodeIndex) => {
        const offset = nodeIndex * 6;
        return Math.hypot(U[offset], U[offset + 1], U[offset + 2]);
      }));
      const displacements: Record<string, Record<DofKey, number>> = {};
      for (const node of inputModel.nodes) {
        const offset = nodeOffset.get(node.id);
        displacements[node.id] = offset === undefined
          ? { ux: 0, uy: 0, uz: 0, rx: 0, ry: 0, rz: 0 }
          : {
              ux: U[offset] / maxAmplitude,
              uy: U[offset + 1] / maxAmplitude,
              uz: U[offset + 2] / maxAmplitude,
              rx: U[offset + 3] / maxAmplitude,
              ry: U[offset + 4] / maxAmplitude,
              rz: U[offset + 5] / maxAmplitude,
            };
      }
      return {
        mode: index + 1,
        eigenvalue,
        omega,
        frequency: omega / (2 * Math.PI),
        displacements,
        maxDisplacement: 1,
      };
    }),
  };
}

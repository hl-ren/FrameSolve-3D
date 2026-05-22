import { solveModalAnalysis, solveStructure } from "../src/core/fem";
import type { StructureModel } from "../src/core/types";

const E = 200e9;
const G = 80e9;
const A = 0.01;
const Iz = 8e-6;
const Iy = 8e-6;
const J = 1e-5;

type Check = {
  name: string;
  actual: number;
  expected: number;
  tolerance: number;
};

function modelBase(): Pick<StructureModel, "materials" | "sections"> {
  return {
    materials: [{ id: "mat", name: "steel", E, G, density: 7850 }],
    sections: [{ id: "sec", name: "test", A, Iy, Iz, J }],
  };
}

function assertClose({ name, actual, expected, tolerance }: Check): void {
  const error = Math.abs(actual - expected);
  if (error > tolerance) {
    throw new Error(`${name}: actual=${actual}, expected=${expected}, error=${error}, tolerance=${tolerance}`);
  }
  console.log(`PASS ${name}: actual=${actual.toExponential(6)}, expected=${expected.toExponential(6)}, error=${error.toExponential(3)}`);
}

function axialBarTest(): void {
  const L = 2;
  const P = 1000;
  const model: StructureModel = {
    ...modelBase(),
    nodes: [
      { id: "N1", x: 0, y: 0, z: 0 },
      { id: "N2", x: L, y: 0, z: 0 },
    ],
    elements: [{ id: "E1", type: "bar3d", startNodeId: "N1", endNodeId: "N2", materialId: "mat", sectionId: "sec" }],
    boundaries: [{ nodeId: "N1", ux: true }],
    loads: [{ nodeId: "N2", fx: P }],
    elementLoads: [],
    nodalMasses: [],
  };
  const result = solveStructure(model);
  assertClose({ name: "axial bar free-end displacement", actual: result.displacements.N2.ux, expected: P * L / (E * A), tolerance: 1e-12 });
  assertClose({ name: "axial bar fixed reaction", actual: result.reactions.N1.ux, expected: -P, tolerance: 1e-6 });
  assertClose({ name: "axial bar tension force", actual: result.elementForces[0].axial, expected: P, tolerance: 1e-6 });
}

function cantileverEndPointTest(): void {
  const L = 3;
  const P = -1000;
  const model: StructureModel = {
    ...modelBase(),
    nodes: [
      { id: "N1", x: 0, y: 0, z: 0 },
      { id: "N2", x: L, y: 0, z: 0 },
    ],
    elements: [{ id: "E1", type: "beam3d", startNodeId: "N1", endNodeId: "N2", materialId: "mat", sectionId: "sec" }],
    boundaries: [{ nodeId: "N1", ux: true, uy: true, uz: true, rx: true, ry: true, rz: true }],
    loads: [{ nodeId: "N2", fz: P }],
    elementLoads: [],
    nodalMasses: [],
  };
  const result = solveStructure(model);
  assertClose({ name: "cantilever end point load displacement", actual: result.displacements.N2.uz, expected: P * L ** 3 / (3 * E * Iz), tolerance: 1e-10 });
  assertClose({ name: "cantilever end point load reaction", actual: result.reactions.N1.uz, expected: -P, tolerance: 1e-6 });
  assertClose({ name: "cantilever end point load fixed-end Mz", actual: result.elementForces[0].momentZStart ?? Number.NaN, expected: P * L, tolerance: 1e-6 });
}

function cantileverDistributedTest(): void {
  const L = 3;
  const q = -500;
  const model: StructureModel = {
    ...modelBase(),
    nodes: [
      { id: "N1", x: 0, y: 0, z: 0 },
      { id: "N2", x: L, y: 0, z: 0 },
    ],
    elements: [{ id: "E1", type: "beam3d", startNodeId: "N1", endNodeId: "N2", materialId: "mat", sectionId: "sec" }],
    boundaries: [{ nodeId: "N1", ux: true, uy: true, uz: true, rx: true, ry: true, rz: true }],
    loads: [],
    elementLoads: [{ id: "L1", elementId: "E1", type: "distributed", coordinate: "global", wz: q }],
    nodalMasses: [],
  };
  const result = solveStructure(model);
  assertClose({ name: "cantilever uniform load displacement", actual: result.displacements.N2.uz, expected: q * L ** 4 / (8 * E * Iz), tolerance: 1e-10 });
  assertClose({ name: "cantilever uniform load reaction", actual: result.reactions.N1.uz, expected: -q * L, tolerance: 1e-6 });
  assertClose({ name: "cantilever uniform load fixed-end Mz", actual: result.elementForces[0].momentZStart ?? Number.NaN, expected: q * L ** 2 / 2, tolerance: 1e-6 });
}

function cantileverElementPointTest(): void {
  const L = 3;
  const a = 1.5;
  const P = -1000;
  const model: StructureModel = {
    ...modelBase(),
    nodes: [
      { id: "N1", x: 0, y: 0, z: 0 },
      { id: "N2", x: L, y: 0, z: 0 },
    ],
    elements: [{ id: "E1", type: "beam3d", startNodeId: "N1", endNodeId: "N2", materialId: "mat", sectionId: "sec" }],
    boundaries: [{ nodeId: "N1", ux: true, uy: true, uz: true, rx: true, ry: true, rz: true }],
    loads: [],
    elementLoads: [{ id: "L1", elementId: "E1", type: "point", coordinate: "global", position: a / L, fz: P }],
    nodalMasses: [],
  };
  const result = solveStructure(model);
  assertClose({ name: "cantilever internal point load displacement", actual: result.displacements.N2.uz, expected: P * a ** 2 * (3 * L - a) / (6 * E * Iz), tolerance: 1e-10 });
  assertClose({ name: "cantilever internal point load reaction", actual: result.reactions.N1.uz, expected: -P, tolerance: 1e-6 });
  assertClose({ name: "cantilever internal point load fixed-end Mz", actual: result.elementForces[0].momentZStart ?? Number.NaN, expected: P * a, tolerance: 1e-6 });
}

function simplySupportedMidpointTest(): void {
  const L = 4;
  const P = -1200;
  const model: StructureModel = {
    ...modelBase(),
    nodes: [
      { id: "N1", x: 0, y: 0, z: 0 },
      { id: "N2", x: L, y: 0, z: 0 },
    ],
    elements: [{ id: "E1", type: "beam3d", startNodeId: "N1", endNodeId: "N2", materialId: "mat", sectionId: "sec" }],
    boundaries: [
      { nodeId: "N1", ux: true, uy: true, uz: true, rx: true },
      { nodeId: "N2", uy: true, uz: true },
    ],
    loads: [],
    elementLoads: [{ id: "L1", elementId: "E1", type: "point", coordinate: "global", position: 0.5, fz: P }],
    nodalMasses: [],
  };
  const result = solveStructure(model);
  assertClose({ name: "simple beam midpoint load left reaction", actual: result.reactions.N1.uz, expected: -P / 2, tolerance: 1e-6 });
  assertClose({ name: "simple beam midpoint load right reaction", actual: result.reactions.N2.uz, expected: -P / 2, tolerance: 1e-6 });
}

function axialFrequencyTest(): void {
  const L = 2;
  const m = 25;
  const model: StructureModel = {
    ...modelBase(),
    materials: [{ id: "mat", name: "steel", E, G, density: 0 }],
    nodes: [
      { id: "N1", x: 0, y: 0, z: 0 },
      { id: "N2", x: L, y: 0, z: 0 },
    ],
    elements: [{ id: "E1", type: "bar3d", startNodeId: "N1", endNodeId: "N2", materialId: "mat", sectionId: "sec" }],
    boundaries: [{ nodeId: "N1", ux: true }],
    loads: [],
    elementLoads: [],
    nodalMasses: [{ nodeId: "N2", mass: m }],
  };
  const result = solveModalAnalysis(model, 1);
  const eigenvalue = (E * A / L) / m;
  assertClose({ name: "axial spring-mass eigenvalue", actual: result.modes[0].eigenvalue, expected: eigenvalue, tolerance: 1e-3 });
  assertClose({ name: "axial spring-mass frequency", actual: result.modes[0].frequency, expected: Math.sqrt(eigenvalue) / (2 * Math.PI), tolerance: 1e-8 });
  assertClose({ name: "axial spring-mass mode shape support", actual: result.modes[0].displacements.N1.ux, expected: 0, tolerance: 1e-12 });
  assertClose({ name: "axial spring-mass mode shape tip", actual: Math.abs(result.modes[0].displacements.N2.ux), expected: 1, tolerance: 1e-12 });
}

function prescribedDisplacementTest(): void {
  const L = 2;
  const prescribed = 0.001;
  const model: StructureModel = {
    ...modelBase(),
    nodes: [
      { id: "N1", x: 0, y: 0, z: 0 },
      { id: "N2", x: L, y: 0, z: 0 },
    ],
    elements: [{ id: "E1", type: "bar3d", startNodeId: "N1", endNodeId: "N2", materialId: "mat", sectionId: "sec" }],
    boundaries: [
      { nodeId: "N1", ux: true },
      { nodeId: "N2", ux: true, values: { ux: prescribed } },
    ],
    loads: [],
    elementLoads: [],
    nodalMasses: [],
  };
  const result = solveStructure(model);
  const reaction = E * A * prescribed / L;
  assertClose({ name: "prescribed displacement value", actual: result.displacements.N2.ux, expected: prescribed, tolerance: 1e-12 });
  assertClose({ name: "prescribed displacement fixed-end reaction", actual: result.reactions.N1.ux, expected: -reaction, tolerance: 1e-6 });
  assertClose({ name: "prescribed displacement moving-end reaction", actual: result.reactions.N2.ux, expected: reaction, tolerance: 1e-6 });
}

function twoDofSpringMassFrequencyTest(): void {
  const L = 2;
  const m = 25;
  const k = E * A / L;
  const model: StructureModel = {
    ...modelBase(),
    materials: [{ id: "mat", name: "steel", E, G, density: 0 }],
    nodes: [
      { id: "N1", x: 0, y: 0, z: 0 },
      { id: "N2", x: L, y: 0, z: 0 },
      { id: "N3", x: 2 * L, y: 0, z: 0 },
    ],
    elements: [
      { id: "E1", type: "bar3d", startNodeId: "N1", endNodeId: "N2", materialId: "mat", sectionId: "sec" },
      { id: "E2", type: "bar3d", startNodeId: "N2", endNodeId: "N3", materialId: "mat", sectionId: "sec" },
    ],
    boundaries: [{ nodeId: "N1", ux: true }],
    loads: [],
    elementLoads: [],
    nodalMasses: [
      { nodeId: "N2", mass: m },
      { nodeId: "N3", mass: m },
    ],
  };
  const result = solveModalAnalysis(model, 2);
  const low = (k / m) * ((3 - Math.sqrt(5)) / 2);
  const high = (k / m) * ((3 + Math.sqrt(5)) / 2);
  assertClose({ name: "two-dof spring-mass first eigenvalue", actual: result.modes[0].eigenvalue, expected: low, tolerance: 1e-3 });
  assertClose({ name: "two-dof spring-mass second eigenvalue", actual: result.modes[1].eigenvalue, expected: high, tolerance: 1e-3 });
}

function densityLumpedMassFrequencyTest(): void {
  const L = 2;
  const density = 7850;
  const model: StructureModel = {
    ...modelBase(),
    materials: [{ id: "mat", name: "steel", E, G, density }],
    nodes: [
      { id: "N1", x: 0, y: 0, z: 0 },
      { id: "N2", x: L, y: 0, z: 0 },
    ],
    elements: [{ id: "E1", type: "bar3d", startNodeId: "N1", endNodeId: "N2", materialId: "mat", sectionId: "sec" }],
    boundaries: [{ nodeId: "N1", ux: true }],
    loads: [],
    elementLoads: [],
    nodalMasses: [],
  };
  const result = solveModalAnalysis(model, 1);
  const freeEndMass = density * A * L / 2;
  const eigenvalue = (E * A / L) / freeEndMass;
  assertClose({ name: "density lumped mass axial eigenvalue", actual: result.modes[0].eigenvalue, expected: eigenvalue, tolerance: 1e-3 });
}

function twoElementDensityFrequencyTest(): void {
  const elementLength = 1.5;
  const density = 7850;
  const k = E * A / elementLength;
  const referenceMass = density * A * elementLength;
  const model: StructureModel = {
    ...modelBase(),
    materials: [{ id: "mat", name: "steel", E, G, density }],
    nodes: [
      { id: "N1", x: 0, y: 0, z: 0 },
      { id: "N2", x: elementLength, y: 0, z: 0 },
      { id: "N3", x: 2 * elementLength, y: 0, z: 0 },
    ],
    elements: [
      { id: "E1", type: "bar3d", startNodeId: "N1", endNodeId: "N2", materialId: "mat", sectionId: "sec" },
      { id: "E2", type: "bar3d", startNodeId: "N2", endNodeId: "N3", materialId: "mat", sectionId: "sec" },
    ],
    boundaries: [{ nodeId: "N1", ux: true }],
    loads: [],
    elementLoads: [],
    nodalMasses: [],
  };
  const result = solveModalAnalysis(model, 2);
  const low = (k / referenceMass) * (2 - Math.sqrt(2));
  const high = (k / referenceMass) * (2 + Math.sqrt(2));
  assertClose({ name: "two-element density first eigenvalue", actual: result.modes[0].eigenvalue, expected: low, tolerance: 1e-3 });
  assertClose({ name: "two-element density second eigenvalue", actual: result.modes[1].eigenvalue, expected: high, tolerance: 1e-3 });
}

function cantileverTipMassBendingFrequencyTest(): void {
  const L = 3;
  const tipMass = 40;
  const model: StructureModel = {
    ...modelBase(),
    materials: [{ id: "mat", name: "steel", E, G, density: 0 }],
    nodes: [
      { id: "N1", x: 0, y: 0, z: 0 },
      { id: "N2", x: L, y: 0, z: 0 },
    ],
    elements: [{ id: "E1", type: "beam3d", startNodeId: "N1", endNodeId: "N2", materialId: "mat", sectionId: "sec" }],
    boundaries: [
      { nodeId: "N1", ux: true, uy: true, uz: true, rx: true, ry: true, rz: true },
      { nodeId: "N2", ux: true, uy: true },
    ],
    loads: [],
    elementLoads: [],
    nodalMasses: [{ nodeId: "N2", mass: tipMass }],
  };
  const result = solveModalAnalysis(model, 1);
  const eigenvalue = (3 * E * Iz / L ** 3) / tipMass;
  assertClose({ name: "cantilever tip-mass bending eigenvalue", actual: result.modes[0].eigenvalue, expected: eigenvalue, tolerance: 1e-3 });
  assertClose({ name: "cantilever tip-mass bending frequency", actual: result.modes[0].frequency, expected: Math.sqrt(eigenvalue) / (2 * Math.PI), tolerance: 1e-8 });
}

axialBarTest();
prescribedDisplacementTest();
cantileverEndPointTest();
cantileverDistributedTest();
cantileverElementPointTest();
simplySupportedMidpointTest();
axialFrequencyTest();
twoDofSpringMassFrequencyTest();
densityLumpedMassFrequencyTest();
twoElementDensityFrequencyTest();
cantileverTipMassBendingFrequencyTest();
console.log("All FrameSolve 3D solver checks passed.");

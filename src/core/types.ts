export type DofKey = "ux" | "uy" | "uz" | "rx" | "ry" | "rz";

export type Vec3 = {
  x: number;
  y: number;
  z: number;
};

export type StructureNode = Vec3 & {
  id: string;
  joint?: "hinged" | "rigid";
};

export type Material = {
  id: string;
  name: string;
  E: number;
  G: number;
  density: number;
};

export type Section = {
  id: string;
  name: string;
  width?: number;
  height?: number;
  A: number;
  Iy: number;
  Iz: number;
  J: number;
};

export type ElementType = "bar3d" | "beam3d";

export type ElementRelease = Partial<Record<DofKey, boolean>>;

export type StructureElement = {
  id: string;
  type: ElementType;
  startNodeId: string;
  endNodeId: string;
  materialId: string;
  sectionId: string;
  localY?: Vec3;
  releaseStart?: ElementRelease;
  releaseEnd?: ElementRelease;
};

export type BoundaryCondition = {
  nodeId: string;
  values?: Partial<Record<DofKey, number>>;
} & Partial<Record<DofKey, boolean>>;

export type NodalLoad = {
  nodeId: string;
  fx?: number;
  fy?: number;
  fz?: number;
  mx?: number;
  my?: number;
  mz?: number;
};

export type NodalMass = {
  nodeId: string;
  mass: number;
};

export type LoadCoordinate = "global" | "local";

export type ElementLoad = {
  id: string;
  elementId: string;
  coordinate: LoadCoordinate;
} & (
  | {
      type: "point";
      position: number;
      fx?: number;
      fy?: number;
      fz?: number;
    }
  | {
      type: "distributed";
      wx?: number;
      wy?: number;
      wz?: number;
    }
);

export type StructureModel = {
  nodes: StructureNode[];
  elements: StructureElement[];
  materials: Material[];
  sections: Section[];
  boundaries: BoundaryCondition[];
  loads: NodalLoad[];
  elementLoads: ElementLoad[];
  nodalMasses: NodalMass[];
};

export type ElementForce = {
  elementId: string;
  type: ElementType;
  localEndForces: number[];
  axial: number;
  shearY?: number;
  shearZ?: number;
  torsion?: number;
  momentYStart?: number;
  momentYEnd?: number;
  momentZStart?: number;
  momentZEnd?: number;
};

export type SolveResult = {
  displacements: Record<string, Record<DofKey, number>>;
  reactions: Record<string, Record<DofKey, number>>;
  elementForces: ElementForce[];
  maxDisplacement: number;
};

export type ModalMode = {
  mode: number;
  eigenvalue: number;
  omega: number;
  frequency: number;
  displacements: Record<string, Record<DofKey, number>>;
  maxDisplacement: number;
};

export type ModalResult = {
  modes: ModalMode[];
};

export const dofKeys: DofKey[] = ["ux", "uy", "uz", "rx", "ry", "rz"];

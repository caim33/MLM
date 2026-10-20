// Coordinate transforms never alter the decoded motion buffer.
export const JOINTS = 22;
export const PAGE_SIZE = 40;
export const DATASETS = ["humanml3d", "motionx", "sonic"];
export const FILTERS = ["all", "text", "video", "cot"];

export function rootOffset(positions, frame, mode = "world") {
  const index = mode === "inplace" ? frame * JOINTS * 3 : 0;
  return [positions[index], positions[index + 2]];
}

export function displayedJoint(positions, frame, joint, floor, mode, target) {
  const index = (frame * JOINTS + joint) * 3;
  const [x, z] = rootOffset(positions, frame, mode);
  return target.set(positions[index] - x, positions[index + 1] - floor, positions[index + 2] - z);
}

// The existing mesh API already preserves root height/yaw but centres X/Z.
export function meshTranslation(positions, frame, mode) {
  if (mode === "inplace") return [0, 0, 0];
  const index = frame * JOINTS * 3;
  return [positions[index] - positions[0], 0, positions[index + 2] - positions[2]];
}

export function analyzeMotion(positions) {
  if (!positions.length || positions.length % (JOINTS * 3)) throw new Error("动作数据尺寸不正确");
  let floor = Infinity;
  let maxY = -Infinity;
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  let localMinX = Infinity, localMaxX = -Infinity, localMinZ = Infinity, localMaxZ = -Infinity;
  let pathLength = 0;
  const frames = positions.length / (JOINTS * 3);
  const trajectory = new Float32Array(frames * 3);
  for (let frame = 0; frame < frames; frame++) {
    const root = frame * JOINTS * 3;
    const x = positions[root] - positions[0], z = positions[root + 2] - positions[2];
    trajectory.set([x, 0.018, z], frame * 3);
    if (frame) pathLength += Math.hypot(x - trajectory[(frame - 1) * 3], z - trajectory[(frame - 1) * 3 + 2]);
    for (let joint = 0; joint < JOINTS; joint++) {
      const i = root + joint * 3;
      const px = positions[i], py = positions[i + 1], pz = positions[i + 2];
      if (![px, py, pz].every(Number.isFinite)) throw new Error("动作数据含无效坐标");
      floor = Math.min(floor, py); maxY = Math.max(maxY, py);
      minX = Math.min(minX, px - positions[0]); maxX = Math.max(maxX, px - positions[0]);
      minZ = Math.min(minZ, pz - positions[2]); maxZ = Math.max(maxZ, pz - positions[2]);
      localMinX = Math.min(localMinX, px - positions[root]); localMaxX = Math.max(localMaxX, px - positions[root]);
      localMinZ = Math.min(localMinZ, pz - positions[root + 2]); localMaxZ = Math.max(localMaxZ, pz - positions[root + 2]);
    }
  }
  return {
    floor, height: maxY - floor, trajectory, pathLength,
    netDistance: Math.hypot(trajectory.at(-3), trajectory.at(-1)),
    world: { minX, maxX, minZ, maxZ },
    inplace: { minX: localMinX, maxX: localMaxX, minZ: localMinZ, maxZ: localMaxZ },
  };
}

export function adjacentSample(page, index, delta, total) {
  if (index < 0 || !total) return null;
  const absolute = page * PAGE_SIZE + index + delta;
  if (absolute < 0 || absolute >= total) return null;
  return { page: Math.floor(absolute / PAGE_SIZE), index: absolute % PAGE_SIZE };
}

export function readLocation(search) {
  const p = new URLSearchParams(search);
  const dataset = DATASETS.includes(p.get("dataset")) ? p.get("dataset") : "humanml3d";
  let filter = FILTERS.includes(p.get("filter")) ? p.get("filter") : "all";
  if (dataset !== "motionx" && ["video", "cot"].includes(filter)) filter = "all";
  const page = Number(p.get("page") || 1);
  const id = p.get("id") || "";
  return {
    dataset, filter, query: (p.get("q") || "").slice(0, 120),
    page: Number.isSafeInteger(page) && page > 0 ? page - 1 : 0,
    selectedId: /^[A-Za-z0-9_.-]{1,200}$/.test(id) ? id : null,
    rootMode: p.get("root") === "inplace" ? "inplace" : "world",
  };
}

export function locationQuery(state) {
  const p = new URLSearchParams({ dataset: state.dataset, page: String(state.page + 1) });
  if (state.selectedId) p.set("id", state.selectedId);
  if (state.query) p.set("q", state.query);
  if (state.filter !== "all") p.set("filter", state.filter);
  if (state.rootMode === "inplace") p.set("root", "inplace");
  return `?${p}`;
}

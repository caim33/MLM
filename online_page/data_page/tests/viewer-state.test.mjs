import test from "node:test";
import assert from "node:assert/strict";
import { analyzeMotion, displayedJoint, meshTranslation, adjacentSample, readLocation, locationQuery } from "../static/viewer-state.mjs";

const point = () => ({ set(x,y,z) { this.x=x; this.y=y; this.z=z; return this; } });
function fixture() {
  // Root starts away from zero; the last frame also changes height and facing.
  const roots = [[10,1,20],[12,1.2,21],[13,.9,24]];
  const values = new Float32Array(3*22*3);
  roots.forEach((root,frame) => {
    for (let joint=0;joint<22;joint++) {
      const local = joint ? (frame===2 ? [0,.1,1] : [1,.1,0]) : [0,0,0];
      values.set(root.map((value,i) => value+local[i]),(frame*22+joint)*3);
    }
  });
  return values;
}
const xyz = p => [p.x,p.y,p.z];
const near = (a,b) => assert.ok(Math.abs(a-b)<1e-5,`${a} != ${b}`);

test("world mode preserves root displacement, vertical motion and facing", () => {
  const data=fixture(), original=data.slice(), a=analyzeMotion(data);
  const first=displayedJoint(data,0,0,a.floor,"world",point());
  const last=displayedJoint(data,2,0,a.floor,"world",point());
  assert.deepEqual([first.x,first.z],[0,0]);
  assert.deepEqual([last.x,last.z],[3,4]);
  near(last.y-first.y,-.1);
  const child=displayedJoint(data,2,1,a.floor,"world",point());
  near(child.x-last.x,0); near(child.z-last.z,1);
  assert.deepEqual(data,original);
});
test("in-place mode removes X/Z translation only, preserving root height and joint orientation", () => {
  const data=fixture(), a=analyzeMotion(data);
  for (let f=0;f<3;f++) {
    const root=displayedJoint(data,f,0,a.floor,"inplace",point());
    assert.deepEqual([root.x,root.z],[0,0]);
    near(root.y,data[f*66+1]-a.floor);
  }
  assert.deepEqual(xyz(displayedJoint(data,2,1,a.floor,"inplace",point())).map(n=>Math.round(n*10)/10),[0,.1,1]);
});
test("root-centred SMPL-H meshes receive the same displayed root translation", () => {
  const data=fixture(), a=analyzeMotion(data);
  for (const mode of ["world","inplace"]) for (let f=0;f<3;f++) {
    const root=displayedJoint(data,f,0,a.floor,mode,point()), translation=meshTranslation(data,f,mode);
    assert.deepEqual([translation[0],translation[2]],[root.x,root.z]);
    assert.equal(translation[1],0); // The API already supplies height, so never add it twice.
  }
});
test("trajectory measures the full path, not just start/end distance", () => {
  const a=analyzeMotion(fixture());
  near(a.pathLength,Math.sqrt(5)+Math.sqrt(10));
  near(a.netDistance,5);
  assert.deepEqual([a.trajectory[0],a.trajectory[2],a.trajectory[6],a.trajectory[8]],[0,0,3,4]);
  assert.ok(a.world.maxZ>a.inplace.maxZ);
});
test("invalid shape and nonfinite coordinates fail before rendering", () => {
  assert.throws(()=>analyzeMotion(new Float32Array()),/尺寸/);
  assert.throws(()=>analyzeMotion(new Float32Array(67)),/尺寸/);
  const data=fixture(); data[10]=NaN;
  assert.throws(()=>analyzeMotion(data),/无效/);
});
test("sample navigation crosses page boundaries and handles final partial page", () => {
  assert.deepEqual(adjacentSample(0,39,1,81),{page:1,index:0});
  assert.deepEqual(adjacentSample(1,0,-1,81),{page:0,index:39});
  assert.deepEqual(adjacentSample(1,39,1,81),{page:2,index:0});
  assert.equal(adjacentSample(2,0,1,81),null);
  assert.equal(adjacentSample(0,0,-1,81),null);
  assert.equal(adjacentSample(0,-1,1,81),null);
});
test("shared sample links retain dataset, query, filter, page and root display", () => {
  const original={dataset:"motionx",filter:"video",query:"walk 001",page:12,selectedId:"sample_123",rootMode:"inplace"};
  assert.deepEqual(readLocation(locationQuery(original)),original);
});
test("invalid link settings use valid defaults and reject path traversal IDs", () => {
  assert.deepEqual(readLocation("?dataset=other&id=../secret&page=Infinity&root=wrong&filter=video"),{
    dataset:"humanml3d",filter:"all",query:"",page:0,selectedId:null,rootMode:"world",
  });
});

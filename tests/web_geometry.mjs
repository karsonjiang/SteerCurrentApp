// No frontend framework or npm dependencies are needed to exercise the editor math.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../src/steercurrent/web/static/geometry.js',import.meta.url));
const {preset,position,nearestDistance,segments,reshape,constrainedReshape,spanPoints,handles,shapeKind,moveResistor,resizeResistor}=await import(`data:text/javascript;base64,${source.toString('base64')}`);
for(const kind of ['step','rectangle','u','stairs'])for(const length of [24,40,80]){
  const g=preset(kind,length);
  assert.equal(shapeKind(g),kind);
  for(const h of handles(g)){
    const handle=h.id,point=[...h.point];
    for(const axis of h.axes){
      const grows=handle==='bottom'||handle.startsWith('indent');
      point[axis]+=(axis===0?(handle==='right'&&length===80?-.04:.04):(length===80?-.04:.04)*(grows?-1:1));
    }
    const next=reshape(g,handle,point);
    assert.ok(next);
    assert.equal(shapeKind(next),kind,'Dragging must preserve the selected shape family.');
    assert.ok(Math.abs(segments(next).reduce((sum,s)=>sum+s.length,0)-next.config.length)<1e-10);
    for(const [i,r] of g.config.resistors.entries()){
      const changed=next.config.resistors[i];
      const before=segments(g),after=segments(next);
      for(const end of ['start','end']){
        const segment=before.findIndex(e=>r[end]<=e.start+e.length+1e-10);
        const fraction=(r[end]-before[segment].start)/before[segment].length;
        const expected=after[segment].start+fraction*after[segment].length;
        assert.ok(Math.abs(changed[end]-expected)<1e-10,'Each resistor end stays attached to its wire section.');
      }
    }
    assert.deepEqual(next.config.battery,g.config.battery);
    for(const edge of segments(next))assert.equal((edge.a[0]!==edge.b[0])+(edge.a[1]!==edge.b[1]),1);
    assert.deepEqual(next.shape.points[0],[0,0]);
  }
  for(const s of [2,7,length-2])assert.ok(Math.abs(nearestDistance(g,position(g,s))-s)<1e-10);
  const batteryPoints=spanPoints(g,-1,1);
  for(const [i,expected] of [[0,-1],[1,0],[2,1]]){
    assert.equal(batteryPoints[i][0],0);
    assert.ok(Math.abs(batteryPoints[i][1]-expected)<1e-12);
  }
  assert.equal(reshape(g,'top',[0,1000]),null);
  const top=handles(g).find(h=>h.id==='top').point;
  const stopped=constrainedReshape(g,'top',top,[top[0],1000]);
  assert.ok(stopped.geometry.config.length<=80+1e-9);
  assert.ok(stopped.point[1]>=top[1]);
  assert.ok(stopped.point[1]<1000);
  assert.ok(reshape(g,'top',stopped.point));
  if(kind==='step')assert.equal(reshape(g,'notch',[0,0]),null);
  if(kind==='stairs')assert.equal(reshape(g,'lower corner',[g.shape.points[2][0],g.shape.points[5][1]]),null);
  if(kind==='u')assert.equal(reshape(g,'indent left',[g.shape.points[4][0],g.shape.points[3][1]]),null);
}
// A corner can move past the old fixed-length limit: the resistor adapts.
const step=preset('step');
step.config.resistors=[{id:'R1',start:10,end:11.4,resistivity:20}];
const notch=handles(step).find(h=>h.id==='notch').point;
const limited=constrainedReshape(step,'notch',notch,[7,notch[1]]);
assert.equal(limited.point[0],7);
assert.ok(reshape(step,'notch',limited.point));
assert.notEqual(limited.geometry.config.resistors[0].end-limited.geometry.config.resistors[0].start,1.4);
const nearCorner=reshape(step,'notch',[6,notch[1]]);
assert.ok(nearCorner,'The resistor endpoint should move clear of the corner.');
assert.ok(nearCorner.config.resistors[0].end<10-.5);
// Numeric edits and pointer drags use the same corner clearance for resistor ends.
const lLoop=preset('step');
const r2={id:'R2',start:26,end:30,resistivity:8};
moveResistor(lLoop,r2,24);
assert.ok(r2.start<23.5 && r2.end<27.5);
assert.ok(Math.abs(r2.end-r2.start-4)<1e-10);
r2.start=26;r2.end=30;
resizeResistor(lLoop,r2,'end',36);
assert.equal(r2.start,26);
assert.ok(r2.end>36.5);
console.log('Resizable shape families and component attachment passed.');

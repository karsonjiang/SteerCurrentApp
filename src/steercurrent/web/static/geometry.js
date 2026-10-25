// All coordinates are model coordinates, with the battery centered at (0, 0).
export const clone = value => JSON.parse(JSON.stringify(value));
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export function segments(geometry) {
  const points = geometry.shape.points;
  let distance = 0;
  return points.map((a, i) => {
    const b = points[(i + 1) % points.length], length = Math.hypot(b[0]-a[0], b[1]-a[1]);
    const segment = {a, b, length, start: distance};
    distance += length;
    return segment;
  });
}
export function position(geometry, distance) {
  const length = geometry.config.length;
  let s = ((distance % length) + length) % length;
  for (const edge of segments(geometry)) {
    if (s <= edge.length) {
      const t = s/edge.length;
      return [edge.a[0]+t*(edge.b[0]-edge.a[0]), edge.a[1]+t*(edge.b[1]-edge.a[1])];
    }
    s -= edge.length;
  }
  return geometry.shape.points[0];
}
export function spanPoints(geometry, start, end) {
  const points = [position(geometry, start)], length = geometry.config.length;
  for (const offset of [-length, 0, length]) {
    for (const edge of segments(geometry)) {
      const s = edge.start+offset;
      if (s > start && s < end) points.push(position(geometry, s));
    }
  }
  points.push(position(geometry, end));
  return points;
}
export function nearestDistance(geometry, point) {
  let best = {distance: Infinity, s: 0};
  for (const edge of segments(geometry)) {
    const dx = edge.b[0]-edge.a[0], dy = edge.b[1]-edge.a[1];
    const t = clamp(((point[0]-edge.a[0])*dx+(point[1]-edge.a[1])*dy)/edge.length**2, 0, 1);
    const distance = Math.hypot(point[0]-edge.a[0]-t*dx, point[1]-edge.a[1]-t*dy);
    if (distance < best.distance) best = {distance, s: edge.start+t*edge.length};
  }
  return best.s;
}
// A component boundary needs a short straight section on either side of a turn.
// Move an end just past that clearance when an edit lands inside it.
function cornerCorrection(geometry, distance, direction) {
  const halfWidth=geometry.config.width/2;
  for(const edge of segments(geometry).slice(1)){
    if(Math.abs(distance-edge.start)<=halfWidth+1e-9)
      return edge.start+(direction<0?-1:1)*(halfWidth+.01)-distance;
  }
  return 0;
}
export function moveResistor(geometry, resistor, start) {
  const length=resistor.end-resistor.start, direction=Math.sign(start-resistor.start);
  if(!direction)return;
  let end=start+length;
  // Moving the whole body can put either end near a different corner.
  for(let i=0;i<segments(geometry).length*2;i++){
    const corrections=[cornerCorrection(geometry,start,direction),cornerCorrection(geometry,end,direction)];
    const shift=direction<0?Math.min(...corrections):Math.max(...corrections);
    if(!shift)break;
    start+=shift;end+=shift;
  }
  resistor.start=start;resistor.end=end;
}
export function resizeResistor(geometry, resistor, end, distance) {
  const direction=Math.sign(distance-resistor[end]);
  resistor[end]=distance+(direction?cornerCorrection(geometry,distance,direction):0);
}
export function bounds(geometry) {
  const p = geometry.shape.points;
  return [Math.min(...p.map(v=>v[0])), Math.max(...p.map(v=>v[0])), Math.min(...p.map(v=>v[1])), Math.max(...p.map(v=>v[1]))];
}
// The point order identifies each preset, so no editor metadata enters the core model.
export function shapeKind(geometry) {
  const p=geometry.shape.points;
  if(p.length===5)return 'rectangle';
  if(p.length===7)return 'step';
  return p[5][1]===p[1][1]?'u':'stairs';
}
export function handles(geometry) {
  const [,right,bottom,top]=bounds(geometry),p=geometry.shape.points,kind=shapeKind(geometry);
  const result=[
    {id:'top',point:[p[2][0]/2,top],axes:[1],offset:[0,1]},
    {id:'bottom',point:[right/2,bottom],axes:[1],offset:[0,-1]},
    {id:'right',point:[right,(bottom+p.at(-3)[1])/2],axes:[0],offset:[1,0]},
  ];
  if(kind==='step')result.push({id:'notch',point:p[3],axes:[0,1],offset:[1,1]});
  if(kind==='stairs')for(const [id,index] of [['upper corner',3],['lower corner',5]])
    result.push({id,point:p[index],axes:[0,1],offset:[1,1]});
  if(kind==='u')for(const [id,index,dx] of [['indent left',3,-1],['indent right',4,1]])
    result.push({id,point:p[index],axes:[0,1],offset:[dx,1]});
  return result;
}
function attachedDistance(oldEdges,newEdges,distance,clearance) {
  // Keep each component end on its original wire segment. If it approaches a
  // corner, place it just inside the usable part of that segment.
  const i=oldEdges.findIndex(edge=>distance<=edge.start+edge.length+1e-10);
  const old=oldEdges[i],updated=newEdges[i];
  const moved=updated.start+(distance-old.start)*updated.length/old.length;
  return clamp(moved,updated.start+clearance,updated.start+updated.length-clearance);
}
export function reshape(geometry,handle,point) {
  if(!point.every(Number.isFinite))return null;
  const next=clone(geometry),p=next.shape.points,kind=shapeKind(geometry);
  let [,w,bottom,top]=bounds(geometry);
  let a=p[2][0],k=p[3][1],c=p[4][0],j=p[5]?.[1];
  if(handle==='top')top=point[1];
  if(handle==='bottom')bottom=point[1];
  if(handle==='right')w=point[0];
  if(handle==='notch'||handle==='upper corner'){a=point[0];k=point[1];}
  if(handle==='lower corner'){c=point[0];j=point[1];}
  if(handle==='indent left'||handle==='indent right'){
    k=point[1];
    if(handle==='indent left')a=point[0];else c=point[0];
  }
  const gap=geometry.config.width+.1;
  if(w<=2*gap||top<=geometry.config.battery.length/2+gap/2||-bottom<=geometry.config.battery.length/2+gap/2)return null;
  if(kind==='rectangle')next.shape.points=[[0,0],[0,top],[w,top],[w,bottom],[0,bottom]];
  if(kind==='step'){
    if(a<=gap||w-a<=gap||k-bottom<=gap||top-k<=gap)return null;
    next.shape.points=[[0,0],[0,top],[a,top],[a,k],[w,k],[w,bottom],[0,bottom]];
  }
  if(kind==='stairs'||kind==='u'){
    if(a<=gap||c-a<=gap||w-c<=gap||k-bottom<=gap||top-k<=gap)return null;
    if(kind==='stairs'&&(k-j<=gap||j-bottom<=gap))return null;
    next.shape.points=kind==='u'
      ?[[0,0],[0,top],[a,top],[a,k],[c,k],[c,top],[w,top],[w,bottom],[0,bottom]]
      :[[0,0],[0,top],[a,top],[a,k],[c,k],[c,j],[w,j],[w,bottom],[0,bottom]];
  }
  const before=segments(geometry),after=segments(next);
  const length=after.reduce((total,edge)=>total+edge.length,0);
  if(length<24||length>80)return null;
  next.config.length=length;
  const clearance=next.config.width/2+1e-4;
  for(const resistor of next.config.resistors){
    resistor.start=attachedDistance(before,after,resistor.start,clearance);
    resistor.end=attachedDistance(before,after,resistor.end,clearance);
  }
  const corners=after.slice(1).map(edge=>edge.start);
  if(corners.some(s=>Math.min(s,length-s)<=next.config.battery.length/2+clearance))return null;
  let previous=next.config.battery.length/2;
  for(const resistor of [...next.config.resistors].sort((a,b)=>a.start-b.start)){
    if(resistor.start<=previous+1e-6||resistor.end<=resistor.start+1e-6||resistor.end>=length-next.config.battery.length/2-1e-6)return null;
    previous=resistor.end;
  }
  return next;
}
// Advance from a valid handle position until the first constraint is reached.
export function constrainedReshape(geometry,handle,from,to) {
  const distance=Math.hypot(to[0]-from[0],to[1]-from[1]);
  const steps=Math.max(1,Math.min(400,Math.ceil(distance/.2)));
  let accepted=reshape(geometry,handle,from)||geometry, fraction=0;
  for(let i=1;i<=steps;i++){
    const t=i/steps,point=from.map((v,axis)=>v+t*(to[axis]-v));
    const candidate=reshape(geometry,handle,point);
    if(!candidate){
      let low=fraction,high=t;
      for(let j=0;j<24;j++){
        const middle=(low+high)/2;
        const probe=reshape(geometry,handle,from.map((v,axis)=>v+middle*(to[axis]-v)));
        if(probe){low=middle;accepted=probe;}else high=middle;
      }
      return {geometry:accepted,point:from.map((v,axis)=>v+low*(to[axis]-v))};
    }
    accepted=candidate;fraction=t;
  }
  return {geometry:accepted,point:to};
}
export function preset(kind,length=40) {
  const w=.3*length,t=.1*length;
  const shapes={
    rectangle:[[0,0],[0,t],[w,t],[w,-t],[0,-t]],
    step:[[0,0],[0,t],[.2*length,t],[.2*length,0],[w,0],[w,-t],[0,-t]],
    stairs:[[0,0],[0,t],[.1*length,t],[.1*length,.025*length],[.2*length,.025*length],[.2*length,-.025*length],[w,-.025*length],[w,-t],[0,-t]],
    u:[[0,0],[0,t],[.1*length,t],[.1*length,.025*length],[.15*length,.025*length],[.15*length,t],[.225*length,t],[.225*length,-t],[0,-t]],
  };
  const geometry={config:{length,width:1,battery:{length:2,emf:1,resistivity:.1,polarity:1},resistors:[],wire_resistivity:1},shape:{points:shapes[kind]}};
  // The bottom wire has room for a resistor in every preset.
  geometry.config.resistors=[{id:'R1',start:.75*length,end:.83*length,resistivity:20}];
  return geometry;
}

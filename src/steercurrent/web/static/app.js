import {clone, clamp, position, spanPoints, nearestDistance, bounds, handles, constrainedReshape, preset, shapeKind, segments, moveResistor, resizeResistor} from './geometry.js';

const $ = id => document.getElementById(id);
const canvas = $('canvas'), context = canvas.getContext('2d');
const lessons = {
  potential: ['Potential V', 'Potential', 'The battery establishes a potential difference around the loop. These wires have finite resistance, so potential changes in the wire too. The electric field points from high to low potential.', 'E = −∇V', 'Select a resistor and increase its resistivity. Compare the potential drop before and after.'],
  electric: ['Electric field E', 'Electric field', 'Electric fields exist both inside the conductor and in the surrounding space. Surface charges arrange themselves so the field guides the current around bends.', 'J = (E + F) / ρ', 'Move a resistor along the loop and compare the surrounding field.'],
  toy: ['Battery-only field', 'Battery-only field', 'This deliberately incomplete model puts uniform charge on the battery terminals. Its field is matched inside the battery, then evaluated throughout the same space as the real circuit.', 'Etoy from battery terminals only', 'Compare its arrows with the actual field, especially far from the battery.'],
  current: ['Current density J', 'Current', 'The arrows show where conventional current flows. The total steady current is the same through each complete cut across this one-loop circuit.', 'I = ∫ J · n dℓ', 'Reshape a corner, then switch to Current strength to compare the flow across the wire.'],
  current_strength: ['Current strength |J|', 'What is current crowding?', 'The current can pack closer together on one side of a bend, often near its inside edge. Pale shows current spread evenly across the wire; red shows where its local strength rises above that even-flow level.', '|J| = √(Jₓ² + Jᵧ²)', 'Reshape a corner and compare its inner and outer edges. Then switch to Current to see the flow direction.'],
  charge: ['Surface charge σ', 'Surface charge', 'Charge on conductor surfaces and at component interfaces shapes the electric field. Magenta is positive; blue is negative. Each colored edge shows its calculated charge density: thicker, darker bands mean greater magnitude.', 'σ = ε (E₂ − E₁) · n', 'Reverse the battery and compare charge signs at its edges.'],
  power_exchange: ['Power exchange', 'Power exchange', 'Orange marks where the battery supplies power. Blue marks where the conductor absorbs power as heat, especially in the resistors. The colors show net power per unit area and depth.', 'Power exchange = J · F − ρ|J|²', 'Move a resistor or change its resistivity. Compare the heating readout.'],
  magnetic: ['Magnetic field Bz', 'Magnetic field', 'In this two-dimensional model, the magnetic field points into or out of the screen. It is constant in the cavity and zero outside the loop. This simplification is specific to an infinitely deep circuit.', '∇ × B = μJ', 'Reverse the battery and compare the magnetic field.'],
  energy: ['Energy flow S', 'Energy flow', 'The electric and magnetic fields carry energy through space into the resistor. The battery supplies energy; the wire and resistors absorb it as heat. The arrows show steady energy flow, not moving particles.', 'S = E × B / μ', 'Switch between Current and Energy flow. Compare the directions of their arrows.'],
};
const vector = symbol => `<mover accent="true"><mi>${symbol}</mi><mo>→</mo></mover>`;
const equation = (expression, meaning, terms) => ({
  markup: `<math display="block">${expression}</math>`,
  meaning,
  terms,
});
const equations = {
  potential: equation(
    `<mrow>${vector('E')}<mo>=</mo><mo>−</mo><mo>∇</mo><mi>V</mi></mrow>`,
    'The electric field points toward decreasing potential.',
    [['E⃗', 'electric field, a vector'], ['V', 'electric potential, a scalar'], ['∇', 'spatial gradient']]
  ),
  electric: equation(
    `<mrow>${vector('J')}<mo>=</mo><mfrac><mrow>${vector('E')}<mo>+</mo>${vector('F')}</mrow><mi>ρ</mi></mfrac></mrow>`,
    'The electric field and the battery driving field together produce current density in the conductor.',
    [['J⃗', 'current density, a vector'], ['E⃗', 'electric field, a vector'], ['F⃗', 'battery driving field, a vector; zero outside the battery'], ['ρ', 'material resistivity']]
  ),
  toy: equation(
    `<mrow><msub>${vector('E')}<mtext>toy</mtext></msub><mo>=</mo><msub>${vector('E')}<mo>+</mo></msub><mo>+</mo><msub>${vector('E')}<mo>−</mo></msub></mrow>`,
    'This comparison adds the fields from charges on the two battery terminals only.',
    [['E⃗toy', 'battery-only electric field, a vector'], ['E⃗+ and E⃗−', 'fields from the positive and negative battery terminals']]
  ),
  current: equation(
    `<mrow><mi>I</mi><mo>=</mo><msub><mo>∫</mo><mi>C</mi></msub>${vector('J')}<mo>·</mo>${vector('n')}<mspace width=".25em"/><mi>dℓ</mi></mrow>`,
    'Add the current density crossing a cut through the wire to get the total current.',
    [['I', 'total current per unit depth'], ['J⃗', 'current density, a vector'], ['n⃗', 'unit normal to the cut, a vector'], ['C', 'cut across the wire'], ['dℓ', 'small length along the cut']]
  ),
  current_strength: equation(
    `<mrow><mo>|</mo>${vector('J')}<mo>|</mo><mo>=</mo><msqrt><msup><msub><mi>J</mi><mi>x</mi></msub><mn>2</mn></msup><mo>+</mo><msup><msub><mi>J</mi><mi>y</mi></msub><mn>2</mn></msup></msqrt></mrow>`,
    'The length of the current-density vector shows how strongly current flows at each point.',
    [['|J⃗|', 'magnitude of current density'], ['Jₓ and Jᵧ', 'horizontal and vertical components of current density']]
  ),
  charge: equation(
    `<mrow><mi>σ</mi><mo>=</mo><mi>ε</mi><mo>(</mo><msub>${vector('E')}<mn>2</mn></msub><mo>−</mo><msub>${vector('E')}<mn>1</mn></msub><mo>)</mo><mo>·</mo>${vector('n')}</mrow>`,
    'The change in the normal electric field across a boundary gives its surface charge density.',
    [['σ', 'surface charge density'], ['ε', 'permittivity of this model'], ['E⃗₁ and E⃗₂', 'electric fields on the two sides of the boundary'], ['n⃗', 'unit normal pointing from side 1 to side 2']]
  ),
  power_exchange: equation(
    `<mrow><msub><mi>p</mi><mtext>exchange</mtext></msub><mo>=</mo>${vector('J')}<mo>·</mo>${vector('F')}<mo>−</mo><mi>ρ</mi><msup><mrow><mo>|</mo>${vector('J')}<mo>|</mo></mrow><mn>2</mn></msup></mrow>`,
    'Positive values show net power supplied by the battery; negative values show net heating.',
    [['pₑₓ', 'net power density'], ['J⃗', 'current density, a vector'], ['F⃗', 'battery driving field, a vector'], ['ρ', 'material resistivity']]
  ),
  magnetic: equation(
    `<mrow><mo>∇</mo><mo>×</mo>${vector('B')}<mo>=</mo><mi>μ</mi>${vector('J')}</mrow>`,
    'Current density determines the curl of the magnetic field; this model displays its out-of-plane component.',
    [['B⃗', 'magnetic field, a vector with only a z component here'], ['J⃗', 'current density, a vector in the circuit plane'], ['μ', 'magnetic permeability'], ['∇ ×', 'curl operator']]
  ),
  energy: equation(
    `<mrow>${vector('S')}<mo>=</mo><mfrac><mrow>${vector('E')}<mo>×</mo>${vector('B')}</mrow><mi>μ</mi></mfrac></mrow>`,
    'The cross product gives the direction and amount of electromagnetic energy flow.',
    [['S⃗', 'energy flux, a vector'], ['E⃗', 'electric field, a vector'], ['B⃗', 'magnetic field, a vector'], ['μ', 'magnetic permeability'], ['×', 'vector cross product']]
  ),
};
function equationHTML(view) {
  const item=equations[view];
  return `<div class="equation-math">${item.markup}</div><p class="equation-meaning">${esc(item.meaning)}</p><dl class="equation-terms">${item.terms.map(([symbol,description])=>`<div><dt>${esc(symbol).replace(/([A-Za-z])⃗/g,'<span class="vector-symbol">$1</span>')}</dt><dd>${esc(description)}</dd></div>`).join('')}</dl>`;
}
const storyScenes=[
  {kicker:'Step 1 of 7',title:'Current',question:'Follow the arrows around each bend in the loop.',action:'Select another loop shape or drag a green handle.',view:'current'},
  {kicker:'Step 2 of 7',title:'Electric field',explanation:'The electric field pushes charges along the wire and around its bends. It also extends into the surrounding space.',question:'Compare the arrows inside and outside the wire.',formula:'J = (E + F) / ρ',action:'Drag a green handle to reshape a corner.',view:'electric'},
  {kicker:'Step 3 of 7',title:'Battery-only field',explanation:'Charges create electric fields. This toy model puts charge only on the two battery terminals, then shows the field it would make.',question:'Compare this field with the full electric field. Can this battery-only field generate the same current?',action:'Use the buttons below to switch between the two fields.',conclusion:'No. The battery-only field misses some bends and fades with distance. Charges elsewhere in the circuit are also needed.',view:'toy',views:[['toy','Battery only'],['electric','Full field']]},
  {kicker:'Step 4 of 7',title:'Surface charge',explanation:'Charge along the wire surfaces and at component interfaces shapes the field that follows the loop.',formula:'σ = ε (E₂ − E₁) · n',action:'Look for thick bands at the battery and resistor edges. Move a resistor or switch to Electric field.',view:'charge',views:[['charge','Charge'],['electric','Electric field']]},
  {kicker:'Step 5 of 7',title:'Power exchange',explanation:'The battery supplies energy to the circuit. The wire and resistors absorb it as heat. Orange shows local net supply; blue shows local net absorption.',question:'Find where power is supplied and absorbed.',formula:'Net supply = J · F − ρ|J|²',action:'Move a resistor and compare its heating readout.',view:'power_exchange'},
  {kicker:'Step 6 of 7',title:'Energy flow',explanation:'The arrows show energy flowing through the space around the wire toward the resistors. These arrows are Poynting vectors, which will be explained in the next step.',question:'Follow the arrows from the battery toward a resistor.',action:'Reshape the loop or move a resistor to see how the flow changes.',view:'energy'},
  {kicker:'Step 7 of 7',title:'Compare fields',explanation:'Charges create an electric field E, and current creates a magnetic field B. Together they give the Poynting vector S: the direction and density of energy flow. Much of that energy travels through the space around the wires.',formula:'S = E × B / μ',action:'Switch among the electric field, magnetic field, and energy flow, then select another loop shape.',view:'energy',views:[['electric','Electric field'],['magnetic','Magnetic field'],['energy','Energy flow']]},
];
let geometry, defaultGeometry, selected='battery', view='current', result=null, resultRevision=-1, revision=0, resultResolution=null;
const shapeStates=new Map();
let storyIndex=0, answerRevealed=false;
let history=[], drag=null, preview=null, checking=false, validationId=0, running=false, timer=null;
let camera={x:6,y:0,zoom:25}, width=600,height=550;
const fmt = n => !Number.isFinite(n) ? '—' : Math.abs(n)<1e-12 ? '0' : Number(n.toPrecision(3)).toString();
const esc = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const currentGeometry = () => preview || geometry;
const fresh = () => result && resultRevision===revision && !preview;
const screen = p => [width/2+(p[0]-camera.x)*camera.zoom, height/2-(p[1]-camera.y)*camera.zoom];
const world = p => [camera.x+(p[0]-width/2)/camera.zoom,camera.y-(p[1]-height/2)/camera.zoom];
function message(text) { $('status').textContent=text; }
function error(text='') { $('error').hidden=!text;$('error').textContent=text; }
function setPanel(name, open) {
  const configuration=name==='controls';
  const panel=$(configuration?'configuration-panel':'physics-panel'),button=$(`toggle-${name}`);
  if(!open&&panel.contains(document.activeElement))button.focus();
  panel.hidden=!open;
  document.body.dataset[configuration?'controlsOpen':'physicsOpen']=String(open);
  button.setAttribute('aria-expanded',String(open));
  if(open&&window.matchMedia(configuration?'(max-width: 760px)':'(max-width: 1100px)').matches)requestAnimationFrame(()=>panel.scrollIntoView({block:'start'}));
}
async function request(path, body) {
  const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json();
  if (!response.ok) throw new Error(data.message||'The server could not complete this request.');
  return data;
}
function fit() {
  if (!geometry) return;
  const [x0,x1,y0,y1]=bounds(geometry);
  camera={x:(x0+x1)/2,y:(y0+y1)/2,zoom:Math.min(width/(x1-x0+6),height/(y1-y0+6))};
  draw();
}
function resize() {
  const box=$('viewport').getBoundingClientRect();
  const changed=width!==box.width||height!==box.height;
  width=box.width;height=box.height;
  const ratio=window.devicePixelRatio||1;
  canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
  context.setTransform(ratio,0,0,ratio,0,0);
  if(geometry&&changed)fit();else draw();
}
function staleState() {
  $('freshness').textContent=fresh()?'Fields up to date':result?'Edits not calculated':'Not calculated';
  $('freshness').classList.toggle('dirty',!fresh());
  $('current-readout').textContent=fresh()?fmt(result.summary.current):'—';
  $('power-readout').textContent=fresh()?fmt(result.summary.source_power):'—';
  $('resistor-readout').textContent=fresh()?fmt(result.summary.resistor_power):'—';
  $('story-current').textContent=fresh()?fmt(result.summary.current):'—';
  $('story-power').textContent=fresh()?fmt(result.summary.source_power):'—';
  $('story-resistor').textContent=fresh()?fmt(result.summary.resistor_power):'—';
  $('undo').disabled=!history.length||checking;
  const automatic=$('auto').checked;
  $('calculate').disabled=automatic||running||checking;
  $('calculate').textContent=automatic?'Automatic updates on':'Calculate fields';
  $('calculate').title=automatic?'Turn off automatic updates to calculate manually.':'';
  $('numerical-state').textContent=running?'Calculating':fresh()?'Complete · current':result?'Previous result · stale':'Not calculated';
  $('numerical-elements').textContent=result?result.summary.triangles.toLocaleString():'—';
  $('numerical-vertices').textContent=result?result.vertices.length.toLocaleString():'—';
  $('numerical-time').textContent=result?`${result.summary.seconds.toFixed(3)} s`:'—';
  $('numerical-resolution').textContent=result?(resultResolution==='fine'?'Finer mesh':'Quick exploration'):'—';
}
function updateLesson() {
  const [label,title,body,,experiment]=lessons[view];
  $('view-label').textContent=label;$('lesson-title').textContent=title;$('lesson-body').textContent=body;
  $('resistor-readout-row').hidden=view!=='power_exchange';
  $('formula').innerHTML=equationHTML(view);
  $('experiment').textContent=experiment;
  $('field-select').value=view==='toy'?'electric':view;
  $('probe').textContent='Move over the diagram to inspect a field value.';
  draw();
}
function renderStory() {
  const scene=storyScenes[storyIndex];
  $('story-count').textContent=`${storyIndex+1} / ${storyScenes.length}`;
  $('story-progress').innerHTML=storyScenes.map((_,i)=>`<span class="${i<=storyIndex?'reached':''}"></span>`).join('');
  $('story-kicker').textContent=scene.kicker;
  $('story-title').textContent=scene.title;
  $('story-explanation-text').textContent=scene.explanation||'';
  $('story-explanation').hidden=!scene.explanation;
  $('guide-explanation').hidden=!scene.explanation;
  $('story-question').textContent=scene.question||'';
  $('story-question').hidden=!scene.question;
  $('story-formula').innerHTML=scene.formula?equationHTML(scene.view):'';
  $('story-formula').hidden=!scene.formula;
  $('story-resistor-row').hidden=storyIndex<4;
  $('story-action').textContent=scene.action;
  $('story-back').disabled=storyIndex===0;
  $('story-next').textContent=storyIndex===storyScenes.length-1?'Explore freely →':'Next →';
  $('story-views').hidden=!scene.views;
  $('story-views').innerHTML=scene.views?.map(([key,label])=>`<button data-story-view="${key}" class="${view===key?'active':''}">${label}</button>`).join('')||'';
  $('story-conclusion').textContent=scene.conclusion||'';
  $('story-conclusion').hidden=!scene.conclusion||!answerRevealed;
  $('story-reveal').hidden=!scene.conclusion||answerRevealed;
  for(const button of $('story-views').querySelectorAll('button'))button.onclick=()=>{
    view=button.dataset.storyView;
    for(const choice of $('story-views').querySelectorAll('button'))choice.classList.toggle('active',choice===button);
    updateLesson();
  };
  updateLesson();
}
function showStory(index) {
  storyIndex=index;answerRevealed=false;view=storyScenes[index].view;renderStory();
  document.querySelector('.story-body').scrollTop=0;
}
function setMode(presentation) {
  document.body.classList.toggle('presentation',presentation);
  document.body.classList.toggle('explore',!presentation);
  for(const [id,active] of [['mode-present',presentation],['mode-explore',!presentation]]){
    $(id).classList.toggle('active',active);$(id).setAttribute('aria-pressed',String(active));
  }
  if(presentation){
    $('guide-panel').open=true;
    $('auto').checked=true;
    view=storyScenes[storyIndex].view;renderStory();
    if(!fresh())schedule();
  } else if(view==='toy'){
    view='electric';updateLesson();
  }
  requestAnimationFrame(resize);
}
function updateControls() {
  const c=geometry.config;
  if (selected!=='battery'&&!c.resistors.some(r=>r.id===selected)) selected=c.resistors[0]?.id||'battery';
  $('shape').value=shapeKind(geometry);$('demo-shape').value=shapeKind(geometry);$('loop-length').value=Number(c.length.toFixed(2));
  const components=[{id:'battery',label:'Battery',tag:`${fmt(c.battery.emf)} strength`},...c.resistors.map(r=>({id:r.id,label:r.id.replaceAll('_',' '),tag:`ρ ${fmt(r.resistivity)}`}))];
  $('components').innerHTML=components.map(r=>`<button class="component ${selected===r.id?'selected':''}" data-component="${esc(r.id)}"><i class="dot"></i>${esc(r.label)}<span>${esc(r.tag)}</span></button>`).join('');
  for (const button of $('components').querySelectorAll('button')) button.onclick=()=>{selected=button.dataset.component;updateControls();draw();};
  if (selected==='battery') {
    $('inspector').innerHTML=`<label for="emf">Battery strength <span class="value-label">${fmt(c.battery.emf)}</span></label><input id="emf" type="range" min="0" max="5" step="0.1" value="${c.battery.emf}"><button id="reverse" class="small" title="Reverse the battery driving field">Reverse polarity ${c.battery.polarity>0?'↑':'↓'}</button><details class="control-help"><summary>Battery help</summary><p>The battery stays straight at the origin.</p></details>`;
    $('emf').oninput=event=>event.target.previousElementSibling.lastElementChild.textContent=event.target.value;
    $('emf').onchange=event=>edit(g=>g.config.battery.emf=Number(event.target.value));
    $('reverse').onclick=()=>edit(g=>g.config.battery.polarity*=-1);
  } else {
    const r=c.resistors.find(r=>r.id===selected);
    $('inspector').innerHTML=`<label for="resistivity">Material resistivity <span class="value-label">${fmt(r.resistivity)}</span></label><input id="resistivity" type="range" min="1" max="60" step="1" value="${r.resistivity}"><label for="resistor-start">Start distance from battery</label><input id="resistor-start" type="number" step="0.1" min="1.1" max="${c.length-1.1}" value="${r.start.toFixed(2)}"><label for="resistor-length">Length along the loop</label><input id="resistor-length" type="number" min="0.5" step="0.1" value="${(r.end-r.start).toFixed(2)}"><details class="control-help"><summary>Resistor help</summary><p>Drag the orange body to move it. Drag its end dots to resize. It can wrap around a corner.</p></details><button id="remove" class="remove">Remove resistor</button>`;
    $('resistivity').oninput=event=>event.target.previousElementSibling.lastElementChild.textContent=event.target.value;
    $('resistivity').onchange=event=>edit(g=>g.config.resistors.find(item=>item.id===selected).resistivity=Number(event.target.value));
    $('resistor-start').onchange=event=>edit(g=>{const item=g.config.resistors.find(item=>item.id===selected);moveResistor(g,item,Number(event.target.value));});
    $('resistor-length').onchange=event=>edit(g=>{const item=g.config.resistors.find(item=>item.id===selected);resizeResistor(g,item,'end',item.start+Number(event.target.value));});
    $('remove').onclick=()=>edit(g=>g.config.resistors=g.config.resistors.filter(r=>r.id!==selected));
  }
  $('mesh-info').textContent=`Length ${fmt(c.length)} · battery at (0, 0)`;
  // Numeric shape controls offer the same operations without requiring dragging.
  $('shape-inputs').innerHTML=handles(geometry).map(h=>`<label>${esc(h.id)} handle</label>${h.axes.map(axis=>`<input aria-label="${esc(h.id)} handle ${axis===0?'x':'y'}" data-handle="${h.id}" data-axis="${axis}" type="number" step="0.1" value="${h.point[axis].toFixed(2)}">`).join('')}`).join('');
  $('shape-inputs').querySelectorAll('input').forEach(input=>input.onchange=()=>{
    const h=handles(geometry).find(h=>h.id===input.dataset.handle),point=[...h.point];point[Number(input.dataset.axis)]=Number(input.value);
    const next=constrainedReshape(geometry,h.id,h.point,point);
    error();
    if(next.point.some((value,axis)=>Math.abs(value-h.point[axis])>1e-6))commit(next.geometry);
    else updateControls();
  });
  staleState();
}
async function commit(next, {resetView=false, undo=false, resetAll=false}={}) {
  if(!next) return;
  const token=++validationId;checking=true;preview=next;error();staleState();draw();message('Checking the new circuit…');
  try {
    const checked=await request('/api/validate',{geometry:next});
    if(token!==validationId) return;
    if(resetAll){history=[];shapeStates.clear();defaultGeometry=clone(checked.geometry);selected='battery';}
    else if(undo) history.pop();else {history.push(clone(geometry));if(history.length>40)history.shift();}
    geometry=checked.geometry;shapeStates.set(shapeKind(geometry),clone(geometry));
    revision++;preview=null;checking=false;updateControls();
    if(resetView)fit();else draw();
    message($('auto').checked?'Circuit updated. Fields will update automatically.':'Circuit updated. Click Calculate fields to update the results.');schedule();
  } catch(e) {
    if(token!==validationId)return;
    checking=false;preview=null;updateControls();draw();error(`${e.message} Your previous circuit has been kept.`);message('Move not applied.');
  }
}
function edit(change) {if(checking)return;const next=clone(geometry);change(next);commit(next);}
function schedule() {clearTimeout(timer);if($('auto').checked)timer=setTimeout(calculate,450);}
async function calculate() {
  if(!geometry||checking||running)return;
  clearTimeout(timer);running=true;const version=revision;
  const body={geometry:clone(geometry),resolution:$('resolution').value};staleState();message('Sending circuit…');error();
  try {
    const {id}=await request('/api/jobs',body);
    let data;
    do {
      await new Promise(resolve=>setTimeout(resolve,350));
      data=await request(`/api/jobs/${id}`);
      message(data.status==='queued'?'Waiting for the calculation worker…':'Calculating the fields… You can still edit or explore.');
    } while(data.status==='queued'||data.status==='running');
    if(data.status==='error')throw new Error(data.message);
    if(version===revision){result=data.result;resultRevision=version;resultResolution=body.resolution;prepareFields();message('Ready.');}
  } catch(e) {if(version===revision){error(e.message);message('Calculation did not finish. You can try again.');}}
  finally {running=false;staleState();draw();if(version!==revision){message('Circuit changed. The earlier result was discarded.');schedule();}}
}
let fieldScales={};
function prepareFields() {
  const absmax=values=>values.reduce((maximum,v)=>Math.max(maximum,Math.abs(v)),0)||1;
  fieldScales={potential:absmax(result.potential),magnetic:absmax(result.magnetic),power_exchange:absmax(result.power_exchange)};
  for(const name of ['electric','current','energy'])fieldScales[name]=absmax(result[name].map(v=>Math.hypot(...v)));
  const conductorStrength=result.current.flatMap((vector,i)=>
    ['outside','cavity'].includes(result.region_names[result.regions[i]])?[]:[Math.hypot(...vector)]).sort((a,b)=>a-b);
  fieldScales.current_strength=conductorStrength[Math.floor(.95*(conductorStrength.length-1))]||1;
  fieldScales.current_reference=Math.abs(result.summary.current)/geometry.config.width;
  fieldScales.toy=fieldScales.electric; // Keep one arrow-strength scale for the comparison.
  fieldScales.charge=absmax(Object.values(result.charges).flatMap(t=>t.density));
}
function color(value,scale,alpha=1) {
  const t=clamp(value/scale,-1,1),from=[47,56,58],to=t>=0?[201,103,62]:[53,112,172];
  return `rgba(${from.map((v,i)=>Math.round(v+(to[i]-v)*Math.abs(t))).join(',')},${alpha})`;
}
function strengthColor(value,reference,scale) {
  const range=Math.max(scale-reference,.05*reference,1e-12);
  const t=clamp((value-reference)/range,0,1);
  const pale=[62,68,67],dark=[191,56,72];
  return `rgb(${pale.map((v,i)=>Math.round(v+(dark[i]-v)*t)).join(',')})`;
}
function chargeColor(value,opacity=1) {
  return value>=0?`rgba(233,105,159,${opacity})`:`rgba(102,169,236,${opacity})`;
}
function chargeBoundaryIsStraight(vertex,first,second,edges,vertices,lengths) {
  const other=index=>vertices[edges[index][0]===vertex?edges[index][1]:edges[index][0]];
  const center=vertices[vertex],a=other(first),b=other(second);
  const dot=(a[0]-center[0])*(b[0]-center[0])+(a[1]-center[1])*(b[1]-center[1]);
  return dot<-.999*lengths[first]*lengths[second];
}
function chargeMarkers(traces,vertices,scale) {
  const markers=[];
  for(const trace of Object.values(traces)){
    const {edges,density}=trace,atVertex=new Map(),lengths=[];
    edges.forEach(([a,b],i)=>{
      for(const vertex of [a,b]){
        if(!atVertex.has(vertex))atVertex.set(vertex,[]);
        atVertex.get(vertex).push(i);
      }
      lengths.push(Math.hypot(vertices[b][0]-vertices[a][0],vertices[b][1]-vertices[a][1]));
    });
    const visited=new Set();
    for(let i=0;i<edges.length;i++){
      if(visited.has(i))continue;
      // A charge segment ends when its sign changes or its boundary turns.
      const positive=density[i]>=0,group=[],pending=[i];
      visited.add(i);
      while(pending.length){
        const edge=pending.pop();group.push(edge);
        for(const vertex of edges[edge])for(const neighbor of atVertex.get(vertex)){
          if(!visited.has(neighbor)&&(density[neighbor]>=0)===positive&&
             chargeBoundaryIsStraight(vertex,edge,neighbor,edges,vertices,lengths)){
            visited.add(neighbor);pending.push(neighbor);
          }
        }
      }
      // Ignore tiny numerical sign patches that are invisible in the charge bands.
      if(Math.max(...group.map(edge=>Math.abs(density[edge])))<.01*scale)continue;
      const degree=new Map();
      for(const edge of group)for(const vertex of edges[edge])degree.set(vertex,(degree.get(vertex)||0)+1);
      let vertex=[...degree].find(([,count])=>count===1)?.[0]??edges[group[0]][0];
      const unused=new Set(group),ordered=[];
      while(unused.size){
        const edge=atVertex.get(vertex).find(index=>unused.has(index));
        if(edge===undefined)break;
        unused.delete(edge);
        const next=edges[edge][0]===vertex?edges[edge][1]:edges[edge][0];
        ordered.push({edge,from:vertex,to:next});
        vertex=next;
      }
      // Average arclength with |charge density| as the weight, then put the
      // sign at that position on this straight part of the boundary.
      let totalCharge=0,moment=0,distance=0;
      for(const {edge} of ordered){
        const charge=Math.abs(density[edge])*lengths[edge];
        totalCharge+=charge;
        moment+=charge*(distance+lengths[edge]/2);
        distance+=lengths[edge];
      }
      const target=moment/totalCharge;
      distance=0;
      for(const {edge,from,to} of ordered){
        if(distance+lengths[edge]>=target){
          const fraction=(target-distance)/lengths[edge],a=vertices[from],b=vertices[to];
          markers.push({point:[a[0]+fraction*(b[0]-a[0]),a[1]+fraction*(b[1]-a[1])],value:density[edge]});
          break;
        }
        distance+=lengths[edge];
      }
    }
  }
  return markers;
}
function path(points,close=false) {context.beginPath();points.forEach((p,i)=>{const [x,y]=screen(p);if(i)context.lineTo(x,y);else context.moveTo(x,y);});if(close)context.closePath();}
function stroke(points,color,width,close=false){path(points,close);context.strokeStyle=color;context.lineWidth=width;context.lineJoin='miter';context.lineCap='butt';context.stroke();}
function circle(p,r,fill,border){const [x,y]=screen(p);context.beginPath();context.arc(x,y,r,0,2*Math.PI);context.fillStyle=fill;context.fill();if(border){context.strokeStyle=border;context.lineWidth=1.5;context.stroke();}}
function text(p,content,color='#c7d6d1',offset=[0,0],size=11){const [x,y]=screen(p);context.save();context.fillStyle=color;context.font=`${size}px system-ui`;context.textAlign='center';context.strokeStyle='#151c1e';context.lineWidth=3;context.lineJoin='round';context.strokeText(content,x+offset[0],y+offset[1]);context.fillText(content,x+offset[0],y+offset[1]);context.restore();}
function arrow(p,vector,strength,fade=false){const norm=Math.hypot(...vector);if(norm<1e-16)return;const [x,y]=screen(p),dx=vector[0]/norm,dy=-vector[1]/norm,length=13;
  context.save();context.translate(x,y);context.rotate(Math.atan2(dy,dx));context.globalAlpha=fade?Math.max(.16,strength):.4+.6*strength;context.strokeStyle='#9ae7ce';context.fillStyle='#9ae7ce';context.lineWidth=1.25;context.beginPath();context.moveTo(-length/2,0);context.lineTo(length/2,0);context.stroke();context.beginPath();context.moveTo(length/2+2,0);context.lineTo(length/2-3,-3);context.lineTo(length/2-3,3);context.closePath();context.fill();context.restore();}
function drawFields() {
  const {vertices,triangles}=result;
  const signed=view==='potential'||view==='magnetic';
  const power=view==='power_exchange';
  const strength=view==='current_strength';
  if(signed || power || strength || $('show-mesh').checked){
    for(let i=0;i<triangles.length;i++){
      const ids=triangles[i],points=ids.map(id=>vertices[id]);
      const [sx,sy]=screen(result.centers[i]);if(sx< -100||sx>width+100||sy< -100||sy>height+100)continue;
      path(points,true);
      if(power && result.power_exchange[i]!==0){
        const fill=color(result.power_exchange[i],fieldScales.power_exchange);
        context.fillStyle=fill;context.fill();
        context.strokeStyle=fill;context.lineWidth=.6;context.stroke();
      }
      if(strength&&!['outside','cavity'].includes(result.region_names[result.regions[i]])){
        const fill=strengthColor(Math.hypot(...result.current[i]),fieldScales.current_reference,fieldScales.current_strength);
        context.fillStyle=fill;context.fill();
        context.strokeStyle=fill;context.lineWidth=.6;context.stroke();
      }
      if(signed){
        // P1 potential and Bz vary linearly inside a triangle. Use that same plane
        // for the color gradient, rather than coloring a whole triangle by its mean.
        const values=ids.map(id=>result[view][id]),p=points.map(screen),scale=fieldScales[view];
        const dx1=p[1][0]-p[0][0],dy1=p[1][1]-p[0][1],dx2=p[2][0]-p[0][0],dy2=p[2][1]-p[0][1];
        const determinant=dx1*dy2-dy1*dx2;
        const gx=((values[1]-values[0])*dy2-(values[2]-values[0])*dy1)/determinant;
        const gy=(dx1*(values[2]-values[0])-dx2*(values[1]-values[0]))/determinant;
        const norm2=gx*gx+gy*gy;
        let fill=color(values[0],scale);
        if(norm2>1e-24){
          const lo=(-scale-values[0])/norm2,hi=(scale-values[0])/norm2;
          fill=context.createLinearGradient(p[0][0]+lo*gx,p[0][1]+lo*gy,p[0][0]+hi*gx,p[0][1]+hi*gy);
          fill.addColorStop(0,color(-scale,scale));fill.addColorStop(.5,color(0,scale));fill.addColorStop(1,color(scale,scale));
        }
        context.fillStyle=fill;context.fill();
        // A narrow matching stroke closes canvas antialiasing seams between cells.
        context.strokeStyle=fill;context.lineWidth=.6;context.stroke();
      }
      if($('show-mesh').checked){context.strokeStyle='#b5c5c358';context.lineWidth=.5;context.stroke();}
    }
  }
  if(['electric','toy','current','energy'].includes(view)){
    // E and J use the same dense row along the conductor. Air keeps its
    // existing sparse grid, so nearby air cannot displace conductor arrows.
    const bins=new Map(),wireBins=new Map(),spacing=23,wireSpacing=18;
    const g=currentGeometry();
    for(let i=0;i<triangles.length;i++){
      const p=result.centers[i],[x,y]=screen(p),v=result[view==='toy'?'toy_electric':view][i],m=Math.hypot(...v);
      if(x<8||x>width-8||y<40||y>height-55)continue;
      const region=result.region_names[result.regions[i]];
      if((view==='electric'||view==='current')&&region!=='outside'&&region!=='cavity'){
        if(m<1e-16)continue;
        const distance=nearestDistance(g,p),onWire=position(g,distance);
        const offset=Math.hypot(p[0]-onWire[0],p[1]-onWire[1]);
        const key=Math.floor(distance*camera.zoom/wireSpacing);
        if(!wireBins.has(key)||offset<wireBins.get(key).offset)wireBins.set(key,{p,v,m,offset});
        continue;
      }
      if(m<fieldScales[view]*1e-7)continue;
      const key=`${Math.floor(x/spacing)},${Math.floor(y/spacing)}`;
      if(!bins.has(key)||bins.get(key).m<m)bins.set(key,{p,v,m});
    }
    for(const {p,v,m} of bins.values()){
      const electricComparison=view==='toy'||view==='electric';
      const strength=electricComparison?Math.sqrt(clamp(m/fieldScales.electric,0,1)):Math.log1p(99*m/fieldScales[view])/Math.log(100);
      arrow(p,v,strength,electricComparison);
    }
    for(const {p,v,m} of wireBins.values()){
      const strength=Math.sqrt(clamp(m/fieldScales[view],0,1));
      arrow(p,v,strength);
    }
  }
  if(view==='charge'||view==='toy'){
    const traces=view==='toy'?[['toy',result.toy_charges]]:Object.entries(result.charges);
    const scale=view==='toy'?Math.max(...result.toy_charges.density.map(Math.abs),1e-12):fieldScales.charge;
    const bands=[];
    for(const [,trace] of traces)for(let i=0;i<trace.edges.length;i++){
      const points=trace.edges[i].map(id=>vertices[id]),value=trace.density[i],strength=clamp(Math.abs(value)/scale,0,1);
      if(view==='toy'&&Math.abs(value)<1e-15)continue;
      bands.push({points,value,strength,width:.75+6*strength});
    }
    // A light outline separates strong charge bands from the resistor color.
    for(const band of bands)if(band.strength>.15){
      stroke(band.points,'rgba(19,26,28,.9)',band.width+1.5);
    }
    for(const {points,value,strength,width} of bands){
      const opacity=.12+.88*Math.sqrt(strength);
      stroke(points,chargeColor(value,opacity),width);
    }
    const markers=chargeMarkers(Object.fromEntries(traces),vertices,scale);
    for(const {point,value} of markers){
      circle(point,6,'#1a2224',chargeColor(value));
      text(point,value>=0?'+':'−',chargeColor(value),[0,4],12);
    }
  }
}
// Shape handles live outside the wire so they cannot compete with resistor drags.
function handlePoint(g, handle) {
  const offset=g.config.width/2+16/camera.zoom;
  const direction=handle.offset;
  return [handle.point[0]+direction[0]*offset,handle.point[1]+direction[1]*offset];
}
function shapeHandleAt(g, pixel) {
  if(!$('show-handles').checked)return null;
  return handles(g).find(handle=>{
    const p=screen(handlePoint(g,handle));
    return Math.hypot(p[0]-pixel[0],p[1]-pixel[1])<15;
  });
}
function draw() {
  if(!geometry)return;
  context.clearRect(0,0,width,height);context.fillStyle='#151c1e';context.fillRect(0,0,width,height);
  const g=currentGeometry(),[wx0,wy1]=world([0,0]),[wx1,wy0]=world([width,height]);
  $('mesh-info').textContent=`Loop length ${fmt(g.config.length)}`;
  const spacing=10**Math.floor(Math.log10(55/camera.zoom));
  context.fillStyle='#8ba19a38';
  for(let x=Math.ceil(wx0/spacing)*spacing;x<wx1;x+=spacing)for(let y=Math.ceil(wy0/spacing)*spacing;y<wy1;y+=spacing){const p=screen([x,y]);context.fillRect(p[0],p[1],1,1);}
  if(fresh()&&(view==='potential'||view==='magnetic'))drawFields();
  const wireWidth=g.config.width*camera.zoom;
  const transparent=fresh()&&view!=='charge';
  const resistorColor=fresh()?'#db97739c':'#b97d5e';
  stroke(g.shape.points,'#82918b70',wireWidth+2,true);
  stroke(g.shape.points,transparent?'#82918b85':'#465653',wireWidth,true);
  if(fresh()&&view==='current_strength')drawFields();
  for(const r of g.config.resistors){if(r.id===selected)stroke(spanPoints(g,r.start,r.end),'#8bd7b9',wireWidth+3);stroke(spanPoints(g,r.start,r.end),resistorColor,wireWidth);text(position(g,(r.start+r.end)/2),r.id.replaceAll('_',' '),'#f0b391',[0,-wireWidth/2-10],11);}
  const b=g.config.battery;
  stroke(spanPoints(g,-b.length/2,b.length/2),'#376e5e',wireWidth);
  text(position(g,.6*b.polarity),'+','#fffefa',[0,4],14);text(position(g,-.6*b.polarity),'−','#fffefa',[0,4],14);
  if(fresh()&&!['potential','magnetic','current_strength'].includes(view))drawFields();
  text([0,0],'battery','#a7d7c2',[-40,4],10);
  if($('show-handles').checked){
    for(const h of handles(g)){
      const p=handlePoint(g,h);
      stroke([h.point,p],'#92c8ac9c',1);
      circle(p,10,'#326c56','#e8e9e7');
      text(p,h.axes.length===2?'✥':h.id==='right'?'↔':'↕','#fffefa',[0,4],15);
    }
    const r=g.config.resistors.find(r=>r.id===selected);
    if(r)for(const endpoint of [r.start,r.end])circle(position(g,endpoint),5,'#1a2224','#dd9c77');
  }
  if(fresh()){
    if(['potential','magnetic'].includes(view))$('legend').textContent=`Blue −${fmt(fieldScales[view])} · neutral 0 · orange +${fmt(fieldScales[view])}`;
    else if(view==='charge')$('legend').textContent=`σ: blue −, magenta + · thicker/darker = larger |σ| · max ${fmt(fieldScales.charge)}`;
    else if(view==='power_exchange')$('legend').textContent=`Blue − absorption · neutral 0 · orange + supply · scale ${fmt(fieldScales.power_exchange)}`;
    else if(view==='toy')$('legend').textContent='Battery-terminal charges: magenta +, blue − · arrows show the field they make';
    else if(view==='electric')$('legend').textContent='E arrows show the force direction for a positive charge · fainter = weaker field';
    else if(view==='current')$('legend').textContent='J arrows show conventional current · fainter = lower current density';
    else if(view==='current_strength')$('legend').textContent=`Red = current crowding above even flow · reference |J| ${fmt(fieldScales.current_reference)}`;
    else $('legend').textContent='Energy-flow arrows show where energy travels · darker = stronger flow';
  } else $('legend').textContent='Circuit preview · calculate to reveal fields';
  staleState();
}
function eventPoint(event){const box=canvas.getBoundingClientRect();return [event.clientX-box.left,event.clientY-box.top];}
function probe(point) {
  if(!fresh()){$('probe').textContent='Drag a resistor or a shape handle. Fields update after you release.';return;}
  for(let i=0;i<result.triangles.length;i++){
    const ids=result.triangles[i],a=result.vertices[ids[0]],b=result.vertices[ids[1]],c=result.vertices[ids[2]];
    const det=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);
    const u=((b[1]-c[1])*(point[0]-c[0])+(c[0]-b[0])*(point[1]-c[1]))/det;
    const v=((c[1]-a[1])*(point[0]-c[0])+(a[0]-c[0])*(point[1]-c[1]))/det;
    if(u<0||v<0||u+v>1)continue;
    let value;
    if(view==='potential'||view==='magnetic')value=fmt(u*result[view][ids[0]]+v*result[view][ids[1]]+(1-u-v)*result[view][ids[2]]);
    else if(view==='power_exchange')value=fmt(result.power_exchange[i]);
    else if(view==='current_strength')value=fmt(Math.hypot(...result.current[i]));
    else if(view==='charge'){
      let nearest=Infinity,density=0;
      for(const trace of Object.values(result.charges))for(let j=0;j<trace.edges.length;j++){
        const [a,b]=trace.edges[j].map(id=>result.vertices[id]),dx=b[0]-a[0],dy=b[1]-a[1];
        const t=clamp(((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(dx*dx+dy*dy),0,1);
        const distance=Math.hypot(point[0]-a[0]-t*dx,point[1]-a[1]-t*dy);if(distance<nearest){nearest=distance;density=trace.density[j];}
      }
      value=nearest*camera.zoom<15?`${fmt(density)} on nearest edge`:'move close to a surface';
    } else {const field=result[view==='toy'?'toy_electric':view];value=`(${fmt(field[i][0])}, ${fmt(field[i][1])})`;}
    $('probe').textContent=`${result.region_names[result.regions[i]]} · ${lessons[view][0]}: ${value} · x ${point[0].toFixed(2)}, y ${point[1].toFixed(2)}`;return;
  }
  $('probe').textContent='Outside the calculated domain. Zoom in to inspect the fields.';
}
canvas.addEventListener('pointerdown',event=>{
  if(!geometry||checking)return;
  const pixel=eventPoint(event),point=world(pixel),g=geometry;
  const distance=p=>{const q=screen(p);return Math.hypot(q[0]-pixel[0],q[1]-pixel[1]);};
  const shapeHandle=shapeHandleAt(g,pixel);
  let action=shapeHandle?{type:'shape',id:shapeHandle.id,anchor:shapeHandle.point}:null;
  if(!action&&$('show-handles').checked){const r=g.config.resistors.find(r=>r.id===selected);if(r)for(const end of ['start','end'])if(distance(position(g,r[end]))<11)action={type:'end',id:r.id,end};}
  if(!action){const s=nearestDistance(g,point);for(const r of g.config.resistors)if(s>=r.start&&s<=r.end&&distance(position(g,s))<Math.max(12,g.config.width*camera.zoom/2))action={type:'resistor',id:r.id,offset:s-(r.start+r.end)/2};}
  if(!action&&distance([0,0])<g.config.battery.length*camera.zoom/2){selected='battery';updateControls();draw();return;}
  drag={...(action||{type:'pan'}),pixel,camera:{...camera},base:clone(g),moved:false};
  if(action?.type==='shape')drag.lastPoint=action.anchor;
  if(action?.type==='resistor'||action?.type==='end'){selected=action.id;updateControls();}
  canvas.setPointerCapture(event.pointerId);canvas.style.cursor=action?'grabbing':'move';event.preventDefault();
});
canvas.addEventListener('pointermove',event=>{
  const pixel=eventPoint(event),point=world(pixel);
  if(!drag){
    if(!geometry)return;
    const handle=shapeHandleAt(geometry,pixel);
    const distance=nearestDistance(geometry,point),onWire=screen(position(geometry,distance));
    const resistor=geometry.config.resistors.some(item=>distance>=item.start&&distance<=item.end)&&Math.hypot(onWire[0]-pixel[0],onWire[1]-pixel[1])<Math.max(12,geometry.config.width*camera.zoom/2);
    canvas.style.cursor=handle?(handle.axes.length===2?'move':handle.id==='right'?'ew-resize':'ns-resize'):resistor?'move':'';
    if(handle)$('probe').textContent=handle.axes.length===2?'Drag this corner while keeping the same loop shape.':`Drag ${handle.id==='right'?'left or right':'up or down'} to resize the loop.`;
    else probe(point);
    return;
  }
  if(Math.hypot(pixel[0]-drag.pixel[0],pixel[1]-drag.pixel[1])<3&&!drag.moved)return;
  drag.moved=true;
  if(drag.type==='pan'){camera.x=drag.camera.x-(pixel[0]-drag.pixel[0])/camera.zoom;camera.y=drag.camera.y+(pixel[1]-drag.pixel[1])/camera.zoom;draw();return;}
  if(drag.type==='shape'){
    // Apply the pointer displacement, not its absolute position: the visible
    // handle is offset from the actual wire, and may be grabbed near its edge.
    const target=[drag.anchor[0]+(pixel[0]-drag.pixel[0])/camera.zoom,
                  drag.anchor[1]-(pixel[1]-drag.pixel[1])/camera.zoom];
    const next=constrainedReshape(drag.base,drag.id,drag.lastPoint,target);
    drag.lastPoint=next.point;preview=next.geometry;error();
  }
  else {
    preview=clone(drag.base);const r=preview.config.resistors.find(r=>r.id===drag.id),s=nearestDistance(preview,point);
    if(drag.type==='resistor'){const length=r.end-r.start,center=clamp(s-drag.offset,length/2+1.1,preview.config.length-length/2-1.1);moveResistor(preview,r,center-length/2);}
    else resizeResistor(preview,r,drag.end,drag.end==='start'?clamp(s,1.1,r.end-.5):clamp(s,r.start+.5,preview.config.length-1.1));
  }
  draw();
});
canvas.addEventListener('pointerup',()=>{if(!drag)return;const changed=preview;const moved=drag.moved;
  const shapeMoved=drag.type!=='shape'||drag.lastPoint.some((value,axis)=>Math.abs(value-drag.anchor[axis])>1e-6);
  drag=null;canvas.style.cursor='';if(changed&&moved&&shapeMoved)commit(changed);else {preview=null;draw();}
});
canvas.addEventListener('pointercancel',()=>{drag=null;preview=null;canvas.style.cursor='';draw();});
canvas.addEventListener('wheel',event=>{event.preventDefault();const p=eventPoint(event),before=world(p);camera.zoom=clamp(camera.zoom*Math.exp(-event.deltaY*.001),5,180);const after=world(p);camera.x+=before[0]-after[0];camera.y+=before[1]-after[1];draw();},{passive:false});
$('calculate').onclick=calculate;$('fit').onclick=fit;
$('mode-present').onclick=()=>setMode(true);$('mode-explore').onclick=()=>setMode(false);
$('story-back').onclick=()=>showStory(storyIndex-1);
$('story-next').onclick=()=>storyIndex===storyScenes.length-1?setMode(false):showStory(storyIndex+1);
$('story-reveal').onclick=()=>{answerRevealed=true;$('story-reveal').hidden=true;$('story-conclusion').hidden=false;};
$('zoom-in').onclick=()=>{camera.zoom=clamp(camera.zoom*1.25,5,180);draw();};$('zoom-out').onclick=()=>{camera.zoom=clamp(camera.zoom/1.25,5,180);draw();};
$('undo').onclick=()=>{if(history.length&&!checking)commit(clone(history.at(-1)),{undo:true});};
$('show-mesh').onchange=draw;$('show-handles').onchange=draw;
$('resolution').onchange=()=>{revision++;draw();schedule();};$('auto').onchange=()=>{clearTimeout(timer);staleState();if($('auto').checked&&!fresh())schedule();};
function changeShape(kind){
  const next=clone(shapeStates.get(kind)||defaultGeometry);
  if(!shapeStates.has(kind))next.shape=preset(kind,defaultGeometry.config.length).shape;
  commit(next,{resetView:true});
}
$('shape').onchange=()=>changeShape($('shape').value);
$('demo-shape').onchange=()=>changeShape($('demo-shape').value);
$('new-loop').onclick=()=>{const length=Number($('loop-length').value);if(length<24||length>80||!Number.isFinite(length)){error('Choose a loop length between 24 and 80.');return;}commit(preset($('shape').value,length),{resetView:true});};
$('reset').onclick=async()=>{try{const demo=await request('/presentation.json');$('auto').checked=true;showStory(0);commit(demo,{resetView:true,resetAll:true});}catch(e){error(e.message);}};
$('add-resistor').onclick=()=>{
  if(checking)return;const next=clone(geometry);if(next.config.resistors.length>=6){error('This prototype supports up to six resistors.');return;}
  const corners=segments(next).slice(1).map(e=>e.start),gap=next.config.width/2+.15,length=2;
  for(let start=2;start<next.config.length-3;start+=.25){const end=start+length;
    if(corners.some(s=>Math.abs(s-start)<gap||Math.abs(s-end)<gap)||next.config.resistors.some(r=>start<r.end+.25&&end>r.start-.25))continue;
    let i=1;while(next.config.resistors.some(r=>r.id===`R${i}`))i++;selected=`R${i}`;next.config.resistors.push({id:selected,start,end,resistivity:20});commit(next);return;}
  error('There is no clear space for another resistor. Move or shorten an existing one.');
};
$('field-select').onchange=()=>{view=$('field-select').value;updateLesson();};
for(const name of ['controls','physics']){
  $(`toggle-${name}`).onclick=()=>setPanel(name,$(`toggle-${name}`).getAttribute('aria-expanded')!=='true');
  $(`close-${name}`).onclick=()=>setPanel(name,false);
}
new ResizeObserver(resize).observe($('viewport'));
try {geometry=await request('/presentation.json');defaultGeometry=clone(geometry);shapeStates.set(shapeKind(geometry),clone(geometry));updateControls();resize();fit();renderStory();calculate();}
catch(e){error(`Could not load the app: ${e.message}`);message('Reload this page after starting the local server.');}

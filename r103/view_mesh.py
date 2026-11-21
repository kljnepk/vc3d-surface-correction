"""Render two local tifxyz meshes as a standalone browser view."""
import html
import json
from pathlib import Path

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Surface comparison in 3D</title><style>
*{box-sizing:border-box}body{margin:0 auto;max-width:1000px;padding:24px;font:16px/1.5 system-ui,sans-serif;color:#203846;background:#fff}h1{font:600 2rem Georgia,serif;margin:.45em 0}
nav{display:flex;flex-wrap:wrap;align-items:center;gap:.7rem 1.2rem}label{display:flex;gap:.35rem;align-items:center}canvas{display:block;background:#f6f8f8;border:1px solid #c9d2d4;width:100%;height:clamp(290px,52vw,470px);touch-action:none}
.vc{color:#24538b}.corrected{color:#b24b09}a{color:#174e78}button{font:inherit}
</style></head><body><a href="VIEW.html">CT surface views and sections</a><h1>Surface comparison in 3D</h1>
<p>CASE_LABEL. Each line connects measured mesh vertices. Both surfaces use the same voxel scale.</p>
<nav aria-label="Visible surfaces"><label class="vc"><input type="checkbox" id="vc" checked> VC3D</label><label class="corrected"><input type="checkbox" id="r103" checked> R103</label></nav>
<canvas id="mesh" aria-label="Wireframes of both strips in the same three-dimensional coordinate frame"></canvas>
<nav aria-label="View controls"><label>Rotation <input type="range" min="-180" max="180" id="yaw" value="26"></label>
<label>Tilt <input type="range" min="-89" max="89" id="tilt" value="23"></label><label>Zoom <input type="range" min="60" max="240" id="zoom" value="100"></label>
<button id="reset" type="button">Reset view</button></nav>
<p>Drag to rotate, or adjust the sliders. The marked point is the center of the shared starting edge. Use the CT sections to inspect the underlying material.</p>
<script>
const surfaces = MESH_DATA;
const $ = id => document.getElementById(id);
const canvas = $('mesh'), ctx = canvas.getContext('2d');
const points = surfaces.flat(2);
const bounds = [0,1,2].map(i => {const p=points.map(v=>v[i]);return [Math.min(...p),Math.max(...p)];});
const center = bounds.map(([a,b])=>(a+b)/2);
const radius = Math.max(1,...points.map(p => Math.hypot(...p.map((x,i)=>x-center[i]))));
function draw(){
 const w=canvas.clientWidth,h=canvas.clientHeight,dpr=Math.min(window.devicePixelRatio||1,3);
 canvas.width=Math.round(w*dpr); canvas.height=Math.round(h*dpr);
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 const a=+$('yaw').value*Math.PI/180,b=+$('tilt').value*Math.PI/180,s=.44*Math.min(w,h)*$('zoom').value/(100*radius);
 function project(p){
  const [x,y,z]=p.map((v,i)=>v-center[i]);
  const u=x*Math.cos(a)-y*Math.sin(a),v=x*Math.sin(a)+y*Math.cos(a);
  return [w/2+s*u,h/2-s*(v*Math.sin(b)+z*Math.cos(b))];
 }
 function line(ps){ctx.beginPath();ps.forEach((p,i)=>{const q=project(p);if(i) ctx.lineTo(...q);else ctx.moveTo(...q);});ctx.stroke();}
 [['vc','#24538b'],['r103','#b24b09']].forEach(([id,color],i)=>{
  if(!$(id).checked)return;
  ctx.strokeStyle=color;ctx.lineWidth=1;ctx.globalAlpha=.65;
  const g=surfaces[i];g.forEach(line);
  for(let c=0;c<g[0].length;c++)line(g.map(row=>row[c]));
  ctx.globalAlpha=1;ctx.lineWidth=1.9;line(g.map(row=>row[4]));
 });
 const origin=project(ANCHOR);ctx.fillStyle='#18242a';ctx.beginPath();ctx.arc(origin[0],origin[1],3.3,0,2*Math.PI);ctx.fill();
}
const limit=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
for(const id of ['yaw','tilt','zoom','vc','r103'])$(id).addEventListener('input',draw);
let drag=null;
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY,+$('yaw').value,+$('tilt').value];canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!drag)return;$('yaw').value=((drag[2]+(e.clientX-drag[0])*.65+540)%360)-180;$('tilt').value=limit(drag[3]+(e.clientY-drag[1])*.5,-89,89);draw();};
canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};
$('reset').onclick=()=>{$('yaw').value=26;$('tilt').value=23;$('zoom').value=100;$('vc').checked=$('r103').checked=true;draw();};
window.addEventListener('resize',draw);draw();
</script></body></html>
"""

def write_view(path, base, corrected, anchor,
               label='PHerc. Paris. 4 · region 01744_02256_04816 · start 0'):
    # Tifxyz is displayed as X,Y,Z. Surface routines store arrays as Z,Y,X.
    ribbons = [mesh[..., ::-1].astype(float).tolist() for mesh in (base, corrected)]
    page = TEMPLATE.replace('MESH_DATA', json.dumps(ribbons, separators=(',', ':'), allow_nan=False))
    page = page.replace('ANCHOR', json.dumps(anchor[::-1].astype(float).tolist(), allow_nan=False))
    Path(path).write_text(page.replace('CASE_LABEL', html.escape(label)), encoding='utf-8')

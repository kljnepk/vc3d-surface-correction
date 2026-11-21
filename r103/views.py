"""Genuine CT samples in the shared local coordinate frame, for inspecting a pair of meshes."""
import html
import json
from pathlib import Path
import numpy as np
from .image import points, png
from .sampling import sample_ct
from .profiles import normals


def midline_length(mesh):
    return float(np.linalg.norm(np.diff(mesh[:, 4].astype(float), axis=0), axis=1).sum())


def sheet(ct, mesh, offset):
    rows = np.linspace(0, len(mesh) - 1, 1 + (len(mesh) - 1) * 5)[None, :]
    cols = np.linspace(0, 8, 49)[:, None]
    surface = points(mesh, rows, cols)
    unit = points(normals(mesh), rows, cols)
    unit /= np.maximum(np.linalg.norm(unit, axis=-1, keepdims=True), 1e-9)
    xyz = surface + offset * unit
    valid = np.all((xyz >= 0) & (xyz <= np.asarray(ct.shape) - 1), axis=-1)
    return png(np.where(valid, sample_ct(ct, np.clip(xyz, 0, np.asarray(ct.shape) - 1)), 0))


def cross(ct, before, after, column=None, row=None):
    if column is not None:
        rows = np.linspace(0, len(before) - 1, 1 + (len(before) - 1) * 6)
        cols = column
    else:
        rows = row
        cols = np.linspace(0, 8, 97)
    p, q = points(before, rows, cols), points(after, rows, cols)
    delta = q - p
    gap = np.linalg.norm(delta, axis=1)
    direction = points(normals(before), rows, cols)
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-9)
    sign = np.where((direction * delta).sum(axis=-1) < 0, -1., 1.)
    moved = gap > 1e-7
    direction[moved] = delta[moved] / gap[moved, None] * sign[moved, None]
    span = max(9., float(np.ceil(gap.max() / 2 + 2)))
    height = 91
    xyz = (p + q)[None, :, :] / 2 + np.linspace(span, -span, height)[:, None, None] * direction
    valid = ((xyz >= 0) & (xyz <= np.array(ct.shape) - 1)).all(axis=-1)
    g = np.where(valid, sample_ct(ct, np.clip(xyz, 0, np.array(ct.shape) - 1)), 0)
    rgb = np.repeat(g[:, :, None], 3, axis=2)
    levels = gap * sign / 2
    y1, y2 = (np.rint((span - t) * (height-1)/(2*span)).astype(int) for t in (-levels, levels))
    x = np.arange(len(gap))
    rgb[y1, x] = (71, 211, 245)
    rgb[y2, x] = (255, 153, 53)
    same = y1 == y2
    rgb[y1[same], x[same]] = (255, 255, 255)
    return [png(g), png(rgb)]


def context_plane(ct, mesh, row):
    # Sample a true flat patch in the local tangent plane. Surrounding pixels are CT context.
    v = mesh.astype(float)
    p = points(v, row, 4)
    along = points(np.gradient(v,axis=0), row, 4)
    along /= max(np.linalg.norm(along), 1e-9)
    across = points(np.gradient(v,axis=1), row, 4)
    across -= np.dot(across,along)*along
    across /= max(np.linalg.norm(across), 1e-9)
    grid = np.linspace(-18,18,109)
    xyz = p + grid[None,:,None]*along + grid[:,None,None]*across
    valid = ((xyz>=0)&(xyz<=np.array(ct.shape)-1)).all(axis=-1)
    values = np.where(valid, sample_ct(ct,np.clip(xyz,0,np.array(ct.shape)-1)),0)
    return png(values)


STYLE = """
:root{color-scheme:light;--ink:#152e32;--muted:#486469;--edge:#c5d4cf;--paper:#fafbf8}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.55 system-ui,sans-serif}
main{max-width:1000px;margin:auto;padding:32px 24px 64px}h1{font:600 clamp(2rem,5vw,3.1rem)/1.05 Georgia,serif;max-width:24ch}
h2{font:600 1.5rem/1.2 Georgia,serif;margin-top:1.8em}p{max-width:85ch}figure{margin:0}figcaption{font-weight:600;margin:0 0 .3rem}
img{display:block;width:100%;background:#151d21}.cut{max-height:290px;object-fit:contain}.sheets{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:.75em 0}select{font:inherit;padding:.22em .45em}
input{accent-color:#245950}small,.note{color:var(--muted)}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:5px 10px;text-align:left;border-bottom:1px solid var(--edge)}a{color:#165b5c}.meta{border-left:3px solid #5b8681;padding-left:12px}
@media(max-width:650px){.sheets{grid-template-columns:1fr}}@media print{.controls{display:none}}
"""

def page(ct, before, after, dest, note, measurements='', official=None,
         label='PHerc. Paris. 4 · region 01744_02256_04816 · start 0'):
    if before.shape != after.shape or before.ndim != 3 or before.shape[1:] != (9, 3):
        raise ValueError('Expected two matched strips.')
    surfaces = [[sheet(ct, mesh, depth) for mesh in (before, after)] for depth in (-2, -1, 0, 1, 2)]
    cuts = [cross(ct, before, after, col) for col in (0, 2, 4, 6, 8)]
    transects = [cross(ct, before, after, row=(len(before)-1)*t/4) for t in range(5)]
    contexts = [[context_plane(ct, mesh, (len(mesh)-1)*t/4) for mesh in (before, after)] for t in range(5)]
    figures = lambda a,b: ('<div class="sheets"><figure><figcaption>Before · VC3D</figcaption><img id="before" alt="CT sampled on the VC3D strip" src="'+a+'"></figure>'
                           '<figure><figcaption>After · R103</figcaption><img id="after" alt="CT sampled on the corrected strip" src="'+b+'"></figure></div>')
    proof = ''
    if official:
        proof = ('<h2>Official VC3D surface renders</h2><p>These two central-layer images were produced by '
                 '<code>vc_render_tifxyz</code> on the same crop and displayed with a common intensity scale. The data above '
                 'use independent trilinear sampling. See <code>VC3D_RENDER/</code> for the TIFFs.</p>'
                 '<div class="sheets">' + ''.join('<figure><figcaption>'+label+'</figcaption><img alt="'+label+
                 ' official render" src="'+img+'"></figure>' for label,img in zip(('VC3D','R103'),official)) + '</div>')
    text = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>PHerc. Paris. 4 · one sheet, two meshes</title><style>'+STYLE+'</style><main>'
            '<h1>Following one papyrus sheet in CT</h1><p class="meta">'+html.escape(label)+'</p>'
            '<p>'+html.escape(note)+'</p><p>Shared window: '+str(len(before))+' rows × 9 columns. Centreline length along the meshes: '
            f'{midline_length(before):.1f} voxels before and {midline_length(after):.1f} after. Both meshes use the same crop.</p>'
            '<h2>Across the full length</h2><p>Follow the texture from left to right. The two panels use the same grayscale, '
            'crop and depth offset in voxels along each mesh normal. The native ribbon is about 12 voxels wide. Increasing display size adds no detail.</p>'
            '<div class="controls"><label>Depth <select id="dep">'+''.join('<option value="'+str(i)+'"'+(' selected' if i==2 else '')+'>'+str(v)+'</option>' for i,v in enumerate((-2,-1,0,1,2)))+
            '</select></label></div>'+figures(*surfaces[2])+
            '<h2>Does the mesh stay on the sheet?</h2><p>A longitudinal CT section passes through both meshes at the chosen column. Cyan marks VC3D, orange marks R103 and white marks their overlap at display resolution. Turn off the lines to inspect the material beneath them.</p>'
            '<div class="controls"><label>Column <select id="col">'+''.join('<option value="'+str(i)+'"'+(' selected' if i==2 else '')+'>'+str(v+1)+' of 9</option>' for i,v in enumerate((0,2,4,6,8)))+
            '</select></label><label><input type="checkbox" id="line" checked> Show mesh lines</label></div>'
            '<img id="cut" class="cut" alt="Longitudinal CT cross-section" src="'+cuts[2][1]+'">'
            +'<h2>Five positions along the strip</h2><p>These sections look across the width of the same sheet at five positions from start to finish. They share the mesh-line toggle above. The chosen row runs from the left to the right edge of each section.</p>'
            '<div class="controls"><label>Position <select id="station">'+''.join('<option value="'+str(i)+'"'+(' selected' if i==2 else '')+'>'+str(i*25)+'% of rows</option>' for i in range(5))+'</select></label></div>'
            '<img id="sect" style="width:min(100%,420px)" alt="Crosswise section through the selected row" src="'+transects[2][1]+'">'
            +'<h2>Material around that position</h2><p>These 36 × 36 voxel fields sample a flat plane tangent to each mesh at its central vertex. '
            'They give a wider view of nearby texture on the same grayscale. Outside the narrow mesh they show surrounding CT; '
            'only the measured strip has a sheet-assignment score. Change Position above to move both planes.</p>'
            '<div class="sheets"><figure><figcaption>Local plane · VC3D</figcaption><img id="context0" alt="Raw CT plane tangent to the public mesh" src="'+contexts[2][0]+'"></figure>'
            '<figure><figcaption>Local plane · R103</figcaption><img id="context1" alt="Raw CT plane tangent to the corrected mesh" src="'+contexts[2][1]+'"></figure></div>'
            +proof+measurements+'<p class="note">Dark pixels beyond the CT boundary indicate missing samples. Annotation overlap describes sheet position. Identifying individual fibers requires visual inspection.</p>'
            '<script>const planes='+json.dumps(surfaces,separators=(',',':'))+', cuts='+json.dumps(cuts,separators=(',',':'))+', transverse='+json.dumps(transects,separators=(',',':'))+', fields='+json.dumps(contexts,separators=(',',':'))+
            ';const e=id=>document.getElementById(id);function update(){const pair=planes[e("dep").value];e("before").src=pair[0];e("after").src=pair[1];e("cut").src=cuts[e("col").value][Number(e("line").checked)];e("sect").src=transverse[e("station").value][Number(e("line").checked)];const f=fields[e("station").value];e("context0").src=f[0];e("context1").src=f[1];}for(const id of ["dep","col","line","station"])e(id).addEventListener("change",update);</script></main></html>')
    Path(dest).write_text(text, encoding='utf-8')

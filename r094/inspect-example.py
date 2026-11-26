#!/usr/bin/env python3
"""Locally render authorized CT against the original frozen output meshes; no ink model."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
HERE=Path(__file__).resolve().parent
ARMS=('VC3D','R094','R103');COLORS=['#52b9e8','#4be0a3','#df9cf4'];STYLES=['--','-',':']
def unit(a):return a/np.linalg.norm(a)
def triangles(g):
    a,b,c,d=g[:-1,:-1],g[1:,:-1],g[:-1,1:],g[1:,1:]
    return np.stack((np.stack((a,b,c),-2),np.stack((c,b,d),-2)),-3).reshape(-1,3,3)
def sample(ct,p):return map_coordinates(ct.astype(float),np.moveaxis(p,-1,0),order=1,mode='constant',cval=np.nan,prefilter=False)
def surface_points(g,rr,cc):
    # Interpolate the same two triangles used by the distance metric, not a bilinear quad.
    r=np.minimum(np.floor(rr).astype(int),g.shape[0]-2);c=np.minimum(np.floor(cc).astype(int),g.shape[1]-2);u=(rr-r)[...,None];v=(cc-c)[...,None];a,b,d,e=g[r,c],g[r+1,c],g[r,c+1],g[r+1,c+1]
    return np.where(u+v<=1,a+u*(b-a)+v*(d-a),e+(1-u)*(d-e)+(1-v)*(b-e))
def section(ct,meshes,row):
    g=meshes['R094'];center=g[row,4];t=unit(g[row+1,4]-g[row-1,4]);lat=g[row,-1]-g[row,0];u=unit(lat-t*np.dot(lat,t));v=unit(np.cross(t,u));radius=15;axis=np.linspace(-radius,radius,181);X,Y=np.meshgrid(axis,axis);img=sample(ct,center+X[...,None]*u+Y[...,None]*v);segments={}
    for a in ARMS:
        tri=triangles(meshes[a][2:]);dist=(tri-center)@t;lines=[]
        for tr,ds in zip(tri,dist):
            pts=[]
            for i,j in ((0,1),(1,2),(2,0)):
                if abs(ds[i])<1e-8:pts.append(tr[i])
                if ds[i]*ds[j]<-1e-16:pts.append(tr[i]+(tr[j]-tr[i])*ds[i]/(ds[i]-ds[j]))
            if len(pts)>=2:
                xy=(np.asarray(pts)-center)@np.stack([u,v],axis=1);i=np.argmax(np.linalg.norm(xy-xy[0],axis=1))
                if np.linalg.norm(xy[i]-xy[0])>1e-7:lines.append([xy[0],xy[i]])
        segments[a]=lines
    return img,segments,radius

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();meta=json.loads((args.prepared/'provenance.json').read_text());e=next(e for e in json.loads((HERE/'examples.json').read_text())['cases'] if e['case']==meta['case']);raw=(args.prepared/'ct.zarr/0/0.0.0').read_bytes();assert hashlib.sha256(raw).hexdigest()==e['normalized_crop_sha256'],'CT mismatch';ct=np.frombuffer(raw,dtype=np.uint8).reshape(128,128,128);meshes={a:np.asarray(m,dtype='<f4') for a,m in e['meshes_zyx'].items()};assert all(hashlib.sha256(m.tobytes()).hexdigest()==e['mesh_sha256'][a] for a,m in meshes.items());meshes={a:m.astype(float) for a,m in meshes.items()}
    if args.out.exists():p.error('Choose a new output filename')
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.9),gridspec_kw={'height_ratios':[.8,1]});fig.subplots_adjust(left=.07,right=.97,bottom=.13,top=.82,hspace=.5,wspace=.3);fig.text(.07,.95,'R094 | local CT inspection',fontsize=22,weight='bold');fig.text(.07,.905,f"PHerc. {e['scroll']} · case {e['case']} · {e['id']}",fontsize=10);fig.text(.07,.87,'Frozen output surfaces; shared grayscale, no ink detector, no manual continuation displayed.',fontsize=10)
    for j,a in enumerate(ARMS):
        ax=axes[0,j];g=meshes[a]; rr,cc=np.meshgrid(np.linspace(2,32,181),np.linspace(0,8,49),indexing='ij');points=surface_points(g,rr,cc);ax.imshow(sample(ct,points).T,origin='lower',aspect='auto',cmap='gray',vmin=0,vmax=255,extent=[2,32,0,8]);ax.set(title=f"{a} · {e['scores'][a]}/41",xlabel='Mesh row',ylabel='Column' if j==0 else '')
    for j,row in enumerate((10,20,30)):
        ax=axes[1,j];im,seg,r=section(ct,meshes,row);ax.imshow(im,cmap='gray',origin='lower',extent=[-r,r,-r,r],vmin=0,vmax=255)
        for a,c,ls in zip(ARMS,COLORS,STYLES):
            if seg[a]:ax.add_collection(LineCollection(seg[a],colors=c,linestyles=ls,linewidths=1.7))
        ax.set(xlim=(-r,r),ylim=(-r,r),title=f'Plane through R094 row {row}',xlabel='Lateral (vox)',ylabel='Normal (vox)' if j==0 else '')
    fig.legend([Line2D([0],[0],color=c,ls=ls,lw=2) for c,ls in zip(COLORS,STYLES)],ARMS,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False);fig.text(.07,.035,'Local diagnostic: planes are defined by R094 geometry, not the held-back fibre. No sheet-identity claim. CT source: Vesuvius Challenge fiber-skeletons (2025-07).',fontsize=8);fig.savefig(args.out,dpi=160);plt.close(fig);print('LOCAL_CT_RENDER=PASS',args.out)
if __name__=='__main__':main()

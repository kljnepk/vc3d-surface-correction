"""Uniform displacement correction for a complete VC3D strip in local Z,Y,X coordinates."""
import numpy as np
from .profile_path import ANCHOR, MAX_STEP, OFFSETS, PROFILE, REGULARITY, STAY, ZERO, best_path, distances, sample_ct, trace_mesh


def correct_uniform(ct, seed, mesh, valid):
    old, trace = trace_mesh(ct,seed,mesh,valid)
    if mesh.shape!=(13,9,3) or ct.shape!=(128,128,128) or not valid.all():
        raise ValueError('Expected a 128-cubed CT crop and a complete 13 by 9 mesh.')
    m = mesh.astype(float)
    normals = np.cross(np.gradient(m,axis=0),np.gradient(m,axis=1))
    length = np.linalg.norm(normals,axis=-1,keepdims=True)
    if np.any(length<1.e-6):
        return mesh.copy()
    normals /= length
    if (np.any(np.sum(normals[1:]*normals[:-1],axis=-1)<=0) or
            np.any(np.sum(normals[:,1:]*normals[:,:-1],axis=-1)<=0)):
        return mesh.copy()
    coords = m[:,:,None,:] + normals[:,:,None,:]*np.arange(-8,8.001,.125)[None,None,:,None]
    if np.any(coords<0) or np.any(coords>np.array(ct.shape)-1):
        return mesh.copy()
    values = sample_ct(ct,coords)
    profiles = values[:,:,np.arange(-24,25)[:,None]+8*PROFILE.astype(int)[None,:]+64]
    centered = profiles-profiles.mean(axis=-1,keepdims=True)
    profiles = centered / np.sqrt(np.mean(centered**2,axis=-1,keepdims=True)+16.)
    middle = profiles[:,3:6,::4].transpose(0,2,1,3)
    anchor = np.median(middle[:3,ZERO],axis=0)
    front = middle[2:]
    change = OFFSETS[:,None]-OFFSETS[None,:]
    penalty = np.where(abs(change)<=MAX_STEP,REGULARITY*change**2,np.inf)
    edges = [distances(a[:,None],b[None,:])+penalty for a,b in zip(front[:-1],front[1:])]
    forward = best_path(edges,ZERO,[ANCHOR*distances(s,anchor)+STAY*OFFSETS**2 for s in front[1:]])
    reverse = best_path([e.T for e in reversed(edges)],forward[-1],[np.zeros(len(OFFSETS)) for _ in edges])[::-1]
    zero = (0,)*13
    paths = {zero}
    for path in (forward,reverse):
        if path[0]==ZERO:
            paths.add((0,0)+tuple(int(round(8*OFFSETS[i])) for i in path))
    oldpath = (0,0)+tuple(int(round(8*v)) for v in trace['forward_vox']) if trace['changed'] else zero
    paths.add(oldpath)
    target = np.median(profiles[:3,:,24],axis=0)
    rows,cols=np.array([4,6,8,10,12])[:,None],np.arange(0,9,2)
    best_key,best_grid=None,None
    for path in paths:
        field=np.broadcast_to(np.asarray(path,dtype=np.int16)[:,None],(13,9)).copy()
        grid=(m+normals*(field/8.)[...,None]).astype('<f4')
        if path not in (zero,oldpath):
            corners=lambda x:(x[:-1,:-1],x[1:,:-1],x[:-1,1:],x[1:,1:])
            base,g=corners(m),corners(grid.astype(float))
            if any(np.any(np.sum(np.cross(base[j]-base[i],base[k]-base[i])*np.cross(g[j]-g[i],g[k]-g[i]),axis=-1)<=1.e-8)
                   for i,j,k in ((0,1,2),(3,2,1))):
                continue
        observed=profiles[rows,cols,field[rows,cols]+24]
        cost=float(np.mean(np.log1p(np.mean((observed-target[cols])**2,axis=-1))))
        cost+=STAY*float(np.mean((field[rows,cols]/8.)**2)) + REGULARITY*float(np.mean((np.diff(field[2:],axis=0)/8.)**2))
        # Break ties by the smallest total displacement, then by the field bytes.
        key=(cost,int(np.abs(field).sum()),field.tobytes())
        if best_key is None or key<best_key:
            best_key,best_grid=key,grid
    return best_grid

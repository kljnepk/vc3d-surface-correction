"""CT sampling, normal estimation and strip geometry."""
from hashlib import sha256
import numpy as np
from .sampling import sample_ct
SENSORS=(0,2,4,6,8)
VALIDATION=(1,3,5,7)
DEPTH=np.arange(-4.,5.)
PROFILE=np.arange(-3.,4.)


def fingerprint(value):
    return sha256(np.ascontiguousarray(value, dtype='<f4').tobytes()).hexdigest()


def normals(grid):
    grid = grid.astype(float)
    along = np.gradient(grid,axis=0)
    across = np.gradient(grid,axis=1)
    n = np.cross(along,across)
    return n / np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1.e-10)


def features(image, points, directions):
    xyz = points[...,None,:] + directions[...,None,:]*PROFILE[:,None]
    limit = np.array(image.shape)-1
    valid = ((xyz>=0)&(xyz<=limit)).all(axis=(-1,-2)) & (np.linalg.norm(directions,axis=-1)>.9)
    values = sample_ct(image,np.clip(xyz,0,limit))
    level = values.mean(axis=-1,keepdims=True)
    shape = values-level
    shape /= np.sqrt(np.mean(shape**2,axis=-1,keepdims=True)+64.)
    return np.concatenate((shape,level/64.),axis=-1), valid


def distance(first, second):
    delta = first-second
    return np.log1p(np.mean(delta[...,:-1]**2,axis=-1)) + .12*np.minimum(np.abs(delta[...,-1]),3.)


def anchor(f,valid):
    # An invalid local seed profile contributes no evidence to its column.
    ref=np.zeros_like(f[0])
    for i in range(f.shape[1]):
        if valid[:,i].any():
            ref[i]=np.median(f[valid[:,i],i],axis=0)
    return ref,valid.any(axis=0)


def five_tracks(ct,base,n):
    m,unit=base[:,SENSORS,:],n[:,SENSORS,:]
    centers=m[...,None,:]+unit[...,None,:]*DEPTH[:,None]
    profile,valid=features(ct,centers,unit[...,None,:])
    zero=len(DEPTH)//2
    ref,visible=anchor(profile[:3,:,zero,:],valid[:3,:,zero])
    unary=distance(profile,ref[:,None,:])+.012*DEPTH**2
    unary=np.where(valid & visible[:,None],unary,3.+.012*DEPTH**2)
    # A missing profile therefore favours returning towards zero.
    change=DEPTH[:,None]-DEPTH[None,:]
    penalty=np.where(np.abs(change)<=1.,.09*change**2,np.inf)
    accumulated=np.full((len(SENSORS),len(DEPTH)),np.inf)
    accumulated[:,zero]=0.
    prior=np.zeros(unary.shape,dtype='i1')
    columns=np.arange(len(SENSORS))[:,None]
    states=np.arange(len(DEPTH))[None,:]
    for row in range(3,len(base)):
        steps=accumulated[:,:,None]+penalty
        parents=np.argmin(steps,axis=1)
        prior[row]=parents
        accumulated=steps[columns,parents,states]+unary[row]
    positions=np.argmin(accumulated,axis=1)
    result=np.zeros((len(base),len(SENSORS)))
    for row in range(len(base)-1,2,-1):
        result[row]=DEPTH[positions]
        positions=prior[row,np.arange(len(SENSORS)),positions]
    return result,int((~valid).sum())


def half_banks(tracks):
    """Each half is stored on its fine grid in eighths of a voxel, including the entire seam."""
    banks=({}, {})
    centers=(np.zeros(len(tracks)),tracks[:,2],np.median(tracks,axis=1))
    for center in centers:
        for weight in (0,1,2):
            shift=center[:,None]+(weight/2)*(tracks-center[:,None])
            shift[:,2]=center
            grid=np.empty((len(center),9))
            grid[:,::2]=shift
            grid[:,1::2]=(shift[:,1:]+shift[:,:-1])/2
            fixed=np.rint(grid*8).astype('<i2')
            for side,half in enumerate((fixed[:,:5],fixed[:,4:])):
                stamp=half.tobytes()
                if stamp not in banks[side]:
                    banks[side][stamp]={'field':half.copy(),'modes':set()}
                banks[side][stamp]['modes'].add(weight)
    return banks


def oriented(base, proposed):
    if not np.isfinite(proposed).all() or np.any(proposed<0) or np.any(proposed>127):
        return False
    def faces(s):
        a,b,c,d=s[:-1,:-1],s[1:,:-1],s[:-1,1:],s[1:,1:]
        return np.stack((np.cross(b-a,c-a),np.cross(c-d,b-d)))
    ref=faces(base.astype(float));test=faces(proposed.astype(float))
    return bool(np.all(np.sum(ref*test,axis=-1)>1.e-8))


def rank_ct(image,grid,ref,visible):
    n=normals(grid)
    f,valid=features(image,grid[:,VALIDATION],n[:,VALIDATION])
    anchor_diff=distance(f[3:],ref)
    transition=distance(f[3:],f[2:-1])
    loss=np.where(valid[3:]&visible,anchor_diff,3.)
    loss+=.25*np.where(valid[3:]&valid[2:-1],transition,3.)
    if len(loss)==0:return 0.
    per_row=loss.mean(axis=1)
    count=(len(per_row)+2)//3
    return float(.5*(per_row.mean()+np.sort(per_row)[-count:].mean()))

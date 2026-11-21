"""Reference-area measurements from frozen surface coordinates and annotations."""
from itertools import product
import numpy as np

def annotation_near(mask):
    h,w,d=mask.shape
    pad=np.pad(mask.astype(bool),2)
    out=np.zeros(mask.shape,bool)
    for z,y,x in product(range(-2,3),repeat=3):
        if z*z+y*y+x*x<=4:
            out |= pad[z+2:z+2+h,y+2:y+2+w,x+2:x+2+d]
    return out


def classify(points,target,other):
    flat=np.asarray(points).reshape(-1,3)
    inside=((flat>=0)&(flat<=np.array(target.shape)-1)).all(axis=1)
    result=np.zeros(len(flat),np.uint8)
    idx=tuple(np.rint(flat[inside]).astype(int).T)
    result[inside]=target[idx].astype(np.uint8)+2*other[idx].astype(np.uint8)
    return result


def triangles(m):
    m=m.astype(float)
    x,y,z,w=m[:-1,:-1],m[1:,:-1],m[:-1,1:],m[1:,1:]
    return np.stack((np.stack((x,y,z),-2),np.stack((z,y,w),-2)),-3)


def score(mesh,seed,target,other):
    if mesh.shape[1:]!=(9,3) or not 3<=len(mesh)<=33 or not np.array_equal(seed,mesh[:3]):
        raise ValueError('Invalid mesh shape or starting edge.')
    tri=triangles(mesh[2:])
    basis=triangles(seed[-2:])[0]
    fixed=.5*np.linalg.norm(np.cross(basis[:,:,1]-basis[:,:,0],basis[:,:,2]-basis[:,:,0]),axis=-1)
    sample=np.einsum('qv,rcsvd->rcsqd',np.eye(3)*.5+np.ones((3,3))/6,tri)
    groups=classify(sample,target,other).reshape(sample.shape[:-1])
    weight=fixed[None,:,:,None]/3.
    areas=np.stack([np.sum(weight*(groups==i),axis=(1,2,3)) for i in range(4)],axis=1)
    blocks=[areas[i:i+10].sum(axis=0).tolist() for i in (0,10,20)]
    line=classify(mesh[2:],target,other).reshape(-1,9)
    switches=0
    for g in line.T:
        g=g[(g==1)|(g==2)]
        switches+=int(np.any((g[:-1]==1)&(g[1:]==2)))
    return dict(rows=len(mesh)-3,complete=len(mesh)==33,
                goal=sum(b[1] for b in blocks),other=sum(b[2] for b in blocks),
                ambiguous=sum(b[3] for b in blocks),unmarked=sum(b[0] for b in blocks),
                absent=float((33-len(mesh))*fixed.sum()),last=blocks[-1][1],switches=switches,blocks=blocks)


def masks(labels):
    if labels.shape!=(128,128,128) or not int(labels[64,64,64]):
        raise ValueError('No labeled instance at the starting point.')
    ident=labels[64,64,64]
    return annotation_near(labels==ident),annotation_near((labels!=ident)&(labels>0))

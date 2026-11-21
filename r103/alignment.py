import numpy as np


def align_public(points,valid,seed):
    height,width=seed.shape[:2]
    options=[]
    for y,x in np.argwhere(valid & (np.max(np.abs(points-seed[0,0]),axis=-1)<=1e-4)):
        part=points[y:y+height,x:x+width]
        if (part.shape==seed.shape and valid[y:y+height,x:x+width].all()
                and np.max(np.abs(part-seed))<=1e-4):
            options.append((int(y),int(x)))
    if len(options)!=1:
        raise ValueError('Could not uniquely locate the 27-vertex starting patch.')
    y,x=options[0]
    if valid[:y].any() or valid[:,:x].any() or valid[:,x+width:].any():
        raise ValueError('The extension added vertices outside the requested edge.')
    taken=valid[y:,x:x+width]
    last=int(np.flatnonzero(taken.any(axis=1))[-1])+1
    if last>height+10:
        raise ValueError('The extension exceeded ten new rows.')
    return points[y:y+last,x:x+width],taken[:last]

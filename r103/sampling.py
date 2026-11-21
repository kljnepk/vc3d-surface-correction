"""Trilinear sampling and CT normalization."""
import numpy as np


def sample_ct(ct, points):
    lo = np.floor(points).astype(np.intp)
    frac = points - lo
    hi = np.minimum(lo + 1, np.array(ct.shape)-1)
    result = np.zeros(points.shape[:-1])
    for side in np.ndindex(2, 2, 2):
        index = tuple(hi[..., k] if side[k] else lo[..., k] for k in range(3))
        result += ct[index] * np.prod(np.where(side, frac, 1.-frac), axis=-1)
    return result

def image8(raw):
    lo, hi = np.percentile(raw[::3,::3,::3], [2,98])
    if not hi > lo:
        raise ValueError('The CT normalization range is empty.')
    return np.rint(255*np.clip((raw.astype(np.float64)-lo)/(hi-lo), 0., 1.)).astype(np.uint8)

"""Transfer a papyrus CT profile across rows and check that the path returns to its seed."""
import numpy as np

# Distances in native voxels. All rows share an offset across their width.
OFFSETS = np.arange(-3., 3.01, .5)
PROFILE = np.arange(-5., 5.01, 1.)
ZERO = len(OFFSETS) // 2
REGULARITY = .055
STAY = .012
ANCHOR = .30
MAX_STEP = 1.
RETURN_TOL = .5
TRACE_TOL = 1.


def sample_ct(ct, points):
    lo = np.floor(points).astype(np.intp)
    frac = points - lo
    hi = np.minimum(lo + 1, np.array(ct.shape)-1)
    result = np.zeros(points.shape[:-1])
    for side in np.ndindex(2, 2, 2):
        index = tuple(hi[..., k] if side[k] else lo[..., k] for k in range(3))
        result += ct[index] * np.prod(np.where(side, frac, 1.-frac), axis=-1)
    return result


def distances(left, right):
    # Per-column normalized error. Median discounts a damaged side of the ribbon.
    return np.median(np.mean((left - right)**2, axis=-1), axis=-1)


def best_path(terms, start, unary):
    cost = np.full(len(OFFSETS), np.inf)
    cost[start] = 0.
    pred = []
    for i, edge in enumerate(terms):
        scores = cost[:, None] + edge
        parent = np.argmin(scores, axis=0)
        cost = scores[parent, np.arange(len(OFFSETS))] + unary[i]
        pred.append(parent)
    path = [int(np.argmin(cost))]
    for parent in reversed(pred):
        path.append(int(parent[path[-1]]))
    return np.asarray(path[::-1], dtype=int)


def trace_mesh(ct, seed, mesh, valid):
    ct, seed, mesh, valid = map(np.asarray, (ct, seed, mesh, valid))
    if (ct.dtype != np.uint8 or ct.ndim != 3 or seed.shape != (3,9,3) or mesh.ndim != 3
            or mesh.shape[1:] != (9,3) or valid.shape != mesh.shape[:2] or valid.dtype != bool
            or len(mesh) < 3 or not valid[:3].all() or not np.isfinite(mesh[valid]).all()
            or not np.allclose(mesh[:3], seed, atol=1e-4, rtol=0)):
        raise ValueError('Invalid mesh or mismatched starting edge.')
    result = mesh.astype('<f4', copy=True)
    def keep(reason, **details):
        return result, dict(state=reason, changed=False, **details)
    if len(mesh) < 4 or not valid.all():
        return keep('INCOMPLETE_SUPPORT')
    grid = mesh.astype(float)
    n = np.cross(np.gradient(grid, axis=0), np.gradient(grid, axis=1))
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    if np.any(length < 1e-6):
        return keep('DEGENERATE_FRAME')
    n /= length
    if np.any(np.sum(n[1:]*n[:-1], axis=-1) <= 0) or np.any(np.sum(n[:,1:]*n[:,:-1], axis=-1) <= 0):
        return keep('FOLDED_FRAME')
    # Evaluate every displacement and normal-profile sample only once.
    shifts = np.arange(OFFSETS[0]+PROFILE[0], OFFSETS[-1]+PROFILE[-1]+.25, .5)
    points = grid[:,:,None,:] + n[:,:,None,:] * shifts[None,None,:,None]
    if (points < 0).any() or (points > np.array(ct.shape)-1).any():
        return keep('OUTSIDE_CT')
    samples = sample_ct(ct, points)
    idx = np.rint((OFFSETS[:,None] + PROFILE[None,:] - shifts[0]) / .5).astype(int)
    profiles = samples[:,:,idx].transpose(0,2,1,3)
    centered = profiles - profiles.mean(axis=-1, keepdims=True)
    feature = centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 4.**2)
    reference = np.median(feature[:3,ZERO], axis=0)
    f = feature[2:]
    change = OFFSETS[None,:] - OFFSETS[:,None]
    penalty = np.where(np.abs(change) <= MAX_STEP, REGULARITY*change**2, np.inf)
    links = [distances(a[:,None], b[None,:]) + penalty for a,b in zip(f[:-1],f[1:])]
    prior = STAY * OFFSETS**2
    fw = best_path(links, ZERO, [ANCHOR*distances(t,reference) + prior for t in f[1:]])
    # Reverse evidence does not know the starting displacement or the seed template.
    # Prefer small changes on ties, without imposing a return to zero.
    backward = best_path([q.T for q in links[::-1]], fw[-1], [np.zeros_like(prior)]*len(links))[::-1]
    return_error = float(abs(OFFSETS[backward[0]]))
    gap = float(np.max(np.abs(OFFSETS[fw]-OFFSETS[backward])))
    details = {'forward_vox': OFFSETS[fw].tolist(), 'reverse_vox': OFFSETS[backward].tolist(),
               'error_at_seed_vox':return_error, 'max_disagreement_vox':gap}
    if not (fw != ZERO).any():
        return keep('ZERO_SHIFT', **details)
    if return_error > RETURN_TOL or gap > TRACE_TOL:
        return keep('ROUND_TRIP_REJECTED', **details)
    result[3:] = grid[3:] + n[3:] * OFFSETS[fw[1:]][:,None,None]
    return result, dict(state='CORRECTED', changed=True, **details)

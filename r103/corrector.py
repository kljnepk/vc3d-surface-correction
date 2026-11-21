"""Correct a surface strip in a normalized local CT crop using five longitudinal tracks."""
import numpy as np

from .profiles import VALIDATION, anchor, features, fingerprint, five_tracks, half_banks, normals, oriented, rank_ct


def smooth(tracks):
    """Smooth displacements across five tracks, advancing one mesh row at a time."""
    scale = 0.4
    phase = np.zeros(5)
    output = np.zeros_like(tracks)
    for row in range(3, len(tracks)):
        target = tracks[row] * scale
        for _ in range(2):
            neighbour_mean = np.array([
                phase[1],
                (phase[0] + phase[2]) / 2,
                (phase[1] + phase[3]) / 2,
                (phase[2] + phase[4]) / 2,
                phase[3],
            ])
            drift = target - phase
            phase = phase + 0.5 * (drift + neighbour_mean - phase)
        output[row] = np.clip(phase / scale, -4.0, 4.0)
    return output


def validate_grid(grid):
    if (grid.dtype != np.dtype("<f4") or grid.ndim != 3 or grid.shape[1:] != (9, 3)
            or not 3 <= len(grid) <= 33 or not np.isfinite(grid).all()
            or ((grid < 0) | (grid > 127)).any()):
        raise ValueError("Expected a complete 3–33 by 9 grid with local coordinates between 0 and 127.")


def correct(ct, base, r094):
    """Return the selected strip, the best smoothed strip and the two CT fit scores."""
    if ct.dtype != np.uint8 or ct.shape != (128, 128, 128):
        raise ValueError("Expected a normalized uint8 CT crop of shape (128, 128, 128).")
    for mesh in (base, r094):
        validate_grid(mesh)
    if not np.array_equal(base[:3], r094[:3]):
        raise ValueError("Both strips must share the first three rows.")

    directions = normals(base)
    observed, outside_count = five_tracks(ct, base, directions)
    left, right = half_banks(smooth(observed))
    ref, visible = anchor(*features(ct, base[:3, VALIDATION], normals(base[:3])[:, VALIDATION]))
    costs = {}

    def cost(mesh):
        stamp = fingerprint(mesh)
        if stamp not in costs:
            costs[stamp] = rank_ct(ct, mesh, ref, visible)
        return costs[stamp]

    best = base
    best_key = (cost(base), 0.0, fingerprint(base))
    for left_half in left.values():
        left_field = left_half["field"]
        seam = (base[:, 4] + directions[:, 4] * (left_field[:, -1, None] / 8.0)).astype("<f4")
        for right_half in right.values():
            right_field = right_half["field"]
            other_seam = (base[:, 4] + directions[:, 4] * (right_field[:, 0, None] / 8.0)).astype("<f4")
            if not np.array_equal(seam, other_seam):
                continue
            fields = np.column_stack((left_field, right_field[:, 1:]))
            trial = (base + directions * (fields / 8.0)[..., None]).astype("<f4")
            if not np.array_equal(base, trial) and not oriented(base, trial):
                continue
            key = (cost(trial), float(np.abs(fields).sum()), fingerprint(trial))
            if key < best_key:
                best, best_key = trial, key

    use_smoothed = len(best) == len(r094) and cost(best) < cost(r094) - 1.e-12
    return (best if use_smoothed else r094), best, {
        "selected": "smoothed" if use_smoothed else "R094",
        "ct_score_fit": cost(best),
        "ct_score_R094": cost(r094),
        "sensor_outside": outside_count,
        "lengths_match": len(best) == len(r094),
    }

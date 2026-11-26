"""Recover the registered NML path and score fibre proximity in local Z,Y,X."""
from xml.etree import ElementTree

import numpy as np

from r103.scoring import triangles


def registered_tree(path, tree_id, member_id):
    """Order one connected component from its lower-numbered endpoint."""
    root = ElementTree.parse(path).getroot()
    for element in root.iter():
        element.tag = element.tag.rsplit('}', 1)[-1]
    trees = [tree for tree in root.iter('thing') if int(tree.attrib['id']) == tree_id]
    if len(trees) != 1:
        raise ValueError(f'Expected one annotation tree {tree_id} in {path.name}')
    tree = trees[0]
    nodes = {int(node.attrib['id']): [float(node.attrib[k]) for k in ('z', 'y', 'x')]
             for node in tree.findall('./nodes/node')}
    edges = {node: set() for node in nodes}
    for edge in tree.findall('./edges/edge'):
        first, second = (int(edge.attrib[k]) for k in ('source', 'target'))
        edges[first].add(second)
        edges[second].add(first)

    component = set()
    pending = [member_id]
    while pending:
        node = pending.pop()
        if node not in component:
            component.add(node)
            pending.extend(edges[node] - component)
    ends = sorted(node for node in component if len(edges[node]) == 1)
    if min(component) != member_id or len(ends) != 2 or any(len(edges[n]) > 2 for n in component):
        raise ValueError(f'Component {tree_id}/{member_id} is not the registered simple path')

    ordered = []
    previous, node = None, ends[0]
    while True:
        ordered.append(nodes[node])
        successors = edges[node] - {previous}
        if not successors:
            return np.asarray(ordered, dtype=np.float64)
        previous, node = node, next(iter(successors))


def reference_points(path, block, start):
    _, tree, member = start['id'].rsplit('_', 2)
    points = registered_tree(path, int(tree[1:]), int(member[1:]))
    core_shift = block['core_shift']
    points = (points - core_shift) - block['nml_offset_after_core_shift_zyx']
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
    points = points[keep]
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    stations = start['anchor_station'] + np.arange(10, 51)
    if stations[-1] > lengths[-1]:
        raise ValueError(f"Reference ends before the last station for {start['id']}")
    sampled = np.column_stack([np.interp(stations, lengths, axis) for axis in points.T])
    return sampled - (np.asarray(start['crop_origin_zyx']) - core_shift)


def distances(points, mesh):
    """Distances to the new triangles, including their edges and vertices."""
    if len(mesh) == 3:
        return np.full(len(points), np.inf)
    a, b, c = triangles(mesh[2:]).reshape(-1, 3, 3).transpose(1, 0, 2)
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    squared_norm = (normal * normal).sum(axis=1)
    valid = squared_norm > 1e-18
    divisor = np.where(valid, squared_norm, 1.)
    result = []
    for point in points:
        delta = point - a
        u = (np.cross(delta, ac) * normal).sum(axis=1) / divisor
        v = (np.cross(ab, delta) * normal).sum(axis=1) / divisor
        inside = valid & (u >= 0) & (v >= 0) & (u + v <= 1)
        best = np.where(inside, (delta * normal).sum(axis=1)**2 / divisor, np.inf)
        for left, right in ((a, b), (b, c), (c, a)):
            edge = right - left
            t = np.clip(((point - left) * edge).sum(axis=1) /
                        np.maximum(1e-18, (edge * edge).sum(axis=1)), 0, 1)
            best = np.minimum(best, ((point - left - t[:, None] * edge)**2).sum(axis=1))
        result.append(np.sqrt(best.min()))
    return np.asarray(result)


def longest_run(reference, mesh):
    visible = ((reference >= 0) & (reference <= 127)).all(axis=1)
    longest = current = 0
    for supported in visible & (distances(reference, mesh) <= 2.):
        current = current + 1 if supported else 0
        longest = max(longest, current)
    return longest

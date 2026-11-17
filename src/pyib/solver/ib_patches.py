"""Constraint patches for the K-matrix force correction.

Port of IBFast ``tools/make_ib_patches.py``.  The multiplier (surface traction)
of the constrained solve is constant on a patch of markers and no-slip is
enforced for the patch sums.  With one marker per cell and a kernel three cells
wide, neighbouring marker constraints are nearly dependent; patches of about
2 x 2 markers remove that near-null band of K and make it four times smaller.

Rule (as validated for IBFast on the Re = 100 plate): rim markers (open boundary
of the surface, or sharp feature edges of a closed one) keep the full resolution
across the edge and are paired along it; all other markers are grouped into
compact patches of up to ``size`` markers by two pairwise matchings (shortest
mesh edges first, then the two pairs whose union is most compact).  Patches are
defined on the reference surface and stay fixed when the body moves.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _distance_graph(points, factor):
    """Edges between markers closer than factor x (median nearest-neighbour distance)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    nearest = tree.query(points, k=2)[0][:, 1]
    return {(min(u, v), max(u, v)) for u, v in tree.query_pairs(factor * float(np.median(nearest)))}


def _greedy_matching(candidates, free):
    """candidates: (key..., a, b) sorted ascending; returns matched (a, b); ``free`` is updated."""
    out = []
    for candidate in candidates:
        a, b = candidate[-2], candidate[-1]
        if free[a] and free[b]:
            free[a] = free[b] = False
            out.append((a, b))
    return out


def make_patches(points, triangles=None, *, size=4, rim="pair", weights="unit",
                 feature_angle=60.0, corner_angle=45.0, max_diameter=2.3,
                 graph_radius=1.6, rim_offset=0.3, groups=None):
    """Patch id (0-based, contiguous) and weight of every marker.

    points : (M, 3) reference marker positions
    triangles : optional (T, 3) surface triangulation; without it neighbours are
        found by distance and the rim by one-sided neighbourhoods
    size : target markers per interior patch (1, 2 or 4)
    rim : 'pair' (default), 'single' or 'none'
    weights : 'unit' (default) or 'area' (vertex area / median edge length^2; needs triangles)
    groups : optional (M,) integers; patches never cross groups
    Returns ``(patch_ids, patch_weights, info)``.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("points must be an (M, 3) array with M >= 2")
    if size not in (1, 2, 4):
        raise ValueError("size must be 1, 2 or 4")
    if rim not in ("pair", "single", "none"):
        raise ValueError("rim must be 'pair', 'single' or 'none'")
    if weights not in ("unit", "area"):
        raise ValueError("weights must be 'unit' or 'area'")
    count = len(points)
    group_of = np.zeros(count, dtype=int) if groups is None else np.asarray(groups, dtype=int)
    if group_of.shape != (count,):
        raise ValueError("groups: one integer per marker expected")
    edge_triangles = defaultdict(list)
    rim_edges = []
    if triangles is None:
        adjacency = defaultdict(list)
        for g in np.unique(group_of):
            index = np.flatnonzero(group_of == g)
            if len(index) < 2:
                continue
            for u, v in _distance_graph(points[index], graph_radius):
                e = (int(index[min(u, v)]), int(index[max(u, v)]))
                edge_triangles[e] = []
                adjacency[e[0]].append(e[1]); adjacency[e[1]].append(e[0])
        weights = "unit"; area_t = np.zeros(0)
        nearest = np.array([min(np.linalg.norm(points[v] - points[u]) for v in adjacency[u]) if adjacency[u] else 0.0
                            for u in range(count)])
        offset = np.array([np.linalg.norm(points[adjacency[u]].mean(axis=0) - points[u]) if adjacency[u] else 0.0
                           for u in range(count)])
        rim_points = offset > rim_offset * np.median(nearest[nearest > 0])
        if rim != "none":
            rim_edges = [(u, v) for (u, v) in edge_triangles if rim_points[u] and rim_points[v]]
    else:
        triangles = np.asarray(triangles, dtype=int)
        for t, (a, b, c) in enumerate(triangles):
            for u, v in ((a, b), (b, c), (c, a)):
                if group_of[u] == group_of[v]:
                    edge_triangles[(min(u, v), max(u, v))].append(t)
        normal = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                          points[triangles[:, 2]] - points[triangles[:, 0]])
        area_t = 0.5 * np.linalg.norm(normal, axis=1)
        normal /= np.maximum(2.0 * area_t[:, None], 1e-300)
        cos_feature = np.cos(np.radians(feature_angle))
        for (u, v), ts in edge_triangles.items():
            if len(ts) != 2 or abs(normal[ts[0]] @ normal[ts[1]]) < cos_feature:
                rim_edges.append((u, v))                    # boundary, non-manifold or sharp
    if not edge_triangles:
        raise ValueError("no neighbour relation between the markers; cannot form patches")
    is_rim = np.zeros(count, dtype=bool)
    if rim != "none":
        for u, v in rim_edges:
            is_rim[u] = is_rim[v] = True

    def length(u, v):
        return float(np.linalg.norm(points[u] - points[v]))

    reference = np.median([length(u, v) for (u, v) in edge_triangles])

    def rounded(x):
        return round(x / reference, 5)                      # lengths compared up to round-off

    patches = []
    # --- rim: pairs along the rim lines, broken at corners and junctions ---
    if rim == "single":
        patches += [[int(v)] for v in np.flatnonzero(is_rim)]
    elif rim == "pair":
        rim_adjacency = defaultdict(list)
        for u, v in rim_edges:
            rim_adjacency[u].append(v); rim_adjacency[v].append(u)
        cos_corner = np.cos(np.radians(corner_angle))

        def is_break(v):
            if len(rim_adjacency[v]) != 2:
                return True
            a, b = rim_adjacency[v]
            d1 = points[v] - points[a]; d2 = points[b] - points[v]
            return (d1 @ d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)) < cos_corner

        breaks = {v for v in rim_adjacency if is_break(v)}
        used = set(); taken = np.zeros(count, dtype=bool)

        def walk(start, following):
            line = [start]; previous, current = start, following
            while True:
                used.add((min(previous, current), max(previous, current)))
                if current in breaks or current == start:
                    return line
                line.append(current)
                candidates = [w for w in rim_adjacency[current] if w != previous]
                previous, current = current, candidates[0]

        lines = []
        for s in sorted(breaks):
            for w in sorted(rim_adjacency[s]):
                if (min(s, w), max(s, w)) not in used:
                    lines.append(walk(s, w))
        for s in sorted(rim_adjacency):                     # closed smooth loops without a break point
            if s not in breaks and not any((min(s, w), max(s, w)) in used for w in rim_adjacency[s]):
                lines.append(walk(s, sorted(rim_adjacency[s])[0]))
        for line in lines:
            line = [v for v in line if not taken[v]]
            for k in range(0, len(line), 2):
                g = line[k:k + 2]
                patches.append([int(v) for v in g]); taken[g] = True
        for v in np.flatnonzero(is_rim & ~taken):           # safety net
            patches.append([int(v)])
    rim_patches = len(patches)
    # --- interior: two pairwise matchings ---
    interior = ~is_rim
    interior_edges = [(u, v) for (u, v) in edge_triangles if interior[u] and interior[v]]
    if size == 1:
        patches += [[int(v)] for v in np.flatnonzero(interior)]
    else:
        free = interior.copy()
        pairs = _greedy_matching(sorted((rounded(length(u, v)), u, v) for u, v in interior_edges), free)
        first = [[u, v] for u, v in pairs] + [[int(v)] for v in np.flatnonzero(free)]
        if size == 2:
            patches += first
        else:
            owner = np.full(count, -1)
            for k, g in enumerate(first):
                owner[g] = k
            candidates = set()
            for u, v in interior_edges:
                a, b = owner[u], owner[v]
                if a != b:
                    candidates.add((min(a, b), max(a, b)))

            def diameter(a, b):
                p = points[first[a] + first[b]]
                return float(np.max(np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)))

            free_groups = np.ones(len(first), dtype=bool)
            scored = [(rounded(diameter(a, b)), a, b) for a, b in candidates]
            merged = _greedy_matching(sorted(s for s in scored if s[0] <= max_diameter), free_groups)
            patches += [first[a] + first[b] for a, b in merged] + [first[k] for k in np.flatnonzero(free_groups)]
    patch_ids = np.full(count, -1, dtype=np.int64)
    for k, g in enumerate(patches):
        patch_ids[g] = k
    if (patch_ids < 0).any():
        raise RuntimeError("internal error: marker without a patch")
    vertex_area = np.zeros(count)
    if triangles is not None:
        np.add.at(vertex_area, triangles.ravel(), np.repeat(area_t / 3.0, 3))
    patch_weights = np.ones(count) if weights == "unit" else vertex_area / reference ** 2
    members = np.bincount(patch_ids)
    largest = max(float(np.max(np.linalg.norm(points[g][:, None] - points[g][None], axis=2))) for g in patches)
    info = {
        "markers": int(count), "patches": int(len(patches)), "rim_patches": int(rim_patches),
        "rim_markers": int(is_rim.sum()),
        "markers_per_patch": {int(k): int((members == k).sum()) for k in range(1, int(members.max()) + 1)},
        "largest_patch_diameter_spacings": largest / float(reference),
        "weights": weights, "size": int(size), "rim": rim,
        "neighbours": "triangulation" if triangles is not None else "distance graph",
    }
    return patch_ids, patch_weights, info

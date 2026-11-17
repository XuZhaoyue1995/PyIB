"""Two-sided surface-pressure correction for triangulated immersed surfaces.

Port of the two-sided normal-force LFC repair of IBFast
(``tools/pressure_lfc.py --two-sided``, 2026-11).  The sphere-only path of
``ib_postprocess.process_snapshot`` is unchanged; this module adds what that
path assumes away:

* the surface is a triangulation of the markers, open (zero-thickness plates,
  wings) or closed, instead of an analytic sphere;
* both sides are repaired.  A lattice face is dropped from the band system when
  the segment between its two cell centres meets a triangle (geometry), not when
  the signed distance changes sign: beyond the rim of an open surface both
  "sides" are the same fluid and such a face does not cross the surface;
* triangles are given one consistent winding by walking across shared edges;
  closed components are turned outward, open ones keep the winding of their
  first triangle.  '+' is the side the normals point to;
* the pressure zero can be placed far from the body (``far_gauge``), so that the
  level on each side is comparable between separate surfaces and between runs.

NumPy/SciPy only; no solver import and no CUDA device are needed.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg, lsqr
from scipy.spatial import ConvexHull, cKDTree

KERNELS = ("h3", "h3s", "h4")


def delta_kernel(distance, spacing=1.0, kind="h3"):
    """One-dimensional regularised delta of the solver, in inverse-length units.

    h3: Roma-Peskin three-point; h3s: its cell average (Yang, Zhang, Li & He,
    JCP 228 (2009) 7821, eq. 20; support 2h); h4: Peskin four-point.
    """
    if kind not in KERNELS:
        raise ValueError(f"unknown delta kernel {kind!r}")
    q = np.abs(np.asarray(distance, dtype=float)) / spacing
    out = np.zeros_like(q)
    if kind == "h4":
        m = q <= 1.0
        out[m] = (3 - 2 * q[m] + np.sqrt(1 + 4 * q[m] - 4 * q[m] ** 2)) / 8
        m = (q > 1.0) & (q <= 2.0)
        out[m] = (5 - 2 * q[m] - np.sqrt(np.maximum(-7 + 12 * q[m] - 4 * q[m] ** 2, 0))) / 8
    elif kind == "h3s":
        s3 = np.sqrt(3.0)
        m = q <= 1.0
        x = q[m]
        out[m] = (17 / 48 + s3 * np.pi / 108 + x / 4 - x * x / 4
                  + (1 - 2 * x) / 16 * np.sqrt(np.maximum(-12 * x * x + 12 * x + 1, 0))
                  - s3 / 12 * np.arcsin(np.clip(s3 / 2 * (2 * x - 1), -1, 1)))
        m = (q > 1.0) & (q <= 2.0)
        x = q[m]
        out[m] = (55 / 48 - s3 * np.pi / 108 - 13 * x / 12 + x * x / 4
                  + (2 * x - 3) / 48 * np.sqrt(np.maximum(-12 * x * x + 36 * x - 23, 0))
                  + s3 / 36 * np.arcsin(np.clip(s3 / 2 * (2 * x - 3), -1, 1)))
    else:
        m = q <= 0.5
        out[m] = (1 + np.sqrt(np.maximum(1 - 3 * q[m] ** 2, 0))) / 3
        m = (q > 0.5) & (q <= 1.5)
        out[m] = (5 - 3 * q[m] - np.sqrt(np.maximum(1 - 3 * (1 - q[m]) ** 2, 0))) / 6
    return out / spacing


def spread_marker_force(lattice, markers, marker_force, marker_volume_weight, kind="h3"):
    """Spread marker force density to the lattice with the solver's kernel and weights."""
    markers = np.asarray(markers, dtype=float)
    force = np.asarray(marker_force, dtype=float)
    weights = np.asarray(marker_volume_weight, dtype=float)
    shape = np.asarray(lattice.shape)
    nearest = np.rint((markers - lattice.origin) / lattice.spacing).astype(int)
    field = np.zeros(lattice.shape + (3,))
    for offset in np.ndindex(5, 5, 5):
        indices = nearest + np.asarray(offset) - 2
        locations = lattice.origin + lattice.spacing * indices
        delta = np.prod(delta_kernel(locations - markers, lattice.spacing, kind), axis=1)
        active = delta > 0
        valid = np.all((indices >= 0) & (indices < shape), axis=1)
        if np.any(active & ~valid):
            raise ValueError("marker force support leaves the saved lattice")
        selected = indices[active]
        if np.any(lattice.cell_ids[selected[:, 0], selected[:, 1], selected[:, 2]] < 0):
            raise ValueError("marker force support encounters missing fine cells")
        np.add.at(field, tuple(selected.T), force[active] * (delta[active] * weights[active])[:, None])
    return field


# --------------------------------------------------------------------------- surface geometry
def orient_triangles(points, triangles):
    """One consistent winding per connected component.

    Returns ``(triangles, closed)``; ``closed`` means every edge is shared by
    exactly two triangles.  Closed components are oriented outward (positive
    volume); open ones keep the winding of their first triangle.
    """
    points = np.asarray(points, dtype=float)
    tri = np.array(triangles, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3 or len(tri) == 0:
        raise ValueError("triangles must be a non-empty integer (count, 3) array")
    if tri.min() < 0 or tri.max() >= len(points):
        raise ValueError("triangle vertex ids exceed the marker count")
    edges = {}
    for t, (a, b, c) in enumerate(tri):
        for u, v in ((a, b), (b, c), (c, a)):
            edges.setdefault((min(u, v), max(u, v)), []).append((t, u < v))
    closed = all(len(v) == 2 for v in edges.values())
    neighbours = [[] for _ in range(len(tri))]
    for v in edges.values():
        if len(v) == 2:
            (t1, d1), (t2, d2) = v
            # the same direction along the shared edge means opposite winding
            neighbours[t1].append((t2, d1 == d2)); neighbours[t2].append((t1, d1 == d2))
        elif len(v) > 2:
            raise ValueError(f"non-manifold surface: an edge is shared by {len(v)} triangles")
    seen = np.zeros(len(tri), bool); flip = np.zeros(len(tri), bool)
    for start in range(len(tri)):
        if seen[start]:
            continue
        component = [start]; seen[start] = True; stack = [start]
        while stack:
            t = stack.pop()
            for t2, opposite in neighbours[t]:
                if not seen[t2]:
                    flip[t2] = flip[t] ^ opposite
                    seen[t2] = True; component.append(t2); stack.append(t2)
        component = np.asarray(component)
        flipped = component[flip[component]]
        tri[flipped] = tri[flipped][:, [0, 2, 1]]
        if closed:
            a, b, c = points[tri[component, 0]], points[tri[component, 1]], points[tri[component, 2]]
            centre = points[np.unique(tri[component])].mean(axis=0)
            if np.sum((a - centre) * np.cross(b - centre, c - centre)) < 0:
                tri[component] = tri[component][:, [0, 2, 1]]
    return tri, closed


def oriented_normals(points, triangles):
    """Unit triangle normals, area-weighted unit marker normals, lumped marker areas."""
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    doubled = np.cross(b - a, c - a)                      # 2 * area * unit normal
    area_t = 0.5 * np.linalg.norm(doubled, axis=1)
    if np.any(area_t <= 0):
        raise ValueError("degenerate triangle in the surface")
    normals = np.zeros_like(points, dtype=float); area = np.zeros(len(points))
    for k in range(3):
        np.add.at(normals, triangles[:, k], doubled)
        np.add.at(area, triangles[:, k], area_t / 3.0)
    length = np.linalg.norm(normals, axis=1)
    if np.any(length == 0):
        raise ValueError("a marker belongs to no triangle (or its normals cancel)")
    return doubled / (2 * area_t)[:, None], normals / length[:, None], area


def _point_triangle_distance(point, a, b, c):
    """Distance from one point to triangles (a, b, c) and the closest points (Ericson)."""
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = np.sum(ab * ap, 1), np.sum(ac * ap, 1)
    bp = point - b
    d3, d4 = np.sum(ab * bp, 1), np.sum(ac * bp, 1)
    cp = point - c
    d5, d6 = np.sum(ab * cp, 1), np.sum(ac * cp, 1)
    closest = np.empty_like(a)
    done = np.zeros(len(a), bool)
    m = (d1 <= 0) & (d2 <= 0); closest[m] = a[m]; done |= m
    m = ~done & (d3 >= 0) & (d4 <= d3); closest[m] = b[m]; done |= m
    m = ~done & (d6 >= 0) & (d5 <= d6); closest[m] = c[m]; done |= m
    vc = d1 * d4 - d3 * d2
    m = ~done & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v = np.where(d1 - d3 != 0, d1 / np.where(d1 - d3 != 0, d1 - d3, 1), 0)
    closest[m] = a[m] + v[m, None] * ab[m]; done |= m
    vb = d5 * d2 - d1 * d6
    m = ~done & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w = np.where(d2 - d6 != 0, d2 / np.where(d2 - d6 != 0, d2 - d6, 1), 0)
    closest[m] = a[m] + w[m, None] * ac[m]; done |= m
    va = d3 * d6 - d5 * d4
    m = ~done & (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    den = (d4 - d3) + (d5 - d6)
    w2 = np.where(den != 0, (d4 - d3) / np.where(den != 0, den, 1), 0)
    closest[m] = b[m] + w2[m, None] * (c[m] - b[m]); done |= m
    m = ~done
    denom = va + vb + vc
    denom = np.where(denom != 0, denom, 1)
    v, w = vb / denom, vc / denom
    closest[m] = a[m] + v[m, None] * ab[m] + w[m, None] * ac[m]
    return np.linalg.norm(point - closest, axis=1), closest


def surface_distance(lattice, points, triangles, triangle_normals, reach=6.0):
    """Signed distance of the lattice cells within ``reach`` spacings of the markers.

    Positive on the side the normal of the nearest triangle points to; NaN
    farther away.  Returns ``(distance, coordinates)`` of lattice shape.
    """
    coordinates = lattice.coordinates()
    flat = coordinates.reshape(-1, 3)
    tree = cKDTree(points)
    nearest_marker, _ = tree.query(flat, k=1)
    selected = np.flatnonzero(nearest_marker <= reach * lattice.spacing)
    incident = [[] for _ in range(len(points))]
    for t, vertices in enumerate(triangles):
        for v in vertices:
            incident[v].append(t)
    _, near = tree.query(flat[selected], k=min(8, len(points)))
    near = np.atleast_2d(near)
    distance = np.full(len(flat), np.nan)
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    for q, cell in enumerate(selected):
        candidates = np.unique(np.concatenate([incident[m] for m in near[q]]))
        d, closest = _point_triangle_distance(flat[cell], a[candidates], b[candidates], c[candidates])
        best = np.argmin(d)
        sign = np.sign(np.dot(flat[cell] - closest[best], triangle_normals[candidates[best]]))
        distance[cell] = d[best] * (sign if sign != 0 else 1.0)
    return distance.reshape(lattice.shape), coordinates


def cut_faces(lattice, points, triangles):
    """Lattice faces crossed by the surface.

    Three boolean arrays ``cut[axis]`` of lattice shape, True at cell (i, j, k)
    when the segment from that cell centre to the next one along ``axis`` meets
    a triangle.  Lines through a shared triangle edge are caught by the inclusive
    barycentric test, so the cut set has no leaks.
    """
    shape = np.asarray(lattice.shape)
    cut = [np.zeros(lattice.shape, bool) for _ in range(3)]
    vertices = (points[triangles] - lattice.origin) / lattice.spacing     # lattice units
    tol = 1e-9
    for axis in range(3):
        b, c = [d for d in range(3) if d != axis]
        for tri in vertices:
            lo = np.ceil(tri[:, [b, c]].min(axis=0) - tol).astype(int)
            hi = np.floor(tri[:, [b, c]].max(axis=0) + tol).astype(int)
            lo = np.maximum(lo, 0); hi = np.minimum(hi, [shape[b] - 1, shape[c] - 1])
            if np.any(hi < lo):
                continue
            (x1, y1), (x2, y2), (x3, y3) = tri[0, [b, c]], tri[1, [b, c]], tri[2, [b, c]]
            det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if abs(det) < 1e-14:
                continue                                           # triangle parallel to the axis
            jj, kk = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
            l1 = ((y2 - y3) * (jj - x3) + (x3 - x2) * (kk - y3)) / det
            l2 = ((y3 - y1) * (jj - x3) + (x1 - x3) * (kk - y3)) / det
            l3 = 1.0 - l1 - l2
            inside = (l1 >= -tol) & (l2 >= -tol) & (l3 >= -tol)
            if not inside.any():
                continue
            along = l1[inside] * tri[0, axis] + l2[inside] * tri[1, axis] + l3[inside] * tri[2, axis]
            i0 = np.floor(along).astype(int)
            ok = (i0 >= 0) & (i0 < shape[axis] - 1)
            index = [None, None, None]
            index[axis] = i0[ok]; index[b] = jj[inside][ok]; index[c] = kk[inside][ok]
            cut[axis][tuple(index)] = True
    return cut


# --------------------------------------------------------------------------- lattice pressure
def _lattice_faces(shape, valid):
    ids = np.arange(int(np.prod(shape))).reshape(shape)
    first, second, axes = [], [], []
    flat = valid.ravel()
    for axis in range(3):
        lower, upper = [slice(None)] * 3, [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        a, b = ids[tuple(lower)].ravel(), ids[tuple(upper)].ravel()
        keep = flat[a] & flat[b]
        first.append(a[keep]); second.append(b[keep])
        axes.append(np.full(np.count_nonzero(keep), axis, dtype=int))
    return np.concatenate(first), np.concatenate(second), np.concatenate(axes)


def lattice_raw_pressure(lattice, gradient, cells, pin_score, *, tolerance=1e-10):
    """Least-squares projection of a lattice gradient on the cells of ``cells``.

    The IBFast local recovery: jumps h/2 (g(c1) + g(c2)) on the lattice faces
    inside ``cells``.  The cells may fall into several disconnected pieces
    (separate surfaces farther apart than twice the reach); each piece has its
    own constant, fixed by a zero at its cell with the largest ``pin_score``
    (lattice-shaped array, for example the distance from the surface).  Cells
    outside the mask return NaN.  Returns ``(pressure, pieces)``.
    """
    first, second, axes = _lattice_faces(lattice.shape, cells)
    g = gradient.reshape(-1, 3)
    jumps = 0.5 * lattice.spacing * (g[first, axes] + g[second, axes])
    flat = np.flatnonzero(cells.ravel())
    if len(flat) < 2:
        raise ValueError("lattice pressure recovery needs at least two cells")
    local = np.full(cells.size, -1, dtype=np.int64); local[flat] = np.arange(len(flat))
    count = len(first)
    incidence = coo_matrix(
        (np.r_[-np.ones(count), np.ones(count)],
         (np.r_[np.arange(count), np.arange(count)], np.r_[local[first], local[second]])),
        shape=(count, len(flat))).tocsr()
    matrix = (incidence.T @ incidence).tocsr()
    components, labels = connected_components(matrix, directed=False)
    score = np.asarray(pin_score, dtype=float).ravel()[flat]
    order = np.lexsort((-score, labels))                   # by piece, largest score first
    pinned = order[np.r_[True, labels[order][1:] != labels[order][:-1]]]
    free = np.setdiff1d(np.arange(len(flat)), pinned)      # exact elimination of the pinned cells
    rhs = np.asarray(incidence.T @ jumps)
    solution = np.zeros(len(flat))
    if len(free):
        reduced = matrix[free][:, free].tocsr()
        solution[free], status = cg(reduced, rhs[free], M=diags(1.0 / reduced.diagonal()),
                                    rtol=tolerance, atol=0.0, maxiter=max(5000, 10 * len(free)))
        if status != 0 or not np.isfinite(solution).all():
            raise RuntimeError(f"lattice pressure solve failed to converge (status {status})")
    out = np.full(cells.size, np.nan)
    out[flat] = solution
    return out.reshape(lattice.shape), int(components)


def repair_pressure_band_cut(lattice, raw_pressure, regular_gradient, distance, cut, *, width=1.5,
                             tolerance=1e-11):
    """Anchored repair of the band |distance| <= width h on both sides of the surface.

    Least squares of the face jumps of the regular gradient on the lattice faces
    the surface does not cross; raw pressure outside the band as Dirichlet
    anchors.  Band cells with no path to an anchor keep the raw value.
    Returns ``(repaired, band, unrepaired, lsqr_iterations)``.
    """
    spacing = lattice.spacing
    valid = np.isfinite(raw_pressure)
    with np.errstate(invalid="ignore"):
        target = valid & (np.abs(distance) <= width * spacing)
    first, second, axes = _lattice_faces(lattice.shape, valid)
    crossed = np.zeros(len(first), bool)
    for axis in range(3):
        m = axes == axis
        crossed[m] = cut[axis].ravel()[first[m]]
    flat_target = target.ravel()
    keep = (flat_target[first] | flat_target[second]) & ~crossed
    first, second, axes = first[keep], second[keep], axes[keep]
    g = regular_gradient.reshape(-1, 3)
    rhs = 0.5 * spacing * (g[first, axes] + g[second, axes])
    target_ids = np.flatnonzero(flat_target)
    if not len(target_ids):
        raise ValueError("pressure repair band contains no cells; refine the grid")
    local = np.full(target.size, -1, dtype=np.int64); local[target_ids] = np.arange(len(target_ids))
    raw = raw_pressure.ravel()
    l1, l2 = local[first], local[second]
    both = (l1 >= 0) & (l2 >= 0)
    adjacency = coo_matrix((np.ones(int(both.sum())), (l1[both], l2[both])),
                           shape=(len(target_ids), len(target_ids)))
    components, labels = connected_components(adjacency, directed=False)
    anchored = np.zeros(components, bool)
    anchored[labels[l1[(l1 >= 0) & (l2 < 0)]]] = True
    anchored[labels[l2[(l2 >= 0) & (l1 < 0)]]] = True
    free = ~anchored[labels]                               # band cells that cannot be repaired
    rows, columns, entries = [], [], []
    for sign, cells, loc in ((-1.0, first, l1), (1.0, second, l2)):
        inside = loc >= 0
        rows.append(np.flatnonzero(inside)); columns.append(loc[inside])
        entries.append(np.full(int(inside.sum()), sign))
        rhs[~inside] -= sign * raw[cells[~inside]]
    incidence = coo_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(columns))),
                           shape=(len(first), len(target_ids))).tocsr()
    use = ~free
    solved = lsqr(incidence[:, use], rhs - incidence[:, free] @ raw[target_ids[free]],
                  atol=tolerance, btol=tolerance, iter_lim=max(500, 20 * len(target_ids)))
    if solved[1] not in (0, 1, 2, 4, 5) or not np.isfinite(solved[0]).all():
        raise RuntimeError(f"two-sided pressure band solve failed (status {solved[1]})")
    repaired = raw.copy()
    repaired[target_ids[use]] = solved[0]
    unrepaired = np.zeros(target.size, bool); unrepaired[target_ids[free]] = True
    return (repaired.reshape(lattice.shape), target, unrepaired.reshape(lattice.shape),
            int(solved[2]))


def surface_fit_side(coordinates, side_distance, pressure, markers, spacing, neighbours=40,
                     report=None):
    """One-sided inverse-distance weighted quadratic extrapolation to the markers.

    ``side_distance`` is +distance for the '+' side and -distance for the '-'
    side; only cells farther than 0.15 h on that side are used.  Where the
    stencil does not determine all ten coefficients (markers at the rim of an
    open surface) the minimum-norm least-squares solution is used, as in
    IBFast; their number is added to ``report['rank_deficient']``.
    """
    with np.errstate(invalid="ignore"):
        side = (side_distance > 0.15 * spacing) & np.isfinite(pressure)
    points, values, wall = coordinates[side], pressure[side], side_distance[side]
    if len(points) < 10:
        raise ValueError("too few cells on one side of the surface for the quadratic fit")
    maximum = min(4 * neighbours, len(points))
    distance, indices = cKDTree(points).query(markers, k=maximum)
    result = np.empty(len(markers))
    deficient = 0
    for marker in range(len(markers)):
        count = min(neighbours, maximum)
        while count < maximum and np.ptp(wall[indices[marker, :count]]) < 1.9 * spacing:
            count = min(2 * count, maximum)
        chosen, dist = indices[marker, :count], distance[marker, :count]
        x, y, z = ((points[chosen] - markers[marker]) / spacing).T
        design = np.column_stack((np.ones(len(x)), x, y, z, x * x, y * y, z * z, x * y, x * z, y * z))
        weight = 1.0 / np.maximum(dist / spacing, 0.35)
        coefficient, _, rank, _ = np.linalg.lstsq(design * weight[:, None], values[chosen] * weight, rcond=None)
        deficient += int(rank < 10)
        result[marker] = coefficient[0]
    if report is not None:
        report["rank_deficient"] = report.get("rank_deficient", 0) + deficient
    return result


# --------------------------------------------------------------------------- driver
def two_sided_surface_pressure(lattice, gradient, markers, triangles, marker_force,
                               marker_volume_weight, *, raw_pressure=None, pressure_width=1.5,
                               reach=6.0, far_gauge=None, kernel="h3", verbose_checks=None):
    """Pressure on both sides of every marker of a triangulated immersed surface.

    Parameters
    ----------
    lattice : ib_postprocess.UniformLattice
    gradient : (nx, ny, nz, 3) lattice array of momentum_residual / dt (NaN where no fine cell)
    markers, triangles : surface; triangles index the markers
    marker_force : (M, 3) force density on the fluid held by the solver (already rho-weighted)
    marker_volume_weight : (M,) spread weight ds^3
    raw_pressure : optional (nx, ny, nz) lattice pressure recovered elsewhere (for example the
        whole-mesh projection of ib_postprocess.recover_pressure).  If None the pressure is
        recovered on the lattice as in IBFast: on the cells within ``reach`` of the surface with
        the zero at the cell farthest from it, or, with ``far_gauge``, on every lattice cell.
    far_gauge : distance D (length units).  The mean of the raw pressure over the lattice cells
        farther than D from all markers is set to zero.

    Returns a dict of arrays and JSON-safe diagnostics under ``"diagnostics"``.
    """
    markers = np.asarray(markers, dtype=float)
    force = np.asarray(marker_force, dtype=float)
    weights = np.asarray(marker_volume_weight, dtype=float)
    if markers.ndim != 2 or markers.shape[1] != 3 or force.shape != markers.shape or weights.shape != (len(markers),):
        raise ValueError("markers, marker_force and marker_volume_weight have inconsistent shapes")
    if not np.isfinite(pressure_width) or pressure_width <= 0 or not reach > pressure_width + 2.5:
        raise ValueError("need pressure_width > 0 and reach > pressure_width + 2.5 spacings")
    spacing = lattice.spacing
    triangles, closed = orient_triangles(markers, triangles)
    triangle_normals, normals, area = oriented_normals(markers, triangles)
    distance, coordinates = surface_distance(lattice, markers, triangles, triangle_normals, reach)
    have = np.isfinite(gradient[..., 0])
    flat_coordinates = coordinates.reshape(-1, 3)
    gauge = None
    recovered_here = raw_pressure is None
    if recovered_here:
        if far_gauge is None:
            work = np.isfinite(distance) & have
            score = np.where(work, np.abs(distance), -np.inf)
        else:
            work = have
            score = np.full(have.size, -np.inf)
            flat = np.flatnonzero(work.ravel())
            score[flat] = cKDTree(markers).query(flat_coordinates[flat], k=1)[0]
        raw_pressure, pieces = lattice_raw_pressure(lattice, gradient, work, score)
    else:
        raw_pressure = np.array(raw_pressure, dtype=float)
        if raw_pressure.shape != lattice.shape:
            raise ValueError("raw_pressure must have the lattice shape")
        work = np.isfinite(raw_pressure) & have
        pieces = 1
    if far_gauge is not None:
        flat = np.flatnonzero(np.isfinite(raw_pressure).ravel())
        marker_distance = cKDTree(markers).query(flat_coordinates[flat], k=1)[0]
        far = flat[marker_distance >= far_gauge]
        if not len(far):
            raise ValueError(
                f"far_gauge={far_gauge:g}: no lattice cell that far from the markers "
                f"(largest distance {marker_distance.max():g})"
            )
        values = raw_pressure.ravel()[far]
        gauge = {"cells": int(len(far)), "rms_about_mean": float(np.std(values)),
                 "shift_removed": float(values.mean()), "distance": float(far_gauge)}
        raw_pressure = raw_pressure - values.mean()
    normal_force = np.sum(force * normals, axis=1)[:, None] * normals
    source = spread_marker_force(lattice, markers, normal_force, weights, kernel)
    cut = cut_faces(lattice, markers, triangles)
    regular = np.where(have[..., None], gradient, 0.0) - source
    repaired, band, unrepaired, iterations = repair_pressure_band_cut(
        lattice, raw_pressure, regular, distance, cut, width=pressure_width)
    flat_distance = distance.ravel()
    out = {
        "markers": markers, "triangles": triangles, "normals": normals, "marker_area": area,
        "closed": bool(closed), "distance": distance, "pressure_raw_lattice": raw_pressure,
        "pressure_lfc_lattice": repaired, "pressure_band": band, "unrepaired": unrepaired,
        "cut": cut,
    }
    fit_report = {}
    for name, sign in (("plus", 1.0), ("minus", -1.0)):
        out[f"surface_pressure_{name}_raw"] = surface_fit_side(
            flat_coordinates, sign * flat_distance, raw_pressure.ravel(), markers, spacing)
        out[f"surface_pressure_{name}"] = surface_fit_side(
            flat_coordinates, sign * flat_distance, repaired.ravel(), markers, spacing,
            report=fit_report)
    out["pressure_force"] = pressure_force(out)
    out["pressure_force_raw"] = pressure_force(out, raw=True)
    weighted_normal = np.sum(force * normals, axis=1) * weights
    out["marker_normal_force"] = -np.sum(weighted_normal[:, None] * normals, axis=0)
    out["diagnostics"] = {
        "method": "IBFast two-sided normal-force LFC band repair (geometric cut faces)",
        "surface": "closed" if closed else "open",
        "markers": int(len(markers)), "triangles": int(len(triangles)), "kernel": kernel,
        "pressure_band_width_h": float(pressure_width), "reach_h": float(reach),
        "lattice_cells_used": int(work.sum()), "pressure_band_cells": int(band.sum()),
        "pressure_band_cells_left_raw": int(unrepaired.sum()),
        "faces_cut": int(sum(int(c.sum()) for c in cut)),
        "pressure_band_iterations": int(iterations),
        "surface_fits_rank_deficient": int(fit_report.get("rank_deficient", 0)),
        "raw_pressure": ("lattice recovery, zero at the farthest cell" if recovered_here and far_gauge is None
                         else "lattice recovery on every lattice cell" if recovered_here
                         else "supplied (whole-mesh projection)"),
        "raw_pressure_pieces": int(pieces),
        "far_gauge": gauge,
        "pressure_force": out["pressure_force"].tolist(),
        "pressure_force_raw": out["pressure_force_raw"].tolist(),
        "marker_normal_force": out["marker_normal_force"].tolist(),
        "pressure_force_definition": (
            "- sum p_plus n A (closed surface, outer side)" if closed
            else "- sum (p_plus - p_minus) n A (open surface)"),
        "normal_area_closure_relative_defect": float(
            np.linalg.norm(np.sum(normals * area[:, None], axis=0)) / np.sum(area)),
    }
    return out


def pressure_force(result, raw=False):
    """Force of the pressure on the body: closed surface, outer side only; open, the jump."""
    suffix = "_raw" if raw else ""
    plus = result["surface_pressure_plus" + suffix]
    minus = result["surface_pressure_minus" + suffix]
    load = plus if result["closed"] else plus - minus
    return -np.sum((load * result["marker_area"])[:, None] * result["normals"], axis=0)


def sphere_triangulation(markers):
    """Closed triangulation of a convex marker cloud (the canonical sphere cases)."""
    return ConvexHull(np.asarray(markers, dtype=float)).simplices.astype(np.int64)

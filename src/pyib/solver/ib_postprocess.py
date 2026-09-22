"""Offline pressure and surface stress from canonical CPU/CUDA field snapshots.

The curl-curl solver eliminates pressure.  A saved final-inner momentum residual
provides its gradient, ``momentum_residual / dt``.  The whole-mesh pressure is
the area/distance weighted graph projection used by IBFast pressure_field.py.
Surface pressure then uses its normal-force LFC, same-phase anchored band
repair.  Shear uses the tangential-force Shortley-Weller band repair and the
wall-pinned quadratic extraction in IBFast wall_shear_band.py.

Only the canonical closed, rigid spheres and their Roma-Peskin h3 kernel are
supported.  Analytic sphere distances replace IBFast's triangulated distance.
The velocity repair is a leading-order thin-band Stokes approximation, not an
exact reconstruction for arbitrary Reynolds number or coarse grids.  Raw and
repaired values are retained separately; stress-integral closure is reported,
never imposed.  Pressure has a boundary-adjacent zero-mean gauge.

All expensive recovery is offline on NumPy/SciPy, even for CUDA snapshots::

    python ib_postprocess.py run/final_field.npz
    python ib_postprocess.py run/field_00001000.npz --field-only

No solver imports, model credentials or CUDA device are required.  Old fields
without a retained momentum residual cannot provide this pressure recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import uuid
from xml.etree.ElementTree import Element, SubElement, ElementTree

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg, lsqr, splu
from scipy.spatial import cKDTree


def _finite(name, values, shape=None):
    values = np.asarray(values, dtype=np.float64)
    if shape is not None and values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def recover_pressure(cell_positions, face_cells, face_area, gradient, *, tolerance=1e-10):
    """Project cell pressure gradients on the complete, connected cell graph.

    Internal face jump: p2-p1 = (g1+g2)/2 dot (x2-x1), weighted by A/|x2-x1|.
    Boundary faces have second cell -1.  Exact elimination pins one cell during
    the solve, then the boundary-adjacent mean is removed.  No penalty gauge is
    used, avoiding the severe conditioning penalty of a large diagonal.
    """
    positions = _finite("cell_positions", cell_positions)
    count = len(positions)
    if positions.shape != (count, 3) or count < 2:
        raise ValueError("pressure recovery needs at least two 3D cells")
    gradient = _finite("gradient", gradient, positions.shape)
    pairs = np.asarray(face_cells)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise ValueError("face_cells must be an integer (number_of_faces, 2) array")
    if np.any(pairs[:, 0] < 0) or np.any(pairs >= count) or np.any(pairs[:, 1] < -1):
        raise ValueError("face_cells contains invalid cell ids")
    areas = _finite("face_area", face_area, (len(pairs),))
    if np.any(areas <= 0):
        raise ValueError("face areas must be positive")
    internal = pairs[:, 1] >= 0
    first, second = pairs[internal].T
    displacement = positions[second] - positions[first]
    distance = np.linalg.norm(displacement, axis=1)
    if not len(distance) or np.any(distance <= 0):
        raise ValueError("pressure graph has no valid internal faces")
    row = np.arange(len(first))
    incidence = coo_matrix(
        (np.r_[-np.ones(len(row)), np.ones(len(row))],
         (np.r_[row, row], np.r_[first, second])),
        shape=(len(row), count),
    ).tocsr()
    weights = areas[internal] / distance
    matrix = (incidence.T @ diags(weights) @ incidence).tocsr()
    if connected_components(matrix, directed=False, return_labels=False) != 1:
        raise ValueError("pressure graph is disconnected; a unique gauge is unavailable")
    jumps = 0.5 * np.sum((gradient[first] + gradient[second]) * displacement, axis=1)
    rhs = np.asarray(incidence.T @ (weights * jumps))
    reduced = matrix[1:, 1:]
    pressure = np.zeros(count)
    pressure[1:], status = cg(
        reduced, rhs[1:], M=diags(1.0 / reduced.diagonal()),
        rtol=tolerance, atol=0.0, maxiter=max(2000, 10 * count),
    )
    if status != 0 or not np.isfinite(pressure).all():
        raise RuntimeError(f"pressure graph solve failed to converge (status {status})")
    gauge_cells = np.unique(pairs[~internal, 0])
    if not len(gauge_cells):
        gauge_cells = np.arange(count)
    pressure -= np.mean(pressure[gauge_cells])
    mismatch = incidence @ pressure - jumps
    diagnostics = {
        "gauge": "mean boundary-adjacent cells = 0" if np.any(~internal) else "mean cells = 0",
        "gradient_projection_weighted_rms": float(np.sqrt(np.sum(weights * mismatch**2) / weights.sum())),
    }
    return pressure, diagnostics


@dataclass
class UniformLattice:
    """Finest uniform block with global cell ids and strict trilinear sampling."""

    cell_ids: np.ndarray
    origin: np.ndarray
    spacing: float

    def __post_init__(self):
        self.cell_ids = np.asarray(self.cell_ids)
        self.origin = _finite("lattice_origin", self.origin, (3,))
        self.spacing = float(self.spacing)
        if self.cell_ids.ndim != 3 or self.cell_ids.dtype.kind not in "iu":
            raise ValueError("index_cell must be a 3D integer array")
        if min(self.cell_ids.shape) < 2 or not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("the lattice needs at least two cells per axis and positive spacing")

    @property
    def shape(self):
        return self.cell_ids.shape

    def coordinates(self):
        indices = np.indices(self.shape).transpose(1, 2, 3, 0)
        return self.origin + self.spacing * indices

    def field(self, cell_field):
        cell_field = np.asarray(cell_field)
        valid = self.cell_ids >= 0
        if np.any(self.cell_ids[valid] >= len(cell_field)):
            raise ValueError("index_cell exceeds the saved field size")
        result = np.full(self.shape + cell_field.shape[1:], np.nan)
        result[valid] = cell_field[self.cell_ids[valid]]
        return result

    def interpolate(self, field, points):
        points = _finite("probe points", points)
        relative = (points - self.origin) / self.spacing
        lower = np.floor(relative).astype(int)
        if np.any(lower < 0) or np.any(lower + 1 >= np.asarray(self.shape)):
            raise ValueError("surface probes leave the finest uniform block; enlarge the refined region")
        fraction = relative - lower
        values = np.zeros((len(points),) + field.shape[3:])
        for offset in np.ndindex(2, 2, 2):
            offset = np.asarray(offset)
            idx = lower + offset
            weight = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
            weight = weight.reshape((len(points),) + (1,) * (field.ndim - 3))
            values += weight * field[idx[:, 0], idx[:, 1], idx[:, 2]]
        if not np.isfinite(values).all():
            raise ValueError("surface probes encounter missing fine cells; enlarge the refined region")
        return values


def delta_h3(distance, spacing=1.0):
    """Roma-Peskin three-point regularized delta, in inverse-length units."""
    q = np.abs(np.asarray(distance)) / spacing
    result = np.zeros_like(q, dtype=float)
    inner = q <= 0.5
    outer = (q > 0.5) & (q <= 1.5)
    result[inner] = (1 + np.sqrt(np.maximum(1 - 3 * q[inner]**2, 0))) / (3 * spacing)
    result[outer] = (5 - 3 * q[outer] - np.sqrt(np.maximum(1 - 3 * (1 - q[outer])**2, 0))) / (6 * spacing)
    return result


def spread_marker_force(lattice, markers, marker_force, marker_volume_weight):
    """Spread stored force using the solver's h3 kernel and ds**3 weights.

    Saved Python force enters density-weighted momentum as dt*force_E.  It is
    already the source in that equation: multiplying it by rho again is wrong.
    """
    markers = _finite("markers", markers)
    force = _finite("marker_force", marker_force, markers.shape)
    weights = _finite("marker_volume_weight", marker_volume_weight, (len(markers),))
    if np.any(weights <= 0):
        raise ValueError("marker volume weights must be positive")
    nearest = np.rint((markers - lattice.origin) / lattice.spacing).astype(int)
    field = np.zeros(lattice.shape + (3,))
    for offset in np.ndindex(5, 5, 5):
        indices = nearest + np.asarray(offset) - 2
        locations = lattice.origin + lattice.spacing * indices
        delta = np.prod(delta_h3(locations - markers, lattice.spacing), axis=1)
        active = delta > 0
        valid = np.all((indices >= 0) & (indices < np.asarray(lattice.shape)), axis=1)
        if np.any(active & ~valid):
            raise ValueError("marker force support leaves the saved lattice")
        selected = indices[active]
        if np.any(lattice.cell_ids[selected[:, 0], selected[:, 1], selected[:, 2]] < 0):
            raise ValueError("marker force support encounters missing fine cells")
        np.add.at(field, tuple(selected.T), force[active] * (delta[active] * weights[active])[:, None])
    return field


def _lattice_edges(shape, valid):
    ids = np.arange(np.prod(shape)).reshape(shape)
    first, second, axis_ids = [], [], []
    for axis in range(3):
        lower, upper = [slice(None)] * 3, [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        a, b = ids[tuple(lower)].ravel(), ids[tuple(upper)].ravel()
        keep = valid.ravel()[a] & valid.ravel()[b]
        first.append(a[keep]); second.append(b[keep])
        axis_ids.append(np.full(np.count_nonzero(keep), axis, dtype=int))
    return np.concatenate(first), np.concatenate(second), np.concatenate(axis_ids)


def repair_pressure_band(lattice, raw_pressure, regular_gradient, signed_distance, *, width=1.5):
    """IBFast same-exterior-phase LFC repair with untouched outer anchors."""
    target = (signed_distance >= 0) & (signed_distance <= width * lattice.spacing) & np.isfinite(raw_pressure)
    valid = (signed_distance >= 0) & np.isfinite(raw_pressure)
    first, second, axes = _lattice_edges(lattice.shape, valid)
    selected = target.ravel()[first] | target.ravel()[second]
    first, second, axes = first[selected], second[selected], axes[selected]
    target_ids = np.flatnonzero(target.ravel())
    if not len(target_ids):
        raise ValueError("pressure repair band contains no cells; refine the grid")
    local = np.full(target.size, -1, dtype=int)
    local[target_ids] = np.arange(len(target_ids))
    gradient = regular_gradient.reshape(-1, 3)
    rhs = 0.5 * lattice.spacing * (gradient[first, axes] + gradient[second, axes])
    raw = raw_pressure.ravel()
    rows, columns, entries = [], [], []
    for sign, cells in ((-1.0, first), (1.0, second)):
        inside = local[cells] >= 0
        rows.append(np.flatnonzero(inside)); columns.append(local[cells[inside]])
        entries.append(np.full(np.count_nonzero(inside), sign))
        rhs[~inside] -= sign * raw[cells[~inside]]
    incidence = coo_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(columns))),
                           shape=(len(first), len(target_ids))).tocsr()
    normal = (incidence.T @ incidence).tocsr()
    count, labels = connected_components(normal, directed=False)
    anchors = np.zeros(len(target_ids), dtype=bool)
    for a, b in ((first, second), (second, first)):
        anchored = (local[a] >= 0) & (local[b] < 0)
        anchors[local[a[anchored]]] = True
    if len(np.unique(labels[anchors])) != count:
        raise ValueError("pressure repair has an unanchored band; enlarge the finest region")
    solved = lsqr(incidence, rhs, atol=1e-11, btol=1e-11, iter_lim=max(500, 20 * len(target_ids)))
    if solved[1] not in (0, 1, 2, 4, 5) or not np.isfinite(solved[0]).all():
        raise RuntimeError(f"pressure band solve failed (status {solved[1]})")
    repaired = raw.copy()
    repaired[target_ids] = solved[0]
    return repaired.reshape(lattice.shape), target, int(solved[2])


def surface_pressure_fit(cell_positions, signed_distance, pressure, markers, spacing, neighbours=40):
    """One-sided inverse-distance weighted quadratic extrapolation, as IBFast."""
    exterior = (signed_distance > 0.15 * spacing) & np.isfinite(pressure)
    points, values = cell_positions[exterior], pressure[exterior]
    wall_distances = signed_distance[exterior]
    if len(points) < 10:
        raise ValueError("too few exterior cells for quadratic surface pressure")
    maximum = min(4 * neighbours, len(points))
    distance, indices = cKDTree(points).query(markers, k=maximum)
    result = np.empty(len(markers))
    for marker in range(len(markers)):
        count = min(neighbours, maximum)
        while count < maximum and np.ptp(wall_distances[indices[marker, :count]]) < 1.9 * spacing:
            count = min(2 * count, maximum)
        chosen, distances = indices[marker, :count], distance[marker, :count]
        x, y, z = ((points[chosen] - markers[marker]) / spacing).T
        design = np.column_stack((np.ones(len(x)), x, y, z, x*x, y*y, z*z, x*y, x*z, y*z))
        weight = 1.0 / np.maximum(distances / spacing, 0.35)
        coefficient, _, rank, _ = np.linalg.lstsq(design * weight[:, None], values[chosen] * weight, rcond=None)
        if rank < 10:
            raise ValueError("surface pressure stencil is rank deficient; refine the grid")
        result[marker] = coefficient[0]
    return result


def effective_wall_offset(normals, resolution=24):
    """IBFast kernel I2 normal projection; returns offset in units of h.

    This preserves the IBFast midpoint quadrature and symmetry cache.  The
    offset is a kernel-dependent approximation, explicitly adjustable by CLI.
    """
    support = 1.5
    grid = (np.arange(resolution) + 0.5) / resolution * 2 * support - support
    coordinates = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    weights = np.prod(delta_h3(coordinates), axis=1) * (2 * support / resolution)**3
    edges = np.linspace(-support, support, 121)
    width = edges[1] - edges[0]
    result, cache = np.empty(len(normals)), {}
    for i, normal in enumerate(normals):
        key = tuple(np.round(np.sort(np.abs(normal)), 2))
        if key not in cache:
            density, _ = np.histogram(coordinates @ normal, bins=edges, weights=weights)
            density /= width
            twice_integrated = np.cumsum(np.cumsum(density) * width) * width
            cache[key] = np.sum(density * twice_integrated) * width
        result[i] = cache[key]
    return result


def rigid_velocity_function(markers, marker_velocity, center):
    """Fit a rigid velocity extension, also removing rigid rotation from shear."""
    relative = markers - center
    # v = translation + Omega cross r; fit all six rigid-body components.
    rotation = np.zeros((len(markers), 3, 3))
    rotation[:, 0, 1], rotation[:, 0, 2] = relative[:, 2], -relative[:, 1]
    rotation[:, 1, 0], rotation[:, 1, 2] = -relative[:, 2], relative[:, 0]
    rotation[:, 2, 0], rotation[:, 2, 1] = relative[:, 1], -relative[:, 0]
    design = np.concatenate((np.broadcast_to(np.eye(3), rotation.shape), rotation), axis=2)
    coefficients, _, rank, _ = np.linalg.lstsq(design.reshape(-1, 6), marker_velocity.ravel(), rcond=None)
    if rank < 6 or not np.allclose((design @ coefficients), marker_velocity, atol=1e-9, rtol=1e-7):
        raise ValueError("surface stress currently requires a rigid sphere velocity")
    def velocity_at(points):
        return coefficients[:3] + np.cross(coefficients[3:], points - center)
    return velocity_at


def repair_velocity_band(lattice, velocity, signed_distance, wall_offset, width, source, wall_velocity):
    """Shortley-Weller solve -lap(delta_u)=source with sharp moving-wall data.

    Source is -spread(tangential_force)/mu.  The wall offset and signed distance
    are physical lengths.  Outer delta_u=0 anchors preserve the original field.
    """
    h, shape = lattice.spacing, np.asarray(lattice.shape)
    band = (signed_distance > wall_offset) & (signed_distance <= width * h)
    if not np.any(band):
        raise ValueError("velocity repair band is empty")
    cells = np.argwhere(band)
    if np.any(cells == 0) or np.any(cells == shape - 1):
        raise ValueError("velocity repair touches finest-region edge; enlarge the refined region")
    ids = np.full(lattice.shape, -1, dtype=int)
    ids[band] = np.arange(len(cells))
    count = len(cells)
    rhs = source[band].copy()
    center_phi = signed_distance[band]
    positions = lattice.origin + cells * h
    rows, columns, entries = [], [], []
    for axis in range(3):
        direction = np.eye(3, dtype=int)[axis]
        sides = []
        for sign in (1, -1):
            neighbour = cells + sign * direction
            neighbour_phi = signed_distance[tuple(neighbour.T)]
            if not np.isfinite(neighbour_phi).all():
                raise ValueError("velocity band touches missing fine cells")
            cut = neighbour_phi <= wall_offset
            fraction = np.ones(count)
            fraction[cut] = (center_phi[cut] - wall_offset) / (center_phi[cut] - neighbour_phi[cut])
            fraction = np.clip(fraction, 0.05, 1.0)
            neighbour_ids = ids[tuple(neighbour.T)]
            regular = (~cut) & (neighbour_ids >= 0)
            sides.append((sign, fraction, cut, regular, neighbour_ids))
        scale = 2.0 / (h*h*(sides[0][1] + sides[1][1]))
        diagonal = scale * (1.0 / sides[0][1] + 1.0 / sides[1][1])
        rows.append(np.arange(count)); columns.append(np.arange(count)); entries.append(diagonal)
        for sign, fraction, cut, regular, neighbour_ids in sides:
            rows.append(np.flatnonzero(regular)); columns.append(neighbour_ids[regular])
            entries.append(-(scale / fraction)[regular])
            if np.any(cut):
                points = positions[cut] + (sign * fraction[cut] * h)[:, None] * direction
                mismatch = wall_velocity(points) - lattice.interpolate(velocity, points)
                rhs[cut] += (scale[cut] / fraction[cut])[:, None] * mismatch
    matrix = coo_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(columns))),
                        shape=(count, count)).tocsc()
    # One factorization for the three velocity components.
    correction = splu(matrix).solve(rhs)
    if not np.isfinite(correction).all():
        raise RuntimeError("velocity band solve produced non-finite values")
    repaired = velocity.copy()
    repaired[band] += correction
    return repaired, band


def extract_wall_shear(lattice, velocity, markers, normals, wall_velocity, wall_offset, viscosity, probes=(1.0, 2.0)):
    """mu times wall-pinned quadratic derivative of relative tangential velocity.

    Subtract the rigid velocity at each probe (including rotation).  Then the
    tangential normal derivative is the viscous traction for a rigid no-slip
    wall; pure solid-body rotation correctly has zero strain and zero shear.
    """
    distances = np.asarray(probes, dtype=float) * lattice.spacing
    if len(distances) < 2 or np.any(distances <= 0) or len(np.unique(distances)) < 2:
        raise ValueError("shear requires at least two distinct positive probe distances")
    samples = []
    for distance in distances:
        points = markers + (wall_offset + distance) * normals
        relative = lattice.interpolate(velocity, points) - wall_velocity(points)
        samples.append(relative - np.sum(relative * normals, axis=1)[:, None] * normals)
    design = np.column_stack((distances, distances**2))
    coefficients = np.linalg.lstsq(design, np.asarray(samples).reshape(len(distances), -1), rcond=None)[0]
    return viscosity * coefficients[0].reshape(len(markers), 3)


def integrate_surface_pressure(pressure, normals, area):
    """Gauge-invariant pressure force for finite equal-area sphere quadrature.

    Finite Fibonacci normals need not sum exactly to zero.  Remove the weighted
    surface mean before integration so a constant pressure cannot create force.
    Preserve the unadjusted integral and normal-area closure defect separately.
    """
    surface_mean = float(np.sum(pressure * area) / np.sum(area))
    raw_force = -np.sum(pressure[:, None] * normals * area[:, None], axis=0)
    force = -np.sum((pressure-surface_mean)[:, None] * normals * area[:, None], axis=0)
    diagnostics = {
        "pressure_force_unadjusted_quadrature": raw_force.tolist(),
        "surface_pressure_mean_removed_for_force": surface_mean,
        "normal_area_closure_relative_defect": float(np.linalg.norm(np.sum(normals*area[:, None], axis=0))/np.sum(area)),
        "pressure_force_quadrature": "area-weighted surface mean removed for gauge invariance; pressure field unchanged",
    }
    return force, diagnostics


def process_snapshot(snapshot, *, field_only=False, skip_shear=False, pressure_width=1.5, velocity_width=3.0, offset="auto"):
    """Recover one selected snapshot.  Returns arrays plus JSON-safe diagnostics."""
    required = ("cpos", "vc", "momentum_residual", "dt", "face_cells", "face_area")
    missing = [key for key in required if key not in snapshot]
    if missing:
        raise ValueError("snapshot lacks pressure-recovery data: " + ", ".join(missing) + "; rerun with selected field output")
    positions = _finite("cpos", snapshot["cpos"])
    velocity = _finite("vc", snapshot["vc"], positions.shape)
    dt = float(snapshot["dt"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("snapshot dt must be positive")
    gradient = _finite("momentum_residual", snapshot["momentum_residual"], positions.shape) / dt
    pressure, diagnostics = recover_pressure(positions, snapshot["face_cells"], snapshot["face_area"], gradient)
    arrays = {"cpos": positions, "vc": velocity, "pressure_raw": pressure,
              "step": np.asarray(snapshot.get("step", -1)), "time": np.asarray(snapshot.get("time", np.nan))}
    diagnostics.update({"method": "IBFast weighted momentum-residual graph projection",
                        "pressure_accuracy": "approximate: depends on inner convergence, spatial and temporal discretization"})
    if field_only:
        return arrays, diagnostics
    required_surface = ("index_cell", "lattice_origin", "lattice_spacing", "sphere_center", "sphere_radius",
                        "markers", "marker_velocity", "ib_force_L", "marker_volume_weight", "density", "kinematic_viscosity")
    missing = [key for key in required_surface if key not in snapshot]
    if missing:
        raise ValueError("snapshot lacks sphere stress data: " + ", ".join(missing))
    center = _finite("sphere_center", snapshot["sphere_center"], (3,))
    radius = float(snapshot["sphere_radius"])
    markers = _finite("markers", snapshot["markers"])
    radial = markers - center
    marker_radius = np.linalg.norm(radial, axis=1)
    if not np.isfinite(radius) or radius <= 0 or not np.allclose(marker_radius, radius, atol=radius*1e-6, rtol=1e-6):
        raise ValueError("surface recovery supports closed spherical marker surfaces only")
    normals = radial / marker_radius[:, None]
    lattice = UniformLattice(snapshot["index_cell"], snapshot["lattice_origin"], snapshot["lattice_spacing"])
    coordinates = lattice.coordinates()
    valid = lattice.cell_ids >= 0
    if not np.allclose(coordinates[valid], positions[lattice.cell_ids[valid]], atol=lattice.spacing*1e-5, rtol=0):
        raise ValueError("lattice coordinates do not match saved cell positions")
    signed_distance = np.linalg.norm(coordinates - center, axis=-1) - radius
    signed_distance[~valid] = np.nan
    if not np.isfinite(pressure_width) or pressure_width <= 0:
        raise ValueError("pressure band width must be positive")
    force = _finite("ib_force_L", snapshot["ib_force_L"], markers.shape)
    normal_force = np.sum(force * normals, axis=1)[:, None] * normals
    spread_normal = spread_marker_force(lattice, markers, normal_force, snapshot["marker_volume_weight"])
    raw_lattice = lattice.field(pressure)
    repaired_lattice, pressure_band, iterations = repair_pressure_band(
        lattice, raw_lattice, lattice.field(gradient) - spread_normal, signed_distance, width=pressure_width)
    repaired_pressure = pressure.copy()
    repaired_pressure[lattice.cell_ids[pressure_band]] = repaired_lattice[pressure_band]
    surface_raw = surface_pressure_fit(coordinates, signed_distance, raw_lattice, markers, lattice.spacing)
    surface_repaired = surface_pressure_fit(coordinates, signed_distance, repaired_lattice, markers, lattice.spacing)
    # Canonical Fibonacci markers have equal-area quadrature; ds**3 is not area.
    area = np.full(len(markers), 4 * np.pi * radius**2 / len(markers))
    pressure_force, force_diagnostics = integrate_surface_pressure(surface_repaired, normals, area)
    diagnostics.update(force_diagnostics)
    arrays.update(pressure_lfc=repaired_pressure, markers=markers, normals=normals, marker_area=area,
                  surface_pressure_raw=surface_raw, surface_pressure_lfc=surface_repaired,
                  pressure_force=pressure_force)
    diagnostics.update(pressure_band_cells=int(pressure_band.sum()), pressure_band_iterations=iterations,
                       geometry="analytic rigid sphere", surface_quadrature="equal area Fibonacci markers",
                       pressure_force=pressure_force.tolist(), pressure_band_width_h=float(pressure_width))
    if not skip_shear:
        density, nu = float(snapshot["density"]), float(snapshot["kinematic_viscosity"])
        if not np.isfinite(density * nu) or density <= 0 or nu <= 0:
            raise ValueError("density and kinematic viscosity must be positive")
        mu = density * nu
        offset_h = float(np.mean(effective_wall_offset(normals))) if offset == "auto" else float(offset)
        if not np.isfinite(offset_h) or not np.isfinite(velocity_width) or offset_h < 0 or velocity_width <= offset_h:
            raise ValueError("velocity band must exceed a finite non-negative wall offset")
        wall_velocity = rigid_velocity_function(markers, _finite("marker_velocity", snapshot["marker_velocity"], markers.shape), center)
        raw_velocity = lattice.field(velocity)
        source = -spread_marker_force(lattice, markers, force - normal_force, snapshot["marker_volume_weight"]) / mu
        repaired_velocity, velocity_band = repair_velocity_band(
            lattice, raw_velocity, signed_distance, offset_h*lattice.spacing, velocity_width, source, wall_velocity)
        raw_shear = extract_wall_shear(lattice, raw_velocity, markers, normals, wall_velocity, offset_h*lattice.spacing, mu)
        shear = extract_wall_shear(lattice, repaired_velocity, markers, normals, wall_velocity, offset_h*lattice.spacing, mu)
        shear_force = np.sum(shear * area[:, None], axis=0)
        surface_force = pressure_force + shear_force
        arrays.update(surface_shear_raw=raw_shear, surface_shear_repaired=shear, shear_force=shear_force, surface_force=surface_force)
        diagnostics.update(velocity_band_cells=int(velocity_band.sum()), effective_wall_offset_h=offset_h,
                           velocity_band_width_h=float(velocity_width), shear_force=shear_force.tolist(),
                           surface_force=surface_force.tolist(), shear_method="IBFast thin-band Stokes repair; rigid-frame quadratic read-off")
        if "body_force" in snapshot:
            body_force = _finite("body_force", snapshot["body_force"], (3,))
            difference = surface_force - body_force
            diagnostics.update(body_force=body_force.tolist(), surface_force_difference=difference.tolist(),
                               surface_force_difference_norm=float(np.linalg.norm(difference)),
                               surface_force_relative_difference=(float(np.linalg.norm(difference)/np.linalg.norm(body_force))
                                                                  if np.linalg.norm(body_force) > 1e-14 else None))
    if "ib_force_E" in snapshot:
        spread_all = spread_marker_force(lattice, markers, force, snapshot["marker_volume_weight"])
        saved = lattice.field(_finite("ib_force_E", snapshot["ib_force_E"], positions.shape))
        difference = float(np.max(np.abs(spread_all[valid] - saved[valid])))
        scale = max(float(np.max(np.abs(saved[valid]))), 1e-14)
        diagnostics["spread_replay_relative_error"] = difference / scale
        tolerance = 1e-4 if np.asarray(snapshot["ib_force_E"]).dtype.itemsize == 4 else 1e-8
        if difference / scale > tolerance:
            raise ValueError("saved marker and Eulerian forces disagree; LFC source is not reproducible")
    return arrays, diagnostics


def _stream_sha256(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024*1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path):
    with Path(path).open("rb") as handle:
        return _stream_sha256(handle)


def _write_point_cloud_vtu(path, positions, fields):
    """VTK vertex cells: sample points only, no inferred fluid/surface topology."""
    document = Element("VTKFile", type="UnstructuredGrid", version="0.1", byte_order="LittleEndian")
    grid = SubElement(document, "UnstructuredGrid")
    piece = SubElement(grid, "Piece", NumberOfPoints=str(len(positions)), NumberOfCells=str(len(positions)))
    def array(parent, name, values, kind="Float64"):
        values = np.asarray(values)
        element = SubElement(parent, "DataArray", type=kind, Name=name, format="ascii",
                             NumberOfComponents=str(values.shape[1] if values.ndim == 2 else 1))
        element.text = " ".join(format(value, ".17g") for value in values.ravel())
    points = SubElement(piece, "Points")
    array(points, "sample_position", positions)
    cells = SubElement(piece, "Cells")
    array(cells, "connectivity", np.arange(len(positions)), "Int64")
    array(cells, "offsets", np.arange(1, len(positions)+1), "Int64")
    array(cells, "types", np.ones(len(positions), dtype=int), "UInt8")
    data = SubElement(piece, "PointData")
    for name, values in fields.items():
        array(data, name, values)
    ElementTree(document).write(path, encoding="utf-8", xml_declaration=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot", type=Path, help="final_field.npz or one selected field snapshot")
    parser.add_argument("--output", type=Path, help="output directory (default: snapshot stem + _postprocess)")
    parser.add_argument("--field-only", action="store_true", help="recover raw whole-field pressure only")
    parser.add_argument("--skip-shear", action="store_true", help="recover pressure/LFC surface pressure without shear")
    parser.add_argument("--pressure-width", type=float, default=1.5, help="pressure repair band width in grid spacings")
    parser.add_argument("--velocity-width", type=float, default=3.0, help="shear repair band width in grid spacings")
    parser.add_argument("--offset", default="auto", help="effective wall offset in grid spacings, or auto")
    parser.add_argument("--vtk", action="store_true", help="also write ParaView VTU point samples; no mesh topology is inferred")
    parser.add_argument("--overwrite", action="store_true", help="archive existing postprocessing products before replacement")
    arguments = parser.parse_args(argv)
    output = arguments.output or arguments.snapshot.with_name(arguments.snapshot.stem + "_postprocess")
    known = ("pressure_and_stress.npz", "surface_stress.csv", "postprocess.json", "pressure_field.vtu", "surface_stress.vtu")
    existing = [output/name for name in known if (output/name).exists()]
    if existing and not arguments.overwrite:
        raise FileExistsError(f"postprocessing output already exists in {output}; use --overwrite to archive it first")
    for path in existing:
        if not path.is_file():
            raise ValueError(f"expected an output file, found {path}")
    # Hash and load the same open file, even if another process atomically
    # replaces the path while this offline recovery is running.
    with arguments.snapshot.open("rb") as source:
        source_hash = _stream_sha256(source)
        source.seek(0)
        with np.load(source, allow_pickle=False) as saved:
            arrays, diagnostics = process_snapshot(saved, field_only=arguments.field_only, skip_shear=arguments.skip_shear,
                                                   pressure_width=arguments.pressure_width, velocity_width=arguments.velocity_width,
                                                   offset=arguments.offset)
            for key in ("run_id", "config_fingerprint", "layout_fingerprint"):
                if key in saved:
                    diagnostics[key] = str(saved[key].item())
    output.mkdir(parents=True, exist_ok=True)
    if existing:
        archive = output/"archive"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
        archive.mkdir(parents=True)
        for path in existing:
            path.replace(archive/path.name)
        diagnostics["previous_output_archive"] = str(archive.resolve())
    np.savez_compressed(output / "pressure_and_stress.npz", **arrays)
    if "markers" in arrays:
        columns = [arrays["markers"], arrays["normals"], arrays["marker_area"], arrays["surface_pressure_raw"], arrays["surface_pressure_lfc"]]
        header = "x,y,z,nx,ny,nz,area,pressure_raw,pressure_lfc"
        if "surface_shear_repaired" in arrays:
            columns.extend((arrays["surface_shear_raw"], arrays["surface_shear_repaired"]))
            header += ",shear_raw_x,shear_raw_y,shear_raw_z,shear_repaired_x,shear_repaired_y,shear_repaired_z"
        np.savetxt(output / "surface_stress.csv", np.column_stack(columns), delimiter=",", header=header, comments="")
    if arguments.vtk:
        fields = {"velocity": arrays["vc"], "pressure_raw": arrays["pressure_raw"]}
        if "pressure_lfc" in arrays:
            fields["pressure_lfc"] = arrays["pressure_lfc"]
        _write_point_cloud_vtu(output/"pressure_field.vtu", arrays["cpos"], fields)
        if "markers" in arrays:
            surface_fields = {key: arrays[key] for key in ("normals", "surface_pressure_raw", "surface_pressure_lfc", "surface_shear_raw", "surface_shear_repaired") if key in arrays}
            _write_point_cloud_vtu(output/"surface_stress.vtu", arrays["markers"], surface_fields)
        diagnostics["vtk_representation"] = "VTK_VERTEX sample points at cell centers / markers; not volume or surface cells"
    diagnostics["source_snapshot"] = str(arguments.snapshot.resolve())
    diagnostics["source_snapshot_sha256"] = source_hash
    diagnostics["postprocess_source_sha256"] = _file_sha256(__file__)
    diagnostics["parameters"] = {key: getattr(arguments, key) for key in ("field_only", "skip_shear", "pressure_width", "velocity_width", "offset", "vtk")}
    (output / "postprocess.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Pressure/stress written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

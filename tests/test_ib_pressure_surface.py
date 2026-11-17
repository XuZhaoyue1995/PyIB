"""Manufactured tests of the two-sided (triangulated-surface) pressure correction."""

import unittest

import numpy as np
from scipy.spatial import ConvexHull

from ib_postprocess import UniformLattice, delta_h3, repair_pressure_band, surface_pressure_fit
from ib_pressure_surface import (
    cut_faces, delta_kernel, lattice_raw_pressure, orient_triangles, oriented_normals,
    pressure_force, repair_pressure_band_cut, spread_marker_force, surface_distance,
    surface_fit_side, two_sided_surface_pressure,
)


def lattice(count=21, spacing=0.1):
    ids = np.arange(count**3).reshape((count,) * 3)
    return UniformLattice(ids, np.full(3, -0.5 * (count - 1) * spacing), spacing)


def sphere_markers(count=260, radius=0.52):
    i = np.arange(count)
    z = 1 - 2 * (i + 0.5) / count
    angle = i * np.pi * (3 - np.sqrt(5))
    return radius * np.column_stack((np.sqrt(1 - z * z) * np.cos(angle), np.sqrt(1 - z * z) * np.sin(angle), z))


def plate(extent=1.4, count=15, height=0.033):
    """Flat sheet wider than the lattice, off the lattice planes, split into triangles."""
    axis = np.linspace(-extent, extent, count)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, height)))
    ids = np.arange(count * count).reshape(count, count)
    a, b, c, d = ids[:-1, :-1].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel(), ids[:-1, 1:].ravel()
    return points, np.concatenate((np.column_stack((a, b, c)), np.column_stack((a, c, d))))


class KernelTests(unittest.TestCase):
    def test_kernels_have_unit_zeroth_and_zero_first_moment(self):
        for kind in ("h3", "h3s", "h4"):
            for shift in (0.0, 0.13, 0.5, 0.71):
                nodes = np.arange(-4, 5) - shift
                weights = delta_kernel(nodes, 1.0, kind)
                self.assertAlmostEqual(weights.sum(), 1.0, places=13, msg=kind)
                self.assertAlmostEqual((weights * nodes).sum(), 0.0, places=13, msg=kind)

    def test_h3_matches_the_existing_kernel(self):
        r = np.linspace(-2.2, 2.2, 401)
        np.testing.assert_allclose(delta_kernel(r, 0.7, "h3"), delta_h3(r, 0.7), atol=1e-15)

    def test_smoothed_kernel_is_the_cell_average_of_h3(self):
        # phi*(r) = integral of phi_3 over [r - 1/2, r + 1/2]  (Yang et al. 2009, eq. 13)
        nodes, weights = np.polynomial.legendre.leggauss(40)
        for r in np.linspace(-2.1, 2.1, 43):
            pieces = np.linspace(r - 0.5, r + 0.5, 9)
            total = 0.0
            for lo, hi in zip(pieces[:-1], pieces[1:]):
                total += 0.5 * (hi - lo) * np.sum(weights * delta_h3(0.5 * (hi + lo) + 0.5 * (hi - lo) * nodes))
            self.assertAlmostEqual(float(delta_kernel(np.array([r]), 1.0, "h3s")[0]), total, places=6)

    def test_spread_conserves_force_for_every_kernel(self):
        grid = lattice()
        markers = sphere_markers(50, 0.4)
        force = np.random.default_rng(3).standard_normal(markers.shape)
        weight = np.full(len(markers), 2.5e-4)
        for kind in ("h3", "h3s", "h4"):
            field = spread_marker_force(grid, markers, force, weight, kind)
            np.testing.assert_allclose(field.sum(axis=(0, 1, 2)) * grid.spacing**3,
                                       (force * weight[:, None]).sum(axis=0), atol=1e-14)


class GeometryTests(unittest.TestCase):
    def test_scrambled_windings_are_made_consistent_and_outward(self):
        points = sphere_markers()
        triangles = ConvexHull(points).simplices.copy()
        rng = np.random.default_rng(1)
        flip = rng.random(len(triangles)) < 0.5
        triangles[flip] = triangles[flip][:, [0, 2, 1]]
        oriented, closed = orient_triangles(points, triangles)
        self.assertTrue(closed)
        triangle_normals, normals, area = oriented_normals(points, oriented)
        centroids = points[oriented].mean(axis=1)
        self.assertTrue(np.all(np.sum(triangle_normals * centroids, axis=1) > 0))
        self.assertGreater(np.min(np.sum(normals * points, axis=1)) / 0.52, 0.99)
        self.assertAlmostEqual(area.sum(), 4 * np.pi * 0.52**2, delta=0.02 * 4 * np.pi * 0.52**2)

    def test_open_surface_is_detected_and_keeps_one_side(self):
        points, triangles = plate()
        triangles[::3] = triangles[::3][:, [0, 2, 1]]
        oriented, closed = orient_triangles(points, triangles)
        self.assertFalse(closed)
        triangle_normals, normals, _ = oriented_normals(points, oriented)
        self.assertTrue(np.all(triangle_normals[:, 2] > 0.999) or np.all(triangle_normals[:, 2] < -0.999))
        np.testing.assert_allclose(np.abs(normals[:, 2]), 1.0, atol=1e-14)

    def test_non_manifold_edges_are_rejected(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0.]])
        with self.assertRaisesRegex(ValueError, "non-manifold"):
            orient_triangles(points, [[0, 1, 2], [0, 1, 3], [0, 1, 4]])

    def test_cut_faces_of_a_closed_surface_are_the_sign_changes(self):
        grid = lattice()
        points = sphere_markers()
        triangles, _ = orient_triangles(points, ConvexHull(points).simplices)
        triangle_normals, _, _ = oriented_normals(points, triangles)
        distance, _ = surface_distance(grid, points, triangles, triangle_normals, reach=6.0)
        cut = cut_faces(grid, points, triangles)
        self.assertGreater(sum(int(c.sum()) for c in cut), 100)
        for axis in range(3):
            low, high = [slice(None)] * 3, [slice(None)] * 3
            low[axis], high[axis] = slice(None, -1), slice(1, None)
            a, b = distance[tuple(low)], distance[tuple(high)]
            known = np.isfinite(a) & np.isfinite(b)
            change = known & (np.sign(a) != np.sign(b))
            np.testing.assert_array_equal(cut[axis][tuple(low)][known], change[known])

    def test_distance_to_a_plane_is_exact_and_signed(self):
        grid = lattice()
        points, triangles = plate()
        triangles, _ = orient_triangles(points, triangles)
        triangle_normals, _, _ = oriented_normals(points, triangles)
        distance, coordinates = surface_distance(grid, points, triangles, triangle_normals, reach=6.0)
        expected = (coordinates[..., 2] - 0.033) * triangle_normals[0, 2]
        known = np.isfinite(distance)
        self.assertGreater(known.sum(), 1000)
        np.testing.assert_allclose(distance[known], expected[known], atol=1e-13)


class RepairTests(unittest.TestCase):
    def test_lattice_recovery_pins_every_disconnected_piece(self):
        grid = lattice(13)
        coordinates = grid.coordinates()
        cells = np.abs(coordinates[..., 0]) > 0.25          # two slabs, not connected
        gradient = np.broadcast_to(np.array([0.3, -0.2, 0.5]), grid.shape + (3,)).copy()
        score = np.abs(coordinates[..., 0])
        pressure, pieces = lattice_raw_pressure(grid, gradient, cells, score)
        self.assertEqual(pieces, 2)
        self.assertTrue(np.all(np.isnan(pressure[~cells])))
        for side in (coordinates[..., 0] > 0.25, coordinates[..., 0] < -0.25):
            field = pressure[side]; x = coordinates[side]
            np.testing.assert_allclose(field - field[0], (x - x[0]) @ np.array([0.3, -0.2, 0.5]), atol=1e-8)
            self.assertAlmostEqual(float(np.min(np.abs(field))), 0.0, places=12)

    def test_two_sided_band_recovers_a_pressure_jump_across_an_open_sheet(self):
        grid = lattice()
        coordinates = grid.coordinates()
        points, triangles = plate()
        triangles, closed = orient_triangles(points, triangles)
        triangle_normals, normals, _ = oriented_normals(points, triangles)
        distance, _ = surface_distance(grid, points, triangles, triangle_normals, reach=6.0)
        slope = np.array([0.4, -0.3, 0.6])
        jump = 2.7
        truth = coordinates @ slope + jump * (distance > 0)
        regular = np.broadcast_to(slope, grid.shape + (3,)).copy()
        band = np.abs(distance) <= 1.5 * grid.spacing
        raw = truth.copy()
        raw[band] += 0.9 * np.sin(7 * coordinates[band, 0])              # defect inside the band only
        cut = cut_faces(grid, points, triangles)
        self.assertEqual(int(cut[2].sum()), grid.shape[0] * grid.shape[1])
        self.assertEqual(int(cut[0].sum() + cut[1].sum()), 0)
        repaired, target, unrepaired, _ = repair_pressure_band_cut(grid, raw, regular, distance, cut)
        np.testing.assert_array_equal(target, band)
        self.assertEqual(int(unrepaired.sum()), 0)
        np.testing.assert_allclose(repaired[target], truth[target], atol=2e-9)
        np.testing.assert_array_equal(repaired[~target], raw[~target])
        flat = coordinates.reshape(-1, 3)
        inner = np.all(np.abs(points[:, :2]) < 0.6, axis=1)               # markers inside the lattice
        plus = surface_fit_side(flat, distance.ravel(), repaired.ravel(), points[inner], grid.spacing)
        minus = surface_fit_side(flat, -distance.ravel(), repaired.ravel(), points[inner], grid.spacing)
        sign = triangle_normals[0, 2]
        np.testing.assert_allclose(plus - minus, jump * np.ones(inner.sum()), atol=1e-8)
        np.testing.assert_allclose(0.5 * (plus + minus), points[inner] @ slope + 0.5 * jump, atol=1e-8)
        self.assertIn(sign, (1.0, -1.0))

    def test_sign_based_repair_would_smear_the_jump(self):
        # The exterior-only repair uses the faces between any two cells with distance >= 0; given
        # the whole lattice (both sides) as "valid" it couples the two sides through the sheet.
        grid = lattice()
        coordinates = grid.coordinates()
        distance = coordinates[..., 2] - 0.033
        truth = 2.7 * (distance > 0)
        raw = truth.copy()
        regular = np.zeros(grid.shape + (3,))
        repaired, target, _ = repair_pressure_band(grid, raw, regular, np.abs(distance))
        self.assertGreater(float(np.max(np.abs(repaired[target] - truth[target]))), 0.5)

    def test_closed_surface_outer_side_matches_the_exterior_only_path(self):
        grid = lattice()
        coordinates = grid.coordinates(); flat = coordinates.reshape(-1, 3)
        points = sphere_markers()
        triangles = ConvexHull(points).simplices
        rng = np.random.default_rng(5)
        force = rng.standard_normal(points.shape) * 0.3 + 2.0 * points
        weight = np.full(len(points), 1.1e-3)
        x, y, z = flat.T
        smooth = np.column_stack((0.2 + 0.8 * x + 0.1 * y, -0.5 - 0.4 * y + 0.1 * x, 0.7 + 0.6 * z))
        tri, _ = orient_triangles(points, triangles)
        _, normals, _ = oriented_normals(points, tri)
        normal_force = np.sum(force * normals, axis=1)[:, None] * normals
        gradient = smooth.reshape(grid.shape + (3,)) + spread_marker_force(grid, points, normal_force, weight)
        result = two_sided_surface_pressure(grid, gradient, points, triangles, force, weight, reach=6.0)
        self.assertTrue(result["closed"])
        self.assertEqual(result["diagnostics"]["surface"], "closed")
        distance = result["distance"]
        outside = np.where(np.isfinite(distance), distance, -1.0)
        raw = result["pressure_raw_lattice"]
        regular = np.where(np.isfinite(gradient), gradient, 0.0) - spread_marker_force(grid, points, normal_force, weight)
        reference, target, _ = repair_pressure_band(grid, raw, regular, outside)
        np.testing.assert_allclose(result["pressure_lfc_lattice"][target], reference[target], atol=5e-9)
        expected = surface_pressure_fit(flat, outside.ravel(), reference.ravel(), points, grid.spacing)
        np.testing.assert_allclose(result["surface_pressure_plus"], expected, atol=5e-9)
        np.testing.assert_allclose(result["pressure_force"], pressure_force(result), atol=0)
        # the regular field is an exact quadratic: the repaired outer pressure reproduces it up to the gauge
        quadratic = 0.2 * x + 0.4 * x * x + 0.1 * x * y - 0.5 * y - 0.2 * y * y + 0.7 * z + 0.3 * z * z
        px, py, pz = points.T
        at_markers = 0.2 * px + 0.4 * px * px + 0.1 * px * py - 0.5 * py - 0.2 * py * py + 0.7 * pz + 0.3 * pz * pz
        far = np.isfinite(raw) & (distance > 3 * grid.spacing)
        shift = float(np.mean(quadratic.reshape(grid.shape)[far] - raw[far]))
        self.assertLess(float(np.max(np.abs(result["surface_pressure_plus"] + shift - at_markers))), 0.05)

    def test_far_gauge_changes_the_level_only(self):
        grid = lattice()
        points = sphere_markers(200, 0.4)
        triangles = ConvexHull(points).simplices
        force = 1.5 * points + 0.2
        weight = np.full(len(points), 1.0e-3)
        flat = grid.coordinates().reshape(-1, 3)
        gradient = np.column_stack((0.3 + flat[:, 1], flat[:, 0], np.full(len(flat), -0.4))).reshape(grid.shape + (3,))
        near = two_sided_surface_pressure(grid, gradient, points, triangles, force, weight, reach=6.0)
        far = two_sided_surface_pressure(grid, gradient, points, triangles, force, weight, reach=6.0, far_gauge=0.5)
        gauge = far["diagnostics"]["far_gauge"]
        self.assertGreater(gauge["cells"], 10)
        for name in ("surface_pressure_plus", "surface_pressure_minus", "surface_pressure_plus_raw"):
            difference = far[name] - near[name]
            self.assertLess(float(np.ptp(difference)), 2e-7)
        np.testing.assert_allclose(far["pressure_force"], near["pressure_force"], atol=2e-7)
        with self.assertRaisesRegex(ValueError, "no lattice cell that far"):
            two_sided_surface_pressure(grid, gradient, points, triangles, force, weight, far_gauge=50.0)


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------- snapshot / CLI
import json
from pathlib import Path
import tempfile

from ib_postprocess import main, process_snapshot, process_snapshot_two_sided


def _polynomial(points):
    x, y, z = np.asarray(points).T
    return 1.3 + 0.2 * x - 0.5 * y + 0.7 * z + 0.4 * x * x - 0.2 * y * y + 0.3 * z * z + 0.1 * x * y


def _polynomial_gradient(points):
    x, y, z = np.asarray(points).T
    return np.column_stack((0.2 + 0.8 * x + 0.1 * y, -0.5 - 0.4 * y + 0.1 * x, 0.7 + 0.6 * z))


def _snapshot(count=17, spacing=0.25, sphere=True):
    ids = np.arange(count**3).reshape((count,) * 3)
    grid = UniformLattice(ids, np.full(3, -0.5 * (count - 1) * spacing), spacing)
    positions = grid.coordinates().reshape(-1, 3)
    faces = []
    for axis in range(3):
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[axis], high[axis] = slice(None, -1), slice(1, None)
        faces.append(np.column_stack((ids[tuple(low)].ravel(), ids[tuple(high)].ravel())))
        for side in (0, -1):
            boundary = [slice(None)] * 3
            boundary[axis] = side
            cells = ids[tuple(boundary)].ravel()
            faces.append(np.column_stack((cells, np.full(len(cells), -1))))
    faces = np.concatenate(faces)
    snapshot = {
        "cpos": positions, "vc": np.zeros_like(positions), "dt": np.array(0.01), "step": np.array(10),
        "time": np.array(0.1), "momentum_residual": 0.01 * _polynomial_gradient(positions),
        "face_cells": faces, "face_area": np.full(len(faces), spacing**2),
        "index_cell": grid.cell_ids, "lattice_origin": grid.origin, "lattice_spacing": np.array(spacing),
    }
    if sphere:
        markers = sphere_markers(96, 0.6)
        snapshot.update(sphere_center=np.zeros(3), sphere_radius=np.array(0.6))
    else:
        axis = np.linspace(-0.9, 0.9, 9)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        markers = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.06)))
    snapshot.update(markers=markers, marker_volume_weight=np.full(len(markers), 0.01),
                    ib_force_L=np.zeros_like(markers), ib_force_E=np.zeros_like(positions))
    return snapshot


def _gauge(snapshot):
    boundary = np.unique(snapshot["face_cells"][snapshot["face_cells"][:, 1] < 0, 0])
    return _polynomial(snapshot["cpos"])[boundary].mean()


class SnapshotTwoSidedTests(unittest.TestCase):
    def test_sphere_snapshot_gives_both_sides_and_keeps_the_exterior_result(self):
        snapshot = _snapshot()
        arrays, diagnostics = process_snapshot_two_sided(snapshot)
        expected = _polynomial(snapshot["markers"]) - _gauge(snapshot)
        self.assertTrue(bool(arrays["surface_closed"]))
        self.assertEqual(diagnostics["triangulation"], "convex hull of the sphere markers")
        np.testing.assert_allclose(arrays["surface_pressure_plus"], expected, atol=2e-8, rtol=1e-8)
        np.testing.assert_allclose(arrays["surface_pressure_minus"], expected, atol=2e-8, rtol=1e-8)
        original, _ = process_snapshot(snapshot)
        np.testing.assert_allclose(arrays["surface_pressure_plus"], original["surface_pressure_lfc"], atol=2e-8)
        np.testing.assert_allclose(arrays["pressure_raw"], original["pressure_raw"], atol=0)
        self.assertEqual(diagnostics["spread_replay_relative_error"], 0.0)

    def test_cli_writes_two_sided_products_and_guards_its_options(self):
        snapshot = _snapshot()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "final_field.npz"
            np.savez(source, **snapshot)
            self.assertEqual(main([str(source), "--two-sided", "--far-gauge", "1.2", "--vtk"]), 0)
            output = source.with_name("final_field_postprocess")
            header = (output / "surface_pressure.csv").read_text().splitlines()[0]
            self.assertEqual(header, "x,y,z,nx,ny,nz,area,p_plus_raw,p_minus_raw,p_plus,p_minus")
            diagnostic = json.loads((output / "postprocess.json").read_text())
            self.assertEqual(diagnostic["surface"], "closed")
            self.assertEqual(diagnostic["parameters"]["far_gauge"], 1.2)
            self.assertEqual(diagnostic["parameters"]["kernel"], "h3")
            self.assertGreater(diagnostic["far_gauge"]["cells"], 100)
            self.assertEqual(len(diagnostic["surface_postprocess_source_sha256"]), 64)
            with np.load(output / "pressure.npz") as result:
                far = np.linalg.norm(snapshot["cpos"], axis=1) - 0.6 >= 1.2 + 1e-9
                self.assertLess(abs(float(result["pressure_raw"][far].mean())), 0.05)
                difference = result["surface_pressure_plus"] - _polynomial(snapshot["markers"])
                self.assertLess(float(np.ptp(difference)), 5e-8)       # only the zero moved
                self.assertEqual(result["triangles"].shape[1], 3)
            self.assertTrue((output / "surface_pressure.vtu").is_file())
            for bad in (["--far-gauge", "1.0"], ["--kernel", "h3s"], ["--two-sided", "--field-only"]):
                with self.assertRaises(SystemExit):
                    main([str(source), "--overwrite"] + bad)

    def test_open_sheet_needs_and_uses_supplied_triangles(self):
        snapshot = _snapshot(sphere=False)
        with self.assertRaisesRegex(ValueError, "triangulated surface"):
            process_snapshot_two_sided(snapshot)
        n = 9
        ids = np.arange(n * n).reshape(n, n)
        a, b, c, d = ids[:-1, :-1].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel(), ids[:-1, 1:].ravel()
        triangles = np.concatenate((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
        arrays, diagnostics = process_snapshot_two_sided(snapshot, triangles=triangles)
        self.assertFalse(bool(arrays["surface_closed"]))
        self.assertEqual(diagnostics["surface"], "open")
        self.assertEqual(diagnostics["pressure_band_cells_left_raw"], 0)
        expected = _polynomial(snapshot["markers"]) - _gauge(snapshot)
        inner = np.all(np.abs(snapshot["markers"][:, :2]) < 0.7, axis=1)
        np.testing.assert_allclose(arrays["surface_pressure_plus"][inner], expected[inner], atol=5e-8)
        np.testing.assert_allclose(arrays["surface_pressure_minus"][inner], expected[inner], atol=5e-8)
        np.testing.assert_allclose(arrays["pressure_force"], 0.0, atol=1e-6)   # no jump, no load

    def test_lattice_raw_pressure_matches_the_mesh_projection_up_to_a_constant(self):
        snapshot = _snapshot()
        mesh, _ = process_snapshot_two_sided(snapshot)
        local, diagnostics = process_snapshot_two_sided(snapshot, raw_pressure="lattice", far_gauge=1.0)
        self.assertNotIn("pressure_lfc", local)
        self.assertEqual(diagnostics["raw_pressure_source"], "lattice")
        difference = local["surface_pressure_plus"] - mesh["surface_pressure_plus"]
        self.assertLess(float(np.ptp(difference)), 5e-7)

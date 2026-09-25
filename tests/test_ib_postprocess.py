"""Manufactured-field and output-contract tests for offline pressure recovery."""

import json
from pathlib import Path
import tempfile
import unittest
from xml.etree.ElementTree import parse

import numpy as np

from ib_postprocess import (
    UniformLattice, integrate_surface_pressure, main, process_snapshot, recover_pressure,
    repair_pressure_band, spread_marker_force,
    surface_pressure_fit,
)


def uniform_grid(count=17, spacing=0.25):
    ids = np.arange(count**3).reshape((count,) * 3)
    lattice = UniformLattice(ids, np.full(3, -0.5*(count-1)*spacing), spacing)
    positions = lattice.coordinates().reshape(-1, 3)
    faces = []
    for axis in range(3):
        low, high = [slice(None)]*3, [slice(None)]*3
        low[axis], high[axis] = slice(None, -1), slice(1, None)
        faces.append(np.column_stack((ids[tuple(low)].ravel(), ids[tuple(high)].ravel())))
        for side in (0, -1):
            boundary = [slice(None)]*3
            boundary[axis] = side
            cells = ids[tuple(boundary)].ravel()
            faces.append(np.column_stack((cells, np.full(len(cells), -1))))
    faces = np.concatenate(faces)
    return lattice, positions, faces, np.full(len(faces), spacing**2)


def sphere_markers(count=96, radius=0.6):
    i = np.arange(count)
    z = 1 - 2*(i+0.5)/count
    angle = i*np.pi*(3-np.sqrt(5))
    return radius*np.column_stack((np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z))


def polynomial(points):
    x, y, z = np.asarray(points).T
    return 1.3 + 0.2*x - 0.5*y + 0.7*z + 0.4*x*x - 0.2*y*y + 0.3*z*z + 0.1*x*y


def polynomial_gradient(points):
    x, y, z = np.asarray(points).T
    return np.column_stack((0.2+0.8*x+0.1*y, -0.5-0.4*y+0.1*x, 0.7+0.6*z))


class PressureTests(unittest.TestCase):
    def test_linear_pressure_integral_matches_analytic_sphere_force(self):
        # The six cardinal directions integrate n_i*n_j exactly on a sphere.
        # For p = p0 + g dot x, the divergence theorem gives Fp = -V*g.
        radius = 0.6
        normals = np.concatenate((np.eye(3), -np.eye(3)))
        gradient = np.array([0.2, -0.5, 0.7])
        area = np.full(6, 4*np.pi*radius**2/6)
        pressure = 17.0 + radius*normals @ gradient
        force, _ = integrate_surface_pressure(pressure, normals, area)
        np.testing.assert_allclose(force, -(4*np.pi*radius**3/3)*gradient, atol=2e-15)

    def test_pressure_force_is_gauge_invariant_even_for_coarse_markers(self):
        markers = sphere_markers(13)
        normals = markers/0.6
        area = np.full(13, 4*np.pi*0.6**2/13)
        force, diagnostic = integrate_surface_pressure(markers[:, 0], normals, area)
        shifted, _ = integrate_surface_pressure(markers[:, 0] + 53., normals, area)
        constant, _ = integrate_surface_pressure(np.full(13, 53.), normals, area)
        np.testing.assert_allclose(force, shifted, atol=1e-13)
        np.testing.assert_allclose(constant, 0., atol=1e-13)
        self.assertGreater(diagnostic["normal_area_closure_relative_defect"], 0.)

    def test_whole_graph_recovers_quadratic_and_boundary_gauge(self):
        _, positions, faces, area = uniform_grid(9)
        pressure, diagnostic = recover_pressure(positions, faces, area, polynomial_gradient(positions))
        expected = polynomial(positions)
        boundary = np.unique(faces[faces[:, 1] < 0, 0])
        expected -= expected[boundary].mean()
        np.testing.assert_allclose(pressure, expected, atol=3e-9, rtol=1e-8)
        self.assertLess(abs(pressure[boundary].mean()), 1e-14)
        self.assertLess(diagnostic["gradient_projection_weighted_rms"], 1e-9)

    def test_pressure_band_removes_defect_and_retains_outer_anchors(self):
        lattice, positions, _, _ = uniform_grid()
        distance = (np.linalg.norm(positions, axis=1)-0.6).reshape(lattice.shape)
        truth = polynomial(positions).reshape(lattice.shape)
        band = (distance >= 0) & (distance <= 1.5*lattice.spacing)
        raw = truth.copy()
        raw[band] += 0.8*np.cos(positions.reshape(lattice.shape+(3,))[band, 0])
        repaired, target, _ = repair_pressure_band(lattice, raw, polynomial_gradient(positions).reshape(lattice.shape+(3,)), distance)
        np.testing.assert_allclose(repaired[target], truth[target], atol=1e-9, rtol=1e-9)
        np.testing.assert_array_equal(repaired[~target], raw[~target])

    def test_surface_fit_exact_for_external_quadratic(self):
        lattice, positions, _, _ = uniform_grid()
        markers = sphere_markers()
        distance = np.linalg.norm(positions, axis=1)-0.6
        fitted = surface_pressure_fit(positions, distance, polynomial(positions), markers, lattice.spacing)
        np.testing.assert_allclose(fitted, polynomial(markers), atol=2e-13)

    def test_surface_fit_widens_a_planar_stencil(self):
        lattice, positions, _, _ = uniform_grid()
        markers = np.array([[0., 0., 0.], [0.05, -0.05, 0.]])
        # The nearest 40 cells can span only two z planes; adaptive widening
        # restores a determined normal quadratic.
        fitted = surface_pressure_fit(positions, positions[:, 2], polynomial(positions), markers, lattice.spacing)
        np.testing.assert_allclose(fitted, polynomial(markers), atol=2e-13)

    def test_disconnected_pressure_graph_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disconnected"):
            recover_pressure(np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]]),
                             np.array([[0, 1], [2, -1]]), np.ones(2), np.zeros((3, 3)))


class ForceSpreadingTests(unittest.TestCase):
    def test_spread_conserves_force(self):
        lattice, _, _, _ = uniform_grid()
        markers = sphere_markers(50)
        rng = np.random.default_rng(12)
        force = rng.normal(size=markers.shape)
        volume = rng.uniform(0.01, 0.03, len(markers))
        spread = spread_marker_force(lattice, markers, force, volume)
        np.testing.assert_allclose(spread.sum(axis=(0, 1, 2))*lattice.spacing**3,
                                   np.sum(force*volume[:, None], axis=0), atol=2e-15)


class SnapshotTests(unittest.TestCase):
    def snapshot(self):
        lattice, positions, faces, area = uniform_grid()
        markers = sphere_markers()
        velocity = np.tile([0.3, -0.2, 0.1], (len(positions), 1))
        return {
            "cpos": positions, "vc": velocity, "dt": np.array(0.01), "step": np.array(10), "time": np.array(0.1),
            "momentum_residual": 0.01*polynomial_gradient(positions), "face_cells": faces, "face_area": area,
            "index_cell": lattice.cell_ids, "lattice_origin": lattice.origin, "lattice_spacing": np.array(lattice.spacing),
            "sphere_center": np.zeros(3), "sphere_radius": np.array(0.6), "markers": markers,
            "marker_velocity": np.tile([0.3, -0.2, 0.1], (len(markers), 1)),
            "marker_volume_weight": np.full(len(markers), 4*np.pi*0.6**2*lattice.spacing/len(markers)),
            "ib_force_L": np.zeros_like(markers), "ib_force_E": np.zeros_like(positions),
            "density": np.array(1.7), "kinematic_viscosity": np.array(0.2),
        }

    def test_complete_offline_recovery_and_cli_artifacts(self):
        snapshot = self.snapshot()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"final_field.npz"
            np.savez(source, **snapshot)
            self.assertEqual(main([str(source), "--vtk"]), 0)
            output = source.with_name("final_field_postprocess")
            with np.load(output/"pressure.npz") as result:
                self.assertFalse(any("shear" in key for key in result.files))
                self.assertNotIn("surface_force", result.files)
                self.assertEqual(result["pressure_force"].shape, (3,))
                boundary = np.unique(snapshot["face_cells"][snapshot["face_cells"][:, 1] < 0, 0])
                gauge = polynomial(snapshot["cpos"])[boundary].mean()
                np.testing.assert_allclose(result["surface_pressure_lfc"], polynomial(snapshot["markers"])-gauge,
                                           atol=2e-8, rtol=1e-8)
            diagnostic = json.loads((output/"postprocess.json").read_text())
            self.assertEqual(diagnostic["spread_replay_relative_error"], 0.)
            self.assertFalse(any("shear" in key for key in diagnostic))
            self.assertNotIn("surface_force", diagnostic)
            self.assertIn("pressure contribution only", diagnostic["pressure_force_definition"])
            self.assertTrue((output/"surface_pressure.csv").is_file())
            self.assertEqual((output/"surface_pressure.csv").read_text().splitlines()[0],
                             "x,y,z,nx,ny,nz,area,pressure_raw,pressure_lfc")
            self.assertEqual(len(diagnostic["source_snapshot_sha256"]), 64)
            piece = parse(output/"pressure_field.vtu").find("./UnstructuredGrid/Piece")
            self.assertEqual(int(piece.attrib["NumberOfPoints"]), len(snapshot["cpos"]))
            surface = parse(output/"surface_pressure.vtu").find("./UnstructuredGrid/Piece")
            self.assertEqual(int(surface.attrib["NumberOfPoints"]), len(snapshot["markers"]))
            with self.assertRaises(FileExistsError):
                main([str(source), "--field-only"])
            self.assertEqual(main([str(source), "--field-only", "--overwrite"]), 0)
            self.assertFalse((output/"surface_pressure.csv").exists())
            self.assertFalse((output/"surface_pressure.vtu").exists())
            self.assertEqual(len(list((output/"archive").glob("*/surface_pressure.csv"))), 1)

    def test_pressure_recovery_needs_no_wall_motion_or_viscosity(self):
        snapshot = self.snapshot()
        for key in ("marker_velocity", "density", "kinematic_viscosity"):
            del snapshot[key]
        arrays, _ = process_snapshot(snapshot)
        boundary = np.unique(snapshot["face_cells"][snapshot["face_cells"][:, 1] < 0, 0])
        gauge = polynomial(snapshot["cpos"])[boundary].mean()
        np.testing.assert_allclose(arrays["surface_pressure_lfc"], polynomial(snapshot["markers"])-gauge,
                                   atol=2e-8, rtol=1e-8)

    def test_field_only_recovers_pressure_without_surface_metadata(self):
        snapshot = self.snapshot()
        minimal = {key: snapshot[key] for key in
                   ("cpos", "vc", "momentum_residual", "dt", "face_cells", "face_area")}
        arrays, _ = process_snapshot(minimal, field_only=True)
        boundary = np.unique(snapshot["face_cells"][snapshot["face_cells"][:, 1] < 0, 0])
        expected = polynomial(snapshot["cpos"])
        expected -= expected[boundary].mean()
        np.testing.assert_allclose(arrays["pressure_raw"], expected, atol=2e-8, rtol=1e-8)
        self.assertNotIn("markers", arrays)
        self.assertNotIn("pressure_force", arrays)

    def test_overwrite_archives_previous_release_products(self):
        snapshot = self.snapshot()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"final_field.npz"
            np.savez(source, **snapshot)
            output = source.with_name("final_field_postprocess")
            output.mkdir()
            previous = ("pressure_and_stress.npz", "surface_stress.csv", "surface_stress.vtu")
            for name in previous:
                (output/name).write_bytes(b"previous-release-artifact")
            with self.assertRaises(FileExistsError):
                main([str(source), "--field-only"])
            self.assertEqual(main([str(source), "--field-only", "--overwrite"]), 0)
            for name in previous:
                self.assertFalse((output/name).exists())
                copies = list((output/"archive").glob("*/"+name))
                self.assertEqual(len(copies), 1)
                self.assertEqual(copies[0].read_bytes(), b"previous-release-artifact")

    def test_old_snapshot_does_not_silently_invent_pressure(self):
        with self.assertRaisesRegex(ValueError, "momentum_residual"):
            process_snapshot({"cpos": np.zeros((10, 3)), "vc": np.zeros((10, 3))})

    def test_non_spherical_geometry_is_rejected(self):
        snapshot = self.snapshot()
        snapshot["markers"][0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "spherical"):
            process_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()

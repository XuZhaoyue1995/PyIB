"""Manufactured-field tests of offline pressure and surface-stress recovery."""

import json
from pathlib import Path
import tempfile
import unittest
from xml.etree.ElementTree import parse

import numpy as np

from ib_postprocess import (
    UniformLattice, extract_wall_shear, integrate_surface_pressure, main, process_snapshot, recover_pressure,
    repair_pressure_band, rigid_velocity_function, spread_marker_force,
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


class ShearTests(unittest.TestCase):
    def test_wall_pinned_quadratic_recovers_nonzero_shear(self):
        lattice, positions, _, _ = uniform_grid(17)
        # Samples at z=h,2h land exactly on lattice planes: no interpolation
        # error is folded into the test of the wall derivative.
        velocity = np.zeros_like(positions)
        velocity[:, 0] = 2.3*positions[:, 2] + 0.4*positions[:, 2]**2
        markers = np.array([[0., 0., 0.], [0.2, -0.1, 0.]])
        normals = np.tile([0., 0., 1.], (len(markers), 1))
        shear = extract_wall_shear(lattice, velocity.reshape(lattice.shape+(3,)), markers, normals,
                                   lambda p: np.zeros_like(p), 0., 1.7)
        np.testing.assert_allclose(shear, np.tile([1.7*2.3, 0, 0], (len(markers), 1)), atol=2e-14)

    def test_rigid_rotation_has_zero_shear(self):
        lattice, positions, _, _ = uniform_grid()
        markers = sphere_markers()
        omega = np.array([0.4, -0.7, 0.2])
        translation = np.array([0.1, 0.3, -0.2])
        marker_velocity = translation + np.cross(omega, markers)
        wall = rigid_velocity_function(markers, marker_velocity, np.zeros(3))
        velocity = (translation + np.cross(omega, positions)).reshape(lattice.shape+(3,))
        shear = extract_wall_shear(lattice, velocity, markers, markers/0.6, wall, 0.3*lattice.spacing, 2.)
        np.testing.assert_allclose(shear, 0., atol=2e-14)

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
            self.assertEqual(main([str(source), "--offset", "0.35", "--vtk"]), 0)
            output = source.with_name("final_field_postprocess")
            with np.load(output/"pressure_and_stress.npz") as result:
                np.testing.assert_allclose(result["surface_shear_repaired"], 0, atol=2e-14)
                boundary = np.unique(snapshot["face_cells"][snapshot["face_cells"][:, 1] < 0, 0])
                gauge = polynomial(snapshot["cpos"])[boundary].mean()
                np.testing.assert_allclose(result["surface_pressure_lfc"], polynomial(snapshot["markers"])-gauge,
                                           atol=2e-8, rtol=1e-8)
            diagnostic = json.loads((output/"postprocess.json").read_text())
            self.assertEqual(diagnostic["spread_replay_relative_error"], 0.)
            self.assertTrue((output/"surface_stress.csv").is_file())
            self.assertEqual(len(diagnostic["source_snapshot_sha256"]), 64)
            piece = parse(output/"pressure_field.vtu").find("./UnstructuredGrid/Piece")
            self.assertEqual(int(piece.attrib["NumberOfPoints"]), len(snapshot["cpos"]))
            with self.assertRaises(FileExistsError):
                main([str(source), "--field-only"])
            self.assertEqual(main([str(source), "--field-only", "--overwrite"]), 0)
            self.assertFalse((output/"surface_stress.csv").exists())
            self.assertFalse((output/"surface_stress.vtu").exists())
            self.assertEqual(len(list((output/"archive").glob("*/surface_stress.csv"))), 1)

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

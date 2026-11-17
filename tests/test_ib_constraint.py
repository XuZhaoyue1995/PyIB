"""Tests of the optional K-matrix force correction (constrained IB update)."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend import GPU, FP32, asnumpy, xp
import ib_case as C
from ib_constraint import ConstraintError, KMatrixCorrection
from ib_patches import make_patches


def case(kind="fixed", k_matrix=True, steps=3, **k_options):
    config = {
        "schema_version": 1,
        "name": f"k_matrix_{kind}",
        "case": {"kind": "fixed_sphere" if kind == "fixed" else "oscillating_sphere"},
        "mesh": {"kind": "uniform_box", "dx": 0.0625,
                 "bounds": [[-0.75, 0.75], [-0.75, 0.75], [-0.75, 0.75]]},
        "physics": {"reynolds": 100.0, "density": 1.0, "inflow_speed": 1.0},
        "time": {"dt": 0.01, "steps": steps, "inner_iterations": 4},
        "ib": {"radius": 0.25, "center": [0.0, 0.0, 0.0], "marker_spacing": 0.0625,
               "force_mode": "accumulated_explicit",
               "k_matrix": {"enabled": bool(k_matrix), **k_options}},
        "solver": {"sparse": True, "cg_check_interval": 1, "cg_max_iterations": 2000},
        "output": {"sample_every": 1, "checkpoint_every": 0, "save_field": True},
    }
    if kind != "fixed":
        config["ib"]["motion"] = {"kind": "oscillation", "axis": [1.0, 0.0, 0.0], "amplitude": 0.02,
                                  "frequency": 0.5, "phase": 0.0}
    return config


@unittest.skipIf(GPU or FP32, "CPU float64 unit tests")
class OperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = C.build_runtime(C.validate_config(case()))
        cls.driver = cls.runtime.driver
        cls.km = cls.driver.ib_kmatrix
        cls.km.rebuild(cls.driver.ib_markers, cls.driver.ib_ds)

    def test_switch_attaches_a_correction_with_one_patch_per_marker(self):
        self.assertIsInstance(self.km, KMatrixCorrection)
        self.assertEqual(self.km.n_patches, self.km.n_markers)
        self.assertEqual(self.km.n3, 3 * self.runtime.reference_markers.shape[0])
        self.assertLess(self.km.volume_defect, 1e-12)

    def test_sparse_b_is_the_interpolation_of_the_curl(self):
        drv, core, ib, km = self.driver, self.runtime.core, self.runtime.ib, self.km
        rng = np.random.default_rng(0)
        psi = rng.standard_normal(core.nedge)
        psi[drv.dir_mask] = 0.0
        velocity = core.interp_f2c(core.curl(psi))
        expected = ib.interp(velocity, drv.ib_markers).reshape(-1)
        np.testing.assert_allclose(km.B @ psi, expected, rtol=0, atol=1e-12 * np.abs(expected).max())

    def test_sparse_bt_is_the_masked_edge_residual_of_the_spread_force(self):
        drv, core, ib, km = self.driver, self.runtime.core, self.runtime.ib, self.km
        rng = np.random.default_rng(1)
        lam = rng.standard_normal(km.n3)
        dt = 0.01
        # lam = dt ds^3 dF_L  ->  the momentum source dt * spread(dF_L), taken to the edges
        force = lam.reshape(-1, 3) / (dt * drv.ib_ds[:, None] ** 3)
        source = dt * ib.spread(force, drv.ib_markers, drv.ib_ds)
        expected = core.rot(core.integr_c2f(source))
        expected[drv.dir_mask] = 0.0
        np.testing.assert_allclose(km.BT @ lam, expected, rtol=0, atol=1e-11 * np.abs(expected).max())

    def test_k_is_positive_definite_and_its_solve_is_accurate(self):
        km = self.km
        k = (km.B.multiply(asnumpy(self.driver.prec)[None, :]).tocsr() @ km.BT).toarray()
        np.testing.assert_allclose(k, k.T, atol=1e-12 * np.abs(k).max())
        eigenvalues = np.linalg.eigvalsh(k)
        self.assertGreater(eigenvalues.min(), 0.0)
        rhs = np.random.default_rng(2).standard_normal(km.n3)
        solution = km.k_solve(rhs)
        self.assertLess(np.linalg.norm(k @ solution - rhs) / np.linalg.norm(rhs), 1e-9)
        self.assertGreater(km.k_info["k_pivot_min"], 0.0)

    def test_inverse_and_cholesky_backends_agree(self):
        drv = self.driver
        other = KMatrixCorrection(drv, self.runtime.ib, k_solver="cholesky")
        other.rebuild(drv.ib_markers, drv.ib_ds)
        self.assertEqual(other.k_info["k_solver"], "cholesky")
        self.assertEqual(self.km.k_info["k_solver"], "inverse")
        rhs = np.random.default_rng(3).standard_normal(self.km.n3)
        a, b = self.km.k_solve(rhs), other.k_solve(rhs)
        self.assertLess(np.linalg.norm(a - b) / np.linalg.norm(b), 1e-9)

    def test_projected_cg_solves_the_saddle_point_system(self):
        drv, core, km = self.driver, self.runtime.core, self.km
        rng = np.random.default_rng(4)
        dt = 0.01
        psi = rng.standard_normal(core.nedge) * 1e-3
        # a consistent right-hand side from a smooth edge field and a random multiplier
        x_true = np.cos(3 * asnumpy(core.epos[:, 0])) * np.sin(2 * asnumpy(core.epos[:, 1])) * 1e-2
        x_true[drv.dir_mask] = 0.0
        lam_true = rng.standard_normal(km.n3) * 1e-3
        b = km._matvec(x_true, dt) - km.BT @ lam_true
        d = km.B @ x_true
        x, lam, iterations, info = km.solve(b, d, psi, dt)
        self.assertGreater(iterations, 0)
        self.assertFalse(info["iteration_cap"])
        momentum = km._matvec(x, dt) - km.BT @ lam - b
        momentum[drv.dir_mask] = 0.0
        self.assertLess(np.linalg.norm(km.B @ x - d) / np.linalg.norm(d), 1e-10)
        self.assertLess(np.linalg.norm(momentum) / np.linalg.norm(b), 1e-4)
        # tighter tolerance -> converges to the constructed solution
        tight = KMatrixCorrection(drv, self.runtime.ib, tolerance=1e-10, tolerance_absolute=0.0)
        tight.rebuild(drv.ib_markers, drv.ib_ds)
        x2, lam2, _, _ = tight.solve(b, d, psi, dt)
        self.assertLess(np.linalg.norm(lam2 - lam_true) / np.linalg.norm(lam_true), 1e-5)
        velocity_error = core.interp_f2c(core.curl(x2 - x_true))
        self.assertLess(float(np.abs(velocity_error).max()) / float(np.abs(core.interp_f2c(core.curl(x_true))).max()), 1e-5)

    def test_dependent_constraints_are_reported(self):
        drv = self.driver
        doubled = np.concatenate([asnumpy(drv.ib_markers)] * 2)      # every marker twice
        width = np.concatenate([asnumpy(drv.ib_ds)] * 2)
        km = KMatrixCorrection(drv, self.runtime.ib)
        with self.assertRaisesRegex(ConstraintError, "not positive definite"):
            km.rebuild(doubled, width)


@unittest.skipIf(GPU or FP32, "CPU float64 unit tests")
class RunTests(unittest.TestCase):
    def run_case(self, config, directory, **kwargs):
        return C.run_case(config, directory, quiet=True, **kwargs)

    def test_default_is_off_and_leaves_no_correction_attached(self):
        config = C.validate_config({})
        self.assertFalse(config["ib"]["k_matrix"]["enabled"])
        runtime = C.build_runtime(case(k_matrix=False))
        self.assertIsNone(runtime.driver.ib_kmatrix)

    def test_fixed_sphere_has_no_slip_after_every_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_case(case("fixed"), Path(temporary) / "run")
            history = result.history
            metadata = json.loads((Path(temporary) / "run" / "metadata.json").read_text())
        self.assertEqual(len(history), 3)
        for row in history:
            self.assertLess(row["slip_rms"], 1e-10)
            self.assertEqual(row["ib_force_updates"], 4)
            self.assertLess(row["force_balance_relerr"], 1e-9)
        summary = metadata["k_matrix"]
        self.assertEqual(summary["rebuilds"], 1)                    # fixed body: K built once
        self.assertEqual(summary["solves"], 12)
        self.assertEqual(summary["iteration_caps"], 0)

    def test_moving_sphere_rebuilds_k_every_step_and_restarts_identically(self):
        config = case("osc", steps=4)
        with tempfile.TemporaryDirectory() as temporary:
            full = self.run_case(config, Path(temporary) / "full")
            metadata = json.loads((Path(temporary) / "full" / "metadata.json").read_text())
            first = copy.deepcopy(config); first["time"]["steps"] = 2
            first["output"]["checkpoint_every"] = 2
            target = Path(temporary) / "split"
            self.run_case(first, target)
            second = copy.deepcopy(first); second["time"]["steps"] = 4
            resumed = self.run_case(second, target, resume=target / "checkpoint.npz")
        self.assertEqual(metadata["k_matrix"]["rebuilds"], 4)
        for row in full.history:
            self.assertLess(row["slip_rms"], 1e-10)
        np.testing.assert_array_equal(resumed.s, full.s)
        np.testing.assert_array_equal(resumed.ib_force_L, full.ib_force_L)
        np.testing.assert_array_equal(resumed.body_force, full.body_force)

    def test_patches_enforce_the_patch_sums(self):
        config = case("fixed", patch_size=4, steps=2)
        runtime = C.build_runtime(C.validate_config(config))
        km = runtime.driver.ib_kmatrix
        self.assertLess(km_patches(km), runtime.reference_markers.shape[0] / 2)
        s, s_old, vc = C._initial_state(runtime)
        for _ in range(2):
            s, vc, s_old = runtime.driver.step(s, s_old, runtime.dt, dt_old=runtime.dt, inner=4)
        slip = runtime.driver.ib_vel - runtime.ib.interp(vc[:runtime.core.ncell], runtime.driver.ib_markers)
        sums = km.patch_sum(slip)
        self.assertLess(float(np.abs(sums).max()), 1e-10)
        self.assertGreater(float(np.abs(slip).max()), 1e-6)         # markers inside a patch may slip

    def test_explicit_and_constrained_agree_on_the_late_force(self):
        # Different force updates, the same converged discrete problem: after the start-up
        # transient of the explicit update the body force must be close.
        steps = 40
        with tempfile.TemporaryDirectory() as temporary:
            explicit = self.run_case(case("fixed", k_matrix=False, steps=steps), Path(temporary) / "e")
            constrained = self.run_case(case("fixed", steps=steps), Path(temporary) / "c")
        a = np.mean([row["body_force_x"] for row in explicit.history[-8:]])
        b = np.mean([row["body_force_x"] for row in constrained.history[-8:]])
        self.assertLess(abs(a - b) / abs(b), 0.05)

    def test_invalid_combinations_are_rejected(self):
        bad = case(); bad["ib"]["force_mode"] = "direct"
        with self.assertRaisesRegex(C.CaseConfigError, "k_matrix"):
            C.validate_config(bad)
        bad = case(); bad["runtime"] = {"device": "cpu", "precision": "float32"}
        with self.assertRaisesRegex(C.CaseConfigError, "float64"):
            C.validate_config(bad)
        bad = case(patch_size=3)
        with self.assertRaisesRegex(C.CaseConfigError, "patch_size"):
            C.validate_config(bad)


def km_patches(km):
    return km.n_patches


class PatchTests(unittest.TestCase):
    def test_structured_sheet_gives_two_by_two_patches_with_paired_rim(self):
        n = 10
        x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(n * n)))
        ids = np.arange(n * n).reshape(n, n)
        a, b, c, d = ids[:-1, :-1].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel(), ids[:-1, 1:].ravel()
        triangles = np.concatenate((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
        patch, weight, info = make_patches(points, triangles)
        self.assertEqual(info["rim_markers"], 4 * (n - 1))
        self.assertEqual(sorted(np.unique(patch).tolist()), list(range(info["patches"])))
        interior = np.ones((n, n), bool); interior[[0, -1], :] = False; interior[:, [0, -1]] = False
        members = np.bincount(patch[interior.ravel()], minlength=info["patches"])
        self.assertEqual(int((members == 4).sum()), ((n - 2) // 2) ** 2)    # 8 x 8 interior -> 16 blocks
        for k in np.flatnonzero(members == 4):
            block = points[(patch == k)]
            self.assertAlmostEqual(float(np.ptp(block[:, 0])), 1.0)
            self.assertAlmostEqual(float(np.ptp(block[:, 1])), 1.0)
        rim_members = np.bincount(patch[~interior.ravel()], minlength=info["patches"])
        self.assertLessEqual(int(rim_members.max()), 2)
        np.testing.assert_array_equal(weight, np.ones(n * n))

    def test_closed_sphere_has_no_rim_and_compact_patches(self):
        from scipy.spatial import ConvexHull
        count = 201
        i = np.arange(count)
        z = 1 - 2 * (i + 0.5) / count
        angle = i * np.pi * (3 - np.sqrt(5))
        points = 0.25 * np.column_stack((np.sqrt(1 - z * z) * np.cos(angle), np.sqrt(1 - z * z) * np.sin(angle), z))
        patch, _, info = make_patches(points, ConvexHull(points).simplices)
        self.assertEqual(info["rim_markers"], 0)
        self.assertLessEqual(max(info["markers_per_patch"]), 4)
        self.assertLess(info["patches"], count / 2)
        self.assertLessEqual(info["largest_patch_diameter_spacings"], 2.3 + 1e-9)
        single, _, info1 = make_patches(points, ConvexHull(points).simplices, size=1)
        self.assertEqual(info1["patches"], count)

    def test_point_cloud_without_triangles_uses_the_distance_graph(self):
        n = 8
        x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(n * n)))
        patch, _, info = make_patches(points)
        self.assertEqual(info["neighbours"], "distance graph")
        self.assertGreater(info["rim_markers"], 0)
        self.assertLess(info["patches"], n * n)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(GPU or FP32, "CPU float64 unit tests")
class StoppingFloorTests(unittest.TestCase):
    def test_floor_is_off_by_default_and_derived_from_the_reference_speed(self):
        config = C.validate_config(case("osc", k_matrix=False))
        self.assertEqual(config["solver"]["cg_reference_speed"], 0.0)
        runtime = C.build_runtime(config)
        self.assertEqual(runtime.driver.cg_xnorm_min, 0.0)
        config = case("osc", k_matrix=True)
        config["solver"]["cg_reference_speed"] = 0.5
        runtime = C.build_runtime(C.validate_config(config))
        core = runtime.core
        uniform = 0.5 * asnumpy(core.epos[:, 1]) * asnumpy(core.te[:, 2])
        self.assertAlmostEqual(runtime.driver.cg_xnorm_min, float(uniform @ uniform), places=12)
        self.assertEqual(runtime.driver.ib_kmatrix.xnorm_min, runtime.driver.cg_xnorm_min)

    def test_floor_reduces_iterations_and_changes_the_force_little(self):
        # Sphere oscillating in fluid at rest inside a closed box (the benchmark set-up, D = 8h):
        # |psi|^2 is small, so without the floor the stopping test is purely relative.
        def quiescent(speed):
            return {
                "schema_version": 1, "name": "floor", "case": {"kind": "oscillating_sphere"},
                "mesh": {"kind": "sphere_concentric", "cells_per_diameter": 8},
                "physics": {"reynolds": 0.2, "density": 1.0, "inflow_speed": 0.0},
                "time": {"dt": 0.0062333187571226, "steps": 4, "inner_iterations": 4},
                "ib": {"radius": 0.25, "center": [0.0, 0.0, 0.0], "marker_spacing": 0.0625,
                       "motion": {"kind": "oscillation", "axis": [1.0, 0.0, 0.0],
                                  "amplitude": 0.0055555555555555558, "frequency": 0.57295779513082334,
                                  "phase": -1.5707963267948966}},
                "solver": {"cg_check_interval": 1, "cg_max_iterations": 1000, "cg_reference_speed": speed},
                "output": {"sample_every": 1, "checkpoint_every": 0, "save_field": False},
            }
        with tempfile.TemporaryDirectory() as temporary:
            strict = C.run_case(quiescent(0.0), Path(temporary) / "a", quiet=True)
            floored = C.run_case(quiescent(0.02), Path(temporary) / "b", quiet=True)
        iterations = lambda result: sum(row["helmholtz_iterations"] for row in result.history)
        self.assertLess(iterations(floored), 0.6 * iterations(strict))
        a = np.array([row["body_force_x"] for row in strict.history])
        b = np.array([row["body_force_x"] for row in floored.history])
        self.assertLess(float(np.max(np.abs(a - b)) / np.max(np.abs(a))), 1e-4)

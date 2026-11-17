"""Tests of the factored Helmholtz operator, the threaded CSR backend and the in-place CG."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix, random as sparse_random

from backend import GPU, FP32
import ib_case as C
from ib_cpu_sparse import ThreadedCSR, execution_identity, validate_options
from ib_sparse import FactoredOperator, build_A, build_factored


def small_case(**solver):
    return {
        "schema_version": 1, "name": "operator", "case": {"kind": "fixed_sphere"},
        "mesh": {"kind": "sphere_wake", "cells_per_diameter": 8},
        "physics": {"reynolds": 100.0, "density": 1.0, "inflow_speed": 1.0},
        "time": {"dt": 0.025, "steps": 2, "inner_iterations": 4},
        "ib": {"radius": 0.25, "center": [0.0, 0.0, 0.0], "marker_spacing": 0.0625},
        "solver": {"sparse": True, **solver},
        "output": {"sample_every": 1, "checkpoint_every": 0, "save_field": False},
    }


class ThreadedCSRTests(unittest.TestCase):
    def test_product_equals_scipy_bit_for_bit(self):
        rng = np.random.default_rng(0)
        for dtype in (np.float64, np.float32):
            for index in (np.int32, np.int64):
                matrix = sparse_random(301, 257, density=0.05, random_state=3, format="csr", dtype=dtype)
                matrix = csr_matrix((matrix.data, matrix.indices.astype(index), matrix.indptr.astype(index)),
                                    shape=matrix.shape)
                vector = rng.standard_normal(257).astype(dtype)
                expected = matrix @ vector
                for threads in (1, 2, 4, 7):
                    with self.subTest(dtype=dtype.__name__, index=index.__name__, threads=threads):
                        operator = ThreadedCSR(matrix, threads)
                        np.testing.assert_array_equal(operator @ vector, expected)
                        self.assertEqual(operator.shape, matrix.shape)
                        self.assertEqual(operator.nnz, matrix.nnz)

    def test_empty_rows_and_more_threads_than_rows(self):
        matrix = csr_matrix(([2.0, -1.0, 4.0], ([0, 3, 3], [1, 0, 2])), shape=(5, 3))
        vector = np.array([1.0, 2.0, 3.0])
        for threads in (1, 3, 16):
            np.testing.assert_array_equal(ThreadedCSR(matrix, threads) @ vector, matrix @ vector)
        empty = csr_matrix((4, 3), dtype=np.float64)
        np.testing.assert_array_equal(ThreadedCSR(empty, 2) @ vector, np.zeros(4))

    def test_wrong_vectors_are_rejected(self):
        matrix = csr_matrix(np.eye(4))
        operator = ThreadedCSR(matrix, 2)
        for bad in (np.ones(3), np.ones(4, dtype=np.float32), np.ones((4, 1)), [1.0, 1.0, 1.0, 1.0]):
            with self.assertRaises(ValueError):
                operator @ bad
        with self.assertRaises(ValueError):
            ThreadedCSR(np.eye(4), 2)

    def test_options_and_identity(self):
        validate_options("threads", 8)
        validate_options("threads", 2, precision="float32")
        for backend, threads, extra in (("threads", 0, {}), ("threads", 2, {"device": "cuda"}),
                                        ("threads", 2, {"sparse": False}), ("threads", 257, {}), ("scipy", 2, {})):
            with self.assertRaises(ValueError):
                validate_options(backend, threads, **extra)
        config = C.validate_config({"solver": {"cpu_sparse_backend": "threads", "cpu_sparse_threads": 3}})
        identity = execution_identity(config)
        self.assertEqual((identity["backend"], identity["threads"], identity["actual_threads"]), ("threads", 3, 3))
        self.assertIn("scipy_extension", identity["implementation_sha256"])
        default = execution_identity(C.validate_config({}))
        self.assertNotEqual(identity, default)                     # a checkpoint is not resumed across backends


@unittest.skipIf(GPU or FP32, "CPU float64 unit tests")
class OperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = C.build_runtime(C.validate_config(small_case(operator="assembled")))
        cls.driver, cls.core = cls.runtime.driver, cls.runtime.core
        cls.kw = dict(alpd=cls.driver.ad, visc=cls.driver.visc, rou=cls.driver.rou, dt=cls.runtime.dt)

    def test_default_operator_is_factored_and_assembled_is_selectable(self):
        self.assertEqual(C.validate_config({})["solver"]["operator"], "factored")
        self.assertEqual(self.driver.sparse_operator, "assembled")
        with self.assertRaisesRegex(C.CaseConfigError, "operator"):
            C.validate_config({"solver": {"operator": "dense"}})

    def test_factored_operator_is_the_assembled_one_up_to_round_off(self):
        assembled = build_A(self.core, **self.kw)
        factored = build_factored(self.core, **self.kw)
        self.assertIsInstance(factored, FactoredOperator)
        self.assertEqual(factored.shape, assembled.shape)
        self.assertLess(factored.nnz, 0.3 * assembled.nnz)
        rng = np.random.default_rng(1)
        for _ in range(3):
            vector = rng.standard_normal(self.core.nedge)
            a, b = assembled @ vector, factored @ vector
            self.assertLess(float(np.max(np.abs(a - b)) / np.max(np.abs(a))), 1e-12)
            matrix_free = self.core.flow_mat(vector, **self.kw)
            self.assertLess(float(np.max(np.abs(matrix_free - b)) / np.max(np.abs(matrix_free))), 1e-12)

    def test_factored_operator_without_the_viscous_block(self):
        factored = build_factored(self.core, alpd=0.0)
        self.assertIsNone(factored.viscous)
        vector = np.random.default_rng(2).standard_normal(self.core.nedge)
        reference = self.core.flow_mat(vector, alpd=0.0)
        self.assertLess(float(np.max(np.abs(factored @ vector - reference)) / np.max(np.abs(reference))), 1e-12)

    def test_helm_matvec_masks_dirichlet_edges_for_every_operator_form(self):
        vector = np.random.default_rng(3).standard_normal(self.core.nedge)
        reference = self.driver.helm_matvec(vector, self.runtime.dt)
        self.assertTrue(np.all(reference[self.driver.dir_mask] == 0.0))
        self.assertGreater(int(self.driver.dir_mask.sum()), 0)
        for operator, backend, threads in (("factored", "scipy", 1), ("factored", "threads", 3), ("assembled", "threads", 2)):
            self.driver.enable_sparse(self.runtime.dt, operator=operator, cpu_backend=backend, cpu_threads=threads)
            result = self.driver.helm_matvec(vector, self.runtime.dt)
            self.assertLess(float(np.max(np.abs(result - reference)) / np.max(np.abs(reference))), 1e-12)
        self.driver.enable_sparse(self.runtime.dt, operator="assembled")

    def test_threads_backend_reproduces_scipy_runs_bit_for_bit(self):
        results = {}
        for name, solver in (("scipy", {}), ("threads", {"cpu_sparse_backend": "threads", "cpu_sparse_threads": 3})):
            runtime = C.build_runtime(C.validate_config(small_case(**solver)))
            s, s_old, vc = C._initial_state(runtime)
            for _ in range(2):
                s, vc, s_old = runtime.driver.step(s, s_old, runtime.dt, dt_old=runtime.dt, inner=4)
            results[name] = (s.copy(), runtime.driver.ib_force_L.copy(), runtime.driver.helmholtz_iterations_last_step)
        np.testing.assert_array_equal(results["scipy"][0], results["threads"][0])
        np.testing.assert_array_equal(results["scipy"][1], results["threads"][1])
        self.assertEqual(results["scipy"][2], results["threads"][2])

    def test_assembled_and_factored_steps_agree_to_solver_round_off(self):
        results = {}
        for operator in ("assembled", "factored"):
            runtime = C.build_runtime(C.validate_config(small_case(operator=operator)))
            s, s_old, vc = C._initial_state(runtime)
            iterations = []
            for _ in range(2):
                s, vc, s_old = runtime.driver.step(s, s_old, runtime.dt, dt_old=runtime.dt, inner=4)
                iterations.append(runtime.driver.helmholtz_iterations_last_step)
            results[operator] = (s.copy(), vc.copy(), iterations)
        self.assertEqual(results["assembled"][2], results["factored"][2])
        velocity = results["assembled"][1]
        self.assertLess(float(np.max(np.abs(velocity - results["factored"][1])) / np.max(np.abs(velocity))), 1e-8)

    def test_step_end_residual_is_closer_to_a_gradient_than_the_last_inner_one(self):
        # The curl-curl solve leaves dt * grad(p) in the cell residual.  The residual kept from the
        # start of the last inner iteration still contains what that iteration removed (with the
        # K-matrix correction also the last force update); the residual of the finished step does not.
        # Closed box, fluid at rest (the benchmark set-up at D = 8h): with inflow/outflow boundaries
        # the boundary update keeps a residual near the outer boundary that no inner iteration removes.
        def circulation(step_end):
            config = {
                "schema_version": 1, "name": "residual", "case": {"kind": "oscillating_sphere"},
                "mesh": {"kind": "sphere_concentric", "cells_per_diameter": 8},
                "physics": {"reynolds": 0.2, "density": 1.0, "inflow_speed": 0.0},
                "time": {"dt": 0.0062333187571226, "steps": 2, "inner_iterations": 2},
                "ib": {"radius": 0.25, "center": [0.0, 0.0, 0.0], "marker_spacing": 0.0625,
                       "k_matrix": {"enabled": True},
                       "motion": {"kind": "oscillation", "axis": [1.0, 0.0, 0.0],
                                  "amplitude": 0.0055555555555555558, "frequency": 0.57295779513082334,
                                  "phase": -1.5707963267948966}},
                "solver": {"cg_check_interval": 1, "cg_max_iterations": 1000},
                "output": {"residual_state": "step_end" if step_end else "last_inner"},
            }
            runtime = C.build_runtime(C.validate_config(config))
            driver = runtime.driver
            self.assertEqual(driver.residual_step_end, step_end)
            s, s_old, vc = C._initial_state(runtime)
            for _ in range(2):
                s, vc, s_old = driver.step(s, s_old, runtime.dt, dt_old=runtime.dt, inner=2)
            edge = runtime.core.rot(runtime.core.integr_c2f(driver.last_momentum_residual))
            edge[driver.dir_mask] = 0.0
            return float(np.linalg.norm(edge))
        last_inner, step_end = circulation(False), circulation(True)
        self.assertLess(step_end, 0.2 * last_inner)

    def test_residual_state_defaults(self):
        explicit = C.build_runtime(C.validate_config(small_case()))
        self.assertFalse(explicit.driver.residual_step_end)             # as before
        config = small_case(); config["output"]["residual_state"] = "step_end"
        self.assertTrue(C.build_runtime(C.validate_config(config)).driver.residual_step_end)
        with self.assertRaisesRegex(C.CaseConfigError, "residual_state"):
            bad = small_case(); bad["output"]["residual_state"] = "final"
            C.validate_config(bad)


if __name__ == "__main__":
    unittest.main()

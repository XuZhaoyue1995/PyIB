"""K-matrix force correction: momentum and no-slip solved together (optional).

Port of the constrained Picard iteration of IBFast (``src/ibf_pcg.f90``,
``ib_update = 'pcg'``).  The default explicit update of PyIB corrects the
Lagrangian force once per step from the slip left by the first inner solve and
accumulates it over the steps.  Here every inner iteration solves the symmetric
saddle-point system

    A dpsi - B^T lam = b            b: masked edge residual with the held force
    B dpsi           = d            d: patch sums of (V_body - J u)

for the stream-function correction ``dpsi`` and the multiplier ``lam``; the held
marker force changes by ``w_m lam_a / (dt ds_m^3)``.  ``B = P J interp_f2c curl``
(patch sum, delta interpolation, face-to-cell interpolation, curl) is assembled
as a sparse matrix for the current marker positions and
``K = B G^-1 B^T`` (``G^-1`` = the diagonal Helmholtz preconditioner of the
driver) is formed and factorised: once for a fixed body, every step for a moving
one.  The system is solved by projected conjugate gradients with the exact
constraint preconditioner [G, B^T; B, 0] (Gould, Hribar & Nocedal 2001): a
feasible start ``G^-1 B^T K^-1 d``, CG on the momentum equation inside the
subspace ``B p = 0``, one final re-projection and the multiplier from the
recomputed residual.

Constraint patches (optional): the multiplier is constant on patches of markers
and no-slip is enforced for the weighted patch sums.  One marker per patch
(default) is the marker-by-marker constraint.

The method needs float64, the marker kernel width equal to the lattice spacing
(``B^T`` is then the exact adjoint of ``B``) and independent constraints (``K``
positive definite: marker spacing not much below one cell).
"""
from __future__ import annotations

import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from backend import xp, GPU, FP32, asnumpy


class ConstraintError(RuntimeError):
    """The constrained system cannot be set up or solved."""


class KMatrixCorrection:
    """Constrained force update attached to a :class:`ib_driver_np.Driver`."""

    def __init__(self, driver, ib, *, patch_ids=None, patch_weights=None,
                 tolerance=1.0e-6, tolerance_absolute=1.0e-10, max_iterations=5000,
                 xnorm_min=0.0, stagnation_window=100, k_solver="auto",
                 inverse_limit_bytes=None):
        if FP32:
            raise ConstraintError("the K-matrix correction requires float64")
        if k_solver not in {"auto", "inverse", "cholesky"}:
            raise ValueError("k_solver must be 'auto', 'inverse' or 'cholesky'")
        self.driver = driver
        self.ib = ib
        self.tolerance = float(tolerance)
        self.tolerance_absolute = float(tolerance_absolute)
        self.max_iterations = int(max_iterations)
        self.xnorm_min = float(xnorm_min)
        self.stagnation_window = int(stagnation_window)
        self.k_solver = k_solver
        self.inverse_limit_bytes = (
            (512 if GPU else 2048) * 1024 * 1024 if inverse_limit_bytes is None
            else int(inverse_limit_bytes)
        )
        if not self.tolerance > 0.0 or not self.tolerance_absolute >= 0.0:
            raise ValueError("constraint tolerances must be positive")
        if self.max_iterations <= 0 or self.stagnation_window <= 0:
            raise ValueError("constraint iteration limits must be positive")
        if not np.isfinite(self.xnorm_min) or self.xnorm_min < 0.0:
            raise ValueError("xnorm_min must be finite and non-negative")
        if abs(ib.hh - ib.dx) > 1.0e-12 * ib.dx:
            raise ConstraintError("the K-matrix correction requires kernel width == lattice spacing")
        self._patch_ids_in = None if patch_ids is None else np.asarray(patch_ids)
        self._patch_weights_in = None if patch_weights is None else np.asarray(patch_weights, dtype=float)
        self._setup_operators()
        self._markers = None            # host snapshot of the positions K was built for
        self._ds = None
        self.n_markers = 0
        self.n_patches = 0
        self.n3 = 0
        # cumulative bookkeeping (host scalars)
        self.rebuild_count = 0
        self.rebuild_seconds = 0.0
        self.total_iterations = 0
        self.stagnation_count = 0
        self.iteration_cap_count = 0
        self.solve_count = 0
        self.records_last_step = []
        self.k_info = {}

    # ------------------------------------------------------------------ setup
    def _setup_operators(self):
        """interp_f2c . curl restricted to the rows of the lattice cells (host CSR)."""
        from ib_sparse import _curl_matrix, _interp_f2c_matrix
        core = self.driver.core
        ids = np.asarray(asnumpy(self.ib.index_cell))
        block = np.unique(ids[ids >= 0])
        self._nblk = int(block.size)
        self._cell2blk = np.full(core.ncell, -1, dtype=np.int64)
        self._cell2blk[block] = np.arange(self._nblk)
        rows = np.concatenate([d * core.ncell + block for d in range(3)])
        interp = _interp_f2c_matrix(core)[rows]                 # (3 nblk, nface)
        pc = (interp @ _curl_matrix(core)).tocsr()              # (3 nblk, nedge)
        free = (~np.asarray(asnumpy(self.driver.dir_mask))).astype(np.float64)
        pc = (pc @ sp.diags(free)).tocsr()                      # no entries on Dirichlet edges
        pc.eliminate_zeros()
        self._pc = pc
        self._prec_host = np.asarray(asnumpy(self.driver.prec), dtype=np.float64)
        # exact adjointness of B needs cell volume == hh^3 on the lattice cells
        volume = 1.0 / np.asarray(asnumpy(core.vol_ci), dtype=np.float64)[block]
        self.volume_defect = float(np.max(np.abs(volume / self.ib.hh ** 3 - 1.0)))
        if self.volume_defect > 1.0e-9:
            raise ConstraintError(
                "lattice cell volumes differ from the kernel volume; B^T would not be the adjoint of B"
            )

    def _set_patches(self, count):
        if self._patch_ids_in is None:
            grp = np.arange(count, dtype=np.int64)
        else:
            grp = np.asarray(self._patch_ids_in)
            if grp.shape != (count,) or grp.dtype.kind not in "iu":
                raise ConstraintError("patch ids: one integer per marker expected")
            grp = grp.astype(np.int64)
            if grp.min() != 0 or np.any(np.bincount(grp) == 0):
                raise ConstraintError("patch ids must be contiguous and start at 0")
        if self._patch_weights_in is None:
            gw = np.ones(count)
        else:
            gw = np.asarray(self._patch_weights_in, dtype=np.float64)
            if gw.shape != (count,) or not np.all(np.isfinite(gw)) or np.any(gw <= 0.0):
                raise ConstraintError("patch weights: one positive number per marker expected")
        self._grp_host, self._gw_host = grp, gw
        self.grp = xp.asarray(grp)
        self.gw = xp.asarray(gw)
        self.n_markers = int(count)
        self.n_patches = int(grp.max()) + 1
        self.n3 = 3 * self.n_patches

    # ---------------------------------------------------------------- rebuild
    def needs_rebuild(self, markers, ds):
        if self._markers is None:
            return True
        m = np.asarray(asnumpy(markers)); w = np.asarray(asnumpy(ds))
        return (m.shape != self._markers.shape or not np.array_equal(m, self._markers)
                or not np.array_equal(w, self._ds))

    def rebuild(self, markers, ds):
        """Assemble B and K = B G^-1 B^T for these marker positions and factorise K."""
        start = time.perf_counter()
        m_host = np.asarray(asnumpy(markers), dtype=np.float64)
        if self.n_markers != m_host.shape[0]:
            self._set_patches(m_host.shape[0])
        cell, weight = self.ib._cells_weights(markers)           # (M, 125) on the backend
        cell = np.asarray(asnumpy(cell)); weight = np.asarray(asnumpy(weight), dtype=np.float64)
        marker, slot = np.nonzero(weight)
        local = self._cell2blk[cell[marker, slot]]
        if np.any(local < 0):
            raise ConstraintError("a marker stencil leaves the finest lattice block")
        value = weight[marker, slot] * self._gw_host[marker] * self.ib.hh ** 3
        patch = self._grp_host[marker]
        rows = np.concatenate([3 * patch + d for d in range(3)])
        cols = np.concatenate([d * self._nblk + local for d in range(3)])
        jw = sp.coo_matrix((np.tile(value, 3), (rows, cols)),
                           shape=(self.n3, 3 * self._nblk)).tocsr()
        b_host = (jw @ self._pc).tocsr()
        b_host.sum_duplicates(); b_host.sort_indices()
        bt_host = b_host.T.tocsr()
        bt_host.sort_indices()
        k = (b_host.multiply(self._prec_host[None, :]).tocsr() @ bt_host).toarray()
        k = 0.5 * (k + k.T)
        try:
            factor = sla.cholesky(k, lower=True, check_finite=True)
        except (sla.LinAlgError, ValueError) as exc:
            raise ConstraintError(
                "K = B G^-1 B^T is not positive definite (dependent constraints: markers "
                "much closer than one cell, or patches without support)"
            ) from exc
        pivots = np.diag(factor)
        use_inverse = self.k_solver == "inverse" or (
            self.k_solver == "auto" and 8 * self.n3 * self.n3 <= self.inverse_limit_bytes
        )
        if use_inverse:
            inverse, info = sla.lapack.dpotri(factor, lower=1)
            if info != 0:
                raise ConstraintError(f"inversion of K failed (LAPACK dpotri info {info})")
            inverse = np.tril(inverse) + np.tril(inverse, -1).T
            self._k_inverse = xp.asarray(inverse)
            self._k_factor = None
        else:
            self._k_inverse = None
            self._k_factor = factor
        if GPU:
            import cupyx.scipy.sparse as csp
            self.B = csp.csr_matrix(b_host)
            self.BT = csp.csr_matrix(bt_host)
        else:
            self.B, self.BT = b_host, bt_host
        self._markers = m_host.copy()
        self._ds = np.asarray(asnumpy(ds)).copy()
        self.rebuild_count += 1
        elapsed = time.perf_counter() - start
        self.rebuild_seconds += elapsed
        self.k_info = {
            "markers": self.n_markers, "patches": self.n_patches, "constraints": self.n3,
            "b_nonzeros": int(b_host.nnz), "k_nonzeros": int(np.count_nonzero(k)),
            "k_pivot_min": float(pivots.min()), "k_pivot_max": float(pivots.max()),
            "k_solver": "inverse" if use_inverse else "cholesky",
            "last_rebuild_seconds": float(elapsed),
        }
        return self

    # ---------------------------------------------------------------- algebra
    def k_solve(self, rhs):
        if self._k_inverse is not None:
            return self._k_inverse @ rhs
        host = sla.cho_solve((self._k_factor, True), np.asarray(asnumpy(rhs)), check_finite=False)
        return xp.asarray(host)

    def patch_sum(self, marker_vectors):
        """(M, 3) marker vectors -> (3 n_patches,) weighted patch sums, patch-major."""
        weighted = marker_vectors * self.gw[:, None]
        if self.n_patches == self.n_markers:
            return weighted.reshape(-1)
        out = xp.zeros((self.n_patches, 3), dtype=weighted.dtype)
        for d in range(3):
            out[:, d] = xp.bincount(self.grp, weighted[:, d], minlength=self.n_patches)
        return out.reshape(-1)

    def _matvec(self, vector, dt):
        return self.driver.helm_matvec(vector, dt)

    def _project(self, residual):
        """(G^-1 - G^-1 B^T K^-1 B G^-1) r"""
        scaled = self.driver.prec * residual
        return scaled - self.driver.prec * (self.BT @ self.k_solve(self.B @ scaled))

    def solve(self, rhs_edge, rhs_constraint, psi, dt):
        """Projected PCG.  Returns (dpsi, lam, iterations, info)."""
        drv = self.driver
        prec = drv.prec
        b = rhs_edge.copy()
        b[drv.dir_mask] = 0.0
        xnorm_raw = float(psi.dot(psi)) + 1.0e-16
        xnorm = max(xnorm_raw, self.xnorm_min)
        x = prec * (self.BT @ self.k_solve(rhs_constraint))      # feasible start
        r = b - self._matvec(x, dt)
        g = self._project(r)
        p = g.copy()
        rg = float(r.dot(g)); rg0 = rg; rgn = rg
        err_tot = xnorm * self.tolerance_absolute ** 2 + self.tolerance ** 2 * abs(rg0)
        iterations = 0; res = 1.0; best = np.inf; it_best = 0
        stagnated = False; capped = False
        if not np.isfinite(rg0):
            raise FloatingPointError("non-finite residual at the start of the constrained solve")
        if abs(rg0) > err_tot:
            capped = True
            for iterations in range(1, self.max_iterations + 1):
                ap = self._matvec(p, dt)
                pap = float(p.dot(ap))
                if pap == 0.0 or not np.isfinite(pap):
                    capped = False
                    break
                alpha = rg / pap
                x += alpha * p
                r -= alpha * ap
                g = self._project(r)
                rgn = float(r.dot(g))
                res = float(np.sqrt(abs(rgn) / abs(rg0)))
                if abs(rgn) <= err_tot:
                    capped = False
                    break
                if res < best:
                    best = res; it_best = iterations
                elif iterations - it_best > self.stagnation_window or res > 100.0 * best:
                    stagnated = True; capped = False
                    break
                p = g + (rgn / rg) * p
                rg = rgn
        # re-projection onto the constraint, residual recomputed, multiplier
        defect = rhs_constraint - self.B @ x
        x += prec * (self.BT @ self.k_solve(defect))
        r = b - self._matvec(x, dt)
        lam = -self.k_solve(self.B @ (prec * r))
        info = {
            "iterations": int(iterations), "relative_residual": float(res),
            "tolerance_ratio": float(np.sqrt(abs(rgn) / err_tot)) if err_tot > 0.0 else 0.0,
            "stagnated": bool(stagnated), "iteration_cap": bool(capped),
            "psi_norm2": float(xnorm_raw),
        }
        return x, lam, int(iterations), info

    # ----------------------------------------------------------- driver hooks
    def begin_step(self):
        """Call once per step after the body was moved to the new time level."""
        drv = self.driver
        self.records_last_step = []
        if self.needs_rebuild(drv.ib_markers, drv.ib_ds):
            self.rebuild(drv.ib_markers, drv.ib_ds)

    def inner_iteration(self, res_e, s, vc, dt):
        """One constrained Picard iteration.  Updates the held force of the driver
        (Lagrangian and spread) and returns the new stream function."""
        drv = self.driver; ib = self.ib
        ncell = drv.core.ncell
        stencil = ib._cells_weights(drv.ib_markers)
        slip = drv.ib_vel - ib._interp_stencil(vc[:ncell], stencil)
        d = self.patch_sum(slip)
        x, lam, iterations, info = self.solve(res_e, d, s, dt)
        if not np.isfinite(info["relative_residual"]):
            raise FloatingPointError("non-finite residual in the constrained solve")
        lam3 = lam.reshape(self.n_patches, 3)
        increment = (self.gw / (dt * drv.ib_ds ** 3))[:, None] * lam3[self.grp]
        drv.ib_force_increment_L = increment
        drv.ib_force_L = drv.ib_force_L + increment
        new_force_E = ib._spread_stencil(drv.ib_force_L, stencil, drv.ib_ds)
        drv.ib_force_increment_E = new_force_E - drv.ib_force_E
        drv.ib_force_E = new_force_E
        drv.ib_force_update_count += 1
        drv.ib_force_updates_last_step += 1
        self.total_iterations += iterations
        self.solve_count += 1
        self.stagnation_count += int(info["stagnated"])
        self.iteration_cap_count += int(info["iteration_cap"])
        info["constraint_rhs_rms"] = float(xp.sqrt(xp.mean(d * d)))
        info["multiplier_norm"] = float(xp.sqrt(lam.dot(lam)))
        self.records_last_step.append(info)
        return s + x, iterations, info

    def summary(self):
        """JSON-safe description for run metadata."""
        return {
            "method": "projected PCG on [A, -B^T; B, 0] with K = B G^-1 B^T (IBFast ib_update='pcg')",
            "tolerance": self.tolerance, "tolerance_absolute": self.tolerance_absolute,
            "max_iterations": self.max_iterations, "xnorm_min": self.xnorm_min,
            "stagnation_window": self.stagnation_window,
            "lattice_volume_defect": self.volume_defect,
            "rebuilds": int(self.rebuild_count), "rebuild_seconds": float(self.rebuild_seconds),
            "solves": int(self.solve_count), "iterations": int(self.total_iterations),
            "stagnations": int(self.stagnation_count),
            "iteration_caps": int(self.iteration_cap_count),
            **self.k_info,
        }

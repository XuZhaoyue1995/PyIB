"""NumPy/CuPy driver for boundary conditions, CG solves and IB time integration.

The stream-function formulation, time discretization and force lifecycle are
covered by operator, manufactured-solution and trajectory regression checks.
"""
import numpy as np
import os

from backend import xp, GPU, FP32, asreal, real, asnumpy   # numpy/cupy; FP32; real = working dtype
from ib_core_np import IBCore

# cupy's default memory pool retains freed blocks AND fragments during the cold
# first-step CG (1000s of iterations) -> peak balloons to ~8GB for a working set of
# ~3GB.  Periodically releasing free blocks keeps the peak ~= the live set, which is
# what lets a finer mesh fit on a small GPU.  No-op on CPU.
if GPU:
    import cupy as _cp
    _mempool = _cp.get_default_memory_pool()
    def free_gpu_pool():
        _mempool.free_all_blocks()
else:
    def free_gpu_pool():
        pass


# ---------------------------------------------------------------------------
# [1] CG elliptic solver  (Helm_s/CG_m with flow_mat as the matrix-free matvec)
# ---------------------------------------------------------------------------
def apply_A(core, pp, dir_mask, **kw):
    """Elliptic operator A = flow_mat + DirBC_s (zero residual at Dirichlet edges)."""
    w = core.flow_mat(pp, **kw)
    w[dir_mask] = 0.0
    return w


def cg_solve(core, b, dir_mask, tol=1e-10, maxiter=3000, prec=None, **kw):
    """Preconditioned CG for A x = b on the non-Dirichlet edge subspace.
    Returns (x, iters, rel_residual).  prec = diagonal preconditioner (None -> identity)."""
    if prec is None:
        prec = np.ones(core.nedge)
    x = np.zeros(core.nedge)
    b = b.copy(); b[dir_mask] = 0.0
    r = b - apply_A(core, x, dir_mask, **kw)        # x0 = 0
    r[dir_mask] = 0.0
    z = r * prec
    p = z.copy()
    eta = z.dot(r)
    eta0 = max(eta, 1e-300)
    last = 1.0
    m = 0
    for m in range(1, maxiter + 1):
        w = apply_A(core, p, dir_mask, **kw)
        denom = p.dot(w)
        if denom == 0.0:
            break
        alf = eta / denom
        x += alf * p
        r -= alf * w
        z = r * prec
        eta_new = z.dot(r)
        last = abs(r.dot(r)) / (b.dot(b) + 1e-300)
        if last < tol * tol:
            break
        p = z + (eta_new / eta) * p
        eta = eta_new
    return x, m, np.sqrt(last)


def boundary_edge_mask(core, lo=-1.0, hi=1.0, eps=1e-9):
    """Edges on the domain box faces (|coord| at lo/hi) -> Dirichlet for the test."""
    e = core.epos
    on = np.zeros(core.nedge, dtype=bool)
    for k in range(3):
        on |= (np.abs(e[:, k] - lo) < eps) | (np.abs(e[:, k] - hi) < eps)
    return on


def diag_preconditioner(core, dir_mask):
    """Diagonal preconditioner (Helm_s): qf=(rf.nf)/Af^2 ; prec=sumup_F2E(qf); invert.
    sumup_F2E[e] = sum over faces touching e of qf[f]  (|fe_sign|, unsigned face->edge sum)."""
    qf = (core.rf * core.nf).sum(axis=1) / (core.Af * core.Af)
    # SUMUP_F2E over the flat link_fe (unsigned face->edge sum)
    prec = xp.bincount(core.lf_edge, xp.abs(core.fe_sign_link) * qf[core.lf_face], minlength=core.nedge)
    prec[dir_mask] = 0.0
    inv = xp.zeros(core.nedge, dtype=real)
    nz = prec != 0.0
    inv[nz] = 1.0 / prec[nz]
    return inv


# ---------------------------------------------------------------------------
# [2][3] full FlowSolve_exp_3 driver: ghost cells + stream-function BC + CN time loop
# ---------------------------------------------------------------------------
class Driver:
    """Crank-Nicolson stream-function/vorticity stepper with optional immersed
    forcing. Ghost cell per boundary face; BC families 1..6 = INLET/OUTLET/TOP/BOT/
    FRONT/BACK (matches make_euler _outer_box_boundary_faces + BC_IC.txt)."""

    # Class defaults also support small injected research/test drivers that do
    # not construct a mesh through __init__.
    cg_max_iterations = 1000
    fail_on_cg_nonconvergence = False

    def __init__(self, core, mesh, visc, rou=1.0, ac=0.5, ad=0.5, *,
                 cg_max_iterations=1000, fail_on_cg_nonconvergence=False):
        self.core = core
        # Momentum is dimensional: rou is density and visc is dynamic
        # viscosity (mu = rho * nu), not kinematic viscosity.
        self.visc = visc; self.rou = rou; self.ac = ac; self.ad = ad
        self.bc_edge = xp.asarray(mesh.bc_edge)
        self.bc_face = xp.asarray(mesh.bc_face)
        bf = xp.where(~core.m1)[0]                       # boundary faces
        self.bf = bf
        self.nghost = int(bf.shape[0])
        self.face_ghost = xp.full(core.nface, -1, dtype=xp.int64)
        self.face_ghost[bf] = core.ncell + xp.arange(self.nghost)
        self.F2C_ext = core.F2C.copy()
        self.F2C_ext[bf, 1] = self.face_ghost[bf]
        self.bf_fam = self.bc_face[bf]                  # 1..6 per boundary face
        self.inlet_edges = self.bc_edge == 1            # LINEAR stream BC
        self.dir_mask = xp.isin(self.bc_edge, xp.asarray([3, 4, 5, 6]))   # FIXED -> DirBC_s residual zero
        self.prec = diag_preconditioner(core, self.dir_mask)
        self.Asp = None                                 # optional sparse matrix for fast CG matvec
        self.ib = None                                  # optional immersed body (set_ib)
        self.ib_alpha = 1.0                             # IB force amplification (compensate Helm^-1 attenuation)
        self.ib_implicit = False                        # True -> solve M f = rhs (coef matrix) instead of explicit f=rhs
        self._step_count = 0                            # for cold-start handling
        self.cold_cg = 0                                # >0: cap CG iters for the first 3 steps (limits cold-CG pool
        #                                                 fragmentation so fine meshes fit a small GPU; transient only)
        self.free_each_step = False                     # >0 GPU mesh: release cupy pool each step (cap cross-step growth)
        self.cavity = False                             # lid-driven cavity BC (no-slip walls + moving lid) instead of inlet/outlet
        self.motion = None                              # optional prescribed kinematics (ib_kinematics), updates markers each step
        self.time = 0.0                                 # physical time, advanced by each step()
        self.last_helm_m = 0                            # iterations in the most recent Helmholtz solve
        self.helmholtz_iterations_last_step = 0         # maximum over all inner solves in one physical step
        self.cg_max_iterations = self._positive_cg_limit(cg_max_iterations)
        if not isinstance(fail_on_cg_nonconvergence, (bool, np.bool_)):
            raise ValueError("fail_on_cg_nonconvergence must be a boolean")
        self.fail_on_cg_nonconvergence = bool(fail_on_cg_nonconvergence)
        self.last_helm_rel_residual = 0.0
        self.last_helm_tolerance_ratio = 0.0
        self.last_helm_converged = True
        self.helmholtz_relative_residual_last_step = 0.0
        self.helmholtz_tolerance_ratio_last_step = 0.0
        self.helmholtz_converged_last_step = True
        self.last_face_flux = None                     # final accepted velocity flux, no extra reconstruction
        self.last_momentum_residual = None             # final inner residual before the curl-curl correction
        self.last_step_dt = None                       # residual / dt has pressure-gradient units
        self.cg_check_interval = int(
            os.environ.get("IB_CG_CHECK_INTERVAL", "20" if GPU else "1")
        )
        if self.cg_check_interval <= 0:
            raise ValueError("IB_CG_CHECK_INTERVAL must be a positive integer")

    def set_cavity(self, U_lid=1.0, spanwise_zerograd=True):
        """Lid-driven cavity: psi=0 (Dirichlet) on ALL boundary edges (walls are
        streamlines); the y=max wall is the moving lid (u=U_lid), the z-faces (thin
        spanwise slab) are ZEROGRAD by default for quasi-2D; pass
        spanwise_zerograd=False for six no-slip walls. Boundary
        faces are classified BY POSITION (robust to the family numbering).  Ghia 1982."""
        self.cavity = True
        self.U_lid = float(U_lid)
        self._lid_vec = xp.asarray([U_lid, 0.0, 0.0], dtype=real)
        self.inlet_edges = xp.zeros(self.core.nedge, dtype=bool)   # no inlet
        self.dir_mask = self.bc_edge > 0                           # psi=0 on every wall edge
        self.prec = diag_preconditioner(self.core, self.dir_mask)
        # classify boundary faces by position: lid=y-max, z-faces=spanwise slab, rest=no-slip
        import numpy as _np
        fp = _np.asarray(asnumpy(self.core.fpos)); bf = asnumpy(self.bf)
        fy = fp[bf, 1]; fz = fp[bf, 2]
        tol = 1e-6 + 1e-3 * (fy.max() - fy.min())
        self._lid_m = xp.asarray(fy > fy.max() - tol)
        z_faces = (fz < fz.min() + tol) | (fz > fz.max() - tol)
        if not spanwise_zerograd:
            z_faces = _np.zeros_like(z_faces, dtype=bool)
        self._zf_m = xp.asarray(z_faces)
        return self

    def set_ib(self, ib, markers, ds, vel_desired, force_mode='replacement'):
        """Attach an immersed body (ib_immersed.ImmersedBoundary).  markers (M,3),
        ds (M,) cube root of marker integration volume, vel_desired (M,3)
        prescribed marker velocity. Surface spacing and ds need not coincide.
        For a moving body, update markers/vel_desired before each step()."""
        # ``fortran_accumulated`` is retained only as a compatibility alias for
        # archived runs.  New Python-facing code uses the method-based name.
        if force_mode not in {
            'replacement', 'accumulated_explicit', 'fortran_accumulated'
        }:
            raise ValueError(f'unsupported IB force_mode: {force_mode!r}')
        self.ib = ib
        self.ib_markers = asreal(markers)
        self.ib_ds = asreal(ds)
        self.ib_vel = asreal(vel_desired)
        self.ib_force_mode = force_mode
        self.ib_force_E = xp.zeros((self.core.ncell, 3), dtype=self.core.cpos.dtype)
        self.ib_force_L = xp.zeros_like(self.ib_vel)
        self.ib_force_increment_E = xp.zeros_like(self.ib_force_E)
        self.ib_force_increment_L = xp.zeros_like(self.ib_force_L)
        self.ib_force_update_count = 0
        self.ib_force_updates_last_step = 0
        return self

    def ib_state_dict(self):
        '''Return persistent IB state as detached CPU data for np.savez.'''
        if self.ib is None:
            return None
        return {
            'force_mode': self.ib_force_mode,
            'force_E': asnumpy(self.ib_force_E).copy(),
            'force_L': asnumpy(self.ib_force_L).copy(),
            'force_update_count': int(self.ib_force_update_count),
        }

    def load_ib_state_dict(self, state):
        '''Restore persistent IB state after set_ib has been called.'''
        if self.ib is None:
            raise RuntimeError('set_ib must be called before loading IB state')
        mode = str(state['force_mode'])
        if mode != self.ib_force_mode:
            raise ValueError(
                f'IB force-mode mismatch: checkpoint={mode!r}, run={self.ib_force_mode!r}'
            )
        force_E = xp.asarray(state['force_E'], dtype=self.ib_force_E.dtype)
        force_L = xp.asarray(state['force_L'], dtype=self.ib_force_L.dtype)
        if force_E.shape != self.ib_force_E.shape:
            raise ValueError(
                f'Eulerian IB-force shape mismatch: {force_E.shape} != {self.ib_force_E.shape}'
            )
        if force_L.shape != self.ib_force_L.shape:
            raise ValueError(
                f'Lagrangian IB-force shape mismatch: {force_L.shape} != {self.ib_force_L.shape}'
            )
        self.ib_force_E = force_E.copy()
        self.ib_force_L = force_L.copy()
        self.ib_force_increment_E = xp.zeros_like(self.ib_force_E)
        self.ib_force_increment_L = xp.zeros_like(self.ib_force_L)
        self.ib_force_update_count = int(state.get('force_update_count', 0))
        self.ib_force_updates_last_step = 0
        return self

    def set_motion(self, motion, t0=0.0):
        """Attach prescribed kinematics (ib_kinematics object: motion(t) ->
        (markers, vel)).  Each step() advances the clock to t+dt and sets
        ib_markers/ib_vel at the NEW time level (Fortran updates the mesh before
        the force calc).  The implicit coef matrix M = interp.spread depends on
        the marker positions, so it is invalidated (rebuilt) every step."""
        self.motion = motion
        self.time = float(t0)
        return self

    def enable_sparse(self, dt):
        """Assemble flow_mat as a constant sparse matrix; CG then uses A@p (fast matvec)
        instead of the python operator chain.  CG (not direct LU) because the stream-
        function operator has a gauge null space that LU can't factor but CG handles."""
        from ib_sparse import build_A
        self.Asp = build_A(self.core, alpd=self.ad, visc=self.visc, rou=self.rou, dt=dt)
        return self

    # cell-vec operators on ghost-extended fields ------------------------------
    def cgrad_ext(self, cell_ext_col):   # scalar component (ncell+nghost,)
        return cell_ext_col[self.F2C_ext[:, 1]] - cell_ext_col[self.F2C_ext[:, 0]]

    def wavg_ext(self, cell_ext_col, wf1, wf2):
        return cell_ext_col[self.F2C_ext[:, 0]] * wf1 + cell_ext_col[self.F2C_ext[:, 1]] * wf2

    # BC + velocity ------------------------------------------------------------
    def calc_vel(self, s):
        """BC_s (INLET=LINEAR) -> U=curl(s) -> vc=interp_f2c(U) -> BC_vc ghosts."""
        core = self.core
        s = s.copy()
        s[self.inlet_edges] = core.epos[self.inlet_edges, 1] * core.te[self.inlet_edges, 2]
        U = core.curl(s)
        vc = xp.zeros((core.ncell + self.nghost, 3), dtype=real)
        vc[:core.ncell] = core.interp_f2c(U)
        bf = self.bf; c1 = core.F2C[bf, 0]; g = self.face_ghost[bf]
        if self.cavity:
            # lid-driven cavity ghosts: face velocity = 0 (no-slip wall), = (U_lid,0,0)
            # (moving lid), ZEROGRAD on the spanwise z-faces.  ghost = 2*face_target - interior.
            v1 = vc[c1]
            gv = -v1                                     # default no-slip walls: face vel 0
            gv[self._lid_m] = 2.0 * self._lid_vec - v1[self._lid_m]   # moving lid: face vel = U_lid
            gv[self._zf_m] = v1[self._zf_m]              # z-faces: ZEROGRAD (quasi-2D slab)
            vc[g] = gv
            return s, U, vc
        nf = core.nf[bf]; Af2 = (core.Af[bf] ** 2)[:, None]; Uf = U[bf][:, None]
        zero = self.bf_fam == 1                          # INLET: ZERO tangential velocity
        vc[g[zero]] = Uf[zero] * nf[zero] / Af2[zero]
        zg = ~zero                                       # others: ZEROGRAD
        v1 = vc[c1[zg]]
        vc[g[zg]] = v1 + (Uf[zg] - (v1 * nf[zg]).sum(axis=1, keepdims=True)) * nf[zg] / Af2[zg]
        return s, U, vc

    def conv_diff(self, U, vc):
        """convection_cell + diffusion_cell (conv_id=1: CENTRAL, no cross-diffusion)."""
        core = self.core; ncell = core.ncell
        conv = xp.zeros((ncell, 3), dtype=real); diff = xp.zeros((ncell, 3), dtype=real)
        wf = 0.5 * U                                     # set_conv CENTRAL: wf1=wf2=0.5*U
        for j in range(3):
            rd = self.cgrad_ext(vc[:, j])                # CGRAD(vc_j) with ghost
            diff[:, j] = core.vol_ci * core.cdiv(core.AoL * self.visc * rd)
            vf = self.wavg_ext(vc[:, j], wf, wf)         # WAVG_C2F(vc_j, wf)
            conv[:, j] = core.vol_ci * core.cdiv(vf)
        return conv, diff

    @staticmethod
    def _positive_cg_limit(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError("cg_max_iterations must be a positive integer")
        return int(value)

    def helm_solve(self, res_e, s, dt, tol1=1e-10, tol2=1e-9, cg_max=None):
        """CG correction from warm start ``s`` and supplied residual ``b-A s``.

        The accepted iterate uses the historical mixed preconditioned stopping
        criterion and check cadence. A final, independently recomputed residual
        records whether it really meets that criterion. ``last_helm_rel_residual``
        is the Euclidean final/initial residual ratio; it is not itself the mixed
        stopping criterion. ``last_helm_tolerance_ratio <= 1`` means that the true
        residual meets the existing mixed tolerance. Strict mode raises on a
        failed true residual, not merely on reaching the iteration limit.
        """
        cg_max = self._positive_cg_limit(
            self.cg_max_iterations if cg_max is None else cg_max
        )
        if not isinstance(self.fail_on_cg_nonconvergence, (bool, np.bool_)):
            raise ValueError("fail_on_cg_nonconvergence must be a boolean")
        if FP32:                                          # float32 CG floor ~1e-6; 1e-9 unreachable
            tol1 = max(tol1, 3e-5); tol2 = max(tol2, 3e-5)
        core = self.core
        kw = dict(alpd=self.ad, visc=self.visc, rou=self.rou, dt=dt)
        r = res_e.copy(); r[self.dir_mask] = 0.0
        initial_l2_squared = r.dot(r)
        x = s.copy()
        pp = r * self.prec
        eta = pp.dot(r); rho0 = eta
        xnorm = x.dot(x) + 1e-16
        err_tot = xnorm * (tol1 ** 2) + (tol2 ** 2) * rho0
        def finish(iterations, true_residual):
            # One batched device-to-host transfer; no per-edge host copy.
            true_eta = (true_residual * self.prec).dot(true_residual)
            values = np.asarray(asnumpy(xp.stack([
                true_eta, true_residual.dot(true_residual), initial_l2_squared,
                err_tot,
            ])), dtype=float)
            eta_final, l2_final, l2_initial, threshold = values
            finite = bool(np.all(np.isfinite(values)) and np.all(values >= 0.0))
            self.last_helm_m = int(iterations)
            self.last_helm_rel_residual = (
                float(np.sqrt(l2_final / l2_initial)) if finite and l2_initial > 0.0
                else (0.0 if finite and l2_final == 0.0 else float("inf"))
            )
            self.last_helm_tolerance_ratio = (
                float(np.sqrt(eta_final / threshold)) if finite and threshold > 0.0
                else (0.0 if finite and eta_final == 0.0 else float("inf"))
            )
            self.last_helm_converged = bool(finite and eta_final <= threshold)
            if self.fail_on_cg_nonconvergence and not self.last_helm_converged:
                # Preserve useful diagnostics even if step() cannot complete.
                self.helmholtz_iterations_last_step = max(
                    getattr(self, "helmholtz_iterations_last_step", 0), int(iterations)
                )
                self.helmholtz_relative_residual_last_step = max(
                    getattr(self, "helmholtz_relative_residual_last_step", 0.0),
                    self.last_helm_rel_residual,
                )
                self.helmholtz_tolerance_ratio_last_step = max(
                    getattr(self, "helmholtz_tolerance_ratio_last_step", 0.0),
                    self.last_helm_tolerance_ratio,
                )
                self.helmholtz_converged_last_step = False
                raise RuntimeError(
                    "Helmholtz CG did not converge: "
                    f"iterations={iterations}/{cg_max}, "
                    f"true_relative_residual={self.last_helm_rel_residual:.6e}, "
                    f"mixed_tolerance_ratio={self.last_helm_tolerance_ratio:.6e}"
                )
            return x, int(iterations)

        if float(eta) < err_tot:                         # already converged (Helm_s: goto 100)
            return finish(0, r)
        # Convergence cadence is a numerical contract, not merely a device
        # optimization: changing it changes the accepted iterate and therefore
        # the subsequent trajectory.  Publication CPU parity runs set it to 20.
        check = self.cg_check_interval
        infinity = xp.asarray(xp.inf, dtype=r.dtype)
        for m in range(1, cg_max + 1):
            w = (self.Asp @ pp) if self.Asp is not None else core.flow_mat(pp, **kw)
            w[self.dir_mask] = 0.0
            denominator = pp.dot(w)
            # Exact convergence can precede the next scheduled device check.
            # Keep the converged iterate instead of producing 0/0 and NaNs.
            alf = eta / xp.where(denominator != 0.0, denominator, infinity)
            x += alf * pp
            r -= alf * w
            z = r * self.prec
            eta_new = z.dot(r)
            if m % check == 0:
                checked_eta = float(eta_new)
                if not np.isfinite(checked_eta) or checked_eta < err_tot:
                    break
            beta = eta_new / xp.where(eta != 0.0, eta, infinity)
            pp = z + beta * pp
            eta = eta_new
        # res_e is b-A(s_initial), not b. This form avoids subtracting two
        # separately reconstructed large b/Ax vectors for a warm start.
        correction = x - s
        correction_image = (self.Asp @ correction) if self.Asp is not None else core.flow_mat(correction, **kw)
        true_residual = res_e - correction_image
        true_residual[self.dir_mask] = 0.0
        return finish(m, true_residual)

    def step(self, s, s_old, dt, dt_old, inner=3):
        """One fluid step with optional immersed forcing and prescribed motion.

        Returns ``(s_new, vc_new, s_old_new)``. Helmholtz diagnostics aggregate
        all fluid inner solves, independently of history/field output cadence.
        """
        core = self.core; rou = self.rou; ac = self.ac; ad = self.ad
        accumulated = (
            self.ib is not None
            and self.ib_force_mode in {
                'accumulated_explicit', 'fortran_accumulated'
            }
        )
        if accumulated and inner < 2:
            raise ValueError('accumulated_explicit requires at least 2 fluid inner iterations')
        self._step_count += 1
        self.time += dt
        self.helmholtz_iterations_last_step = 0
        self.helmholtz_relative_residual_last_step = 0.0
        self.helmholtz_tolerance_ratio_last_step = 0.0
        self.helmholtz_converged_last_step = True
        self.last_helm_rel_residual = 0.0
        self.last_helm_tolerance_ratio = 0.0
        self.last_helm_converged = True
        self.last_face_flux = None
        self.last_momentum_residual = None
        self.last_step_dt = None
        if self.ib is not None:
            self.ib_force_updates_last_step = 0
        if self.ib is not None and self.motion is not None:
            mk, vel = self.motion(self.time)             # body state at the new time level
            self.ib_markers = asreal(mk)
            self.ib_vel = asreal(vel)
            if self.ib_implicit:
                self.ib.invalidate_implicit()            # M is position-dependent -> rebuild
            if accumulated:
                # The held Lagrangian force follows a moving body.
                self.ib_force_E = self.ib.spread(
                    self.ib_force_L, self.ib_markers, self.ib_ds
                )
        cg = self._positive_cg_limit(self.cg_max_iterations)
        if self.cold_cg and self._step_count <= 3:
            cg = min(self._positive_cg_limit(self.cold_cg), cg)
        # state from current s
        _, U, vc = self.calc_vel(s)
        conv0, diff0 = self.conv_diff(U, vc)
        res_c0 = rou * vc[:core.ncell] - (1 - ac) * dt * rou * conv0 + (1 - ad) * dt * diff0
        # Guess_s
        qe = s.copy()
        s = s + (s - s_old) * dt / dt_old
        s, U, vc = self.calc_vel(s)
        s_old = qe
        for i in range(1, inner + 1):
            conv, diff = self.conv_diff(U, vc)
            res_c = res_c0 - rou * vc[:core.ncell] - ac * dt * rou * conv + ad * dt * diff
            if self.ib is not None:
                if accumulated:
                    # All solves use the currently held total force.
                    res_c = res_c + dt * self.ib_force_E
                else:
                    # Legacy Python path: replace force at every inner iteration.
                    if self.ib_implicit:
                        fE, fL = self.ib.implicit_force(
                            vc[:core.ncell], self.ib_markers, self.ib_vel, dt, self.ib_ds
                        )
                    else:
                        fE, fL = self.ib.direct_force(
                            vc[:core.ncell], self.ib_markers, self.ib_vel, dt, self.ib_ds
                        )
                    # IB interpolation/spreading returns acceleration; the
                    # momentum equation and reported loads require rho * a.
                    self.ib_force_E = (self.rou * self.ib_alpha) * fE
                    self.ib_force_L = (self.rou * self.ib_alpha) * fL
                    self.ib_force_increment_E = self.ib_force_E
                    self.ib_force_increment_L = self.ib_force_L
                    self.ib_force_update_count += 1
                    self.ib_force_updates_last_step += 1
                    res_c = res_c + dt * self.ib_force_E
            if i == inner:
                # Retain the exact final residual, including dt * IB force.
                # Pressure recovery uses residual / dt, as in the IBFast
                # momentum reconstruction. Assignment keeps only a reference.
                self.last_momentum_residual = res_c
            res_f = core.integr_c2f(res_c)
            res_e = core.rot(res_f)
            s, helm_m = self.helm_solve(res_e, s, dt, cg_max=cg)
            self.helmholtz_iterations_last_step = max(
                self.helmholtz_iterations_last_step, int(helm_m)
            )
            self.helmholtz_relative_residual_last_step = max(
                self.helmholtz_relative_residual_last_step, self.last_helm_rel_residual
            )
            self.helmholtz_tolerance_ratio_last_step = max(
                self.helmholtz_tolerance_ratio_last_step, self.last_helm_tolerance_ratio
            )
            self.helmholtz_converged_last_step = (
                self.helmholtz_converged_last_step and self.last_helm_converged
            )
            s, U, vc = self.calc_vel(s)
            if accumulated and i == 1:
                # Evaluate one explicit increment from the post-predictor
                # velocity, then add it to the persistent force state.
                if self.ib_implicit:
                    delta_E, delta_L = self.ib.implicit_force(
                        vc[:core.ncell], self.ib_markers, self.ib_vel, dt, self.ib_ds
                    )
                else:
                    delta_E, delta_L = self.ib.direct_force(
                        vc[:core.ncell], self.ib_markers, self.ib_vel, dt, self.ib_ds
                    )
                self.ib_force_increment_E = (self.rou * self.ib_alpha) * delta_E
                self.ib_force_increment_L = (self.rou * self.ib_alpha) * delta_L
                self.ib_force_E = self.ib_force_E + self.ib_force_increment_E
                self.ib_force_L = self.ib_force_L + self.ib_force_increment_L
                self.ib_force_update_count += 1
                self.ib_force_updates_last_step += 1
        self.last_face_flux = U
        self.last_step_dt = float(dt)
        if self.free_each_step:
            free_gpu_pool()                             # release pool between steps (fine-mesh memory cap)
        return s, vc, s_old


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"e:\ImmerseBoundaryProject\IBZhaoyue")
    sys.path.insert(0, r"e:\ImmerseBoundaryProject\IBZhaoyue\solver")
    sys.argv.append("--nometis")
    import make_euler_mesh as M
    mesh = M.make_nested_mesh(0.25, [M._build_box(("center+size", (0, 0, 0), (2, 2, 2)))],
                              refinement_ratio=2)
    core = IBCore(mesh)
    dmask = boundary_edge_mask(core)
    print(f"edges={core.nedge}  Dirichlet(boundary)={int(dmask.sum())}  interior={int((~dmask).sum())}")

    # ---- manufactured-solution test: pick x_true (0 on boundary), b = A x_true, solve, compare ----
    rng = np.random.default_rng(0)
    e = core.epos
    smooth = np.cos(e[:, 0]) * np.cos(e[:, 1]) * np.cos(e[:, 2])      # smooth interior field
    x_true = smooth.copy(); x_true[dmask] = 0.0
    b = apply_A(core, x_true, dmask)

    prec = diag_preconditioner(core, dmask)
    for label, P in [("no-prec", None), ("diag-prec", prec)]:
        x, iters, res = cg_solve(core, b, dmask, tol=1e-12, maxiter=5000, prec=P)
        rA = apply_A(core, x, dmask) - b
        res_rel = np.linalg.norm(rA) / (np.linalg.norm(b) + 1e-300)
        # solution error on interior (operator may have a null space -> report both)
        err = np.linalg.norm(x - x_true) / (np.linalg.norm(x_true) + 1e-300)
        print(f"  [{label:9s}] iters={iters:4d}  ‖Ax-b‖/‖b‖={res_rel:.2e}  ‖x-x_true‖/‖x_true‖={err:.2e}")

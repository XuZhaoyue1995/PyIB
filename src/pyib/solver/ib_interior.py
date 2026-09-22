"""Interior-fluid load correction for closed immersed bodies.

The immersed-boundary force is applied to all fluid covered by the regularised
surface, including the fictitious fluid enclosed by a closed body.  Therefore
the hydrodynamic load on that body is not, in general, just the negative of the
force applied to the fluid.  Following Uhlmann's momentum bookkeeping,

    F_body = -F_fluid + d/dt integral_body(rho * u) dV,

with the analogous angular-momentum term for the moment about the origin.

The canonical Python runner currently supports closed spheres only.  It uses
the same one-cell, linearly smoothed analytic sphere indicator as the validated
oscillating-sphere and Magnus research cases.  NumPy and CuPy share this module
through :mod:`backend`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from backend import asnumpy, real, xp


@dataclass
class InteriorMomenta:
    """Volume, linear momentum and angular momentum inside a closed body."""

    linear_momentum: Any
    angular_momentum: Any
    indicator_volume: Any


@dataclass
class CorrectedBodyLoad:
    """Raw fluid load and the corresponding interior-momentum-corrected load."""

    force_fluid: Any
    moment_fluid: Any
    body_force: Any
    body_moment: Any
    interior_momenta: InteriorMomenta
    linear_momentum_rate: Any
    angular_momentum_rate: Any


def _validate_sphere_parameters(radius, smoothing_width, density) -> None:
    if not all(
        math.isfinite(float(value)) and float(value) > 0.0
        for value in (radius, smoothing_width, density)
    ):
        raise ValueError("radius, smoothing_width and density must be finite and positive")


def _sphere_weighted_volumes(positions, volumes, center, radius, smoothing_width):
    distance = xp.linalg.norm(positions - center, axis=1)
    indicator = xp.clip(
        (radius + 0.5 * smoothing_width - distance) / smoothing_width, 0.0, 1.0
    )
    return indicator * volumes


def _integrate_weighted_velocity(velocity, positions, weighted_volumes, density):
    linear = density * (velocity * weighted_volumes[:, None]).sum(axis=0)
    angular = density * (
        xp.cross(positions, velocity) * weighted_volumes[:, None]
    ).sum(axis=0)
    return InteriorMomenta(linear, angular, weighted_volumes.sum())


def sphere_interior_momenta(
    velocity,
    cell_positions,
    cell_volumes,
    center,
    *,
    radius: float,
    smoothing_width: float,
    density: float,
) -> InteriorMomenta:
    """Integrate fictitious-fluid momentum over a smoothed sphere indicator.

    ``smoothing_width`` is normally the finest Eulerian spacing.  The indicator
    is one inside ``radius - width/2``, zero outside ``radius + width/2``, and
    linear across the intervening cell.  Angular momentum is about the global
    origin so it is paired with a fluid moment computed about that same origin.
    """

    velocity = xp.asarray(velocity, dtype=real)
    cell_positions = xp.asarray(cell_positions, dtype=real)
    cell_volumes = xp.asarray(cell_volumes, dtype=real)
    center = xp.asarray(center, dtype=real)

    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError("velocity must have shape (number_of_cells, 3)")
    if cell_positions.shape != velocity.shape:
        raise ValueError("cell_positions must have the same shape as velocity")
    if cell_volumes.shape != (velocity.shape[0],):
        raise ValueError("cell_volumes must have shape (number_of_cells,)")
    if center.shape != (3,):
        raise ValueError("center must be a three-vector")
    _validate_sphere_parameters(radius, smoothing_width, density)
    weighted_volumes = _sphere_weighted_volumes(
        cell_positions, cell_volumes, center, float(radius), float(smoothing_width)
    )
    return _integrate_weighted_velocity(
        velocity, cell_positions, weighted_volumes, float(density)
    )


class ClosedSphereLoadCorrection:
    """Track and correct loads for one closed sphere at every physical step."""

    def __init__(
        self,
        cell_positions,
        cell_volumes,
        *,
        reference_center,
        maximum_translation,
        radius: float,
        smoothing_width: float,
        density: float,
        force_support_width: float | None = None,
    ) -> None:
        """Preselect cells within the prescribed sphere's swept support.

        ``maximum_translation`` bounds displacement from ``reference_center``
        componentwise.  A zero bound declares a fixed centre (also valid for a
        rotating sphere), allowing indicator weights to be cached.

        By default raw Eulerian force and moment are integrated over every
        cell.  When the caller guarantees a compact IB force kernel,
        ``force_support_width`` may give its maximum componentwise distance
        from the marker surface; the Roma three-point kernel uses ``1.5*h``.
        This avoids whole-domain load reductions at each time step.
        """

        positions = xp.asarray(cell_positions, dtype=real)
        volumes = xp.asarray(cell_volumes, dtype=real)
        reference_center = xp.asarray(reference_center, dtype=real)
        maximum_translation = xp.asarray(maximum_translation, dtype=real)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("cell_positions must have shape (number_of_cells, 3)")
        if volumes.shape != (positions.shape[0],):
            raise ValueError("cell_volumes must have shape (number_of_cells,)")
        if reference_center.shape != (3,) or maximum_translation.shape != (3,):
            raise ValueError(
                "reference_center and maximum_translation must be three-vectors"
            )
        _validate_sphere_parameters(radius, smoothing_width, density)
        if not all(
            bool(asnumpy(xp.all(xp.isfinite(value))))
            for value in (reference_center, maximum_translation)
        ):
            raise ValueError("reference_center and maximum_translation must be finite")
        if force_support_width is not None and (
            not math.isfinite(float(force_support_width)) or force_support_width < 0.0
        ):
            raise ValueError("force_support_width must be finite and non-negative")

        # Only cells in the swept indicator support can contribute.  Restricting
        # this once avoids a whole-domain distance calculation on every step.
        half_width = (
            float(radius) + 0.5 * float(smoothing_width) + xp.abs(maximum_translation)
        )
        lower = reference_center - half_width
        upper = reference_center + half_width
        candidate_mask = xp.all((positions >= lower) & (positions <= upper), axis=1)
        self._candidate_indices = xp.nonzero(candidate_mask)[0]
        if int(self._candidate_indices.size) == 0:
            raise ValueError("the closed-sphere indicator contains no Eulerian cells")

        self._all_positions = positions
        self._all_volumes = volumes
        self._positions = positions[self._candidate_indices]
        self._volumes = volumes[self._candidate_indices]
        self._force_indices = None
        self._force_positions = positions
        self._force_volumes = volumes
        if force_support_width is not None:
            force_half_width = (
                float(radius) + float(force_support_width) + xp.abs(maximum_translation)
            )
            # Include roundoff at the declared support boundary.  Marker motion
            # and centre arithmetic can otherwise differ by a final bit.
            padding = 8.0 * xp.finfo(real).eps * xp.maximum(
                1.0, xp.abs(reference_center) + force_half_width
            )
            force_half_width = force_half_width + padding
            force_mask = xp.all(
                (positions >= reference_center - force_half_width)
                & (positions <= reference_center + force_half_width),
                axis=1,
            )
            self._force_indices = xp.nonzero(force_mask)[0]
            self._force_positions = positions[self._force_indices]
            self._force_volumes = volumes[self._force_indices]
        self.radius = float(radius)
        self.smoothing_width = float(smoothing_width)
        self.density = float(density)
        self._fixed_weighted_volumes = None
        if bool(asnumpy(xp.all(maximum_translation == 0.0))):
            self._fixed_weighted_volumes = _sphere_weighted_volumes(
                self._positions,
                self._volumes,
                reference_center,
                self.radius,
                self.smoothing_width,
            )
        self.current: InteriorMomenta | None = None
        self.latest_load: CorrectedBodyLoad | None = None

    def evaluate(self, velocity, center) -> InteriorMomenta:
        """Return the interior momenta without changing the time-history state."""

        velocity = xp.asarray(velocity, dtype=real)
        if velocity.shape != self._all_positions.shape:
            raise ValueError("velocity shape does not match the Eulerian cell layout")
        center = xp.asarray(center, dtype=real)
        if center.shape != (3,):
            raise ValueError("center must be a three-vector")
        weighted_volumes = self._fixed_weighted_volumes
        if weighted_volumes is None:
            weighted_volumes = _sphere_weighted_volumes(
                self._positions,
                self._volumes,
                center,
                self.radius,
                self.smoothing_width,
            )
        return _integrate_weighted_velocity(
            velocity[self._candidate_indices],
            self._positions,
            weighted_volumes,
            self.density,
        )

    def initialize(self, velocity, center) -> InteriorMomenta:
        """Record the initial/restart state used by the next backward difference."""

        self.current = self.evaluate(velocity, center)
        self.latest_load = None
        return self.current

    def restore(self, linear, angular, indicator_volume) -> None:
        """Restore a validated post-step momentum state from a checkpoint."""

        linear = xp.asarray(linear, dtype=real)
        angular = xp.asarray(angular, dtype=real)
        indicator_volume = xp.asarray(indicator_volume, dtype=real)
        if (
            linear.shape != (3,)
            or angular.shape != (3,)
            or indicator_volume.shape != ()
        ):
            raise ValueError("invalid closed-body checkpoint momentum shapes")
        if not all(
            bool(asnumpy(xp.all(xp.isfinite(value))))
            for value in (linear, angular, indicator_volume)
        ) or float(asnumpy(indicator_volume)) <= 0.0:
            raise ValueError("closed-body checkpoint momenta must be finite with positive volume")
        self.current = InteriorMomenta(linear, angular, indicator_volume)
        self.latest_load = None

    def advance(
        self, velocity, center, dt: float, eulerian_force_density
    ) -> CorrectedBodyLoad:
        """Advance the bookkeeping once and return the corrected post-step load."""

        if self.current is None:
            raise RuntimeError(
                "closed-body correction must be initialized before advance"
            )
        if not math.isfinite(float(dt)) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        force_density = xp.asarray(eulerian_force_density, dtype=real)
        if force_density.shape != self._all_positions.shape:
            raise ValueError(
                "eulerian_force_density shape does not match the cell layout"
            )

        previous = self.current
        current = self.evaluate(velocity, center)
        linear_rate = (current.linear_momentum - previous.linear_momentum) / float(dt)
        angular_rate = (current.angular_momentum - previous.angular_momentum) / float(
            dt
        )
        if self._force_indices is not None:
            force_density = force_density[self._force_indices]
        force_fluid = (force_density * self._force_volumes[:, None]).sum(axis=0)
        moment_fluid = (
            xp.cross(self._force_positions, force_density) * self._force_volumes[:, None]
        ).sum(axis=0)
        load = CorrectedBodyLoad(
            force_fluid=force_fluid,
            moment_fluid=moment_fluid,
            body_force=-force_fluid + linear_rate,
            body_moment=-moment_fluid + angular_rate,
            interior_momenta=current,
            linear_momentum_rate=linear_rate,
            angular_momentum_rate=angular_rate,
        )
        self.current = current
        self.latest_load = load
        return load

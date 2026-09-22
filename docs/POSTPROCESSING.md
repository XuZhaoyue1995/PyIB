# Postprocessing methods and recorded test coverage

The included JSON examples and packaging regression checks use spheres.
That example coverage does not define the general geometry capability of the
immersed-boundary method. The author describes thin-wall treatment using a
one-sided, locally infinite-thickness interpretation. This packaging update
did not implement or validate a new thin-wall adapter.

The bundled offline command reads the standard field snapshot, reconstructs
whole-field pressure from the saved momentum residual, applies an exterior
pressure-band correction and fits surface pressure. Near-wall velocity repair
and a wall-pinned fit provide tangential wall shear. The existing canonical
snapshot processor and its tests use analytic sphere geometry; arbitrary
geometry handling and the author's thin-wall treatment require their own
entry points and test evidence.

The current implementation requires suitable fine-grid padding for the repair
and interpolation band. It reports an error when that support is insufficient.
For example, the very coarse supplied smoke configuration can be processed
with `--field-only`; full surface recovery should use an appropriately resolved
case. Older snapshots without the saved momentum residual cannot faithfully
recover the same pressure from the final velocity alone.

Pressure is determined up to a reference constant, with a boundary-adjacent
zero-mean gauge. The velocity-band correction uses a thin-layer Stokes
approximation. Recovered wall shear is tangential traction rather than the
entire stress tensor. Raw and repaired fields remain separate, and optional
VTU output contains cell-centre and marker sample points.

The separate 4,096-cell audit field uses dx=0.125, 13 markers and five steps.
Its full postprocessing completed, with surface-integrated and volume-IB loads
differing by 0.648684018993278 (about 64.87%). This is integration evidence,
not an accepted accuracy benchmark. Existing numerical records in `checks/`
retain the exact tests and their results. D32/D40 physical validation remains
separate ongoing work.

`pyib-validate` checks artifact integrity and internal consistency. A `VALID`
result does not certify physical accuracy. The runner's closed-body
internal-fluid momentum correction is a separate force-accounting operation
from pressure and shear reconstruction; it should not be applied to a
zero-volume thin wall by assuming that the sphere bookkeeping is universal.

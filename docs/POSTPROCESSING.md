# Pressure-field reconstruction and surface-pressure correction

```sh
pyib-postprocess RUN/final_field.npz --output RUN/postprocess --vtk
```

The command reconstructs pressure from the saved solver state, corrects the
exterior pressure band and evaluates surface pressure. Its numerical methods
are unchanged from the D32/D40 oscillating-sphere validation. The measured
surface-pressure errors and error definitions are in [VALIDATION.md](VALIDATION.md).

## Methods

- **Pressure-field reconstruction:** an area/distance-weighted graph
  projection of the saved momentum residual divided by the time step.
  The reference pressure is the mean at boundary-adjacent cells.
- **Surface-pressure correction:** the normal component of the spread IB
  forcing is removed from the pressure-gradient target within the exterior
  pressure band. A least-squares band solve retains the outer pressure
  anchors; a one-sided quadratic pressure fit evaluates the surface value.
- **Pressure-force integration:** corrected surface pressure is integrated
  against the outward normals and marker areas. Removing the area-weighted
  surface mean during force quadrature makes this force independent of the
  pressure reference; it does not change the saved pressure field.

The surface integral is the pressure contribution to force. The solver's
runtime body force is available separately in `history.csv` and includes
the internal-fluid momentum-rate correction. Wall-shear recovery and
surface-total-force output are not part of this release.

## Parameters and output

```sh
pyib-postprocess RUN/final_field.npz --output RUN/postprocess --pressure-width 1.5 --vtk
```

`--pressure-width` sets the corrected exterior band width in finest-grid
spacings, with a default of 1.5. Use `--field-only` to reconstruct the
whole-field pressure without surface processing. `--overwrite` allows
replacement of existing postprocessing outputs in the selected directory.

Outputs are `pressure.npz`, `surface_pressure.csv` and `postprocess.json`.
The NPZ stores `pressure_raw`, corrected `pressure_lfc`,
`surface_pressure_raw`, corrected `surface_pressure_lfc`, marker positions,
normals, areas and `pressure_force`. Raw pressure remains available for
comparison. With `--field-only`, surface arrays and the surface CSV are omitted.

The JSON records source and snapshot hashes, parameters, pressure projection
and pressure-band diagnostics, surface quadrature and pressure force.
`--vtk` additionally writes `pressure_field.vtu` and, when surface processing
is enabled, `surface_pressure.vtu`. These contain cell-centre and
surface-marker sample points for ParaView.

## Input geometry and saved fields

Use a standard snapshot containing the momentum residual, time step,
mesh connectivity, velocity, IB forcing and surface information. The saved
residual provides the pressure-gradient target. Surface processing also
uses the finest-grid lattice and exterior pressure-fit samples.

The bundled snapshot processor uses analytic sphere geometry, outward
normals and equal-area Fibonacci marker quadrature. A thin-wall geometry
adapter must supply the selected fluid side, normals, distances and surface
quadrature under the author's one-sided, locally infinite-thickness
convention.

Use the same density, force convention, pressure reference and surface
quadrature when comparing numerical and analytical values. The validation
record gives the exact benchmark parameters and normalization.

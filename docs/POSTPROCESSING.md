# Pressure-field reconstruction and surface-pressure correction

```sh
pyib-postprocess RUN/final_field.npz --output RUN/postprocess --vtk
```

The command reconstructs pressure from the saved solver state, corrects the
exterior pressure band and evaluates surface pressure. Its numerical methods
are unchanged from the D32/D40 oscillating-sphere validation. A second path,
`--two-sided`, handles triangulated surfaces and both of their sides. The measured
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

## Two-sided correction for triangulated surfaces

```sh
pyib-postprocess RUN/final_field.npz --output RUN/postprocess2 --two-sided
pyib-postprocess RUN/final_field.npz --output RUN/postprocess2 --two-sided --far-gauge 1.0
pyib-postprocess SNAPSHOT.npz --two-sided --surface plate_triangles.npz --kernel h3s
```

`--two-sided` selects the port of IBFast `tools/pressure_lfc.py --two-sided`
(`ib_pressure_surface.py`).  The command without it is the sphere-exterior
correction described above and is unchanged.  The method is the same
normal-force band repair; what differs is what the sphere path assumes:

| | default (sphere exterior) | `--two-sided` |
|---|---|---|
| Surface | analytic sphere, radial normals, equal-area markers | triangulation of the markers, open (zero-thickness plates, wings) or closed; area-weighted normals and lumped marker areas |
| Sides | exterior band 0–1.5 h | band of 1.5 h on both sides; `p_plus` and `p_minus` at every marker |
| Which lattice faces are dropped | where the signed distance changes sign | where the segment between two cell centres meets a triangle (beyond the rim of an open surface the two sides are the same fluid) |
| Orientation | outward from the centre | one winding per connected component by walking across shared edges; closed surfaces turned outward, '+' is the side the normals point to |
| Pressure zero | mean of the boundary-adjacent cells of the whole mesh | the same, or with `--far-gauge D` the mean over the finest-lattice cells farther than `D` from all markers |
| Kernel of the spread force | h3 | h3, h3s (cell-averaged h3, Yang et al. 2009) or h4 |
| Pressure force | exterior pressure, surface mean removed | closed: `-sum p_plus n A`; open: `-sum (p_plus - p_minus) n A` |

A snapshot of the bundled sphere cases has no triangles; its markers are
triangulated by their convex hull.  For any other surface give the triangles in
the snapshot (`triangles`) or with `--surface FILE.npz` (integer array
`triangles`, shape `(count, 3)`, indexing the markers).  `--raw-pressure lattice`
recovers the raw pressure on the finest lattice only, as IBFast does (cells
within six spacings of the surface, or every lattice cell with `--far-gauge`),
instead of repairing the whole-mesh projection.

Outputs: `surface_pressure.csv` with the columns
`x,y,z,nx,ny,nz,area,p_plus_raw,p_minus_raw,p_plus,p_minus`; `pressure.npz` with
the same surface arrays, `triangles`, `surface_closed`, the lattice fields
`pressure_raw_lattice`, `pressure_lfc_lattice`, `surface_distance`, and
`pressure_force`, `pressure_force_raw`, `marker_normal_force`.
`postprocess.json` records the band size, the number of faces cut, band cells
left unrepaired (no path to an anchor), rank-deficient surface fits (markers at
the rim of an open surface) and the far-gauge cells with the scatter of the raw
pressure over them.

Checks of this path are in [VALIDATION.md](VALIDATION.md): agreement with the
IBFast tool on an IBFast dump of an impulsively started zero-thickness disc, and
with the sphere-exterior path on the oscillating sphere.  The PyIB solver itself
still runs sphere cases only; open surfaces reach this path through snapshots
that carry their own markers and triangles.

## Input geometry and saved fields

Use a standard snapshot containing the momentum residual, time step,
mesh connectivity, velocity, IB forcing and surface information. The saved
residual provides the pressure-gradient target. Surface processing also
uses the finest-grid lattice and exterior pressure-fit samples.

The default snapshot processor uses analytic sphere geometry, outward
normals and equal-area Fibonacci marker quadrature. Thin walls and other
triangulated surfaces are handled by `--two-sided` (previous section).

Use the same density, force convention, pressure reference and surface
quadrature when comparing numerical and analytical values. The validation
record gives the exact benchmark parameters and normalization.

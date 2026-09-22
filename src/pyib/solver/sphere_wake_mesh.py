"""Literature-guided nested mesh for steady flow past a sphere at Re=100.

The sphere is centred at the origin and has diameter ``D=0.5``. Extents are
specified in sphere diameters so the resolution study uses the same physical
domain at D/dx = 16, 24, and 32.

The refinement follows the wake instead of using the old concentric cubes:

    level  spacing  x/D        y/D, z/D
      0       h     [-1,  3]   [-1,  1]
      1      2h     [-2,  6]   [-2,  2]
      2      4h     [-4, 12]   [-4,  4]
      3      8h     [-8, 16]   [-8,  8]

At D/dx=24 this produces 746,496 cells. The finest region contains the
entire recirculation bubble reported for Re=100, while successively coarser
regions retain the downstream wake without paying for an oversized uniform
far field.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

# The single-GPU path does not partition the mesh.
if "--nometis" not in sys.argv:
    sys.argv.append("--nometis")

import make_euler_mesh as M


DIAMETER = 0.5
DEFAULT_DDX = 24

# Bounds are dimensionless multiples of the sphere diameter, innermost first.
BOUNDS_D = (
    (-1.0, 3.0, -1.0, 1.0, -1.0, 1.0),
    (-2.0, 6.0, -2.0, 2.0, -2.0, 2.0),
    (-4.0, 12.0, -4.0, 4.0, -4.0, 4.0),
    (-8.0, 16.0, -8.0, 8.0, -8.0, 8.0),
)


def box_specs(diameter: float = DIAMETER):
    """Return grid-aligned physical boxes for ``make_nested_mesh``."""
    return [
        M._build_box(("bounds", tuple(diameter * np.asarray(bounds_d))))
        for bounds_d in BOUNDS_D
    ]


def expected_layer_cells(ddx: int) -> list[int]:
    """Analytic cell counts, including only each outer box's shell."""
    counts: list[int] = []
    for level, bounds_d in enumerate(BOUNDS_D):
        spacing_d = (2**level) / ddx
        lo = np.asarray(bounds_d)[[0, 2, 4]]
        hi = np.asarray(bounds_d)[[1, 3, 5]]
        whole = int(np.prod(np.rint((hi - lo) / spacing_d).astype(np.int64)))
        if level:
            inner = np.asarray(BOUNDS_D[level - 1])
            inner_lo = inner[[0, 2, 4]]
            inner_hi = inner[[1, 3, 5]]
            hole = int(np.prod(np.rint((inner_hi - inner_lo) / spacing_d).astype(np.int64)))
            whole -= hole
        counts.append(whole)
    return counts


def build_mesh(ddx: int = DEFAULT_DDX, *, verbose: bool = True):
    if ddx <= 0 or ddx % 8:
        raise ValueError("D/dx must be a positive multiple of 8 for this four-level mesh")
    dx_fine = DIAMETER / ddx
    return M.make_nested_mesh(dx_fine, box_specs(), refinement_ratio=2, verbose=verbose)


def mesh_summary(mesh, ddx: int) -> dict:
    expected = expected_layer_cells(ddx)
    actual = [int(v) for v in mesh.layer_cell_counts]
    bounds = np.asarray(BOUNDS_D, dtype=float)
    return {
        "diameter": DIAMETER,
        "D_over_dx": ddx,
        "dx_fine": DIAMETER / ddx,
        "outer_domain_D": {
            "x": [float(bounds[-1, 0]), float(bounds[-1, 1])],
            "y": [float(bounds[-1, 2]), float(bounds[-1, 3])],
            "z": [float(bounds[-1, 4]), float(bounds[-1, 5])],
        },
        "layer_bounds_D": bounds.tolist(),
        "layer_spacing_D": [(2**k) / ddx for k in range(len(BOUNDS_D))],
        "layer_cells": actual,
        "expected_layer_cells": expected,
        "total_cells": int(mesh.cells.shape[0]),
        "expected_total_cells": int(sum(expected)),
        "nodes": int(mesh.nodes.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddx", type=int, default=DEFAULT_DDX, help="sphere cells per diameter")
    parser.add_argument("--json", type=Path, help="optional path for a compact mesh summary")
    parser.add_argument("--quiet", action="store_true", help="suppress generator progress")
    args, _unknown = parser.parse_known_args()

    mesh = build_mesh(args.ddx, verbose=not args.quiet)
    summary = mesh_summary(mesh, args.ddx)
    if summary["layer_cells"] != summary["expected_layer_cells"]:
        raise RuntimeError("generated layer counts do not match the analytic counts")

    report = json.dumps(summary, indent=2)
    print(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(report + os.linesep, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

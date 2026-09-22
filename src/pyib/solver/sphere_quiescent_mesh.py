"""Concentric nested mesh for a sphere oscillating in otherwise quiescent fluid.

The sphere diameter is ``D=0.5`` and its mean centre is the origin.  Unlike the
steady-wake case, the disturbance is symmetric and localized, so concentric
2:1 boxes are appropriate:

    level  spacing  x/D, y/D, z/D
      0       h          [-1, 1]
      1      2h          [-2, 2]
      2      4h          [-4, 4]
      3      8h          [-8, 8]
      4     16h        [-16, 16]

Every complete box contains ``(2N)^3`` cells at its own spacing and every
outer shell removes an ``N^3`` inner hole, where ``N=D/h``.  The total is
therefore exactly ``36 N^3`` cells.
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

if '--nometis' not in sys.argv:
    sys.argv.append('--nometis')

import make_euler_mesh as M


DIAMETER = 0.5
DEFAULT_DDX = 16
HALF_WIDTHS_D = (1.0, 2.0, 4.0, 8.0, 16.0)


def box_specs(diameter: float = DIAMETER):
    boxes = []
    for half_width_d in HALF_WIDTHS_D:
        half_width = diameter * half_width_d
        boxes.append(
            M._build_box(
                (
                    'bounds',
                    (
                        -half_width, half_width,
                        -half_width, half_width,
                        -half_width, half_width,
                    ),
                )
            )
        )
    return boxes


def expected_layer_cells(ddx: int) -> list[int]:
    if ddx <= 0:
        raise ValueError('D/dx must be positive')
    n3 = int(ddx) ** 3
    return [8 * n3] + [7 * n3] * (len(HALF_WIDTHS_D) - 1)


def build_mesh(ddx: int = DEFAULT_DDX, *, verbose: bool = True):
    if ddx <= 0 or ddx % 8:
        raise ValueError('D/dx must be a positive multiple of 8')
    return M.make_nested_mesh(
        DIAMETER / ddx,
        box_specs(),
        refinement_ratio=2,
        verbose=verbose,
    )


def mesh_summary(mesh, ddx: int) -> dict:
    expected = expected_layer_cells(ddx)
    actual = [int(value) for value in mesh.layer_cell_counts]
    return {
        'diameter': DIAMETER,
        'D_over_dx': int(ddx),
        'dx_fine': DIAMETER / ddx,
        'outer_domain_D': {
            'x': [-HALF_WIDTHS_D[-1], HALF_WIDTHS_D[-1]],
            'y': [-HALF_WIDTHS_D[-1], HALF_WIDTHS_D[-1]],
            'z': [-HALF_WIDTHS_D[-1], HALF_WIDTHS_D[-1]],
        },
        'layer_half_widths_D': list(HALF_WIDTHS_D),
        'layer_spacing_D': [(2**level) / ddx for level in range(len(HALF_WIDTHS_D))],
        'layer_cells': actual,
        'expected_layer_cells': expected,
        'total_cells': int(mesh.cells.shape[0]),
        'expected_total_cells': int(sum(expected)),
        'nodes': int(mesh.nodes.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ddx', type=int, default=DEFAULT_DDX)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--quiet', action='store_true')
    args, _unknown = parser.parse_known_args()

    mesh = build_mesh(args.ddx, verbose=not args.quiet)
    summary = mesh_summary(mesh, args.ddx)
    if summary['layer_cells'] != summary['expected_layer_cells']:
        raise RuntimeError('generated layer counts do not match analytic counts')
    report = json.dumps(summary, indent=2)
    print(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(report + os.linesep, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

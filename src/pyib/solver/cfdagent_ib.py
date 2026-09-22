"""Run a reproducible CPU or CUDA immersed-boundary case from JSON."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path


def _read_raw_config(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read case file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit("case file must contain a JSON object")
    return raw


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="versioned JSON case file")
    parser.add_argument("--outdir", type=Path, required=True, help="run output directory")
    parser.add_argument("--resume", type=Path, help="checkpoint.npz to continue")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="override runtime.device")
    parser.add_argument(
        "--precision", choices=("float64", "float32"), help="override runtime.precision"
    )
    parser.add_argument("--steps", type=int, help="override the target total step count")
    parser.add_argument("--overwrite", action="store_true", help="replace known run artifacts")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    raw = copy.deepcopy(_read_raw_config(args.case))
    runtime = raw.setdefault("runtime", {})
    if args.device is not None:
        runtime["device"] = args.device
    device = str(runtime.get("device", "cpu"))
    if args.precision is not None:
        runtime["precision"] = args.precision
    precision = str(runtime.get("precision", "float64"))
    if args.steps is not None:
        if args.steps <= 0:
            raise SystemExit("--steps must be positive")
        raw.setdefault("time", {})["steps"] = args.steps

    # The existing array backend is intentionally selected at import time.  Set
    # it before importing ib_case and keep one device per process.
    os.environ["IB_GPU"] = "1" if device == "cuda" else "0"
    os.environ["IB_FP32"] = "1" if precision == "float32" else "0"
    here = Path(__file__).resolve().parent
    project = here.parent
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(project))

    from ib_case import CaseConfigError, run_case, validate_config

    try:
        config = validate_config(raw)
        run_case(
            config,
            args.outdir,
            resume=args.resume,
            overwrite=args.overwrite,
            quiet=args.quiet,
        )
    except (CaseConfigError, FileExistsError, RuntimeError, FloatingPointError) as exc:
        print(f"CFDAgent-IB error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


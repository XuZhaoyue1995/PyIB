"""Backend-neutral case runner for the Python immersed-boundary solver.

The numerical kernels live in ``ib_core_np.py``, ``ib_driver_np.py`` and
``ib_immersed.py``.  This module supplies the missing application layer:

* validated, versioned JSON case configuration;
* complete fixed-, oscillating-, and rotating-sphere immersed-boundary setup;
* diagnostics and force histories;
* atomic checkpoints and deterministic restart;
* identical NumPy/CuPy execution through the existing backend.

The command-line entry point is ``cfdagent_ib.py``.  Importing this module
directly selects the backend using ``IB_GPU``/``IB_FP32`` exactly once, so a
process must not switch devices after import.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import copy
import csv
import hashlib
import json
import os
import platform
import signal
import sys
import time
import uuid
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import make_euler_mesh as M
from backend import FP32, GPU, asnumpy, real, xp
from ib_core_np import IBCore
from ib_driver_np import Driver
from ib_immersed import ImmersedBoundary
from ib_interior import ClosedSphereLoadCorrection, CorrectedBodyLoad
from ib_kinematics import Oscillation


SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 5
CHECKPOINT_SCHEMA_VERSION = 3

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "name": "fixed_sphere",
    "case": {"kind": "fixed_sphere"},
    "runtime": {"device": "cpu", "precision": "float64"},
    "mesh": {
        "kind": "uniform_box",
        "dx": 0.25,
        "bounds": [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
        "verbose": False,
    },
    "physics": {
        "reynolds": 100.0,
        "density": 1.0,
        "inflow_speed": 1.0,
    },
    "time": {"dt": 0.01, "steps": 5, "inner_iterations": 4},
    "ib": {
        "radius": 0.25,
        "center": [0.0, 0.0, 0.0],
        "marker_spacing": 0.25,
        "force_mode": "accumulated_explicit",
        "implicit": False,
        "alpha": 1.0,
        "motion": {
            "kind": "none",
            "axis": [0.0, 1.0, 0.0],
            "amplitude": 0.0,
            "frequency": 1.0,
            "phase": 0.0,
        },
    },
    "solver": {"sparse": True, "cg_check_interval": 20,
               "cg_max_iterations": 1000, "fail_on_cg_nonconvergence": False},
    "output": {
        "sample_every": 10,
        "checkpoint_every": 1000,
        "field_every": 0,
        "field_start_step": 0,
        "save_field": True,
    },
}

_ALLOWED_KEYS = {
    "root": {"schema_version", "name", "case", "runtime", "mesh", "physics", "time", "ib", "solver", "output"},
    "case": {"kind"},
    "runtime": {"device", "precision"},
    "mesh": {"kind", "dx", "bounds", "cells_per_diameter", "verbose"},
    "physics": {"reynolds", "density", "inflow_speed"},
    "time": {"dt", "steps", "inner_iterations"},
    "ib": {"radius", "center", "marker_spacing", "force_mode", "implicit", "alpha", "motion"},
    "motion": {"kind", "axis", "amplitude", "frequency", "phase", "spin_ratio"},
    "solver": {"sparse", "cg_check_interval", "cg_max_iterations", "fail_on_cg_nonconvergence"},
    "output": {"sample_every", "checkpoint_every", "field_every", "field_start_step", "save_field"},
}

HISTORY_COLUMNS = [
    "step",
    "time",
    "Cd",
    "Cy",
    "Cz",
    "body_force_x",
    "body_force_y",
    "body_force_z",
    "force_fluid_x",
    "force_fluid_y",
    "force_fluid_z",
    "body_moment_x",
    "body_moment_y",
    "body_moment_z",
    "moment_fluid_x",
    "moment_fluid_y",
    "moment_fluid_z",
    "interior_momentum_x",
    "interior_momentum_y",
    "interior_momentum_z",
    "interior_angular_momentum_x",
    "interior_angular_momentum_y",
    "interior_angular_momentum_z",
    "interior_momentum_rate_x",
    "interior_momentum_rate_y",
    "interior_momentum_rate_z",
    "interior_angular_momentum_rate_x",
    "interior_angular_momentum_rate_y",
    "interior_angular_momentum_rate_z",
    "interior_indicator_volume",
    "slip_max",
    "slip_rms",
    "speed_max",
    "cfl",
    "mass_residual_inf",
    "force_balance_relerr",
    "ib_force_updates",
    "helmholtz_iterations",
    "helmholtz_relative_residual",
    "helmholtz_tolerance_ratio",
    "milliseconds_per_step",
]


class CaseConfigError(ValueError):
    """The case configuration is incomplete, inconsistent or unsupported."""


@dataclass
class CaseRuntime:
    config: dict[str, Any]
    mesh: Any
    mesh_summary: dict[str, Any]
    core: IBCore
    driver: Driver
    ib: ImmersedBoundary
    load_correction: ClosedSphereLoadCorrection
    reference_markers: np.ndarray
    reference_force: float
    reference_speed: float
    dx: float
    dt: float
    steps: int
    inner_iterations: int
    layout_fingerprint: str
    source_hashes: dict[str, str]


@dataclass
class RunResult:
    output_directory: Path
    start_step: int
    completed_step: int
    target_steps: int
    stopped: bool
    s: np.ndarray
    s_old: np.ndarray
    vc: np.ndarray
    ib_force_E: np.ndarray
    ib_force_L: np.ndarray
    body_force: np.ndarray
    body_moment: np.ndarray
    interior_momentum: np.ndarray
    interior_angular_momentum: np.ndarray
    markers: np.ndarray
    marker_velocity: np.ndarray
    history: list[dict[str, float | int]]


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _reject_unknown(section: str, values: dict[str, Any]) -> None:
    unknown = sorted(set(values).difference(_ALLOWED_KEYS[section]))
    if unknown:
        raise CaseConfigError(f"unknown {section} configuration keys: {unknown}")


def _positive(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CaseConfigError(f"{name} must be numeric") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise CaseConfigError(f"{name} must be finite and positive")
    return number


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise CaseConfigError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise CaseConfigError(f"{name} must be a positive integer") from exc
    if number <= 0 or number != float(value):
        raise CaseConfigError(f"{name} must be a positive integer")
    return number


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise CaseConfigError(f"{name} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise CaseConfigError(f"{name} must be a non-negative integer") from exc
    try:
        exact = number == float(value)
    except (TypeError, ValueError):
        exact = False
    if number < 0 or not exact:
        raise CaseConfigError(f"{name} must be a non-negative integer")
    return number


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise CaseConfigError(f"{name} must be true or false")
    return value


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge defaults, reject typos and return a normalized configuration."""
    if not isinstance(raw, dict):
        raise CaseConfigError("the case file must contain a JSON object")
    _reject_unknown("root", raw)
    for section in ("case", "runtime", "mesh", "physics", "time", "ib", "solver", "output"):
        if section in raw:
            if not isinstance(raw[section], dict):
                raise CaseConfigError(f"{section} must be a JSON object")
            _reject_unknown(section, raw[section])
    if isinstance(raw.get("ib"), dict) and "motion" in raw["ib"]:
        if not isinstance(raw["ib"]["motion"], dict):
            raise CaseConfigError("ib.motion must be a JSON object")
        _reject_unknown("motion", raw["ib"]["motion"])

    config = _deep_merge(DEFAULT_CONFIG, raw)
    if isinstance(config["schema_version"], bool):
        raise CaseConfigError("schema_version must be an integer")
    try:
        schema_version = int(config["schema_version"])
        exact_schema = schema_version == float(config["schema_version"])
    except (TypeError, ValueError):
        raise CaseConfigError("schema_version must be an integer") from None
    if not exact_schema or schema_version != SCHEMA_VERSION:
        raise CaseConfigError(
            f"unsupported schema_version={config['schema_version']}; expected {SCHEMA_VERSION}"
        )
    config["schema_version"] = schema_version
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise CaseConfigError("name must be a non-empty string")

    case_kind = str(config["case"]["kind"])
    if case_kind not in {"fixed_sphere", "oscillating_sphere", "rotating_sphere"}:
        raise CaseConfigError(f"unsupported case.kind={case_kind!r}")

    device = str(config["runtime"]["device"])
    precision = str(config["runtime"]["precision"])
    if device not in {"cpu", "cuda"}:
        raise CaseConfigError("runtime.device must be 'cpu' or 'cuda'")
    if precision not in {"float64", "float32"}:
        raise CaseConfigError("runtime.precision must be 'float64' or 'float32'")

    mesh = config["mesh"]
    mesh_kind = str(mesh["kind"])
    if mesh_kind not in {"uniform_box", "sphere_wake", "sphere_concentric"}:
        raise CaseConfigError(f"unsupported mesh.kind={mesh_kind!r}")
    if mesh_kind == "uniform_box":
        mesh["dx"] = _positive("mesh.dx", mesh["dx"])
        bounds = np.asarray(mesh["bounds"], dtype=float)
        if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)):
            raise CaseConfigError("mesh.bounds must be three finite [lower, upper] pairs")
        if np.any(bounds[:, 1] <= bounds[:, 0]):
            raise CaseConfigError("every mesh bound must have upper > lower")
        cells = (bounds[:, 1] - bounds[:, 0]) / mesh["dx"]
        if not np.allclose(cells, np.rint(cells), rtol=0.0, atol=1.0e-10):
            raise CaseConfigError("uniform-box side lengths must be integer multiples of mesh.dx")
        mesh["bounds"] = bounds.tolist()
    else:
        mesh["cells_per_diameter"] = _positive_int(
            "mesh.cells_per_diameter", mesh.get("cells_per_diameter", 16)
        )
        if mesh["cells_per_diameter"] % 8:
            raise CaseConfigError("nested-sphere cells_per_diameter must be a multiple of 8")
        # Remove uniform-mesh defaults so persisted configuration describes only
        # the selected mesh model.
        mesh.pop("dx", None)
        mesh.pop("bounds", None)

    physics = config["physics"]
    physics["reynolds"] = _positive("physics.reynolds", physics["reynolds"])
    physics["density"] = _positive("physics.density", physics["density"])
    physics["inflow_speed"] = float(physics["inflow_speed"])
    if not np.isfinite(physics["inflow_speed"]):
        raise CaseConfigError("physics.inflow_speed must be finite")

    time_cfg = config["time"]
    time_cfg["dt"] = _positive("time.dt", time_cfg["dt"])
    time_cfg["steps"] = _positive_int("time.steps", time_cfg["steps"])
    time_cfg["inner_iterations"] = _positive_int(
        "time.inner_iterations", time_cfg["inner_iterations"]
    )

    ib = config["ib"]
    ib["radius"] = _positive("ib.radius", ib["radius"])
    centre = np.asarray(ib["center"], dtype=float)
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise CaseConfigError("ib.center must contain three finite coordinates")
    ib["center"] = centre.tolist()
    ib["marker_spacing"] = _positive("ib.marker_spacing", ib["marker_spacing"])
    ib["alpha"] = _positive("ib.alpha", ib["alpha"])
    if ib["alpha"] > 1.0:
        raise CaseConfigError("ib.alpha must be <= 1.0")
    force_mode = str(ib["force_mode"])
    if force_mode not in {"direct", "accumulated_explicit"}:
        raise CaseConfigError(
            "ib.force_mode must be 'accumulated_explicit' or 'direct'"
        )
    ib["force_mode"] = force_mode
    ib["implicit"] = _boolean("ib.implicit", ib["implicit"])
    if force_mode == "accumulated_explicit":
        if ib["implicit"]:
            raise CaseConfigError("accumulated_explicit is an explicit IB method")
        if ib["alpha"] != 1.0:
            raise CaseConfigError("accumulated_explicit requires ib.alpha=1.0")
        if time_cfg["inner_iterations"] < 2:
            raise CaseConfigError(
                "accumulated_explicit requires at least two fluid inner iterations"
            )
    elif ib["implicit"] and ib["alpha"] > 0.7:
        raise CaseConfigError("implicit IB currently requires ib.alpha <= 0.7")

    motion = ib["motion"]
    motion_kind = str(motion["kind"])
    expected_motion = {
        "fixed_sphere": "none",
        "oscillating_sphere": "oscillation",
        "rotating_sphere": "rotation",
    }[case_kind]
    if motion_kind != expected_motion:
        raise CaseConfigError(
            f"case.kind={case_kind!r} requires ib.motion.kind={expected_motion!r}"
        )
    axis = np.asarray(motion["axis"], dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) == 0.0:
        raise CaseConfigError("ib.motion.axis must be a finite non-zero vector")
    motion["axis"] = (axis / np.linalg.norm(axis)).tolist()
    motion["amplitude"] = float(motion["amplitude"])
    motion["frequency"] = _positive("ib.motion.frequency", motion["frequency"])
    motion["phase"] = float(motion["phase"])
    if not np.isfinite(motion["amplitude"]) or motion["amplitude"] < 0.0:
        raise CaseConfigError("ib.motion.amplitude must be finite and non-negative")
    if not np.isfinite(motion["phase"]):
        raise CaseConfigError("ib.motion.phase must be finite")
    if motion_kind == "rotation":
        if "spin_ratio" not in motion:
            raise CaseConfigError("rotating_sphere requires ib.motion.spin_ratio")
        motion["spin_ratio"] = float(motion["spin_ratio"])
        if not np.isfinite(motion["spin_ratio"]) or motion["spin_ratio"] == 0.0:
            raise CaseConfigError("ib.motion.spin_ratio must be finite and non-zero")
    elif "spin_ratio" in motion:
        raise CaseConfigError("ib.motion.spin_ratio is only valid for rotating_sphere")
    if case_kind == "oscillating_sphere" and motion["amplitude"] == 0.0:
        raise CaseConfigError("oscillating_sphere requires a positive motion amplitude")
    if case_kind == "fixed_sphere" and motion["amplitude"] != 0.0:
        raise CaseConfigError("fixed_sphere requires zero motion amplitude")
    if case_kind == "rotating_sphere" and motion["amplitude"] != 0.0:
        raise CaseConfigError("rotating_sphere requires zero translational amplitude")
    if case_kind in {"fixed_sphere", "rotating_sphere"} and physics["inflow_speed"] == 0.0:
        raise CaseConfigError(f"{case_kind} requires non-zero physics.inflow_speed")

    output = config["output"]
    output["sample_every"] = _positive_int("output.sample_every", output["sample_every"])
    output["checkpoint_every"] = _nonnegative_int(
        "output.checkpoint_every", output["checkpoint_every"]
    )
    output["save_field"] = _boolean("output.save_field", output["save_field"])
    output["field_every"] = _nonnegative_int("output.field_every", output["field_every"])
    output["field_start_step"] = _nonnegative_int("output.field_start_step", output["field_start_step"])
    config["mesh"]["verbose"] = _boolean(
        "mesh.verbose", config["mesh"].get("verbose", False)
    )
    config["solver"]["sparse"] = _boolean(
        "solver.sparse", config["solver"]["sparse"]
    )
    config["solver"]["cg_check_interval"] = _positive_int(
        "solver.cg_check_interval", config["solver"]["cg_check_interval"]
    )
    config["solver"]["cg_max_iterations"] = _positive_int(
        "solver.cg_max_iterations", config["solver"]["cg_max_iterations"]
    )
    config["solver"]["fail_on_cg_nonconvergence"] = _boolean(
        "solver.fail_on_cg_nonconvergence", config["solver"]["fail_on_cg_nonconvergence"]
    )

    if mesh_kind in {"sphere_wake", "sphere_concentric"}:
        if not np.allclose(centre, 0.0) or not np.isclose(ib["radius"], 0.25):
            raise CaseConfigError(
                "nested sphere meshes require ib.center=[0,0,0] and ib.radius=0.25"
            )
    return config


def load_case_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CaseConfigError(f"cannot read case file {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CaseConfigError(f"invalid JSON in {source}: {exc}") from exc
    return validate_config(raw)


def _immutable_config(config: dict[str, Any]) -> dict[str, Any]:
    stable = copy.deepcopy(config)
    stable.pop("name", None)
    stable.pop("output", None)
    stable["time"].pop("steps", None)
    return stable


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(
        _immutable_config(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numerical_source_hashes() -> dict[str, str]:
    return {
        "case_runner": _source_hash(HERE / "ib_case.py"),
        "backend": _source_hash(HERE / "backend.py"),
        "core": _source_hash(HERE / "ib_core_np.py"),
        "driver": _source_hash(HERE / "ib_driver_np.py"),
        "immersed_boundary": _source_hash(HERE / "ib_immersed.py"),
        "interior_load": _source_hash(HERE / "ib_interior.py"),
        "kinematics": _source_hash(HERE / "ib_kinematics.py"),
        "sparse_operator": _source_hash(HERE / "ib_sparse.py"),
        "mesh_builder": _source_hash(PROJECT / "make_euler_mesh.py"),
        "sphere_wake_mesh": _source_hash(HERE / "sphere_wake_mesh.py"),
        "sphere_quiescent_mesh": _source_hash(HERE / "sphere_quiescent_mesh.py"),
    }


def _layout_fingerprint(mesh: Any, markers: np.ndarray, marker_widths: np.ndarray) -> str:
    """Hash topology, ordering, boundary labels and Lagrangian quadrature."""
    digest = hashlib.sha256()
    names = (
        "nodes", "cells", "boundary_faces", "boundary_fam", "pos_ref",
        "dx_uniform", "index_cell", "faces_to_nodes", "faces_to_cells",
        "edges_to_nodes", "faces_to_edges", "bc_node", "bc_edge", "bc_face",
        "bc_cell", "interface_pairs", "link_fe_insert",
    )
    for name in names:
        array = np.ascontiguousarray(np.asarray(getattr(mesh, name)))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    for name, value in (("markers", markers), ("marker_widths", marker_widths)):
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez(temporary, **arrays)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def fibonacci_sphere(count: int, radius: float, centre: list[float] | np.ndarray) -> np.ndarray:
    if count < 4:
        raise ValueError("a sphere surface requires at least four markers")
    index = np.arange(count, dtype=float)
    phi = np.arccos(1.0 - 2.0 * (index + 0.5) / count)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * index
    unit = np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1
    )
    return np.asarray(centre, dtype=float) + radius * unit


def rotating_sphere_velocity(
    markers: np.ndarray,
    centre: list[float] | np.ndarray,
    axis: list[float] | np.ndarray,
    spin_ratio: float,
    radius: float,
    reference_speed: float,
) -> np.ndarray:
    """Return rigid surface velocity for ``spin_ratio = Omega*R/U``."""
    points = np.asarray(markers, dtype=float)
    origin = np.asarray(centre, dtype=float)
    direction = np.asarray(axis, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or origin.shape != (3,):
        raise ValueError("markers must be (N,3) and centre must be a three-vector")
    norm = float(np.linalg.norm(direction))
    if direction.shape != (3,) or not np.isfinite(norm) or norm == 0.0:
        raise ValueError("rotation axis must be a finite non-zero three-vector")
    omega = direction / norm * float(spin_ratio) * float(reference_speed) / float(radius)
    return np.cross(omega, points - origin)


def sphere_center(config: dict[str, Any], time_value: float) -> np.ndarray:
    """Return the prescribed center of the canonical closed sphere."""

    center = np.asarray(config["ib"]["center"], dtype=float)
    motion = config["ib"]["motion"]
    if motion["kind"] != "oscillation":
        return center
    angle = (
        2.0 * np.pi * float(motion["frequency"]) * float(time_value)
        + float(motion["phase"])
    )
    displacement = (
        float(motion["amplitude"])
        * np.sin(angle)
        * np.asarray(motion["axis"], dtype=float)
    )
    return center + displacement


def _uniform_bounds_tuple(bounds: np.ndarray) -> tuple[float, ...]:
    return (
        float(bounds[0, 0]), float(bounds[0, 1]),
        float(bounds[1, 0]), float(bounds[1, 1]),
        float(bounds[2, 0]), float(bounds[2, 1]),
    )


def _validate_ib_support(config: dict[str, Any], dx: float, bounds: np.ndarray) -> None:
    ib = config["ib"]
    centre = np.asarray(ib["center"], dtype=float)
    radius = float(ib["radius"])
    motion = ib["motion"]
    excursion = np.zeros(3)
    if motion["kind"] == "oscillation":
        excursion = abs(float(motion["amplitude"])) * np.abs(np.asarray(motion["axis"]))
    # The current 3-point kernel is evaluated over the implementation's 5^3 search cube.
    clearance = radius + 2.5 * dx
    low = centre - excursion - clearance
    high = centre + excursion + clearance
    if np.any(low < bounds[:, 0] - 1.0e-12) or np.any(high > bounds[:, 1] + 1.0e-12):
        raise CaseConfigError(
            "the sphere or its IB support leaves the finest uniform mesh region; "
            "increase the domain/refined region, reduce radius/amplitude, or reduce dx"
        )


def build_runtime(config: dict[str, Any]) -> CaseRuntime:
    config = validate_config(config)
    mesh_cfg = config["mesh"]
    ib_cfg = config["ib"]
    radius = float(ib_cfg["radius"])
    centre = np.asarray(ib_cfg["center"], dtype=float)

    if mesh_cfg["kind"] == "uniform_box":
        dx = float(mesh_cfg["dx"])
        bounds = np.asarray(mesh_cfg["bounds"], dtype=float)
        _validate_ib_support(config, dx, bounds)
        mesh = M.make_nested_mesh(
            dx,
            [M._build_box(("bounds", _uniform_bounds_tuple(bounds)))],
            refinement_ratio=2,
            verbose=bool(mesh_cfg["verbose"]),
        )
        mesh_summary = {
            "kind": "uniform_box",
            "dx": dx,
            "bounds": bounds.tolist(),
            "cells": int(mesh.cells.shape[0]),
        }
    elif mesh_cfg["kind"] == "sphere_wake":
        import sphere_wake_mesh as SWM

        ddx = int(mesh_cfg["cells_per_diameter"])
        dx = 2.0 * radius / ddx
        finest_bounds_d = np.asarray(SWM.BOUNDS_D[0], dtype=float)
        bounds = np.asarray(
            [
                [finest_bounds_d[0] * 2.0 * radius, finest_bounds_d[1] * 2.0 * radius],
                [finest_bounds_d[2] * 2.0 * radius, finest_bounds_d[3] * 2.0 * radius],
                [finest_bounds_d[4] * 2.0 * radius, finest_bounds_d[5] * 2.0 * radius],
            ]
        )
        _validate_ib_support(config, dx, bounds)
        mesh = SWM.build_mesh(ddx, verbose=bool(mesh_cfg["verbose"]))
        mesh_summary = {"kind": "sphere_wake", **SWM.mesh_summary(mesh, ddx)}
    else:
        import sphere_quiescent_mesh as SQM

        ddx = int(mesh_cfg["cells_per_diameter"])
        dx = 2.0 * radius / ddx
        finest_half_width = float(SQM.HALF_WIDTHS_D[0]) * 2.0 * radius
        bounds = np.asarray(
            [[-finest_half_width, finest_half_width]] * 3, dtype=float
        )
        _validate_ib_support(config, dx, bounds)
        mesh = SQM.build_mesh(ddx, verbose=bool(mesh_cfg["verbose"]))
        raw_summary = SQM.mesh_summary(mesh, ddx)
        half_widths = raw_summary.pop("layer_half_widths_D")
        raw_summary["layer_bounds_D"] = [
            [-half, half, -half, half, -half, half] for half in half_widths
        ]
        mesh_summary = {"kind": "sphere_concentric", **raw_summary}

    core = IBCore(mesh)
    physics = config["physics"]
    inflow_speed = float(physics["inflow_speed"])
    reynolds = float(physics["reynolds"])
    density = float(physics["density"])
    reference_length = 2.0 * radius

    motion_cfg = ib_cfg["motion"]
    if inflow_speed != 0.0:
        viscosity = abs(inflow_speed) * reference_length / reynolds
        reference_speed = abs(inflow_speed)
    elif motion_cfg["kind"] == "oscillation":
        reference_speed = (
            2.0 * np.pi * float(motion_cfg["frequency"]) * float(motion_cfg["amplitude"])
        )
        viscosity = reference_speed * reference_length / reynolds
    else:
        raise CaseConfigError("a stationary sphere requires non-zero physics.inflow_speed")

    # Driver advances physical momentum (rho*u); its diffusion coefficient is
    # dynamic viscosity, whereas Re specifies the kinematic viscosity above.
    driver = Driver(core, mesh, visc=density * viscosity, rou=density)
    # Explicit numerical contract: identical CPU/GPU stopping checks, persisted
    # in config and its restart fingerprint instead of a hidden environment default.
    driver.cg_check_interval = int(config["solver"]["cg_check_interval"])
    driver.cg_max_iterations = int(config["solver"]["cg_max_iterations"])
    driver.fail_on_cg_nonconvergence = bool(config["solver"]["fail_on_cg_nonconvergence"])
    if mesh_cfg["kind"] == "sphere_concentric":
        driver.set_cavity(U_lid=0.0, spanwise_zerograd=False)
    dt = float(config["time"]["dt"])
    if config["solver"]["sparse"]:
        driver.enable_sparse(dt)

    marker_spacing = float(ib_cfg["marker_spacing"])
    marker_count = max(12, int(round(4.0 * np.pi * radius**2 / marker_spacing**2)))
    markers = fibonacci_sphere(marker_count, radius, centre)
    marker_volume_width = (4.0 * np.pi * radius**2 * dx / marker_count) ** (1.0 / 3.0)
    marker_widths = np.full(marker_count, marker_volume_width)
    layout_fingerprint = _layout_fingerprint(mesh, markers, marker_widths)
    source_hashes = _numerical_source_hashes()

    ib = ImmersedBoundary(mesh)
    desired_velocity = np.zeros((marker_count, 3))
    public_force_mode = str(ib_cfg["force_mode"])
    driver_force_mode = (
        "accumulated_explicit" if public_force_mode == "accumulated_explicit" else "replacement"
    )
    if motion_cfg["kind"] == "oscillation":
        motion = Oscillation(
            markers,
            axis=motion_cfg["axis"],
            A=float(motion_cfg["amplitude"]),
            freq=float(motion_cfg["frequency"]),
            phase=float(motion_cfg["phase"]),
        )
        markers_at_zero, desired_velocity = motion(0.0)
        driver.set_ib(
            ib, markers_at_zero, marker_widths, desired_velocity, force_mode=driver_force_mode
        )
        driver.set_motion(motion, t0=0.0)
    elif motion_cfg["kind"] == "rotation":
        desired_velocity = rotating_sphere_velocity(
            markers,
            centre,
            motion_cfg["axis"],
            float(motion_cfg["spin_ratio"]),
            radius,
            reference_speed,
        )
        driver.set_ib(
            ib, markers, marker_widths, desired_velocity, force_mode=driver_force_mode
        )
    else:
        driver.set_ib(ib, markers, marker_widths, desired_velocity, force_mode=driver_force_mode)

    driver.ib_alpha = float(ib_cfg["alpha"])
    driver.ib_implicit = bool(ib_cfg["implicit"])
    maximum_translation = np.zeros(3)
    if motion_cfg["kind"] == "oscillation":
        maximum_translation = (
            float(motion_cfg["amplitude"])
            * np.abs(np.asarray(motion_cfg["axis"], dtype=float))
        )
    load_correction = ClosedSphereLoadCorrection(
        core.cpos[: core.ncell],
        1.0 / core.vol_ci,
        reference_center=centre,
        maximum_translation=maximum_translation,
        radius=radius,
        smoothing_width=dx,
        density=density,
        force_support_width=1.5 * ib.hh,
    )
    reference_area = np.pi * radius**2
    reference_force = 0.5 * density * reference_speed**2 * reference_area
    mesh_summary.update(
        {
            "faces": int(core.nface),
            "edges": int(core.nedge),
            "markers": marker_count,
        }
    )
    return CaseRuntime(
        config=config,
        mesh=mesh,
        mesh_summary=mesh_summary,
        core=core,
        driver=driver,
        ib=ib,
        load_correction=load_correction,
        reference_markers=markers,
        reference_force=reference_force,
        reference_speed=reference_speed,
        dx=dx,
        dt=dt,
        steps=int(config["time"]["steps"]),
        inner_iterations=int(config["time"]["inner_iterations"]),
        layout_fingerprint=layout_fingerprint,
        source_hashes=source_hashes,
    )


def _initial_state(runtime: CaseRuntime):
    core = runtime.core
    inflow_speed = float(runtime.config["physics"]["inflow_speed"])
    s = inflow_speed * core.epos[:, 1] * core.te[:, 2]
    s, _, vc = runtime.driver.calc_vel(s)
    runtime.load_correction.initialize(
        vc[: core.ncell], sphere_center(runtime.config, runtime.driver.time)
    )
    return s, s.copy(), vc


def _checkpoint_payload(
    runtime: CaseRuntime, run_id: str, step: int, s, s_old
) -> dict[str, Any]:
    state = runtime.driver.ib_state_dict()
    if state is None:
        raise RuntimeError("cannot checkpoint an IB case before Driver.set_ib")
    body_load = runtime.load_correction.latest_load
    if body_load is None or runtime.load_correction.current is None:
        raise RuntimeError("cannot checkpoint before closed-body load bookkeeping")
    return {
        "checkpoint_schema": np.int64(CHECKPOINT_SCHEMA_VERSION),
        "update_phase": np.str_("post_step"),
        "run_id": np.str_(run_id),
        "config_fingerprint": np.str_(config_fingerprint(runtime.config)),
        "layout_fingerprint": np.str_(runtime.layout_fingerprint),
        "step": np.int64(step),
        "physical_time": np.float64(runtime.driver.time),
        "dt": np.float64(runtime.dt),
        "previous_dt": np.float64(runtime.dt),
        "precision": np.str_("float32" if FP32 else "float64"),
        "device": np.str_("cuda" if GPU else "cpu"),
        "source_hashes_json": np.str_(
            json.dumps(runtime.source_hashes, sort_keys=True, separators=(",", ":"))
        ),
        "s": asnumpy(s),
        "s_old": asnumpy(s_old),
        "markers": asnumpy(runtime.driver.ib_markers),
        "marker_velocity": asnumpy(runtime.driver.ib_vel),
        # Persist only the public algorithm name; legacy Driver aliases remain
        # an implementation detail and never leak into canonical artifacts.
        "ib_force_mode": np.str_(runtime.config["ib"]["force_mode"]),
        "ib_force_E": state["force_E"],
        "ib_force_L": state["force_L"],
        "ib_force_update_count": np.int64(state["force_update_count"]),
        "force_fluid": asnumpy(body_load.force_fluid),
        "moment_fluid": asnumpy(body_load.moment_fluid),
        "body_force": asnumpy(body_load.body_force),
        "body_moment": asnumpy(body_load.body_moment),
        "interior_momentum": asnumpy(body_load.interior_momenta.linear_momentum),
        "interior_angular_momentum": asnumpy(
            body_load.interior_momenta.angular_momentum
        ),
        "interior_indicator_volume": asnumpy(
            body_load.interior_momenta.indicator_volume
        ),
        "interior_momentum_rate": asnumpy(body_load.linear_momentum_rate),
        "interior_angular_momentum_rate": asnumpy(body_load.angular_momentum_rate),
    }


def save_checkpoint(
    path: Path, runtime: CaseRuntime, run_id: str, step: int, s, s_old
) -> None:
    _atomic_npz(path, **_checkpoint_payload(runtime, run_id, step, s, s_old))


def save_field_snapshot(path: Path, runtime: CaseRuntime, run_id: str, step: int, s, s_old, vc) -> None:
    """Persist one self-contained field for offline pressure and surface stress.

    Retaining the already-computed momentum residual does not run a pressure
    solve in the time loop. GPU-to-host copies occur only at selected snapshots.
    """
    driver, core = runtime.driver, runtime.core
    if driver.last_momentum_residual is None or driver.last_step_dt != runtime.dt:
        raise RuntimeError("no current-step momentum residual for pressure recovery")
    payload = _checkpoint_payload(runtime, run_id, step, s, s_old)
    for name in ("checkpoint_schema", "update_phase", "physical_time", "previous_dt"):
        payload.pop(name)
    payload.update(
        result_schema=np.int64(RESULT_SCHEMA_VERSION),
        time=np.float64(driver.time),
        vc=asnumpy(vc[:core.ncell]),
        cpos=asnumpy(core.cpos),
        cell_volume=asnumpy(1.0 / core.vol_ci),
        marker_volume_weight=asnumpy(driver.ib_ds**3),
        momentum_residual=asnumpy(driver.last_momentum_residual[:core.ncell]),
        helmholtz_relative_residual=np.float64(driver.helmholtz_relative_residual_last_step),
        helmholtz_tolerance_ratio=np.float64(driver.helmholtz_tolerance_ratio_last_step),
        face_cells=asnumpy(core.F2C),
        face_area=asnumpy(core.Af),
        index_cell=asnumpy(runtime.ib.index_cell),
        lattice_origin=asnumpy(runtime.ib.pos_ref),
        lattice_spacing=np.float64(runtime.dx),
        sphere_center=np.asarray(sphere_center(runtime.config, driver.time), dtype=real),
        sphere_radius=np.float64(runtime.config["ib"]["radius"]),
        density=np.float64(driver.rou),
        kinematic_viscosity=np.float64(driver.visc / driver.rou),
    )
    _atomic_npz(path, **payload)


def load_checkpoint(path: Path, runtime: CaseRuntime):
    try:
        checkpoint = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise CaseConfigError(f"cannot read checkpoint {path}: {exc}") from exc
    with checkpoint:
        required = {
            "checkpoint_schema", "update_phase", "run_id", "config_fingerprint",
            "layout_fingerprint", "step",
            "physical_time", "dt", "previous_dt", "precision", "device",
            "source_hashes_json", "s", "s_old", "markers", "marker_velocity",
            "ib_force_mode", "ib_force_E", "ib_force_L", "ib_force_update_count",
            "force_fluid", "moment_fluid", "body_force", "body_moment",
            "interior_momentum", "interior_angular_momentum",
            "interior_indicator_volume", "interior_momentum_rate",
            "interior_angular_momentum_rate",
        }
        missing = sorted(required.difference(checkpoint.files))
        if missing:
            raise CaseConfigError(f"checkpoint is missing fields: {missing}")
        schema = int(checkpoint["checkpoint_schema"])
        if schema != CHECKPOINT_SCHEMA_VERSION:
            raise CaseConfigError(
                f"unsupported checkpoint schema {schema}; expected {CHECKPOINT_SCHEMA_VERSION}"
            )
        phase = str(np.asarray(checkpoint["update_phase"]).item())
        if phase != "post_step":
            raise CaseConfigError(f"unsupported checkpoint update phase {phase!r}")
        old_fingerprint = str(np.asarray(checkpoint["config_fingerprint"]).item())
        new_fingerprint = config_fingerprint(runtime.config)
        if old_fingerprint != new_fingerprint:
            raise CaseConfigError("checkpoint configuration does not match the requested case")
        old_layout = str(np.asarray(checkpoint["layout_fingerprint"]).item())
        if old_layout != runtime.layout_fingerprint:
            raise CaseConfigError("checkpoint mesh/marker layout does not match this case")
        if not np.isclose(float(checkpoint["dt"]), runtime.dt, rtol=0.0, atol=0.0):
            raise CaseConfigError("checkpoint dt does not match the requested case")
        if not np.isclose(float(checkpoint["previous_dt"]), runtime.dt, rtol=0.0, atol=0.0):
            raise CaseConfigError("checkpoint previous_dt does not match the constant-step case")
        expected_precision = "float32" if FP32 else "float64"
        if str(np.asarray(checkpoint["precision"]).item()) != expected_precision:
            raise CaseConfigError("checkpoint precision does not match the requested case")
        expected_device = "cuda" if GPU else "cpu"
        if str(np.asarray(checkpoint["device"]).item()) != expected_device:
            raise CaseConfigError("checkpoint device does not match the requested case")
        try:
            old_hashes = json.loads(str(np.asarray(checkpoint["source_hashes_json"]).item()))
        except (json.JSONDecodeError, TypeError) as exc:
            raise CaseConfigError("checkpoint source identity is invalid") from exc
        if old_hashes != runtime.source_hashes:
            raise CaseConfigError("checkpoint numerical source identity does not match this solver")
        run_id = str(np.asarray(checkpoint["run_id"]).item())
        try:
            uuid.UUID(run_id)
        except (ValueError, AttributeError) as exc:
            raise CaseConfigError("checkpoint run_id is invalid") from exc
        step = int(checkpoint["step"])
        physical_time = float(checkpoint["physical_time"])
        expected_time = step * runtime.dt
        time_tolerance = max(
            1.0e-12,
            16.0 * np.finfo(np.float64).eps * max(abs(expected_time), runtime.dt) * max(step, 1),
        )
        if step < 0 or not np.isclose(
            physical_time, expected_time, rtol=0.0, atol=time_tolerance
        ):
            raise CaseConfigError("checkpoint step and physical time are inconsistent")
        s_raw = np.asarray(checkpoint["s"])
        s_old_raw = np.asarray(checkpoint["s_old"])
        if s_raw.dtype.kind != "f" or s_old_raw.dtype.kind != "f":
            raise CaseConfigError("checkpoint flow fields must use floating-point arrays")
        s = xp.asarray(s_raw, dtype=real)
        s_old = xp.asarray(s_old_raw, dtype=real)
        if s.shape != (runtime.core.nedge,) or s_old.shape != (runtime.core.nedge,):
            raise CaseConfigError("checkpoint field shapes do not match the mesh")
        markers = np.asarray(checkpoint["markers"])
        marker_velocity = np.asarray(checkpoint["marker_velocity"])
        expected_shape = tuple(runtime.driver.ib_markers.shape)
        if markers.shape != expected_shape or marker_velocity.shape != expected_shape:
            raise CaseConfigError("checkpoint marker shapes do not match the case")
        force_e = np.asarray(checkpoint["ib_force_E"])
        force_l = np.asarray(checkpoint["ib_force_L"])
        if force_e.dtype.kind != "f" or force_l.dtype.kind != "f":
            raise CaseConfigError("checkpoint IB forces must use floating-point arrays")
        if force_e.shape != (runtime.core.ncell, 3) or force_l.shape != expected_shape:
            raise CaseConfigError("checkpoint IB-force shapes do not match the case")
        vector_fields = {
            name: np.asarray(checkpoint[name])
            for name in (
                "force_fluid", "moment_fluid", "body_force", "body_moment",
                "interior_momentum", "interior_angular_momentum",
                "interior_momentum_rate", "interior_angular_momentum_rate",
            )
        }
        if any(value.dtype.kind != "f" or value.shape != (3,) for value in vector_fields.values()):
            raise CaseConfigError("checkpoint closed-body load vectors are invalid")
        indicator_volume = np.asarray(checkpoint["interior_indicator_volume"])
        if indicator_volume.dtype.kind != "f" or indicator_volume.shape != ():
            raise CaseConfigError("checkpoint interior indicator volume is invalid")
        arrays_to_check = (
            np.asarray(checkpoint["s"]), np.asarray(checkpoint["s_old"]), markers,
            marker_velocity, force_e, force_l, indicator_volume,
            *vector_fields.values(),
        )
        if not all(np.all(np.isfinite(array)) for array in arrays_to_check):
            raise CaseConfigError("checkpoint contains non-finite state")
        load_tolerance = 2.0e-5 if FP32 else 1.0e-11
        positions = asnumpy(runtime.core.cpos[:runtime.core.ncell])
        volume = asnumpy(1.0 / runtime.core.vol_ci)
        integrated_force = (force_e * volume[:, None]).sum(axis=0)
        integrated_moment = (np.cross(positions, force_e) * volume[:, None]).sum(axis=0)
        expected_loads = {
            "force_fluid": integrated_force,
            "moment_fluid": integrated_moment,
            "body_force": -integrated_force + vector_fields["interior_momentum_rate"],
            "body_moment": -integrated_moment + vector_fields["interior_angular_momentum_rate"],
        }
        if float(indicator_volume) <= 0 or any(
            not np.allclose(vector_fields[name], value, rtol=load_tolerance, atol=load_tolerance)
            for name, value in expected_loads.items()
        ):
            raise CaseConfigError("checkpoint closed-body loads disagree with its IB field or momentum rates")
        force_mode = str(np.asarray(checkpoint["ib_force_mode"]).item())
        expected_public_mode = str(runtime.config["ib"]["force_mode"])
        if force_mode != expected_public_mode:
            raise CaseConfigError("checkpoint IB-force mode does not match the case")
        driver_force_mode = (
            "accumulated_explicit" if force_mode == "accumulated_explicit" else "replacement"
        )
        force_update_count = int(checkpoint["ib_force_update_count"])
        expected_force_updates = (
            step
            if runtime.config["ib"]["force_mode"] == "accumulated_explicit"
            else step * runtime.inner_iterations
        )
        if force_update_count != expected_force_updates:
            raise CaseConfigError(
                "checkpoint IB-force update count is inconsistent with its step"
            )
    runtime.driver.time = physical_time
    runtime.driver._step_count = step
    if runtime.driver.motion is not None:
        expected_markers, expected_velocity = runtime.driver.motion(physical_time)
        tolerance = 2.0e-6 if FP32 else 1.0e-12
        if not np.allclose(markers, expected_markers, rtol=0.0, atol=tolerance):
            raise CaseConfigError("checkpoint markers disagree with prescribed motion at its time")
        if not np.allclose(marker_velocity, expected_velocity, rtol=0.0, atol=tolerance):
            raise CaseConfigError("checkpoint marker velocity disagrees with prescribed motion")
        if runtime.driver.ib_implicit:
            runtime.ib.invalidate_implicit()
    else:
        tolerance = 2.0e-6 if FP32 else 1.0e-12
        if not np.allclose(markers, runtime.reference_markers, rtol=0.0, atol=tolerance):
            raise CaseConfigError("checkpoint markers disagree with the fixed body")
        motion_cfg = runtime.config["ib"]["motion"]
        if motion_cfg["kind"] == "rotation":
            expected_velocity = rotating_sphere_velocity(
                runtime.reference_markers,
                runtime.config["ib"]["center"],
                motion_cfg["axis"],
                float(motion_cfg["spin_ratio"]),
                float(runtime.config["ib"]["radius"]),
                runtime.reference_speed,
            )
        else:
            expected_velocity = np.zeros_like(runtime.reference_markers)
        if not np.allclose(
            marker_velocity, expected_velocity, rtol=0.0, atol=tolerance
        ):
            raise CaseConfigError("checkpoint marker velocity disagrees with prescribed motion")
    runtime.driver.load_ib_state_dict(
        {
            "force_mode": driver_force_mode,
            "force_E": force_e,
            "force_L": force_l,
            "force_update_count": force_update_count,
        }
    )
    runtime.driver.ib_markers = xp.asarray(markers, dtype=real)
    runtime.driver.ib_vel = xp.asarray(marker_velocity, dtype=real)
    _, _, vc = runtime.driver.calc_vel(s)
    reconstructed = runtime.load_correction.evaluate(
        vc[: runtime.core.ncell], sphere_center(runtime.config, physical_time)
    )
    momentum_tolerance = 2.0e-5 if FP32 else 1.0e-11
    if not np.allclose(
        asnumpy(reconstructed.linear_momentum),
        vector_fields["interior_momentum"],
        rtol=momentum_tolerance,
        atol=momentum_tolerance,
    ) or not np.allclose(
        asnumpy(reconstructed.angular_momentum),
        vector_fields["interior_angular_momentum"],
        rtol=momentum_tolerance,
        atol=momentum_tolerance,
    ) or not np.isclose(
        float(asnumpy(reconstructed.indicator_volume)),
        float(indicator_volume),
        rtol=momentum_tolerance,
        atol=momentum_tolerance,
    ):
        raise CaseConfigError("checkpoint interior momentum disagrees with its flow field")
    runtime.load_correction.restore(
        vector_fields["interior_momentum"],
        vector_fields["interior_angular_momentum"],
        indicator_volume,
    )
    return step, s, s_old, vc, run_id


def _synchronize() -> None:
    if GPU:
        xp.cuda.Stream.null.synchronize()


def diagnostics(runtime: CaseRuntime, step: int, s, vc, milliseconds_per_step: float) -> dict[str, float | int]:
    core = runtime.core
    driver = runtime.driver
    body_load = runtime.load_correction.latest_load
    if body_load is None:
        raise RuntimeError("closed-body load correction was not advanced for this step")
    fluid_force = body_load.force_fluid
    body_force = asnumpy(body_load.body_force)
    force_fluid = asnumpy(body_load.force_fluid)
    body_moment = asnumpy(body_load.body_moment)
    moment_fluid = asnumpy(body_load.moment_fluid)
    interior_momentum = asnumpy(body_load.interior_momenta.linear_momentum)
    interior_angular_momentum = asnumpy(body_load.interior_momenta.angular_momentum)
    interior_momentum_rate = asnumpy(body_load.linear_momentum_rate)
    interior_angular_momentum_rate = asnumpy(body_load.angular_momentum_rate)
    coefficients = body_force / runtime.reference_force
    marker_velocity = runtime.ib.interp(vc[:core.ncell], driver.ib_markers)
    slip = xp.linalg.norm(marker_velocity - driver.ib_vel, axis=1)
    speed = xp.linalg.norm(vc[:core.ncell], axis=1)
    face_flux = driver.last_face_flux
    if face_flux is None:
        _, face_flux, _ = driver.calc_vel(s)
    divergence = core.vol_ci * core.cdiv(face_flux)
    lagrangian_force = (
        driver.ib_force_L * (driver.ib_ds**3)[:, None]
    ).sum(axis=0)
    balance_denominator = max(
        float(asnumpy(xp.linalg.norm(lagrangian_force))),
        float(asnumpy(xp.linalg.norm(fluid_force))),
        1.0e-300,
    )
    balance = float(asnumpy(xp.linalg.norm(fluid_force - lagrangian_force))) / balance_denominator
    speed_max = float(asnumpy(xp.max(speed)))
    row: dict[str, float | int] = {
        "step": int(step),
        "time": float(driver.time),
        "Cd": float(coefficients[0]),
        "Cy": float(coefficients[1]),
        "Cz": float(coefficients[2]),
        "body_force_x": float(body_force[0]),
        "body_force_y": float(body_force[1]),
        "body_force_z": float(body_force[2]),
        "force_fluid_x": float(force_fluid[0]),
        "force_fluid_y": float(force_fluid[1]),
        "force_fluid_z": float(force_fluid[2]),
        "body_moment_x": float(body_moment[0]),
        "body_moment_y": float(body_moment[1]),
        "body_moment_z": float(body_moment[2]),
        "moment_fluid_x": float(moment_fluid[0]),
        "moment_fluid_y": float(moment_fluid[1]),
        "moment_fluid_z": float(moment_fluid[2]),
        "interior_momentum_x": float(interior_momentum[0]),
        "interior_momentum_y": float(interior_momentum[1]),
        "interior_momentum_z": float(interior_momentum[2]),
        "interior_angular_momentum_x": float(interior_angular_momentum[0]),
        "interior_angular_momentum_y": float(interior_angular_momentum[1]),
        "interior_angular_momentum_z": float(interior_angular_momentum[2]),
        "interior_momentum_rate_x": float(interior_momentum_rate[0]),
        "interior_momentum_rate_y": float(interior_momentum_rate[1]),
        "interior_momentum_rate_z": float(interior_momentum_rate[2]),
        "interior_angular_momentum_rate_x": float(interior_angular_momentum_rate[0]),
        "interior_angular_momentum_rate_y": float(interior_angular_momentum_rate[1]),
        "interior_angular_momentum_rate_z": float(interior_angular_momentum_rate[2]),
        "interior_indicator_volume": float(
            asnumpy(body_load.interior_momenta.indicator_volume)
        ),
        "slip_max": float(asnumpy(xp.max(slip))),
        "slip_rms": float(asnumpy(xp.sqrt(xp.mean(slip**2)))),
        "speed_max": speed_max,
        "cfl": speed_max * runtime.dt / runtime.dx,
        "mass_residual_inf": float(asnumpy(xp.max(xp.abs(divergence)))),
        "force_balance_relerr": balance,
        "ib_force_updates": int(driver.ib_force_updates_last_step),
        "helmholtz_iterations": int(
            getattr(driver, "helmholtz_iterations_last_step", driver.last_helm_m)
        ),
        "helmholtz_relative_residual": float(driver.helmholtz_relative_residual_last_step),
        "helmholtz_tolerance_ratio": float(driver.helmholtz_tolerance_ratio_last_step),
        "milliseconds_per_step": float(milliseconds_per_step),
    }
    numeric = np.asarray([row[name] for name in HISTORY_COLUMNS], dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise FloatingPointError(f"non-finite diagnostic at step {step}: {row}")
    if balance > (2.0e-5 if FP32 else 1.0e-9):
        raise RuntimeError(f"IB force-spread balance failed at step {step}: {balance:.3e}")
    expected_updates = (
        1
        if runtime.config["ib"]["force_mode"] == "accumulated_explicit"
        else runtime.inner_iterations
    )
    if row["ib_force_updates"] != expected_updates:
        raise RuntimeError(
            "IB update-count invariant failed at step "
            f"{step}: {row['ib_force_updates']} != {expected_updates}"
        )
    return row


def _validated_history_prefix(
    path: Path, start_step: int, runtime: CaseRuntime
) -> list[dict[str, float | int]]:
    """Read the durable prefix through a checkpoint without modifying the file."""
    if not path.exists():
        raise CaseConfigError("resume requires history.csv beside checkpoint.npz")
    retained: list[dict[str, float | int]] = []
    previous_step = -1
    saw_checkpoint = False
    integer_fields = {"step", "ib_force_updates", "helmholtz_iterations"}
    expected_updates = (
        1
        if runtime.config["ib"]["force_mode"] == "accumulated_explicit"
        else runtime.inner_iterations
    )
    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != HISTORY_COLUMNS:
            raise CaseConfigError("history.csv schema does not match this runner")
        for line_number, raw in enumerate(reader, start=2):
            try:
                if raw.get(None) or any(raw.get(name) is None for name in HISTORY_COLUMNS):
                    raise ValueError("wrong number of CSV fields")
                row: dict[str, float | int] = {}
                for name in HISTORY_COLUMNS:
                    value = float(raw[name])
                    if not np.isfinite(value):
                        raise ValueError(f"non-finite {name}")
                    if name in integer_fields:
                        integer = int(value)
                        if integer != value:
                            raise ValueError(f"non-integer {name}")
                        row[name] = integer
                    else:
                        row[name] = value
                step = int(row["step"])
                if step <= previous_step:
                    raise ValueError("steps are not strictly increasing")
                if step > start_step:
                    if saw_checkpoint:
                        break
                    raise ValueError("history skipped the checkpoint step")
                expected_time = step * runtime.dt
                tolerance = max(
                    1.0e-12,
                    16.0 * np.finfo(np.float64).eps * max(abs(expected_time), runtime.dt) * max(step, 1),
                )
                if not np.isclose(float(row["time"]), expected_time, rtol=0.0, atol=tolerance):
                    raise ValueError("time is inconsistent with step and dt")
                if int(row["ib_force_updates"]) != expected_updates:
                    raise ValueError("IB update count is inconsistent with the configured method")
            except (KeyError, TypeError, ValueError) as exc:
                # A torn final append after a durable checkpoint is recoverable.
                if saw_checkpoint:
                    break
                raise CaseConfigError(
                    f"invalid history.csv row {line_number} before checkpoint: {exc}"
                ) from exc
            retained.append(row)
            previous_step = step
            if step == start_step:
                saw_checkpoint = True
    if not saw_checkpoint:
        raise CaseConfigError("history.csv does not contain the checkpoint step")
    return retained


def _atomic_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=HISTORY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prepare_history(
    path: Path, retained: list[dict[str, float | int]]
) -> tuple[Any, csv.DictWriter]:
    _atomic_history(path, retained)
    handle = path.open("a", newline="", encoding="utf-8")
    return handle, csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)


def _read_history(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | int] = {}
            for name in HISTORY_COLUMNS:
                row[name] = int(raw[name]) if name in {"step", "ib_force_updates", "helmholtz_iterations"} else float(raw[name])
            rows.append(row)
    return rows


def _metadata(
    runtime: CaseRuntime,
    run_id: str,
    output_directory: Path,
    start_step: int,
    resume: Path | None,
) -> dict[str, Any]:
    backend_version = xp.__version__
    return {
        "result_schema": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_name": runtime.config["name"],
        "case_kind": runtime.config["case"]["kind"],
        "backend": "cuda/cupy" if GPU else "cpu/numpy",
        "precision": "float32" if FP32 else "float64",
        "python": platform.python_version(),
        "array_backend_version": backend_version,
        "config_fingerprint": config_fingerprint(runtime.config),
        "layout_fingerprint": runtime.layout_fingerprint,
        "config": runtime.config,
        "mesh": runtime.mesh_summary,
        "dt": runtime.dt,
        "target_steps": runtime.steps,
        "start_step": start_step,
        "resumed_from": str(resume.resolve()) if resume else None,
        "output_directory": str(output_directory.resolve()),
        "source_sha256": runtime.source_hashes,
        "closed_body_load": {
            "method": "Uhlmann interior-fluid momentum correction",
            "force": "-force_fluid + backward_difference(interior_momentum)",
            "moment": "-moment_fluid + backward_difference(interior_angular_momentum)",
            "indicator": "analytic sphere with one-cell linear smoothing",
            "moment_origin": [0.0, 0.0, 0.0],
        },
    }


def _read_resume_metadata(
    path: Path, runtime: CaseRuntime, run_id: str
) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaseConfigError(f"cannot read existing metadata.json: {exc}") from exc
    expected = {
        "result_schema": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "config_fingerprint": config_fingerprint(runtime.config),
        "layout_fingerprint": runtime.layout_fingerprint,
        "backend": "cuda/cupy" if GPU else "cpu/numpy",
        "precision": "float32" if FP32 else "float64",
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise CaseConfigError(f"existing metadata {name} does not match the checkpoint")
    if metadata.get("source_sha256") != runtime.source_hashes:
        raise CaseConfigError("existing metadata source identity does not match this solver")
    if not isinstance(metadata.get("created_utc"), str) or not metadata["created_utc"]:
        raise CaseConfigError("existing metadata created_utc is invalid")
    try:
        old_dt = float(metadata["dt"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CaseConfigError("existing metadata dt is invalid") from exc
    if not np.isclose(old_dt, runtime.dt, rtol=0.0, atol=0.0):
        raise CaseConfigError("existing metadata dt does not match the checkpoint")
    return metadata


def _archive_known_outputs(outdir: Path, paths: tuple[Path, ...]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = outdir / "archive" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        os.replace(path, archive / path.name)
    _fsync_parent(archive)
    return archive


def run_case(
    config: dict[str, Any],
    output_directory: str | os.PathLike[str],
    *,
    resume: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    quiet: bool = False,
) -> RunResult:
    """Run one complete single-device IB case and persist reproducible artifacts."""
    config = validate_config(config)
    expected_gpu = config["runtime"]["device"] == "cuda"
    expected_fp32 = config["runtime"]["precision"] == "float32"
    if expected_gpu != GPU or expected_fp32 != FP32:
        raise RuntimeError(
            "backend was imported with different settings; start a fresh process through "
            "cfdagent_ib.py so runtime.device/runtime.precision are applied before import"
        )

    outdir = Path(output_directory)
    outdir.mkdir(parents=True, exist_ok=True)
    history_path = outdir / "history.csv"
    checkpoint_path = outdir / "checkpoint.npz"
    metadata_path = outdir / "metadata.json"
    final_field_path = outdir / "final_field.npz"
    snapshots_path = outdir / "snapshots"
    known_outputs = (history_path, checkpoint_path, metadata_path, final_field_path, snapshots_path)
    resume_path = Path(resume) if resume is not None else None
    if resume_path is not None and resume_path.resolve() != checkpoint_path.resolve():
        raise CaseConfigError(
            "resume must use checkpoint.npz in the requested output directory"
        )
    if resume_path is None and not overwrite and any(path.exists() for path in known_outputs):
        raise FileExistsError(
            f"{outdir} already contains run artifacts; pass overwrite=True or resume a checkpoint"
        )

    setup_start = time.perf_counter()
    runtime = build_runtime(config)
    if resume_path is not None:
        start_step, s, s_old, vc, run_id = load_checkpoint(resume_path, runtime)
        if start_step >= runtime.steps:
            raise CaseConfigError(
                f"target steps ({runtime.steps}) must exceed checkpoint step ({start_step})"
            )
        retained_history = _validated_history_prefix(history_path, start_step, runtime)
        previous_metadata = _read_resume_metadata(metadata_path, runtime, run_id)
        _archive_known_outputs(outdir, (final_field_path,))
        # A crash can leave a snapshot newer than the last durable checkpoint.
        # Preserve such files in an archive instead of mixing different futures.
        stale_snapshots = tuple(
            p for p in snapshots_path.glob("field_*.npz")
            if p.stem[6:].isdigit() and int(p.stem[6:]) > start_step
        )
        _archive_known_outputs(outdir, stale_snapshots)
    else:
        start_step = 0
        s, s_old, vc = _initial_state(runtime)
        run_id = str(uuid.uuid4())
        retained_history = []
        previous_metadata = None
        if overwrite:
            _archive_known_outputs(outdir, known_outputs)

    metadata = _metadata(runtime, run_id, outdir, start_step, resume_path)
    if previous_metadata is not None:
        metadata["created_utc"] = previous_metadata["created_utc"]
        metadata["resume_count"] = int(previous_metadata.get("resume_count", 0)) + 1
        metadata["last_resume_utc"] = datetime.now(timezone.utc).isoformat()
    else:
        metadata["resume_count"] = 0
    metadata["setup_seconds"] = time.perf_counter() - setup_start
    _atomic_json(metadata_path, metadata)
    history_handle, history_writer = _prepare_history(history_path, retained_history)

    stop = {"requested": False}
    old_handlers: dict[int, Any] = {}

    def request_stop(signum, _frame):
        stop["requested"] = True
        if not quiet:
            print(f"[signal] received {signum}; stopping after the current step", flush=True)

    for signal_name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, signal_name):
            signum = int(getattr(signal, signal_name))
            try:
                old_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, request_stop)
            except (OSError, ValueError):
                pass

    completed_step = start_step
    interval_start = time.perf_counter()
    interval_step = start_step
    if not quiet:
        print(
            f"CFDAgent-IB case={config['name']} backend={'cuda/cupy' if GPU else 'cpu/numpy'} "
            f"dtype={'float32' if FP32 else 'float64'} cells={runtime.core.ncell:,} "
            f"edges={runtime.core.nedge:,} markers={runtime.reference_markers.shape[0]:,} "
            f"steps={start_step + 1}..{runtime.steps}",
            flush=True,
        )
        print(f"output={outdir.resolve()}", flush=True)

    try:
        for step in range(start_step + 1, runtime.steps + 1):
            # Anchor time to the integer step so long runs and restarts are identical.
            runtime.driver.time = (step - 1) * runtime.dt
            s, vc, s_old = runtime.driver.step(
                s, s_old, runtime.dt, dt_old=runtime.dt, inner=runtime.inner_iterations
            )
            runtime.driver.time = step * runtime.dt
            runtime.load_correction.advance(
                vc[: runtime.core.ncell],
                sphere_center(runtime.config, runtime.driver.time),
                runtime.dt,
                runtime.driver.ib_force_E,
            )
            completed_step = step
            checkpoint_every = int(config["output"]["checkpoint_every"])
            checkpoint_due = bool(
                (checkpoint_every and step % checkpoint_every == 0)
                or step == runtime.steps
                or stop["requested"]
            )
            sample = (
                step <= 3
                or step % int(config["output"]["sample_every"]) == 0
                or step == runtime.steps
                or checkpoint_due
            )
            if sample:
                _synchronize()
                now = time.perf_counter()
                milliseconds_per_step = (now - interval_start) * 1000.0 / max(
                    step - interval_step, 1
                )
                interval_start = now
                interval_step = step
                row = diagnostics(runtime, step, s, vc, milliseconds_per_step)
                history_writer.writerow(row)
                history_handle.flush()
                if not quiet:
                    print(
                        f"step={step:6d} t={row['time']:9.5f} Cd={row['Cd']:10.5f} "
                        f"slip={row['slip_max']:.4e} mass={row['mass_residual_inf']:.2e} "
                        f"CG={row['helmholtz_iterations']:4d} "
                        f"{row['milliseconds_per_step']:8.1f} ms/step",
                        flush=True,
                    )

            if checkpoint_due:
                # History must be durable BEFORE the checkpoint naming its row.
                # Ordinary samples only flush userspace buffers, avoiding fsync
                # latency on every sampled step (especially network storage).
                history_handle.flush()
                os.fsync(history_handle.fileno())
                save_checkpoint(checkpoint_path, runtime, run_id, step, s, s_old)
            field_every = int(config["output"]["field_every"])
            if field_every and step >= int(config["output"]["field_start_step"]) and step % field_every == 0 and not (
                step == runtime.steps and config["output"]["save_field"]
            ):
                snapshots_path.mkdir(exist_ok=True)
                save_field_snapshot(
                    snapshots_path / f"field_{step:09d}.npz", runtime, run_id, step, s, s_old, vc
                )
            # A signal can arrive after checkpoint_due was first evaluated.
            # Persist this completed step before honoring it.
            if stop["requested"] and not checkpoint_due:
                if not sample:
                    _synchronize()
                    now = time.perf_counter()
                    milliseconds_per_step = (now - interval_start) * 1000.0 / max(
                        step - interval_step, 1
                    )
                    interval_start = now
                    interval_step = step
                    row = diagnostics(runtime, step, s, vc, milliseconds_per_step)
                    history_writer.writerow(row)
                history_handle.flush()
                os.fsync(history_handle.fileno())
                save_checkpoint(checkpoint_path, runtime, run_id, step, s, s_old)
            if stop["requested"]:
                break
    finally:
        history_handle.close()
        for signum, handler in old_handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                pass

    if completed_step == runtime.steps and bool(config["output"]["save_field"]):
        save_field_snapshot(final_field_path, runtime, run_id, completed_step, s, s_old, vc)

    metadata.update(
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "completed_step": completed_step,
            "complete": completed_step == runtime.steps,
        }
    )
    _atomic_json(metadata_path, metadata)
    history = _read_history(history_path)
    body_load = runtime.load_correction.latest_load
    if body_load is None:
        raise RuntimeError("closed-body load state is unavailable at run completion")
    if not quiet:
        print(
            f"CFDAgent-IB {'COMPLETE' if completed_step == runtime.steps else 'STOPPED'} "
            f"step={completed_step} time={runtime.driver.time:.6f}",
            flush=True,
        )
    return RunResult(
        output_directory=outdir,
        start_step=start_step,
        completed_step=completed_step,
        target_steps=runtime.steps,
        stopped=completed_step != runtime.steps,
        s=asnumpy(s).copy(),
        s_old=asnumpy(s_old).copy(),
        vc=asnumpy(vc[: runtime.core.ncell]).copy(),
        ib_force_E=asnumpy(runtime.driver.ib_force_E).copy(),
        ib_force_L=asnumpy(runtime.driver.ib_force_L).copy(),
        body_force=asnumpy(body_load.body_force).copy(),
        body_moment=asnumpy(body_load.body_moment).copy(),
        interior_momentum=asnumpy(
            body_load.interior_momenta.linear_momentum
        ).copy(),
        interior_angular_momentum=asnumpy(
            body_load.interior_momenta.angular_momentum
        ).copy(),
        markers=asnumpy(runtime.driver.ib_markers).copy(),
        marker_velocity=asnumpy(runtime.driver.ib_vel).copy(),
        history=history,
    )

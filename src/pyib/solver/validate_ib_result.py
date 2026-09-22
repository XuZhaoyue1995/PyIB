"""Strict acceptance checks for artifacts produced by :mod:`ib_case`.

The validator is intentionally independent of the numerical backend: it never
imports CuPy or rebuilds the mesh.  It validates the durable evidence written by
``ib_case.run_case`` and cross-checks every value duplicated across its three
mandatory artifacts and optional final-field artifact.

Usage::

    python validate_ib_result.py PATH_TO_RESULT_DIRECTORY
    python validate_ib_result.py PATH_TO_RESULT_DIRECTORY \
        --force-balance-atol 2e-5

Exit status is zero only when all checks pass.  Invalid results and unexpected
validator failures return a non-zero status and print a concise reason to
stderr.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
import uuid
import zipfile

import numpy as np


RESULT_SCHEMA_VERSION = 5
CHECKPOINT_SCHEMA_VERSION = 3

HISTORY_COLUMNS = (
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
)

INTEGER_HISTORY_COLUMNS = {"step", "ib_force_updates", "helmholtz_iterations"}
NONNEGATIVE_HISTORY_COLUMNS = {
    "helmholtz_relative_residual", "helmholtz_tolerance_ratio",
    "slip_max",
    "slip_rms",
    "speed_max",
    "cfl",
    "mass_residual_inf",
    "force_balance_relerr",
    "ib_force_updates",
    "helmholtz_iterations",
    "milliseconds_per_step",
    "interior_indicator_volume",
}

METADATA_FIELDS = {
    "result_schema",
    "run_id",
    "created_utc",
    "case_name",
    "case_kind",
    "backend",
    "precision",
    "python",
    "array_backend_version",
    "config_fingerprint",
    "layout_fingerprint",
    "config",
    "mesh",
    "dt",
    "target_steps",
    "start_step",
    "resumed_from",
    "output_directory",
    "source_sha256",
    "closed_body_load",
    "resume_count",
    "setup_seconds",
    "completed_utc",
    "completed_step",
    "complete",
}
METADATA_OPTIONAL_FIELDS = {"last_resume_utc"}

CHECKPOINT_FIELDS = {
    "checkpoint_schema",
    "update_phase",
    "run_id",
    "config_fingerprint",
    "layout_fingerprint",
    "step",
    "physical_time",
    "dt",
    "previous_dt",
    "precision",
    "device",
    "source_hashes_json",
    "s",
    "s_old",
    "markers",
    "marker_velocity",
    "ib_force_mode",
    "ib_force_E",
    "ib_force_L",
    "ib_force_update_count",
    "force_fluid",
    "moment_fluid",
    "body_force",
    "body_moment",
    "interior_momentum",
    "interior_angular_momentum",
    "interior_indicator_volume",
    "interior_momentum_rate",
    "interior_angular_momentum_rate",
}

FINAL_FIELD_FIELDS = {
    "helmholtz_relative_residual", "helmholtz_tolerance_ratio",
    "momentum_residual", "face_cells", "face_area", "index_cell",
    "lattice_origin", "lattice_spacing", "sphere_center", "sphere_radius",
    "density", "kinematic_viscosity",
    "result_schema",
    "run_id",
    "step",
    "time",
    "dt",
    "s",
    "s_old",
    "vc",
    "cpos",
    "cell_volume",
    "markers",
    "marker_velocity",
    "marker_volume_weight",
    "ib_force_mode",
    "ib_force_E",
    "ib_force_L",
    "ib_force_update_count",
    "force_fluid",
    "moment_fluid",
    "body_force",
    "body_moment",
    "interior_momentum",
    "interior_angular_momentum",
    "interior_indicator_volume",
    "interior_momentum_rate",
    "interior_angular_momentum_rate",
    "config_fingerprint",
    "layout_fingerprint",
    "source_hashes_json",
    "device",
    "precision",
}

CONFIG_FIELDS = {
    "schema_version",
    "name",
    "case",
    "runtime",
    "mesh",
    "physics",
    "time",
    "ib",
    "solver",
    "output",
}

SOURCE_FILES = {
    "case_runner": "ib_case.py",
    "backend": "backend.py",
    "core": "ib_core_np.py",
    "driver": "ib_driver_np.py",
    "immersed_boundary": "ib_immersed.py",
    "interior_load": "ib_interior.py",
    "kinematics": "ib_kinematics.py",
    "sparse_operator": "ib_sparse.py",
    "mesh_builder": "../make_euler_mesh.py",
    "sphere_wake_mesh": "sphere_wake_mesh.py",
    "sphere_quiescent_mesh": "sphere_quiescent_mesh.py",
}
NUMERICAL_SOURCE_KEYS = set(SOURCE_FILES)

DEFAULT_FORCE_BALANCE_ATOL = {
    "float64": 1.0e-9,
    "float32": 2.0e-5,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")


class ValidationError(RuntimeError):
    """A persisted result violates the acceptance contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))
    _require(not missing and not extra, f"{label} schema mismatch: missing={missing}, extra={extra}")


def _allowed_keys(
    value: Mapping[str, Any], required: set[str], allowed: set[str], label: str
) -> None:
    actual = set(value)
    missing = sorted(required.difference(actual))
    extra = sorted(actual.difference(allowed))
    _require(not missing and not extra, f"{label} schema mismatch: missing={missing}, extra={extra}")


def _json_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    if minimum is not None:
        _require(value >= minimum, f"{label} must be >= {minimum}")
    return value


def _json_number(value: Any, label: str, *, positive: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    number = float(value)
    _require(math.isfinite(number), f"{label} must be finite")
    if positive:
        _require(number > 0.0, f"{label} must be positive")
    return number


def _json_string(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be a non-empty string")
    return value


def _check_json_finite(value: Any, label: str) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains a non-finite number")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_json_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_json_finite(item, f"{label}[{index}]")


def _strict_json_loads(text: str, label: str) -> Any:
    def reject_constant(token: str) -> None:
        raise ValidationError(f"{label} contains non-standard JSON constant {token!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except ValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {label}: {exc}") from exc
    _check_json_finite(value, label)
    return value


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read {path.name}: {exc}") from exc
    _require(bool(text.strip()), f"{path.name} is empty")
    metadata = _as_mapping(_strict_json_loads(text, path.name), path.name)
    _allowed_keys(
        metadata,
        METADATA_FIELDS,
        METADATA_FIELDS | METADATA_OPTIONAL_FIELDS,
        "metadata.json",
    )
    return dict(metadata)


def _parse_utc(value: Any, label: str) -> datetime:
    text = _json_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{label} is not a valid ISO-8601 timestamp") from exc
    _require(parsed.tzinfo is not None, f"{label} must include a timezone")
    _require(parsed.utcoffset() is not None, f"{label} has an invalid timezone")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValidationError(f"cannot hash numerical source {path}: {exc}") from exc
    return digest.hexdigest()


def _validate_sha256(value: Any, label: str) -> str:
    text = _json_string(value, label)
    _require(bool(_SHA256_RE.fullmatch(text)), f"{label} must be a lowercase SHA-256 digest")
    return text


def _validate_uuid(value: Any, label: str) -> str:
    text = _json_string(value, label)
    try:
        parsed = uuid.UUID(text)
    except (ValueError, AttributeError) as exc:
        raise ValidationError(f"{label} must be a UUID") from exc
    _require(str(parsed) == text, f"{label} must use canonical lowercase UUID syntax")
    return text


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    stable = copy.deepcopy(dict(config))
    stable.pop("name", None)
    stable.pop("output", None)
    _as_mapping(stable.get("time"), "config.time").pop("steps", None)
    try:
        payload = json.dumps(
            stable,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"config cannot be fingerprinted: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _validate_config(config_value: Any) -> dict[str, Any]:
    config = dict(_as_mapping(config_value, "metadata.config"))
    _exact_keys(config, CONFIG_FIELDS, "metadata.config")
    _require(_json_int(config["schema_version"], "config.schema_version") == 1, "unsupported config schema")
    _json_string(config["name"], "config.name")

    case = _as_mapping(config["case"], "config.case")
    _exact_keys(case, {"kind"}, "config.case")
    case_kind = _json_string(case["kind"], "config.case.kind")
    _require(
        case_kind in {"fixed_sphere", "oscillating_sphere", "rotating_sphere"},
        "unsupported config.case.kind",
    )

    runtime = _as_mapping(config["runtime"], "config.runtime")
    _exact_keys(runtime, {"device", "precision"}, "config.runtime")
    device = _json_string(runtime["device"], "config.runtime.device")
    precision = _json_string(runtime["precision"], "config.runtime.precision")
    _require(device in {"cpu", "cuda"}, "config.runtime.device must be cpu or cuda")
    _require(precision in DEFAULT_FORCE_BALANCE_ATOL, "unsupported config.runtime.precision")

    mesh = _as_mapping(config["mesh"], "config.mesh")
    common_mesh_fields = {"kind", "verbose"}
    _allowed_keys(
        mesh,
        common_mesh_fields,
        common_mesh_fields | {"dx", "bounds", "cells_per_diameter"},
        "config.mesh",
    )
    mesh_kind = _json_string(mesh["kind"], "config.mesh.kind")
    _require(
        mesh_kind in {"uniform_box", "sphere_wake", "sphere_concentric"},
        "unsupported config.mesh.kind",
    )
    if mesh_kind == "uniform_box":
        _allowed_keys(
            mesh,
            common_mesh_fields | {"dx", "bounds"},
            common_mesh_fields | {"dx", "bounds", "cells_per_diameter"},
            "config.mesh",
        )
        mesh_dx = _json_number(mesh["dx"], "config.mesh.dx", positive=True)
        try:
            bounds = np.asarray(mesh["bounds"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValidationError("config.mesh.bounds must be numeric") from exc
        _require(
            bounds.shape == (3, 2) and np.all(np.isfinite(bounds)),
            "config.mesh.bounds schema mismatch",
        )
        _require(
            np.all(bounds[:, 1] > bounds[:, 0]),
            "config.mesh.bounds must have upper > lower",
        )
        uniform_cell_counts = (bounds[:, 1] - bounds[:, 0]) / mesh_dx
        _require(
            np.allclose(uniform_cell_counts, np.rint(uniform_cell_counts), rtol=0.0, atol=1.0e-10),
            "uniform-box side lengths must be integer multiples of config.mesh.dx",
        )
        if "cells_per_diameter" in mesh:
            _json_int(
                mesh["cells_per_diameter"],
                "config.mesh.cells_per_diameter",
                minimum=1,
            )
    else:
        _exact_keys(
            mesh,
            common_mesh_fields | {"cells_per_diameter"},
            "config.mesh",
        )
        cells_per_diameter = _json_int(
            mesh["cells_per_diameter"], "config.mesh.cells_per_diameter", minimum=1
        )
        _require(
            cells_per_diameter % 8 == 0,
            "config.mesh.cells_per_diameter must be a multiple of 8",
        )
    _require(isinstance(mesh["verbose"], bool), "config.mesh.verbose must be boolean")

    physics = _as_mapping(config["physics"], "config.physics")
    _exact_keys(physics, {"reynolds", "density", "inflow_speed"}, "config.physics")
    _json_number(physics["reynolds"], "config.physics.reynolds", positive=True)
    _json_number(physics["density"], "config.physics.density", positive=True)
    _json_number(physics["inflow_speed"], "config.physics.inflow_speed")

    time_cfg = _as_mapping(config["time"], "config.time")
    _exact_keys(time_cfg, {"dt", "steps", "inner_iterations"}, "config.time")
    _json_number(time_cfg["dt"], "config.time.dt", positive=True)
    _json_int(time_cfg["steps"], "config.time.steps", minimum=1)
    _json_int(time_cfg["inner_iterations"], "config.time.inner_iterations", minimum=1)

    ib = _as_mapping(config["ib"], "config.ib")
    _exact_keys(
        ib,
        {"radius", "center", "marker_spacing", "force_mode", "implicit", "alpha", "motion"},
        "config.ib",
    )
    _json_number(ib["radius"], "config.ib.radius", positive=True)
    try:
        centre = np.asarray(ib["center"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError("config.ib.center must be numeric") from exc
    _require(centre.shape == (3,) and np.all(np.isfinite(centre)), "config.ib.center schema mismatch")
    _json_number(ib["marker_spacing"], "config.ib.marker_spacing", positive=True)
    force_mode = _json_string(ib["force_mode"], "config.ib.force_mode")
    _require(
        force_mode in {"direct", "accumulated_explicit"},
        "config.ib.force_mode must be direct or accumulated_explicit",
    )
    _require(isinstance(ib["implicit"], bool), "config.ib.implicit must be boolean")
    alpha = _json_number(ib["alpha"], "config.ib.alpha", positive=True)
    _require(alpha <= 1.0, "config.ib.alpha must be <= 1.0")
    inner_iterations = int(time_cfg["inner_iterations"])
    if ib["implicit"]:
        _require(alpha <= 0.7, "implicit IB requires config.ib.alpha <= 0.7")
    if force_mode == "accumulated_explicit":
        _require(not ib["implicit"], "accumulated_explicit must be explicit")
        _require(alpha == 1.0, "accumulated_explicit requires config.ib.alpha=1.0")
        _require(
            inner_iterations >= 2,
            "accumulated_explicit requires at least two inner iterations",
        )
    motion = _as_mapping(ib["motion"], "config.ib.motion")
    motion_kind = _json_string(motion.get("kind"), "config.ib.motion.kind")
    common_motion_fields = {"kind", "axis", "amplitude", "frequency", "phase"}
    if motion_kind == "rotation":
        _exact_keys(motion, common_motion_fields | {"spin_ratio"}, "config.ib.motion")
    else:
        _exact_keys(motion, common_motion_fields, "config.ib.motion")
    _require(
        motion_kind in {"none", "oscillation", "rotation"},
        "unsupported config.ib.motion.kind",
    )
    try:
        axis = np.asarray(motion["axis"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError("config.ib.motion.axis must be numeric") from exc
    _require(
        axis.shape == (3,) and np.all(np.isfinite(axis)) and np.linalg.norm(axis) > 0.0,
        "config.ib.motion.axis must be a finite non-zero vector",
    )
    _require(
        math.isclose(float(np.linalg.norm(axis)), 1.0, rel_tol=0.0, abs_tol=1.0e-12),
        "normalized config.ib.motion.axis must have unit length",
    )
    amplitude = _json_number(motion["amplitude"], "config.ib.motion.amplitude")
    _require(amplitude >= 0.0, "config.ib.motion.amplitude must be non-negative")
    _json_number(motion["frequency"], "config.ib.motion.frequency", positive=True)
    _json_number(motion["phase"], "config.ib.motion.phase")
    expected_motion = {
        "fixed_sphere": "none",
        "oscillating_sphere": "oscillation",
        "rotating_sphere": "rotation",
    }[case_kind]
    _require(
        motion_kind == expected_motion,
        f"config.case.kind={case_kind} requires motion.kind={expected_motion}",
    )
    if case_kind == "fixed_sphere":
        _require(amplitude == 0.0, "fixed_sphere requires zero motion amplitude")
        _require(
            float(physics["inflow_speed"]) != 0.0,
            "a stationary sphere requires non-zero config.physics.inflow_speed",
        )
    elif case_kind == "oscillating_sphere":
        _require(amplitude > 0.0, "oscillating_sphere requires positive motion amplitude")
    else:
        _require(amplitude == 0.0, "rotating_sphere requires zero translational amplitude")
        spin_ratio = _json_number(motion["spin_ratio"], "config.ib.motion.spin_ratio")
        _require(spin_ratio != 0.0, "config.ib.motion.spin_ratio must be non-zero")
        _require(
            float(physics["inflow_speed"]) != 0.0,
            "a rotating sphere requires non-zero config.physics.inflow_speed",
        )

    if mesh_kind in {"sphere_wake", "sphere_concentric"}:
        _require(np.allclose(centre, 0.0), "nested sphere mesh requires config.ib.center=[0,0,0]")
        _require(
            math.isclose(float(ib["radius"]), 0.25, rel_tol=0.0, abs_tol=0.0),
            "nested sphere mesh requires config.ib.radius=0.25",
        )

    solver = _as_mapping(config["solver"], "config.solver")
    _exact_keys(solver, {"sparse", "cg_check_interval", "cg_max_iterations", "fail_on_cg_nonconvergence"}, "config.solver")
    _require(isinstance(solver["sparse"], bool), "config.solver.sparse must be boolean")
    _json_int(solver["cg_check_interval"], "config.solver.cg_check_interval", minimum=1)
    _json_int(solver["cg_max_iterations"], "config.solver.cg_max_iterations", minimum=1)
    _require(isinstance(solver["fail_on_cg_nonconvergence"], bool), "config.solver.fail_on_cg_nonconvergence must be boolean")
    output = _as_mapping(config["output"], "config.output")
    _exact_keys(output, {"sample_every", "checkpoint_every", "field_every", "field_start_step", "save_field"}, "config.output")
    _json_int(output["sample_every"], "config.output.sample_every", minimum=1)
    _json_int(output["checkpoint_every"], "config.output.checkpoint_every", minimum=0)
    _json_int(output["field_every"], "config.output.field_every", minimum=0)
    _json_int(output["field_start_step"], "config.output.field_start_step", minimum=0)
    _require(isinstance(output["save_field"], bool), "config.output.save_field must be boolean")
    return config


def _validate_mesh_metadata(mesh_value: Any, config: Mapping[str, Any]) -> tuple[int, int, int]:
    mesh = _as_mapping(mesh_value, "metadata.mesh")
    kind = config["mesh"]["kind"]
    common = {"kind", "faces", "edges", "markers"}
    if kind == "uniform_box":
        expected = common | {"dx", "bounds", "cells"}
        _exact_keys(mesh, expected, "metadata.mesh")
        cells = _json_int(mesh["cells"], "metadata.mesh.cells", minimum=1)
        mesh_dx = _json_number(mesh["dx"], "metadata.mesh.dx", positive=True)
        _require(mesh_dx == float(config["mesh"]["dx"]), "metadata.mesh.dx disagrees with config")
        try:
            mesh_bounds = np.asarray(mesh["bounds"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValidationError("metadata.mesh.bounds must be numeric") from exc
        _require(
            mesh_bounds.shape == (3, 2)
            and np.all(np.isfinite(mesh_bounds))
            and np.array_equal(mesh_bounds, np.asarray(config["mesh"]["bounds"], dtype=float)),
            "metadata.mesh.bounds disagrees with config",
        )
        expected_counts = np.rint(
            (mesh_bounds[:, 1] - mesh_bounds[:, 0]) / mesh_dx
        ).astype(np.int64)
        _require(
            cells == int(np.prod(expected_counts, dtype=np.int64)),
            "metadata.mesh.cells disagrees with uniform bounds and dx",
        )
    else:
        expected = common | {
            "diameter",
            "D_over_dx",
            "dx_fine",
            "outer_domain_D",
            "layer_bounds_D",
            "layer_spacing_D",
            "layer_cells",
            "expected_layer_cells",
            "total_cells",
            "expected_total_cells",
            "nodes",
        }
        _exact_keys(mesh, expected, "metadata.mesh")
        cells = _json_int(mesh["total_cells"], "metadata.mesh.total_cells", minimum=1)
        expected_cells = _json_int(
            mesh["expected_total_cells"], "metadata.mesh.expected_total_cells", minimum=1
        )
        _require(cells == expected_cells, "sphere_wake total cell count failed its analytic check")
        layer_cells = mesh["layer_cells"]
        expected_layers = mesh["expected_layer_cells"]
        _require(isinstance(layer_cells, list), "metadata.mesh.layer_cells must be an array")
        _require(isinstance(expected_layers, list), "metadata.mesh.expected_layer_cells must be an array")
        for index, value in enumerate(layer_cells):
            _json_int(value, f"metadata.mesh.layer_cells[{index}]", minimum=1)
        for index, value in enumerate(expected_layers):
            _json_int(value, f"metadata.mesh.expected_layer_cells[{index}]", minimum=1)
        _require(bool(layer_cells), "metadata.mesh.layer_cells must not be empty")
        _require(layer_cells == expected_layers, "sphere_wake layer cell counts failed their analytic check")
        _require(sum(layer_cells) == cells, "sphere_wake layer cells do not sum to total_cells")
        diameter_cells = _json_int(mesh["D_over_dx"], "metadata.mesh.D_over_dx", minimum=1)
        _require(
            diameter_cells == config["mesh"]["cells_per_diameter"],
            "metadata.mesh.D_over_dx disagrees with config",
        )
        diameter = _json_number(mesh["diameter"], "metadata.mesh.diameter", positive=True)
        _require(
            math.isclose(diameter, 2.0 * float(config["ib"]["radius"]), rel_tol=0.0, abs_tol=0.0),
            "metadata.mesh.diameter disagrees with the IB radius",
        )
        dx_fine = _json_number(mesh["dx_fine"], "metadata.mesh.dx_fine", positive=True)
        _require(
            dx_fine == diameter / diameter_cells,
            "metadata.mesh.dx_fine disagrees with diameter/D_over_dx",
        )
        try:
            layer_bounds = np.asarray(mesh["layer_bounds_D"], dtype=float)
            layer_spacing = np.asarray(mesh["layer_spacing_D"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValidationError("sphere_wake layer geometry must be numeric") from exc
        layer_count = len(layer_cells)
        _require(
            layer_bounds.shape == (layer_count, 6)
            and np.all(np.isfinite(layer_bounds)),
            "metadata.mesh.layer_bounds_D schema mismatch",
        )
        _require(
            layer_spacing.shape == (layer_count,)
            and np.all(np.isfinite(layer_spacing))
            and np.all(layer_spacing > 0.0),
            "metadata.mesh.layer_spacing_D schema mismatch",
        )
        expected_spacing = np.asarray(
            [(2**level) / diameter_cells for level in range(layer_count)], dtype=float
        )
        _require(
            np.array_equal(layer_spacing, expected_spacing),
            "metadata.mesh.layer_spacing_D disagrees with refinement levels",
        )
        _require(
            np.all(layer_bounds[:, [1, 3, 5]] > layer_bounds[:, [0, 2, 4]]),
            "metadata.mesh.layer_bounds_D has non-positive extents",
        )
        lower = layer_bounds[:, [0, 2, 4]]
        upper = layer_bounds[:, [1, 3, 5]]
        if layer_count > 1:
            _require(
                np.all(lower[1:] <= lower[:-1])
                and np.all(upper[1:] >= upper[:-1]),
                "metadata.mesh.layer_bounds_D is not nested inner-to-outer",
            )
        analytic_layer_cells: list[int] = []
        for level in range(layer_count):
            whole_ratio = (upper[level] - lower[level]) / layer_spacing[level]
            _require(
                np.allclose(whole_ratio, np.rint(whole_ratio), rtol=0.0, atol=1.0e-10),
                f"metadata.mesh layer {level} extents are not grid-aligned",
            )
            whole = int(np.prod(np.rint(whole_ratio).astype(np.int64), dtype=np.int64))
            hole = 0
            if level:
                hole_ratio = (upper[level - 1] - lower[level - 1]) / layer_spacing[level]
                offset_ratio = (lower[level - 1] - lower[level]) / layer_spacing[level]
                _require(
                    np.allclose(hole_ratio, np.rint(hole_ratio), rtol=0.0, atol=1.0e-10)
                    and np.allclose(
                        offset_ratio,
                        np.rint(offset_ratio),
                        rtol=0.0,
                        atol=1.0e-10,
                    ),
                    f"metadata.mesh layer {level} interface is not grid-aligned",
                )
                hole = int(
                    np.prod(np.rint(hole_ratio).astype(np.int64), dtype=np.int64)
                )
            analytic_layer_cells.append(whole - hole)
        _require(
            layer_cells == analytic_layer_cells,
            "metadata.mesh.layer_cells disagrees with layer geometry",
        )
        outer = _as_mapping(mesh["outer_domain_D"], "metadata.mesh.outer_domain_D")
        _exact_keys(outer, {"x", "y", "z"}, "metadata.mesh.outer_domain_D")
        try:
            outer_bounds = np.asarray([outer["x"], outer["y"], outer["z"]], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValidationError("metadata.mesh.outer_domain_D must be numeric") from exc
        _require(
            outer_bounds.shape == (3, 2)
            and np.all(np.isfinite(outer_bounds))
            and np.array_equal(
                outer_bounds,
                layer_bounds[-1].reshape(3, 2),
            ),
            "metadata.mesh.outer_domain_D disagrees with the outer layer",
        )
        _json_int(mesh["nodes"], "metadata.mesh.nodes", minimum=1)

    _require(mesh["kind"] == kind, "metadata.mesh.kind disagrees with config")
    faces = _json_int(mesh["faces"], "metadata.mesh.faces", minimum=1)
    edges = _json_int(mesh["edges"], "metadata.mesh.edges", minimum=1)
    markers = _json_int(mesh["markers"], "metadata.mesh.markers", minimum=1)
    radius = float(config["ib"]["radius"])
    marker_spacing = float(config["ib"]["marker_spacing"])
    expected_markers = max(12, int(round(4.0 * math.pi * radius**2 / marker_spacing**2)))
    _require(markers == expected_markers, "metadata.mesh.markers disagrees with IB geometry")
    return cells, edges, markers


def _load_npz(path: Path, expected_fields: set[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            actual = set(archive.files)
            _require(
                len(actual) == len(archive.files),
                f"{path.name} contains duplicate array names",
            )
            missing = sorted(expected_fields.difference(actual))
            extra = sorted(actual.difference(expected_fields))
            _require(
                not missing and not extra,
                f"{path.name} schema mismatch: missing={missing}, extra={extra}",
            )
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except ValidationError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise ValidationError(f"cannot read {path.name} safely: {exc}") from exc
    for name, array in arrays.items():
        _require(array.dtype.kind != "O", f"{path.name}:{name} has forbidden object dtype")
        if array.dtype.kind in {"f", "c"}:
            _require(np.all(np.isfinite(array)), f"{path.name}:{name} contains non-finite values")
    return arrays


def _scalar(array: np.ndarray, label: str) -> Any:
    _require(array.shape == (), f"{label} must be a scalar array")
    return array.item()


def _integer_scalar(array: np.ndarray, label: str, *, minimum: int | None = None) -> int:
    _require(array.dtype.kind in {"i", "u"}, f"{label} must have integer dtype")
    value = int(_scalar(array, label))
    if minimum is not None:
        _require(value >= minimum, f"{label} must be >= {minimum}")
    return value


def _float_scalar(array: np.ndarray, label: str) -> float:
    _require(array.dtype.kind == "f", f"{label} must have floating dtype")
    value = float(_scalar(array, label))
    _require(math.isfinite(value), f"{label} must be finite")
    return value


def _string_scalar(array: np.ndarray, label: str) -> str:
    _require(array.dtype.kind in {"U", "S"}, f"{label} must have string dtype")
    value = _scalar(array, label)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(f"{label} is not UTF-8") from exc
    return _json_string(value, label)


def _validate_npz_source_hashes(
    array: np.ndarray,
    label: str,
    metadata_hashes: Mapping[str, Any],
) -> dict[str, str]:
    text = _string_scalar(array, label)
    parsed = _as_mapping(_strict_json_loads(text, label), label)
    _exact_keys(parsed, NUMERICAL_SOURCE_KEYS, label)
    validated: dict[str, str] = {}
    for key in sorted(NUMERICAL_SOURCE_KEYS):
        digest = _validate_sha256(parsed[key], f"{label}.{key}")
        _require(
            digest == metadata_hashes[key],
            f"{label} disagrees with metadata for {SOURCE_FILES[key]}",
        )
        validated[key] = digest
    return validated


def _float_array(
    array: np.ndarray,
    label: str,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> None:
    _require(array.dtype == dtype, f"{label} dtype {array.dtype} != expected {dtype}")
    _require(array.shape == shape, f"{label} shape {array.shape} != expected {shape}")
    _require(np.all(np.isfinite(array)), f"{label} contains non-finite values")


def _arrays_identical(left: np.ndarray, right: np.ndarray, label: str) -> None:
    _require(left.dtype == right.dtype, f"{label} dtype differs between checkpoint and final field")
    _require(left.shape == right.shape, f"{label} shape differs between checkpoint and final field")
    _require(np.array_equal(left, right), f"{label} differs between checkpoint and final field")


def _time_tolerance(step: int, dt: float) -> float:
    expected_time = step * dt
    return max(
        1.0e-12,
        16.0
        * np.finfo(np.float64).eps
        * max(abs(expected_time), dt)
        * max(step, 1),
    )


def _expected_marker_state(
    config: Mapping[str, Any], marker_count: int, physical_time: float
) -> tuple[np.ndarray, np.ndarray]:
    radius = float(config["ib"]["radius"])
    centre = np.asarray(config["ib"]["center"], dtype=float)
    index = np.arange(marker_count, dtype=float)
    phi = np.arccos(1.0 - 2.0 * (index + 0.5) / marker_count)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * index
    unit = np.stack(
        [
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ],
        axis=1,
    )
    reference = centre + radius * unit
    motion = config["ib"]["motion"]
    if motion["kind"] == "none":
        return reference, np.zeros_like(reference)
    axis = np.asarray(motion["axis"], dtype=float)
    if motion["kind"] == "rotation":
        reference_speed = abs(float(config["physics"]["inflow_speed"]))
        omega = axis * float(motion["spin_ratio"]) * reference_speed / radius
        return reference, np.cross(omega, reference - centre)
    amplitude = float(motion["amplitude"])
    angular_frequency = 2.0 * np.pi * float(motion["frequency"])
    angle = angular_frequency * physical_time + float(motion["phase"])
    displacement = amplitude * np.sin(angle) * axis
    velocity = amplitude * angular_frequency * np.cos(angle) * axis
    return reference + displacement, np.tile(velocity, (marker_count, 1))


def _cell_volumes(
    mesh: Mapping[str, Any],
    config: Mapping[str, Any],
    cpos: np.ndarray,
    cell_count: int,
    dtype: np.dtype[Any],
) -> tuple[np.ndarray, float]:
    if config["mesh"]["kind"] == "uniform_box":
        finest_dx = float(mesh["dx"])
        widths = np.full(cell_count, finest_dx, dtype=dtype)
        bounds = np.asarray(mesh["bounds"], dtype=float)
        tolerance = max(1.0e-12, finest_dx * 1.0e-9)
        _require(
            np.all(cpos >= bounds[:, 0] - tolerance)
            and np.all(cpos <= bounds[:, 1] + tolerance),
            "final_field.cpos leaves the uniform mesh bounds",
        )
    else:
        diameter = float(mesh["diameter"])
        spacing = np.asarray(mesh["layer_spacing_D"], dtype=float)
        counts = [int(value) for value in mesh["layer_cells"]]
        finest_dx = float(mesh["dx_fine"])
        widths = np.concatenate(
            [
                np.full(count, diameter * spacing_D, dtype=dtype)
                for count, spacing_D in zip(counts, spacing)
            ]
        )
        _require(
            widths.shape == (cell_count,),
            "sphere_wake layer geometry does not cover every final cell",
        )
        layer_bounds = np.asarray(mesh["layer_bounds_D"], dtype=float)
        position_D = np.asarray(cpos, dtype=float) / diameter
        assigned = np.full(cell_count, -1, dtype=np.int64)
        tolerance_D = max(1.0e-12, float(np.min(spacing)) * 1.0e-9)
        for level, bounds_D in enumerate(layer_bounds):
            lower = bounds_D[[0, 2, 4]]
            upper = bounds_D[[1, 3, 5]]
            inside = np.all(position_D >= lower - tolerance_D, axis=1) & np.all(
                position_D <= upper + tolerance_D, axis=1
            )
            assigned[(assigned < 0) & inside] = level
        _require(
            np.all(assigned >= 0),
            "final_field.cpos contains cells outside sphere_wake layer bounds",
        )
        expected_assignment = np.repeat(
            np.arange(len(counts), dtype=np.int64), np.asarray(counts, dtype=np.int64)
        )
        _require(
            np.array_equal(assigned, expected_assignment),
            "final_field.cpos ordering/counts disagree with sphere_wake layers",
        )
    volumes = widths * widths * widths
    _require(
        volumes.shape == (cell_count,)
        and np.all(np.isfinite(volumes))
        and np.all(volumes > 0.0),
        "derived cell volumes are invalid",
    )
    return volumes, finest_dx


def _recompute_force_evidence(
    force_e: np.ndarray,
    force_l: np.ndarray,
    cell_volume: np.ndarray,
    marker_volume_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    fluid_force = np.sum(force_e * cell_volume[:, None], axis=0)
    lagrangian_force = np.sum(force_l * marker_volume_weight[:, None], axis=0)
    denominator = max(
        float(np.linalg.norm(lagrangian_force)),
        float(np.linalg.norm(fluid_force)),
        1.0e-300,
    )
    balance = float(np.linalg.norm(fluid_force - lagrangian_force)) / denominator
    _require(
        np.all(np.isfinite(fluid_force))
        and np.all(np.isfinite(lagrangian_force))
        and math.isfinite(balance),
        "recomputed IB force evidence is non-finite",
    )
    return fluid_force, lagrangian_force, balance


def _expected_sphere_center(config: Mapping[str, Any], physical_time: float) -> np.ndarray:
    center = np.asarray(config["ib"]["center"], dtype=float)
    motion = config["ib"]["motion"]
    if motion["kind"] != "oscillation":
        return center
    angle = (
        2.0 * np.pi * float(motion["frequency"]) * physical_time
        + float(motion["phase"])
    )
    return (
        center
        + float(motion["amplitude"])
        * np.sin(angle)
        * np.asarray(motion["axis"], dtype=float)
    )


def _recompute_interior_momenta(
    config: Mapping[str, Any],
    physical_time: float,
    cell_positions: np.ndarray,
    cell_velocity: np.ndarray,
    cell_volume: np.ndarray,
    finest_dx: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Independent NumPy reconstruction of the canonical sphere indicator."""

    center = _expected_sphere_center(config, physical_time)
    radius = float(config["ib"]["radius"])
    density = float(config["physics"]["density"])
    distance = np.linalg.norm(cell_positions - center, axis=1)
    indicator = np.clip((radius + 0.5 * finest_dx - distance) / finest_dx, 0.0, 1.0)
    weighted_volume = indicator * cell_volume
    linear = density * np.sum(cell_velocity * weighted_volume[:, None], axis=0)
    angular = density * np.sum(
        np.cross(cell_positions, cell_velocity) * weighted_volume[:, None], axis=0
    )
    volume = float(np.sum(weighted_volume))
    _require(
        np.all(np.isfinite(linear))
        and np.all(np.isfinite(angular))
        and math.isfinite(volume)
        and volume > 0.0,
        "recomputed interior-fluid evidence is invalid",
    )
    return linear, angular, volume


def _history_vector(row: Mapping[str, float | int], stem: str) -> np.ndarray:
    return np.asarray([row[f"{stem}_{axis}"] for axis in "xyz"], dtype=float)


def _validate_load_algebra(
    *,
    force_fluid: np.ndarray,
    moment_fluid: np.ndarray,
    body_force: np.ndarray,
    body_moment: np.ndarray,
    momentum_rate: np.ndarray,
    angular_momentum_rate: np.ndarray,
    rtol: float,
    atol: float,
    label: str,
) -> None:
    _require(
        np.allclose(
            body_force,
            -force_fluid + momentum_rate,
            rtol=rtol,
            atol=atol,
        ),
        f"{label} body force violates the interior-momentum correction",
    )
    _require(
        np.allclose(
            body_moment,
            -moment_fluid + angular_momentum_rate,
            rtol=rtol,
            atol=atol,
        ),
        f"{label} body moment violates the interior-angular-momentum correction",
    )


def _validate_history_loads(
    row: Mapping[str, float | int],
    evidence: Mapping[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> None:
    pairs = {
        "body_force": "body_force",
        "force_fluid": "force_fluid",
        "body_moment": "body_moment",
        "moment_fluid": "moment_fluid",
        "interior_momentum": "interior_momentum",
        "interior_angular_momentum": "interior_angular_momentum",
        "interior_momentum_rate": "interior_momentum_rate",
        "interior_angular_momentum_rate": "interior_angular_momentum_rate",
    }
    for history_stem, evidence_name in pairs.items():
        _require(
            np.allclose(
                _history_vector(row, history_stem),
                evidence[evidence_name],
                rtol=rtol,
                atol=atol,
            ),
            f"history final {history_stem} disagrees with persisted load evidence",
        )
    _require(
        math.isclose(
            float(row["interior_indicator_volume"]),
            float(evidence["interior_indicator_volume"]),
            rel_tol=rtol,
            abs_tol=atol,
        ),
        "history final interior indicator volume disagrees with persisted evidence",
    )


def _validate_cg_evidence(history, config, final=None):
    if config["solver"]["fail_on_cg_nonconvergence"]:
        _require(all(row["helmholtz_tolerance_ratio"] <= 1.0 for row in history),
                 "strict run contains unconverged CG history")
    if final is not None:
        for name in ("helmholtz_relative_residual", "helmholtz_tolerance_ratio"):
            value = _float_scalar(final[name], f"final_field.{name}")
            _require(value == history[-1][name], f"final_field.{name} disagrees with history")


def _read_history(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValidationError("history.csv is empty") from exc
            _require(tuple(header) == HISTORY_COLUMNS, "history.csv header/schema mismatch")
            _require(len(header) == len(set(header)), "history.csv contains duplicate columns")
            for line_number, values in enumerate(reader, start=2):
                _require(bool(values), f"history.csv contains blank row at line {line_number}")
                _require(
                    len(values) == len(HISTORY_COLUMNS),
                    f"history.csv line {line_number} has {len(values)} fields; expected {len(HISTORY_COLUMNS)}",
                )
                row: dict[str, float | int] = {}
                for name, raw in zip(HISTORY_COLUMNS, values):
                    _require(bool(raw.strip()), f"history.csv line {line_number} has empty {name}")
                    if name in INTEGER_HISTORY_COLUMNS:
                        _require(
                            bool(_INTEGER_RE.fullmatch(raw.strip())),
                            f"history.csv line {line_number} {name} is not an integer",
                        )
                        row[name] = int(raw)
                    else:
                        try:
                            number = float(raw)
                        except ValueError as exc:
                            raise ValidationError(
                                f"history.csv line {line_number} {name} is not numeric"
                            ) from exc
                        _require(
                            math.isfinite(number),
                            f"history.csv line {line_number} {name} is non-finite",
                        )
                        row[name] = number
                rows.append(row)
    except ValidationError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValidationError(f"cannot read history.csv: {exc}") from exc
    _require(bool(rows), "history.csv contains no data rows")
    return rows


def _validate_history(
    rows: Sequence[Mapping[str, float | int]],
    *,
    completed_step: int,
    dt: float,
    expected_force_updates: int,
    sample_every: int,
    checkpoint_every: int,
    force_balance_atol: float,
    load_rtol: float,
    load_atol: float,
) -> None:
    _require(int(rows[0]["step"]) == 1, "history.csv must begin at step 1")
    previous_step = 0
    previous_time = -math.inf
    for index, row in enumerate(rows, start=2):
        step = int(row["step"])
        physical_time = float(row["time"])
        _require(step > previous_step, f"history.csv step is not strictly increasing at line {index}")
        _require(
            physical_time > previous_time,
            f"history.csv time is not strictly increasing at line {index}",
        )
        _require(step <= completed_step, f"history.csv line {index} exceeds completed_step")
        _require(
            math.isclose(
                physical_time,
                step * dt,
                rel_tol=0.0,
                abs_tol=_time_tolerance(step, dt),
            ),
            f"history.csv line {index} time disagrees with step*dt",
        )
        for name in NONNEGATIVE_HISTORY_COLUMNS:
            _require(float(row[name]) >= 0.0, f"history.csv line {index} {name} is negative")
        _require(
            float(row["interior_indicator_volume"]) > 0.0,
            f"history.csv line {index} interior_indicator_volume is not positive",
        )
        _validate_load_algebra(
            force_fluid=_history_vector(row, "force_fluid"),
            moment_fluid=_history_vector(row, "moment_fluid"),
            body_force=_history_vector(row, "body_force"),
            body_moment=_history_vector(row, "body_moment"),
            momentum_rate=_history_vector(row, "interior_momentum_rate"),
            angular_momentum_rate=_history_vector(
                row, "interior_angular_momentum_rate"
            ),
            rtol=load_rtol,
            atol=load_atol,
            label=f"history.csv line {index}",
        )
        _require(
            int(row["ib_force_updates"]) == expected_force_updates,
            f"history.csv line {index} ib_force_updates != {expected_force_updates}",
        )
        balance = float(row["force_balance_relerr"])
        _require(
            balance <= force_balance_atol,
            f"history.csv line {index} force balance {balance:.6e} exceeds {force_balance_atol:.6e}",
        )
        previous_step = step
        previous_time = physical_time
    _require(int(rows[-1]["step"]) == completed_step, "history.csv final step disagrees with metadata")
    recorded_steps = {int(row["step"]) for row in rows}
    required_steps = set(range(1, min(completed_step, 3) + 1))
    required_steps.update(range(sample_every, completed_step + 1, sample_every))
    if checkpoint_every:
        required_steps.update(range(checkpoint_every, completed_step + 1, checkpoint_every))
    required_steps.add(completed_step)
    missing_steps = sorted(required_steps.difference(recorded_steps))
    _require(
        not missing_steps,
        f"history.csv omits required sampled steps: {missing_steps}",
    )


def _positive_atol(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return number


def _force_balance_limit(precision: str, requested: float | None) -> float:
    if requested is None:
        return DEFAULT_FORCE_BALANCE_ATOL[precision]
    try:
        value = float(requested)
    except (TypeError, ValueError) as exc:
        raise ValidationError("force_balance_atol must be numeric") from exc
    _require(
        math.isfinite(value) and value >= 0.0,
        "force_balance_atol must be finite and non-negative",
    )
    return value


def validate_result(
    result_directory: str | Path,
    *,
    force_balance_atol: float | None = None,
) -> dict[str, Any]:
    """Validate a complete result directory and return a compact summary."""
    root = Path(result_directory).expanduser().resolve()
    _require(root.is_dir(), f"result directory does not exist: {root}")
    paths = {
        "metadata": root / "metadata.json",
        "history": root / "history.csv",
        "checkpoint": root / "checkpoint.npz",
        "final": root / "final_field.npz",
    }
    missing = [
        paths[name].name
        for name in ("metadata", "history", "checkpoint")
        if not paths[name].is_file()
    ]
    _require(not missing, f"missing required result artifacts: {missing}")

    metadata = _read_metadata(paths["metadata"])
    _require(
        _json_int(metadata["result_schema"], "metadata.result_schema") == RESULT_SCHEMA_VERSION,
        f"unsupported result_schema; expected {RESULT_SCHEMA_VERSION}",
    )
    created = _parse_utc(metadata["created_utc"], "metadata.created_utc")
    completed = _parse_utc(metadata["completed_utc"], "metadata.completed_utc")
    _require(completed >= created, "metadata.completed_utc predates created_utc")
    run_id = _validate_uuid(metadata["run_id"], "metadata.run_id")
    layout_fingerprint = _validate_sha256(
        metadata["layout_fingerprint"], "metadata.layout_fingerprint"
    )
    config = _validate_config(metadata["config"])
    load_metadata = _as_mapping(metadata["closed_body_load"], "metadata.closed_body_load")
    _exact_keys(
        load_metadata,
        {"method", "force", "moment", "indicator", "moment_origin"},
        "metadata.closed_body_load",
    )
    _require(
        load_metadata["method"] == "Uhlmann interior-fluid momentum correction",
        "metadata closed-body load method is unsupported",
    )
    _require(
        load_metadata["force"]
        == "-force_fluid + backward_difference(interior_momentum)",
        "metadata closed-body force definition is unsupported",
    )
    _require(
        load_metadata["moment"]
        == "-moment_fluid + backward_difference(interior_angular_momentum)",
        "metadata closed-body moment definition is unsupported",
    )
    _require(
        load_metadata["indicator"] == "analytic sphere with one-cell linear smoothing",
        "metadata closed-body indicator definition is unsupported",
    )
    try:
        moment_origin = np.asarray(load_metadata["moment_origin"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError("metadata.closed_body_load.moment_origin must be numeric") from exc
    _require(
        moment_origin.shape == (3,)
        and np.all(np.isfinite(moment_origin))
        and np.array_equal(moment_origin, np.zeros(3)),
        "metadata closed-body moment origin must be the global origin",
    )
    save_field = bool(config["output"]["save_field"])
    if save_field:
        _require(paths["final"].is_file(), "missing required result artifact: final_field.npz")
    else:
        _require(
            not paths["final"].exists(),
            "final_field.npz must be absent when config.output.save_field=false",
        )
    _require(metadata["case_name"] == config["name"], "metadata.case_name disagrees with config")
    _require(metadata["case_kind"] == config["case"]["kind"], "metadata.case_kind disagrees with config")

    precision = _json_string(metadata["precision"], "metadata.precision")
    _require(precision == config["runtime"]["precision"], "metadata precision disagrees with config")
    expected_backend = "cuda/cupy" if config["runtime"]["device"] == "cuda" else "cpu/numpy"
    _require(metadata["backend"] == expected_backend, "metadata backend disagrees with config")
    _json_string(metadata["python"], "metadata.python")
    _json_string(metadata["array_backend_version"], "metadata.array_backend_version")
    _json_string(metadata["output_directory"], "metadata.output_directory")
    _require(
        metadata["resumed_from"] is None or isinstance(metadata["resumed_from"], str),
        "metadata.resumed_from must be null or a string",
    )
    _require(isinstance(metadata["complete"], bool), "metadata.complete must be boolean")
    _require(metadata["complete"] is True, "strict acceptance requires a complete run")
    dt = _json_number(metadata["dt"], "metadata.dt", positive=True)
    _require(dt == float(config["time"]["dt"]), "metadata.dt disagrees with config.time.dt")
    target_steps = _json_int(metadata["target_steps"], "metadata.target_steps", minimum=1)
    start_step = _json_int(metadata["start_step"], "metadata.start_step", minimum=0)
    completed_step = _json_int(metadata["completed_step"], "metadata.completed_step", minimum=1)
    resume_count = _json_int(metadata["resume_count"], "metadata.resume_count", minimum=0)
    _require(target_steps == config["time"]["steps"], "metadata.target_steps disagrees with config")
    _require(completed_step == target_steps, "complete metadata must have completed_step == target_steps")
    _require(start_step < completed_step, "metadata.start_step must be before completed_step")
    resumed_from = metadata["resumed_from"]
    if start_step == 0:
        _require(resumed_from is None, "fresh-run metadata.resumed_from must be null")
        _require(resume_count == 0, "fresh-run metadata.resume_count must be zero")
        _require(
            "last_resume_utc" not in metadata,
            "fresh-run metadata must not contain last_resume_utc",
        )
    else:
        resumed_path = Path(_json_string(resumed_from, "metadata.resumed_from"))
        _require(
            resumed_path.name == "checkpoint.npz",
            "metadata.resumed_from must identify the canonical checkpoint.npz",
        )
        _require(resume_count >= 1, "resumed metadata.resume_count must be positive")
        _require("last_resume_utc" in metadata, "resumed metadata lacks last_resume_utc")
        last_resume = _parse_utc(metadata["last_resume_utc"], "metadata.last_resume_utc")
        _require(
            created <= last_resume <= completed,
            "metadata.last_resume_utc lies outside the run lifetime",
        )
    _json_number(metadata["setup_seconds"], "metadata.setup_seconds")
    _require(float(metadata["setup_seconds"]) >= 0.0, "metadata.setup_seconds is negative")

    fingerprint = _validate_sha256(metadata["config_fingerprint"], "metadata.config_fingerprint")
    _require(fingerprint == _config_fingerprint(config), "metadata config fingerprint is incorrect")
    cells, edges, markers = _validate_mesh_metadata(metadata["mesh"], config)

    source_hashes = _as_mapping(metadata["source_sha256"], "metadata.source_sha256")
    _exact_keys(source_hashes, set(SOURCE_FILES), "metadata.source_sha256")
    solver_directory = Path(__file__).resolve().parent
    for key, filename in SOURCE_FILES.items():
        recorded = _validate_sha256(source_hashes[key], f"metadata.source_sha256.{key}")
        current = _sha256(solver_directory / filename)
        _require(recorded == current, f"numerical source hash mismatch for {filename}")

    checkpoint = _load_npz(paths["checkpoint"], CHECKPOINT_FIELDS)
    final = _load_npz(paths["final"], FINAL_FIELD_FIELDS) if save_field else None
    _require(
        _integer_scalar(checkpoint["checkpoint_schema"], "checkpoint.checkpoint_schema")
        == CHECKPOINT_SCHEMA_VERSION,
        f"unsupported checkpoint schema; expected {CHECKPOINT_SCHEMA_VERSION}",
    )
    _require(
        _string_scalar(checkpoint["update_phase"], "checkpoint.update_phase") == "post_step",
        "checkpoint update_phase must be post_step",
    )
    checkpoint_run_id = _validate_uuid(
        _string_scalar(checkpoint["run_id"], "checkpoint.run_id"), "checkpoint.run_id"
    )
    _require(checkpoint_run_id == run_id, "checkpoint run_id disagrees with metadata")
    checkpoint_layout = _validate_sha256(
        _string_scalar(checkpoint["layout_fingerprint"], "checkpoint.layout_fingerprint"),
        "checkpoint.layout_fingerprint",
    )
    _require(
        checkpoint_layout == layout_fingerprint,
        "checkpoint layout fingerprint disagrees with metadata",
    )
    checkpoint_step = _integer_scalar(checkpoint["step"], "checkpoint.step", minimum=1)
    _require(checkpoint_step == completed_step, "checkpoint step disagrees with metadata")
    checkpoint_time = _float_scalar(checkpoint["physical_time"], "checkpoint.physical_time")
    _require(
        math.isclose(
            checkpoint_time,
            checkpoint_step * dt,
            rel_tol=0.0,
            abs_tol=_time_tolerance(checkpoint_step, dt),
        ),
        "checkpoint physical_time disagrees with step*dt",
    )
    _require(_float_scalar(checkpoint["dt"], "checkpoint.dt") == dt, "checkpoint dt mismatch")
    _require(
        _float_scalar(checkpoint["previous_dt"], "checkpoint.previous_dt") == dt,
        "checkpoint previous_dt mismatch",
    )
    _require(
        _string_scalar(checkpoint["precision"], "checkpoint.precision") == precision,
        "checkpoint precision mismatch",
    )
    device = str(config["runtime"]["device"])
    _require(
        _string_scalar(checkpoint["device"], "checkpoint.device") == device,
        "checkpoint device mismatch",
    )
    _require(
        _string_scalar(checkpoint["config_fingerprint"], "checkpoint.config_fingerprint")
        == fingerprint,
        "checkpoint config fingerprint mismatch",
    )
    public_force_mode = str(config["ib"]["force_mode"])
    checkpoint_force_mode = _string_scalar(
        checkpoint["ib_force_mode"], "checkpoint.ib_force_mode"
    )
    expected_checkpoint_mode = public_force_mode
    _require(
        checkpoint_force_mode == expected_checkpoint_mode,
        "checkpoint IB force mode disagrees with config",
    )

    checkpoint_sources = _validate_npz_source_hashes(
        checkpoint["source_hashes_json"],
        "checkpoint.source_hashes_json",
        source_hashes,
    )

    inner_iterations = int(config["time"]["inner_iterations"])
    expected_force_updates = 1 if public_force_mode == "accumulated_explicit" else inner_iterations
    update_count = _integer_scalar(
        checkpoint["ib_force_update_count"], "checkpoint.ib_force_update_count", minimum=1
    )
    _require(
        update_count == completed_step * expected_force_updates,
        "checkpoint IB force update count disagrees with completed physical steps",
    )

    dtype = np.dtype(precision)
    _float_array(checkpoint["s"], "checkpoint.s", dtype=dtype, shape=(edges,))
    _float_array(checkpoint["s_old"], "checkpoint.s_old", dtype=dtype, shape=(edges,))
    _float_array(checkpoint["markers"], "checkpoint.markers", dtype=dtype, shape=(markers, 3))
    _float_array(
        checkpoint["marker_velocity"],
        "checkpoint.marker_velocity",
        dtype=dtype,
        shape=(markers, 3),
    )
    _float_array(checkpoint["ib_force_E"], "checkpoint.ib_force_E", dtype=dtype, shape=(cells, 3))
    _float_array(checkpoint["ib_force_L"], "checkpoint.ib_force_L", dtype=dtype, shape=(markers, 3))
    load_vector_names = (
        "force_fluid",
        "moment_fluid",
        "body_force",
        "body_moment",
        "interior_momentum",
        "interior_angular_momentum",
        "interior_momentum_rate",
        "interior_angular_momentum_rate",
    )
    checkpoint_load: dict[str, np.ndarray] = {}
    for name in load_vector_names:
        _float_array(checkpoint[name], f"checkpoint.{name}", dtype=dtype, shape=(3,))
        checkpoint_load[name] = checkpoint[name]
    checkpoint_load["interior_indicator_volume"] = np.asarray(
        _float_scalar(
            checkpoint["interior_indicator_volume"],
            "checkpoint.interior_indicator_volume",
        )
    )
    _require(
        float(checkpoint_load["interior_indicator_volume"]) > 0.0,
        "checkpoint interior indicator volume must be positive",
    )
    load_rtol = 5.0e-6 if precision == "float32" else 1.0e-12
    load_atol = 5.0e-6 if precision == "float32" else 1.0e-12
    _validate_load_algebra(
        force_fluid=checkpoint_load["force_fluid"],
        moment_fluid=checkpoint_load["moment_fluid"],
        body_force=checkpoint_load["body_force"],
        body_moment=checkpoint_load["body_moment"],
        momentum_rate=checkpoint_load["interior_momentum_rate"],
        angular_momentum_rate=checkpoint_load["interior_angular_momentum_rate"],
        rtol=load_rtol,
        atol=load_atol,
        label="checkpoint",
    )

    expected_checkpoint_markers, expected_checkpoint_velocity = _expected_marker_state(
        config, markers, checkpoint_time
    )
    marker_atol = 2.0e-6 if precision == "float32" else 1.0e-12
    _require(
        np.allclose(
            checkpoint["markers"],
            expected_checkpoint_markers,
            rtol=0.0,
            atol=marker_atol,
        ),
        "checkpoint markers disagree with the prescribed body state",
    )
    _require(
        np.allclose(
            checkpoint["marker_velocity"],
            expected_checkpoint_velocity,
            rtol=0.0,
            atol=marker_atol,
        ),
        "checkpoint marker velocity disagrees with prescribed motion",
    )

    if final is None:
        balance_atol = _force_balance_limit(precision, force_balance_atol)
        history = _read_history(paths["history"])
        _validate_cg_evidence(history, config)
        _validate_history(
            history,
            completed_step=completed_step,
            dt=dt,
            expected_force_updates=expected_force_updates,
            sample_every=int(config["output"]["sample_every"]),
            checkpoint_every=int(config["output"]["checkpoint_every"]),
            force_balance_atol=balance_atol,
            load_rtol=load_rtol,
            load_atol=load_atol,
        )
        _validate_history_loads(
            history[-1], checkpoint_load, rtol=load_rtol, atol=load_atol
        )
        _require(
            math.isclose(
                float(history[-1]["time"]),
                checkpoint_time,
                rel_tol=0.0,
                abs_tol=_time_tolerance(completed_step, dt),
            ),
            "history final time disagrees with checkpoint",
        )
        return {
            "result_directory": str(root),
            "case_name": metadata["case_name"],
            "case_kind": metadata["case_kind"],
            "backend": metadata["backend"],
            "precision": precision,
            "completed_step": completed_step,
            "history_rows": len(history),
            "force_balance_max": max(
                float(row["force_balance_relerr"]) for row in history
            ),
            "force_balance_recomputed": None,
            "force_balance_atol": balance_atol,
            "config_fingerprint": fingerprint,
            "coverage": "checkpoint_only",
        }

    _require(
        _integer_scalar(final["result_schema"], "final_field.result_schema")
        == RESULT_SCHEMA_VERSION,
        f"unsupported final_field result_schema; expected {RESULT_SCHEMA_VERSION}",
    )
    final_run_id = _validate_uuid(
        _string_scalar(final["run_id"], "final_field.run_id"), "final_field.run_id"
    )
    _require(final_run_id == run_id, "final_field run_id disagrees with metadata")
    final_step = _integer_scalar(final["step"], "final_field.step", minimum=1)
    _require(final_step == completed_step, "final_field step disagrees with metadata")
    final_time = _float_scalar(final["time"], "final_field.time")
    _require(final_time == checkpoint_time, "final_field time disagrees with checkpoint")
    _require(_float_scalar(final["dt"], "final_field.dt") == dt, "final_field dt mismatch")
    _require(
        _string_scalar(final["device"], "final_field.device") == device,
        "final_field device mismatch",
    )
    _require(
        _string_scalar(final["precision"], "final_field.precision") == precision,
        "final_field precision mismatch",
    )
    _require(
        _string_scalar(final["config_fingerprint"], "final_field.config_fingerprint")
        == fingerprint,
        "final_field config fingerprint mismatch",
    )
    final_layout = _validate_sha256(
        _string_scalar(final["layout_fingerprint"], "final_field.layout_fingerprint"),
        "final_field.layout_fingerprint",
    )
    _require(final_layout == layout_fingerprint, "final_field layout fingerprint mismatch")
    final_sources = _validate_npz_source_hashes(
        final["source_hashes_json"], "final_field.source_hashes_json", source_hashes
    )
    _require(final_sources == checkpoint_sources, "final/checkpoint source identities differ")
    _require(
        _string_scalar(final["ib_force_mode"], "final_field.ib_force_mode")
        == checkpoint_force_mode,
        "final_field IB force mode disagrees with checkpoint",
    )
    _require(
        _integer_scalar(
            final["ib_force_update_count"],
            "final_field.ib_force_update_count",
            minimum=1,
        )
        == update_count,
        "final_field IB force update count disagrees with checkpoint",
    )
    _float_array(final["s"], "final_field.s", dtype=dtype, shape=(edges,))
    _float_array(final["s_old"], "final_field.s_old", dtype=dtype, shape=(edges,))
    _float_array(final["vc"], "final_field.vc", dtype=dtype, shape=(cells, 3))
    _float_array(final["momentum_residual"], "final_field.momentum_residual", dtype=dtype, shape=(cells, 3))
    faces = int(metadata["mesh"]["faces"])
    _float_array(final["face_area"], "final_field.face_area", dtype=dtype, shape=(faces,))
    _require(np.all(final["face_area"] > 0), "final_field.face_area must be positive")
    face_cells = final["face_cells"]
    _require(face_cells.dtype.kind in "iu" and face_cells.shape == (faces, 2),
             "final_field.face_cells must be integer face connectivity")
    _require(np.all((face_cells[:, 0] >= 0) & (face_cells[:, 0] < cells)) and
             np.all((face_cells[:, 1] >= -1) & (face_cells[:, 1] < cells)),
             "final_field.face_cells contains invalid cell ids")
    index_cell = final["index_cell"]
    _require(index_cell.dtype.kind in "iu" and index_cell.ndim == 3 and index_cell.size > 0,
             "final_field.index_cell must be a nonempty integer lattice")
    _require(np.all((index_cell >= -1) & (index_cell < cells)),
             "final_field.index_cell contains invalid cell ids")
    _float_array(final["lattice_origin"], "final_field.lattice_origin", dtype=dtype, shape=(3,))
    _float_array(final["sphere_center"], "final_field.sphere_center", dtype=dtype, shape=(3,))
    _require(np.allclose(final["sphere_center"], _expected_sphere_center(config, final_time),
                        rtol=0, atol=2e-6 if dtype == np.dtype("float32") else 1e-12),
             "final_field.sphere_center disagrees with prescribed motion")
    radius = float(config["ib"]["radius"])
    inflow = abs(float(config["physics"]["inflow_speed"]))
    if inflow == 0:
        motion = config["ib"]["motion"]
        inflow = 2 * np.pi * float(motion["frequency"]) * float(motion["amplitude"])
    expected_scalars = {
        "lattice_spacing": float(metadata["mesh"].get("dx", metadata["mesh"].get("dx_fine"))),
        "sphere_radius": radius,
        "density": float(config["physics"]["density"]),
        "kinematic_viscosity": inflow * 2 * radius / float(config["physics"]["reynolds"]),
    }
    for name, expected in expected_scalars.items():
        value = _float_scalar(final[name], f"final_field.{name}")
        _require(value > 0 and math.isclose(value, expected, rel_tol=1e-12, abs_tol=0),
                 f"final_field.{name} disagrees with configuration")
    for name in ("helmholtz_relative_residual", "helmholtz_tolerance_ratio"):
        value = _float_scalar(final[name], f"final_field.{name}")
        _require(value >= 0, f"final_field.{name} must be nonnegative")
        if name == "helmholtz_tolerance_ratio" and config["solver"]["fail_on_cg_nonconvergence"]:
            _require(value <= 1.0, "strict run contains an unconverged final CG solve")
    _float_array(final["cpos"], "final_field.cpos", dtype=dtype, shape=(cells, 3))
    _float_array(
        final["cell_volume"], "final_field.cell_volume", dtype=dtype, shape=(cells,)
    )
    _require(
        np.all(final["cell_volume"] > 0.0),
        "final_field.cell_volume must be strictly positive",
    )
    _float_array(final["markers"], "final_field.markers", dtype=dtype, shape=(markers, 3))
    _float_array(
        final["marker_velocity"], "final_field.marker_velocity", dtype=dtype, shape=(markers, 3)
    )
    _float_array(final["ib_force_E"], "final_field.ib_force_E", dtype=dtype, shape=(cells, 3))
    _float_array(final["ib_force_L"], "final_field.ib_force_L", dtype=dtype, shape=(markers, 3))
    final_load: dict[str, np.ndarray] = {}
    for name in load_vector_names:
        _float_array(final[name], f"final_field.{name}", dtype=dtype, shape=(3,))
        final_load[name] = final[name]
    final_load["interior_indicator_volume"] = np.asarray(
        _float_scalar(
            final["interior_indicator_volume"],
            "final_field.interior_indicator_volume",
        )
    )
    _require(
        float(final_load["interior_indicator_volume"]) > 0.0,
        "final_field interior indicator volume must be positive",
    )
    _validate_load_algebra(
        force_fluid=final_load["force_fluid"],
        moment_fluid=final_load["moment_fluid"],
        body_force=final_load["body_force"],
        body_moment=final_load["body_moment"],
        momentum_rate=final_load["interior_momentum_rate"],
        angular_momentum_rate=final_load["interior_angular_momentum_rate"],
        rtol=load_rtol,
        atol=load_atol,
        label="final_field",
    )
    _float_array(
        final["marker_volume_weight"],
        "final_field.marker_volume_weight",
        dtype=dtype,
        shape=(markers,),
    )
    _require(
        np.all(final["marker_volume_weight"] > 0.0),
        "final_field.marker_volume_weight must be strictly positive",
    )
    for name in (
        "s", "s_old", "markers", "marker_velocity", "ib_force_E", "ib_force_L",
        *load_vector_names, "interior_indicator_volume",
    ):
        _arrays_identical(checkpoint[name], final[name], name)

    expected_markers, expected_marker_velocity = _expected_marker_state(
        config, markers, final_time
    )
    marker_atol = 2.0e-6 if precision == "float32" else 1.0e-12
    _require(
        np.allclose(final["markers"], expected_markers, rtol=0.0, atol=marker_atol),
        "final/checkpoint markers disagree with the prescribed body state",
    )
    _require(
        np.allclose(
            final["marker_velocity"],
            expected_marker_velocity,
            rtol=0.0,
            atol=marker_atol,
        ),
        "final/checkpoint marker velocity disagrees with prescribed motion",
    )
    expected_cell_volume, finest_dx = _cell_volumes(
        metadata["mesh"], config, final["cpos"], cells, dtype
    )
    marker_width = (
        4.0
        * np.pi
        * float(config["ib"]["radius"]) ** 2
        * finest_dx
        / markers
    ) ** (1.0 / 3.0)
    expected_marker_width = np.full(markers, marker_width, dtype=dtype)
    expected_marker_volume = (
        expected_marker_width * expected_marker_width * expected_marker_width
    )
    geometry_rtol = 1.0e-5 if precision == "float32" else 1.0e-12
    _require(
        np.allclose(
            final["cell_volume"],
            expected_cell_volume,
            rtol=geometry_rtol,
            atol=0.0,
        ),
        "final_field.cell_volume disagrees with persisted mesh geometry",
    )
    _require(
        np.allclose(
            final["marker_volume_weight"],
            expected_marker_volume,
            rtol=geometry_rtol,
            atol=0.0,
        ),
        "final_field.marker_volume_weight disagrees with IB quadrature",
    )

    balance_atol = _force_balance_limit(precision, force_balance_atol)
    history = _read_history(paths["history"])
    _validate_cg_evidence(history, config, final)
    _validate_history(
        history,
        completed_step=completed_step,
        dt=dt,
        expected_force_updates=expected_force_updates,
        sample_every=int(config["output"]["sample_every"]),
        checkpoint_every=int(config["output"]["checkpoint_every"]),
        force_balance_atol=balance_atol,
        load_rtol=load_rtol,
        load_atol=load_atol,
    )
    fluid_force, _lagrangian_force, recomputed_balance = _recompute_force_evidence(
        final["ib_force_E"],
        final["ib_force_L"],
        final["cell_volume"],
        final["marker_volume_weight"],
    )
    fluid_moment = np.sum(
        np.cross(final["cpos"], final["ib_force_E"])
        * final["cell_volume"][:, None],
        axis=0,
    )
    _require(
        np.allclose(
            final_load["force_fluid"], fluid_force, rtol=load_rtol, atol=load_atol
        ),
        "persisted force_fluid disagrees with the Eulerian IB force field",
    )
    _require(
        np.allclose(
            final_load["moment_fluid"], fluid_moment, rtol=load_rtol, atol=load_atol
        ),
        "persisted moment_fluid disagrees with the Eulerian IB force field",
    )
    recomputed_momentum, recomputed_angular_momentum, recomputed_indicator_volume = (
        _recompute_interior_momenta(
            config,
            final_time,
            final["cpos"],
            final["vc"],
            final["cell_volume"],
            finest_dx,
        )
    )
    _require(
        np.allclose(
            final_load["interior_momentum"],
            recomputed_momentum,
            rtol=load_rtol,
            atol=load_atol,
        ),
        "persisted interior momentum disagrees with the final velocity field",
    )
    _require(
        np.allclose(
            final_load["interior_angular_momentum"],
            recomputed_angular_momentum,
            rtol=load_rtol,
            atol=load_atol,
        ),
        "persisted interior angular momentum disagrees with the final velocity field",
    )
    _require(
        math.isclose(
            float(final_load["interior_indicator_volume"]),
            recomputed_indicator_volume,
            rel_tol=load_rtol,
            abs_tol=load_atol,
        ),
        "persisted interior indicator volume disagrees with the final geometry",
    )
    _require(
        recomputed_balance <= balance_atol,
        "recomputed final IB force balance "
        f"{recomputed_balance:.6e} exceeds {balance_atol:.6e}",
    )
    balance_match_atol = 5.0e-6 if precision == "float32" else 1.0e-12
    _require(
        math.isclose(
            float(history[-1]["force_balance_relerr"]),
            recomputed_balance,
            rel_tol=5.0e-3,
            abs_tol=balance_match_atol,
        ),
        "history final force_balance_relerr disagrees with persisted IB forces",
    )
    _validate_history_loads(history[-1], final_load, rtol=load_rtol, atol=load_atol)
    body_force = final_load["body_force"]
    recorded_force = _history_vector(history[-1], "body_force")
    reference_speed = abs(float(config["physics"]["inflow_speed"]))
    if reference_speed == 0.0:
        motion = config["ib"]["motion"]
        reference_speed = (
            2.0
            * np.pi
            * float(motion["frequency"])
            * float(motion["amplitude"])
        )
    reference_force = (
        0.5
        * float(config["physics"]["density"])
        * reference_speed**2
        * np.pi
        * float(config["ib"]["radius"]) ** 2
    )
    expected_coefficients = body_force / reference_force
    recorded_coefficients = np.asarray(
        [history[-1]["Cd"], history[-1]["Cy"], history[-1]["Cz"]], dtype=float
    )
    diagnostic_rtol = 5.0e-4 if precision == "float32" else 1.0e-10
    diagnostic_atol = 5.0e-6 if precision == "float32" else 1.0e-12
    _require(
        np.allclose(
            recorded_force,
            body_force,
            rtol=diagnostic_rtol,
            atol=diagnostic_atol,
        ),
        "history final body-force vector disagrees with persisted corrected load",
    )
    _require(
        np.allclose(
            recorded_coefficients,
            expected_coefficients,
            rtol=diagnostic_rtol,
            atol=diagnostic_atol,
        ),
        "history final force coefficients disagree with persisted corrected body load",
    )
    _require(
        math.isclose(
            float(history[-1]["time"]),
            final_time,
            rel_tol=0.0,
            abs_tol=_time_tolerance(completed_step, dt),
        ),
        "history final time disagrees with final_field",
    )
    return {
        "result_directory": str(root),
        "case_name": metadata["case_name"],
        "case_kind": metadata["case_kind"],
        "backend": metadata["backend"],
        "precision": precision,
        "completed_step": completed_step,
        "history_rows": len(history),
        "force_balance_max": max(float(row["force_balance_relerr"]) for row in history),
        "force_balance_recomputed": recomputed_balance,
        "force_balance_atol": balance_atol,
        "config_fingerprint": fingerprint,
        "coverage": "full_field",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_directory",
        type=Path,
        help="directory containing the mandatory run artifacts and optional final field",
    )
    parser.add_argument(
        "--force-balance-atol",
        type=_positive_atol,
        default=None,
        help=(
            "maximum accepted history.force_balance_relerr; defaults to 1e-9 for "
            "float64 and 2e-5 for float32"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = validate_result(
            args.result_directory,
            force_balance_atol=args.force_balance_atol,
        )
    except ValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # A validator crash must never be reported as acceptance.
        print(f"INVALID: validator failure ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(
        "VALID: "
        f"case={summary['case_name']} backend={summary['backend']} "
        f"precision={summary['precision']} step={summary['completed_step']} "
        f"history_rows={summary['history_rows']} "
        f"coverage={summary['coverage']} "
        f"force_balance_max={summary['force_balance_max']:.6e} "
        f"atol={summary['force_balance_atol']:.6e}"
    )
    if summary["force_balance_recomputed"] is None:
        print(
            "coverage_note=final_field.npz was intentionally disabled; "
            "force balance is verified from history only"
        )
    else:
        print(
            "force_balance_recomputed="
            f"{summary['force_balance_recomputed']:.6e}"
        )
    print(f"fingerprint={summary['config_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

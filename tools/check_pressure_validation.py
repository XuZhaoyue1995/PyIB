"""Recompute the published pressure and runtime-load benchmark from CSV evidence.

Requires NumPy; imports no PyIB code. Accepts the supplied ZIP or its extracted
directory. The default is checks/pressure-validation-20260925.zip in this source
distribution. A supplied full-field pressure gauge is an input to this audit,
not independently regenerated here. See the evidence README for scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def vector(table, prefix):
    return np.column_stack([table[prefix + "_" + d] for d in "xyz"])


def fit(times, values, omega, phase):
    angle = omega * times + phase
    design = np.column_stack([np.cos(angle), np.sin(angle), np.ones_like(angle)])
    coeff, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank != 3:
        raise ValueError("Harmonic fit does not have full rank")
    return coeff, float(np.sqrt(np.mean((values - design @ coeff) ** 2)))


def audit(root):
    checks = []

    def check(name, actual, expected, atol=1e-12, rtol=1e-9):
        a, b = np.asarray(actual), np.asarray(expected)
        ok = a.shape == b.shape and bool(np.allclose(a, b, atol=atol, rtol=rtol))
        delta = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.shape == b.shape else None
        checks.append({"name": name, "passed": ok, "max_absolute_difference": delta})

    manifest = load(root / "MANIFEST.json")
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Manifest path leaves evidence directory")
        check("manifest hash " + item["path"], path.is_file() and sha(path) == item["sha256"], True)
    expected = load(root / "expected.json")
    provenance = load(root / "provenance.json")
    runs = []
    for label in ("D32", "D40"):
        run = root / "runs" / label
        metadata, phases = load(run / "metadata.json"), load(run / "phases.json")
        original = next(r for r in provenance["runs"] if r["run"] == label)
        check(label + " original history hash", sha(run / "history.csv") == original["original_history_sha256"], True)
        c = metadata["config"]
        check(label + " published case matches metadata", load(root / "cases" / (label + ".json")) == c, True)
        history = np.genfromtxt(run / "history.csv", names=True, delimiter=",")
        check(label + " finite history", all(np.all(np.isfinite(history[k])) for k in history.dtype.names), True)
        radius, rho = c["ib"]["radius"], c["physics"]["density"]
        motion = c["ib"]["motion"]
        omega, phase = 2 * np.pi * motion["frequency"], motion["phase"]
        amplitude, axis = motion["amplitude"], np.array(motion["axis"], float)
        axis /= np.linalg.norm(axis)
        speed, period = amplitude * omega, 2 * np.pi / omega
        nu = speed * 2 * radius / c["physics"]["reynolds"]
        mu, k = rho * nu, radius / np.sqrt(2 * nu / omega)
        lam = (1 + 1j) * k
        # Independent complex evaluation of the reference; no production analyzer imported.
        p_hat = mu * speed / (2 * radius) * (3 + 3 * lam + lam**2)
        f_hat = -6 * np.pi * mu * radius * speed * (1 + lam + lam**2 / 9)
        reference_force_coeff = np.array([f_hat.real, -f_hat.imag])
        force_amplitude = abs(f_hat)
        check(label + " completed run", metadata["complete"], True)
        check(label + " completed steps", [metadata["completed_step"], history["step"][-1]], [c["time"]["steps"]] * 2)
        times = history["time"]
        check(label + " increasing times", np.all(np.diff(times) > 0), True)
        check(label + " time = step dt", times, history["step"] * c["time"]["dt"])
        check(label + " final time 6T", times[-1], 6 * period)
        body = vector(history, "body_force")
        check(label + " body-force accounting", body, -vector(history, "force_fluid") + vector(history, "interior_momentum_rate"))
        use = (times > 4 * period + 1e-10 * period) & (times <= 6 * period + 1e-10 * period)
        prev = (times > 4 * period + 1e-10 * period) & (times <= 5 * period + 1e-10 * period)
        last = (times > 5 * period + 1e-10 * period) & (times <= 6 * period + 1e-10 * period)
        axial_body = body @ axis
        coeff, residual = fit(times[use], axial_body[use], omega, phase)
        amplitude_error = np.linalg.norm(coeff[:2]) / force_amplitude - 1
        phase_error = (np.degrees(np.arctan2(coeff[1], coeff[0]) - np.arctan2(reference_force_coeff[1], reference_force_coeff[0])) + 180) % 360 - 180
        force_error = float(np.linalg.norm(coeff[:2] - reference_force_coeff) / force_amplitude)
        measured = {"cosine": coeff[0], "sine": coeff[1], "mean": coeff[2],
                    "amplitude": np.linalg.norm(coeff[:2]), "amplitude_error_relative": amplitude_error,
                    "phase_error_degrees": phase_error,
                    "harmonic_vector_error_over_theory_amplitude": force_error,
                    "fit_residual_rms_over_theory_amplitude": residual / force_amplitude}
        for name, value in measured.items():
            check(label + " runtime fit " + name, value, expected[label]["body_force"][name])
        previous_coeff, _ = fit(times[prev], axial_body[prev], omega, phase)
        last_coeff, _ = fit(times[last], axial_body[last], omega, phase)
        cycle_coeff = float(np.linalg.norm(last_coeff - previous_coeff) / force_amplitude)
        cycle_rms = float(np.sqrt(np.mean((axial_body[last] - np.interp(times[last] - period, times, axial_body))**2)) / force_amplitude)
        check(label + " cycle coefficient change", cycle_coeff, expected[label]["cycle_coefficient_change_over_theory_amplitude"])
        check(label + " cycle waveform repeat", cycle_rms, expected[label]["cycle_waveform_repeat_rms_over_theory_amplitude"])
        check(label + " phase count", len(phases), 5)
        check(label + " phase positions", [r["time"] / period for r in phases], [5, 5.25, 5.5, 5.75, 6])
        numerators = {"pressure_raw": [], "pressure_corrected": []}
        denominators = []
        phase_metrics = []
        for row in phases:
            name = label + " step " + str(row["step"])
            table = np.genfromtxt(run / row["file"], names=True, delimiter=",")
            check(name + " pressure-only columns", table.dtype.names == ("x", "y", "z", "nx", "ny", "nz", "area", "pressure_raw", "pressure_corrected"), True)
            check(name + " finite pressure table", all(np.all(np.isfinite(table[k])) for k in table.dtype.names), True)
            angle = omega * row["time"] + phase
            xyz = np.column_stack([table[d] for d in "xyz"])
            normals = np.column_stack([table["n" + d] for d in "xyz"])
            area = table["area"]
            center = np.asarray(c["ib"]["center"]) + amplitude * np.sin(angle) * axis
            check(name + " radius", np.linalg.norm(xyz - center, axis=1), np.full(len(area), radius))
            check(name + " normal", normals, (xyz - center) / radius)
            check(name + " positive areas", np.all(area > 0), True)
            check(name + " total area", np.sum(area), 4 * np.pi * radius**2)
            check(name + " run identity", row["run_id"] == metadata["run_id"], True)
            check(name + " time = step dt", row["time"], row["step"] * c["time"]["dt"])
            exact = (p_hat * np.exp(1j * angle)).real * (normals @ axis)
            denominator = float(np.sum(area * exact**2))
            denominators.append(denominator)
            metrics = {}
            for field in numerators:
                error = table[field] + row["pressure_gauge_shift"] - exact
                num = float(np.sum(area * error**2))
                numerators[field].append(num)
                rms = float(np.sqrt(num / np.sum(area)))
                peak = float(np.max(np.abs(exact)))
                values = {"absolute_rms": rms, "rms_over_phase_max_analytic": rms / peak,
                          "max_error_over_phase_max_analytic": float(np.max(np.abs(error)) / peak),
                          "phase_max_analytic": peak}
                for metric, value in values.items():
                    check(name + " " + field + " " + metric, value, row["expected"][field][metric])
                metrics[field] = {**values, "relative_area_L2": float(np.sqrt(num / denominator))}
            p = table["pressure_corrected"]
            def integrate_pressure(value):
                return -np.sum((value - np.average(value, weights=area))[:, None] * normals * area[:, None], axis=0)
            pressure_force = integrate_pressure(p)
            check(name + " pressure-force quadrature", pressure_force, row["expected"]["pressure_force"]["numerical"])
            check(name + " pressure-force gauge invariance", integrate_pressure(p + 0.73), pressure_force)
            exact_pressure_force = -4 * np.pi * radius**2 / 3 * (p_hat * np.exp(1j * angle)).real * axis
            check(name + " analytic pressure force", exact_pressure_force, row["expected"]["pressure_force"]["analytic"])
            check(name + " pressure-force error normalization", np.linalg.norm(pressure_force - exact_pressure_force) / force_amplitude,
                  row["expected"]["pressure_force"]["difference_norm_over_total_theory_amplitude"])
            indices = np.flatnonzero(history["step"] == row["step"])
            check(name + " history phase sample exists", len(indices), 1)
            if len(indices) == 1:
                check(name + " body force at phase", body[indices[0]], row["expected"]["body_force"])
            phase_metrics.append({"step": row["step"], "time_over_period": row["time"] / period, "pressure": metrics})
        aggregate = {}
        for field, numerator in numerators.items():
            result = float(np.sqrt(sum(numerator[:4]) / sum(denominators[:4])))
            aggregate[field] = result
            check(label + " " + field + " four-phase area/time L2", result, expected[label][field]["four_phase_relative_area_time_L2"])
        runs.append({"name": label, "reynolds": c["physics"]["reynolds"], "cells": metadata["mesh"]["total_cells"],
                     "completed_steps": metadata["completed_step"], "runtime_force": measured,
                     "cycle_waveform_repeat_rms_over_theory_amplitude": cycle_rms,
                     "surface_pressure_relative_area_time_L2": aggregate, "phase_metrics": phase_metrics})
    return {"schema_version": 1, "scope": "Independent pressure CSV and runtime-load arithmetic/consistency audit for the Re0=0.2 oscillating-sphere benchmark.",
            "limitations": ["Full-volume fields are omitted; supplied pressure gauge shifts and raw-field hashes are provenance inputs, not regenerated by this checker.",
                            "Two jointly refined spatial/time resolutions do not isolate convergence order or domain error.",
                            "Reference is linear unbounded unsteady Stokes; simulations use finite-Re equations in a finite box."],
            "runs": runs, "checks_passed": sum(c["passed"] for c in checks), "checks_total": len(checks), "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "checks" / "pressure-validation-20260925.zip")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="pyib_pressure_check_") as temp:
        if args.evidence.suffix.lower() == ".zip":
            root = Path(temp)
            with zipfile.ZipFile(args.evidence) as z:
                for member in z.infolist():
                    if not (root / member.filename).resolve().is_relative_to(root.resolve()):
                        raise ValueError("Archive path leaves extraction directory")
                z.extractall(root)
            root = root / "pressure-validation-20260925"
        else:
            root = args.evidence.resolve()
        result = audit(root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"checks_passed": result["checks_passed"], "checks_total": result["checks_total"],
                      "failed": [c for c in result["checks"] if not c["passed"]],
                      "runs": [{"name": r["name"],
                                "corrected_pressure_L2_percent": 100 * r["surface_pressure_relative_area_time_L2"]["pressure_corrected"],
                                "body_force_harmonic_vector_error_percent": 100 * r["runtime_force"]["harmonic_vector_error_over_theory_amplitude"]}
                               for r in result["runs"]]}, indent=2))
    return 0 if result["checks_passed"] == result["checks_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

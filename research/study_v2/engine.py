"""Fresh, protocol-locked study-v2 integrations of the supplied physical model.

Public API: evaluate_job(job), run_baseline(job). No previous experiment runner
or result cache is imported. One process may execute jobs sequentially; parallel
calls must use processes, since the supplied solver's BDF symbol is instrumented
locally while a solve is active. Existing job IDs are never reused or overwritten.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL_PATH = HERE / "protocol.json"
FROZEN_PROTOCOL_SHA256 = "9c8d446971a1fa6085cbc74aa8565e97151b3c2266a2db7005be292176a042e5"
PHASES = {"search", "validation", "perturbation", "waveform", "baseline"}
COUNTERS = ("aux", "rxn", "elec", "charge", "conv", "wout")
OVERRIDES = {"h", "lambda0", "gamma_f_factor", "initial_C", "ambient_C", "heater_realization_factor"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_protocol():
    raw = PROTOCOL_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("Frozen study-v2 protocol hash mismatch; an amendment is required.")
    return json.loads(raw.decode("utf-8"))


PROTOCOL = read_protocol()
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["NUMBA_CACHE_DIR"] = str(HERE / "runtime" / "numba")
sys.dont_write_bytecode = True
sys.path.append(str(ROOT / "research" / "deps"))
sys.path.insert(0, str(ROOT / "代码"))
import numpy as np
import scipy
import numba
from numpy.polynomial import polynomial as poly
from 模型物理 import Config, FROZEN_CONFIG, FuelCellModel, build_shared, TM
import 数值求解 as solver

_ACTIVE = False


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=encode,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def source_hashes():
    return {relative(p): digest(p) for p in
            (Path(__file__), PROTOCOL_PATH, ROOT / "代码" / "模型物理.py", ROOT / "代码" / "数值求解.py")}


def normalize_job(job, *, baseline=False):
    read_protocol()  # Every worker and every job checks the frozen bytes again.
    result = json.loads(json_bytes(job))
    name = result.get("job_id")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("job_id must be a nonempty path-free ASCII identifier")
    result.setdefault("phase", "baseline" if baseline else "search")
    if result["phase"] not in PHASES or (baseline and result["phase"] != "baseline"):
        raise ValueError("Unknown or inconsistent phase")
    refine = result.setdefault("refine", 1)
    if isinstance(refine, bool) or refine not in (1, 2, 4):
        raise ValueError("refine must be 1, 2 or 4")
    result.setdefault("overrides", {})
    if set(result["overrides"]) - OVERRIDES:
        raise ValueError("Unsupported physical override")
    for key, value in result["overrides"].items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Nonfinite or nonnumeric override: {key}")
    if baseline:
        policy = result.get("policy", result.get("scheme"))
        if policy not in ("preheat", "concurrent"):
            raise ValueError("Baseline policy must be preheat or concurrent")
        if result["overrides"]:
            raise ValueError("Frozen first-crossing baselines use nominal parameters")
        result.update(policy=policy, powers=[1.] * 5, dense=True, profile="constant")
        return result
    powers = np.asarray(result.get("powers"), dtype=float)
    if powers.shape != (5,) or not np.isfinite(powers).all() or np.any(powers < 0) or np.any(powers > 1):
        raise ValueError("powers must contain five finite command densities in [0,1]")
    result["powers"] = powers.tolist()
    for key, default in (("allowance_K", 0.), ("dense", False), ("profile", "constant")):
        result.setdefault(key, default)
    if result["profile"] not in ("constant", "front_loaded", "back_loaded"):
        raise ValueError("Unknown heat profile")
    for key in ("cutoff_s", "reserve_C", "allowance_K"):
        if not isinstance(result.get(key), (int, float)) or not math.isfinite(result[key]):
            raise ValueError(f"Missing or invalid {key}")
    if not 10 <= result["cutoff_s"] <= 90 or result["allowance_K"] < 0:
        raise ValueError("Cutoff or numerical allowance outside the frozen domain")
    if result["reserve_C"] not in PROTOCOL["operating_protocol"]["temperature_reserves_C"]:
        raise ValueError("Temperature reserve is not in the frozen protocol")
    if not isinstance(result["dense"], bool):
        raise ValueError("dense must be Boolean")
    return result


@contextmanager
def reserve_output(job):
    folder = HERE / "runs" / job["phase"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (job["job_id"] + ".json")
    archive = path.with_suffix(".npz")
    lock = path.with_suffix(".lock")
    if path.exists() or archive.exists() or lock.exists():
        raise FileExistsError(f"Job ID already exists; no cache or overwrite permitted: {path}")
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started_utc": utc_now(), "job_id": job["job_id"]}))
    try:
        if path.exists() or archive.exists():
            raise FileExistsError(f"Output appeared while reserving job: {path}")
        yield path, archive
    finally:
        lock.unlink(missing_ok=True)


def make_model(job):
    specification = PROTOCOL["model"]
    overrides = job["overrides"]
    cfg = replace(Config(**FROZEN_CONFIG),
                  area_cm2=float(specification["active_area_cm2"]),
                  h=float(overrides.get("h", specification["h_W_m2K"])),
                  lambda0=float(overrides.get("lambda0", specification["lambda0"])),
                  gamma_f=FROZEN_CONFIG["gamma_f"] * float(overrides.get("gamma_f_factor", 1.)),
                  rtol=float(PROTOCOL["numerics"]["rtol"]),
                  max_step=float(PROTOCOL["numerics"]["max_step_s"]),
                  success_margin_K=float(PROTOCOL["fresh_supporting_experiments"]["first_crossing_and_continuation"]["first_crossing_C"]))
    model = FuelCellModel(n_cells=5, refine=job["refine"], endplates=True,
                          config=cfg, backend="numba", mesh_builder=build_shared)
    initial_C = float(overrides.get("initial_C", specification["initial_C"]))
    ambient_C = float(overrides.get("ambient_C", specification["ambient_C"]))
    realization = float(overrides.get("heater_realization_factor", 1.))
    if realization <= 0:
        raise ValueError("heater_realization_factor must be positive")
    if realization != 1.:
        original_rhs = model.rhs

        def realized_rhs(t, y, j_c, q, ambient):
            return original_rhs(t, y, j_c, np.asarray(q) * realization, ambient)

        model.rhs = realized_rhs
    return model, initial_C, ambient_C, realization


def current_ramp(t):
    return min(.005 * max(float(t), 0.), .3)


def waveform(powers, cutoff, profile):
    powers = np.asarray(powers, dtype=float)
    delta = np.minimum(.25 * powers, .95 * (1. - powers)) if profile != "constant" else np.zeros(5)
    sign = 1. if profile == "front_loaded" else -1.

    def command(t):
        if t >= cutoff:
            return np.zeros(5)
        return powers + sign * delta if t < cutoff / 2 else powers - sign * delta

    def dose_until(t):
        duration = min(max(float(t), 0.), cutoff)
        first = min(duration, cutoff / 2)
        return powers * duration + sign * delta * (2 * first - duration)

    knots = [cutoff, 60.]
    if profile != "constant":
        knots.append(cutoff / 2)
    return command, dose_until, sorted(set(knots)), delta


def real_roots(coefficients, lo, hi):
    coefficients = np.trim_zeros(np.asarray(coefficients), "b")
    roots = poly.polyroots(coefficients) if len(coefficients) > 1 else []
    return [float(np.clip(z.real, lo, hi)) for z in roots
            if abs(z.imag) < 1e-7 and lo - 1e-10 <= z.real <= hi + 1e-10]


def analyze_dense(segments, interval, actual_end, *, threshold=None, require_complete=True):
    """Extrema/continuous durations of accepted numerical interpolants only."""
    begin, end = map(float, interval)
    observed_end = min(end, float(actual_end))
    minimum = math.inf
    when = cell_number = None
    coverage = []
    positive = []
    errors = []
    for segment in segments:
        t0, t1, cc = segment["t0"], segment["t1"], segment["coefficients"]
        if t1 <= begin or t0 >= observed_end:
            continue
        lo = max(0., (begin - t0) / (t1 - t0))
        hi = min(1., (observed_end - t0) / (t1 - t0))
        if hi <= lo:
            continue
        coverage.append((t0 + lo * (t1 - t0), t0 + hi * (t1 - t0)))
        errors.append(segment["reconstruction_error"])
        cuts = [lo, hi]
        for cell in range(cc.shape[1]):
            candidates = [lo, hi] + real_roots(poly.polyder(cc[:, cell]), lo, hi)
            values = poly.polyval(candidates, cc[:, cell])
            index = int(np.argmin(values))
            if values[index] < minimum:
                minimum = float(values[index])
                when = float(t0 + candidates[index] * (t1 - t0))
                cell_number = cell + 1
            if threshold is not None:
                shifted = cc[:, cell].copy()
                shifted[0] -= threshold
                cuts.extend(real_roots(shifted, lo, hi))
        if threshold is not None:
            cuts = np.unique(np.clip(cuts, lo, hi))
            for left, right in zip(cuts[:-1], cuts[1:]):
                if np.min(poly.polyval((left + right) / 2, cc)) >= threshold:
                    a, b = t0 + left * (t1 - t0), t0 + right * (t1 - t0)
                    if positive and abs(a - positive[-1][1]) <= 1e-8:
                        positive[-1][1] = b
                    else:
                        positive.append([float(a), float(b)])
    coverage_ok = bool(coverage and abs(coverage[0][0] - begin) <= 1e-8
                       and abs(coverage[-1][1] - end) <= 1e-8
                       and all(abs(a[1] - b[0]) <= 1e-8 for a, b in zip(coverage, coverage[1:])))
    finite = math.isfinite(minimum)
    valid = finite and (coverage_ok or not require_complete)
    return dict(valid=bool(valid), method="Extrema and crossings of accepted BDF temperature polynomials; numerical interpolation, not an exact-solution proof",
                interval_s=[begin, end], threshold_C=threshold, continuous_window_coverage=coverage_ok,
                minimum_mea_mean_C=minimum if valid else None, minimum_time_s=when if valid else None,
                minimum_cell_one_based=cell_number if valid else None,
                observed_partial_minimum_C=minimum if finite and not coverage_ok else None,
                segment_count=len(coverage), polynomial_reconstruction_max_error_K=max(errors, default=None),
                all_means_ge_threshold_intervals_s=positive if threshold is not None else None,
                longest_continuous_ge_threshold_s=max((b - a for a, b in positive), default=0.) if threshold is not None else None,
                all_means_above_threshold_for_full_interval=bool(coverage_ok and finite and threshold is not None and minimum >= threshold))


def integrate(model, horizon, initial_C, ambient_C, current, heating, *, breakpoints=(),
              stop_success=False, initial_state=None, dense_interval=None, full_dense=False,
              sample_dt=None, wall_s=None):
    """Observe the original solver without altering its physical or retry rules."""
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("Engine is process-parallel, not thread-parallel")
    _ACTIVE = True
    segments = []
    original_bdf = solver.BDF
    original_jacobian = solver.phase_branch_jacobian

    class RecordedBDF(original_bdf):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Discard a just-observed step if the supplied solver retries it.
            while segments and segments[-1]["t1"] > self.t + 1e-12:
                segments.pop()

        def step(self):
            before = float(self.t)
            message = super().step()
            after = float(self.t)
            capture = dense_interval is not None and after > dense_interval[0] and before < dense_interval[1]
            if self.status != "failed" and after > before and capture:
                dense = self.dense_output()
                order = int(dense.order)
                z = np.linspace(0., 1., order + 1)

                def means(at):
                    state = dense(before + (after - before) * at)
                    temperature = state[model.sl["T"]].T * model.scale[model.sl["T"]] - TM
                    return model.mesh.mea_means(temperature)

                coefficients = poly.polyfit(z, means(z), order)
                checks = np.array([.37, .63])
                error = float(np.max(np.abs(poly.polyval(checks, coefficients).T - means(checks))))
                segment = dict(t0=before, t1=after, coefficients=coefficients,
                               reconstruction_error=error, order=order)
                if full_dense:
                    segment.update(D_scaled=np.asarray(dense.D).copy(),
                                   t_shift=np.asarray(dense.t_shift).copy(),
                                   denom=np.asarray(dense.denom).copy())
                segments.append(segment)
            return message

    if dense_interval is not None:
        solver.BDF = RecordedBDF
    solver.phase_branch_jacobian = solver.active_jacobian
    options = dict(stop_success=stop_success,
                   voltage_floor=float(PROTOCOL["operating_protocol"]["voltage_floor_V"]),
                   charge_cap=None, breakpoints=list(breakpoints), save_fields=True,
                   sample_dt=float(PROTOCOL["numerics"]["sample_dt_s"] if sample_dt is None else sample_dt),
                   max_wall_s=float(PROTOCOL["numerics"]["max_wall_s"] if wall_s is None else wall_s),
                   reuse_jacobian=False, initial_state=initial_state)
    try:
        result, final = solver.simulate(model, float(horizon), initial_C, ambient_C,
                                        current, heating, **options)
    finally:
        solver.BDF = original_bdf
        solver.phase_branch_jacobian = original_jacobian
        _ACTIVE = False
    recipe = {k: v for k, v in options.items() if k != "initial_state"}
    recipe.update(method="scipy.integrate.BDF", jacobian="supplied active_jacobian, branches='Rvl', batch_size=64",
                  initial_state_mode="fresh model.initial" if initial_state is None else "physical terminal state with counters reset",
                  first_interval_max_step_s=.01, first_interval_end_s=.1,
                  native_negative_inventory_retry="Supplied solver, up to 20 local retries; same physical inputs, tightened numerical tolerance/step")
    return result, final, segments, recipe


def model_arrays(model):
    mesh = model.mesh
    arrays = dict(state_scale=model.scale, atol_vector=model.atol, x_m=mesh.x,
                  mesh_kind=np.asarray(mesh.kind, dtype="U8"), mesh_cell_ids=mesh.cell_ids,
                  mesh_dx_m=mesh.dx, mesh_capacity_J_K=mesh.capacity_J_K,
                  mesh_conductance_W_K=mesh.conductance_W_K,
                  mesh_boundary_conductance_W_K=mesh.boundary_conductance_W_K,
                  pidx=model.pidx, dx_p_m=model.dx_p, heat_b=model.heat_b, heat_c=model.heat_c,
                  bidx=model.bidx, dx_b_m=model.dx_b)
    for cell, indices in enumerate(mesh.mea_indices):
        arrays[f"mea_indices_{cell}"] = indices
    return arrays


def archive_stage(result, final, initial, segments, prefix=""):
    arrays = {prefix + "states_scaled": result.pop("states_scaled"),
              prefix + "final_state_scaled": final,
              prefix + "initial_state_scaled": initial,
              prefix + "time_s": np.asarray(result["series"]["time_s"])}
    # The common metadata is exported once by model_arrays.
    result.pop("state_scale", None)
    result.pop("x_m", None)
    if segments:
        coefficients = np.zeros((len(segments), 6, 5))
        for i, segment in enumerate(segments):
            coefficients[i, :len(segment["coefficients"])] = segment["coefficients"]
        arrays[prefix + "dense_segment_times_s"] = np.array([[s["t0"], s["t1"]] for s in segments])
        arrays[prefix + "dense_temperature_coefficients_C"] = coefficients
        arrays[prefix + "dense_orders"] = np.array([s["order"] for s in segments], dtype=np.int8)
        arrays[prefix + "dense_reconstruction_error_K"] = np.array([s["reconstruction_error"] for s in segments])
        if "D_scaled" in segments[0]:
            D = np.zeros((len(segments), 6, len(final)))
            shifts = np.zeros((len(segments), 5))
            denominator = np.zeros((len(segments), 5))
            for i, segment in enumerate(segments):
                order = segment["order"]
                D[i, :order + 1] = segment["D_scaled"]
                shifts[i, :order] = segment["t_shift"]
                denominator[i, :order] = segment["denom"]
            arrays[prefix + "full_dense_D_scaled"] = D
            arrays[prefix + "full_dense_t_shift_s"] = shifts
            arrays[prefix + "full_dense_denom_s"] = denominator
    return arrays


def add_accounting(result, model, initial, final, current):
    start = solver.state_metrics(model, initial, current(0.))
    terminal = solver.state_metrics(model, final, current(result["time_s"]))
    result.update(net_input_energy_J=result["aux_energy_J"] + result["hydrogen_thermoneutral_energy_J"] - result["electrical_output_J"],
                  input_energy_J=result["aux_energy_J"] + result["hydrogen_thermoneutral_energy_J"],
                  initial_enthalpy_J=start["enthalpy_J"], final_enthalpy_J=terminal["enthalpy_J"],
                  enthalpy_change_J=terminal["enthalpy_J"] - start["enthalpy_J"],
                  initial_water_kg=start["water_total_kg"], final_water_kg=terminal["water_total_kg"],
                  conv_energy_J=terminal["conv_energy_J"], water_out_kg=terminal["water_out_kg"],
                  final_local_temperature_min_C=float(np.min(final[model.sl["T"]] * model.scale[model.sl["T"]]) - TM),
                  final_ice_max=terminal["ice_max"], final_bulk_ice_max=terminal["ice_volume_max"])
    result["energy_counter_identity_residual_J"] = result["hydrogen_thermoneutral_energy_J"] - result["electrical_output_J"] - result["rxn_energy_J"]


def base_metadata(job):
    sources = source_hashes()
    signature = dict(job=job, frozen_protocol_sha256=FROZEN_PROTOCOL_SHA256, sources=sources)
    return dict(job_id=job["job_id"], eval_id=job["job_id"], phase=job["phase"], job=job,
                refine=job["refine"], started_utc=utc_now(), finished=False,
                frozen_protocol_sha256=FROZEN_PROTOCOL_SHA256, protocol=PROTOCOL,
                sources=sources, source_sha256=sources, request_signature=signature,
                signature_hash=hashlib.sha256(json_bytes(signature)).hexdigest(),
                fresh_integration=True, reused_prior_result=False,
                versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__))


def numerical_exception(metadata, exc):
    metadata.update(status="NUMERICAL_FAILURE", failure_reason=f"engine_exception: {type(exc).__name__}: {exc}",
                    exception_traceback=traceback.format_exc(), complete_180s=False,
                    numerically_resolved=False, g_window_K=None, g_dense_K=None,
                    min_window_mea_C=None, feasible=False,
                    audit=dict(passed=False, violations=["engine_exception"]),
                    terminal_state_available=False)
    return metadata


def finish(path, archive, result, arrays, started):
    if arrays:
        with archive.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        result.update(states_archive=relative(archive), states_archive_sha256=digest(archive),
                      archive_keys=sorted(arrays))
    result.update(finished=True, finished_utc=utc_now(), job_elapsed_s=time.perf_counter() - started)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, default=encode, allow_nan=False)
    return result


def evaluate_job(job: dict) -> dict:
    job = normalize_job(job)
    with reserve_output(job) as (path, archive):
        started = time.perf_counter()
        metadata = base_metadata(job)
        arrays = {}
        try:
            model, initial_C, ambient_C, realization = make_model(job)
            initial = model.initial(initial_C)
            command, dose, knots, delta = waveform(job["powers"], job["cutoff_s"], job["profile"])
            window = PROTOCOL["operating_protocol"]["window_s"]
            raw, final, segments, recipe = integrate(
                model, 180., initial_C, ambient_C, current_ramp, command,
                breakpoints=knots, dense_interval=window if job["dense"] else None)
            raw.update(metadata)
            complete = raw["status"] == "COMPLETE" and abs(raw["time_s"] - 180.) <= 1e-8
            times = np.asarray(raw["series"]["time_s"])
            means = np.asarray(raw["series"]["temperature_C"])
            mask = (times >= window[0] - 1e-8) & (times <= window[1] + 1e-8)
            expected_samples = int(round((window[1] - window[0]) / PROTOCOL["numerics"]["sample_dt_s"])) + 1
            sample_valid = complete and int(mask.sum()) == expected_samples
            minimum = float(np.min(means[mask])) if sample_valid else None
            target = float(job["reserve_C"] + job["allowance_K"])
            margin = minimum - target if minimum is not None else None
            dense = analyze_dense(segments, window, raw["time_s"], threshold=target) if job["dense"] else None
            if dense is not None and not complete:
                dense["valid"] = False
                dense["minimum_mea_mean_C"] = None
                dense["all_means_above_threshold_for_full_interval"] = False
            dense_margin = dense["minimum_mea_mean_C"] - target if dense and dense["valid"] else None
            audit = solver.validate_constraints(raw, require_success=False)
            planned = model.cfg.area_cm2 * float(np.sum(job["powers"])) * job["cutoff_s"]
            expected_actual_cells = model.cfg.area_cm2 * realization * dose(raw["time_s"])
            energy_error = float(np.max(np.abs(np.asarray(raw["aux_energy_cells_J"]) - expected_actual_cells)))
            if energy_error > 1e-4:
                audit["violations"].append("realized_heater_energy_integration_mismatch")
                audit["passed"] = False
            add_accounting(raw, model, initial, final, current_ramp)
            raw.update(complete_180s=complete, numerically_resolved=raw["status"] not in ("NUMERICAL_LIMIT", "NUMERICAL_FAILURE"),
                       terminal_state_available=True, window_sample_count=int(mask.sum()),
                       sampled_window_covered=sample_valid, min_window_mea_C=minimum,
                       g_window_K=margin, g_dense_K=dense_margin, dense_dwell=dense,
                       reserve_C=job["reserve_C"], allowance_K=job["allowance_K"], target_temperature_C=target,
                       feasible=bool(complete and audit["passed"] and margin is not None and margin >= 0.
                                     and (not job["dense"] or dense_margin is not None and dense_margin >= 0.)),
                       audit=audit, planned_energy_J=planned, actual_aux_energy_J=raw["aux_energy_J"],
                       expected_realized_full_aux_energy_J=planned * realization,
                       realized_energy_integration_error_per_cell_max_J=energy_error,
                       expected_realized_energy_cells_to_terminal_J=expected_actual_cells,
                       powers=job["powers"], cutoff_s=job["cutoff_s"], profile=job["profile"], waveform_delta_W_cm2=delta,
                       heater_realization_factor=realization, initial_C=initial_C, ambient_C=ambient_C,
                       backend=model.backend, model_kwargs=dict(n_cells=5, refine=job["refine"], endplates=True,
                                                               backend="numba", mesh_builder="build_shared"),
                       solver_kwargs=recipe, state_slices={k: [v.start, v.stop] for k, v in model.sl.items()},
                       thermal_nodes=model.nt, state_count=model.size,
                       realized_heater_rule="Original RHS receives command times realization factor; auxiliary counters and thermal source both scale.")
            raw["series"]["q_command_W_cm2"] = np.asarray(raw["series"]["q_W_cm2"])
            raw["series"]["q_realized_W_cm2"] = realization * np.asarray(raw["series"]["q_W_cm2"])
            arrays = model_arrays(model)
            arrays.update(archive_stage(raw, final, initial, segments))
            result = raw
        except Exception as exc:
            result = numerical_exception(metadata, exc)
        return finish(path, archive, result, arrays, started)


def run_baseline(job: dict) -> dict:
    job = normalize_job(job, baseline=True)
    with reserve_output(job) as (path, archive):
        started = time.perf_counter()
        metadata = base_metadata(job)
        arrays = {}
        try:
            model, initial_C, ambient_C, _ = make_model(job)
            initial = model.initial(initial_C)
            current = (lambda t: 0.) if job["policy"] == "preheat" else current_ramp
            first, final, pre_segments, recipe = integrate(
                model, 300., initial_C, ambient_C, current, lambda t: np.ones(5),
                breakpoints=[60.], stop_success=True, dense_interval=[0., 300.], full_dense=True)
            add_accounting(first, model, initial, final, current)
            arrays = model_arrays(model)
            arrays.update(archive_stage(first, final, initial, pre_segments, "pre_"))
            first.update(metadata)
            first.update(policy=job["policy"], backend=model.backend, initial_C=initial_C, ambient_C=ambient_C,
                         first_threshold_C=.01, first_threshold_reached=first["status"] == "SUCCESS",
                         first_threshold_time_s=first["time_s"] if first["status"] == "SUCCESS" else None,
                         first_phase_audit=solver.validate_constraints(first, require_success=True),
                         solver_kwargs=recipe, state_slices={k: [v.start, v.stop] for k, v in model.sl.items()},
                         thermal_nodes=model.nt, state_count=model.size,
                         full_dense_reconstruction="y_scaled(t)=D[0]+sum_{j=1..order} D[j]*product_{i=0..j-1}((t-t_shift[i])/denom[i]); retain only order+1 rows. Clip at the recorded stage endpoint.",
                         first_dense=analyze_dense(pre_segments, [0., first["time_s"]], first["time_s"]),
                         numerically_resolved=first["status"] not in ("NUMERICAL_LIMIT", "NUMERICAL_FAILURE"))
            if first["status"] == "SUCCESS":
                post_initial = final.copy()
                for key in COUNTERS:
                    post_initial[model.sl[key]] = 0.
                physical_mask = np.ones(model.size, dtype=bool)
                for key in COUNTERS:
                    physical_mask[model.sl[key]] = False
                continuity_error = float(np.max(np.abs((post_initial - final)[physical_mask])))
                post, post_final, post_segments, post_recipe = integrate(
                    model, 60., initial_C, ambient_C, lambda t: .3, lambda t: np.zeros(5),
                    initial_state=post_initial, dense_interval=[0., 60.], full_dense=True)
                add_accounting(post, model, post_initial, post_final, lambda t: .3)
                post_complete = post["status"] == "COMPLETE" and abs(post["time_s"] - 60.) <= 1e-8
                diagnoses = {}
                for threshold in (0., .01):
                    strict = analyze_dense(post_segments, [0., 60.], post["time_s"], threshold=threshold,
                                           require_complete=False)
                    tolerant = analyze_dense(post_segments, [0., 60.], post["time_s"], threshold=threshold - 1e-8,
                                             require_complete=False)
                    strict.update(complete_60s_protocol=post_complete, temperature_tolerance_K=1e-8,
                                  sustained_60s_with_numerical_tolerance=bool(post_complete and tolerant["all_means_above_threshold_for_full_interval"]),
                                  longest_continuous_ge_threshold_with_tolerance_s=tolerant["longest_continuous_ge_threshold_s"])
                    diagnoses[f"{threshold:.2f}"] = strict
                post.update(dense_temperature_diagnostics=diagnoses, audit=solver.validate_constraints(post, require_success=False),
                            solver_kwargs=post_recipe, previous_stage_physical_state_preserved=continuity_error == 0.,
                            previous_stage_physical_scaled_state_max_error=continuity_error,
                            counters_reset=list(COUNTERS))
                arrays.update(archive_stage(post, post_final, post_initial, post_segments, "post_"))
                first.update(postload=post, persistence_status=post["status"],
                             minimum_postload_mea_mean_C=diagnoses["0.00"]["minimum_mea_mean_C"],
                             minimum_postload_voltage_V=post["min_voltage_V"],
                             sustained_60s_at_0_C=diagnoses["0.00"]["sustained_60s_with_numerical_tolerance"],
                             sustained_60s_at_0p01_C=diagnoses["0.01"]["sustained_60s_with_numerical_tolerance"],
                             total_aux_energy_J=first["aux_energy_J"] + post["aux_energy_J"],
                             total_hydrogen_thermoneutral_energy_J=first["hydrogen_thermoneutral_energy_J"] + post["hydrogen_thermoneutral_energy_J"],
                             total_electrical_output_J=first["electrical_output_J"] + post["electrical_output_J"],
                             total_net_input_energy_J=first["net_input_energy_J"] + post["net_input_energy_J"])
            else:
                first.update(postload=None, persistence_status="NOT_EVALUATED_FIRST_THRESHOLD_NOT_REACHED")
            result = first
        except Exception as exc:
            result = numerical_exception(metadata, exc)
        return finish(path, archive, result, arrays, started)


def verify_engine(smoke=True):
    """Local implementation checks; optional simulation horizon is only 0.02 s."""
    checks = {}
    protocol_hash = digest(PROTOCOL_PATH)
    checks["frozen_protocol"] = protocol_hash == FROZEN_PROTOCOL_SHA256
    coefficients = np.tile(np.array([.16, -1., 1.])[:, None], (1, 5))
    segment = dict(t0=0., t1=60., coefficients=coefficients, reconstruction_error=0., order=2)
    diagnostic = analyze_dense([segment], [0., 60.], 60., threshold=0.)
    checks["internal_polynomial_minimum"] = abs(diagnostic["minimum_mea_mean_C"] + .09) < 1e-12
    checks["threshold_crossing_duration"] = abs(diagnostic["longest_continuous_ge_threshold_s"] - 12.) < 1e-8
    short = analyze_dense([segment], [0., 60.], 10., threshold=0.)
    checks["incomplete_window_not_certified"] = not short["valid"] and short["minimum_mea_mean_C"] is None
    for profile in ("constant", "front_loaded", "back_loaded"):
        powers = [.2, .5, 1., .5, .2]
        command, dose, knots, delta = waveform(powers, 38., profile)
        checks[f"{profile}_matched_energy"] = bool(np.allclose(dose(38.), 38. * np.asarray(powers), rtol=0., atol=1e-14))
        checks[f"{profile}_bounded"] = bool(np.all(command(0.) >= 0) and np.all(command(0.) <= 1)
                                             and np.all(command(20.) >= 0) and np.all(command(20.) <= 1))
    smoke_record = None
    if smoke:
        job = dict(refine=1, overrides={"heater_realization_factor": 1.05})
        model, initial_C, ambient_C, factor = make_model(job)
        initial = model.initial(initial_C)
        raw, final, segments, recipe = integrate(model, .02, initial_C, ambient_C, current_ramp,
                                                lambda t: np.ones(5), dense_interval=[0., .02],
                                                full_dense=True, sample_dt=.01, wall_s=120.)
        expected = 5 * 25 * factor * .02
        checks["smoke_complete_0p02s"] = raw["status"] == "COMPLETE" and abs(raw["time_s"] - .02) < 1e-12
        checks["actuator_realization_above_one"] = abs(raw["aux_energy_J"] - expected) < 1e-9
        checks["dense_capture"] = bool(segments)
        native_error = 0.
        for segment in segments:
            order = segment["order"]
            at = segment["t0"] + .37 * (segment["t1"] - segment["t0"])
            products = np.cumprod((at - segment["t_shift"]) / segment["denom"])
            y = segment["D_scaled"][0] + np.dot(products, segment["D_scaled"][1:order + 1])
            mea = model.mesh.mea_means(y[model.sl["T"]] * model.scale[model.sl["T"]] - TM)
            native_error = max(native_error, float(np.max(abs(mea - poly.polyval(.37, segment["coefficients"])))))
        checks["native_full_state_dense_reconstruction"] = native_error < 1e-9
        smoke_record = dict(horizon_s=.02, status=raw["status"], time_s=raw["time_s"], backend=model.backend,
                            realized_aux_energy_J=raw["aux_energy_J"], expected_aux_energy_J=expected,
                            water_residual_relative=raw["water_residual_relative"], energy_residual_relative=raw["energy_residual_relative"],
                            native_full_state_reconstruction_error_K=native_error, steps=raw["steps"], solver_kwargs=recipe)
    record = dict(verification_utc=utc_now(), sources=source_hashes(), checks=checks, passed=all(checks.values()),
                  smoke=smoke_record, scientific_jobs_executed=0,
                  audit_layers=dict(source="Frozen protocol and original model/solver hashes retained; no legacy runner/result imports.",
                                    logic="Analytic heat-dose, extrema, crossing and incomplete-window checks; original numerical rules retained.",
                                    scope="Dense minima certify accepted numerical interpolants only. The 0.02 s smoke does not validate a 180 s physical prediction."))
    path = HERE / "engine_verification.json"
    if path.exists():
        raise FileExistsError("Use a new verification record name after an implementation amendment")
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=encode, allow_nan=False), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, default=encode), flush=True)
    if not record["passed"]:
        raise AssertionError("Engine implementation verification failed")
    return record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no-smoke", action="store_true")
    arguments = parser.parse_args()
    if not arguments.verify:
        parser.error("Scientific jobs are scheduled through evaluate_job/run_baseline by the orchestrator")
    verify_engine(smoke=not arguments.no_smoke)

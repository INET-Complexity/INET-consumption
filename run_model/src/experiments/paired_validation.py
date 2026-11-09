"""Serial per-revision/seed paired validation, with self-contained local artifacts."""

import hashlib
import json
import platform
import resource
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

import numba
import numpy as np
import pandas as pd

from .income_transfer import OneTimeIncomeTransfer
from .lfa_endowment import OneTimeLFAEndowment
from .paired_checkpoint import PairedCheckpoint, _RandomState
from .paired_response import _ids, measure_paired_response
from .quarter_outcomes import QuarterOutcomeCollector
from .two_pass_diagnostics import TwoPassDiagnosticCapture

# Household case artifacts must stay under this root (overridable in tests only).
ARTIFACT_ROOT = Path("/private/tmp")


def save_artifact(directory, payload):
    """Exclusive JSON + numeric NPZ; retain raw NaNs without nonstandard JSON."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {}

    def pack(value):
        if isinstance(value, (pd.Series, pd.DataFrame)):
            if isinstance(value.index, pd.MultiIndex) or (
                isinstance(value, pd.DataFrame) and isinstance(value.columns, pd.MultiIndex)
            ):
                raise TypeError("MultiIndex artifacts require an explicit schema extension.")
            data = {
                "index": pack(list(value.index)),
                "index_name": pack(value.index.name),
                "data": pack(value.to_numpy()),
            }
            if isinstance(value, pd.Series):
                data["name"] = pack(value.name)
                return {"__series__": data}
            data.update(columns=pack(list(value.columns)), columns_name=pack(value.columns.name))
            return {"__dataframe__": data}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("Object arrays cannot be saved in validation artifacts.")
            key = f"a{len(arrays)}"
            arrays[key] = value
            return {"__ndarray__": key}
        if isinstance(value, np.generic):
            return pack(value.item())
        if is_dataclass(value):
            return {field.name: pack(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, Mapping):
            if len({str(key) for key in value}) != len(value):
                raise ValueError("JSON key conversion would lose provenance.")
            return {str(key): pack(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(item) for item in value]
        if isinstance(value, bytes):
            return {"__bytes_hex__": value.hex()}
        if isinstance(value, Enum):
            return pack(value.value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not np.isfinite(value):
            return {"__nonfinite__": str(value)}
        if value is None or isinstance(value, (str, float, int, bool)):
            return value
        raise TypeError(f"Unsupported artifact value: {type(value).__name__}")

    metadata = pack(payload)
    with (directory / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with (directory / "records.json").open("x") as stream:
        json.dump({"schema_version": 1, "payload": metadata}, stream, allow_nan=False, indent=2)


def load_artifact(directory):
    """Read this module's artifact schema without pickle or executable objects."""
    directory = Path(directory)
    metadata = json.loads((directory / "records.json").read_text())
    if metadata["schema_version"] != 1:
        raise ValueError("Unsupported validation artifact schema.")
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:

        def unpack(value):
            if isinstance(value, dict):
                if set(value) in ({"__series__"}, {"__dataframe__"}):
                    kind = next(iter(value))
                    data = unpack(value[kind])
                    index = pd.Index(data["index"], name=data["index_name"])
                    if kind == "__series__":
                        return pd.Series(data["data"], index=index, name=data["name"])
                    return pd.DataFrame(
                        data["data"], index=index, columns=pd.Index(data["columns"], name=data["columns_name"])
                    )
                if set(value) == {"__ndarray__"}:
                    return arrays[value["__ndarray__"]].copy()
                if set(value) == {"__bytes_hex__"}:
                    return bytes.fromhex(value["__bytes_hex__"])
                if set(value) == {"__nonfinite__"}:
                    return float(value["__nonfinite__"])
                return {key: unpack(item) for key, item in value.items()}
            if isinstance(value, list):
                return [unpack(item) for item in value]
            return value

        return unpack(metadata["payload"])


def _equal(a, b, path="zero"):
    if is_dataclass(a) and is_dataclass(b):
        for field in fields(a):
            _equal(getattr(a, field.name), getattr(b, field.name), f"{path}.{field.name}")
    elif isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            raise ValueError(f"Zero-control keys differ at {path}.")
        for key in a:
            _equal(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f"Zero-control numeric divergence at {path}.")
    elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            raise ValueError(f"Zero-control length divergence at {path}.")
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            _equal(left, right, f"{path}[{index}]")
    elif isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if not np.isclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f"Zero-control scalar divergence at {path}.")
    elif a != b:
        raise ValueError(f"Zero-control divergence at {path}.")


def _measure(control, shock, control_passes, shock_passes, *, amounts, kind, pre_deflator, horizons, groups):
    common = dict(
        intervention_kind=kind,
        value_units="nominal",
        control_ids=control[0].household_ids,
        shock_ids=shock[0].household_ids,
        control_periods=[q.completed_row for q in control],
        shock_periods=[q.completed_row for q in shock],
        control_deflators=[q.deflator for q in control],
        shock_deflators=[q.deflator for q in shock],
        intervention_nominal=amounts,
        pre_shock_control_deflator=pre_deflator,
        horizons=horizons,
        groups=groups,
    )
    result = {}
    for key in ("consumption", "consumption_cash_expenditure", "consumption_including_housing"):
        if any(key not in q.nominal_outcomes for q in (*control, *shock)):
            raise ValueError(f"Required realised outcome unavailable: {key}.")
        result[key] = measure_paired_response(
            outcome=key,
            control_nominal=np.stack([q.nominal_outcomes[key] for q in control]),
            shock_nominal=np.stack([q.nominal_outcomes[key] for q in shock]),
            **common,
        )
    for pass_kind in ("first", "replacement"):
        left = [q for q in control_passes if q.pass_kind == pass_kind]
        right = [q for q in shock_passes if q.pass_kind == pass_kind]
        if [q.period_id for q in left] != list(common["control_periods"]) or [q.period_id for q in right] != list(
            common["shock_periods"]
        ):
            raise ValueError("Target pass records are incomplete or misaligned.")
        for key in ("goods_demand", "target_consumption_calibrated_total", "target_consumption_long_run"):
            label = f"{pass_kind}/{key}"
            if key != "goods_demand" and any(key not in q.components for q in (*left, *right)):
                result[label] = {"status": "unavailable", "component": key}
                continue

            def values(records):
                return np.stack(
                    [q.goods_demand.sum(axis=1) if key == "goods_demand" else q.components[key] for q in records]
                )

            result[label] = measure_paired_response(
                outcome=label, control_nominal=values(left), shock_nominal=values(right), **common
            )
    return result


def run_paired_case(
    model,
    *,
    country_code,
    revision,
    seed,
    start_iteration,
    horizon_periods,
    household_ids,
    transfer_nominal,
    endowment_nominal,
    horizons,
    include_zero,
    output_dir,
    provenance,
    groups=None,
    wall_seconds=2700,
    output_bytes=20 * 1024**3,
    peak_rss_bytes=16 * 1024**3,
):
    """Run ONE prepared revision/seed case; no data loading, fitting or registration.

    Caller must obtain experiment approval and pin real inputs before production
    use. Synthetic fixture tests exercise this API without a production campaign.
    Use separate processes/source environments for the two revisions. Provenance
    is recorded as caller-supplied, not falsely certified by this runner. Resource
    limits are checked at iteration/artifact boundaries, not preemptive kills.
    Case budgets do not replace the registered campaign-wide budget.
    All household artifacts must stay under /private/tmp, in a fresh directory.
    """
    started = time.monotonic()
    if any(
        isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
        for x in (seed, start_iteration, horizon_periods)
    ):
        raise ValueError("Seed/start/horizon must be integers.")
    if start_iteration < 1 or horizon_periods < 1 or not isinstance(include_zero, bool):
        raise ValueError("Require unshocked initialization, positive horizon and explicit zero-arm choice.")
    if not isinstance(revision, str) or not revision or not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("Declare revision and caller-supplied provenance.")
    if set(model.countries) != {country_code} or getattr(model, "random_seed", seed) != seed:
        raise ValueError("Case requires one declared country and the initialized simulation seed.")
    json.dumps(dict(provenance), allow_nan=False)  # Fail preflight on invalid metadata.
    ids = _ids(household_ids)
    country = model.countries[country_code]
    if len(country.households.states["corr_individuals"]) != len(ids):
        raise ValueError("Registered household IDs must cover the whole population.")
    initial_iteration = len(country.households.ts.historic("income")) - 1
    if initial_iteration > start_iteration:
        raise ValueError("Simulation has already passed the intervention checkpoint.")
    probes = [
        OneTimeIncomeTransfer(country_code, year=1, month=1, amounts_nominal=x)
        for x in (transfer_nominal, endowment_nominal)
    ]
    if any(p.amounts_nominal.shape != (len(ids),) for p in probes):
        raise ValueError("Frozen intervention vectors must cover the population.")
    horizons = tuple(horizons)
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(
            isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)) or h < 1 or h > horizon_periods
            for h in horizons
        )
    ):
        raise ValueError("Declared measurement horizons must fit the case.")
    if any(not np.isfinite(x) or x <= 0 for x in (wall_seconds, output_bytes, peak_rss_bytes)):
        raise ValueError("Resource budgets must be finite and positive.")
    root = Path(output_dir).resolve()
    if not root.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError(f"Household case artifacts must remain under {ARTIFACT_ROOT}.")
    initial_checkpoint = PairedCheckpoint(model)  # Reject bound hooks before any output.
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = dict(
        schema_version=1,
        status="running",
        revision=revision,
        seed=int(seed),
        country_code=country_code,
        start_iteration=int(start_iteration),
        horizon_periods=int(horizon_periods),
        horizons=list(horizons),
        include_zero=include_zero,
        provenance=dict(provenance),
        provenance_verified=False,
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        numba=numba.__version__,
        pandas=pd.__version__,
        executable=sys.executable,
        resource_limits=dict(
            wall_seconds=float(wall_seconds), output_bytes=int(output_bytes), peak_rss_bytes=int(peak_rss_bytes)
        ),
        completed_arms=[],
    )

    def write_manifest():
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    def check_limits():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if time.monotonic() - started > wall_seconds or rss > peak_rss_bytes or used > output_bytes:
            raise RuntimeError("Case resource budget exceeded; abort before further work.")

    try:
        write_manifest()
        check_limits()
        warmed = initial_checkpoint.clone()
        with initial_checkpoint.activate_rng():
            for t in range(initial_iteration, start_iteration):
                check_limits()
                warmed.iterate(t=t)
            checkpoint = PairedCheckpoint(warmed)
            rng_state = _RandomState.capture()
        pre_country = warmed.countries[country_code]
        economy = pre_country.economy
        pre_deflator = economy.current_consumer_price_level() / economy.initial_consumer_price_level()
        owned_rng = getattr(pre_country.households.functions["consumption"], "_idiosyncratic_rng", None)
        save_artifact(
            root / "checkpoint_evidence",
            {
                "scope": "audit evidence, not a portable whole-model restart file",
                "household_ids": ids,
                "household_memberships": pre_country.households.states["corr_individuals"],
                "household_histories": pre_country.households.ts.dicts,
                "global_rng_state": rng_state,
                "consumption_rng_state": None if owned_rng is None else owned_rng.bit_generator.state,
                "income_belief_states": {
                    k: v for k, v in pre_country.households.states.items() if "income_belief" in k
                },
                "residual": getattr(pre_country.households.functions["consumption"], "_household_epsilon", None),
                "pre_shock_deflator": pre_deflator,
                "configuration": warmed.configuration.model_dump(mode="json"),
            },
        )
        arms = ["control"] + (["zero"] if include_zero else []) + ["income_transfer", "lfa_endowment"]
        collected = {}
        for name in arms:
            check_limits()
            arm = checkpoint.clone()
            intervention = None
            if name != "control":
                amounts = (
                    np.zeros(len(ids))
                    if name == "zero"
                    else probes[0 if name == "income_transfer" else 1].amounts_nominal
                )
                cls = OneTimeLFAEndowment if name == "lfa_endowment" else OneTimeIncomeTransfer
                intervention = cls(
                    country_code, year=arm.timestep.year, month=arm.timestep.month, amounts_nominal=amounts
                )
                intervention.install(arm)
            capture = TwoPassDiagnosticCapture(arm)
            outcomes = QuarterOutcomeCollector(
                arm,
                revision=revision,
                arm=name,
                seed=seed,
                start_iteration=start_iteration,
                household_ids={country_code: ids},
                interventions={} if intervention is None else {country_code: intervention},
            )
            error = None
            try:
                with checkpoint.activate_rng(), capture, outcomes:
                    for t in range(start_iteration, start_iteration + horizon_periods):
                        check_limits()
                        arm.iterate(t=t)
            except BaseException as exc:
                error = exc
                raise
            finally:
                try:
                    save_artifact(
                        root / name,
                        {
                            "status": "failed" if error else "complete",
                            "passes": capture.records,
                            "quarters": outcomes.records,
                            "booking": None if intervention is None else intervention.booking,
                            "opening_endowment_snapshot": getattr(intervention, "opening_snapshot", None),
                        },
                    )
                except BaseException as save_error:
                    if error is None:
                        raise
                    manifest["partial_save_error"] = repr(save_error)
            collected[name] = (outcomes.records, capture.records)
            if name == "zero":
                control, passes = collected["control"]
                zero, zero_passes = collected[name]
                for a, b in zip(control, zero, strict=True):
                    for field in (
                        "household_ids",
                        "deflator",
                        "nominal_outcomes",
                        "household_state",
                        "funding_plan",
                        "banks",
                        "government",
                        "economy",
                        "unavailable",
                        "source_rows",
                    ):
                        _equal(getattr(a, field), getattr(b, field), field)
                _equal(passes, zero_passes, "target_passes")
                manifest["zero_control"] = "passed"
            manifest["completed_arms"].append(name)
            write_manifest()
            check_limits()
        responses = {
            name: _measure(
                collected["control"][0],
                collected[name][0],
                collected["control"][1],
                collected[name][1],
                amounts=probe.amounts_nominal,
                kind=name,
                pre_deflator=pre_deflator,
                horizons=horizons,
                groups=groups,
            )
            for name, probe in zip(("income_transfer", "lfa_endowment"), probes, strict=True)
        }
        save_artifact(root / "responses", responses)
        check_limits()
        manifest.update(
            status="complete",
            elapsed_seconds=time.monotonic() - started,
            zero_control=manifest.get("zero_control", "not_requested"),
        )
        manifest["artifact_sha256"] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        write_manifest()
        return manifest
    except BaseException as exc:
        manifest.update(
            status="failed", error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.monotonic() - started
        )
        (root / "failure.txt").write_text(traceback.format_exc())
        write_manifest()
        raise

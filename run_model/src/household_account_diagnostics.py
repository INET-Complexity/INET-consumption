"""Household account diagnostics for Stage 0 consumption validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from macro_data.processing.synthetic_population.hfcs_synthetic_population import compute_notebook_household_accounts

PAPER_ACCOUNT_COLUMNS = ("lfa", "ifa", "ha", "mr", "db", "nw")
RUNTIME_HOUSEHOLD_FIELDS = (
    "income",
    "consumption",
    "wealth_deposits",
    "wealth_other_financial_assets",
    "wealth_main_residence",
    "wealth_other_properties",
    "mortgage_debt",
    "consumption_loan_debt",
    "debt",
    "net_wealth",
)
OPTIONAL_RUNTIME_HOUSEHOLD_FIELDS = ("wealth_other_real_assets",)
DEFAULT_QUANTILES = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


@dataclass(frozen=True)
class HouseholdAccountDiagnostics:
    panel: pd.DataFrame
    moments: pd.DataFrame
    quantiles: pd.DataFrame
    reconciliation: pd.DataFrame


def _country_entry(mapping: dict, country_code: str):
    if country_code in mapping:
        return mapping[country_code]
    for key, value in mapping.items():
        if str(key) == str(country_code):
            return value
    available = ", ".join(sorted(str(key) for key in mapping))
    raise KeyError(f"Country {country_code} not found in data pickle. Available countries: {available}")


def _population_frame(datawrapper, country_code: str) -> pd.DataFrame:
    country = _country_entry(datawrapper.synthetic_countries, country_code)
    population = getattr(country, "population", None)
    household_data = getattr(population, "household_data", None)
    if household_data is None:
        raise ValueError(f"Country {country_code} population does not expose household_data.")
    if not isinstance(household_data, pd.DataFrame):
        raise TypeError("population.household_data must be a pandas DataFrame.")
    return household_data


def make_paper_account_panel(datawrapper, country_code: str) -> pd.DataFrame:
    """Compute notebook-style accounts and expose the helper index as household_id."""
    household_data = _population_frame(datawrapper, country_code)
    try:
        accounts = compute_notebook_household_accounts(household_data)
    except KeyError as exc:
        raise ValueError(
            "Cannot compute paper household accounts from the data pickle. "
            "The required HFCS source columns are unavailable."
        ) from exc
    return accounts.reset_index(names="household_id")


def _read_household_field(group: h5py.Group, field: str) -> np.ndarray:
    if field not in group:
        raise KeyError(f"Missing HDF5 household field: {field}")
    values = np.asarray(group[field], dtype=float)
    if values.ndim != 2:
        raise ValueError(f"HDF5 household field {field} must be 2D, got shape {values.shape}.")
    return values


def _period_index(period: int | None, n_rows: int) -> int:
    if n_rows <= 0:
        raise ValueError("HDF5 household fields must contain at least one row.")
    if period is None:
        return n_rows - 1
    row = int(period)
    if row < 0:
        row = n_rows + row
    if row < 0 or row >= n_rows:
        raise ValueError(f"period {period} is outside available HDF5 rows 0..{n_rows - 1}.")
    return row


def _read_cpi_series(economy_group: h5py.Group, cpi_source: str) -> np.ndarray:
    if cpi_source not in economy_group:
        raise KeyError(f"Missing HDF5 economy CPI field: {cpi_source}")
    cpi = np.asarray(economy_group[cpi_source], dtype=float).reshape(-1)
    if np.any(~np.isfinite(cpi)) or np.any(cpi <= 0.0):
        raise ValueError(f"{cpi_source} must be finite and strictly positive for household account diagnostics.")
    return cpi


def read_runtime_household_panel(
    h5_path: str | Path,
    country_code: str,
    *,
    period: int | None = None,
    fixed_cpi_source: str = "cpi_fixed_basket",
    transaction_cpi_source: str = "cpi_transaction",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read one HDF5 household row and attach fixed/transaction CPI deflators."""
    with h5py.File(h5_path, "r") as handle:
        if country_code not in handle:
            available = ", ".join(sorted(handle.keys()))
            raise KeyError(f"Country {country_code} not found in model HDF5. Available groups: {available}")
        country_group = handle[country_code]
        households = country_group["households"]
        economy = country_group["economy"]

        arrays = {field: _read_household_field(households, field) for field in RUNTIME_HOUSEHOLD_FIELDS}
        optional_arrays = {
            field: _read_household_field(households, field)
            for field in OPTIONAL_RUNTIME_HOUSEHOLD_FIELDS
            if field in households
        }
        rows = {values.shape[0] for values in arrays.values()}
        widths = {values.shape[1] for values in arrays.values()}
        for field, values in optional_arrays.items():
            if values.shape[0] not in rows or values.shape[1] not in widths:
                raise ValueError(f"Optional HDF5 household field {field} must match required household field shapes.")
        if len(rows) != 1:
            raise ValueError(f"Runtime household fields must have matching row counts, got {sorted(rows)}.")
        if len(widths) != 1:
            raise ValueError(f"Runtime household fields must have matching household widths, got {sorted(widths)}.")
        row = _period_index(period, rows.pop())
        n_households = widths.pop()

        fixed_cpi = _read_cpi_series(economy, fixed_cpi_source)
        transaction_cpi = _read_cpi_series(economy, transaction_cpi_source)
        if row >= fixed_cpi.size:
            raise ValueError(f"{fixed_cpi_source} length {fixed_cpi.size} does not include period {row}.")
        if row >= transaction_cpi.size:
            raise ValueError(f"{transaction_cpi_source} length {transaction_cpi.size} does not include period {row}.")

    panel = pd.DataFrame({"household_id": np.arange(n_households, dtype=int)})
    for field, values in arrays.items():
        column = "model_spendable_income" if field == "income" else f"runtime_{field}"
        panel[column] = values[row]
    for field, values in optional_arrays.items():
        panel[f"runtime_{field}"] = values[row]
    panel["period"] = row
    panel["cpi_fixed_basket"] = fixed_cpi[row]
    panel["cpi_transaction"] = transaction_cpi[row]
    metadata = {
        "period": row,
        "n_households": n_households,
        "fixed_cpi_source": fixed_cpi_source,
        "transaction_cpi_source": transaction_cpi_source,
        "fixed_cpi": float(fixed_cpi[row]),
        "transaction_cpi": float(transaction_cpi[row]),
        "missing_runtime_fields": [],
        "missing_optional_runtime_fields": [
            field for field in OPTIONAL_RUNTIME_HOUSEHOLD_FIELDS if field not in optional_arrays
        ],
        "cpi_valid": True,
    }
    return panel, metadata


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_values = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    denominator_values = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    return pd.Series(
        np.divide(
            numerator_values,
            denominator_values,
            out=np.full(numerator_values.shape, np.nan),
            where=np.isfinite(denominator_values) & (denominator_values > 0.0),
        ),
        index=numerator.index,
    )


def build_household_account_panel(
    datawrapper,
    h5_path: str | Path,
    country_code: str,
    *,
    period: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    paper = make_paper_account_panel(datawrapper, country_code)
    runtime, metadata = read_runtime_household_panel(h5_path, country_code, period=period)
    metadata["household_count_mismatch"] = len(paper) != metadata["n_households"]
    if metadata["household_count_mismatch"]:
        raise ValueError(
            "Household-count mismatch between paper accounts and HDF5 runtime fields: "
            f"{len(paper)} != {metadata['n_households']}."
        )

    panel = paper.merge(runtime, on="household_id", how="inner", validate="one_to_one")
    if len(panel) != len(paper):
        raise ValueError("Household account join lost rows; household_id values must match HDF5 household order.")

    runtime_assets = (
        panel["runtime_wealth_deposits"]
        + panel["runtime_wealth_other_financial_assets"]
        + panel["runtime_wealth_main_residence"]
        + panel["runtime_wealth_other_properties"]
    )
    panel["runtime_gross_assets"] = runtime_assets
    panel["runtime_other_real_assets"] = panel.get("runtime_wealth_other_real_assets", 0.0)
    panel["runtime_total_assets"] = panel["runtime_gross_assets"] + panel["runtime_other_real_assets"]
    panel["runtime_total_debt_components"] = panel["runtime_mortgage_debt"] + panel["runtime_consumption_loan_debt"]
    panel["runtime_net_wealth_identity"] = panel["runtime_total_assets"] - panel["runtime_total_debt_components"]

    nominal_columns = list(PAPER_ACCOUNT_COLUMNS) + [
        "model_spendable_income",
        "runtime_consumption",
        "runtime_wealth_deposits",
        "runtime_wealth_other_financial_assets",
        "runtime_wealth_main_residence",
        "runtime_wealth_other_properties",
        "runtime_mortgage_debt",
        "runtime_consumption_loan_debt",
        "runtime_debt",
        "runtime_net_wealth",
        "runtime_gross_assets",
        "runtime_other_real_assets",
        "runtime_total_assets",
        "runtime_total_debt_components",
    ]
    for column in nominal_columns:
        panel[f"real_{column}_fixed_cpi"] = panel[column] / panel["cpi_fixed_basket"]
        panel[f"real_{column}_transaction_cpi"] = panel[column] / panel["cpi_transaction"]

    income = panel["model_spendable_income"]
    assets = panel["lfa"] + panel["ifa"] + panel["ha"]
    for column in PAPER_ACCOUNT_COLUMNS:
        panel[f"{column}_to_income"] = _safe_ratio(panel[column], income)
    panel["consumption_to_income"] = _safe_ratio(panel["runtime_consumption"], income)
    panel["paper_debt_to_income"] = _safe_ratio(panel["mr"] + panel["db"], income)
    panel["runtime_debt_to_income"] = _safe_ratio(panel["runtime_debt"], income)
    panel["paper_debt_to_assets"] = _safe_ratio(panel["mr"] + panel["db"], assets)
    panel["runtime_debt_to_assets"] = _safe_ratio(panel["runtime_debt"], panel["runtime_gross_assets"])
    return panel, metadata


def summarize_moments(panel: pd.DataFrame) -> pd.DataFrame:
    numeric = panel.select_dtypes(include=[np.number])
    rows = []
    for column in numeric.columns:
        values = numeric[column].replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "variable": column,
                "count": int(values.count()),
                "missing": int(values.isna().sum()),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def summarize_quantiles(panel: pd.DataFrame, quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> pd.DataFrame:
    numeric = panel.select_dtypes(include=[np.number])
    rows = []
    for column in numeric.columns:
        values = numeric[column].replace([np.inf, -np.inf], np.nan).dropna()
        row: dict[str, object] = {"variable": column}
        for quantile in quantiles:
            row[f"q{quantile:g}"] = values.quantile(quantile) if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_reconciliation(panel: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    paper_identity = panel["lfa"] + panel["ifa"] + panel["ha"] - panel["mr"] - panel["db"]
    runtime_paper_assets_gap = panel["runtime_gross_assets"] - (panel["lfa"] + panel["ifa"] + panel["ha"])
    runtime_paper_debt_gap = panel["runtime_debt"] - (panel["mr"] + panel["db"])
    rows = [
        ("period", metadata["period"]),
        ("households", len(panel)),
        ("household_count_mismatch", bool(metadata.get("household_count_mismatch", False))),
        ("missing_runtime_fields", ",".join(metadata.get("missing_runtime_fields", []))),
        ("missing_optional_runtime_fields", ",".join(metadata.get("missing_optional_runtime_fields", []))),
        ("fixed_cpi_source", metadata["fixed_cpi_source"]),
        ("transaction_cpi_source", metadata["transaction_cpi_source"]),
        ("fixed_cpi", metadata["fixed_cpi"]),
        ("transaction_cpi", metadata["transaction_cpi"]),
        ("cpi_valid", bool(metadata.get("cpi_valid", False))),
        ("max_abs_nw_identity_error", float((panel["nw"] - paper_identity).abs().max())),
        ("mean_runtime_minus_paper_assets", float(runtime_paper_assets_gap.mean())),
        ("mean_runtime_minus_paper_debt", float(runtime_paper_debt_gap.mean())),
        ("mean_runtime_minus_paper_net_wealth", float((panel["runtime_net_wealth"] - panel["nw"]).mean())),
        (
            "mean_consumption_loan_debt_minus_paper_db",
            float((panel["runtime_consumption_loan_debt"] - panel["db"]).mean()),
        ),
        (
            "max_abs_runtime_debt_component_gap",
            float((panel["runtime_debt"] - panel["runtime_total_debt_components"]).abs().max()),
        ),
        (
            "max_abs_runtime_net_wealth_identity_gap",
            float((panel["runtime_net_wealth"] - panel["runtime_net_wealth_identity"]).abs().max()),
        ),
    ]
    return pd.DataFrame(rows, columns=["check", "value"])


def run_household_account_diagnostics(
    *,
    datawrapper,
    model_h5: str | Path,
    country_code: str,
    period: int | None = None,
) -> HouseholdAccountDiagnostics:
    panel, metadata = build_household_account_panel(datawrapper, model_h5, country_code, period=period)
    moments = summarize_moments(panel)
    quantiles = summarize_quantiles(panel)
    reconciliation = build_reconciliation(panel, metadata)
    return HouseholdAccountDiagnostics(
        panel=panel,
        moments=moments,
        quantiles=quantiles,
        reconciliation=reconciliation,
    )


def _summary_markdown(diagnostics: HouseholdAccountDiagnostics, country_code: str) -> str:
    reconciliation = diagnostics.reconciliation.set_index("check")["value"]
    lines = [
        f"# Household Account Diagnostics: {country_code}",
        "",
        f"- Period: {reconciliation.get('period')}",
        f"- Households: {reconciliation.get('households')}",
        f"- Max paper net-wealth identity error: {reconciliation.get('max_abs_nw_identity_error')}",
        f"- Mean runtime minus paper debt: {reconciliation.get('mean_runtime_minus_paper_debt')}",
        "- Runtime consumption_loan_debt is reported separately from paper db.",
        f"- Fixed CPI source: {reconciliation.get('fixed_cpi_source')} = {reconciliation.get('fixed_cpi')}",
        f"- Transaction CPI source: {reconciliation.get('transaction_cpi_source')} = {reconciliation.get('transaction_cpi')}",
    ]
    return "\n".join(lines) + "\n"


def write_household_account_outputs(
    diagnostics: HouseholdAccountDiagnostics,
    output_dir: str | Path,
    *,
    country_code: str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "panel": output_path / "household_account_panel.csv",
        "moments": output_path / "household_account_moments.csv",
        "quantiles": output_path / "household_account_quantiles.csv",
        "reconciliation": output_path / "household_account_reconciliation.csv",
        "summary": output_path / "household_account_summary.md",
    }
    diagnostics.panel.to_csv(paths["panel"], index=False)
    diagnostics.moments.to_csv(paths["moments"], index=False)
    diagnostics.quantiles.to_csv(paths["quantiles"], index=False)
    diagnostics.reconciliation.to_csv(paths["reconciliation"], index=False)
    paths["summary"].write_text(_summary_markdown(diagnostics, country_code), encoding="utf-8")
    return paths

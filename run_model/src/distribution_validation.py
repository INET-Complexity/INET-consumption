"""Small Plotly/HFCS helpers for ``visualise_model_results.ipynb``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MODEL_COLOR = "#1f77b4"
HFCS_COLOR = "#d62728"
OTHER_COLORS = ("#2ca02c", "#9467bd", "#ff7f0e", "#17becf")

HFCS_COLUMNS = {
    "ID": "ID",
    "SA0100": "Country",
    "HW0010": "Weight",
    "DI2000": "Income",
    "DA1000": "Wealth in Real Assets",
    "DA2100": "Wealth in Financial Assets",
    "DA2101": "Wealth in Deposits",
    "DL1000": "Debt",
    "DN3001": "Net Wealth",
}


def prepare_model_values(
    values: Any,
    timestep: int = -1,
    *,
    scale: float | None = None,
    drop_nonpositive: bool = False,
) -> np.ndarray:
    """Return finite model values for one timestep, optionally rescaled."""
    selected = np.asarray(values, dtype=float)
    if selected.ndim > 1:
        selected = selected[timestep]

    array = _clean_array(selected)
    if scale is not None:
        array = array * scale
    if drop_nonpositive:
        array = array[array > 0]
    return array


def trim_series_to_common_percentile(
    series_by_name: Mapping[str, Any],
    *,
    percentile: float = 99.0,
    lower_bound: float | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    """Trim all series to the lowest requested percentile across them."""
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in the interval (0, 100].")

    cleaned = {name: _clean_array(values) for name, values in series_by_name.items()}
    for name, values in cleaned.items():
        _require_nonempty(values, name)

    upper_limit = min(float(np.percentile(values, percentile)) for values in cleaned.values())
    trimmed = {}
    for name, values in cleaned.items():
        mask = values <= upper_limit
        if lower_bound is not None:
            mask &= values >= lower_bound
        trimmed[name] = values[mask]
    return trimmed, upper_limit


def load_hfcs_wave_dataframes(
    hfcs_data_path: str | Path,
    country_iso2: str,
    years: Sequence[int],
    *,
    eur_to_lcu_by_year: Mapping[int, float] | None = None,
    num_surveys: int = 5,
) -> dict[int, pd.DataFrame]:
    """Load compact household HFCS frames for the requested country and years."""
    hfcs_data_path = Path(hfcs_data_path)
    country_iso2 = country_iso2.upper()
    eur_to_lcu_by_year = eur_to_lcu_by_year or {}

    waves = {}
    for year in years:
        frames = []
        for implicate in range(1, num_surveys + 1):
            path = _find_hfcs_file(hfcs_data_path / str(year), f"D{implicate}.csv")
            frame = _read_hfcs_frame(path, country_iso2, eur_to_lcu_by_year.get(year, 1.0))
            frame["Implicate"] = implicate
            frames.append(frame)
        waves[year] = pd.concat(frames, ignore_index=True)
    return waves


def weighted_hfcs_total(frame: pd.DataFrame, column: str) -> float:
    """Return a survey-weighted total, averaged across HFCS implicates."""
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame["Weight"], errors="coerce")
    mask = values.notna() & weights.notna()
    n_implicates = frame["Implicate"].nunique() if "Implicate" in frame else 1
    return float((values[mask] * weights[mask]).sum() / max(int(n_implicates), 1))


def build_multi_histogram_figure(
    series_by_name: Mapping[str, Any],
    *,
    title: str,
    xaxis_title: str,
    nbinsx: int = 80,
    histnorm: str = "probability",
    opacity: float = 0.45,
    x_tickformat: str = ",.4~f",
    y_tickformat: str = ".2~f",
) -> go.Figure:
    """Build an overlaid histogram with shared bin edges."""
    cleaned = _clean_named_series(series_by_name)
    xbins = _shared_histogram_xbins(cleaned.values(), nbinsx)

    fig = go.Figure()
    for idx, (name, values) in enumerate(cleaned.items()):
        lower_name = name.lower()
        color = MODEL_COLOR if lower_name.startswith("model") else HFCS_COLOR if lower_name.startswith("hfcs") else OTHER_COLORS[
            idx % len(OTHER_COLORS)
        ]
        fig.add_trace(
            go.Histogram(
                x=values,
                name=name,
                marker={"color": color},
                xbins=xbins,
                histnorm=histnorm,
                opacity=opacity,
            )
        )

    fig.update_layout(title_text=title, template="plotly_white", barmode="overlay", xaxis_title=xaxis_title)
    fig.update_xaxes(tickformat=x_tickformat)
    fig.update_yaxes(tickformat=y_tickformat)
    return fig


def build_multi_cdf_figure(
    series_by_name: Mapping[str, Any],
    *,
    title: str,
    xaxis_title: str,
    x_tickformat: str = ",.4~f",
    y_tickformat: str = ".2~f",
) -> go.Figure:
    """Build empirical CDF lines for named series."""
    fig = go.Figure()
    for idx, (name, values) in enumerate(_clean_named_series(series_by_name).items()):
        x_values, y_values = _empirical_cdf(values)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=name,
                line=_comparison_line(name, idx),
                opacity=_series_opacity(name),
            )
        )

    fig.update_layout(title_text=title, template="plotly_white", xaxis_title=xaxis_title)
    fig.update_xaxes(tickformat=x_tickformat)
    fig.update_yaxes(range=[0, 1], tickformat=y_tickformat)
    return fig


def build_multi_lorenz_figure(
    series_by_name: Mapping[str, Any],
    *,
    title: str,
    equality_color: str = "#6c757d",
) -> go.Figure:
    """Build Lorenz curves for named non-negative series."""
    fig = go.Figure()
    for idx, (name, values) in enumerate(_clean_named_series(series_by_name).items()):
        x_values, y_values = _lorenz_curve(values)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=name,
                line=_comparison_line(name, idx),
                opacity=_series_opacity(name),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Equality",
            line={"color": equality_color, "dash": "dash"},
        )
    )

    fig.update_layout(
        title_text=title,
        template="plotly_white",
        xaxis_title="Cumulative share of households",
    )
    fig.update_xaxes(range=[0, 1], tickformat=".2~f")
    fig.update_yaxes(range=[0, 1], tickformat=".2~f")
    return fig


def _read_hfcs_frame(path: Path, country_iso2: str, exchange_rate: float) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, encoding="unicode_escape", engine="pyarrow")
    except (ImportError, ValueError):
        frame = pd.read_csv(path, encoding="unicode_escape", low_memory=False)

    frame.columns = [str(column).upper() for column in frame.columns]
    frame = frame[frame["SA0100"].astype(str).str.upper() == country_iso2].copy()

    columns = [column for column in HFCS_COLUMNS if column in frame]
    frame = frame[columns].rename(columns=HFCS_COLUMNS)
    for column in frame.columns:
        if column not in {"ID", "Country"}:
            frame[column] = pd.to_numeric(frame[column].replace(["A", "M"], np.nan), errors="coerce")

    monetary_columns = [column for column in frame.columns if column not in {"ID", "Country", "Weight"}]
    frame.loc[:, monetary_columns] *= exchange_rate
    frame["Income"] = frame["Income"].clip(lower=0.0)

    for column in ("Wealth in Deposits", "Wealth in Financial Assets", "Wealth in Real Assets", "Debt"):
        if column not in frame:
            frame[column] = 0.0
    frame["Wealth"] = frame["Wealth in Real Assets"] + frame["Wealth in Financial Assets"]
    frame["Wealth in Other Financial Assets"] = frame["Wealth in Financial Assets"] - frame["Wealth in Deposits"]
    if "Net Wealth" not in frame:
        frame["Net Wealth"] = frame["Wealth"] - frame["Debt"]
    return frame


def _find_hfcs_file(year_dir: Path, filename: str) -> Path:
    for candidate in (year_dir / filename, year_dir / filename.lower()):
        if candidate.exists():
            return candidate

    target = filename.lower()
    for candidate in year_dir.glob("*.csv"):
        if candidate.name.lower() == target:
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in {year_dir}.")


def _clean_named_series(series_by_name: Mapping[str, Any]) -> dict[str, np.ndarray]:
    cleaned = {name: _clean_array(values) for name, values in series_by_name.items()}
    for name, values in cleaned.items():
        _require_nonempty(values, name)
    return cleaned


def _clean_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def _require_nonempty(values: np.ndarray, name: str) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one finite value.")


def _shared_histogram_xbins(arrays: Sequence[np.ndarray], nbinsx: int) -> dict[str, float]:
    if nbinsx <= 0:
        raise ValueError("nbinsx must be positive.")

    values = np.concatenate([array for array in arrays if len(array) > 0])
    start = float(np.min(values))
    end = float(np.max(values))
    if start == end:
        padding = max(abs(start) * 0.01, 0.5)
        start -= padding
        end += padding
    return {"start": start, "end": end, "size": (end - start) / nbinsx}


def _empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.sort(values)
    y_values = np.arange(1, len(x_values) + 1, dtype=float) / len(x_values)
    return x_values, y_values


def _lorenz_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.any(values < 0):
        raise ValueError("Lorenz curve values must be non-negative.")

    sorted_values = np.sort(values)
    population_share = np.linspace(0, 1, len(sorted_values) + 1)
    total = sorted_values.sum()
    if total == 0:
        return population_share, np.zeros(len(sorted_values) + 1)
    value_share = np.concatenate(([0.0], np.cumsum(sorted_values) / total))
    return population_share, value_share


def _comparison_line(name: str, idx: int) -> dict[str, Any]:
    is_model = _is_model_series(name)
    lower_name = name.lower()
    color = MODEL_COLOR if lower_name.startswith("model") else HFCS_COLOR if lower_name.startswith("hfcs") else OTHER_COLORS[
        idx % len(OTHER_COLORS)
    ]
    return {
        "color": color,
        "dash": "solid" if is_model else "dot",
        "width": 4 if is_model else 2,
    }


def _series_opacity(name: str) -> float:
    return 1.0 if _is_model_series(name) else 0.75


def _is_model_series(name: str) -> bool:
    return name.lower().startswith("model")

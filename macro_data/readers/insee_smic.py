"""INSEE SMIC data helpers."""

from pathlib import Path

import pandas as pd

DEFAULT_INSEE_SMIC_CSV = (
    Path(__file__).resolve().parents[2] / "run_model" / "data" / "raw_data" / "INSEE" / "SMIC_monthly_values.csv"
)


def load_insee_smic_annual_table(path: str | Path | None = None) -> pd.Series:
    """Load annual net monthly SMIC values from an INSEE monthly CSV.

    ``path`` defaults to :data:`DEFAULT_INSEE_SMIC_CSV` when ``None`` so callers that
    resolve the path from a :class:`DataPaths` registry can pass ``None`` as a fallback.

    The INSEE file is monthly. For model annual bases, only December rows
    (``YYYY-12``) are retained.
    """
    if path is None:
        path = DEFAULT_INSEE_SMIC_CSV
    raw = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["period", "net_monthly_smic", "code"],
        dtype=str,
    )
    annual = raw.loc[raw["period"].str.fullmatch(r"\d{4}-12", na=False), ["period", "net_monthly_smic"]].copy()
    annual["year"] = annual["period"].str[:4].astype(int)
    annual["net_monthly_smic"] = pd.to_numeric(annual["net_monthly_smic"], errors="raise")
    annual = annual.drop_duplicates(subset="year", keep="first").sort_values("year")
    return annual.set_index("year")["net_monthly_smic"].rename("net_monthly_smic")

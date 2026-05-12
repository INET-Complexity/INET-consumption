"""Eurostat CFC-based capital depreciation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country


@dataclass(frozen=True)
class JsonStatDataset:
    """Small JSON-stat 2.0 reader for dense dimension lookups."""

    ids: list[str]
    sizes: list[int]
    dimensions: dict[str, Any]
    values: dict[int, float]

    @classmethod
    def from_file(cls, path: Path) -> "JsonStatDataset":
        if not path.exists():
            raise FileNotFoundError(f"Required Eurostat JSON-stat file not found: {path}")
        with path.open("r") as handle:
            data = json.load(handle)

        sizes = list(data["size"])
        strides: list[int] = []
        product = 1
        for size in reversed(sizes):
            strides.insert(0, product)
            product *= size

        values = {int(index): float(value) for index, value in data.get("value", {}).items()}
        return cls(
            ids=list(data["id"]),
            sizes=sizes,
            dimensions=data["dimension"],
            values=values,
        )

    @property
    def strides(self) -> list[int]:
        strides: list[int] = []
        product = 1
        for size in reversed(self.sizes):
            strides.insert(0, product)
            product *= size
        return strides

    def get(self, **coordinates: str) -> float:
        flat_index = 0
        for dim_offset, dim_name in enumerate(self.ids):
            if dim_name not in coordinates:
                raise ValueError(f"Missing JSON-stat coordinate {dim_name!r}.")
            categories = self.dimensions[dim_name]["category"]["index"]
            code = coordinates[dim_name]
            if code not in categories:
                raise ValueError(f"Eurostat dimension {dim_name!r} does not contain code {code!r}.")
            flat_index += categories[code] * self.strides[dim_offset]

        if flat_index not in self.values:
            raise ValueError(f"Eurostat value is missing for coordinates {coordinates}.")
        return self.values[flat_index]


def _eurostat_geo_code(country_name: Country | str) -> str:
    if isinstance(country_name, Country):
        return country_name.to_two_letter_code()
    country = str(country_name)
    if len(country) == 2:
        return country
    return Country(country).to_two_letter_code()


def _sector_value(dataset: JsonStatDataset, sector: str, **coordinates: str) -> float:
    if sector == "R_S":
        return _sector_value(dataset, "R", **coordinates) + _sector_value(dataset, "S", **coordinates)
    return dataset.get(nace_r2=sector, **coordinates)


def load_eurostat_capital_depreciation_data(
    eurostat_path: Path | str,
    country_name: Country | str,
    year: int,
    industries: list[str],
) -> pd.DataFrame:
    """Load CFC, income-account levels, stock, and model-ready ratios by industry.

    The monetary levels are read from Eurostat current-price national-currency
    datasets. Ratios are unit invariant and are the values used by the model.
    """

    eurostat_path = Path(eurostat_path)
    income = JsonStatDataset.from_file(eurostat_path / "nama_10_a64.json")
    stock = JsonStatDataset.from_file(eurostat_path / "nama_10_nfa_st.json")
    geo = _eurostat_geo_code(country_name)
    year_code = str(year)

    rows = []
    for industry in industries:
        cfc = _sector_value(
            income,
            industry,
            freq="A",
            unit="CP_MNAC",
            na_item="P51C",
            geo=geo,
            time=year_code,
        )
        output = _sector_value(
            income,
            industry,
            freq="A",
            unit="CP_MNAC",
            na_item="P1",
            geo=geo,
            time=year_code,
        )
        intermediate = _sector_value(
            income,
            industry,
            freq="A",
            unit="CP_MNAC",
            na_item="P2",
            geo=geo,
            time=year_code,
        )
        gross_value_added = _sector_value(
            income,
            industry,
            freq="A",
            unit="CP_MNAC",
            na_item="B1G",
            geo=geo,
            time=year_code,
        )
        labour_compensation = _sector_value(
            income,
            industry,
            freq="A",
            unit="CP_MNAC",
            na_item="D1",
            geo=geo,
            time=year_code,
        )
        production_taxes_less_subsidies = _sector_value(
            income,
            industry,
            freq="A",
            unit="CP_MNAC",
            na_item="D29X39",
            geo=geo,
            time=year_code,
        )
        net_operating_surplus = _sector_value(
            income,
            industry,
            freq="A",
            unit="CP_MNAC",
            na_item="B2A3N",
            geo=geo,
            time=year_code,
        )
        net_stock = _sector_value(
            stock,
            industry,
            freq="A",
            unit="CRC_MNAC",
            asset10="N11N",
            geo=geo,
            time=year_code,
        )

        if output <= 0.0:
            raise ValueError(f"Eurostat output must be positive for {country_name} {year} sector {industry}.")
        if net_stock <= 0.0:
            raise ValueError(
                f"Eurostat net fixed capital stock must be positive for {country_name} {year} sector {industry}."
            )

        rows.append(
            {
                "Industry": industry,
                "Consumption of Fixed Capital": cfc,
                "Output": output,
                "Intermediate Consumption": intermediate,
                "Gross Value Added": gross_value_added,
                "Labour Compensation": labour_compensation,
                "Production Taxes Less Subsidies": production_taxes_less_subsidies,
                "Net Operating Surplus": net_operating_surplus,
                "Gross Operating Surplus": net_operating_surplus + cfc,
                "Net Fixed Capital Stock": net_stock,
                "Capital Depreciation Rate": cfc / net_stock,
                "CFC Output Ratio": cfc / output,
                "Labour Output Ratio": labour_compensation / output,
                "Production Tax Output Ratio": production_taxes_less_subsidies / output,
                "Intermediate Output Ratio": intermediate / output,
            }
        )

    return pd.DataFrame(rows).set_index("Industry")


def build_cfc_output_replacement_matrix(
    industries: list[str],
    capital_good_mix: np.ndarray,
    cfc_output_ratios: pd.Series | np.ndarray,
    yearly_factor: float,
) -> pd.DataFrame:
    """Build a replacement matrix with period-output CFC/output column totals."""

    mix = np.asarray(capital_good_mix, dtype=float)
    if mix.ndim != 1 or mix.shape[0] != len(industries):
        raise ValueError("capital_good_mix must be a 1D vector with one value per industry.")
    if mix.sum() <= 0.0:
        raise ValueError("capital_good_mix must have a positive sum.")
    mix = mix / mix.sum()

    ratios = np.asarray(cfc_output_ratios, dtype=float)
    if ratios.ndim != 1 or ratios.shape[0] != len(industries):
        raise ValueError("cfc_output_ratios must be a 1D vector with one value per industry.")
    if np.any(~np.isfinite(ratios)) or np.any(ratios < 0.0):
        raise ValueError("cfc_output_ratios must be finite and non-negative.")
    if yearly_factor <= 0.0:
        raise ValueError("yearly_factor must be positive.")

    matrix = mix[:, None] * ratios[None, :]
    return pd.DataFrame(
        data=matrix,
        index=pd.Index(industries, name="Industries"),
        columns=pd.Index(industries, name="Industries"),
    )

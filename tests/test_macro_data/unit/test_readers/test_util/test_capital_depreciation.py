import json

import numpy as np

from macro_data.readers.util.capital_depreciation import (
    build_cfc_output_replacement_matrix,
    load_eurostat_capital_depreciation_data,
)


def _write_jsonstat(path, ids, dimensions, observations):
    sizes = [len(dimensions[dim]) for dim in ids]
    strides = []
    product = 1
    for size in reversed(sizes):
        strides.insert(0, product)
        product *= size

    dim_payload = {}
    for dim, codes in dimensions.items():
        dim_payload[dim] = {
            "category": {
                "index": {code: index for index, code in enumerate(codes)},
                "label": {code: code for code in codes},
            }
        }

    values = {}
    for coords, value in observations:
        flat_index = sum(dimensions[dim].index(coords[dim]) * strides[i] for i, dim in enumerate(ids))
        values[str(flat_index)] = value

    path.write_text(
        json.dumps(
            {
                "version": "2.0",
                "class": "dataset",
                "id": ids,
                "size": sizes,
                "dimension": dim_payload,
                "value": values,
            }
        )
    )


def test_load_eurostat_capital_depreciation_data_maps_r_s(tmp_path):
    eurostat_path = tmp_path
    income_ids = ["freq", "unit", "nace_r2", "na_item", "geo", "time"]
    income_dims = {
        "freq": ["A"],
        "unit": ["CP_MNAC"],
        "nace_r2": ["A", "R", "S"],
        "na_item": ["P1", "P2", "B1G", "D1", "D29X39", "P51C", "B2A3N"],
        "geo": ["FR"],
        "time": ["2014"],
    }
    income_obs = {
        ("A", "P1"): 100.0,
        ("A", "P2"): 55.0,
        ("A", "B1G"): 45.0,
        ("A", "D1"): 25.0,
        ("A", "D29X39"): 5.0,
        ("A", "P51C"): 10.0,
        ("A", "B2A3N"): 10.0,
        ("R", "P1"): 30.0,
        ("R", "P2"): 18.0,
        ("R", "B1G"): 12.0,
        ("R", "D1"): 6.0,
        ("R", "D29X39"): 1.0,
        ("R", "P51C"): 3.0,
        ("R", "B2A3N"): 2.0,
        ("S", "P1"): 70.0,
        ("S", "P2"): 35.0,
        ("S", "B1G"): 35.0,
        ("S", "D1"): 14.0,
        ("S", "D29X39"): 4.0,
        ("S", "P51C"): 7.0,
        ("S", "B2A3N"): 14.0,
    }
    _write_jsonstat(
        eurostat_path / "nama_10_a64.json",
        income_ids,
        income_dims,
        [
            (
                {
                    "freq": "A",
                    "unit": "CP_MNAC",
                    "nace_r2": sector,
                    "na_item": item,
                    "geo": "FR",
                    "time": "2014",
                },
                value,
            )
            for (sector, item), value in income_obs.items()
        ],
    )

    stock_ids = ["freq", "unit", "nace_r2", "asset10", "geo", "time"]
    stock_dims = {
        "freq": ["A"],
        "unit": ["CRC_MNAC"],
        "nace_r2": ["A", "R", "S"],
        "asset10": ["N11N"],
        "geo": ["FR"],
        "time": ["2014"],
    }
    stock_obs = {"A": 200.0, "R": 50.0, "S": 150.0}
    _write_jsonstat(
        eurostat_path / "nama_10_nfa_st.json",
        stock_ids,
        stock_dims,
        [
            (
                {
                    "freq": "A",
                    "unit": "CRC_MNAC",
                    "nace_r2": sector,
                    "asset10": "N11N",
                    "geo": "FR",
                    "time": "2014",
                },
                value,
            )
            for sector, value in stock_obs.items()
        ],
    )

    result = load_eurostat_capital_depreciation_data(eurostat_path, "FRA", 2014, ["A", "R_S"])

    assert result.loc["A", "Capital Depreciation Rate"] == 0.05
    assert result.loc["A", "CFC Output Ratio"] == 0.1
    assert result.loc["A", "Labour Output Ratio"] == 0.25
    assert result.loc["A", "Production Tax Output Ratio"] == 0.05
    assert result.loc["R_S", "Consumption of Fixed Capital"] == 10.0
    assert result.loc["R_S", "Output"] == 100.0
    assert result.loc["R_S", "Labour Compensation"] == 20.0
    assert result.loc["R_S", "Production Taxes Less Subsidies"] == 5.0
    assert result.loc["R_S", "Net Operating Surplus"] == 16.0
    assert result.loc["R_S", "Gross Operating Surplus"] == 26.0
    assert result.loc["R_S", "Net Fixed Capital Stock"] == 200.0
    assert result.loc["R_S", "Capital Depreciation Rate"] == 0.05


def test_build_cfc_output_replacement_matrix_column_sums():
    matrix = build_cfc_output_replacement_matrix(
        industries=["A", "B"],
        capital_good_mix=np.array([2.0, 1.0]),
        cfc_output_ratios=np.array([0.12, 0.24]),
        yearly_factor=4.0,
    )

    assert np.allclose(matrix.sum(axis=0).values, np.array([0.12, 0.24]))

"""Opt-in calibration support of the HFCS consumption source, built from the survey records in the model."""

import hashlib
import os
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from macro_data.configuration import DataConfiguration, split_country_configs
from macro_data.processing.synthetic_population import hfcs_consumption_support as support
from macro_data.processing.synthetic_population.balance_sheet_scaling import (
    HOUSEHOLD_STOCK_COLUMNS,
    scale_household_balance_sheet,
    scale_housing_market,
)
from macro_data.processing.synthetic_population.hfcs_synthetic_population import (
    eligible_hfcs_consumption_households,
    sample_households,
)
from macro_data.readers.population_data.hfcs_reader import HFCSReader
from macro_data.util.consumption_initialisation import winsorise_hfcs_consumption_ratios
from macro_data.util.consumption_source_config import consumption_data_signature
from macro_data.util.hfcs_consumption_source import (
    HFCSConsumptionRegistration,
    HFCSConsumptionSource,
    read_hfcs_consumption_source,
    source_ids_fingerprint,
)
from tests.test_macro_data.unit.test_balance_sheet_scaling import TAXES, housing_table, set_hfcs_gross, toy_population
from tests.test_macro_data.unit.test_consumption_source_config import configured  # noqa: F401

TAX, SIW, SHARE = 0.2, 0.1, 0.15


def survey():
    """Four survey households in the reader's column names. Monthly spending, payments; annual income and stocks."""
    zero = [0.0] * 4
    return pd.DataFrame(
        {
            "HFCS Source ID": ["a", "b", "c", "d"],
            "Weight": [1.0, 1.0, 1.0, 1.0],
            "HFCS Gross Employee Income": [30000.0, 20000.0, 40000.0, 25000.0],
            "HFCS Gross Self-Employment Income": zero,
            "HFCS Gross Social Transfers": zero,
            "Income from Pensions": zero,
            "Rental Income from Real Estate": zero,
            "Income from Financial Assets": zero,
            "Interest Payments": [600.0, 0.0, 1000.0, 100.0],
            "Amount spent on Consumption of Goods and Services": [1500.0, 1000.0, 2000.0, 1200.0],
            "Rent Paid": [0.0, 400.0, 0.0, 0.0],
            "Rent Paid for Partially Owned Dwelling": zero,
            "Value of the Main Residence": [100000.0, 0.0, 200000.0, 80000.0],
            "Wealth in Deposits": [3000.0, 500.0, 1000.0, 200.0],
            "Money Market Funds": [500.0, 0.0, 0.0, 0.0],
            "Mutual Funds": [2000.0, 0.0, 0.0, 0.0],
            "Bonds": zero,
            "Shares": [1000.0, 0.0, 300.0, 0.0],
            "Managed Accounts": zero,
            "Money owed to Households": zero,
            "Other Assets": zero,
            "Voluntary Pension": zero,
            "Outstanding Balance of Mortgage Debt": [40000.0, 0.0, 90000.0, 4000.0],
            "Outstanding Balance of Non-Mortgage Debt": [2000.0, 0.0, 0.0, 0.0],
            "Mortgage Payments (flow)": [300.0, np.nan, 600.0, np.nan],
            "HMR Mortgage Payments (flow)": [np.nan, np.nan, np.nan, 5000.0],
            "Monthly Leasing Payments": zero,
            "Additional Consumer Loan Payments": [50.0, 0.0, 0.0, 0.0],
        }
    )


def test_support_variables_follow_the_calibration_definitions():
    h = survey()
    out = support.build_support_variables(
        h, income_tax=TAX, employee_social_insurance_tax=SIW, imputed_rent_share=SHARE
    )
    # Imputed rent: rate x DA1110 with total s/(1-s) x sum_w 12 (G + R), all four households observed.
    nd_less = 12 * (h["Amount spent on Consumption of Goods and Services"] + h["Rent Paid"])
    rate = SHARE / (1 - SHARE) * nd_less.sum() / h["Value of the Main Residence"].sum()
    imputed = rate * h["Value of the Main Residence"]
    income = h["HFCS Gross Employee Income"] * (1 - SIW) * (1 - TAX) - h["Interest Payments"] + imputed
    np.testing.assert_allclose(out["income"], income)
    lfa = np.array([3500.0, 500.0, 1000.0, 200.0])  # deposits + money-market funds
    ifa = np.array([2500.0, 0.0, 300.0, 0.0])  # mutual funds + shares, funds moved to lfa
    debt = np.array([42000.0, 0.0, 90000.0, 4000.0])  # DL1100 + DL1200
    # Debt service: mortgage 12 x payment; household d keyed principal as payment (5000 >= 4000) and is repaired to
    # balance x weighted median annual-payment/balance of plausible payers (0.08 for a: 0.09, c: 0.08).
    mortgage_service = np.array([3600.0, 0.0, 7200.0, 0.08 * 4000.0])
    nonmortgage_service = np.array([600.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(out["ifa_y"], ifa / income)
    np.testing.assert_allclose(out["ha_y"], h["Value of the Main Residence"] / income)
    np.testing.assert_allclose(out["nla_y"], (lfa - debt) / income)
    np.testing.assert_allclose(out["r_y"], (lfa - mortgage_service - nonmortgage_service) / income)
    assert out["valid"].tolist() == [True, True, True, True]


def test_validity_excludes_nonfinite_ratios_bounds_and_debt_service_above_balance():
    h = survey()
    h.loc[1, "HFCS Gross Employee Income"] = 0.0  # no income: ratios not finite
    h.loc[2, "Value of the Main Residence"] = 5e7  # ha_y far above the hard bound 30
    h.loc[3, "HMR Mortgage Payments (flow)"] = 1.0  # plausible payment, but then 12 x 300 exceeds a tiny balance
    h.loc[3, "Outstanding Balance of Mortgage Debt"] = 100.0
    h.loc[3, "Mortgage Payments (flow)"] = 50.0
    out = support.build_support_variables(
        h, income_tax=TAX, employee_social_insurance_tax=SIW, imputed_rent_share=SHARE
    )
    assert out["valid"].tolist()[1:] == [False, False, False]
    assert out["valid"].tolist()[0]


def test_missing_support_columns_are_an_error():
    with pytest.raises(ValueError, match="support_inputs=True"):
        support.build_support_variables(
            survey().drop(columns="Interest Payments"),
            income_tax=TAX,
            employee_social_insurance_tax=SIW,
            imputed_rent_share=SHARE,
        )


def test_imputed_rent_share_must_be_a_share():
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="share"):
            support.hfcs_imputed_rent(survey(), bad)


def calibration_weighted_quantile(x, w, q):
    """Verbatim rule of consumption_calibration.weighted_quantile."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    order = np.argsort(x)
    x_sorted, w_sorted = x[order], w[order]
    cum_w = np.cumsum(w_sorted) - 0.5 * w_sorted
    cum_w /= w_sorted.sum()
    return float(np.interp(q, cum_w, x_sorted))


def test_weighted_percentile_matches_the_calibration_rule_and_the_existing_winsoriser():
    rng = np.random.default_rng(3)
    x = rng.lognormal(size=500)
    w = rng.uniform(0.2, 9.0, size=500)
    for q in (0.01, 0.5, 0.99):
        assert support.weighted_quantile(x, w, q) == calibration_weighted_quantile(x, w, q)
    wins = winsorise_hfcs_consumption_ratios(c_y=x, survey_weights=w, hard_valid=np.ones(500, dtype=bool))
    assert (support.weighted_quantile(x, w, 0.01), support.weighted_quantile(x, w, 0.99)) == pytest.approx(
        (wins.lower, wins.upper)
    )


def make_source(n=20, quantiles=(0.1, 0.9), **fields):
    rng = np.random.default_rng(7)
    c_y = rng.uniform(0.2, 1.5, n)
    weights = rng.uniform(1.0, 5.0, n)
    valid = np.ones(n, dtype=bool)
    valid[0] = False
    wins = winsorise_hfcs_consumption_ratios(
        c_y=np.where(valid, c_y, np.nan), survey_weights=weights, hard_valid=valid, lower_quantile=quantiles[0],
        upper_quantile=quantiles[1],
    )  # fmt: skip
    ids = np.array([str(i) for i in range(n)], dtype=object)
    return rng, build_source(ids, wins, weights, c_y, quantiles, **fields)


def build_source(ids, wins, weights, c_y, quantiles, **fields):
    return HFCSConsumptionSource(
        ids, 2014, wins, "0" * 64, weights, c_y=c_y, threshold_pool=np.ones(len(ids), dtype=bool), quantiles=quantiles,
        **fields,
    )  # fmt: skip


def test_support_is_the_joint_weighted_trim_on_the_hard_valid_pool():
    rng, source = make_source()
    n = len(source.source_ids)
    variables = pd.DataFrame(
        {name: rng.normal(size=n) for name in ("ifa_y", "nla_y", "ha_y", "r_y")} | {"valid": np.ones(n, dtype=bool)}
    )
    variables.loc[1, "valid"] = False  # model-side validity removes it from the pool too
    result = support.calibration_support(source, variables, lower_quantile=0.1, upper_quantile=0.9)
    pool = source.winsorised.valid & variables["valid"].to_numpy()
    np.testing.assert_array_equal(result.hard_valid, pool)
    expected = pool.copy()
    for name, x in {
        "c_y": source.c_y,
        **{k: variables[k].to_numpy() for k in ("ifa_y", "nla_y", "ha_y", "r_y")},
    }.items():
        lower = calibration_weighted_quantile(x[pool], source.survey_weights[pool], 0.1)
        upper = calibration_weighted_quantile(x[pool], source.survey_weights[pool], 0.9)
        assert result.thresholds[name] == (lower, upper)
        expected &= (x >= lower) & (x <= upper)
    np.testing.assert_array_equal(result.support, expected)
    assert 0 < result.support.sum() < pool.sum()
    assert not result.support[0] and not result.support[1]


def test_thresholds_use_only_the_registered_threshold_pool():
    rng, source = make_source()
    n = len(source.source_ids)
    variables = pd.DataFrame(
        {name: rng.normal(size=n) for name in ("ifa_y", "nla_y", "ha_y", "r_y")} | {"valid": np.ones(n, dtype=bool)}
    )
    pool = np.arange(n) < n // 2
    restricted = replace(source, threshold_pool=pool)
    full = support.calibration_support(source, variables, lower_quantile=0.1, upper_quantile=0.9)
    part = support.calibration_support(restricted, variables, lower_quantile=0.1, upper_quantile=0.9)
    assert part.thresholds["ifa_y"] != full.thresholds["ifa_y"]
    mask = restricted.winsorised.valid & pool
    x = variables["ifa_y"].to_numpy()
    assert part.thresholds["ifa_y"][0] == calibration_weighted_quantile(x[mask], source.survey_weights[mask], 0.1)


@pytest.fixture
def sampling_inputs(tmp_path):
    ids = np.array(["001", "002", "003", "004"])
    frame = pd.DataFrame(
        dict(
            source_id=ids,
            wave=2014,
            c_y=[0.1, 0.9, 0.5, np.nan],
            survey_weight=[1.0, 3.0, 2.0, 100.0],
            hard_valid=[True, True, True, False],
            threshold_pool=True,
        )
    )
    frame.attrs.update(
        consumption_concept="total",
        income_concept="net_of_interest_plus_imputed_rent",
        income_scale_var="disp_geom_avg_income",
        preparation_stage="before_percentile_trimming",
        preparation_manifest_sha256="a" * 64,
    )
    path = tmp_path / "private.parquet"
    frame.to_parquet(path, index=False)
    source = read_hfcs_consumption_source(
        path,
        wave=2014,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        registration=HFCSConsumptionRegistration(4, source_ids_fingerprint(ids), "a" * 64),
    )
    households = pd.DataFrame({"HFCS Source ID": ids, "Weight": [1.0, 3.0, 2.0, 100.0]}, index=[10, 20, 30, 40])
    individuals = pd.DataFrame({"Corresponding Household ID": [10, 20, 30, 40], "Age": [40, 50, 30, 60]})
    return households, individuals, source


def test_default_sampler_eligibility_is_unchanged(sampling_inputs, monkeypatch):
    households, individuals, source = sampling_inputs
    assert source.support is None and source.stock_anchor_factors is None

    def choice(indices, n, p, replace):
        np.testing.assert_array_equal(indices, [10, 20, 30])
        np.testing.assert_allclose(p, np.array([1.0, 3.0, 2.0]) / 6.0)
        return np.full(n, 20)

    monkeypatch.setattr(np.random, "choice", choice)
    sample_households(households, individuals, 5, consumption_source=source)


def test_support_sampler_never_draws_outside_the_support(sampling_inputs):
    households, individuals, source = sampling_inputs
    restricted = replace(source, support=np.array([True, False, True, False]))
    eligible = eligible_hfcs_consumption_households(households, individuals, restricted)
    assert eligible["HFCS Source ID"].tolist() == ["001", "003"]
    np.random.seed(1)
    drawn, _ = sample_households(households, individuals, 400, consumption_source=restricted)
    assert set(drawn["HFCS Source ID"]) == {"001", "003"}
    np.random.seed(1)
    default, _ = sample_households(households, individuals, 400, consumption_source=source)
    assert "002" in set(default["HFCS Source ID"])
    share = (drawn["HFCS Source ID"] == "003").mean()
    assert abs(share - 2.0 / 3.0) < 0.1  # survey weights still drive the draws


def test_empty_support_is_an_error(sampling_inputs):
    households, individuals, source = sampling_inputs
    with pytest.raises(ValueError, match="No hard-valid"):
        eligible_hfcs_consumption_households(households, individuals, replace(source, support=np.zeros(4, dtype=bool)))


def test_anchor_factor_restores_the_default_aggregate_stock_to_income_exactly():
    rng = np.random.default_rng(11)
    n = 200
    w = rng.uniform(0.5, 6.0, n)
    y = rng.lognormal(10, 0.5, n)
    stocks = {"A": rng.lognormal(9, 1.0, n), "B": rng.uniform(0, 5e4, n) * (rng.uniform(size=n) < 0.3)}
    valid = rng.uniform(size=n) < 0.9
    in_support = valid & (rng.uniform(size=n) < 0.8)
    factors = support.stock_anchor_factors(stocks, y, w, valid, in_support)
    for name, s in stocks.items():
        default_ratio = (w * s)[valid].sum() / (w * y)[valid].sum()
        support_ratio = (w * s)[in_support].sum() / (w * y)[in_support].sum()
        assert factors[name] * support_ratio == pytest.approx(default_ratio, rel=1e-13)
    assert support.stock_anchor_factors(stocks, y, w, valid, valid) == pytest.approx({"A": 1.0, "B": 1.0})


def test_anchor_factor_handles_stocks_the_support_does_not_hold():
    w, y = np.ones(4), np.array([1.0, 1.0, 1.0, 1.0])
    valid = np.ones(4, dtype=bool)
    in_support = np.array([True, True, False, False])
    assert support.stock_anchor_factors({"none": np.zeros(4)}, y, w, valid, in_support) == {"none": 1.0}
    with pytest.raises(ValueError, match="holds none"):
        support.stock_anchor_factors({"tail": np.array([0.0, 0.0, 5.0, 0.0])}, y, w, valid, in_support)


def test_survey_stock_definitions_cover_exactly_the_scaled_stocks():
    h = survey().assign(
        **{
            "Outstanding Balance of HMR Mortgages": 1.0,
            "Outstanding Balance of Mortgages on other Properties": 2.0,
            "Outstanding Balance of other Non-Mortgage Loans": 3.0,
            "Value of other Properties": 4.0,
            "Value of Household Vehicles": 5.0,
            "Value of Household Valuables": 6.0,
            "Value of Self-Employment Businesses": 7.0,
            "Mutual Funds": [-5.0, 0.0, 0.0, 0.0],  # negative components are clipped to zero, as in the model's IFA
        }
    )
    stocks = support.survey_stocks(h)
    assert tuple(stocks) == HOUSEHOLD_STOCK_COLUMNS
    assert stocks["Wealth Other Real Assets"].tolist() == [18.0] * 4
    assert stocks["Wealth in Other Financial Assets"].tolist() == [1000.0, 0.0, 300.0, 0.0]


def test_scaling_applies_anchor_after_k_i_and_leaves_default_untouched():
    kwargs = dict(concept="disp_geom_avg_income", financial_asset_yield=0.04, **TAXES)

    def run(anchor):
        pop = toy_population()
        set_hfcs_gross(pop, [2.0, 3.0, 5.0])
        housing, diagnostics = scale_household_balance_sheet(
            pop, housing_table(), stock_anchor_factors=anchor, **kwargs
        )
        return pop.household_data, housing, diagnostics

    base, base_housing, base_diag = run(None)
    ones, ones_housing, _ = run({name: 1.0 for name in HOUSEHOLD_STOCK_COLUMNS})
    pd.testing.assert_frame_equal(base, ones)
    pd.testing.assert_frame_equal(base_housing, ones_housing)
    assert "stock_anchor_factors" not in base_diag

    anchor = {name: 1.0 + 0.1 * i for i, name in enumerate(HOUSEHOLD_STOCK_COLUMNS)}
    scaled, housing, diagnostics = run(anchor)
    for name in HOUSEHOLD_STOCK_COLUMNS:
        np.testing.assert_allclose(scaled[name], base[name] * anchor[name], rtol=1e-13)
    np.testing.assert_allclose(
        scaled["Income from Financial Assets"],
        base["Income from Financial Assets"] * anchor["Wealth in Other Financial Assets"],
    )
    np.testing.assert_allclose(
        housing["Value"].to_numpy()[:2],
        base_housing["Value"].to_numpy()[:2]
        * [anchor["Value of the Main Residence"], anchor["Value of other Properties"]],
    )
    assert housing["Value"].iloc[2] == base_housing["Value"].iloc[2]  # social housing is not anchored
    assert diagnostics["stock_anchor_factors"] == anchor


def test_housing_anchor_defaults_to_no_change():
    table, factors = housing_table(), np.array([2.0, 3.0])
    pd.testing.assert_frame_equal(scale_housing_market(table, factors), scale_housing_market(table, factors, None))


def test_reader_adds_support_columns_only_on_request(tmp_path):
    path = tmp_path / "D1.csv"
    path.write_text("ID,SA0010,SA0100,HW0010,DI1412,DL2100,DL2110,HC0110,HD1320c,DA2101\n10,001,FR,1,5,6,7,8,9,10\n")
    rates = SimpleNamespace(from_eur_to_lcu=lambda **_: 2.0)
    default = HFCSReader.read_csv(path, "FRA", "FR", 2014, rates)
    assert set(default.columns) == {"HFCS Source ID", "Weight", "Wealth in Deposits"}
    extra = HFCSReader.read_csv(path, "FRA", "FR", 2014, rates, support_inputs=True)
    assert extra.loc[10, ["Interest Payments", "Mortgage Payments (flow)", "Money Market Funds"]].tolist() == [
        10,
        12,
        18,
    ]
    assert extra.loc[10, ["HMR Mortgage Payments (flow)", "Monthly Leasing Payments"]].tolist() == [14, 16]
    assert set(default.columns) < set(extra.columns)
    assert set(extra.columns) - set(default.columns) == set(support.SUPPORT_INPUT_COLUMNS)


def test_attach_support_matches_records_by_id_and_sets_anchor(monkeypatch):
    h = survey()
    h["Outstanding Balance of HMR Mortgages"] = 0.0
    h["Outstanding Balance of Mortgages on other Properties"] = 0.0
    h["Outstanding Balance of other Non-Mortgage Loans"] = 0.0
    h["Value of other Properties"] = 0.0
    h["Value of Household Vehicles"] = 0.0
    h["Value of Household Valuables"] = 0.0
    h["Value of Self-Employment Businesses"] = 0.0
    ids = np.array(["d", "c", "b", "a"], dtype=object)  # source order differs from survey order
    c_y = np.array([0.8, 0.8, 0.8, 0.8])
    weights = h.set_index("HFCS Source ID").loc[ids, "Weight"].to_numpy()
    wins = winsorise_hfcs_consumption_ratios(c_y=c_y, survey_weights=weights, hard_valid=np.ones(4, dtype=bool))
    source = build_source(ids, wins, weights, c_y, (0.0, 1.0))
    out = support.attach_calibration_support(
        source,
        h.sample(frac=1.0, random_state=0),
        income_tax=TAX,
        employee_social_insurance_tax=SIW,
        imputed_rent_share=SHARE,
        scaling_income_concept="disp_geom_avg_income",
    )
    # Quantiles 0 and 1 cut nothing: the whole valid pool is the support and every k_S is 1.
    assert out.support.tolist() == [True, True, True, True]
    assert out.stock_anchor_factors == pytest.approx({name: 1.0 for name in HOUSEHOLD_STOCK_COLUMNS})
    assert set(out.support_thresholds) == set(support.TRIM_VARIABLES)
    assert source.support is None  # the registered source object is not mutated
    with pytest.raises(ValueError, match="differ from the registered source ID set"):
        support.attach_calibration_support(
            source,
            h.iloc[:3],
            income_tax=TAX,
            employee_social_insurance_tax=SIW,
            imputed_rent_share=SHARE,
            scaling_income_concept="disp_geom_avg_income",
        )


def test_imputed_rent_share_comes_from_the_country_input_output_table():
    icio = SimpleNamespace(
        imputed_rents={"FRA": 15.0}, actual_rents={"FRA": 5.0}, get_hh_consumption=lambda c: np.array([40.0, 40.0])
    )
    rates = SimpleNamespace(from_usd_to_lcu=lambda country, year: 0.5)
    assert support.imputed_rent_share_of_household_consumption(icio, rates, "FRA", 2014) == pytest.approx(15 / 60)
    with pytest.raises(ValueError, match="No imputed rent"):
        support.imputed_rent_share_of_household_consumption(
            SimpleNamespace(imputed_rents={}, actual_rents={}), rates, "FRA", 2014
        )


def test_config_default_and_validation(configured):  # noqa: F811
    config, _ = configured
    country = config.country_configs["FRA"]
    assert country.consumption_source_support == "hard_valid"
    country.household_consumption_initialisation.balance_sheet_scaling = True
    fields = country.model_dump()
    ok = type(country).model_validate({**fields, "consumption_source_support": "calibration_support"})
    assert ok.consumption_source_support == "calibration_support"
    with pytest.raises(ValidationError):
        type(country).model_validate({**fields, "consumption_source_support": "trimmed"})
    scaling_off = deepcopy(fields)
    scaling_off["household_consumption_initialisation"]["balance_sheet_scaling"] = False
    with pytest.raises(ValidationError, match="balance_sheet_scaling"):
        type(country).model_validate({**scaling_off, "consumption_source_support": "calibration_support"})
    assert type(country).model_validate(scaling_off).consumption_source_support == "hard_valid"
    legacy = deepcopy(fields)
    legacy["household_consumption_initialisation"] = {"method": "legacy"}
    with pytest.raises(ValidationError, match="hfcs_cy"):
        type(country).model_validate({**legacy, "consumption_source_support": "calibration_support"})
    assert type(country).model_validate({**legacy}).consumption_source_support == "hard_valid"


def test_data_override_selects_support_and_its_own_cache(configured):  # noqa: F811
    run_model_path = str(Path(__file__).resolve().parents[3] / "run_model")
    if run_model_path not in sys.path:
        sys.path.insert(0, run_model_path)
    from src.notebook_workflow import apply_data_config_overrides, data_cache_name_for

    config, root = configured
    config.country_configs["FRA"].household_consumption_initialisation.balance_sheet_scaling = True
    overridden = apply_data_config_overrides(config, {"consumption_source_support": "calibration_support"})
    assert overridden.country_configs["FRA"].consumption_source_support == "calibration_support"
    assert data_cache_name_for("data.pkl", {"consumption_source_support": "calibration_support"}) != "data.pkl"
    assert data_cache_name_for("data.pkl", None) == "data.pkl"


def test_signature_is_unchanged_by_default_and_changes_when_support_is_on(configured):  # noqa: F811
    config, root = configured
    default = consumption_data_signature(config, root)
    assert "consumption_source_support" not in default["configuration"]["country_configs"]["FRA"]
    assert "calibration_support" not in default
    # Same signature as a configuration written before the field existed (key absent in the file dict).
    legacy_style = deepcopy(config.model_dump(mode="json"))
    legacy_style["country_configs"]["FRA"].pop("consumption_source_support")
    assert consumption_data_signature(legacy_style, root) == default
    on = deepcopy(config.model_dump(mode="json"))
    on["country_configs"]["FRA"]["consumption_source_support"] = "calibration_support"
    signature = consumption_data_signature(on, root)
    assert signature != default
    assert signature["calibration_support"]["countries"] == ["FRA"]


def test_config_in_repo_stays_on_the_default():
    path = Path(__file__).resolve().parents[3] / "run_model/config/data_config_FRA.yaml"
    cfg = yaml.safe_load(path.read_text())
    assert cfg["country_configs"]["FRA"].get("consumption_source_support", "hard_valid") == "hard_valid"
    DataConfiguration(**{**cfg, "country_configs": split_country_configs(cfg["country_configs"])})


RAW_DATA = Path(os.environ.get("INET_RAW_DATA", Path(__file__).resolve().parents[3] / "run_model/data/raw_data"))
FRA_THRESHOLDS = {  # calibration, FRA 2014, p1/p99 on the hard-valid pool (consumption_calibration, fast notebook)
    "c_y": (0.2560, 2.1227),
    "ifa_y": (0.0, 8.178),
    "nla_y": (-7.116, 3.920),
    "ha_y": (0.0, 16.140),
    "r_y": (-0.4215, 4.221),
}


@pytest.mark.skipif(
    not (RAW_DATA / "hfcs/consumption_initialisation/FRA_2014.parquet").exists(),
    reason="private HFCS microdata and registered source are not available",
)
def test_fra_2014_support_reproduces_the_calibration_trim_within_input_differences():
    """Needs private data (set INET_RAW_DATA). Rates and share are the FRA 2014 values of the model's own readers."""
    import json

    source_dir = RAW_DATA / "hfcs/consumption_initialisation"
    registration = HFCSConsumptionRegistration(**json.loads((source_dir / "FRA_2014.registration.json").read_text()))
    source_path = source_dir / "FRA_2014.parquet"
    source = read_hfcs_consumption_source(
        source_path,
        wave=2014,
        expected_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        registration=registration,
    )
    rates = SimpleNamespace(from_eur_to_lcu=lambda **_: 1.0)
    hfcs = RAW_DATA / "hfcs/2014"
    households = HFCSReader.read_csv(hfcs / "H1.csv", "FRA", "FR", 2014, rates, support_inputs=True)
    derived = HFCSReader.read_csv(hfcs / "D1.csv", "FRA", "FR", 2014, rates, support_inputs=True)
    households = households.join(derived.drop(columns=["Weight", "HFCS Source ID"], errors="ignore"))
    ids = pd.Index(households["HFCS Source ID"].to_numpy())
    positions = ids.get_indexer(source.source_ids)
    variables = support.build_support_variables(
        households, income_tax=0.2191, employee_social_insurance_tax=0.1405, imputed_rent_share=0.1623
    ).iloc[positions]
    result = support.calibration_support(source, variables, lower_quantile=0.01, upper_quantile=0.99)
    weights = source.survey_weights
    share = weights[result.support].sum() / weights[source.winsorised.valid].sum()
    assert share == pytest.approx(0.9354, abs=0.005)  # calibration: 9,679 households, 93.54% of hard-valid weight
    assert abs(result.support.sum() - 9679) / 9679 < 0.01
    for name, (lower, upper) in FRA_THRESHOLDS.items():
        assert result.thresholds[name][0] == pytest.approx(lower, abs=0.05 * max(1.0, abs(lower)))
        assert result.thresholds[name][1] == pytest.approx(upper, rel=0.10)

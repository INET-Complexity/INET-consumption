import pytest

from macro_data.readers.income_belief_priors import (
    lookup_income_belief_priors,
    load_income_belief_priors_table,
)


def test__load_income_belief_priors_table_has_twelve_rows():
    table = load_income_belief_priors_table()
    assert len(table) == 12
    assert set(table.index.get_level_values("employment_status")) == {"employed", "retired", "unemployed"}
    assert set(table.index.get_level_values("age_group")) == {"18-34", "35-49", "50-70", "71-85"}


def test__unemployed_71_85_falls_back_to_unemployed_50_70():
    table = load_income_belief_priors_table()
    fallback = table.loc[("unemployed", "50-70")]
    target = table.loc[("unemployed", "71-85")]
    for column in ["mean_income_uncertainty_var_prior", "sigma2_xi", "sigma2_v"]:
        assert target[column] == fallback[column]
    # mean_income_exp_dev is NOT replaced by the fallback
    assert target["mean_income_exp_dev"] != fallback["mean_income_exp_dev"]


def test__lookup_income_belief_priors_returns_expected_values():
    table = load_income_belief_priors_table()
    priors = lookup_income_belief_priors("employed", "35-49", table)
    assert priors["rho"] == 0.9518878627264576
    assert priors["mu_1_0"] == pytest.approx(0.01008815296089871)
    assert priors["P_1_0"] == pytest.approx(0.003828699143248362)
    assert priors["sigma2_xi"] == pytest.approx(0.0111010082630461)
    assert priors["sigma2_v"] == pytest.approx(0.0003454310251577192)

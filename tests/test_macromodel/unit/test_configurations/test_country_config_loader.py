from pathlib import Path

import pytest
import yaml

from macromodel.configurations import load_country_configuration

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _minimal_country_payload(wealth_parameters):
    return {
        "FRA": {
            "households": {
                "functions": {
                    "wealth": {
                        "name": "PaperAssetReturnWealthSetter",
                        "path_name": "wealth",
                        "parameters": wealth_parameters,
                    }
                }
            }
        }
    }


def test__load_country_configuration_resolves_paper_parameter_ref(tmp_path):
    paper_params = {
        "stage_1": {
            "asset_returns": {
                "paper_stochastic_v1": {
                    "other_real_assets_depreciation_rate": 0.05,
                    "mu_eq": 0.0029,
                    "mu_bond": 0.0081,
                    "sigma_eq": 0.0935,
                    "sigma_bond": 0.0316,
                    "rho": -0.2585,
                    "equity_weight": 0.5,
                    "draw_scope": "country_period",
                }
            }
        }
    }
    (tmp_path / "paper.yaml").write_text(yaml.safe_dump(paper_params))
    country_path = tmp_path / "country.yaml"
    country_path.write_text(
        yaml.safe_dump(
            _minimal_country_payload(
                {
                    "paper_parameter_file": "paper.yaml",
                    "paper_parameter_ref": "stage_1.asset_returns.paper_stochastic_v1",
                }
            )
        )
    )

    config = load_country_configuration(country_path, country_iso3="FRA")

    params = config.households.functions.wealth.parameters
    assert "paper_parameter_file" not in params
    assert "paper_parameter_ref" not in params
    assert params["mu_eq"] == 0.0029
    assert params["equity_weight"] == 0.5


def test__load_country_configuration_unwraps_single_country_file_without_iso3(tmp_path):
    paper_params = {
        "stage_1": {
            "asset_returns": {
                "paper_stochastic_v1": {
                    "other_real_assets_depreciation_rate": 0.05,
                    "mu_eq": 0.0029,
                    "mu_bond": 0.0081,
                    "sigma_eq": 0.0935,
                    "sigma_bond": 0.0316,
                    "rho": -0.2585,
                    "equity_weight": 0.5,
                    "draw_scope": "country_period",
                }
            }
        }
    }
    (tmp_path / "paper.yaml").write_text(yaml.safe_dump(paper_params))
    country_path = tmp_path / "country.yaml"
    country_path.write_text(
        yaml.safe_dump(
            _minimal_country_payload(
                {
                    "paper_parameter_file": "paper.yaml",
                    "paper_parameter_ref": "stage_1.asset_returns.paper_stochastic_v1",
                }
            )
        )
    )

    config = load_country_configuration(country_path)

    assert config.households.functions.wealth.name == "PaperAssetReturnWealthSetter"
    assert config.households.functions.wealth.parameters["mu_eq"] == 0.0029


def test__load_country_configuration_merges_embedded_parameter_overrides(tmp_path):
    paper_params = {
        "stage_1": {
            "asset_returns": {
                "paper_stochastic_v1": {
                    "other_real_assets_depreciation_rate": 0.05,
                    "mu_eq": 0.0029,
                    "mu_bond": 0.0081,
                    "sigma_eq": 0.0935,
                    "sigma_bond": 0.0316,
                    "rho": -0.2585,
                    "equity_weight": 0.5,
                    "draw_scope": "country_period",
                }
            }
        }
    }
    (tmp_path / "paper.yaml").write_text(yaml.safe_dump(paper_params))
    country_path = tmp_path / "country.yaml"
    country_path.write_text(
        yaml.safe_dump(
            _minimal_country_payload(
                {
                    "paper_parameter_file": "paper.yaml",
                    "paper_parameter_ref": "stage_1.asset_returns.paper_stochastic_v1",
                    "equity_weight": 0.25,
                }
            )
        )
    )

    config = load_country_configuration(country_path, country_iso3="FRA")

    params = config.households.functions.wealth.parameters
    assert "paper_parameter_file" not in params
    assert "paper_parameter_ref" not in params
    assert params["mu_eq"] == 0.0029
    assert params["equity_weight"] == 0.25
    assert params["draw_scope"] == "country_period"


def test__load_country_configuration_merges_multiple_paper_parameter_refs(tmp_path):
    paper_params = {
        "asset_returns": {
            "paper_stochastic_v1": {
                "mu_eq": 0.0029,
                "equity_weight": 0.5,
            }
        },
        "portfolio_composition": {
            "uses_portfolio_choice": False,
            "phi_1": 5.0,
            "lambda_kappa": 0.1,
            "fixed_cost_share": 0.0,
        },
    }
    (tmp_path / "paper.yaml").write_text(yaml.safe_dump(paper_params))
    country_path = tmp_path / "country.yaml"
    country_path.write_text(
        yaml.safe_dump(
            _minimal_country_payload(
                {
                    "paper_parameter_refs": [
                        {
                            "paper_parameter_file": "paper.yaml",
                            "paper_parameter_ref": "asset_returns.paper_stochastic_v1",
                        },
                        {
                            "paper_parameter_file": "paper.yaml",
                            "paper_parameter_ref": "portfolio_composition",
                        },
                    ],
                }
            )
        )
    )

    config = load_country_configuration(country_path, country_iso3="FRA")

    params = config.households.functions.wealth.parameters
    assert "paper_parameter_refs" not in params
    assert "paper_parameter_file" not in params
    assert "paper_parameter_ref" not in params
    assert params["mu_eq"] == 0.0029
    assert params["equity_weight"] == 0.5
    assert params["uses_portfolio_choice"] is False
    assert params["phi_1"] == 5.0
    assert params["lambda_kappa"] == 0.1
    assert params["fixed_cost_share"] == 0.0


def test__load_country_configuration_rejects_duplicate_keys_across_paper_parameter_refs(tmp_path):
    paper_params = {
        "asset_returns": {
            "paper_stochastic_v1": {
                "mu_eq": 0.0029,
                "fixed_cost_share": 0.2,
            }
        },
        "portfolio_composition": {
            "uses_portfolio_choice": False,
            "fixed_cost_share": 0.0,
        },
    }
    (tmp_path / "paper.yaml").write_text(yaml.safe_dump(paper_params))
    country_path = tmp_path / "country.yaml"
    country_path.write_text(
        yaml.safe_dump(
            _minimal_country_payload(
                {
                    "paper_parameter_refs": [
                        {
                            "paper_parameter_file": "paper.yaml",
                            "paper_parameter_ref": "asset_returns.paper_stochastic_v1",
                        },
                        {
                            "paper_parameter_file": "paper.yaml",
                            "paper_parameter_ref": "portfolio_composition",
                        },
                    ],
                }
            )
        )
    )

    with pytest.raises(ValueError, match="Duplicate paper parameter keys.*fixed_cost_share"):
        load_country_configuration(country_path, country_iso3="FRA")


def test__load_country_configuration_rejects_duplicate_local_keys_with_paper_parameter_refs(tmp_path):
    paper_params = {
        "asset_returns": {
            "paper_stochastic_v1": {
                "mu_eq": 0.0029,
            }
        },
    }
    (tmp_path / "paper.yaml").write_text(yaml.safe_dump(paper_params))
    country_path = tmp_path / "country.yaml"
    country_path.write_text(
        yaml.safe_dump(
            _minimal_country_payload(
                {
                    "paper_parameter_refs": [
                        {
                            "paper_parameter_file": "paper.yaml",
                            "paper_parameter_ref": "asset_returns.paper_stochastic_v1",
                        }
                    ],
                    "mu_eq": 0.003,
                }
            )
        )
    )

    with pytest.raises(ValueError, match="Duplicate paper parameter keys.*mu_eq"):
        load_country_configuration(country_path, country_iso3="FRA")


def test__load_country_configuration_rejects_literal_mapping_inside_paper_parameter_refs(tmp_path):
    country_path = tmp_path / "country.yaml"
    country_path.write_text(
        yaml.safe_dump(
            _minimal_country_payload(
                {
                    "paper_parameter_refs": [{"mu_eq": 0.0029}],
                }
            )
        )
    )

    with pytest.raises(ValueError, match="entries must define only paper_parameter_file/paper_parameter_ref"):
        load_country_configuration(country_path, country_iso3="FRA")


def test__load_country_configuration_rejects_missing_file_key_inside_paper_parameter_refs(tmp_path):
    country_path = tmp_path / "country.yaml"
    country_path.write_text(
        yaml.safe_dump(
            _minimal_country_payload(
                {
                    "paper_parameter_refs": [
                        {
                            "paper_parameter_ref": "asset_returns.paper_stochastic_v1",
                        }
                    ],
                }
            )
        )
    )

    with pytest.raises(ValueError, match="entries must define only paper_parameter_file/paper_parameter_ref"):
        load_country_configuration(country_path, country_iso3="FRA")


def test__load_country_configuration_rejects_extra_keys_inside_paper_parameter_refs(tmp_path):
    country_path = tmp_path / "country.yaml"
    country_path.write_text(
        yaml.safe_dump(
            _minimal_country_payload(
                {
                    "paper_parameter_refs": [
                        {
                            "paper_parameter_file": "paper.yaml",
                            "paper_parameter_ref": "asset_returns.paper_stochastic_v1",
                            "mu_eq": 0.0029,
                        }
                    ],
                }
            )
        )
    )

    with pytest.raises(ValueError, match="entries must define only paper_parameter_file/paper_parameter_ref"):
        load_country_configuration(country_path, country_iso3="FRA")


def test__load_country_configuration_resolves_nested_paper_parameter_ref_in_parameter_section(tmp_path):
    paper_params = {
        "desired_consumption": {
            "credit_augmented_v1": {
                "income_belief_learning_horizon": {
                    "paper_parameter_file": "paper.yaml",
                    "paper_parameter_ref": "income_belief_learning.permanent_income_log_ratio",
                },
            }
        },
        "income_belief_learning": {"permanent_income_log_ratio": {"delta": 0.95, "S": 40}},
    }
    (tmp_path / "paper.yaml").write_text(yaml.safe_dump(paper_params))
    country_path = tmp_path / "country.yaml"
    country_path.write_text(
        yaml.safe_dump(
            {
                "FRA": {
                    "households": {
                        "functions": {
                            "consumption": {
                                "name": "CreditAugmentedConsumption",
                                "path_name": "consumption",
                                "parameters": {
                                    "paper_parameter_file": "paper.yaml",
                                    "paper_parameter_ref": "desired_consumption.credit_augmented_v1",
                                },
                            }
                        }
                    }
                }
            }
        )
    )

    config = load_country_configuration(country_path, country_iso3="FRA")

    params = config.households.functions.consumption.parameters
    assert "paper_parameter_file" not in params
    assert "paper_parameter_ref" not in params
    assert params["income_belief_learning_horizon"] == {"delta": 0.95, "S": 40}


def test__load_country_configuration_resolves_real_fra_cacf_parameters():
    config = load_country_configuration(_REPO_ROOT / "run_model/config/country_config_FRA.yaml", country_iso3="FRA")
    with (_REPO_ROOT / "run_model/config/consumption_paper_parameters.yaml").open() as f:
        paper_parameters = yaml.safe_load(f)
    # FRA resolves credit_augmented_v2 (continuous calibration estimated on HFCS
    # France, 2026-08). credit_augmented_v1 is retained in the paper-parameter file
    # for reproducibility of pre-v2 baselines but is no longer the FRA reference.
    v2 = paper_parameters["desired_consumption"]["credit_augmented_v2"]
    expected_income_growth_propensity = v2["income_growth_propensity"]

    params = config.households.functions.consumption.parameters
    assert "paper_parameter_file" not in params
    assert "paper_parameter_ref" not in params
    # Inert under the continuous mapping, but still resolved; see the yaml comment.
    assert params["permanent_income_propensity"] == 0.55
    assert params["income_growth_propensity"] == expected_income_growth_propensity
    assert params["interest_rate_cashflow_propensity"] == -0.003
    assert params["uncertainty_propensity"] == -0.005
    assert params["partial_adjustment_speed"] == 0.56
    assert params["long_run_mpc_lower_bound"] == 0.0
    assert params["long_run_mpc_upper_bound"] == 2.0
    assert params["consumption_smoothing_fraction"] == 0.0
    assert params["consumption_smoothing_window"] == 12
    assert params["elasticity_of_substitution"] == 1.0
    assert params["minimum_consumption_fraction"] == 1.0
    assert params["income_belief_learning_horizon"] == {"delta": 0.95, "S": 40}
    # v2-specific: the estimated intercept (v1 used 0.08 plus a downstream recentring),
    # the idiosyncratic term, and the calibration's own smoothed income denominator.
    assert params["long_run_intercept"] == -0.4638
    assert params["idiosyncratic_sd"] == 0.3308
    assert params["idiosyncratic_persistence"] == "fixed_effect"
    assert params["income_denominator"] == "geometric_average"
    assert params["income_denominator_window"] == 20
    assert params["uses_continuous_wealth_calibration"] is True
    # Decoupled logistics: the estimator rejects v1's shared-slope restriction by two
    # orders of magnitude, so these must resolve as four distinct numbers.
    calibration = params["continuous_wealth_calibration"]
    assert calibration["index_construction"] == "raw_ratio"
    assert calibration["alpha_2_steepness"] == 2.012
    assert calibration["gamma_1_steepness"] == 148.413
    assert calibration["alpha_2_midpoint"] == 1.000
    assert calibration["gamma_1_midpoint"] == 0.0532
    assert calibration["weight_net_liquid_assets"] == 0.6719


def test__load_country_configuration_resolves_real_fra_wealth_parameter_refs():
    config = load_country_configuration(_REPO_ROOT / "run_model/config/country_config_FRA.yaml", country_iso3="FRA")

    params = config.households.functions.wealth.parameters
    assert "paper_parameter_refs" not in params
    assert "paper_parameter_file" not in params
    assert "paper_parameter_ref" not in params
    assert params["mu_eq"] == 0.0029
    assert params["equity_weight"] == 0.5
    assert params["uses_portfolio_choice"] is True
    assert params["target_share_source"] == "scalar"
    assert params["default_target_illiquid_share"] == 0.65
    assert params["phi_1"] == 5.0
    assert params["lambda_kappa"] == 0.1
    assert params["fixed_cost_share"] == 0.001
    assert params["frm_coefficients_path"] == "data/raw_data/portfolio/FR_portfolio_frm_coefficients.json"


def test__load_country_configuration_resolves_real_fra_tax_overrides():
    config = load_country_configuration(_REPO_ROOT / "run_model/config/country_config_FRA.yaml", country_iso3="FRA")

    overrides = config.central_government.tax_overrides
    assert overrides.employer_social_insurance_rate == pytest.approx(0.30552085148943465)
    assert overrides.value_added_tax_rate == pytest.approx(0.13)
    assert overrides.household_investment_vat_rate == pytest.approx(0.13)
    assert overrides.household_capital_formation_rate == 0.0
    assert overrides.firm_capital_formation_rate == pytest.approx(0.24856698371134814)
    assert overrides.other_product_production_tax_rate is None


def test__load_country_configuration_rejects_missing_paper_parameter_ref(tmp_path):
    (tmp_path / "paper.yaml").write_text(yaml.safe_dump({"stage_1": {}}))
    country_path = tmp_path / "country.yaml"
    country_path.write_text(
        yaml.safe_dump(
            _minimal_country_payload(
                {
                    "paper_parameter_file": "paper.yaml",
                    "paper_parameter_ref": "stage_1.asset_returns.paper_stochastic_v1",
                }
            )
        )
    )

    with pytest.raises(ValueError, match="Missing paper parameter section"):
        load_country_configuration(country_path, country_iso3="FRA")


def test__load_country_configuration_rejects_non_mapping_paper_parameter_ref(tmp_path):
    (tmp_path / "paper.yaml").write_text(yaml.safe_dump({"stage_1": {"scalar": 1.0}}))
    country_path = tmp_path / "country.yaml"
    country_path.write_text(
        yaml.safe_dump(
            _minimal_country_payload(
                {
                    "paper_parameter_file": "paper.yaml",
                    "paper_parameter_ref": "stage_1.scalar",
                }
            )
        )
    )

    with pytest.raises(ValueError, match="Paper parameter section must be a mapping"):
        load_country_configuration(country_path, country_iso3="FRA")


def test__load_country_configuration_rejects_circular_paper_parameter_ref(tmp_path):
    paper_params = {
        "stage_1": {
            "asset_returns": {
                "paper_stochastic_v1": {
                    "paper_parameter_file": "paper.yaml",
                    "paper_parameter_ref": "stage_1.asset_returns.paper_stochastic_v1",
                }
            }
        }
    }
    (tmp_path / "paper.yaml").write_text(yaml.safe_dump(paper_params))
    country_path = tmp_path / "country.yaml"
    country_path.write_text(
        yaml.safe_dump(
            _minimal_country_payload(
                {
                    "paper_parameter_file": "paper.yaml",
                    "paper_parameter_ref": "stage_1.asset_returns.paper_stochastic_v1",
                }
            )
        )
    )

    with pytest.raises(ValueError, match="Circular paper parameter reference detected"):
        load_country_configuration(country_path, country_iso3="FRA")

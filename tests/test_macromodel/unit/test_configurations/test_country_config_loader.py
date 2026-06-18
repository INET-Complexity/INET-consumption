from pathlib import Path

import pytest
import yaml

from macromodel.configurations import load_country_configuration


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
                "fixed_cost_share": 0.2,
            }
        },
        "portfolio_composition": {
            "uses_portfolio_choice": False,
            "phi_1": 5.0,
            "lambda_kappa": 0.1,
            "fixed_cost_share": None,
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
                    "fixed_cost_share": 0.0,
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
    config = load_country_configuration(Path("run_model/config/country_config_FRA.yaml"), country_iso3="FRA")

    params = config.households.functions.consumption.parameters
    assert "paper_parameter_file" not in params
    assert "paper_parameter_ref" not in params
    assert params["permanent_income_propensity"] == 0.55
    assert params["income_growth_propensity"] == 0.45
    assert params["interest_rate_cashflow_propensity"] == -0.003
    assert params["uncertainty_propensity"] == -0.005
    assert params["partial_adjustment_speed"] == 0.56
    assert params["consumption_smoothing_fraction"] == 0.0
    assert params["consumption_smoothing_window"] == 12
    assert params["elasticity_of_substitution"] == 1.0
    assert params["minimum_consumption_fraction"] == 1.0
    assert params["income_belief_learning_horizon"] == {"delta": 0.95, "S": 40}


def test__load_country_configuration_resolves_real_fra_wealth_parameter_refs():
    config = load_country_configuration(Path("run_model/config/country_config_FRA.yaml"), country_iso3="FRA")

    params = config.households.functions.wealth.parameters
    assert "paper_parameter_refs" not in params
    assert "paper_parameter_file" not in params
    assert "paper_parameter_ref" not in params
    assert params["mu_eq"] == 0.0029
    assert params["equity_weight"] == 0.5
    assert params["uses_portfolio_choice"] is False
    assert params["target_share_source"] == "scalar"
    assert params["default_target_illiquid_share"] == 0.65
    assert params["phi_1"] == 5.0
    assert params["lambda_kappa"] == 0.1
    assert params["fixed_cost_share"] == 0.0
    assert params["frm_coefficients_path"] == "portfolio/FR_portfolio_frm_coefficients.json"


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

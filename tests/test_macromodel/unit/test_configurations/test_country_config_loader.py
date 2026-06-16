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


def test__load_country_configuration_rejects_embedded_parameter_overrides(tmp_path):
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

    with pytest.raises(ValueError, match="cannot define sibling parameter keys: equity_weight"):
        load_country_configuration(country_path, country_iso3="FRA")


def test__load_country_configuration_resolves_nested_paper_parameter_ref_in_parameter_section(tmp_path):
    paper_params = {
        "stage_2": {
            "desired_consumption": {
                "credit_augmented_v1": {
                    "income_belief_learning_horizon": {
                        "paper_parameter_file": "paper.yaml",
                        "paper_parameter_ref": "stage_3.income_belief_learning.permanent_income_log_ratio",
                    },
                }
            }
        },
        "stage_3": {"income_belief_learning": {"permanent_income_log_ratio": {"delta": 0.95, "S": 40}}},
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
                                    "paper_parameter_ref": "stage_2.desired_consumption.credit_augmented_v1",
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
    assert params["income_belief_learning_horizon"] == {"delta": 0.95, "S": 40}


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

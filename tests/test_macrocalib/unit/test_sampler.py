from types import SimpleNamespace

import yaml

import macrocalib.sampler.sampler as sampler_module
from macrocalib.sampler import PriorSampler, Sampler


def test_sampler(sampler: Sampler, prior_sampler: PriorSampler):
    sampler.base_configuration.t_max = 5
    sampler.base_configuration.seed = None

    n_runs = 5

    data = sampler.parallel_run(n_runs, prior_sampler)

    # Each element in data is a dict with 'simulations' key, which is a list of length n_runs
    total_runs = sum(len(core_result["simulations"]) for core_result in data)
    assert total_runs == n_runs * sampler.n_cores


def test_sampler_default_loads_keyed_country_config(tmp_path, monkeypatch):
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
    country_path = tmp_path / "country_config_FRA.yaml"
    country_path.write_text(
        yaml.safe_dump(
            {
                "FRA": {
                    "households": {
                        "functions": {
                            "wealth": {
                                "name": "PaperAssetReturnWealthSetter",
                                "path_name": "wealth",
                                "parameters": {
                                    "paper_parameter_file": "paper.yaml",
                                    "paper_parameter_ref": "stage_1.asset_returns.paper_stochastic_v1",
                                },
                            }
                        }
                    }
                }
            }
        )
    )
    data = SimpleNamespace(n_industries=3, synthetic_countries={"FRA": object()})
    monkeypatch.setattr(
        sampler_module.DataWrapper,
        "init_from_pickle",
        staticmethod(lambda path: data),
    )

    sampler = Sampler.default(
        configuration_updater=lambda configuration, parameters: configuration,
        observer=lambda simulation: [],
        pickle_path=tmp_path / "data.pkl",
        country_conf_path=country_path,
        countries=["FRA"],
        n_cores=1,
    )

    wealth_config = sampler.base_configuration.country_configurations["FRA"].households.functions.wealth
    assert wealth_config.name == "PaperAssetReturnWealthSetter"
    assert wealth_config.parameters["mu_eq"] == 0.0029

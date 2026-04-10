from pathlib import Path

from macro_data import DataWrapper
from macro_data.configuration_utils import default_data_configuration
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation
from config import Config
from src.visual_helpers import plot_output


def main() -> None:
    cfg = Config.from_env()

    data_config = default_data_configuration(countries=[cfg.country_iso3])

    creator = DataWrapper.from_config(
        configuration=data_config,
        raw_data_path=cfg.raw_data_path,
        single_hfcs_survey=True,
    )

    output_dir = cfg.output_path
    data_pkl_path = output_dir / "data.pkl"
    output_dir.mkdir(parents=True, exist_ok=True)

    creator.save(str(data_pkl_path))
    data = DataWrapper.init_from_pickle(str(data_pkl_path))

    configuration = SimulationConfiguration(
        country_configurations={cfg.country_iso3: CountryConfiguration()},
        t_max=200,
        seed=47,
    )

    model = Simulation.from_datawrapper(
        datawrapper=data,
        simulation_configuration=configuration,
    )

    model.run()
    model.save(save_dir=output_dir, file_name="multi_country_simulation.h5")

    df = model.shallow_df_dict()[cfg.country_iso3]
    plot_output(df=df, no_rows=4, no_cols=4, country_code=cfg.country_iso3)


if __name__ == "__main__":
    main()

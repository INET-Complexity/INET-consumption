from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import environs


@dataclass
class Config:
    data_dir: Path = Path("data/hfcs")
    api_dotenv_path: Path = Path("/Users/andone/.ssh/macro_ts_api_keys.env")
    model_dotenv_path: Path = Path(__file__).with_name(".env")
    country_name: str = "France"
    country_iso3: str = "FRA"
    country_iso2: str = "FR"
    seed: int = 45
    t_max: int = 100
    raw_data_path: Path = Path("data/raw_data")
    output_path: Path = Path("data/output_data")
    fred_api_key: str | None = None
    webstat_api_key: str | None = None

    @classmethod
    def from_env(
        cls,
        api_dotenv_path: str | Path = "/Users/andone/.ssh/macro_ts_api_keys.env",
        model_dotenv_path: str | Path | None = None,
        **overrides,
    ) -> "Config":
        run_model_dir = Path(__file__).resolve().parents[1]
        model_env_path = Path(model_dotenv_path) if model_dotenv_path is not None else Path(__file__).with_name(".env")

        model_env = environs.Env()
        model_env.read_env(str(model_env_path), recurse=False)

        api_env = environs.Env()
        api_env.read_env(str(api_dotenv_path), recurse=False)

        raw_data_path = Path(model_env("RAW_DATA_PATH"))
        output_path = Path(model_env("OUTPUT_DATA_PATH"))

        if not raw_data_path.is_absolute():
            raw_data_path = run_model_dir / raw_data_path
        if not output_path.is_absolute():
            output_path = run_model_dir / output_path

        return cls(
            api_dotenv_path=Path(api_dotenv_path),
            model_dotenv_path=model_env_path,
            raw_data_path=raw_data_path,
            output_path=output_path,
            fred_api_key=api_env("FRED_API_KEY", default=None),
            webstat_api_key=api_env("WEBSTAT_API_KEY", default=None),
            **overrides,
        )

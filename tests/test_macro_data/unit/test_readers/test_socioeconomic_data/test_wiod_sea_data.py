import pandas as pd
import pytest

from macro_data.readers.socioeconomic_data.wiod_sea_data import WIODSEAReader


class DummyExchangeRates:
    def exchange_rates_dict(self, year):
        return {"FRA": 1.0}


def test_wiod_sea_rescale_preserves_industry_alignment(tmp_path):
    csv_path = tmp_path / "wiod_sea.csv"
    rows = []
    for code, va, comp, cap, stock in [
        ("A01", 100.0, 20.0, 80.0, 1000.0),
        ("B", 50.0, 40.0, 10.0, 500.0),
    ]:
        rows.extend(
            [
                {"country": "FRA", "variable": "VA", "description": code, "code": code, "2014": va},
                {"country": "FRA", "variable": "COMP", "description": code, "code": code, "2014": comp},
                {"country": "FRA", "variable": "CAP", "description": code, "code": code, "2014": cap},
                {"country": "FRA", "variable": "K", "description": code, "code": code, "2014": stock},
            ]
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    reader = WIODSEAReader.agg_from_csv(
        path=csv_path,
        aggregation_type="Aggregate",
        year=2014,
        country_names=["FRA"],
        industries=["A", "B"],
        exchange_rates=DummyExchangeRates(),
        value_added_dict={
            "FRA": pd.Series(
                [2000.0, 1000.0],
                index=pd.Index(["B", "A"], name="Industry"),
            )
        },
    )

    result = reader.df.loc["FRA"]

    assert result.loc["A", "Value Added"] == pytest.approx(1000.0)
    assert result.loc["B", "Value Added"] == pytest.approx(2000.0)
    assert result.loc["A", "Labour Compensation"] == pytest.approx(200.0)
    assert result.loc["B", "Labour Compensation"] == pytest.approx(1600.0)

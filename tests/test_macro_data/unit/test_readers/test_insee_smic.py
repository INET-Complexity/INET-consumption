import pandas as pd

from macro_data.readers.insee_smic import load_insee_smic_annual_table


def test__load_insee_smic_annual_table_uses_december_rows(tmp_path):
    csv_path = tmp_path / "smic.csv"
    csv_path.write_text(
        '"Label";"Net monthly amount";"Codes"\n'
        '"idBank";"000879878";""\n'
        '"Period";"";""\n'
        '"2021-11";"1200.0";"A"\n'
        '"2021-12";"1210.0";"A"\n'
        '"2022-01";"1220.0";"A"\n'
        '"2022-12";"1300.0";"A"\n',
        encoding="utf-8",
    )

    annual = load_insee_smic_annual_table(csv_path)

    expected = pd.Series(
        [1210.0, 1300.0],
        index=pd.Index([2021, 2022], name="year"),
        name="net_monthly_smic",
    )
    pd.testing.assert_series_equal(annual, expected)

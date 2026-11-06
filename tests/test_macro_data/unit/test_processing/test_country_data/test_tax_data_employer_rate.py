from unittest.mock import MagicMock

import pytest

from macro_data.processing.country_data.tax_data import TaxData


@pytest.mark.parametrize("configured, expected", [(None, 0.4186), (0.3055, 0.3055)])
def test__employer_rate_uses_configured_value_or_reader(configured, expected):
    readers = MagicMock()
    readers.oecd_econ.read_tau_sif.return_value = 0.4186

    tax_data = TaxData.from_readers(readers, "FRA", 2014, employer_social_insurance_rate=configured)

    assert tax_data.employer_social_insurance_tax == expected

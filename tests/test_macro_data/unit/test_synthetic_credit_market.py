from types import SimpleNamespace

import numpy as np
import pandas as pd

from macro_data.processing.synthetic_credit_market.synthetic_credit_market import SyntheticCreditMarket


def test_zero_firm_debt_zeros_firm_loan_principals():
    bank_data = pd.DataFrame(
        {
            "Long-Term Interest Rates on Firm Loans": [0.05, 0.06],
            "Interest Rates on Household Consumption Loans": [0.10, 0.12],
            "Interest Rates on Mortgages": [0.03, 0.035],
        }
    )

    firm_data = pd.DataFrame(
        {
            "Debt": [100.0, 200.0, 300.0],
            "Corresponding Bank ID": [0, 1, 0],
        }
    )

    household_data = pd.DataFrame(
        {
            "Outstanding Balance of other Non-Mortgage Loans": [10.0, 0.0, 5.0, 20.0],
            "Outstanding Balance of HMR Mortgages": [50.0, 100.0, 0.0, 25.0],
            "Corresponding Bank ID": [0, 0, 1, 1],
        }
    )

    banks = SimpleNamespace(bank_data=bank_data)
    firms = SimpleNamespace(country_name="FRA", year=2014, firm_data=firm_data)
    population = SimpleNamespace(household_data=household_data)

    credit_market_zero = SyntheticCreditMarket.create_from_agents(
        firms=firms,
        population=population,
        banks=banks,
        zero_firm_debt=True,
        firm_loan_maturity=60,
        hh_consumption_maturity=12,
        mortgage_maturity=120,
    )
    assert np.isclose(credit_market_zero.longterm_loans.principal.sum(), 0.0)
    assert np.isclose(credit_market_zero.longterm_loans.interest.sum(), 0.0)
    assert np.isclose(credit_market_zero.longterm_loans.installments.sum(), 0.0)

    credit_market_nonzero = SyntheticCreditMarket.create_from_agents(
        firms=firms,
        population=population,
        banks=banks,
        zero_firm_debt=False,
        firm_loan_maturity=60,
        hh_consumption_maturity=12,
        mortgage_maturity=120,
    )
    assert credit_market_nonzero.longterm_loans.principal.sum() > 0.0
    assert credit_market_nonzero.longterm_loans.installments.sum() > 0.0

from macromodel.configurations.bank_configuration import BankParameters


def test_bank_parameters_accept_capital_stock_collateral_ratio_name():
    params = BankParameters(firm_loans_capital_stock_collateral_ratio=0.8)

    assert params.firm_loans_capital_stock_collateral_ratio == 0.8


def test_bank_parameters_accept_legacy_debt_to_equity_ratio_name():
    params = BankParameters(firm_loans_debt_to_equity_ratio=0.7)

    assert params.firm_loans_capital_stock_collateral_ratio == 0.7


def test_bank_parameters_default_firm_loan_restriction_switches_preserve_legacy_behaviour():
    params = BankParameters()

    assert params.enable_firm_loans_return_on_assets_restriction is True
    assert params.enable_firm_loans_return_on_equity_restriction is True
    assert params.enable_firm_loans_dscr_restriction is False
    assert params.firm_loans_min_dscr == 1.25
    assert params.firm_loans_cfads_window == 4
    assert params.firm_loans_cfads_haircut == 1.0
    assert params.firm_loans_dscr_underwriting_rate_mode == "max_bank_rate"

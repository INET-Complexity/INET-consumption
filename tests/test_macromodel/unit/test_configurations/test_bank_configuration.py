from macromodel.configurations.bank_configuration import BankParameters


def test_bank_parameters_accept_capital_stock_collateral_ratio_name():
    params = BankParameters(firm_loans_capital_stock_collateral_ratio=0.8)

    assert params.firm_loans_capital_stock_collateral_ratio == 0.8


def test_bank_parameters_accept_legacy_debt_to_equity_ratio_name():
    params = BankParameters(firm_loans_debt_to_equity_ratio=0.7)

    assert params.firm_loans_capital_stock_collateral_ratio == 0.7

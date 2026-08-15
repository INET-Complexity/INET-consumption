"""Independent verification of the CACF lag row indexing (pre-existing, not GH #120).

Structural, not numerical: it captures the exact array the consumption rule is
handed and compares it against identifiable rows of the timeseries, so the
result cannot be confounded by VAT factors, deflators or floors.
"""

import numpy as np


def _stub_downstream(test_country, monkeypatch, n_households):
    """Neutralise the Stage 4/5 plumbing that runs after the target is set."""
    monkeypatch.setattr(
        test_country.credit_market,
        "compute_scheduled_mortgage_payments_by_household",
        lambda: np.zeros(n_households),
    )
    monkeypatch.setattr(
        test_country.credit_market,
        "compute_scheduled_consumption_loan_payments_by_household",
        lambda: np.zeros(n_households),
    )
    test_country.banks.ts.override_current(
        "interest_rates_on_household_consumption_loans",
        np.asarray([0.10, 0.14]),
    )
    monkeypatch.setattr(
        test_country.households,
        "build_stage4_borrow_vs_sell_inputs",
        lambda **_kwargs: {
            "delta_tilde": np.zeros(n_households),
            "opening_tfa_scale": np.full(n_households, 100.0),
            "post_return_ifa": np.full(n_households, 25.0),
            "r_kappa": np.full(n_households, 0.02),
        },
    )


class TestLagRowIndexing:
    def test__prev_consumption_resolves_two_periods_back_at_planning_time(self, test_country, monkeypatch):
        households = test_country.households
        n_households = households.ts.current("n_households")
        n_industries = len(test_country.firms.ts.current("price"))
        _stub_downstream(test_country, monkeypatch, n_households)

        # Give each period's realised consumption an identifiable level. After
        # this the series holds rows for periods 0..2, so a call planning
        # period 3 should see period 2 as its previous consumption.
        households.ts.override_current("consumption", np.full(n_households, 100.0))  # period 0
        households.ts.consumption.append(np.full(n_households, 200.0))  # period 1
        households.ts.consumption.append(np.full(n_households, 300.0))  # period 2
        rows_at_call = len(households.ts.consumption)

        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return np.zeros((n_households, n_industries))

        monkeypatch.setattr(households.functions["consumption"], "compute_target_consumption", _capture)
        test_country._set_household_target_demand(replace_current=False)

        assert rows_at_call == 3, rows_at_call
        current_row = float(np.asarray(households.ts.current("consumption"))[0])
        prev_row = float(np.asarray(households.ts.prev("consumption"))[0])
        passed = float(np.asarray(captured["lagged_consumption"])[0])

        # Realised series are appended only after goods-market clearing, so at
        # planning time no row exists for the period being planned. `current` is
        # therefore already the previous period (300.0) and `prev` is the one
        # before that (200.0).
        assert current_row == 300.0, current_row
        assert prev_row == 200.0, prev_row

        # The legacy wiring hands the rule `prev`, i.e. two periods back, where
        # the intended concept is one period back.
        assert passed == 200.0, passed
        assert passed != current_row

    def test__persisted_budget_lag_resolves_one_period_back(self, test_country, monkeypatch):
        """The GH #120 replacement resolves to t-1 on the planning pass."""
        households = test_country.households
        n_households = households.ts.current("n_households")
        n_industries = len(test_country.firms.ts.current("price"))
        _stub_downstream(test_country, monkeypatch, n_households)

        households.ts.override_current("cacf_real_consumption_budget", np.full(n_households, 100.0))  # period 0
        households.ts.cacf_real_consumption_budget.append(np.full(n_households, 200.0))  # period 1
        households.ts.cacf_real_consumption_budget.append(np.full(n_households, 300.0))  # period 2

        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return np.zeros((n_households, n_industries))

        monkeypatch.setattr(households.functions["consumption"], "compute_target_consumption", _capture)
        test_country._set_household_target_demand(replace_current=False)

        passed = float(np.asarray(captured["lagged_real_consumption_budget"])[0])
        assert passed == 300.0, passed

"""Serial diagnostic replay of housing matches with native accounting.

Replay proposals before credit clearing. Certify completed transactions after
settlement; report cash-flow differences while financing responds endogenously.
This is not permission to overwrite booked payments or bank balance sheets.
"""
from contextlib import contextmanager
from copy import deepcopy
import threading

import numpy as np
import pandas as pd

_LOCK = threading.Lock()


class HousingReplayMismatch(RuntimeError):
    """Baseline housing flows cannot be reproduced under native settlement."""


class HousingReplayTape:
    def __init__(self, shock_period, horizon_periods=1, *, require_financing_match=False):
        self.require_financing_match = bool(require_financing_match)
        self.start = int(shock_period)
        self.stop = self.start + int(horizon_periods)
        self.proposals = {}
        self.completed = {}
        self.cash = {}
        self.audit = []
        self.failure_detail = []
        self.financing_cache = {}

    def active(self, period):
        return self.start <= period < self.stop

    def capture_proposals(self, seed, period, country):
        self.proposals[seed, period] = deepcopy(country.housing_market.states['current_sales'])

    def replay_proposals(self, seed, period, country):
        key = seed, period
        if key not in self.completed:
            raise HousingReplayMismatch(f'Missing completed baseline housing transactions: {key}')
        country.housing_market.states['current_sales'] = deepcopy(self.completed[key])

    def record_or_verify(self, seed, period, country, *, treatment):
        key = seed, period
        hh = country.households
        cash = np.stack([hh.current_net_housing_cash_need(), hh.current_housing_lfa_draw()])
        if not np.isfinite(cash).all():
            raise HousingReplayMismatch(f'Nonfinite housing cash: {key}')
        completed = country.housing_market.states['current_sales'].copy(deep=True)
        financial = {}
        for name in ('income', 'financing_income', 'expected_income', 'liquid_financial_assets', 'received_mortgages', 'target_consumption'):
            if name in getattr(getattr(hh, 'ts', None), 'dicts', {}):
                financial[name] = np.asarray(hh.ts.current(name), dtype=float).copy()
        if not treatment:
            self.financing_cache[key] = financial
            self.completed[key] = completed
            self.cash[key] = cash.copy()
            return
        try:
            pd.testing.assert_frame_equal(completed, self.completed[key], check_exact=True)
        except AssertionError as error:
            compared = self.completed[key].merge(completed, on=list(completed.columns), how='left', indicator=True)
            missing = compared.loc[compared['_merge'].eq('left_only')].drop(columns='_merge')
            for _, transaction in missing.iterrows():
                buyer = int(transaction['buyer_id'])
                detail = dict(seed=seed, period=period, transaction=transaction.to_dict(), buyer_id=buyer)
                for name, values in financial.items():
                    detail[name + '_treatment'] = np.asarray(values[buyer]).tolist()
                    detail[name + '_baseline'] = np.asarray(self.financing_cache[key][name][buyer]).tolist()
                self.failure_detail.append(detail)
            raise HousingReplayMismatch(f'Completed housing transactions differ: {key}') from error
        difference = np.abs(cash - self.cash[key])
        matched = np.allclose(cash, self.cash[key], rtol=1e-10, atol=1e-6)
        self.audit.append(dict(seed=seed, period=period, matched=True, financing_matched=bool(matched),
                               max_cash_difference=float(difference.max()),
                               max_net_housing_cash_difference=float(difference[0].max()),
                               max_housing_lfa_draw_difference=float(difference[1].max())))
        if self.require_financing_match and not matched:
            raise HousingReplayMismatch(
                f'Housing cash differs at {key}: maximum {difference.max():.6g}; '
                'baseline financing cannot be imposed by replaying matches alone.'
            )


@contextmanager
def housing_replay(tape):
    """Install serial experiment instrumentation and restore it on every exit."""
    from src.experiments import mpc
    from macromodel.country.country import Country
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('Housing replay is already active in this process.')
    original_run = mpc._run_one
    original_clear = Country.clear_housing_market
    original_process = Country.process_housing_market_clearing
    context = {}

    def run_one(**kwargs):
        context.update(seed=int(kwargs['seed']), treatment=bool(kwargs['shock']),
                       code=kwargs['country_code'], period=-1)
        try:
            return original_run(**kwargs)
        finally:
            context.clear()

    def clear(country):
        original_clear(country)
        if not context or country.country_name != context['code']:
            return
        context['period'] += 1
        period = context['period']
        if tape.active(period):
            if context['treatment']:
                tape.replay_proposals(context['seed'], period, country)
            else:
                tape.capture_proposals(context['seed'], period, country)

    def process(country):
        original_process(country)
        if context and country.country_name == context['code'] and tape.active(context['period']):
            tape.record_or_verify(context['seed'], context['period'], country,
                                  treatment=context['treatment'])

    mpc._run_one = run_one
    Country.clear_housing_market = clear
    Country.process_housing_market_clearing = process
    try:
        yield tape
    finally:
        mpc._run_one = original_run
        Country.clear_housing_market = original_clear
        Country.process_housing_market_clearing = original_process
        _LOCK.release()


def run_housing_replay_experiment(*, control_periods=1, **kwargs):
    """Run MPCs with a verified housing replay in the specified impact window.

    One period controls the impact experiment. Claim a cumulative controlled
    response only when all periods of that cumulative window are controlled.
    """
    from pathlib import Path
    from src.experiments.mpc import run_mpc_experiment
    if kwargs.get('n_jobs', 1) != 1:
        raise ValueError('Housing replay requires n_jobs=1 in a dedicated process.')
    if not 1 <= control_periods <= kwargs['horizon_periods']:
        raise ValueError('control_periods must be within the MPC measurement window.')
    tape = HousingReplayTape(kwargs['shock_period'], control_periods)
    try:
        with housing_replay(tape):
            outputs = run_mpc_experiment(**kwargs)
    finally:
        out = Path(kwargs['output_dir'])
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tape.audit).to_csv(out / 'housing_replay_audit.csv', index=False)
        import json
        (out / 'housing_replay_failures.json').write_text(json.dumps(tape.failure_detail, indent=2, default=str))
    expected = len(kwargs['seeds']) * control_periods
    if len(tape.audit) != expected:
        raise HousingReplayMismatch(f'Expected {expected} verified periods, got {len(tape.audit)}')
    return outputs

# Household Property Income Power Warning

This note records the runtime warning seen during the FRA property-market
calibration investigation:

```text
macromodel/agents/households/func/property.py:276: RuntimeWarning: invalid value encountered in power
  * household_income[ind_dec] ** self.maximum_price_income_exponent
```

## Symptom

During `DefaultHouseholdDemandForProperty.compute_demand`, NumPy emits
`RuntimeWarning: invalid value encountered in power` while computing the maximum
price households are willing to pay for property.

The warning occurs before housing-market clearing when households in social
housing, renters who do not stay, or owners who do not stay evaluate whether to
buy or rent.

## Immediate Trigger

The FRA configuration uses fractional income exponents:

```yaml
maximum_price_income_exponent: 0.789
maximum_rent_income_exponent: 0.3464
```

With a fractional exponent, a negative real-valued income base is invalid in
NumPy's real domain:

```python
household_income[ind_dec] ** self.maximum_price_income_exponent
```

If any deciding household has negative `expected_income`, the expression returns
`nan` for that household and emits the warning.

The rent-willingness calculation has the same failure mode for deciding renters:

```python
household_income[ind_deciding_to_rent] ** self.maximum_rent_income_exponent
```

## Resolution

`household-property-income-power-warning` chooses a narrow affordability-rule
fix: keep upstream `expected_income` unchanged, but use a local non-negative
income base before applying the fractional buy/rent affordability exponents.
Non-positive expected income therefore implies zero buy/rent willingness in
these power terms rather than `nan`.

The same branch also ports the stable `scipy.special.expit` buy-probability
sigmoid from `INET-Complexity/macro-main#90`.

Regression coverage:

```text
uv run pytest tests/test_macromodel/unit/test_agents/test_households/func/test_property_probability.py
uv run ruff check macromodel/agents/households/func/property.py tests/test_macromodel/unit/test_agents/test_households/func/test_property_probability.py
```

## Data Path

The property demand function receives:

```python
household_income=self.ts.current("expected_income")
```

from `Households.prepare_housing_market_clearing`.

`expected_income` is computed as:

```python
expected_income_employee
+ expected_income_social_transfers
+ income_rental
+ expected_income_financial_assets
```

One possible route to negative household expected income is negative
`wealth_other_financial_assets`, because expected financial-asset income is
currently linear in current other financial assets:

```python
return income_coefficient * current_other_financial_assets
```

However, the saved FRA diagnostic that motivated this note did **not** show
negative `wealth_other_financial_assets` or negative
`expected_income_financial_assets`. It showed negative `expected_income` for
household `5169` in periods `35-50`, driven by large negative `income_rental`.
That source is tracked separately as a `compute_rental_income` diagnostic bug.

## Expected Diagnostic Checks

When reproducing the warning, inspect the deciding households immediately before
the property affordability calculation:

```python
ind_dec = (
    household_residence_tenure_status == -1
    | (
        (household_residence_tenure_status == 0)
        & (np.random.random(n_households) > probability_stay_in_rented_property)
    )
    | (
        (household_residence_tenure_status == 1)
        & (np.random.random(n_households) > probability_stay_in_owned_property)
    )
)

np.nanmin(household_income)
np.sum(household_income < 0)
np.sum(household_income[ind_dec] < 0)
```

Also inspect the income components for affected households:

```python
expected_income_employee
expected_income_social_transfers
income_rental
expected_income_financial_assets
wealth_other_financial_assets
```

## Model Decision

For property affordability powers, non-positive-income households use a zero
affordability base:

- upstream income components remain unchanged and diagnosable;
- the local power base is clamped to zero;
- households are not excluded from the tenure decision solely because expected
  income is non-positive.

If zero willingness creates downstream housing-market issues, reopen that as a
participation-policy question rather than silently introducing a positive floor.

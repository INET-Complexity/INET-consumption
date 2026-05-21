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

One plausible route to negative household expected income is negative
`wealth_other_financial_assets`, because expected financial-asset income is
currently linear in current other financial assets:

```python
return income_coefficient * current_other_financial_assets
```

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

## Open Model Decision

The code needs an explicit modeling decision for non-positive-income households
in the property affordability equations. Candidate behaviors include:

- clamp the income base used for fractional powers to zero;
- use a small positive floor if zero willingness creates downstream issues;
- exclude non-positive-income households from buy/rent affordability decisions;
- prevent the upstream income components from producing negative expected
  household income.

Until that decision is made, suppressing the warning would hide a real
affordability-state inconsistency.

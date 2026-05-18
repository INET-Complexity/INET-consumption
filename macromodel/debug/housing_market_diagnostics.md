# Housing Market Diagnostics

This note records the housing wealth accounting diagnostic for the
`bug/housing-initialisation` investigation.

## Symptom

`wealth_other_properties` can stay flat or decline when an owner-occupier moves
out of a previous home. The expected accounting movement is:

- the new owner-occupied home contributes to `wealth_main_residence`;
- the old owned but vacant home contributes to `wealth_other_properties`.

If the old home is removed from main-residence wealth but does not enter
other-property wealth, the housing state is internally inconsistent.

## Cause

`wealth_other_properties` is computed from properties where
`Is Owner-Occupied == 0`. The flag must therefore stay aligned with the
derived owner/inhabitant state:

```python
owner_id >= 0 and owner_id == inhabitant_id
```

The failure mode was a stale flag after a move:

```text
Corresponding Owner Household ID      = household_id
Corresponding Inhabitant Household ID = -1
Is Owner-Occupied                     = 1
```

This is an owned vacant property. It should count as other-property wealth, not
as owner-occupied housing.

## Useful Derived Checks

When debugging housing wealth, record these derived values from
`housing_market.states["properties"]`:

- `owner_occupied_flag_mismatch_count`: count where `Is Owner-Occupied` differs
  from `owner_id >= 0 and owner_id == inhabitant_id`;
- `owner_occupied_flag_mismatch_value`: total value of those mismatched
  properties;
- `vacant_owned_flagged_oo_value`: total value where `owner_id >= 0`,
  `inhabitant_id == -1`, and `Is Owner-Occupied == 1`;
- `truth_main_value`: total value where `owner_id >= 0` and
  `owner_id == inhabitant_id`;
- `truth_other_value`: total value where `owner_id >= 0` and
  `owner_id != inhabitant_id`.

The expected invariant after housing-market processing is:

```python
owner_occupied_flag_mismatch_count == 0
vacant_owned_flagged_oo_value == 0
```

## Depreciation

There is no explicit depreciation rule for housing properties in this path.
Property values can change through stochastic revaluation and through completed
sales updating `Value` to the transaction price. Depreciation logic applies to
other real assets and firms' capital accounting, not to
`wealth_other_properties`.

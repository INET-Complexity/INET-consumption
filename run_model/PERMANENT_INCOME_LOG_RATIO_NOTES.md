# Permanent-Income Log-Ratio Notes

This note documents the notebook diagnostics around the saved household series
`ln(y^p / y)`, exposed in the plotting helpers as:

- `ln_y_p_over_p`: aggregate `ln(y^p / y)`
- `zeta_times_posterior_mean`: household-specific term
- `common_log_ratio`: Stage 3 common forecast term

## Definition

The saved decomposition is:

```text
ln(y^p / y) = zeta * posterior_mean + common_log_ratio
```

The label is `ln(y^p / y)`, not `ln(y^p / p)`.

## Timing

`common_log_ratio` is simultaneous with current consumption and the other
current-period model outcomes. In the saved HDF5 output, row `0` is the initial
state, so simulation period `t` is stored at saved row `t + 1`.

## Behavior in the FRA benchmark

For the FRA benchmark discussed in the notebook:

- `zeta * posterior_mean` is nearly flat in the aggregate series.
- Most of the time variation in `ln(y^p / y)` comes from `common_log_ratio`.
- Within `common_log_ratio`, the dominant moving regressor is
  `log_real_pc_income`.

The estimated Stage 3 coefficient on `log_real_pc_income` in that run is about
`-1.018`, so quarter-to-quarter changes in this regressor move the common
forecast almost one-for-one with the opposite sign:

```text
Delta common_log_ratio_from_income
  ~= -1.018 * Delta log_real_pc_income
```

In the FRA benchmark periods we inspected, the income term explained nearly all
of the quarter-to-quarter change in `common_log_ratio`, with the remaining
regressors contributing only small offsets.

# Unemployment shock with temporary rehiring restriction

Set `prevent_rehiring=True` on an `unemployment_rate` ShockSpec to keep workers directly separated by that shock ineligible for hiring until `period + duration` (exclusive end). Separations still occur each active quarter. A worker displaced in the last active quarter is released with the earlier cohorts at the common end of the window. Release restores eligibility; it does not force a hire.

Workers remain unemployed, retaining the existing benefit treatment. Initially unemployed workers and endogenous separations remain eligible. The filter covers the default, reservation-wage-binding and Poledna clearers. Other shocks reject the option. The default is False, preserving the original experiment. Initial separation draws are unchanged when toggling the flag with the same name/seed; later cohorts can differ because employment paths differ.

`run_model/run_irf.ipynb` enables `PREVENT_REHIRING = True` and writes to an output directory suffixed `-no-rehiring`. Set False to reproduce the original design. The shock name stays `unemp_2pp`, avoiding an artificial change in the keyed separation draw. The notebook limits parallel simulations to four; measure RAM before running it. This implementation has unit regression checks, not new macroeconomic results. The paper-figures notebook continues reading its configured experiment directory; explicitly select the restricted directory when analysing the new run.

YAML shock files accepted by the CLI can specify:

```yaml
shocks:
  - name: unemp_2pp
    kind: unemployment_rate
    period: 35
    magnitude: 0.02
    duration: 4
    mode: additive
    prevent_rehiring: true
```

Use a distinct output directory for this variant. `analysis/irf_shock_specs.csv` records `shock_prevent_rehiring`, so the restriction is visible in exported metadata. Existing output files are not relabelled as restricted results.

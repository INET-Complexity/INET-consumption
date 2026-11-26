# Alternative unemployment shock designs

The IRF notebook includes two designs, replacing the previous 2% separation flow in each of four quarters:

| Shock name | Separations | Rehiring eligibility |
|---|---|---|
| `unemp_2pp_once` | Once, 2% of employed-plus-unemployed labour force | Restored after four quarters |
| `unemp_05pp_quarterly` | 0.5% of that labour force in each of four quarters | Common release after four quarters; cohorts blocked for 4/3/2/1 quarters |

Both directly displace about 2% of the labour force in total, subject to available workers and integer rounding. They differ in unemployment exposure: the one-off design blocks the whole cohort for a year; the staggered design gives later cohorts shorter spells. Release restores eligibility without forcing a hire. Endogenous separations and employment changes remain possible.

`ShockSpec.duration` controls how long separations repeat. `prevent_rehiring=True` excludes directly displaced workers from hiring until the common release at `period + rehiring_lockout_duration`. The optional lockout duration defaults to `duration`, must be an integer at least as long as `duration`, and requires `prevent_rehiring=True`. This separates a one-quarter separation intervention from a four-quarter unemployment restriction.

Workers remain unemployed with the existing benefit treatment. Initially unemployed workers and endogenous separations remain eligible. The filter covers default, reservation-wage-binding and Poledna clearers. The library defaults preserve unrestricted hiring. Other shock kinds reject the restriction.

`run_model/run_irf.ipynb` enables `PREVENT_REHIRING=True`. Set it False for unrestricted versions of these two designs. Results go to a directory with `-job-loss-designs-no-rehiring` appended (or `-job-loss-designs` without the restriction), preserving earlier outputs. The notebook limits concurrent runs to four; measure RAM before launching. Existing figures are cleared because they describe the earlier experiment. The paper-figures notebook must explicitly select the new directory and shock name; old outputs are not relabelled.

Example YAML for the CLI:

```yaml
shocks:
  - name: unemp_2pp_once
    kind: unemployment_rate
    period: 35
    magnitude: 0.02
    duration: 1
    mode: additive
    prevent_rehiring: true
    rehiring_lockout_duration: 4
  - name: unemp_05pp_quarterly
    kind: unemployment_rate
    period: 35
    magnitude: 0.005
    duration: 4
    mode: additive
    prevent_rehiring: true
    rehiring_lockout_duration: 4
```

Use a distinct output directory. `analysis/irf_shock_specs.csv` records the restriction and effective lockout duration. For a fixed shock name/seed the restriction does not alter first-wave separation draws; later cohorts may differ as employment paths diverge. Differently named designs use different keyed draws. This change includes unit regression checks, not new macroeconomic estimates.

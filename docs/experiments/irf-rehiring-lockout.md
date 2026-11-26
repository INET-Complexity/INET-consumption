# Alternative unemployment shock designs

The notebook now defaults to `REHIRING_SCOPE="all_unemployed"`: no hiring from the unemployed pool is allowed during the four-quarter restriction. This prevents replacement hiring from offsetting the aggregate employment loss. Job-to-job moves remain allowed. Benefits, work effort and normal endogenous separations continue, so the unemployment-rate response is endogenous and can exceed the initial rise; this does not impose a fixed +2 pp unemployment-rate plateau.

Two separation profiles are available, replacing the previous 2% flow in each of four quarters:

| Shock name | Separations | Rehiring eligibility |
|---|---|---|
| `unemp_2pp_once` | Once, 2% of employed-plus-unemployed labour force | Restored after four quarters |
| `unemp_05pp_quarterly` | 0.5% of that labour force in each of four quarters | Common release after four quarters; cohorts blocked for 4/3/2/1 quarters |

Both directly displace about 2% of the labour force in total, subject to available workers and integer rounding. They differ in unemployment exposure: the one-off design blocks the whole cohort for a year; the staggered design gives later cohorts shorter spells. Release restores eligibility without forcing a hire. Endogenous separations and employment changes remain possible.

`ShockSpec.duration` controls how long separations repeat. `prevent_rehiring=True` excludes directly displaced workers from hiring until the common release at `period + rehiring_lockout_duration`. The optional lockout duration defaults to `duration`, must be an integer at least as long as `duration`, and requires `prevent_rehiring=True`. This separates a one-quarter separation intervention from a four-quarter unemployment restriction.

`rehiring_scope="displaced"` retains the earlier cohort-only restriction: initially unemployed workers and endogenous separations remain eligible. With `rehiring_scope="all_unemployed"`, everyone in the unemployed pool is excluded, including workers separated after the prehook during labour clearing. Activity status remains unemployed and benefit treatment is unchanged. The filter applies only to unemployment-to-employment matching, so job-to-job matching is preserved. All three clearers (default, reservation-wage-binding and Poledna) respect the restriction. The library defaults preserve unrestricted hiring; specifying the all-unemployed scope requires `prevent_rehiring=True`. Other shock kinds reject the restriction.

`run_model/run_irf.ipynb` enables `PREVENT_REHIRING=True` and `REHIRING_SCOPE="all_unemployed"`. Select `"displaced"` to reproduce the earlier cohort-only experiment. Set it False for unrestricted versions of these two designs. The all-unemployed experiment adds `-job-loss-designs-no-rehiring-all-unemployed` to the run directory; the earlier cohort-only experiment uses `-job-loss-designs-no-rehiring`. Without restrictions the suffix is `-job-loss-designs`. Earlier outputs are preserved. The notebook limits concurrent runs to four; measure RAM before launching. Existing figures are cleared because they describe the earlier experiment. The paper-figures notebook must explicitly select the new directory and shock name; old outputs are not relabelled.

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
    rehiring_scope: all_unemployed
    rehiring_lockout_duration: 4
  - name: unemp_05pp_quarterly
    kind: unemployment_rate
    period: 35
    magnitude: 0.005
    duration: 4
    mode: additive
    prevent_rehiring: true
    rehiring_scope: all_unemployed
    rehiring_lockout_duration: 4
```

Use a distinct output directory. `analysis/irf_shock_specs.csv` records the restriction, scope and effective lockout duration. For a fixed shock name/seed the restriction does not alter first-wave separation draws; later cohorts may differ as employment paths diverge. Differently named designs use different keyed draws. This change includes unit regression checks, not new macroeconomic estimates.


## Separation size versus the unemployment-rate response

The magnitude is the directly imposed separation flow, not the total realised rate difference. The one-off profile uses duration=1: it applies the 2%-of-labour-force separation only once. The staggered profile applies 0.5% in each of four quarters. Both also shut down ordinary job finding under the all-unemployed scope, so their unemployment responses can converge even though their exogenous separation profiles differ. Unemployment panels show percentage-point differences from the contemporaneous control, not annualised rates.

The saved 24-seed all-unemployed run inspected on 2026-11-26 shows the one-off unemployment response at +3.768, +5.358, +7.044 and +8.611 pp in the first four quarters; the staggered response is +2.286, +4.472, +6.633 and +8.682 pp. The one-off specification still records separation duration=1. These are results from the saved run before the small-firm sampler correction, not new validation results. The aggregates cannot separately identify foregone ordinary hires from other equilibrium responses. A freeze-only arm would be needed for that attribution.

Sampling retains at least one employee per firm. If sampled workers would empty a firm, replacements are drawn from other firms' remaining capacity. If the total requested separations exceed that capacity, the shock is capped and emits a warning with requested and feasible counts. Such capped runs do not deliver the nominal shock size and must be reported accordingly.

The committed paper notebook now includes dataset, restriction-scope and job-loss-shock selectors, validates required shock names, and labels the exogenous separation profiles explicitly. Old outputs are never treated as new macro-mode results.

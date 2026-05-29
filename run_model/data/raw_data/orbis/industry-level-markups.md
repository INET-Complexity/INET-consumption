# Constructing Industry-Level Markups from ORBIS Global Financials

## Dataset

Dataset used:

```text
orbis_academics_quarterly_industry_global_financials_and_ratios_2014
```

This document describes how to construct:

1. Operating markups
2. Full-cost markups
3. Funding-inclusive markups
4. Unit cost measures
5. Suggested data cleaning procedures
6. Recommended aggregation procedures for France NACE Rev.2 industries

---

## 1. Key Conceptual Distinction

The markup object depends on the pricing framework.

### Standard marginal-cost pricing

$$
P = \mu \times MC
$$

Typically excludes:

* financing costs,
* accounting profits,
* capital structure.

### Full-cost pricing

$$
P = (1+m) \times (\text{labor} + \text{materials} + \text{capital costs})
$$

Includes:

* depreciation,
* potentially financing costs.

This guide focuses on empirically constructing:

* operating markups,
* full-cost markups,
* financing-inclusive markups.

---

## 2. Core Variables from ORBIS

### Revenue Variables

Preferred revenue variable:

| Economic concept | ORBIS variable |
| --- | --- |
| Operating revenue | `operating_revenue_turnover_` |

Alternative:

| Economic concept | ORBIS variable |
| --- | --- |
| Sales | `sales` |

Recommendation:
Use `operating_revenue_turnover_` as baseline because it is more consistently populated internationally.

---

### Cost Variables

| Economic concept | ORBIS variable |
| --- | --- |
| Materials/intermediate inputs | `material_costs` |
| Labor costs | `costs_of_employees` |
| Depreciation/amortization | `depreciation_amortization` |
| Interest costs | `interest_paid` |

Important:

* Prefer `interest_paid` over `financial_expenses`.
* `financial_expenses` may include non-operating financial items.

---

## 3. Unit Cost Definitions

### 3.1 Variable Operating Cost

$$
UC^{OP} = M + L
$$

where:

* M: material costs
* L: labor costs

Operational implementation:

```text
UC_OP = material_costs + costs_of_employees
```

---

### 3.2 Full Operating Cost

$$
UC^{FC} = M + L + D
$$

where:

* D: depreciation/amortization

Operational implementation:

```text
UC_FC = material_costs + costs_of_employees + depreciation_amortization
```

Interpretation:

* Includes capital consumption.
* Appropriate for full-cost pricing frameworks.
* Recommended baseline for medium-run industry pricing models.

---

### 3.3 Funding-Inclusive Unit Cost

$$
UC^{ALL} = M + L + D + I
$$

where:

* I: interest paid

Operational implementation:

```text
UC_ALL = material_costs
       + costs_of_employees
       + depreciation_amortization
       + interest_paid
```

Interpretation:

* Includes financing pressure.
* Appropriate for models where debt servicing affects pricing.
* Particularly relevant for:
    * construction,
    * infrastructure,
    * transport,
    * capital-intensive sectors.

Caution:

* This measure reflects financing structure as well as operational costs.
* High leverage mechanically lowers measured markup.

---

## 4. Markup Definitions

### 4.1 Operating Markup

$$
\mu^{OP} = \frac{R}{M + L}
$$

Operational implementation:

```text
MU_OP = operating_revenue_turnover_
      / (material_costs + costs_of_employees)
```

Interpretation:

* Closest to variable-cost markup.
* Most comparable to standard industrial organization measures.

---

### 4.2 Full-Cost Markup

$$
\mu^{FC} = \frac{R}{M + L + D}
$$

Operational implementation:

```text
MU_FC = operating_revenue_turnover_
      / (
          material_costs
        + costs_of_employees
        + depreciation_amortization
        )
```

Interpretation:

* Includes capital consumption.
* Recommended baseline for cost-plus pricing models.
* Likely the most appropriate measure for industry pricing calibration.

---

### 4.3 Funding-Inclusive Markup

$$
\mu^{ALL} = \frac{R}{M + L + D + I}
$$

Operational implementation:

```text
MU_ALL = operating_revenue_turnover_
       / (
           material_costs
         + costs_of_employees
         + depreciation_amortization
         + interest_paid
         )
```

Interpretation:

* Includes operational and financing costs.
* Appropriate if the pricing model explicitly includes funding costs.

---

## 5. Recommended Data Cleaning

### 5.1 Consolidation

Strong recommendation:

Use unconsolidated accounts whenever possible.

Variable:

```text
consolidation_code
```

Reason:

* Consolidated accounts may include intra-group flows,
* transfer pricing,
* duplicated financing structures,
* distorted cost measures.

---

### 5.2 Keep Only Firms with Positive Economic Activity

Recommended filters:

```text
operating_revenue_turnover_ > 0
material_costs >= 0
costs_of_employees >= 0
```

For markup construction:

```text
denominator > 0
```

---

### 5.3 Sector Exclusions

Recommended exclusions:

| Sector | Reason |
| --- | --- |
| Finance (NACE K) | Financial statements not comparable |
| Public administration | Non-market pricing |
| Real estate (optional) | Extremely noisy markups |
| Holding companies | Non-operating entities |

---

### 5.4 Outlier Treatment

Recommended:

* Winsorize top/bottom 1% or 2.5%.
* Especially important for:
    * small firms,
    * service firms,
    * firms with missing cost items.

---

## 6. Recommended Aggregation Procedure

### Aggregation Level

Recommended:

* France,
* NACE Rev.2 2-digit industries,
* annual aggregation.

---

### Weighting

Recommended weighting variable:

```text
operating_revenue_turnover_
```

Alternative:

```text
number_of_employees
```

---

### Preferred Statistics

Recommended:

1. Weighted median markup
2. Sales-weighted mean markup
3. Employment-weighted mean markup

Weighted median is often more robust.

---

## 7. Additional Useful Variables

### Capital Intensity

| Concept | ORBIS variable |
| --- | --- |
| Fixed assets | `fixed_assets` |
| Tangible assets | `tangible_fixed_assets` |

Useful for:

* capital-intensity controls,
* heterogeneity analysis.

---

### Leverage / Financial Structure

| Concept | ORBIS variable |
| --- | --- |
| Loans | `loans` |
| Long-term debt | `long_term_debt` |
| Financial leverage | `gearing_` |

Useful for:

* analyzing funding-cost sensitivity,
* debt-pricing transmission.

---

## 8. Recommended Baseline Specification

Recommended baseline markup:

$$
\mu^{FC} = \frac{R}{M + L + D}
$$

Recommended robustness specification:

$$
\mu^{ALL} = \frac{R}{M + L + D + I}
$$

Rationale:

* Separates operational pricing from financing-driven pricing.
* Allows direct evaluation of interest-cost sensitivity.
* Particularly useful in high-rate environments.

---

## 9. Suggested Empirical Workflow

### Step 1

Select unconsolidated firms.

### Step 2

Keep firms with:

```text
operating_revenue_turnover_ > 0
```

and positive markup denominator.

### Step 3

Construct:

* UC_OP
* UC_FC
* UC_ALL
* MU_OP
* MU_FC
* MU_ALL

### Step 4

Winsorize markup measures.

### Step 5

Aggregate by:

* year,
* France,
* NACE Rev.2 industry.

### Step 6

Compute:

* weighted median,
* weighted mean.

---

## 10. Interval Columns in Aggregated Output

The file:

```text
orbis_markups_by_nace_rev_2_main_section.csv
```

is produced by:

```text
run_model/src/orbis_sector_markups.py
```

Rows are aggregated by:

* year,
* NACE Rev.2 main section.

Weights are:

```text
operating_revenue_turnover_
```

The interval columns are not statistical confidence intervals.
They are deterministic within-sector spread measures computed from the weighted firm-level markup distribution after winsorization.

For each markup measure:

* MU_OP
* MU_FC
* MU_ALL

the script computes:

```text
weighted_mean
weighted_median
median_interval_low
median_interval_high
mean_interval_low
mean_interval_high
```

Median interval columns are weighted quantiles around the weighted median.
With a median interval half-width of 0.40:

```text
median_interval_low  = weighted 10th percentile
median_interval_high = weighted 90th percentile
```

Mean interval columns are centered on the weighted mean.
With the default mean interval standard-deviation multiplier of 1.0:

```text
mean_interval_low  = weighted_mean - weighted_standard_deviation
mean_interval_high = weighted_mean + weighted_standard_deviation
```

For the France configuration, the active markup columns are:

```text
markup_central_column = mu_all_weighted_median
markup_lower_column   = mu_all_median_interval_low
markup_upper_column   = mu_all_median_interval_high
```

Thus, the model uses the funding-inclusive markup median as the central markup, and the revenue-weighted 10th and 90th percentiles as the lower and upper markup bounds.

---

## 11. Notes on Interpretation

### Operating markup

Closest to standard economic markup.

### Full-cost markup

Closest to managerial/accounting pricing.

### Funding-inclusive markup

Captures:

* leverage pressure,
* interest-rate sensitivity,
* financing conditions.

But it should not be interpreted purely as product-market power.

It combines:

* operational pricing,
* financing structure.

---

## 12. Recommended Variables to Export from ORBIS

Minimum recommended export:

```text
bvd_id_number
closing_date
consolidation_code
operating_revenue_turnover_
material_costs
costs_of_employees
depreciation_amortization
interest_paid
fixed_assets
long_term_debt
number_of_employees
```

Additional useful controls:

```text
gearing_
profit_margin_
solvency_ratio_asset_based_
current_ratio_x_
```

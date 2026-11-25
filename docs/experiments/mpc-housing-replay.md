# Housing replay: implementation and unresolved financing control

The serial experimental runner `run_housing_replay_experiment` records baseline housing matches before credit clearing, replays those matches in treatment, and retains the model's native purchase-completion, seller, mortgage and bank settlement. It checks the completed transaction table and each household's net housing cash need and housing liquid-asset draw. It aborts when those controls do not hold. It does not overwrite household cash or bank accounts.

The control window is explicit: `control_periods=1` addresses impact; cumulative controlled estimates require control of the whole cumulative window. The runner requires `n_jobs=1` in a dedicated process. Temporary method instrumentation is restored on success and failure. The ordinary MPC runner is unchanged.

Seed 3, transfer period 60, 10% median positive income, current minimum-income regime: replayed completed transactions match but the strict cash/funding check fails (maximum difference about 3.63 million model units). A successful replay of matches is therefore insufficient to claim that housing resources are held fixed. No controlled MPC is reported from this failed smoke run.

Two estimands remain possible: fix transactions while allowing endogenous financing, or fix transactions and financing. The latter requires intervention before mortgage origination/purchase funding and matching counterpart entries, not a post-settlement cash overwrite. The user selected transaction control with endogenous financing, as described below. The previously verified unrestricted 30-seed results remain valid.

## Selected specification

User selected transaction control with endogenous financing. Default validation requires the entire completed transaction table (identities, prices, sales and rentals) to match baseline exactly. Mortgage funding and housing-related asset draws may respond, and their deviations are reported in the audit. The earlier 3.63-million funding difference is therefore diagnostic, not a failure under this specification. Strict financing matching remains an optional lower-level diagnostic.

The comparison controls period 60 only and reports impact MPCs. It uses the original unrestricted run's filtered household keys to prevent post-intervention sample composition from driving the comparison. Separate worker processes can run independent seed batches, each serial internally; concurrent model runs within one process are unsupported.

## Full-campaign boundary

The first campaign verified 28 seed pairs, failed on seed 15 and did not run seed 16. Replaying only completed baseline transactions instead of all proposals still fails in seed 15. The affected purchase passes DSTI but the provisional mortgage hits the 80% LTV ceiling, and the own-funds completion test cancels it. The mortgage is then zeroed. Thus no all-seed controlled estimate is available.

The downpayment-reservation question is an open issue; the user deferred further intervention work. Such a reservation would change financing priority and is not implemented. Existing lending limits and accounting remain enforced. No selected-seed paper result is substituted.

RNG stream redesign is also deferred. The PR adds diagnostic tooling, not an endorsed all-seed housing-controlled estimate or a change to financing priority.

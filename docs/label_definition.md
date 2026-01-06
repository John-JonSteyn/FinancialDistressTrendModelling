# Distress proxy label definition

## Purpose
This project does not aim to predict legal bankruptcy or default. Instead, it labels company-period observations as **stable** or **deteriorating** based on **forward-looking early-warning signals** observed in subsequent filings.

The label is designed to capture **deterioration trajectories**, consistent with longitudinal monitoring approaches used in restructuring and credit analysis.

## Unit of analysis
- **Entity:** company (`cik`)
- **Time:** fiscal reporting period (`period`, parsed as `YYYYMMDD`)
- **Observation:** one company-period row (typically quarterly cadence)

## Lookahead window
For each company-period observation at time *t*, we evaluate the **next 3 company periods** (approximately the next 3 quarters).

We define a forward-looking count:
- `lookahead_window_quarters = 3`
- `minimum_future_quarters_required = 2`
- `minimum_trigger_quarters_required = 2`

## Quarterly distress signal (evaluated at each future quarter)
A future quarter is marked as a distress signal quarter if **any** of the following conditions holds:

1. **Negative operating cash flow**
   - `NetCashProvidedByUsedInOperatingActivities < 0`

2. **Material quarter-on-quarter increase in leverage proxy**
   - `ratio_liabilities_to_assets__change_qoq > 0.05`
   - Interpreted as a rise of more than 5 percentage points in liabilities-to-assets.

3. **Material quarter-on-quarter decline in cash intensity**
   - `ratio_cash_to_assets__change_qoq < -0.02`
   - Interpreted as a drop of more than 2 percentage points in cash-to-assets.

4. **Negative net income**
   - `NetIncomeLoss < 0`

## Label rule
An observation at time *t* is labelled **deteriorating** if, within the next 3 quarters:

- At least **2** of the next **3** quarters are distress signal quarters, and
- At least **2** future quarters are actually available (to avoid labelling from insufficient forward data)

Formally:
- `deteriorating_at_t = (distress_signal_quarter_count_in_t+1..t+3 >= 2) AND (available_future_quarters_in_t+1..t+3 >= 2)`

Otherwise:
- `stable_at_t = not deteriorating_at_t`

## Output fields
The labelling script produces:
- `distress_proxy_label` (0 = stable, 1 = deteriorating)
- `distress_signal_future_quarter_count` (0..3)
- `future_quarter_count_available` (0..3)

## Rationale
- The label is **forward-looking** and based on **persistent deterioration**, rather than a single noisy quarter.
- Thresholds are explicitly defined, reproducible, and aligned with early-warning monitoring logic.
- The approach avoids using bankruptcy outcomes, reducing outcome leakage and focusing on deterioration dynamics.

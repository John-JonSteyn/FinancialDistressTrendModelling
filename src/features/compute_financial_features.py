"""
Financial feature engineering for FSDS company-period panel data.

This script reads the processed company-period panel and produces a modelling-ready
feature table containing:

- Core ratio features
    - Leverage
    - Profitability
    - Cash-flow quality
    - Cash intensity

- Trend features
    - Quarter-on-quarter change
    - Year-on-year change
    - Rolling means
    - Short-term slopes

"""

from __future__ import annotations

import pathlib
import numpy
import pandas

# -------------------------
# Configuration constants
# -------------------------

PROCESSED_DATA_DIRECTORY = pathlib.Path("data/processed")
OUTPUT_TABLE_DIRECTORY = pathlib.Path("outputs/tables")

INPUT_PANEL_FILENAME = "panel.parquet"
OUTPUT_FEATURES_FILENAME = "features.parquet"

FEATURE_LIST_FILENAME = "feature_list.csv"
FEATURE_MISSINGNESS_FILENAME = "feature_missingness.csv"
FEATURE_ROW_COUNTS_BY_YEAR_FILENAME = "feature_row_counts_by_year.csv"

MODELLING_START_YEAR = 2019
MODELLING_END_YEAR = 2024

PERIOD_DATE_FORMAT = "%Y%m%d"

ROLLING_WINDOW_QUARTERS = 4
SLOPE_WINDOW_QUARTERS = 4

# Parse FSDS period values into a proper datetime series using explicit YYYYMMDD formatting.
def parse_period_to_datetime(period_series: pandas.Series) -> pandas.Series:
    print("Parsing fiscal period values to datetime...")
    period_as_integer_series = pandas.to_numeric(period_series, errors="coerce").astype("Int64")
    period_as_string_series = period_as_integer_series.astype("string")

    period_datetime_series = pandas.to_datetime(
        period_as_string_series,
        format=PERIOD_DATE_FORMAT,
        errors="coerce",
    )

    return period_datetime_series


# Filter the dataset to a fixed modelling horison by fiscal year.
def filter_to_modelling_horison(panel_dataframe: pandas.DataFrame) -> pandas.DataFrame:
    print(f"Filtering panel to modelling horison: {MODELLING_START_YEAR}–{MODELLING_END_YEAR}...")
    period_datetime_series = parse_period_to_datetime(panel_dataframe["period"])
    fiscal_year_series = period_datetime_series.dt.year

    filtered_dataframe = panel_dataframe.loc[
        (fiscal_year_series >= MODELLING_START_YEAR)
        & (fiscal_year_series <= MODELLING_END_YEAR)
    ].copy()

    filtered_dataframe["period_datetime"] = period_datetime_series.loc[filtered_dataframe.index]
    filtered_dataframe["fiscal_year"] = fiscal_year_series.loc[filtered_dataframe.index]

    print(f"Rows after horison filter: {len(filtered_dataframe):,}")
    return filtered_dataframe


# Safely compute a ratio while avoiding divide-by-zero and non-finite outputs.
def compute_safe_ratio(
    numerator_series: pandas.Series,
    denominator_series: pandas.Series,
) -> pandas.Series:
    denominator_nonzero_series = denominator_series.replace(0, numpy.nan)
    ratio_series = numerator_series / denominator_nonzero_series
    ratio_series = ratio_series.replace([numpy.inf, -numpy.inf], numpy.nan)
    return ratio_series


# Create a list of base financial columns that must exist for feature creation.
def get_base_financial_column_names() -> list[str]:
    return [
        "Assets",
        "Liabilities",
        "StockholdersEquity",
        "Revenues",
        "NetIncomeLoss",
        "OperatingIncomeLoss",
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "NetCashProvidedByUsedInOperatingActivities",
        "DebtCurrent",
        "LongTermDebtCurrent",
        "LongTermDebtNoncurrent",
        "InterestExpense",
    ]


# Compute ratio features from the panel-level statement values.
def compute_ratio_features(panel_dataframe: pandas.DataFrame) -> pandas.DataFrame:
    print("Computing ratio features...")
    features_dataframe = panel_dataframe.copy()

    assets_series = features_dataframe.get("Assets")
    liabilities_series = features_dataframe.get("Liabilities")
    equity_series = features_dataframe.get("StockholdersEquity")
    revenues_series = features_dataframe.get("Revenues")
    net_income_series = features_dataframe.get("NetIncomeLoss")
    operating_income_series = features_dataframe.get("OperatingIncomeLoss")
    operating_cash_flow_series = features_dataframe.get("NetCashProvidedByUsedInOperatingActivities")

    cash_series = features_dataframe.get("CashAndCashEquivalentsAtCarryingValue")
    cash_and_restricted_cash_series = features_dataframe.get(
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
    )

    debt_current_series = features_dataframe.get("DebtCurrent")
    long_term_debt_current_series = features_dataframe.get("LongTermDebtCurrent")
    long_term_debt_noncurrent_series = features_dataframe.get("LongTermDebtNoncurrent")
    interest_expense_series = features_dataframe.get("InterestExpense")

    total_debt_series = (
        debt_current_series.fillna(0)
        + long_term_debt_current_series.fillna(0)
        + long_term_debt_noncurrent_series.fillna(0)
    )

    features_dataframe["ratio_liabilities_to_assets"] = compute_safe_ratio(liabilities_series, assets_series)
    features_dataframe["ratio_equity_to_assets"] = compute_safe_ratio(equity_series, assets_series)
    features_dataframe["ratio_total_debt_to_assets"] = compute_safe_ratio(total_debt_series, assets_series)

    features_dataframe["ratio_net_income_margin"] = compute_safe_ratio(net_income_series, revenues_series)
    features_dataframe["ratio_operating_income_margin"] = compute_safe_ratio(operating_income_series, revenues_series)
    features_dataframe["ratio_operating_cash_flow_margin"] = compute_safe_ratio(operating_cash_flow_series, revenues_series)

    features_dataframe["ratio_cash_to_assets"] = compute_safe_ratio(cash_series, assets_series)
    features_dataframe["ratio_cash_and_restricted_cash_to_assets"] = compute_safe_ratio(
        cash_and_restricted_cash_series,
        assets_series,
    )

    features_dataframe["ratio_interest_coverage_operating_income"] = compute_safe_ratio(
        operating_income_series,
        interest_expense_series,
    )

    print("Ratio feature computation complete.")
    return features_dataframe


# Compute quarter-on-quarter and year-on-year changes for selected features.
def compute_change_features(
    features_dataframe: pandas.DataFrame,
    feature_column_names: list[str],
) -> pandas.DataFrame:
    print("Computing quarter-on-quarter and year-on-year changes...")
    output_dataframe = features_dataframe.sort_values(
        ["cik", "period_datetime"], ascending=[True, True]
    ).copy()

    for feature_column_name in feature_column_names:
        output_dataframe[f"{feature_column_name}__change_qoq"] = (
            output_dataframe.groupby("cik")[feature_column_name].diff(1)
        )
        output_dataframe[f"{feature_column_name}__change_yoy"] = (
            output_dataframe.groupby("cik")[feature_column_name].diff(ROLLING_WINDOW_QUARTERS)
        )

    print("Change feature computation complete.")
    return output_dataframe


# Compute rolling means for selected features over a fixed number of quarters.
def compute_rolling_mean_features(
    features_dataframe: pandas.DataFrame,
    feature_column_names: list[str],
    rolling_window_quarters: int,
) -> pandas.DataFrame:
    print(f"Computing rolling means over {rolling_window_quarters} quarters...")
    output_dataframe = features_dataframe.sort_values(
        ["cik", "period_datetime"], ascending=[True, True]
    ).copy()

    for feature_column_name in feature_column_names:
        output_dataframe[f"{feature_column_name}__rolling_mean_{rolling_window_quarters}q"] = (
            output_dataframe.groupby("cik")[feature_column_name]
            .rolling(window=rolling_window_quarters, min_periods=rolling_window_quarters)
            .mean()
            .reset_index(level=0, drop=True)
        )

    print("Rolling mean computation complete.")
    return output_dataframe


# Compute a trailing slope over a fixed window using ordinary least squares.
def compute_trailing_slope_over_window(values_array: numpy.ndarray) -> float:
    if values_array.size == 0 or numpy.any(numpy.isnan(values_array)):
        return numpy.nan

    time_index_array = numpy.arange(values_array.size, dtype=float)
    return float(numpy.polyfit(time_index_array, values_array.astype(float), deg=1)[0])


# Compute short-term trend slopes for selected features.
def compute_slope_features(
    features_dataframe: pandas.DataFrame,
    feature_column_names: list[str],
    slope_window_quarters: int,
) -> pandas.DataFrame:
    print(f"Computing trend slopes over {slope_window_quarters} quarters...")
    output_dataframe = features_dataframe.sort_values(
        ["cik", "period_datetime"], ascending=[True, True]
    ).copy()

    for feature_column_name in feature_column_names:
        output_dataframe[f"{feature_column_name}__slope_{slope_window_quarters}q"] = (
            output_dataframe.groupby("cik")[feature_column_name]
            .rolling(window=slope_window_quarters, min_periods=slope_window_quarters)
            .apply(
                lambda rolling_series: compute_trailing_slope_over_window(
                    rolling_series.to_numpy()
                ),
                raw=False,
            )
            .reset_index(level=0, drop=True)
        )

    print("Slope feature computation complete.")
    return output_dataframe


# Build and save evidence tables describing feature completeness and coverage.
def write_feature_evidence_tables(features_dataframe: pandas.DataFrame) -> None:
    print("Writing feature evidence tables...")
    OUTPUT_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    feature_column_names = [
        column_name
        for column_name in features_dataframe.columns
        if column_name not in {"cik", "period", "period_datetime", "fiscal_year", "latest_filed"}
    ]

    pandas.DataFrame({"feature_name": sorted(feature_column_names)}).to_csv(
        OUTPUT_TABLE_DIRECTORY / FEATURE_LIST_FILENAME,
        index=False,
    )

    (
        features_dataframe[feature_column_names]
        .isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "feature_name", 0: "missing_value_count"})
        .sort_values("missing_value_count", ascending=False)
        .to_csv(OUTPUT_TABLE_DIRECTORY / FEATURE_MISSINGNESS_FILENAME, index=False)
    )

    (
        features_dataframe.groupby("fiscal_year")
        .size()
        .reset_index(name="row_count")
        .sort_values("fiscal_year")
        .to_csv(OUTPUT_TABLE_DIRECTORY / FEATURE_ROW_COUNTS_BY_YEAR_FILENAME, index=False)
    )

    print("Evidence tables written.")


# Orchestrate feature engineering from the processed panel through to saved features.
def main() -> None:
    print("Starting financial feature engineering pipeline...")
    input_panel_path = PROCESSED_DATA_DIRECTORY / INPUT_PANEL_FILENAME
    if not input_panel_path.exists():
        raise FileNotFoundError(f"Input panel not found: {input_panel_path.resolve()}")

    panel_dataframe = pandas.read_parquet(input_panel_path)
    print(f"Loaded panel with {len(panel_dataframe):,} rows.")

    filtered_panel_dataframe = filter_to_modelling_horison(panel_dataframe)

    features_with_ratios_dataframe = compute_ratio_features(filtered_panel_dataframe)

    base_financial_column_names = [
        column_name
        for column_name in get_base_financial_column_names()
        if column_name in features_with_ratios_dataframe.columns
    ]

    ratio_feature_column_names = [
        column_name
        for column_name in features_with_ratios_dataframe.columns
        if column_name.startswith("ratio_")
    ]

    feature_source_column_names = base_financial_column_names + ratio_feature_column_names
    print(f"Computing trends for {len(feature_source_column_names)} base features...")

    features_with_changes_dataframe = compute_change_features(
        features_dataframe=features_with_ratios_dataframe,
        feature_column_names=feature_source_column_names,
    )

    features_with_rolling_means_dataframe = compute_rolling_mean_features(
        features_dataframe=features_with_changes_dataframe,
        feature_column_names=feature_source_column_names,
        rolling_window_quarters=ROLLING_WINDOW_QUARTERS,
    )

    features_with_slopes_dataframe = compute_slope_features(
        features_dataframe=features_with_rolling_means_dataframe,
        feature_column_names=feature_source_column_names,
        slope_window_quarters=SLOPE_WINDOW_QUARTERS,
    )

    output_features_path = PROCESSED_DATA_DIRECTORY / OUTPUT_FEATURES_FILENAME
    features_with_slopes_dataframe.to_parquet(
        output_features_path,
        engine="pyarrow",
        index=False,
    )

    write_feature_evidence_tables(features_with_slopes_dataframe)

    print(f"Feature engineering complete. Output written to: {output_features_path.resolve()}")


if __name__ == "__main__":
    main()

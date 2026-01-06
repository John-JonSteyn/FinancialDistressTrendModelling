"""
Labels company-period observations as stable or deteriorating using a conservative
forward-looking rule:

An observation at time t is labelled deteriorating if at least 2 of the next 3
quarters show distress signals, and at least 2 future quarters are available.
"""

from __future__ import annotations
import pathlib
import pandas

# -------------------------
# Configuration constants
# -------------------------

PROCESSED_DATA_DIRECTORY = pathlib.Path("data/processed")
OUTPUT_TABLE_DIRECTORY = pathlib.Path("outputs/tables")

INPUT_MODELLING_FEATURES_FILENAME = "modelling_features.parquet"
OUTPUT_MODELLING_TABLE_FILENAME = "modelling_table.parquet"

LABEL_DISTRIBUTION_FILENAME = "label_distribution.csv"
LABEL_RULE_PARAMETERS_FILENAME = "label_rule_parameters.csv"

LOOKAHEAD_WINDOW_QUARTERS = 3
MINIMUM_FUTURE_QUARTERS_REQUIRED = 2
MINIMUM_TRIGGER_QUARTERS_REQUIRED = 2

LEVERAGE_CHANGE_QOQ_THRESHOLD = 0.05
CASH_CHANGE_QOQ_THRESHOLD = -0.02

# Load modelling features from parquet.
def load_modelling_features(modelling_features_path: pathlib.Path) -> pandas.DataFrame:
    if not modelling_features_path.exists():
        raise FileNotFoundError(f"Modelling features not found: {modelling_features_path.resolve()}")
    return pandas.read_parquet(modelling_features_path)


# Validate that required columns exist for the chosen label rule.
def validate_required_columns(modelling_dataframe: pandas.DataFrame) -> None:
    required_column_names = [
        "cik",
        "period_datetime",
        "NetCashProvidedByUsedInOperatingActivities",
        "ratio_liabilities_to_assets__change_qoq",
        "ratio_cash_to_assets__change_qoq",
        "NetIncomeLoss",
    ]

    missing_column_names = [column_name for column_name in required_column_names if column_name not in modelling_dataframe.columns]
    if missing_column_names:
        raise ValueError(f"Missing required columns for labelling: {missing_column_names}")


# Compute per-row distress signal flags based on contemporaneous values.
def compute_distress_signal_flags(modelling_dataframe: pandas.DataFrame) -> pandas.DataFrame:
    output_dataframe = modelling_dataframe.copy()

    output_dataframe["signal_negative_operating_cash_flow"] = (
        output_dataframe["NetCashProvidedByUsedInOperatingActivities"] < 0
    )

    output_dataframe["signal_leverage_increase_qoq"] = (
        output_dataframe["ratio_liabilities_to_assets__change_qoq"] > LEVERAGE_CHANGE_QOQ_THRESHOLD
    )

    output_dataframe["signal_cash_decline_qoq"] = (
        output_dataframe["ratio_cash_to_assets__change_qoq"] < CASH_CHANGE_QOQ_THRESHOLD
    )

    output_dataframe["signal_negative_net_income"] = (
        output_dataframe["NetIncomeLoss"] < 0
    )

    output_dataframe["distress_signal_quarter"] = (
        output_dataframe["signal_negative_operating_cash_flow"]
        | output_dataframe["signal_leverage_increase_qoq"]
        | output_dataframe["signal_cash_decline_qoq"]
        | output_dataframe["signal_negative_net_income"]
    )

    return output_dataframe


# Compute forward-looking distress counts within each company using a fixed lookahead window.
def compute_forward_lookahead_counts(
    modelling_dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    output_dataframe = modelling_dataframe.sort_values(
        ["cik", "period_datetime"],
        ascending=[True, True],
    ).copy()

    def compute_group_forward_counts(distress_signal_series: pandas.Series) -> pandas.DataFrame:
        shifted_future_series = distress_signal_series.shift(-1)

        future_trigger_count_series = (
            shifted_future_series.rolling(
                window=LOOKAHEAD_WINDOW_QUARTERS,
                min_periods=MINIMUM_FUTURE_QUARTERS_REQUIRED,
            )
            .sum()
        )

        future_available_count_series = (
            shifted_future_series.rolling(
                window=LOOKAHEAD_WINDOW_QUARTERS,
                min_periods=MINIMUM_FUTURE_QUARTERS_REQUIRED,
            )
            .count()
        )

        return pandas.DataFrame(
            {
                "distress_signal_future_quarter_count": future_trigger_count_series,
                "future_quarter_count_available": future_available_count_series,
            },
            index=distress_signal_series.index,
        )

    forward_counts_dataframe = (
        output_dataframe.groupby("cik", group_keys=False)["distress_signal_quarter"]
        .apply(compute_group_forward_counts)
    )

    output_dataframe["distress_signal_future_quarter_count"] = forward_counts_dataframe["distress_signal_future_quarter_count"]
    output_dataframe["future_quarter_count_available"] = forward_counts_dataframe["future_quarter_count_available"]

    return output_dataframe


# Apply the rule to produce the final binary label.
def apply_distress_proxy_label(modelling_dataframe: pandas.DataFrame) -> pandas.DataFrame:
    output_dataframe = modelling_dataframe.copy()

    trigger_condition_series = (
        output_dataframe["distress_signal_future_quarter_count"] >= MINIMUM_TRIGGER_QUARTERS_REQUIRED
    )

    availability_condition_series = (
        output_dataframe["future_quarter_count_available"] >= MINIMUM_FUTURE_QUARTERS_REQUIRED
    )

    output_dataframe["distress_proxy_label"] = (trigger_condition_series & availability_condition_series).astype("Int64")

    return output_dataframe


# Write evidence tables describing label distribution and rule parameters.
def write_label_evidence_tables(labelled_dataframe: pandas.DataFrame) -> None:
    OUTPUT_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    label_distribution_dataframe = (
        labelled_dataframe["distress_proxy_label"]
        .value_counts(dropna=False)
        .rename_axis("distress_proxy_label")
        .reset_index(name="row_count")
        .sort_values("distress_proxy_label")
    )

    label_distribution_dataframe.to_csv(
        OUTPUT_TABLE_DIRECTORY / LABEL_DISTRIBUTION_FILENAME,
        index=False,
    )

    rule_parameters_dataframe = pandas.DataFrame(
        [
            {"parameter_name": "LOOKAHEAD_WINDOW_QUARTERS", "parameter_value": LOOKAHEAD_WINDOW_QUARTERS},
            {"parameter_name": "MINIMUM_FUTURE_QUARTERS_REQUIRED", "parameter_value": MINIMUM_FUTURE_QUARTERS_REQUIRED},
            {"parameter_name": "MINIMUM_TRIGGER_QUARTERS_REQUIRED", "parameter_value": MINIMUM_TRIGGER_QUARTERS_REQUIRED},
            {"parameter_name": "LEVERAGE_CHANGE_QOQ_THRESHOLD", "parameter_value": LEVERAGE_CHANGE_QOQ_THRESHOLD},
            {"parameter_name": "CASH_CHANGE_QOQ_THRESHOLD", "parameter_value": CASH_CHANGE_QOQ_THRESHOLD},
        ]
    )

    rule_parameters_dataframe.to_csv(
        OUTPUT_TABLE_DIRECTORY / LABEL_RULE_PARAMETERS_FILENAME,
        index=False,
    )


# Orchestrate forward-looking labelling and save the modelling table.
def main() -> None:
    print("Starting distress proxy label construction...")

    input_modelling_features_path = PROCESSED_DATA_DIRECTORY / INPUT_MODELLING_FEATURES_FILENAME
    output_modelling_table_path = PROCESSED_DATA_DIRECTORY / OUTPUT_MODELLING_TABLE_FILENAME

    PROCESSED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    modelling_dataframe = load_modelling_features(input_modelling_features_path)
    print(f"Loaded modelling features with {len(modelling_dataframe):,} rows and {modelling_dataframe.shape[1]:,} columns.")

    validate_required_columns(modelling_dataframe)
    print("Required columns validated.")

    modelling_dataframe = compute_distress_signal_flags(modelling_dataframe)
    print("Computed per-quarter distress signal flags.")

    modelling_dataframe = compute_forward_lookahead_counts(modelling_dataframe)
    print("Computed forward-looking distress counts.")

    labelled_dataframe = apply_distress_proxy_label(modelling_dataframe)
    print("Applied distress proxy label rule.")

    labelled_dataframe.to_parquet(
        output_modelling_table_path,
        engine="pyarrow",
        index=False,
    )
    print(f"Modelling table written to: {output_modelling_table_path.resolve()}")

    write_label_evidence_tables(labelled_dataframe)
    print("Label evidence tables written.")

    label_value_counts = labelled_dataframe["distress_proxy_label"].value_counts(dropna=False).to_dict()
    print(f"Label distribution: {label_value_counts}")
    print("Distress proxy label construction complete.")


if __name__ == "__main__":
    main()

"""
Implements the academically strongest evaluation approach for a longitudinal
early-warning model by providing two deterministic splits:

1) Primary split (time holdout, monitoring-realistic)
   - Training: 2019–2022 (all companies)
   - Testing:  2023–2024 (all companies, including those seen in training)

2) Secondary split (future unseen companies, robustness)
   - Training: 2019–2022 (all companies)
   - Testing:  2023–2024, restricted to companies not present in training years
"""

from __future__ import annotations
import pathlib
import pandas

# -------------------------
# Configuration constants
# -------------------------

PROCESSED_DATA_DIRECTORY = pathlib.Path("data/processed")
OUTPUT_TABLE_DIRECTORY = pathlib.Path("outputs/tables")

INPUT_MODELLING_TABLE_FILENAME = "modelling_table.parquet"

TIME_HOLDOUT_SPLIT_SUMMARY_FILENAME = "train_test_split_summary_time_holdout.csv"
FUTURE_UNSEEN_COMPANIES_SPLIT_SUMMARY_FILENAME = "train_test_split_summary_future_unseen_companies.csv"

DEFAULT_TRAIN_START_YEAR = 2019
DEFAULT_TRAIN_END_YEAR = 2022
DEFAULT_TEST_START_YEAR = 2023
DEFAULT_TEST_END_YEAR = 2024

LABEL_COLUMN_NAME = "distress_proxy_label"

IDENTIFIER_COLUMN_NAMES = {
    "cik",
    "period",
    "period_datetime",
    "fiscal_year",
    "latest_filed",
}

NON_FEATURE_COLUMN_NAMES = IDENTIFIER_COLUMN_NAMES | {
    LABEL_COLUMN_NAME,
    "distress_signal_quarter",
    "distress_signal_future_quarter_count",
    "future_quarter_count_available",
    "signal_negative_operating_cash_flow",
    "signal_leverage_increase_qoq",
    "signal_cash_decline_qoq",
    "signal_negative_net_income",
}

SPLIT_STRATEGY_TIME_HOLDOUT = "time_holdout"
SPLIT_STRATEGY_FUTURE_UNSEEN_COMPANIES = "future_unseen_companies"

# Load the modelling table from parquet.
def load_modelling_table(modelling_table_path: pathlib.Path) -> pandas.DataFrame:
    if not modelling_table_path.exists():
        raise FileNotFoundError(f"Modelling table not found: {modelling_table_path.resolve()}")
    return pandas.read_parquet(modelling_table_path)


# Select feature columns by excluding known non-feature columns.
def get_feature_column_names(modelling_dataframe: pandas.DataFrame) -> list[str]:
    feature_column_names: list[str] = []
    for column_name in modelling_dataframe.columns:
        if column_name not in NON_FEATURE_COLUMN_NAMES:
            feature_column_names.append(column_name)
    return sorted(feature_column_names)


# Filter rows to a fiscal year range and require a non-missing label.
def filter_rows_for_year_range(
    modelling_dataframe: pandas.DataFrame,
    start_year: int,
    end_year: int,
) -> pandas.DataFrame:
    if "fiscal_year" not in modelling_dataframe.columns:
        raise ValueError("Modelling dataframe is missing required column: fiscal_year")

    filtered_dataframe = modelling_dataframe.loc[
        (modelling_dataframe["fiscal_year"] >= start_year)
        & (modelling_dataframe["fiscal_year"] <= end_year)
    ].copy()

    filtered_dataframe = filtered_dataframe.loc[
        filtered_dataframe[LABEL_COLUMN_NAME].notna()
    ].copy()

    return filtered_dataframe


# Build the primary time-holdout split for monitoring-realistic evaluation.
def build_time_holdout_split(
    modelling_dataframe: pandas.DataFrame,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
) -> dict[str, object]:
    training_dataframe = filter_rows_for_year_range(
        modelling_dataframe=modelling_dataframe,
        start_year=train_start_year,
        end_year=train_end_year,
    )

    testing_dataframe = filter_rows_for_year_range(
        modelling_dataframe=modelling_dataframe,
        start_year=test_start_year,
        end_year=test_end_year,
    )

    feature_column_names = get_feature_column_names(modelling_dataframe)

    training_features_dataframe = training_dataframe[feature_column_names].copy()
    training_label_series = training_dataframe[LABEL_COLUMN_NAME].astype(int).copy()

    testing_features_dataframe = testing_dataframe[feature_column_names].copy()
    testing_label_series = testing_dataframe[LABEL_COLUMN_NAME].astype(int).copy()

    training_company_set = set(training_dataframe["cik"].astype(int).unique().tolist())
    testing_company_set = set(testing_dataframe["cik"].astype(int).unique().tolist())
    overlapping_company_count = len(training_company_set.intersection(testing_company_set))

    return {
        "split_strategy": SPLIT_STRATEGY_TIME_HOLDOUT,
        "training_features": training_features_dataframe,
        "training_labels": training_label_series,
        "testing_features": testing_features_dataframe,
        "testing_labels": testing_label_series,
        "feature_column_names": feature_column_names,
        "train_row_count": int(len(training_dataframe)),
        "test_row_count": int(len(testing_dataframe)),
        "train_unique_company_count": int(len(training_company_set)),
        "test_unique_company_count": int(len(testing_company_set)),
        "overlapping_company_count": int(overlapping_company_count),
    }


# Build the robustness split that tests only on future companies unseen in training years.
def build_future_unseen_companies_split(
    modelling_dataframe: pandas.DataFrame,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
) -> dict[str, object]:
    training_dataframe = filter_rows_for_year_range(
        modelling_dataframe=modelling_dataframe,
        start_year=train_start_year,
        end_year=train_end_year,
    )

    candidate_testing_dataframe = filter_rows_for_year_range(
        modelling_dataframe=modelling_dataframe,
        start_year=test_start_year,
        end_year=test_end_year,
    )

    training_company_set = set(training_dataframe["cik"].astype(int).unique().tolist())

    testing_dataframe = candidate_testing_dataframe.loc[
        ~candidate_testing_dataframe["cik"].astype(int).isin(training_company_set)
    ].copy()

    feature_column_names = get_feature_column_names(modelling_dataframe)

    training_features_dataframe = training_dataframe[feature_column_names].copy()
    training_label_series = training_dataframe[LABEL_COLUMN_NAME].astype(int).copy()

    testing_features_dataframe = testing_dataframe[feature_column_names].copy()
    testing_label_series = testing_dataframe[LABEL_COLUMN_NAME].astype(int).copy()

    testing_company_set = set(testing_dataframe["cik"].astype(int).unique().tolist())
    overlapping_company_count = len(training_company_set.intersection(testing_company_set))

    return {
        "split_strategy": SPLIT_STRATEGY_FUTURE_UNSEEN_COMPANIES,
        "training_features": training_features_dataframe,
        "training_labels": training_label_series,
        "testing_features": testing_features_dataframe,
        "testing_labels": testing_label_series,
        "feature_column_names": feature_column_names,
        "train_row_count": int(len(training_dataframe)),
        "test_row_count": int(len(testing_dataframe)),
        "train_unique_company_count": int(len(training_company_set)),
        "test_unique_company_count": int(len(testing_company_set)),
        "overlapping_company_count": int(overlapping_company_count),
        "candidate_test_row_count_before_company_filter": int(len(candidate_testing_dataframe)),
    }


# Write a split evidence table describing coverage, label balance, and company overlap.
def write_split_summary_table(
    modelling_dataframe: pandas.DataFrame,
    split_result: dict[str, object],
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    output_filename: str,
) -> None:
    OUTPUT_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    labelled_dataframe = modelling_dataframe.loc[
        modelling_dataframe[LABEL_COLUMN_NAME].notna()
    ].copy()

    labelled_dataframe["split_bucket"] = "excluded"

    labelled_dataframe.loc[
        (labelled_dataframe["fiscal_year"] >= train_start_year)
        & (labelled_dataframe["fiscal_year"] <= train_end_year),
        "split_bucket",
    ] = "train_candidate"

    labelled_dataframe.loc[
        (labelled_dataframe["fiscal_year"] >= test_start_year)
        & (labelled_dataframe["fiscal_year"] <= test_end_year),
        "split_bucket",
    ] = "test_candidate"

    split_strategy = str(split_result["split_strategy"])

    if split_strategy == SPLIT_STRATEGY_TIME_HOLDOUT:
        labelled_dataframe.loc[labelled_dataframe["split_bucket"] == "train_candidate", "split_bucket"] = "train"
        labelled_dataframe.loc[labelled_dataframe["split_bucket"] == "test_candidate", "split_bucket"] = "test"

    elif split_strategy == SPLIT_STRATEGY_FUTURE_UNSEEN_COMPANIES:
        training_company_set = set(
            labelled_dataframe.loc[labelled_dataframe["split_bucket"] == "train_candidate", "cik"]
            .astype(int)
            .unique()
            .tolist()
        )

        labelled_dataframe.loc[
            labelled_dataframe["split_bucket"] == "train_candidate",
            "split_bucket",
        ] = "train"

        labelled_dataframe.loc[
            (labelled_dataframe["split_bucket"] == "test_candidate")
            & (~labelled_dataframe["cik"].astype(int).isin(training_company_set)),
            "split_bucket",
        ] = "test"

        labelled_dataframe.loc[
            (labelled_dataframe["split_bucket"] == "test_candidate")
            & (labelled_dataframe["cik"].astype(int).isin(training_company_set)),
            "split_bucket",
        ] = "excluded"

    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    year_label_counts_dataframe = (
        labelled_dataframe.groupby(["split_bucket", "fiscal_year", LABEL_COLUMN_NAME])
        .size()
        .reset_index(name="row_count")
        .rename(columns={LABEL_COLUMN_NAME: "distress_proxy_label"})
        .sort_values(["split_bucket", "fiscal_year", "distress_proxy_label"])
    )

    company_counts_dataframe = (
        labelled_dataframe.groupby("split_bucket")["cik"]
        .nunique()
        .reset_index(name="unique_company_count")
        .sort_values("split_bucket")
    )

    split_metadata_dataframe = pandas.DataFrame(
        [
            {"metric_name": "split_strategy", "metric_value": split_strategy},
            {"metric_name": "train_years", "metric_value": f"{train_start_year}-{train_end_year}"},
            {"metric_name": "test_years", "metric_value": f"{test_start_year}-{test_end_year}"},
            {"metric_name": "train_row_count", "metric_value": int(split_result["train_row_count"])},
            {"metric_name": "test_row_count", "metric_value": int(split_result["test_row_count"])},
            {"metric_name": "train_unique_company_count", "metric_value": int(split_result["train_unique_company_count"])},
            {"metric_name": "test_unique_company_count", "metric_value": int(split_result["test_unique_company_count"])},
            {"metric_name": "overlapping_company_count", "metric_value": int(split_result["overlapping_company_count"])},
        ]
    )

    if split_strategy == SPLIT_STRATEGY_FUTURE_UNSEEN_COMPANIES:
        split_metadata_dataframe = pandas.concat(
            [
                split_metadata_dataframe,
                pandas.DataFrame(
                    [
                        {
                            "metric_name": "candidate_test_row_count_before_company_filter",
                            "metric_value": int(split_result["candidate_test_row_count_before_company_filter"]),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    output_path = OUTPUT_TABLE_DIRECTORY / output_filename

    with open(output_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("# Split metadata\n")
    split_metadata_dataframe.to_csv(output_path, mode="a", index=False)

    with open(output_path, "a", encoding="utf-8") as file_handle:
        file_handle.write("\n# Unique company counts by split bucket\n")
    company_counts_dataframe.to_csv(output_path, mode="a", index=False)

    with open(output_path, "a", encoding="utf-8") as file_handle:
        file_handle.write("\n# Row counts by split bucket, fiscal year, and label\n")
    year_label_counts_dataframe.to_csv(output_path, mode="a", index=False)


# Provide a single entry point to load data, split, and write evidence for both splits.
def load_and_split_modelling_data(
    split_strategy: str = SPLIT_STRATEGY_TIME_HOLDOUT,
    train_start_year: int = DEFAULT_TRAIN_START_YEAR,
    train_end_year: int = DEFAULT_TRAIN_END_YEAR,
    test_start_year: int = DEFAULT_TEST_START_YEAR,
    test_end_year: int = DEFAULT_TEST_END_YEAR,
) -> dict[str, object]:
    print("Loading modelling table and preparing train-test split...")
    print(f"Split strategy: {split_strategy}")
    print(f"Training years: {train_start_year}–{train_end_year}")
    print(f"Testing years:  {test_start_year}–{test_end_year}")

    modelling_table_path = PROCESSED_DATA_DIRECTORY / INPUT_MODELLING_TABLE_FILENAME
    modelling_dataframe = load_modelling_table(modelling_table_path)
    print(f"Loaded modelling table with {len(modelling_dataframe):,} rows and {modelling_dataframe.shape[1]:,} columns.")

    if split_strategy == SPLIT_STRATEGY_TIME_HOLDOUT:
        split_result = build_time_holdout_split(
            modelling_dataframe=modelling_dataframe,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
        )

        write_split_summary_table(
            modelling_dataframe=modelling_dataframe,
            split_result=split_result,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
            output_filename=TIME_HOLDOUT_SPLIT_SUMMARY_FILENAME,
        )

        print(f"Split evidence written to: {(OUTPUT_TABLE_DIRECTORY / TIME_HOLDOUT_SPLIT_SUMMARY_FILENAME).resolve()}")

    elif split_strategy == SPLIT_STRATEGY_FUTURE_UNSEEN_COMPANIES:
        split_result = build_future_unseen_companies_split(
            modelling_dataframe=modelling_dataframe,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
        )

        write_split_summary_table(
            modelling_dataframe=modelling_dataframe,
            split_result=split_result,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
            output_filename=FUTURE_UNSEEN_COMPANIES_SPLIT_SUMMARY_FILENAME,
        )

        print(
            "Note: future unseen companies split restricts the test set to companies not present in training years."
        )
        print(
            f"Split evidence written to: {(OUTPUT_TABLE_DIRECTORY / FUTURE_UNSEEN_COMPANIES_SPLIT_SUMMARY_FILENAME).resolve()}"
        )

    else:
        raise ValueError(
            f"Unsupported split strategy: {split_strategy}. "
            f"Use '{SPLIT_STRATEGY_TIME_HOLDOUT}' or '{SPLIT_STRATEGY_FUTURE_UNSEEN_COMPANIES}'."
        )

    print(f"Training rows: {split_result['train_row_count']:,}")
    print(f"Testing rows:  {split_result['test_row_count']:,}")
    print(f"Training unique companies: {split_result['train_unique_company_count']:,}")
    print(f"Testing unique companies:  {split_result['test_unique_company_count']:,}")
    print(f"Overlapping companies:     {split_result['overlapping_company_count']:,}")
    print(f"Feature column count:      {len(split_result['feature_column_names']):,}")

    return split_result


# Run both splits to generate evidence tables without training a model.
def main() -> None:
    print("Generating train-test split evidence tables for both evaluation splits...")

    load_and_split_modelling_data(split_strategy=SPLIT_STRATEGY_TIME_HOLDOUT)
    load_and_split_modelling_data(split_strategy=SPLIT_STRATEGY_FUTURE_UNSEEN_COMPANIES)

    print("Split evidence generation complete.")


if __name__ == "__main__":
    main()

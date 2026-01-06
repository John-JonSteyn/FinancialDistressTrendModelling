"""
Loads the engineered feature table and produces a modelling-ready dataset by
dropping features that exceed a predefined missingness threshold.
"""

from __future__ import annotations
import pathlib
import pandas

# -------------------------
# Configuration constants
# -------------------------

PROCESSED_DATA_DIRECTORY = pathlib.Path("data/processed")
OUTPUT_TABLE_DIRECTORY = pathlib.Path("outputs/tables")

INPUT_FEATURES_FILENAME = "features.parquet"
OUTPUT_MODELLING_FEATURES_FILENAME = "modelling_features.parquet"

SELECTED_FEATURE_LIST_FILENAME = "selected_feature_list.csv"
DROPPED_FEATURE_LIST_FILENAME = "dropped_feature_list.csv"
FEATURE_MISSINGNESS_WITH_DECISION_FILENAME = "feature_missingness_with_decision.csv"

IDENTIFIER_COLUMN_NAMES = {"cik", "period", "period_datetime", "fiscal_year", "latest_filed"}

MAX_MISSINGNESS_FRACTION = 0.70

# Load the engineered feature table from parquet.
def load_engineered_features(features_parquet_path: pathlib.Path) -> pandas.DataFrame:
    if not features_parquet_path.exists():
        raise FileNotFoundError(f"Engineered features not found: {features_parquet_path.resolve()}")

    return pandas.read_parquet(features_parquet_path)


# Identify candidate feature columns by excluding identifier columns.
def get_candidate_feature_column_names(features_dataframe: pandas.DataFrame) -> list[str]:
    candidate_column_names: list[str] = []
    for column_name in features_dataframe.columns:
        if column_name not in IDENTIFIER_COLUMN_NAMES:
            candidate_column_names.append(column_name)
    return candidate_column_names


# Compute missingness statistics for each candidate feature column.
def compute_feature_missingness_statistics(
    features_dataframe: pandas.DataFrame,
    candidate_feature_column_names: list[str],
) -> pandas.DataFrame:
    total_row_count = int(len(features_dataframe))

    missing_value_counts_series = features_dataframe[candidate_feature_column_names].isna().sum()
    missingness_dataframe = missing_value_counts_series.reset_index()
    missingness_dataframe.columns = ["feature_name", "missing_value_count"]

    missingness_dataframe["total_row_count"] = total_row_count
    missingness_dataframe["missingness_fraction"] = (
        missingness_dataframe["missing_value_count"] / missingness_dataframe["total_row_count"]
    )

    missingness_dataframe = missingness_dataframe.sort_values(
        by=["missingness_fraction", "feature_name"],
        ascending=[False, True],
    )

    return missingness_dataframe


# Apply the missingness threshold to decide which features to keep.
def decide_features_by_missingness_threshold(
    missingness_dataframe: pandas.DataFrame,
    maximum_missingness_fraction: float,
) -> pandas.DataFrame:
    decision_dataframe = missingness_dataframe.copy()
    decision_dataframe["selection_decision"] = decision_dataframe["missingness_fraction"].apply(
        lambda fraction_value: "keep" if float(fraction_value) <= maximum_missingness_fraction else "drop"
    )
    return decision_dataframe


# Build a modelling dataframe retaining identifiers and only selected feature columns.
def build_modelling_features_dataframe(
    features_dataframe: pandas.DataFrame,
    decision_dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    selected_feature_names = decision_dataframe.loc[
        decision_dataframe["selection_decision"] == "keep",
        "feature_name",
    ].tolist()

    identifier_column_names_present = [
        column_name for column_name in features_dataframe.columns if column_name in IDENTIFIER_COLUMN_NAMES
    ]

    modelling_column_names = identifier_column_names_present + selected_feature_names
    modelling_dataframe = features_dataframe[modelling_column_names].copy()

    return modelling_dataframe


# Write the selection evidence tables to CSV.
def write_selection_evidence_tables(
    decision_dataframe: pandas.DataFrame,
    output_table_directory: pathlib.Path,
) -> None:
    output_table_directory.mkdir(parents=True, exist_ok=True)

    decision_dataframe.to_csv(
        output_table_directory / FEATURE_MISSINGNESS_WITH_DECISION_FILENAME,
        index=False,
    )

    selected_features_dataframe = decision_dataframe.loc[
        decision_dataframe["selection_decision"] == "keep",
        ["feature_name", "missingness_fraction", "missing_value_count", "total_row_count"],
    ].sort_values(by=["feature_name"], ascending=[True])

    dropped_features_dataframe = decision_dataframe.loc[
        decision_dataframe["selection_decision"] == "drop",
        ["feature_name", "missingness_fraction", "missing_value_count", "total_row_count"],
    ].sort_values(by=["missingness_fraction", "feature_name"], ascending=[False, True])

    selected_features_dataframe.to_csv(
        output_table_directory / SELECTED_FEATURE_LIST_FILENAME,
        index=False,
    )
    dropped_features_dataframe.to_csv(
        output_table_directory / DROPPED_FEATURE_LIST_FILENAME,
        index=False,
    )


# Orchestrate feature selection from engineered features to modelling-ready output.
def main() -> None:
    print("Starting feature selection for modelling...")
    print(f"Maximum missingness fraction threshold: {MAX_MISSINGNESS_FRACTION:.2f}")

    input_features_path = PROCESSED_DATA_DIRECTORY / INPUT_FEATURES_FILENAME
    output_modelling_features_path = PROCESSED_DATA_DIRECTORY / OUTPUT_MODELLING_FEATURES_FILENAME

    PROCESSED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    features_dataframe = load_engineered_features(input_features_path)
    print(f"Loaded engineered features with {len(features_dataframe):,} rows and {features_dataframe.shape[1]:,} columns.")

    candidate_feature_column_names = get_candidate_feature_column_names(features_dataframe)
    print(f"Candidate feature column count: {len(candidate_feature_column_names):,}")

    missingness_dataframe = compute_feature_missingness_statistics(
        features_dataframe=features_dataframe,
        candidate_feature_column_names=candidate_feature_column_names,
    )

    decision_dataframe = decide_features_by_missingness_threshold(
        missingness_dataframe=missingness_dataframe,
        maximum_missingness_fraction=MAX_MISSINGNESS_FRACTION,
    )

    kept_feature_count = int((decision_dataframe["selection_decision"] == "keep").sum())
    dropped_feature_count = int((decision_dataframe["selection_decision"] == "drop").sum())
    print(f"Selected feature count: {kept_feature_count:,}")
    print(f"Dropped feature count: {dropped_feature_count:,}")

    modelling_dataframe = build_modelling_features_dataframe(
        features_dataframe=features_dataframe,
        decision_dataframe=decision_dataframe,
    )
    print(f"Modelling dataset columns: {modelling_dataframe.shape[1]:,}")

    modelling_dataframe.to_parquet(
        output_modelling_features_path,
        engine="pyarrow",
        index=False,
    )
    print(f"Modelling features written to: {output_modelling_features_path.resolve()}")

    write_selection_evidence_tables(
        decision_dataframe=decision_dataframe,
        output_table_directory=OUTPUT_TABLE_DIRECTORY,
    )
    print("Selection evidence tables written.")


if __name__ == "__main__":
    main()

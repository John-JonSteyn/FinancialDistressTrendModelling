"""
Builds a longitudinal company-period panel from parsed SEC Financial Statement Data
Sets (FSDS) parquet files. The output contains one row per company per fiscal reporting
period, with selected financial statement tags pivoted into columns.
"""

from __future__ import annotations

import pathlib
import pandas

from fsds_tag_shortlist import get_default_fsds_tag_shortlist_set

# -------------------------
# Configuration constants
# -------------------------

INTERIM_PARQUET_ROOT_DIRECTORY = pathlib.Path("data/interim/fsds_parquet")
PROCESSED_DATA_DIRECTORY = pathlib.Path("data/processed")
OUTPUT_TABLE_DIRECTORY = pathlib.Path("outputs/tables")

PANEL_PARQUET_FILENAME = "panel.parquet"
MISSINGNESS_TABLE_FILENAME = "panel_missingness_by_column.csv"
DUPLICATE_KEYS_TABLE_FILENAME = "panel_duplicate_key_counts.csv"
ROW_COUNTS_BY_YEAR_FILENAME = "panel_row_counts_by_year.csv"

ALLOWED_FORMS = {"10-K", "10-Q"}

# Discover available quarter parquet directories.
def discover_parquet_quarter_directories(
    parquet_root_directory: pathlib.Path,
) -> list[pathlib.Path]:
    if not parquet_root_directory.exists():
        raise FileNotFoundError(
            f"Parquet root directory not found: {parquet_root_directory.resolve()}"
        )

    quarter_directories: list[pathlib.Path] = []
    for child_path in parquet_root_directory.iterdir():
        if child_path.is_dir():
            quarter_directories.append(child_path)

    return sorted(quarter_directories, key=lambda path_item: path_item.name)


# Load submission data for a given quarter.
def load_submission_parquet(quarter_parquet_directory: pathlib.Path) -> pandas.DataFrame:
    submission_parquet_path = quarter_parquet_directory / "sub.parquet"
    if not submission_parquet_path.exists():
        raise FileNotFoundError(f"Missing sub.parquet in {quarter_parquet_directory}")
    return pandas.read_parquet(submission_parquet_path)


# Load numeric data for a given quarter from parquet parts.
def load_numeric_parquet_parts(quarter_parquet_directory: pathlib.Path) -> pandas.DataFrame:
    numeric_parts_directory = quarter_parquet_directory / "num_parts"
    if not numeric_parts_directory.exists():
        raise FileNotFoundError(
            f"Missing num_parts directory in {quarter_parquet_directory}"
        )
    return pandas.read_parquet(numeric_parts_directory)


# Filter submission data to valid 10-K and 10-Q filings only (excluding amended forms).
def filter_valid_submissions(submission_dataframe: pandas.DataFrame) -> pandas.DataFrame:
    filtered_dataframe = submission_dataframe[
        submission_dataframe["form"].isin(ALLOWED_FORMS)
    ].copy()
    return filtered_dataframe


# Filter numeric data to the selected FSDS tag shortlist.
def filter_numeric_to_tag_shortlist(
    numeric_dataframe: pandas.DataFrame,
    tag_shortlist_set: set[str],
) -> pandas.DataFrame:
    filtered_dataframe = numeric_dataframe[
        numeric_dataframe["tag"].isin(tag_shortlist_set)
    ].copy()
    return filtered_dataframe


# Resolve duplicate numeric facts by keeping the latest filed value.
def resolve_numeric_duplicates(
    merged_dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    merged_dataframe_sorted = merged_dataframe.sort_values(
        by=["cik", "period", "tag", "filed"],
        ascending=[True, True, True, True],
    )

    deduplicated_dataframe = (
        merged_dataframe_sorted.groupby(["cik", "period", "tag"], as_index=False)
        .tail(1)
        .copy()
    )

    return deduplicated_dataframe


# Pivot numeric facts into a wide company-period format while preserving the latest filing date.
def pivot_numeric_facts(
    deduplicated_dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    latest_filed_dataframe = (
        deduplicated_dataframe.groupby(["cik", "period"], as_index=False)["filed"]
        .max()
        .rename(columns={"filed": "latest_filed"})
    )

    pivoted_values_dataframe = deduplicated_dataframe.pivot_table(
        index=["cik", "period"],
        columns="tag",
        values="value",
        aggfunc="first",
    ).reset_index()

    pivoted_values_dataframe.columns.name = None

    merged_pivoted_dataframe = pivoted_values_dataframe.merge(
        latest_filed_dataframe,
        on=["cik", "period"],
        how="left",
    )

    return merged_pivoted_dataframe


# Deduplicate the full panel by keeping the latest filed row per (cik, period).
def deduplicate_panel_by_latest_filed(
    panel_dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    if "latest_filed" not in panel_dataframe.columns:
        raise ValueError("Panel dataframe is missing required column: latest_filed")

    panel_dataframe_sorted = panel_dataframe.sort_values(
        by=["cik", "period", "latest_filed"],
        ascending=[True, True, True],
    )

    deduplicated_panel_dataframe = panel_dataframe_sorted.drop_duplicates(
        subset=["cik", "period"],
        keep="last",
    ).copy()

    return deduplicated_panel_dataframe


# Compute missingness counts for each column in the panel.
def compute_panel_missingness(panel_dataframe: pandas.DataFrame) -> pandas.DataFrame:
    missingness_series = panel_dataframe.isna().sum()
    missingness_dataframe = missingness_series.reset_index()
    missingness_dataframe.columns = ["column_name", "missing_value_count"]
    return missingness_dataframe


# Compute duplicate key counts for (cik, period) combinations.
def compute_duplicate_key_counts(panel_dataframe: pandas.DataFrame) -> pandas.DataFrame:
    duplicate_counts_dataframe = (
        panel_dataframe.groupby(["cik", "period"])
        .size()
        .reset_index(name="row_count")
    )
    duplicates_only_dataframe = duplicate_counts_dataframe[
        duplicate_counts_dataframe["row_count"] > 1
    ]
    return duplicates_only_dataframe


# Compute panel row counts by fiscal year using explicit YYYYMMDD parsing.
def compute_row_counts_by_year(panel_dataframe: pandas.DataFrame) -> pandas.DataFrame:
    period_as_integer_series = pandas.to_numeric(panel_dataframe["period"], errors="coerce").astype("Int64")
    period_as_string_series = period_as_integer_series.astype("string")

    period_datetime_series = pandas.to_datetime(
        period_as_string_series,
        format="%Y%m%d",
        errors="coerce",
    )

    fiscal_year_series = period_datetime_series.dt.year

    year_counts_dataframe = (
        pandas.DataFrame({"fiscal_year": fiscal_year_series})
        .dropna()
        .groupby("fiscal_year")
        .size()
        .reset_index(name="row_count")
        .sort_values("fiscal_year")
    )

    return year_counts_dataframe


# Build the company-period panel across all available quarters.
def main() -> None:
    PROCESSED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    tag_shortlist_set = get_default_fsds_tag_shortlist_set()
    quarter_directories = discover_parquet_quarter_directories(
        INTERIM_PARQUET_ROOT_DIRECTORY
    )

    panel_rows: list[pandas.DataFrame] = []

    for quarter_directory in quarter_directories:
        print(f"Processing quarter: {quarter_directory.name}")

        submission_dataframe = load_submission_parquet(quarter_directory)
        numeric_dataframe = load_numeric_parquet_parts(quarter_directory)

        submission_dataframe = filter_valid_submissions(submission_dataframe)
        numeric_dataframe = filter_numeric_to_tag_shortlist(
            numeric_dataframe,
            tag_shortlist_set,
        )

        merged_dataframe = numeric_dataframe.merge(
            submission_dataframe[["adsh", "cik", "period", "filed"]],
            on="adsh",
            how="inner",
        )

        deduplicated_dataframe = resolve_numeric_duplicates(merged_dataframe)
        pivoted_dataframe = pivot_numeric_facts(deduplicated_dataframe)

        panel_rows.append(pivoted_dataframe)

    full_panel_dataframe = pandas.concat(panel_rows, ignore_index=True)
    full_panel_dataframe = deduplicate_panel_by_latest_filed(full_panel_dataframe)

    panel_parquet_path = PROCESSED_DATA_DIRECTORY / PANEL_PARQUET_FILENAME
    full_panel_dataframe.to_parquet(
        panel_parquet_path,
        engine="pyarrow",
        index=False,
    )

    missingness_dataframe = compute_panel_missingness(full_panel_dataframe)
    duplicate_keys_dataframe = compute_duplicate_key_counts(full_panel_dataframe)
    row_counts_by_year_dataframe = compute_row_counts_by_year(full_panel_dataframe)

    missingness_dataframe.to_csv(
        OUTPUT_TABLE_DIRECTORY / MISSINGNESS_TABLE_FILENAME,
        index=False,
    )
    duplicate_keys_dataframe.to_csv(
        OUTPUT_TABLE_DIRECTORY / DUPLICATE_KEYS_TABLE_FILENAME,
        index=False,
    )
    row_counts_by_year_dataframe.to_csv(
        OUTPUT_TABLE_DIRECTORY / ROW_COUNTS_BY_YEAR_FILENAME,
        index=False,
    )

    print(f"Panel written to: {panel_parquet_path.resolve()}")


if __name__ == "__main__":
    main()

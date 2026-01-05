"""
FSDS parsing and normalisation script.

This script converts unpacked SEC Financial Statement Data Sets (FSDS) quarterly text files
into parquet for efficient downstream processing.

Input (expected):
- data/raw/fsds_txt/<year>q<quarter>/sub.txt
- data/raw/fsds_txt/<year>q<quarter>/num.txt
- data/raw/fsds_txt/<year>q<quarter>/tag.txt

Output (created as needed):
- data/interim/fsds_parquet/<year>q<quarter>/sub.parquet
- data/interim/fsds_parquet/<year>q<quarter>/tag.parquet
- data/interim/fsds_parquet/<year>q<quarter>/num_parts/part-00000.parquet (chunked)
- outputs/logs/fsds_parse_summary.csv
"""

from __future__ import annotations

import datetime
import json
import pathlib
import re
from typing import Any, Iterable

import pandas
import pyarrow
import pyarrow.parquet


# ------------------------
# Configuration constants
# ------------------------
RAW_TEXT_ROOT_DIRECTORY = pathlib.Path("data/raw/fsds_txt")
INTERIM_PARQUET_ROOT_DIRECTORY = pathlib.Path("data/interim/fsds_parquet")
OUTPUT_LOG_DIRECTORY = pathlib.Path("outputs/logs")

QUARTER_DIRECTORY_NAME_PATTERN = re.compile(r"^\d{4}q[1-4]$")

SUBMISSION_TEXT_FILENAME = "sub.txt"
NUMERIC_TEXT_FILENAME = "num.txt"
TAG_TEXT_FILENAME = "tag.txt"

SUBMISSION_PARQUET_FILENAME = "sub.parquet"
TAG_PARQUET_FILENAME = "tag.parquet"
NUMERIC_PARQUET_PARTS_DIRECTORY_NAME = "num_parts"

PARSE_SUMMARY_CSV_FILENAME = "fsds_parse_summary.csv"

# Chunk size for num.txt (tune if needed). Larger chunks use more memory.
NUMERIC_TABLE_CHUNK_ROW_COUNT = 1_000_000

# Parquet compression (snappy is a strong default for speed and size).
PARQUET_COMPRESSION = "snappy"


# Discover quarter directories under the raw FSDS text root.
def discover_quarter_directories(raw_text_root_directory: pathlib.Path) -> list[pathlib.Path]:
    if not raw_text_root_directory.exists():
        raise FileNotFoundError(
            f"Raw FSDS text directory not found: {raw_text_root_directory.resolve()}"
        )

    quarter_directories: list[pathlib.Path] = []
    for child_path in raw_text_root_directory.iterdir():
        if child_path.is_dir() and QUARTER_DIRECTORY_NAME_PATTERN.match(child_path.name):
            quarter_directories.append(child_path)

    return sorted(quarter_directories, key=lambda path_item: path_item.name)


# Read a tab-delimited FSDS text file into a pandas DataFrame.
def read_tab_delimited_text_file(text_file_path: pathlib.Path) -> pandas.DataFrame:
    return pandas.read_csv(
        text_file_path,
        sep="\t",
        low_memory=False,
    )


# Read a tab-delimited FSDS text file as an iterator of pandas DataFrames (chunked).
def read_tab_delimited_text_file_in_chunks(
    text_file_path: pathlib.Path,
    chunk_row_count: int,
) -> Iterable[pandas.DataFrame]:
    return pandas.read_csv(
        text_file_path,
        sep="\t",
        low_memory=False,
        chunksize=chunk_row_count,
    )


# Write a pandas DataFrame to parquet using stable defaults.
def write_dataframe_to_parquet(
    dataframe: pandas.DataFrame,
    parquet_file_path: pathlib.Path,
) -> None:
    parquet_file_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(
        parquet_file_path,
        engine="pyarrow",
        index=False,
        compression=PARQUET_COMPRESSION,
    )


# Write numeric FSDS data to chunked parquet parts for memory safety.
def write_numeric_table_to_parquet_parts(
    numeric_text_file_path: pathlib.Path,
    numeric_parquet_parts_directory: pathlib.Path,
    chunk_row_count: int,
) -> dict[str, Any]:
    numeric_parquet_parts_directory.mkdir(parents=True, exist_ok=True)

    total_row_count = 0
    part_file_count = 0

    # Track missingness for key fields across the entire numeric table.
    key_column_names = ["adsh", "tag", "ddate", "qtrs", "uom", "value"]
    missing_value_counter_by_column: dict[str, int] = {column_name: 0 for column_name in key_column_names}

    for chunk_index, numeric_chunk_dataframe in enumerate(
        read_tab_delimited_text_file_in_chunks(numeric_text_file_path, chunk_row_count)
    ):
        if numeric_chunk_dataframe.empty:
            continue

        total_row_count += int(numeric_chunk_dataframe.shape[0])

        for column_name in key_column_names:
            if column_name in numeric_chunk_dataframe.columns:
                missing_value_counter_by_column[column_name] += int(
                    numeric_chunk_dataframe[column_name].isna().sum()
                )

        part_filename = f"part-{chunk_index:05d}.parquet"
        part_parquet_file_path = numeric_parquet_parts_directory / part_filename
        write_dataframe_to_parquet(numeric_chunk_dataframe, part_parquet_file_path)

        part_file_count += 1

    return {
        "numeric_row_count": total_row_count,
        "numeric_part_file_count": part_file_count,
        "numeric_missing_counts": missing_value_counter_by_column,
    }


# Compute a compact missingness summary for selected columns.
def compute_missingness_summary(
    dataframe: pandas.DataFrame,
    selected_column_names: list[str],
) -> dict[str, int]:
    missingness_summary: dict[str, int] = {}
    for column_name in selected_column_names:
        if column_name in dataframe.columns:
            missingness_summary[column_name] = int(dataframe[column_name].isna().sum())
        else:
            missingness_summary[column_name] = -1  # Column not present
    return missingness_summary


# Compute the most common form values from the submission table.
def compute_top_form_values(
    submission_dataframe: pandas.DataFrame,
    top_value_count: int = 10,
) -> dict[str, int]:
    if "form" not in submission_dataframe.columns:
        return {}

    form_value_counts = submission_dataframe["form"].value_counts(dropna=False).head(top_value_count)
    return {str(index_value): int(count_value) for index_value, count_value in form_value_counts.items()}


# Append a row to the in-memory parse summary list.
def append_parse_summary_row(
    parse_summary_rows: list[dict[str, Any]],
    summary_row: dict[str, Any],
) -> None:
    parse_summary_rows.append(summary_row)


# Write the parse summary rows to a CSV log file.
def write_parse_summary_csv(
    parse_summary_rows: list[dict[str, Any]],
    output_log_directory: pathlib.Path,
    output_csv_filename: str,
) -> pathlib.Path:
    output_log_directory.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_log_directory / output_csv_filename

    parse_summary_dataframe = pandas.DataFrame(parse_summary_rows)
    parse_summary_dataframe.to_csv(output_csv_path, index=False)

    return output_csv_path


# Parse a single quarter directory (sub, tag to single parquet; num to chunked parquet parts).
def parse_single_quarter_to_parquet(quarter_text_directory: pathlib.Path) -> dict[str, Any]:
    quarter_name = quarter_text_directory.name

    submission_text_file_path = quarter_text_directory / SUBMISSION_TEXT_FILENAME
    numeric_text_file_path = quarter_text_directory / NUMERIC_TEXT_FILENAME
    tag_text_file_path = quarter_text_directory / TAG_TEXT_FILENAME

    if not submission_text_file_path.exists():
        raise FileNotFoundError(f"Missing {SUBMISSION_TEXT_FILENAME} in {quarter_text_directory.resolve()}")
    if not numeric_text_file_path.exists():
        raise FileNotFoundError(f"Missing {NUMERIC_TEXT_FILENAME} in {quarter_text_directory.resolve()}")
    if not tag_text_file_path.exists():
        raise FileNotFoundError(f"Missing {TAG_TEXT_FILENAME} in {quarter_text_directory.resolve()}")

    quarter_output_directory = INTERIM_PARQUET_ROOT_DIRECTORY / quarter_name
    submission_parquet_file_path = quarter_output_directory / SUBMISSION_PARQUET_FILENAME
    tag_parquet_file_path = quarter_output_directory / TAG_PARQUET_FILENAME
    numeric_parquet_parts_directory = quarter_output_directory / NUMERIC_PARQUET_PARTS_DIRECTORY_NAME

    # Load and write sub.txt
    submission_dataframe = read_tab_delimited_text_file(submission_text_file_path)
    write_dataframe_to_parquet(submission_dataframe, submission_parquet_file_path)

    # Load and write tag.txt
    tag_dataframe = read_tab_delimited_text_file(tag_text_file_path)
    write_dataframe_to_parquet(tag_dataframe, tag_parquet_file_path)

    # Chunk-load and write num.txt
    numeric_write_statistics = write_numeric_table_to_parquet_parts(
        numeric_text_file_path=numeric_text_file_path,
        numeric_parquet_parts_directory=numeric_parquet_parts_directory,
        chunk_row_count=NUMERIC_TABLE_CHUNK_ROW_COUNT,
    )

    # Compute evidence summaries
    submission_missingness_summary = compute_missingness_summary(
        submission_dataframe,
        selected_column_names=["adsh", "cik", "form", "period", "filed"],
    )
    tag_missingness_summary = compute_missingness_summary(
        tag_dataframe,
        selected_column_names=["tag", "tlabel", "doc"],
    )
    top_form_values = compute_top_form_values(submission_dataframe)

    return {
        "quarter": quarter_name,
        "submission_row_count": int(submission_dataframe.shape[0]),
        "tag_row_count": int(tag_dataframe.shape[0]),
        "numeric_row_count": int(numeric_write_statistics["numeric_row_count"]),
        "numeric_part_file_count": int(numeric_write_statistics["numeric_part_file_count"]),
        "top_forms_json": json.dumps(top_form_values, ensure_ascii=False),
        "submission_missingness_json": json.dumps(submission_missingness_summary, ensure_ascii=False),
        "tag_missingness_json": json.dumps(tag_missingness_summary, ensure_ascii=False),
        "numeric_missingness_json": json.dumps(numeric_write_statistics["numeric_missing_counts"], ensure_ascii=False),
    }


# Parse all discovered quarters under the raw text root and write a summary CSV.
def main() -> None:
    run_date_string = datetime.date.today().isoformat()
    print(f"FSDS parse run date: {run_date_string}")
    print(f"Raw text root directory: {RAW_TEXT_ROOT_DIRECTORY.resolve()}")
    print(f"Interim parquet root directory: {INTERIM_PARQUET_ROOT_DIRECTORY.resolve()}")

    quarter_directories = discover_quarter_directories(RAW_TEXT_ROOT_DIRECTORY)
    if not quarter_directories:
        raise FileNotFoundError(
            f"No quarter directories found under: {RAW_TEXT_ROOT_DIRECTORY.resolve()}"
        )

    parse_summary_rows: list[dict[str, Any]] = []

    for quarter_text_directory in quarter_directories:
        quarter_name = quarter_text_directory.name
        print(f"Parsing quarter: {quarter_name}")

        try:
            quarter_summary = parse_single_quarter_to_parquet(quarter_text_directory)
            quarter_summary["run_date"] = run_date_string
            append_parse_summary_row(parse_summary_rows, quarter_summary)
            print(f"Completed quarter: {quarter_name}")

        except Exception as unexpected_error:
            print(f"Failed quarter: {quarter_name} ({unexpected_error})")
            append_parse_summary_row(
                parse_summary_rows,
                {
                    "run_date": run_date_string,
                    "quarter": quarter_name,
                    "error": str(unexpected_error),
                },
            )

    output_csv_path = write_parse_summary_csv(
        parse_summary_rows=parse_summary_rows,
        output_log_directory=OUTPUT_LOG_DIRECTORY,
        output_csv_filename=PARSE_SUMMARY_CSV_FILENAME,
    )

    print(f"Parse summary written to: {output_csv_path.resolve()}")


if __name__ == "__main__":
    main()

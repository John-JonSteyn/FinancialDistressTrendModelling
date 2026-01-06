"""
FSDS acquisition script for a fixed modelling horison.

This script downloads and unpacks the SEC Financial Statement Data Sets (FSDS)
for the period 2019 Q1 through 2024 Q4.

The date range is intentionally fixed to ensure reproducibility of the
financial distress trend modelling experiment. If the modelling horison
changes, a new script should be created rather than modifying this one.
"""

from __future__ import annotations

import pathlib
import requests
import time
import zipfile
from typing import Iterable


# ------------------------
# Fixed modelling horison 
# ------------------------
START_YEAR = 2019
START_QUARTER = 1
END_YEAR = 2024
END_QUARTER = 4


# Build the official SEC FSDS ZIP download URL for a given year and quarter.
def build_financial_statement_dataset_zip_url(year: int, quarter: int) -> str:
    return (
        "https://www.sec.gov/files/dera/data/financial-statement-data-sets/"
        f"{year}q{quarter}.zip"
    )


# Generate an inclusive sequence of (year, quarter) pairs.
def generate_quarter_sequence(
    start_year: int,
    start_quarter: int,
    end_year: int,
    end_quarter: int,
) -> Iterable[tuple[int, int]]:
    current_year = start_year
    current_quarter = start_quarter

    while (current_year < end_year) or (
        current_year == end_year and current_quarter <= end_quarter
    ):
        yield current_year, current_quarter

        current_quarter += 1
        if current_quarter == 5:
            current_quarter = 1
            current_year += 1


# Download a single FSDS ZIP file with retries, backoff, and atomic file writes.
def download_financial_statement_dataset_zip(
    file_url: str,
    output_zip_path: pathlib.Path,
    user_agent_string: str,
    maximum_attempts: int = 5,
) -> None:
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip_path = output_zip_path.with_suffix(output_zip_path.suffix + ".part")

    request_headers = {
        "User-Agent": user_agent_string,
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }

    for attempt_number in range(1, maximum_attempts + 1):
        try:
            with requests.get(
                file_url,
                headers=request_headers,
                stream=True,
                timeout=(30, 180),
            ) as response:

                if response.status_code == 404:
                    raise FileNotFoundError(
                        f"Dataset not published at URL: {file_url}"
                    )

                response.raise_for_status()

                with open(temporary_zip_path, "wb") as output_file:
                    for data_chunk in response.iter_content(chunk_size=1024 * 1024):
                        if data_chunk:
                            output_file.write(data_chunk)

            temporary_zip_path.replace(output_zip_path)
            return

        except FileNotFoundError:
            if temporary_zip_path.exists():
                temporary_zip_path.unlink()
            raise

        except requests.HTTPError as http_error:
            status_code = getattr(http_error.response, "status_code", None)
            if status_code in {429, 500, 502, 503, 504} and attempt_number < maximum_attempts:
                time.sleep(min(2 ** attempt_number, 30))
                continue

            if temporary_zip_path.exists():
                temporary_zip_path.unlink()
            raise

        except requests.RequestException:
            if attempt_number < maximum_attempts:
                time.sleep(min(2 ** attempt_number, 30))
                continue

            if temporary_zip_path.exists():
                temporary_zip_path.unlink()
            raise


# Unzip a downloaded FSDS archive into a deterministic quarter folder.
def unpack_financial_statement_dataset_zip(
    input_zip_path: pathlib.Path,
    output_quarter_directory: pathlib.Path,
) -> None:
    output_quarter_directory.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_zip_path, "r") as zip_file_handle:
        zip_file_handle.extractall(output_quarter_directory)


# Download and unpack FSDS quarterly ZIP files for the fixed modelling horison.
def main() -> None:
    output_zip_directory = pathlib.Path("data/raw/fsds_zip")
    output_text_root_directory = pathlib.Path("data/raw/fsds_txt")

    user_agent_string = (
        "FinancialDistressTrendModelling (academic research)"
    )

    for year, quarter in generate_quarter_sequence(
        START_YEAR,
        START_QUARTER,
        END_YEAR,
        END_QUARTER,
    ):
        zip_filename = f"{year}q{quarter}.zip"
        zip_file_url = build_financial_statement_dataset_zip_url(year, quarter)
        output_zip_path = output_zip_directory / zip_filename

        output_quarter_directory = output_text_root_directory / f"{year}q{quarter}"
        expected_sub_file_path = output_quarter_directory / "sub.txt"
        expected_num_file_path = output_quarter_directory / "num.txt"
        expected_tag_file_path = output_quarter_directory / "tag.txt"

        quarter_already_unpacked = (
            expected_sub_file_path.exists()
            and expected_num_file_path.exists()
            and expected_tag_file_path.exists()
        )

        if quarter_already_unpacked:
            print(f"Already unpacked: {year}Q{quarter}")
            continue

        if not output_zip_path.exists():
            print(f"Downloading: {zip_file_url}")
            try:
                download_financial_statement_dataset_zip(
                    file_url=zip_file_url,
                    output_zip_path=output_zip_path,
                    user_agent_string=user_agent_string,
                )
                print(f"Saved ZIP: {output_zip_path}")

            except FileNotFoundError:
                print(f"Quarter not available at URL, stopping at: {zip_filename}")
                break

            except Exception as unexpected_error:
                print(f"Failed download: {zip_filename} ({unexpected_error})")
                continue

            time.sleep(0.5)

        print(f"Unpacking to: {output_quarter_directory}")
        try:
            unpack_financial_statement_dataset_zip(
                input_zip_path=output_zip_path,
                output_quarter_directory=output_quarter_directory,
            )
        except Exception as unexpected_error:
            print(f"Failed unpack: {zip_filename} ({unexpected_error})")


if __name__ == "__main__":
    main()

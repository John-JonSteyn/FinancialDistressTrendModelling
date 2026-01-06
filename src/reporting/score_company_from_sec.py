"""
Fetches recent SEC XBRL "company facts" for a given company (CIK or ticker),
reconstructs the same base tag panel used in our pipeline, recomputes engineered
features (ratios and trends), and applies a saved trained model to produce:

- A distress probability for the most recent quarter available
- A score history table for the latest N quarters (default 4)
- Report artefacts (tables + figures), including a combined PNG
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from typing import Any

import joblib
import matplotlib.pyplot as matplotlib_plot
import numpy
import pandas
import requests

# -------------------------
# Configuration constants
# -------------------------

SEC_USER_AGENT_STRING = "FinancialDistressTrendModelling (academic research)"

SEC_COMPANY_FACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

DEFAULT_MODEL_PATH = pathlib.Path("outputs/models/gradient_boosting_time_holdout.joblib")
DEFAULT_SELECTED_FEATURE_LIST_PATH = pathlib.Path("outputs/tables/selected_feature_list.csv")

DEFAULT_SCORE_QUARTER_COUNT = 8
DEFAULT_FETCH_QUARTER_COUNT = 12

HTTP_TIMEOUT_SECONDS = 60
REQUEST_SLEEP_SECONDS = 0.25

MAXIMUM_RANDOM_COMPANY_ATTEMPTS = 25

PERIOD_DATE_FORMAT = "%Y%m%d"
ROLLING_WINDOW_QUARTERS = 4
SLOPE_WINDOW_QUARTERS = 4

POSITIVE_CLASS_THRESHOLD = 0.50

# Base FSDS tags engineered from (mirrors compute_financial_features.py).
BASE_TAG_COLUMN_NAMES = [
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

OUTPUT_FIGURE_DIRECTORY = pathlib.Path("outputs/figures")
OUTPUT_TABLE_DIRECTORY = pathlib.Path("outputs/tables")

DEFAULT_PERMUTATION_IMPORTANCE_ENABLED = True
DEFAULT_PERMUTATION_IMPORTANCE_SHUFFLE_REPEATS = 15
DEFAULT_TOP_IMPORTANCE_FEATURE_COUNT = 15
REPORT_DPI = 180


# -------------------------
# Result container
# -------------------------

@dataclass(frozen=True)
class CompanyScoreResult:
    cik: int
    company_name: str
    most_recent_period_end_date: str
    most_recent_distress_probability: float
    most_recent_distress_label: int
    score_history_table: pandas.DataFrame
    engineered_features_dataframe: pandas.DataFrame


# -------------------------
# SEC helpers
# -------------------------

def build_sec_request_headers() -> dict[str, str]:
    """Build SEC-friendly request headers using the fixed User-Agent."""
    return {
        "User-Agent": SEC_USER_AGENT_STRING,
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }


def get_json_with_polite_rate_limit(url: str) -> dict[str, Any]:
    """Request JSON and apply a small delay to respect SEC rate-limiting guidance."""
    request_headers = build_sec_request_headers()

    response: requests.Response | None = None
    try:
        response = requests.get(url, headers=request_headers, timeout=HTTP_TIMEOUT_SECONDS)
    finally:
        time.sleep(REQUEST_SLEEP_SECONDS)

    if response is None:
        raise requests.RequestException(f"No response returned for URL: {url}")

    if response.status_code == 404:
        raise FileNotFoundError(f"SEC endpoint returned 404 for URL: {url}")

    response.raise_for_status()
    return response.json()


def format_cik_as_padded_string(cik: int) -> str:
    """Convert a CIK integer into the SEC 10-digit zero-padded string format."""
    return str(int(cik)).zfill(10)


def resolve_cik_from_ticker(ticker: str) -> int:
    """Resolve a ticker symbol to a CIK using SEC company_tickers.json."""
    ticker_upper = ticker.strip().upper()
    if not ticker_upper:
        raise ValueError("Ticker must be a non-empty string.")

    company_tickers_payload = get_json_with_polite_rate_limit(SEC_COMPANY_TICKERS_URL)

    for _, record in company_tickers_payload.items():
        record_ticker = str(record.get("ticker", "")).upper()
        if record_ticker == ticker_upper:
            return int(record["cik_str"])

    raise ValueError(f"Ticker not found in SEC ticker mapping: {ticker_upper}")


def choose_random_cik_from_local_modelling_table(
    modelling_table_path: pathlib.Path,
    random_seed: int,
) -> int:
    """Choose a deterministic random CIK from local modelling_table.parquet."""
    if not modelling_table_path.exists():
        raise FileNotFoundError(f"Local modelling table not found: {modelling_table_path.resolve()}")

    modelling_table_dataframe = pandas.read_parquet(modelling_table_path, columns=["cik"])
    unique_cik_list = sorted(set(modelling_table_dataframe["cik"].dropna().astype(int).tolist()))
    if not unique_cik_list:
        raise ValueError("No CIK values found in local modelling table.")

    random_generator = random.Random(int(random_seed))
    return int(random_generator.choice(unique_cik_list))


def fetch_company_facts_json(cik: int) -> dict[str, Any]:
    """Fetch SEC company facts JSON for a given CIK."""
    cik_padded = format_cik_as_padded_string(cik)
    company_facts_url = SEC_COMPANY_FACTS_URL_TEMPLATE.format(cik_padded=cik_padded)
    return get_json_with_polite_rate_limit(company_facts_url)


def extract_company_name(company_facts_payload: dict[str, Any]) -> str:
    """Extract a readable company name from the SEC company facts payload."""
    company_name = company_facts_payload.get("entityName")
    return str(company_name) if company_name else "Unknown company name"


def is_supported_filing_record(record: dict[str, Any]) -> bool:
    """Return True for usable 10-Q / 10-K fact records with end/filed/value present."""
    form_value = str(record.get("form", "")).strip().upper()
    if form_value not in {"10-Q", "10-K"}:
        return False

    end_date_value = str(record.get("end", "")).strip()
    filed_date_value = str(record.get("filed", "")).strip()
    if not end_date_value or not filed_date_value:
        return False

    value = record.get("val")
    if value is None:
        return False

    return True


def convert_end_date_to_fsds_period_value(end_date_text: str) -> int:
    """Convert an SEC end date (YYYY-MM-DD) to an FSDS-style integer period (YYYYMMDD)."""
    cleaned_text = end_date_text.strip()
    if len(cleaned_text) != 10 or cleaned_text[4] != "-" or cleaned_text[7] != "-":
        raise ValueError(f"Unexpected end date format (expected YYYY-MM-DD): {end_date_text}")

    year_text, month_text, day_text = cleaned_text.split("-")
    return int(f"{year_text}{month_text}{day_text}")


def parse_period_to_datetime(period_series: pandas.Series) -> pandas.Series:
    """Parse FSDS period values (YYYYMMDD) into a datetime series."""
    period_as_integer_series = pandas.to_numeric(period_series, errors="coerce").astype("Int64")
    period_as_string_series = period_as_integer_series.astype("string")

    return pandas.to_datetime(
        period_as_string_series,
        format=PERIOD_DATE_FORMAT,
        errors="coerce",
    )


def build_base_tag_panel_from_company_facts(
    cik: int,
    company_facts_payload: dict[str, Any],
    base_tag_column_names: list[str],
    maximum_period_count: int,
) -> pandas.DataFrame:
    """
    Build a (cik, period) panel for base tags, keeping the latest filed value per (period, tag).
    """
    facts_root = company_facts_payload.get("facts", {})
    us_gaap_root = facts_root.get("us-gaap", {})

    extracted_rows: list[dict[str, Any]] = []

    for tag_name in base_tag_column_names:
        tag_object = us_gaap_root.get(tag_name)
        if not isinstance(tag_object, dict):
            continue

        units_object = tag_object.get("units", {})
        if not isinstance(units_object, dict):
            continue

        preferred_unit_key = "USD" if "USD" in units_object else next(iter(units_object.keys()), None)
        if not preferred_unit_key:
            continue

        unit_records = units_object.get(preferred_unit_key, [])
        if not isinstance(unit_records, list):
            continue

        for record in unit_records:
            if not isinstance(record, dict):
                continue
            if not is_supported_filing_record(record):
                continue

            end_date_text = str(record.get("end", "")).strip()
            filed_date_text = str(record.get("filed", "")).strip()

            try:
                period_value = convert_end_date_to_fsds_period_value(end_date_text)
            except ValueError:
                continue

            value = record.get("val")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            extracted_rows.append(
                {
                    "cik": int(cik),
                    "period": float(period_value),
                    "tag": str(tag_name),
                    "value": float(numeric_value),
                    "filed": str(filed_date_text),
                }
            )

    if not extracted_rows:
        return pandas.DataFrame(columns=["cik", "period"] + base_tag_column_names + ["latest_filed", "period_datetime"])

    extracted_dataframe = pandas.DataFrame(extracted_rows)

    extracted_dataframe.sort_values(
        by=["cik", "period", "tag", "filed"],
        ascending=[True, True, True, True],
        inplace=True,
    )

    latest_by_tag_period_dataframe = (
        extracted_dataframe.groupby(["cik", "period", "tag"], as_index=False)
        .tail(1)
        .copy()
    )

    pivoted_dataframe = latest_by_tag_period_dataframe.pivot_table(
        index=["cik", "period"],
        columns="tag",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivoted_dataframe.columns.name = None

    latest_filed_by_period_dataframe = (
        latest_by_tag_period_dataframe.groupby(["cik", "period"], as_index=False)["filed"]
        .max()
        .rename(columns={"filed": "latest_filed"})
    )

    panel_dataframe = pivoted_dataframe.merge(
        latest_filed_by_period_dataframe,
        on=["cik", "period"],
        how="left",
    )

    panel_dataframe["period_datetime"] = parse_period_to_datetime(panel_dataframe["period"])
    panel_dataframe = panel_dataframe.dropna(subset=["period_datetime"]).copy()
    panel_dataframe = panel_dataframe.sort_values(["cik", "period_datetime"], ascending=[True, True])

    panel_dataframe = panel_dataframe.tail(int(maximum_period_count)).reset_index(drop=True)
    return panel_dataframe


# -------------------------
# Feature engineering (mirrors training pipeline)
# -------------------------

def get_numeric_series_or_all_missing(dataframe: pandas.DataFrame, column_name: str) -> pandas.Series:
    """Return a numeric series for a column, or an all-missing numeric series if absent."""
    if column_name in dataframe.columns:
        return pandas.to_numeric(dataframe[column_name], errors="coerce")
    return pandas.Series(numpy.nan, index=dataframe.index, dtype="float64")


def compute_safe_ratio(numerator_series: pandas.Series, denominator_series: pandas.Series) -> pandas.Series:
    """Compute a ratio safely by avoiding divide-by-zero and non-finite values."""
    denominator_nonzero_series = denominator_series.replace(0, numpy.nan)
    ratio_series = numerator_series / denominator_nonzero_series
    return ratio_series.replace([numpy.inf, -numpy.inf], numpy.nan)


def compute_ratio_features(panel_dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """Compute ratio features from statement values."""
    features_dataframe = panel_dataframe.copy()

    assets_series = get_numeric_series_or_all_missing(features_dataframe, "Assets")
    liabilities_series = get_numeric_series_or_all_missing(features_dataframe, "Liabilities")
    equity_series = get_numeric_series_or_all_missing(features_dataframe, "StockholdersEquity")
    revenues_series = get_numeric_series_or_all_missing(features_dataframe, "Revenues")
    net_income_series = get_numeric_series_or_all_missing(features_dataframe, "NetIncomeLoss")
    operating_income_series = get_numeric_series_or_all_missing(features_dataframe, "OperatingIncomeLoss")
    operating_cash_flow_series = get_numeric_series_or_all_missing(features_dataframe, "NetCashProvidedByUsedInOperatingActivities")

    cash_series = get_numeric_series_or_all_missing(features_dataframe, "CashAndCashEquivalentsAtCarryingValue")
    cash_and_restricted_cash_series = get_numeric_series_or_all_missing(
        features_dataframe,
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    )

    debt_current_series = get_numeric_series_or_all_missing(features_dataframe, "DebtCurrent")
    long_term_debt_current_series = get_numeric_series_or_all_missing(features_dataframe, "LongTermDebtCurrent")
    long_term_debt_noncurrent_series = get_numeric_series_or_all_missing(features_dataframe, "LongTermDebtNoncurrent")
    interest_expense_series = get_numeric_series_or_all_missing(features_dataframe, "InterestExpense")

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

    return features_dataframe


def compute_change_features(features_dataframe: pandas.DataFrame, feature_column_names: list[str]) -> pandas.DataFrame:
    """Compute quarter-on-quarter and year-on-year changes for selected features."""
    output_dataframe = features_dataframe.copy()
    output_dataframe = output_dataframe.sort_values(["cik", "period_datetime"], ascending=[True, True])

    for feature_column_name in feature_column_names:
        quarter_on_quarter_change_column_name = f"{feature_column_name}__change_qoq"
        year_on_year_change_column_name = f"{feature_column_name}__change_yoy"

        output_dataframe[quarter_on_quarter_change_column_name] = (
            output_dataframe.groupby("cik")[feature_column_name].diff(1)
        )
        output_dataframe[year_on_year_change_column_name] = (
            output_dataframe.groupby("cik")[feature_column_name].diff(ROLLING_WINDOW_QUARTERS)
        )

    return output_dataframe


def compute_rolling_mean_features(
    features_dataframe: pandas.DataFrame,
    feature_column_names: list[str],
    rolling_window_quarters: int,
) -> pandas.DataFrame:
    """Compute rolling means for selected features over a fixed number of quarters."""
    output_dataframe = features_dataframe.copy()
    output_dataframe = output_dataframe.sort_values(["cik", "period_datetime"], ascending=[True, True])

    for feature_column_name in feature_column_names:
        rolling_mean_column_name = f"{feature_column_name}__rolling_mean_{int(rolling_window_quarters)}q"
        output_dataframe[rolling_mean_column_name] = (
            output_dataframe.groupby("cik")[feature_column_name]
            .rolling(window=int(rolling_window_quarters), min_periods=int(rolling_window_quarters))
            .mean()
            .reset_index(level=0, drop=True)
        )

    return output_dataframe


def compute_trailing_slope_over_window(values_array: numpy.ndarray) -> float:
    """Compute an OLS slope over the window using index positions as time."""
    if values_array.size == 0:
        return numpy.nan
    if numpy.any(numpy.isnan(values_array)):
        return numpy.nan

    time_index_array = numpy.arange(values_array.size, dtype=float)
    slope_value = numpy.polyfit(time_index_array, values_array.astype(float), deg=1)[0]
    return float(slope_value)


def compute_slope_features(
    features_dataframe: pandas.DataFrame,
    feature_column_names: list[str],
    slope_window_quarters: int,
) -> pandas.DataFrame:
    """Compute short-window slopes for selected features."""
    output_dataframe = features_dataframe.copy()
    output_dataframe = output_dataframe.sort_values(["cik", "period_datetime"], ascending=[True, True])

    for feature_column_name in feature_column_names:
        slope_column_name = f"{feature_column_name}__slope_{int(slope_window_quarters)}q"
        output_dataframe[slope_column_name] = (
            output_dataframe.groupby("cik")[feature_column_name]
            .rolling(window=int(slope_window_quarters), min_periods=int(slope_window_quarters))
            .apply(
                lambda rolling_series: compute_trailing_slope_over_window(rolling_series.to_numpy()),
                raw=False,
            )
            .reset_index(level=0, drop=True)
        )

    return output_dataframe


def load_selected_feature_name_list(selected_feature_list_path: pathlib.Path) -> list[str]:
    """Load the selected modelling features list produced by select_modelling_features.py."""
    if not selected_feature_list_path.exists():
        raise FileNotFoundError(f"Selected feature list not found: {selected_feature_list_path.resolve()}")

    selected_feature_dataframe = pandas.read_csv(selected_feature_list_path)
    if "feature_name" not in selected_feature_dataframe.columns:
        raise ValueError("Selected feature list CSV must contain a 'feature_name' column.")

    return selected_feature_dataframe["feature_name"].astype(str).tolist()


def align_features_to_model_expectations(
    engineered_features_dataframe: pandas.DataFrame,
    selected_feature_name_list: list[str],
) -> pandas.DataFrame:
    """Align engineered features to the exact feature list used during training."""
    aligned_dataframe = engineered_features_dataframe.copy()

    for feature_name in selected_feature_name_list:
        if feature_name not in aligned_dataframe.columns:
            aligned_dataframe[feature_name] = numpy.nan

    return aligned_dataframe[selected_feature_name_list].copy()


def score_recent_periods(
    trained_pipeline: Any,
    engineered_features_dataframe: pandas.DataFrame,
    selected_feature_name_list: list[str],
    score_quarter_count: int,
) -> pandas.DataFrame:
    """Score the most recent N periods using the trained model pipeline."""
    if "period_datetime" not in engineered_features_dataframe.columns:
        raise ValueError("Engineered features dataframe must include 'period_datetime'.")

    ordered_dataframe = engineered_features_dataframe.sort_values(["period_datetime"], ascending=[True]).copy()

    available_row_count = int(len(ordered_dataframe))
    if available_row_count == 0:
        raise ValueError("No engineered rows available to score after period parsing.")

    effective_score_quarter_count = min(int(score_quarter_count), available_row_count)
    recent_periods_dataframe = ordered_dataframe.tail(effective_score_quarter_count).copy()

    aligned_feature_matrix = align_features_to_model_expectations(
        engineered_features_dataframe=recent_periods_dataframe,
        selected_feature_name_list=selected_feature_name_list,
    )

    probability_array = trained_pipeline.predict_proba(aligned_feature_matrix)[:, 1]
    predicted_label_array = (probability_array >= POSITIVE_CLASS_THRESHOLD).astype(int)

    recent_periods_dataframe["distress_probability"] = probability_array.astype(float)
    recent_periods_dataframe["distress_label"] = predicted_label_array.astype(int)

    output_table = recent_periods_dataframe[
        ["period_datetime", "period", "latest_filed", "distress_probability", "distress_label"]
    ].copy()

    output_table["period_end_date"] = output_table["period_datetime"].dt.strftime("%Y-%m-%d")
    output_table = output_table.drop(columns=["period_datetime"])

    return output_table[["period_end_date", "period", "latest_filed", "distress_probability", "distress_label"]]


def engineer_features_for_company_base_panel(company_base_panel_dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """Engineer the full feature set for the company base panel."""
    features_with_ratios_dataframe = compute_ratio_features(company_base_panel_dataframe)

    base_feature_source_column_names = [
        column_name for column_name in BASE_TAG_COLUMN_NAMES if column_name in features_with_ratios_dataframe.columns
    ]
    ratio_feature_column_names = [
        column_name for column_name in features_with_ratios_dataframe.columns if str(column_name).startswith("ratio_")
    ]
    feature_source_column_names = base_feature_source_column_names + ratio_feature_column_names

    features_with_changes_dataframe = compute_change_features(
        features_dataframe=features_with_ratios_dataframe,
        feature_column_names=feature_source_column_names,
    )

    features_with_rolling_means_dataframe = compute_rolling_mean_features(
        features_dataframe=features_with_changes_dataframe,
        feature_column_names=feature_source_column_names,
        rolling_window_quarters=ROLLING_WINDOW_QUARTERS,
    )

    return compute_slope_features(
        features_dataframe=features_with_rolling_means_dataframe,
        feature_column_names=feature_source_column_names,
        slope_window_quarters=SLOPE_WINDOW_QUARTERS,
    )


# -------------------------
# Scoring
# -------------------------

def score_company_by_cik(
    cik: int,
    trained_pipeline: Any,
    selected_feature_name_list: list[str],
    fetch_quarter_count: int,
    score_quarter_count: int,
) -> CompanyScoreResult:
    """Score a company by CIK from SEC facts through to model prediction."""
    print(f"Fetching SEC company facts for CIK {cik}...")
    company_facts_payload = fetch_company_facts_json(cik)
    company_name = extract_company_name(company_facts_payload)
    print(f"Company: {company_name}")

    print("Building base tag panel from company facts...")
    base_panel_dataframe = build_base_tag_panel_from_company_facts(
        cik=int(cik),
        company_facts_payload=company_facts_payload,
        base_tag_column_names=BASE_TAG_COLUMN_NAMES,
        maximum_period_count=int(fetch_quarter_count),
    )

    if base_panel_dataframe.empty:
        raise ValueError("No usable 10-Q/10-K facts found for this company in the selected tags.")

    print(f"Base panel rows available: {len(base_panel_dataframe)}")
    print("Computing engineered features (ratios and trends)...")
    engineered_features_dataframe = engineer_features_for_company_base_panel(base_panel_dataframe)

    print("Scoring recent periods...")
    score_history_table = score_recent_periods(
        trained_pipeline=trained_pipeline,
        engineered_features_dataframe=engineered_features_dataframe,
        selected_feature_name_list=selected_feature_name_list,
        score_quarter_count=int(score_quarter_count),
    )

    most_recent_row = score_history_table.tail(1).iloc[0]
    most_recent_probability = float(most_recent_row["distress_probability"])
    most_recent_label = int(most_recent_row["distress_label"])
    most_recent_period_end_date = str(most_recent_row["period_end_date"])

    return CompanyScoreResult(
        cik=int(cik),
        company_name=str(company_name),
        most_recent_period_end_date=most_recent_period_end_date,
        most_recent_distress_probability=float(most_recent_probability),
        most_recent_distress_label=int(most_recent_label),
        score_history_table=score_history_table,
        engineered_features_dataframe=engineered_features_dataframe,
    )


# -------------------------
# Reporting artefacts
# -------------------------

def write_score_history_table(cik: int, score_history_table: pandas.DataFrame) -> pathlib.Path:
    """Write the score history table to CSV."""
    OUTPUT_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_TABLE_DIRECTORY / f"company_score_history_cik_{int(cik)}.csv"
    score_history_table.to_csv(output_path, index=False)
    return output_path


def write_score_summary_json(
    cik: int,
    company_name: str,
    model_path: pathlib.Path,
    score_history_table: pandas.DataFrame,
) -> pathlib.Path:
    """Write a small summary JSON for the most recent score."""
    OUTPUT_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    most_recent_row = score_history_table.tail(1).iloc[0]
    payload = {
        "company_name": str(company_name),
        "cik": int(cik),
        "model_path": model_path.as_posix(),
        "most_recent_period_end_date": str(most_recent_row["period_end_date"]),
        "most_recent_latest_filed": str(most_recent_row["latest_filed"]),
        "most_recent_distress_probability": float(most_recent_row["distress_probability"]),
        "most_recent_distress_label": int(most_recent_row["distress_label"]),
        "score_quarter_count": int(len(score_history_table)),
        "threshold": float(POSITIVE_CLASS_THRESHOLD),
    }

    output_path = OUTPUT_TABLE_DIRECTORY / f"company_score_summary_cik_{int(cik)}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def build_feature_coverage_table(
    cik: int,
    engineered_features_dataframe: pandas.DataFrame,
    selected_feature_name_list: list[str],
    score_quarter_count: int,
) -> pandas.DataFrame:
    """Build a table describing feature missingness across the scored rows."""
    recent_rows_dataframe = (
        engineered_features_dataframe.sort_values(["period_datetime"], ascending=[True])
        .tail(int(score_quarter_count))
        .copy()
    )

    aligned_feature_matrix = align_features_to_model_expectations(
        engineered_features_dataframe=recent_rows_dataframe,
        selected_feature_name_list=selected_feature_name_list,
    )

    missing_value_count_by_feature = aligned_feature_matrix.isna().sum()

    coverage_dataframe = missing_value_count_by_feature.reset_index()
    coverage_dataframe.columns = ["feature_name", "missing_value_count_in_scored_rows"]
    coverage_dataframe["scored_row_count"] = int(len(aligned_feature_matrix))
    coverage_dataframe["missing_fraction_in_scored_rows"] = (
        coverage_dataframe["missing_value_count_in_scored_rows"] / coverage_dataframe["scored_row_count"]
    )
    coverage_dataframe["cik"] = int(cik)

    coverage_dataframe = coverage_dataframe.sort_values(
        ["missing_fraction_in_scored_rows", "feature_name"],
        ascending=[False, True],
    )

    return coverage_dataframe


def write_feature_coverage_table(cik: int, feature_coverage_table: pandas.DataFrame) -> pathlib.Path:
    """Write feature coverage table to CSV."""
    OUTPUT_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_TABLE_DIRECTORY / f"company_feature_coverage_cik_{int(cik)}.csv"
    feature_coverage_table.to_csv(output_path, index=False)
    return output_path


def compute_permutation_importance_for_scored_rows(
    trained_pipeline: Any,
    engineered_features_dataframe: pandas.DataFrame,
    selected_feature_name_list: list[str],
    score_quarter_count: int,
    shuffle_repeats: int,
    random_seed: int,
) -> pandas.DataFrame:
    """
    Compute permutation importance for the scored rows as mean probability drop after shuffling each feature.
    """
    recent_rows_dataframe = (
        engineered_features_dataframe.sort_values(["period_datetime"], ascending=[True])
        .tail(int(score_quarter_count))
        .copy()
    )

    aligned_feature_matrix = align_features_to_model_expectations(
        engineered_features_dataframe=recent_rows_dataframe,
        selected_feature_name_list=selected_feature_name_list,
    )

    baseline_probability_array = trained_pipeline.predict_proba(aligned_feature_matrix)[:, 1]
    baseline_mean_probability = float(numpy.nanmean(baseline_probability_array))

    random_generator = numpy.random.default_rng(int(random_seed))
    importance_rows: list[dict[str, float]] = []

    for feature_name in selected_feature_name_list:
        if feature_name not in aligned_feature_matrix.columns:
            continue

        probability_drop_values: list[float] = []

        for _ in range(int(shuffle_repeats)):
            shuffled_matrix = aligned_feature_matrix.copy()

            column_values = shuffled_matrix[feature_name].to_numpy()
            shuffled_values = column_values.copy()
            random_generator.shuffle(shuffled_values)
            shuffled_matrix[feature_name] = shuffled_values

            shuffled_probability_array = trained_pipeline.predict_proba(shuffled_matrix)[:, 1]
            shuffled_mean_probability = float(numpy.nanmean(shuffled_probability_array))

            probability_drop_values.append(baseline_mean_probability - shuffled_mean_probability)

        importance_rows.append(
            {
                "feature_name": str(feature_name),
                "mean_probability_drop": float(numpy.mean(probability_drop_values)),
                "standard_deviation_probability_drop": float(numpy.std(probability_drop_values)),
            }
        )

    importance_dataframe = pandas.DataFrame(importance_rows).sort_values(
        "mean_probability_drop",
        ascending=False,
    )

    return importance_dataframe


def write_permutation_importance_table(cik: int, permutation_importance_table: pandas.DataFrame) -> pathlib.Path:
    """Write permutation importance table to CSV."""
    OUTPUT_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_TABLE_DIRECTORY / f"company_permutation_importance_cik_{int(cik)}.csv"
    permutation_importance_table.to_csv(output_path, index=False)
    return output_path


def write_score_history_figure(cik: int, company_name: str, score_history_table: pandas.DataFrame) -> pathlib.Path:
    """Write a score history line chart to PNG."""
    OUTPUT_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    figure_handle = matplotlib_plot.figure(figsize=(10, 4.5))
    axis_handle = figure_handle.add_subplot(1, 1, 1)

    x_values = pandas.to_datetime(score_history_table["period_end_date"], errors="coerce")
    y_values = score_history_table["distress_probability"].astype(float)

    axis_handle.plot(x_values, y_values, marker="o")
    axis_handle.set_ylim(0.0, 1.0)
    axis_handle.set_xlabel("Period end date")
    axis_handle.set_ylabel("Distress probability")
    axis_handle.set_title(f"Distress score history (CIK {int(cik)}) - {company_name}")
    axis_handle.grid(True)

    output_path = OUTPUT_FIGURE_DIRECTORY / f"company_score_history_cik_{int(cik)}.png"
    figure_handle.savefig(output_path, dpi=REPORT_DPI, bbox_inches="tight")
    matplotlib_plot.close(figure_handle)

    return output_path


def write_permutation_importance_figure(
    cik: int,
    company_name: str,
    permutation_importance_table: pandas.DataFrame,
    top_feature_count: int,
) -> pathlib.Path:
    """Write a permutation importance bar chart to PNG."""
    OUTPUT_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    top_table = permutation_importance_table.head(int(top_feature_count)).copy()
    top_table = top_table.sort_values("mean_probability_drop", ascending=True)

    figure_handle = matplotlib_plot.figure(figsize=(10, 6))
    axis_handle = figure_handle.add_subplot(1, 1, 1)

    axis_handle.barh(
        top_table["feature_name"].astype(str),
        top_table["mean_probability_drop"].astype(float),
    )
    axis_handle.set_xlabel("Mean probability drop after shuffling feature")
    axis_handle.set_ylabel("Feature name")
    axis_handle.set_title(f"Permutation importance (top {len(top_table)}) - CIK {int(cik)} - {company_name}")
    axis_handle.grid(True, axis="x")

    output_path = OUTPUT_FIGURE_DIRECTORY / f"company_permutation_importance_cik_{int(cik)}.png"
    figure_handle.savefig(output_path, dpi=REPORT_DPI, bbox_inches="tight")
    matplotlib_plot.close(figure_handle)

    return output_path


def shorten_feature_name_for_display(
    feature_name: str,
    maximum_character_count: int,
) -> str:
    cleaned_name = str(feature_name).replace("__", "_")

    if len(cleaned_name) <= maximum_character_count:
        return cleaned_name

    return cleaned_name[: maximum_character_count - 3] + "..."

def convert_feature_name_to_human_readable_label(feature_name: str) -> str:
    """
   Convert a pipeline feature name into a readable label for plots and reports.
    """
    raw_feature_name = str(feature_name)

    base_tag_label_by_name = {
        "Assets": "Total assets",
        "Liabilities": "Total liabilities",
        "StockholdersEquity": "Shareholders' equity",
        "Revenues": "Revenue",
        "NetIncomeLoss": "Net income (loss)",
        "OperatingIncomeLoss": "Operating income (loss)",
        "CashAndCashEquivalentsAtCarryingValue": "Cash and cash equivalents",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": "Cash and restricted cash",
        "NetCashProvidedByUsedInOperatingActivities": "Operating cash flow",
        "DebtCurrent": "Current debt",
        "LongTermDebtCurrent": "Long-term debt (current portion)",
        "LongTermDebtNoncurrent": "Long-term debt (non-current portion)",
        "InterestExpense": "Interest expense",
    }

    ratio_label_by_name = {
        "ratio_liabilities_to_assets": "Liabilities to assets",
        "ratio_equity_to_assets": "Equity to assets",
        "ratio_total_debt_to_assets": "Total debt to assets",
        "ratio_net_income_margin": "Net income margin",
        "ratio_operating_income_margin": "Operating margin",
        "ratio_operating_cash_flow_margin": "Operating cash flow margin",
        "ratio_cash_to_assets": "Cash to assets",
        "ratio_cash_and_restricted_cash_to_assets": "Cash and restricted cash to assets",
        "ratio_interest_coverage_operating_income": "Interest coverage (operating income)",
    }

    suffix_label_by_suffix = {
        "__change_qoq": "QoQ change",
        "__change_yoy": "YoY change",
        "__rolling_mean_4q": "4-quarter rolling mean",
        "__slope_4q": "4-quarter slope",
    }

    base_name = raw_feature_name
    transformation_label: str | None = None

    for suffix, candidate_label in suffix_label_by_suffix.items():
        if raw_feature_name.endswith(suffix):
            base_name = raw_feature_name[: -len(suffix)]
            transformation_label = candidate_label
            break

    if base_name in base_tag_label_by_name:
        base_label = base_tag_label_by_name[base_name]
    elif base_name in ratio_label_by_name:
        base_label = ratio_label_by_name[base_name]
    else:
        base_label = base_name.replace("ratio_", "").replace("__", "_").replace("_", " ").strip().title()

    if transformation_label is None:
        return base_label

    return f"{base_label} ({transformation_label})"


def truncate_label_for_plotting(label_text: str, maximum_character_count: int) -> str:
    """
   Hard truncate label text for plots while preserving identity.
    """
    cleaned_label_text = str(label_text).strip()
    if len(cleaned_label_text) <= maximum_character_count:
        return cleaned_label_text

    return cleaned_label_text[: maximum_character_count - 3] + "..."


def get_distress_colour(distress_probability: float) -> str:
    """
    Map distress probability to a semantic colour.

    Blue   : Very low risk
    Green  : Low to moderate risk
    Orange : Elevated risk
    Red    : High risk
    """
    if distress_probability < 0.25:
        return "tab:blue"
    if distress_probability < 0.50:
        return "tab:green"
    if distress_probability < 0.75:
        return "tab:orange"
    return "tab:red"


def write_combined_company_report_png(
    cik: int,
    company_name: str,
    model_path: pathlib.Path,
    score_history_table: pandas.DataFrame,
    feature_coverage_table: pandas.DataFrame,
    permutation_importance_table: pandas.DataFrame | None,
    top_importance_feature_count: int,
) -> pathlib.Path:
    OUTPUT_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    most_recent_row = score_history_table.tail(1).iloc[0]
    most_recent_probability = float(most_recent_row["distress_probability"])
    most_recent_period_end_date = str(most_recent_row["period_end_date"])
    most_recent_latest_filed = str(most_recent_row["latest_filed"])

    scored_row_count = int(feature_coverage_table["scored_row_count"].iloc[0])
    missing_feature_count = int((feature_coverage_table["missing_value_count_in_scored_rows"] > 0).sum())
    total_feature_count = int(len(feature_coverage_table))
    present_feature_count = int(total_feature_count - missing_feature_count)

    figure_handle = matplotlib_plot.figure(figsize=(12, 13))

    # Title block
    title_axis = figure_handle.add_axes([0.06, 0.90, 0.88, 0.08])
    title_axis.axis("off")
    title_axis.text(0.0, 0.78, "Financial distress scoring report", fontsize=16, fontweight="bold")
    title_axis.text(0.0, 0.46, f"Company: {company_name}", fontsize=12)
    title_axis.text(0.0, 0.18, f"CIK: {cik}", fontsize=12)
    title_axis.text(0.0, -0.10, f"Model: {model_path.as_posix()}", fontsize=9)

    # Summary block
    summary_axis = figure_handle.add_axes([0.06, 0.83, 0.88, 0.06])
    summary_axis.axis("off")
    summary_axis.text(0.0, 0.70, f"Most recent period end date: {most_recent_period_end_date}", fontsize=11)
    summary_axis.text(0.0, 0.35, f"Latest filed date (facts): {most_recent_latest_filed}", fontsize=11)
    summary_axis.text(
        0.0,
        0.00,
        f"Distress probability: {most_recent_probability:.4f} | Threshold: {POSITIVE_CLASS_THRESHOLD:.2f}",
        fontsize=12,
        fontweight="bold",
        color=get_distress_colour(most_recent_probability),
    )

    # Score history chart
    score_axis = figure_handle.add_axes([0.08, 0.55, 0.86, 0.25])
    period_end_date_series = pandas.to_datetime(score_history_table["period_end_date"], errors="coerce")
    distress_probability_series = score_history_table["distress_probability"].astype(float)

    line_colour = get_distress_colour(most_recent_probability)
    score_axis.plot(
        period_end_date_series,
        distress_probability_series,
        marker="o",
        color=line_colour,
        linewidth=2.0,
    )
    score_axis.scatter(
        period_end_date_series.iloc[-1],
        distress_probability_series.iloc[-1],
        color=line_colour,
        s=90,
        zorder=5,
    )
    score_axis.set_ylim(0.0, 1.0)
    score_axis.set_xlabel("Period end date")
    score_axis.set_ylabel("Distress probability")
    score_axis.set_title("Recent distress score history")
    score_axis.grid(True)
    score_axis.tick_params(axis="x", rotation=30)

    # Score history table
    table_axis = figure_handle.add_axes([0.08, 0.33, 0.86, 0.18])
    table_axis.axis("off")

    table_dataframe = score_history_table.copy()
    table_dataframe = table_dataframe[["period_end_date", "latest_filed", "distress_probability", "distress_label"]].copy()
    table_dataframe["distress_probability"] = table_dataframe["distress_probability"].astype(float).map(lambda value: f"{value:.4f}")

    table_handle = table_axis.table(
        cellText=table_dataframe.values.tolist(),
        colLabels=list(table_dataframe.columns),
        loc="center",
        cellLoc="center",
    )
    table_handle.auto_set_font_size(False)
    table_handle.set_fontsize(9)
    table_handle.scale(1.0, 1.35)

    # Coverage block
    coverage_axis = figure_handle.add_axes([0.06, 0.24, 0.88, 0.07])
    coverage_axis.axis("off")
    coverage_axis.text(
        0.0,
        0.55,
        (
            f"Feature coverage across the last {scored_row_count} periods: "
            f"{present_feature_count}/{total_feature_count} features present at least once"
        ),
        fontsize=11,
    )

    # Permutation importance bar chart
    if permutation_importance_table is not None and not permutation_importance_table.empty:
        importance_axis = figure_handle.add_axes([0.18, 0.05, 0.76, 0.17])

        top_importance_dataframe = permutation_importance_table.head(int(top_importance_feature_count)).copy()
        top_importance_dataframe = top_importance_dataframe.sort_values("mean_probability_drop", ascending=True)

        raw_feature_name_list = top_importance_dataframe["feature_name"].astype(str).tolist()

        human_readable_feature_label_list = [
            convert_feature_name_to_human_readable_label(feature_name)
            for feature_name in raw_feature_name_list
        ]

        plot_label_list = [
            truncate_label_for_plotting(label_text=label, maximum_character_count=52)
            for label in human_readable_feature_label_list
        ]

        mean_probability_drop_series = top_importance_dataframe["mean_probability_drop"].astype(float)

        importance_axis.barh(plot_label_list, mean_probability_drop_series)
        importance_axis.set_xlabel("Mean probability drop after shuffling feature")
        importance_axis.set_ylabel("Feature name")
        importance_axis.set_title("Permutation importance (top features)")
        importance_axis.grid(True, axis="x")
        importance_axis.tick_params(axis="y", labelsize=8, pad=2)

    output_path = OUTPUT_FIGURE_DIRECTORY / f"company_report_cik_{cik}.png"
    figure_handle.savefig(output_path, dpi=REPORT_DPI, bbox_inches="tight")
    matplotlib_plot.close(figure_handle)

    return output_path


# -------------------------
# Main
# -------------------------

def main() -> None:
    argument_parser = argparse.ArgumentParser(description="Score a company using the trained distress trend model.")
    argument_parser.add_argument("--cik", type=int, default=None, help="Company CIK as an integer (preferred).")
    argument_parser.add_argument("--ticker", type=str, default=None, help="Company ticker symbol (optional).")
    argument_parser.add_argument(
        "--local_modelling_table_path",
        type=str,
        default="data/processed/modelling_table.parquet",
        help="Path to local modelling_table.parquet used for deterministic random demo selection.",
    )
    argument_parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed used for deterministic demo selection.",
    )
    argument_parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the saved trained model pipeline (.joblib).",
    )
    argument_parser.add_argument(
        "--selected_feature_list_path",
        type=str,
        default=str(DEFAULT_SELECTED_FEATURE_LIST_PATH),
        help="Path to outputs/tables/selected_feature_list.csv.",
    )
    argument_parser.add_argument(
        "--fetch_quarter_count",
        type=int,
        default=DEFAULT_FETCH_QUARTER_COUNT,
        help="How many recent periods to fetch for feature computation (recommended: 8).",
    )
    argument_parser.add_argument(
        "--score_quarter_count",
        type=int,
        default=DEFAULT_SCORE_QUARTER_COUNT,
        help="How many most recent periods to score and print (recommended: 4).",
    )

    parsed_arguments = argument_parser.parse_args()

    print("Starting company scoring from SEC data...")

    model_path = pathlib.Path(parsed_arguments.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

    selected_feature_list_path = pathlib.Path(parsed_arguments.selected_feature_list_path)
    selected_feature_name_list = load_selected_feature_name_list(selected_feature_list_path)

    print(f"Loading trained model pipeline from: {model_path}")
    trained_pipeline = joblib.load(model_path)

    score_result: CompanyScoreResult | None = None

    if parsed_arguments.cik is not None:
        cik_value = int(parsed_arguments.cik)
        print(f"Using provided CIK: {cik_value}")

        score_result = score_company_by_cik(
            cik=cik_value,
            trained_pipeline=trained_pipeline,
            selected_feature_name_list=selected_feature_name_list,
            fetch_quarter_count=int(parsed_arguments.fetch_quarter_count),
            score_quarter_count=int(parsed_arguments.score_quarter_count),
        )

    elif parsed_arguments.ticker is not None:
        print(f"Resolving CIK from ticker: {parsed_arguments.ticker}")
        cik_value = resolve_cik_from_ticker(parsed_arguments.ticker)
        print(f"Resolved CIK: {cik_value}")

        score_result = score_company_by_cik(
            cik=cik_value,
            trained_pipeline=trained_pipeline,
            selected_feature_name_list=selected_feature_name_list,
            fetch_quarter_count=int(parsed_arguments.fetch_quarter_count),
            score_quarter_count=int(parsed_arguments.score_quarter_count),
        )

    else:
        print("No company specified. Selecting a random company from local modelling_table.parquet...")

        modelling_table_path = pathlib.Path(parsed_arguments.local_modelling_table_path)
        base_random_seed = int(parsed_arguments.random_seed)

        last_failure_message: str | None = None

        for attempt_number in range(1, MAXIMUM_RANDOM_COMPANY_ATTEMPTS + 1):
            attempt_seed = base_random_seed + (attempt_number - 1)
            candidate_cik_value = choose_random_cik_from_local_modelling_table(
                modelling_table_path=modelling_table_path,
                random_seed=int(attempt_seed),
            )

            try:
                score_result = score_company_by_cik(
                    cik=candidate_cik_value,
                    trained_pipeline=trained_pipeline,
                    selected_feature_name_list=selected_feature_name_list,
                    fetch_quarter_count=int(parsed_arguments.fetch_quarter_count),
                    score_quarter_count=int(parsed_arguments.score_quarter_count),
                )
                break

            except ValueError as value_error:
                last_failure_message = str(value_error)
                print(f"Rejected CIK {candidate_cik_value}: {last_failure_message}")
                print("Trying another company...")
                continue

            except (requests.RequestException, FileNotFoundError) as request_error:
                last_failure_message = str(request_error)
                print(f"Network or SEC endpoint error for CIK {candidate_cik_value}: {last_failure_message}")
                print("Trying another company...")
                continue

        if score_result is None:
            raise ValueError(
                "Unable to find a random company with usable SEC facts after "
                f"{MAXIMUM_RANDOM_COMPANY_ATTEMPTS} attempts. "
                "Provide a specific company using --ticker or --cik. "
                f"Last failure: {last_failure_message}"
            )

    if score_result is None:
        raise RuntimeError("Internal error: no score result produced.")

    print("")
    print("============================================================")
    print("Company distress score summary")
    print("============================================================")
    print(f"Company name: {score_result.company_name}")
    print(f"CIK:          {score_result.cik}")
    print(f"Model path:   {model_path.as_posix()}")
    print(f"Most recent period end date: {score_result.most_recent_period_end_date}")
    print(f"Distress probability:        {score_result.most_recent_distress_probability:.4f}")
    print(f"Distress label (threshold {POSITIVE_CLASS_THRESHOLD:.2f}): {score_result.most_recent_distress_label}")
    print("")

    print("Recent score history:")
    print(score_result.score_history_table.to_string(index=False))
    print("")

    print("Writing report artefacts (tables and figures)...")

    score_history_csv_path = write_score_history_table(
        cik=int(score_result.cik),
        score_history_table=score_result.score_history_table,
    )

    score_summary_json_path = write_score_summary_json(
        cik=int(score_result.cik),
        company_name=str(score_result.company_name),
        model_path=model_path,
        score_history_table=score_result.score_history_table,
    )

    feature_coverage_table = build_feature_coverage_table(
        cik=int(score_result.cik),
        engineered_features_dataframe=score_result.engineered_features_dataframe,
        selected_feature_name_list=selected_feature_name_list,
        score_quarter_count=int(parsed_arguments.score_quarter_count),
    )

    feature_coverage_csv_path = write_feature_coverage_table(
        cik=int(score_result.cik),
        feature_coverage_table=feature_coverage_table,
    )

    score_history_figure_path = write_score_history_figure(
        cik=int(score_result.cik),
        company_name=str(score_result.company_name),
        score_history_table=score_result.score_history_table,
    )

    permutation_importance_table: pandas.DataFrame | None = None
    permutation_importance_csv_path: pathlib.Path | None = None
    permutation_importance_figure_path: pathlib.Path | None = None

    if DEFAULT_PERMUTATION_IMPORTANCE_ENABLED:
        print("Computing permutation importance for scored rows...")
        permutation_importance_table = compute_permutation_importance_for_scored_rows(
            trained_pipeline=trained_pipeline,
            engineered_features_dataframe=score_result.engineered_features_dataframe,
            selected_feature_name_list=selected_feature_name_list,
            score_quarter_count=int(parsed_arguments.score_quarter_count),
            shuffle_repeats=int(DEFAULT_PERMUTATION_IMPORTANCE_SHUFFLE_REPEATS),
            random_seed=int(parsed_arguments.random_seed),
        )

        permutation_importance_csv_path = write_permutation_importance_table(
            cik=int(score_result.cik),
            permutation_importance_table=permutation_importance_table,
        )

        permutation_importance_figure_path = write_permutation_importance_figure(
            cik=int(score_result.cik),
            company_name=str(score_result.company_name),
            permutation_importance_table=permutation_importance_table,
            top_feature_count=int(DEFAULT_TOP_IMPORTANCE_FEATURE_COUNT),
        )

    combined_report_png_path = write_combined_company_report_png(
        cik=int(score_result.cik),
        company_name=str(score_result.company_name),
        model_path=model_path,
        score_history_table=score_result.score_history_table,
        feature_coverage_table=feature_coverage_table,
        permutation_importance_table=permutation_importance_table,
        top_importance_feature_count=int(DEFAULT_TOP_IMPORTANCE_FEATURE_COUNT),
    )

    print("Report artefacts written:")
    print(f"- Score history CSV: {score_history_csv_path.resolve()}")
    print(f"- Score summary JSON: {score_summary_json_path.resolve()}")
    print(f"- Feature coverage CSV: {feature_coverage_csv_path.resolve()}")
    print(f"- Score history figure: {score_history_figure_path.resolve()}")
    if permutation_importance_csv_path is not None:
        print(f"- Permutation importance CSV: {permutation_importance_csv_path.resolve()}")
    if permutation_importance_figure_path is not None:
        print(f"- Permutation importance figure: {permutation_importance_figure_path.resolve()}")
    print(f"- Combined report PNG: {combined_report_png_path.resolve()}")

    print("")
    print("Company scoring complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)

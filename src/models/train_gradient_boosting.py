"""
Gradient boosting training for FSDS longitudinal distress proxy modelling.

This script trains a non-linear gradient boosting model by running two
deterministic splits:

1) time_holdout: 2019-2022 train, 2023-2024 test
2) future_unseen_companies: 2019-2022 train, 2023-2024 test restricted
   to companies not present in training years
"""

from __future__ import annotations

import pathlib
import sys
import time

import joblib
import matplotlib.pyplot as matplotlib_plot
import numpy
import pandas
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


# -------------------------
# Configuration constants
# -------------------------

OUTPUT_MODEL_DIRECTORY = pathlib.Path("outputs/models")
OUTPUT_TABLE_DIRECTORY = pathlib.Path("outputs/tables")
OUTPUT_FIGURE_DIRECTORY = pathlib.Path("outputs/figures")

RANDOM_SEED = 42

POSITIVE_CLASS_THRESHOLD = 0.50
TOP_IMPORTANCE_COUNT = 50

SPLIT_STRATEGY_LIST = [
    "time_holdout",
    "future_unseen_companies",
]

MAXIMUM_ITERATIONS = 400
LEARNING_RATE = 0.06
MAXIMUM_LEAF_NODES = 31
MAXIMUM_TREE_DEPTH = None
MINIMUM_SAMPLES_PER_LEAF = 40
L2_REGULARISATION = 0.0
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.10

# Print a step message with elapsed time from a step start.
def log_step(step_message: str, step_start_time: float) -> float:
    current_time = time.time()
    elapsed_seconds = current_time - step_start_time
    print(f"{step_message} (elapsed: {elapsed_seconds:.1f}s)", flush=True)
    return time.time()


# Ensure the repository root is on sys.path so local imports work when running as a script.
def add_repository_root_to_python_path() -> None:
    current_file_path = pathlib.Path(__file__).resolve()
    repository_root_path = current_file_path.parents[2]
    if str(repository_root_path) not in sys.path:
        sys.path.insert(0, str(repository_root_path))


# Build a pipeline: median imputation + histogram gradient boosting classifier.
def build_gradient_boosting_pipeline() -> Pipeline:
    estimator = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=LEARNING_RATE,
        max_iter=MAXIMUM_ITERATIONS,
        max_leaf_nodes=MAXIMUM_LEAF_NODES,
        max_depth=MAXIMUM_TREE_DEPTH,
        min_samples_leaf=MINIMUM_SAMPLES_PER_LEAF,
        l2_regularization=L2_REGULARISATION,
        early_stopping=EARLY_STOPPING,
        validation_fraction=VALIDATION_FRACTION,
        random_state=RANDOM_SEED,
    )

    pipeline = Pipeline(
        steps=[
            ("median_imputer", SimpleImputer(strategy="median")),
            ("gradient_boosting", estimator),
        ]
    )

    return pipeline


# Compute standard evaluation metrics for binary classification.
def compute_classification_metrics(
    true_labels: pandas.Series,
    predicted_labels: numpy.ndarray,
    predicted_probabilities: numpy.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "roc_auc": float(roc_auc_score(true_labels, predicted_probabilities)),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision": float(precision_score(true_labels, predicted_labels, zero_division=0)),
        "recall": float(recall_score(true_labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(true_labels, predicted_labels, zero_division=0)),
    }
    return metrics


# Compute balanced sample weights from the training labels to address class imbalance.
def compute_balanced_sample_weights(training_label_series: pandas.Series) -> numpy.ndarray:
    label_value_counts = training_label_series.value_counts(dropna=False).to_dict()
    count_label_zero = float(label_value_counts.get(0, 0))
    count_label_one = float(label_value_counts.get(1, 0))

    total_count = count_label_zero + count_label_one
    if total_count <= 0:
        raise ValueError("Training label series is empty; cannot compute sample weights.")

    if count_label_zero <= 0 or count_label_one <= 0:
        return numpy.ones(len(training_label_series), dtype=float)

    weight_zero = total_count / (2.0 * count_label_zero)
    weight_one = total_count / (2.0 * count_label_one)

    sample_weights = training_label_series.map({0: weight_zero, 1: weight_one}).to_numpy(dtype=float)
    return sample_weights


# Write evidence tables for metrics, confusion matrix, and feature importances.
def write_evidence_tables(
    split_strategy: str,
    metrics: dict[str, float],
    confusion_matrix_array: numpy.ndarray,
    feature_column_names: list[str],
    feature_importance_array: numpy.ndarray,
) -> None:
    OUTPUT_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    metrics_dataframe = pandas.DataFrame(
        [{"metric_name": key, "metric_value": value} for key, value in metrics.items()]
    )
    metrics_dataframe.to_csv(
        OUTPUT_TABLE_DIRECTORY / f"gradient_boosting_metrics_{split_strategy}.csv",
        index=False,
    )

    confusion_matrix_dataframe = pandas.DataFrame(
        confusion_matrix_array,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    ).reset_index().rename(columns={"index": "row_label"})
    confusion_matrix_dataframe.to_csv(
        OUTPUT_TABLE_DIRECTORY / f"gradient_boosting_confusion_matrix_{split_strategy}.csv",
        index=False,
    )

    importance_dataframe = pandas.DataFrame(
        {
            "feature_name": feature_column_names,
            "feature_importance": feature_importance_array,
        }
    ).sort_values("feature_importance", ascending=False)

    top_importance_dataframe = importance_dataframe.head(TOP_IMPORTANCE_COUNT).copy()
    top_importance_dataframe.to_csv(
        OUTPUT_TABLE_DIRECTORY / f"gradient_boosting_feature_importance_{split_strategy}.csv",
        index=False,
    )


# Save a ROC curve figure for the test set evaluation.
def write_roc_curve_figure(
    split_strategy: str,
    true_labels: pandas.Series,
    predicted_probabilities: numpy.ndarray,
) -> None:
    OUTPUT_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    false_positive_rate_array, true_positive_rate_array, _ = roc_curve(true_labels, predicted_probabilities)

    figure_handle = matplotlib_plot.figure()
    axis_handle = figure_handle.add_subplot(1, 1, 1)
    axis_handle.plot(false_positive_rate_array, true_positive_rate_array)
    axis_handle.set_xlabel("False positive rate")
    axis_handle.set_ylabel("True positive rate")
    axis_handle.set_title(f"ROC curve (gradient boosting, {split_strategy})")
    axis_handle.grid(True)

    output_figure_path = OUTPUT_FIGURE_DIRECTORY / f"gradient_boosting_roc_curve_{split_strategy}.png"
    figure_handle.savefig(output_figure_path, dpi=160, bbox_inches="tight")
    matplotlib_plot.close(figure_handle)


# Train and evaluate gradient boosting for one split strategy and save outputs.
def train_and_evaluate_for_split(split_strategy: str) -> dict[str, object]:
    print("", flush=True)
    print("------------------------------------------------------------", flush=True)
    print(f"Training gradient boosting for split strategy: {split_strategy}", flush=True)
    print("------------------------------------------------------------", flush=True)

    step_start_time = time.time()

    print("Preparing imports for data split logic...", flush=True)
    from src.models.model_data_split import load_and_split_modelling_data
    step_start_time = log_step("Imported split module.", step_start_time)

    print("Creating train-test split and loading dataframes...", flush=True)
    split_result = load_and_split_modelling_data(split_strategy=split_strategy)
    step_start_time = log_step("Split data loaded.", step_start_time)

    training_features_dataframe = split_result["training_features"]
    training_label_series = split_result["training_labels"]
    testing_features_dataframe = split_result["testing_features"]
    testing_label_series = split_result["testing_labels"]
    feature_column_names = split_result["feature_column_names"]

    print(
        f"Training features shape: {training_features_dataframe.shape} | "
        f"Testing features shape: {testing_features_dataframe.shape}",
        flush=True,
    )
    print(
        f"Training label distribution: {training_label_series.value_counts(dropna=False).to_dict()}",
        flush=True,
    )
    print(
        f"Testing label distribution:  {testing_label_series.value_counts(dropna=False).to_dict()}",
        flush=True,
    )

    print("Computing balanced sample weights for training...", flush=True)
    training_sample_weights = compute_balanced_sample_weights(training_label_series)
    step_start_time = log_step("Sample weights computed.", step_start_time)

    print("Building pipeline (median imputation, histogram gradient boosting)...", flush=True)
    pipeline = build_gradient_boosting_pipeline()
    step_start_time = log_step("Pipeline built.", step_start_time)

    print("Fitting model on training data...", flush=True)
    pipeline.fit(training_features_dataframe, training_label_series, gradient_boosting__sample_weight=training_sample_weights)
    step_start_time = log_step("Model fit complete.", step_start_time)

    print("Predicting probabilities on the test set...", flush=True)
    predicted_probabilities = pipeline.predict_proba(testing_features_dataframe)[:, 1]
    predicted_labels = (predicted_probabilities >= POSITIVE_CLASS_THRESHOLD).astype(int)
    step_start_time = log_step("Predictions complete.", step_start_time)

    print("Computing metrics and confusion matrix...", flush=True)
    metrics = compute_classification_metrics(
        true_labels=testing_label_series,
        predicted_labels=predicted_labels,
        predicted_probabilities=predicted_probabilities,
    )
    confusion_matrix_array = confusion_matrix(testing_label_series, predicted_labels)
    step_start_time = log_step("Metrics computed.", step_start_time)

    print("Extracting feature importances...", flush=True)
    estimator: HistGradientBoostingClassifier = pipeline.named_steps["gradient_boosting"]

    if hasattr(estimator, "feature_importances_"):
        feature_importance_array = estimator.feature_importances_
    else:
        feature_importance_array = numpy.zeros(len(feature_column_names), dtype=float)

    step_start_time = log_step("Feature importance extracted.", step_start_time)

    OUTPUT_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_output_path = OUTPUT_MODEL_DIRECTORY / f"gradient_boosting_{split_strategy}.joblib"

    print(f"Saving model artefact to: {model_output_path}", flush=True)
    joblib.dump(pipeline, model_output_path)
    step_start_time = log_step("Model saved.", step_start_time)

    print("Writing evidence tables (metrics, confusion matrix, feature importance)...", flush=True)
    write_evidence_tables(
        split_strategy=split_strategy,
        metrics=metrics,
        confusion_matrix_array=confusion_matrix_array,
        feature_column_names=feature_column_names,
        feature_importance_array=feature_importance_array,
    )
    step_start_time = log_step("Evidence tables written.", step_start_time)

    print("Writing ROC curve figure...", flush=True)
    write_roc_curve_figure(
        split_strategy=split_strategy,
        true_labels=testing_label_series,
        predicted_probabilities=predicted_probabilities,
    )
    step_start_time = log_step("ROC curve figure written.", step_start_time)

    print("Completed split run.", flush=True)
    print(f"Test ROC-AUC:   {metrics['roc_auc']:.4f}", flush=True)
    print(f"Test F1:        {metrics['f1']:.4f}", flush=True)
    print(f"Test recall:    {metrics['recall']:.4f}", flush=True)
    print(f"Test precision: {metrics['precision']:.4f}", flush=True)

    return {
        "split_strategy": split_strategy,
        "metrics": metrics,
        "model_output_path": str(model_output_path),
        "test_row_count": int(len(testing_features_dataframe)),
    }


# Run training for both splits and write a combined metrics summary table.
def main() -> None:
    overall_start_time = time.time()

    print("Starting gradient boosting training for both evaluation splits...", flush=True)
    print(f"Split strategies: {SPLIT_STRATEGY_LIST}", flush=True)

    print("Adding repository root to Python path...", flush=True)
    add_repository_root_to_python_path()
    print("Repository root added.", flush=True)

    print("Ensuring output directories exist...", flush=True)
    OUTPUT_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print("Output directories ready.", flush=True)

    result_rows: list[dict[str, object]] = []

    for split_strategy in SPLIT_STRATEGY_LIST:
        split_result = train_and_evaluate_for_split(split_strategy=split_strategy)

        metrics = split_result["metrics"]
        result_rows.append(
            {
                "split_strategy": split_strategy,
                "test_row_count": split_result["test_row_count"],
                "roc_auc": metrics["roc_auc"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    print("Writing combined metrics summary table...", flush=True)
    combined_metrics_dataframe = pandas.DataFrame(result_rows)
    combined_metrics_path = OUTPUT_TABLE_DIRECTORY / "gradient_boosting_metrics_summary.csv"
    combined_metrics_dataframe.to_csv(combined_metrics_path, index=False)

    total_elapsed_seconds = time.time() - overall_start_time

    print("", flush=True)
    print("All gradient boosting runs complete.", flush=True)
    print(f"Combined metrics written to: {combined_metrics_path.resolve()}", flush=True)
    print(f"Total elapsed time: {total_elapsed_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()

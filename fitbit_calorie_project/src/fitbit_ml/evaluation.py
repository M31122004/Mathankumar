from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .modeling import RegressionResult


def ensure_output_dirs(base_dir: str | Path) -> dict[str, Path]:
    base = Path(base_dir)
    dirs = {
        "figures": base / "figures",
        "models": base / "models",
        "reports": base / "reports",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def save_regression_outputs(
    metrics: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_importance: pd.DataFrame,
    best_result: RegressionResult,
    output_dirs: dict[str, Path],
) -> None:
    metrics.to_csv(output_dirs["reports"] / "regression_metrics.csv", index=False)
    holdout.to_csv(output_dirs["reports"] / "holdout_predictions.csv", index=False)
    feature_importance.to_csv(output_dirs["reports"] / "feature_importance.csv", index=False)
    joblib.dump(best_result.pipeline, output_dirs["models"] / "best_regression_model.joblib")


def save_clustering_outputs(cluster_output: dict[str, object], output_dirs: dict[str, Path]) -> None:
    cluster_output["silhouette_scores"].to_csv(
        output_dirs["reports"] / "kmeans_silhouette_scores.csv", index=False
    )
    cluster_output["assignments"].to_csv(
        output_dirs["reports"] / "cluster_assignments.csv", index=False
    )
    cluster_output["cluster_summary"].to_csv(
        output_dirs["reports"] / "cluster_summary.csv", index=False
    )
    cluster_output["experience_mix"].to_csv(
        output_dirs["reports"] / "cluster_experience_mix.csv", index=False
    )
    cluster_output["workout_mix"].to_csv(
        output_dirs["reports"] / "cluster_workout_type_mix.csv", index=False
    )
    cluster_output["optional_scores"].to_csv(
        output_dirs["reports"] / "optional_clustering_scores.csv", index=False
    )


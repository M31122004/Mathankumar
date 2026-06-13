from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fitbit_ml.clustering import run_workout_clustering
from fitbit_ml.data import load_dataset
from fitbit_ml.evaluation import ensure_output_dirs, save_clustering_outputs, save_regression_outputs
from fitbit_ml.modeling import extract_feature_importance, train_and_compare_models
from fitbit_ml.visualization import save_cluster_plots, save_eda_plots, save_model_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fitbit regression and clustering pipeline.")
    parser.add_argument(
        "--data",
        default=ROOT / "data" / "processed" / "fitbit_workouts_sample.csv",
        help="Input Fitbit CSV path.",
    )
    parser.add_argument(
        "--outputs",
        default=ROOT / "outputs",
        help="Directory for reports, figures, and models.",
    )
    args = parser.parse_args()

    data = load_dataset(args.data)
    output_dirs = ensure_output_dirs(args.outputs)

    save_eda_plots(data, output_dirs["figures"])

    _, best_model, metrics, holdout = train_and_compare_models(data)
    feature_importance = extract_feature_importance(best_model.pipeline)
    save_regression_outputs(metrics, holdout, feature_importance, best_model, output_dirs)
    save_model_plots(metrics, holdout, output_dirs["figures"])

    cluster_output = run_workout_clustering(data)
    save_clustering_outputs(cluster_output, output_dirs)
    save_cluster_plots(cluster_output, output_dirs["figures"])

    print("\nRegression model comparison:")
    print(metrics.to_string(index=False))
    print(f"\nBest model: {best_model.name}")
    print(f"Best model R2: {best_model.r2:.4f}")
    print(f"\nBest KMeans clusters: {cluster_output['best_k']}")
    print(f"Silhouette score: {cluster_output['silhouette']:.4f}")
    print(f"\nOutputs saved under: {Path(args.outputs).resolve()}")


if __name__ == "__main__":
    main()


from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_eda_plots(data: pd.DataFrame, figure_dir: str | Path) -> None:
    figure_dir = Path(figure_dir)

    plt.figure(figsize=(8, 5))
    plt.hist(data["Calories_Burned"], bins=30, color="#287c78", edgecolor="white")
    plt.title("Calories Burned Distribution")
    plt.xlabel("Calories Burned")
    plt.ylabel("Workout Sessions")
    plt.tight_layout()
    plt.savefig(figure_dir / "calories_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    data.groupby("Workout_Type")["Calories_Burned"].mean().sort_values().plot(
        kind="barh", color="#c05a34"
    )
    plt.title("Average Calories Burned by Workout Type")
    plt.xlabel("Average Calories Burned")
    plt.tight_layout()
    plt.savefig(figure_dir / "calories_by_workout_type.png", dpi=150)
    plt.close()


def save_model_plots(metrics: pd.DataFrame, holdout: pd.DataFrame, figure_dir: str | Path) -> None:
    figure_dir = Path(figure_dir)

    plt.figure(figsize=(9, 5))
    ordered = metrics.sort_values("R2")
    plt.barh(ordered["Model"], ordered["R2"], color="#3568a8")
    plt.axvline(0.8, color="#c05a34", linestyle="--", linewidth=1, label="Target R2")
    plt.title("Regression Model Comparison")
    plt.xlabel("R2 Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "model_r2_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(holdout["Actual_Calories"], holdout["Predicted_Calories"], alpha=0.65, color="#287c78")
    min_val = min(holdout["Actual_Calories"].min(), holdout["Predicted_Calories"].min())
    max_val = max(holdout["Actual_Calories"].max(), holdout["Predicted_Calories"].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#333333", linewidth=1)
    plt.title("Actual vs Predicted Calories")
    plt.xlabel("Actual Calories")
    plt.ylabel("Predicted Calories")
    plt.tight_layout()
    plt.savefig(figure_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()


def save_cluster_plots(cluster_output: dict[str, object], figure_dir: str | Path) -> None:
    figure_dir = Path(figure_dir)
    assignments = cluster_output["assignments"]
    silhouette_scores = cluster_output["silhouette_scores"]

    plt.figure(figsize=(8, 5))
    plt.plot(silhouette_scores["Clusters"], silhouette_scores["Silhouette"], marker="o", color="#3568a8")
    plt.axhline(0.15, color="#c05a34", linestyle="--", linewidth=1, label="Acceptance threshold")
    plt.title("KMeans Silhouette Scores")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "kmeans_silhouette_scores.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        assignments["PCA1"],
        assignments["PCA2"],
        c=assignments["Cluster"],
        cmap="viridis",
        alpha=0.75,
    )
    plt.title("Workout Pattern Clusters in PCA Space")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(figure_dir / "pca_kmeans_clusters.png", dpi=150)
    plt.close()


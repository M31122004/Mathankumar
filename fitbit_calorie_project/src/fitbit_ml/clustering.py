from __future__ import annotations

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TARGET, cap_outliers_iqr


CLUSTER_NUMERIC_FEATURES = [
    "Max_BPM",
    "Avg_BPM",
    "Resting_BPM",
    "Session_Duration (hours)",
    "Water_Intake (liters)",
    "Workout_Frequency (days/week)",
    TARGET,
]


def build_cluster_preprocessor() -> ColumnTransformer:
    categorical_without_workout = [col for col in CATEGORICAL_FEATURES if col != "Workout_Type"]
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, CLUSTER_NUMERIC_FEATURES),
            ("cat", categorical_pipe, categorical_without_workout),
        ]
    )


def run_workout_clustering(data: pd.DataFrame, max_clusters: int = 6) -> dict[str, object]:
    clean = cap_outliers_iqr(data, CLUSTER_NUMERIC_FEATURES)
    cluster_input = clean.drop(columns=["Workout_Type"], errors="ignore")
    transformed = build_cluster_preprocessor().fit_transform(cluster_input)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(transformed)

    scores: list[dict[str, float | int]] = []
    best_model: KMeans | None = None
    best_score = -1.0
    best_k = 0

    for k in range(2, max_clusters + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(components)
        score = silhouette_score(components, labels)
        scores.append({"Clusters": k, "Silhouette": round(float(score), 4)})
        if score > best_score:
            best_score = float(score)
            best_model = model
            best_k = k

    if best_model is None:
        raise RuntimeError("KMeans failed to produce a clustering model.")

    labels = best_model.labels_
    assignments = clean.copy()
    assignments["Cluster"] = labels
    assignments["PCA1"] = components[:, 0]
    assignments["PCA2"] = components[:, 1] if components.shape[1] > 1 else 0

    summary = assignments.groupby("Cluster")[CLUSTER_NUMERIC_FEATURES].mean().round(2)
    summary["Cluster_Size"] = assignments["Cluster"].value_counts().sort_index().values

    experience_mix = (
        assignments.groupby(["Cluster", "Experience_Level"]).size().unstack(fill_value=0)
    )
    workout_mix = assignments.groupby(["Cluster", "Workout_Type"]).size().unstack(fill_value=0)

    optional_scores = compare_optional_clusterers(components)

    return {
        "pca": pca,
        "components": components,
        "labels": labels,
        "best_k": best_k,
        "silhouette": best_score,
        "silhouette_scores": pd.DataFrame(scores),
        "assignments": assignments,
        "cluster_summary": summary.reset_index(),
        "experience_mix": experience_mix.reset_index(),
        "workout_mix": workout_mix.reset_index(),
        "optional_scores": optional_scores,
    }


def compare_optional_clusterers(components) -> pd.DataFrame:
    comparisons: list[dict[str, float | str]] = []

    hierarchical = AgglomerativeClustering(n_clusters=3)
    h_labels = hierarchical.fit_predict(components)
    comparisons.append(
        {
            "Method": "Hierarchical Clustering",
            "Silhouette": round(float(silhouette_score(components, h_labels)), 4),
        }
    )

    dbscan = DBSCAN(eps=1.8, min_samples=12)
    d_labels = dbscan.fit_predict(components)
    valid_labels = set(d_labels) - {-1}
    if len(valid_labels) >= 2:
        comparisons.append(
            {
                "Method": "DBSCAN",
                "Silhouette": round(float(silhouette_score(components, d_labels)), 4),
            }
        )
    else:
        comparisons.append({"Method": "DBSCAN", "Silhouette": float("nan")})

    return pd.DataFrame(comparisons)

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fitbit_ml.clustering import run_workout_clustering
from fitbit_ml.data import generate_sample_dataset, load_dataset
from fitbit_ml.modeling import extract_feature_importance, train_and_compare_models


st.set_page_config(page_title="Fitbit ML Analytics", layout="wide")

st.title("Fitbit Calorie Prediction & Workout Clustering")

uploaded = st.sidebar.file_uploader("Upload Fitbit CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample data", value=uploaded is None)

if uploaded is not None and not use_sample:
    data_path = ROOT / "data" / "raw" / "uploaded_fitbit.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(uploaded.getvalue())
    data = load_dataset(data_path)
else:
    rows = st.sidebar.slider("Sample rows", min_value=300, max_value=2500, value=1200, step=100)
    data = generate_sample_dataset(rows=rows)

tabs = st.tabs(["EDA", "Regression", "Clustering"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Workout Sessions", f"{len(data):,}")
    c2.metric("Average Calories", f"{data['Calories_Burned'].mean():.0f}")
    c3.metric("Average Duration", f"{data['Session_Duration (hours)'].mean():.2f} hr")

    left, right = st.columns(2)
    left.subheader("Calories Distribution")
    left.bar_chart(data["Calories_Burned"].value_counts(bins=20).sort_index())

    right.subheader("Average Calories by Workout Type")
    right.bar_chart(data.groupby("Workout_Type")["Calories_Burned"].mean().sort_values())

    st.subheader("Preview")
    st.dataframe(data.head(30), use_container_width=True)

with tabs[1]:
    if st.button("Train Regression Models", type="primary"):
        with st.spinner("Training regression models..."):
            _, best, metrics, holdout = train_and_compare_models(data)
            importance = extract_feature_importance(best.pipeline)
        st.success(f"Best model: {best.name} | R2: {best.r2:.3f}")
        st.dataframe(metrics, use_container_width=True)
        st.scatter_chart(holdout, x="Actual_Calories", y="Predicted_Calories")
        if not importance.empty:
            st.subheader("Top Feature Importance")
            st.dataframe(importance.head(15), use_container_width=True)

with tabs[2]:
    if st.button("Run PCA + KMeans Clustering", type="primary"):
        with st.spinner("Finding workout behavior clusters..."):
            cluster_output = run_workout_clustering(data)
        st.success(
            f"Best clusters: {cluster_output['best_k']} | "
            f"Silhouette: {cluster_output['silhouette']:.3f}"
        )
        st.dataframe(cluster_output["cluster_summary"], use_container_width=True)
        st.scatter_chart(cluster_output["assignments"], x="PCA1", y="PCA2", color="Cluster")
        st.subheader("Experience Level Mix")
        st.dataframe(cluster_output["experience_mix"], use_container_width=True)


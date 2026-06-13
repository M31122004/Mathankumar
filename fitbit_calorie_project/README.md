# Fitbit Calorie Burn Prediction & Workout Pattern Clustering

This project builds a machine learning workflow for fitness analytics:

- Predict calories burned per workout session with supervised regression.
- Discover hidden workout behavior patterns with PCA and KMeans clustering.
- Export metrics, feature importance, cluster summaries, and visualizations.

The pipeline works with a real Fitbit-style CSV. If no dataset is available, it can generate a realistic sample dataset for practice and live evaluation demos.

## Project Structure

```text
fitbit_calorie_project/
|-- data/
|   |-- raw/                  # Place real Fitbit CSV here
|   `-- processed/            # Generated sample dataset
|-- outputs/
|   |-- figures/              # Saved EDA/model/cluster plots
|   |-- models/               # Saved best model pipeline
|   `-- reports/              # Metrics and cluster summaries
|-- scripts/
|   |-- generate_sample_data.py
|   `-- run_fitbit_ml.py
|-- src/
|   `-- fitbit_ml/
|       |-- __init__.py
|       |-- clustering.py
|       |-- data.py
|       |-- evaluation.py
|       |-- modeling.py
|       `-- visualization.py
|-- PROJECT_REPORT.md
|-- requirements.txt
`-- streamlit_app.py
```

## Setup

```bash
cd fitbit_calorie_project
pip install -r requirements.txt
```

## Run With Sample Data

```bash
python scripts/generate_sample_data.py
python scripts/run_fitbit_ml.py --data data/processed/fitbit_workouts_sample.csv
```

## Run With Real Data

Place your CSV in `data/raw/`, then run:

```bash
python scripts/run_fitbit_ml.py --data data/raw/your_fitbit_file.csv
```

Required columns:

`Age`, `Gender`, `Weight (kg)`, `Height (m)`, `BMI`, `Fat_Percentage`, `Max_BPM`, `Avg_BPM`, `Resting_BPM`, `Session_Duration (hours)`, `Workout_Type`, `Water_Intake (liters)`, `Workout_Frequency (days/week)`, `Experience_Level`, `Calories_Burned`

## Launch Dashboard

```bash
streamlit run streamlit_app.py
```

## Outputs

- Regression model comparison: `outputs/reports/regression_metrics.csv`
- Best-model feature importance: `outputs/reports/feature_importance.csv`
- Cluster summaries: `outputs/reports/cluster_summary.csv`
- PCA cluster assignments: `outputs/reports/cluster_assignments.csv`
- Saved model: `outputs/models/best_regression_model.joblib`
- Visualizations: `outputs/figures/`

## Acceptance Targets

- Regression: target `R2 >= 0.80`
- Clustering: target `Silhouette Score >= 0.15`


# Project Report: Fitbit Calorie Burn Prediction & Workout Pattern Clustering

## Problem Statement

Fitness applications need reliable calorie estimates and useful workout behavior insights. This project uses Fitbit-style workout data to predict calories burned and identify user workout patterns based on physiological, demographic, and workout-context features.

## Dataset

The expected dataset contains age, gender, body metrics, heart-rate measures, workout duration, workout type, hydration, workout frequency, experience level, and the target variable `Calories_Burned`.

If a real dataset is unavailable, `scripts/generate_sample_data.py` creates a realistic practice dataset with the same schema.

## Data Preprocessing

- Missing numeric values are filled with median values.
- Missing categorical values are filled with the most frequent category.
- Numeric outliers are capped with the IQR method.
- Categorical variables are one-hot encoded.
- Numeric features are standardized for distance-sensitive models, SVR, KNN, PCA, and clustering.

## Feature Engineering

The dataset already includes important derived and sensor features such as `BMI`, `Avg_BPM`, `Resting_BPM`, and `Session_Duration (hours)`. The modeling pipeline keeps the feature space interpretable and uses encoded categorical context such as workout type and experience level.

## Supervised Learning

Target variable: `Calories_Burned`

Models trained:

- Linear Regression
- Ridge Regression
- Lasso Regression
- KNN Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression
- XGBoost Regressor, if installed

Evaluation metrics:

- MAE
- RMSE
- R2 Score
- 5-fold cross-validation R2

The target acceptance goal is `R2 >= 0.80`.

## Unsupervised Learning

For clustering, the workflow does not use `Workout_Type` as an input label. This encourages the model to find hidden workout behavior patterns from intensity, duration, calories, hydration, frequency, and experience level.

Steps:

- Preprocess features
- Apply PCA to 2 components for interpretable cluster visualization
- Run KMeans for cluster counts from 2 to 6
- Select the best cluster count by Silhouette Score
- Compare optional Hierarchical Clustering and DBSCAN outputs

The target acceptance goal is `Silhouette Score >= 0.15`.

## Business Interpretation

Useful cluster interpretations usually follow these patterns:

- Higher average BPM and shorter duration often indicate high-intensity interval behavior.
- Longer duration and moderate BPM often indicate steady cardio or endurance sessions.
- Lower BPM and longer duration may represent low-intensity recovery or yoga-style activity.
- More frequent workouts and lower resting BPM can indicate advanced or conditioned users.

These insights support personalized coaching, nutrition planning, workout recommendations, and wearable-device product optimization.

## How to Reproduce

```bash
python scripts/generate_sample_data.py
python scripts/run_fitbit_ml.py --data data/processed/fitbit_workouts_sample.csv
```

Review the generated CSV reports under `outputs/reports/` and visualizations under `outputs/figures/`.

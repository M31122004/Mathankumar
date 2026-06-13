from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TARGET, cap_outliers_iqr


@dataclass
class RegressionResult:
    name: str
    pipeline: Pipeline
    mae: float
    rmse: float
    r2: float
    cv_r2_mean: float
    cv_r2_std: float


def build_preprocessor() -> ColumnTransformer:
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
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def get_regressors() -> dict[str, object]:
    models: dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000, random_state=RANDOM_STATE),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=9, weights="distance"),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVR": SVR(kernel="rbf", C=25, epsilon=0.12),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
        )
    except Exception:
        pass
    return models


def train_and_compare_models(data: pd.DataFrame) -> tuple[list[RegressionResult], RegressionResult, pd.DataFrame, pd.DataFrame]:
    clean = cap_outliers_iqr(data, NUMERIC_FEATURES + [TARGET])
    x = clean.drop(columns=[TARGET])
    y = clean[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: list[RegressionResult] = []

    for name, model in get_regressors().items():
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring="r2", n_jobs=None)
        results.append(
            RegressionResult(
                name=name,
                pipeline=pipeline,
                mae=mean_absolute_error(y_test, predictions),
                rmse=rmse,
                r2=r2_score(y_test, predictions),
                cv_r2_mean=float(np.mean(cv_scores)),
                cv_r2_std=float(np.std(cv_scores)),
            )
        )

    best = max(results, key=lambda item: item.r2)
    metrics = pd.DataFrame(
        [
            {
                "Model": result.name,
                "MAE": round(result.mae, 3),
                "RMSE": round(result.rmse, 3),
                "R2": round(result.r2, 4),
                "CV_R2_Mean": round(result.cv_r2_mean, 4),
                "CV_R2_Std": round(result.cv_r2_std, 4),
            }
            for result in results
        ]
    ).sort_values("R2", ascending=False)

    holdout = x_test.copy()
    holdout["Actual_Calories"] = y_test.values
    holdout["Predicted_Calories"] = best.pipeline.predict(x_test).round(1)
    return results, best, metrics, holdout


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    importance = pd.DataFrame({"Feature": feature_names, "Importance": values})
    return importance.sort_values("Importance", ascending=False).reset_index(drop=True)


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "Calories_Burned"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Age",
    "Weight (kg)",
    "Height (m)",
    "BMI",
    "Fat_Percentage",
    "Max_BPM",
    "Avg_BPM",
    "Resting_BPM",
    "Session_Duration (hours)",
    "Water_Intake (liters)",
    "Workout_Frequency (days/week)",
]

CATEGORICAL_FEATURES = ["Gender", "Workout_Type", "Experience_Level"]

REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return data[REQUIRED_COLUMNS].copy()


def cap_outliers_iqr(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    capped = data.copy()
    for column in columns:
        q1 = capped[column].quantile(0.25)
        q3 = capped[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        capped[column] = capped[column].clip(lower=lower, upper=upper)
    return capped


def generate_sample_dataset(rows: int = 1200, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    workout_types = np.array(["Cardio", "Strength", "HIIT", "Yoga"])
    experience_levels = np.array(["Beginner", "Intermediate", "Advanced"])
    genders = np.array(["Male", "Female"])

    workout = rng.choice(workout_types, size=rows, p=[0.34, 0.28, 0.22, 0.16])
    experience = rng.choice(experience_levels, size=rows, p=[0.36, 0.42, 0.22])
    gender = rng.choice(genders, size=rows, p=[0.52, 0.48])

    age = rng.integers(18, 61, size=rows)
    height = np.where(gender == "Male", rng.normal(1.75, 0.07, rows), rng.normal(1.63, 0.06, rows))
    height = np.clip(height, 1.45, 1.95)
    bmi = np.clip(rng.normal(24.8, 3.8, rows), 17.5, 36.0)
    weight = bmi * height**2
    fat_pct = np.clip(
        np.where(gender == "Male", rng.normal(21, 6, rows), rng.normal(29, 7, rows))
        + (bmi - 24.5) * 0.9,
        8,
        48,
    )

    exp_bonus = np.select(
        [experience == "Beginner", experience == "Intermediate", experience == "Advanced"],
        [-4, 2, 7],
    )
    workout_intensity = np.select(
        [workout == "Yoga", workout == "Strength", workout == "Cardio", workout == "HIIT"],
        [0.48, 0.66, 0.76, 0.9],
    )

    max_bpm = np.clip(206.9 - 0.67 * age + rng.normal(0, 7, rows), 145, 205)
    resting_bpm = np.clip(72 - exp_bonus * 0.6 + rng.normal(0, 6, rows), 48, 92)
    avg_bpm = np.clip(resting_bpm + (max_bpm - resting_bpm) * workout_intensity + rng.normal(0, 5, rows), 82, 188)

    duration = np.select(
        [workout == "Yoga", workout == "Strength", workout == "Cardio", workout == "HIIT"],
        [
            rng.normal(1.0, 0.24, rows),
            rng.normal(0.9, 0.22, rows),
            rng.normal(0.82, 0.25, rows),
            rng.normal(0.55, 0.16, rows),
        ],
    )
    duration = np.clip(duration, 0.25, 1.8)
    water = np.clip(0.35 + duration * workout_intensity * 1.7 + rng.normal(0, 0.25, rows), 0.25, 3.5)
    frequency = np.select(
        [experience == "Beginner", experience == "Intermediate", experience == "Advanced"],
        [rng.integers(1, 4, rows), rng.integers(3, 6, rows), rng.integers(4, 8, rows)],
    )

    gender_factor = np.where(gender == "Male", 1.05, 0.96)
    exp_factor = np.select(
        [experience == "Beginner", experience == "Intermediate", experience == "Advanced"],
        [0.92, 1.0, 1.08],
    )
    calories = (
        duration
        * weight
        * workout_intensity
        * (avg_bpm / 100)
        * 6.25
        * gender_factor
        * exp_factor
        + frequency * 8
        - resting_bpm * 0.55
        + rng.normal(0, 28, rows)
    )
    calories = np.clip(calories, 90, 1200)

    return pd.DataFrame(
        {
            "Age": age,
            "Gender": gender,
            "Weight (kg)": weight.round(1),
            "Height (m)": height.round(2),
            "BMI": bmi.round(1),
            "Fat_Percentage": fat_pct.round(1),
            "Max_BPM": max_bpm.round(0).astype(int),
            "Avg_BPM": avg_bpm.round(0).astype(int),
            "Resting_BPM": resting_bpm.round(0).astype(int),
            "Session_Duration (hours)": duration.round(2),
            "Workout_Type": workout,
            "Water_Intake (liters)": water.round(2),
            "Workout_Frequency (days/week)": frequency.astype(int),
            "Experience_Level": experience,
            "Calories_Burned": calories.round(0).astype(int),
        }
    )


def save_sample_dataset(path: str | Path, rows: int = 1200, seed: int = RANDOM_STATE) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_dataset(rows=rows, seed=seed).to_csv(output_path, index=False)
    return output_path


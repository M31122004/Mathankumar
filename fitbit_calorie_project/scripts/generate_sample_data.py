from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fitbit_ml.data import save_sample_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a realistic Fitbit workout sample dataset.")
    parser.add_argument("--rows", type=int, default=1200, help="Number of rows to generate.")
    parser.add_argument(
        "--output",
        default=ROOT / "data" / "processed" / "fitbit_workouts_sample.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    output_path = save_sample_dataset(args.output, rows=args.rows)
    print(f"Sample dataset saved to: {output_path}")


if __name__ == "__main__":
    main()


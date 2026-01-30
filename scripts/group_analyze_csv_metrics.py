"""Analyze and aggregate metrics from multiple versioned CSV files."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_metrics_across_versions(csv_dir_path: str, keep_only_last_row: bool = False):
    """Loads all 'metrics.csv' files from 'version_*' subdirectories, then calculates and prints
    the mean and standard deviation for each column. Also produces a formatted 'mean ± std' output
    for easy copy-pasting.

    Args:
        csv_dir_path: The path to the directory containing the
                            'version_N' subdirectories.
        keep_only_last_row: If True, only the last row of each CSV file will be considered.
    """
    base_path = Path(csv_dir_path)

    # 1. Validate the input path
    if not base_path.is_dir():
        print(f"Error: Directory not found at '{base_path}'", file=sys.stderr)
        sys.exit(1)

    # 2. Find all 'metrics.csv' files using a glob pattern
    metric_files = sorted(list(base_path.glob("version_*/metrics.csv")))

    if not metric_files:
        print(
            f"Error: No 'metrics.csv' files found in 'version_*' subdirectories "
            f"under '{base_path}'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(metric_files)} metric files to analyze.")
    for f in metric_files:
        print(f"  - {f}")

    try:
        # 3. Load each CSV into a pandas DataFrame and store them in a list
        all_dfs = [pd.read_csv(f) for f in metric_files]

        if keep_only_last_row:
            all_dfs = [df.tail(1).reset_index(drop=True) for df in all_dfs]

        # Assumption check: ensure all dataframes have the same structure
        first_shape = all_dfs[0].shape
        first_columns = all_dfs[0].columns
        for i, df in enumerate(all_dfs[1:], 1):
            if df.shape != first_shape or not df.columns.equals(first_columns):
                print(
                    f"Error: File '{metric_files[i]}' has a different structure "
                    "from the first file.",
                    file=sys.stderr,
                )
                sys.exit(1)

        # 4. Stack the DataFrames into a single 3D NumPy array
        stacked_data = np.stack([df.to_numpy() for df in all_dfs], axis=0)

        # 5. Calculate the mean and standard deviation across all files
        mean_data = np.mean(stacked_data, axis=0)
        std_data = np.std(stacked_data, axis=0)

        # 6. Convert the results back to DataFrames for nice printing
        mean_df = pd.DataFrame(mean_data, columns=all_dfs[0].columns)
        std_df = pd.DataFrame(std_data, columns=all_dfs[0].columns)

        # 7. Create a new DataFrame with the formatted "mean ± std" strings
        print("\n--- Aggregated Metrics (mean ± std) ---")
        formatted_df = pd.DataFrame(index=mean_df.index, columns=mean_df.columns)
        for col in mean_df.columns:
            formatted_df[col] = [
                f"{mean_val:.4f} ± {std_val:.4f}"
                for mean_val, std_val in zip(mean_df[col], std_df[col])
            ]
        # Using .to_string() ensures the full DataFrame is printed to the console
        print(formatted_df.to_string())

        # 8. Save all results to CSV files
        mean_output_path = base_path / "aggregated_metrics_mean.csv"
        std_output_path = base_path / "aggregated_metrics_std.csv"
        formatted_output_path = base_path / "aggregated_metrics_formatted.csv"

        mean_df.to_csv(mean_output_path, index=False)
        std_df.to_csv(std_output_path, index=False)
        formatted_df.to_csv(formatted_output_path, index=False)

        print(f"\nSuccessfully analyzed metrics. Results saved in '{base_path}':")
        print(f"  - Mean values: {mean_output_path.name}")
        print(f"  - Std dev values: {std_output_path.name}")
        print(f"  - Formatted values: {formatted_output_path.name}")

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate mean and standard deviation for 'metrics.csv' files "
        "across multiple 'version_N' directories."
    )
    parser.add_argument(
        "csv_directory",
        type=str,
        help="Path to the 'csv' directory containing 'version_N' subdirectories.",
    )
    parser.add_argument(
        "--keep_only_last_row",
        action="store_true",
        help="If set, only the last row of each CSV will be considered.",
    )
    args = parser.parse_args()

    analyze_metrics_across_versions(args.csv_directory, keep_only_last_row=args.keep_only_last_row)

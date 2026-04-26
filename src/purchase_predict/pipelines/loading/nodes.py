from pathlib import Path

import pandas as pd


def load_csv_files(input_path: str, pattern: str = "*.csv") -> pd.DataFrame:
    """Load and concatenate CSV files from a local file or directory."""
    path = Path(input_path)
    if path.is_file():
        return pd.read_csv(path)

    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {path} with pattern {pattern}")

    return pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

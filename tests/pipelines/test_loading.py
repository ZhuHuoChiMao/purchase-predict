import pandas as pd

from purchase_predict.pipelines.loading.nodes import load_csv_files


def test_load_csv_files_from_directory(tmp_path):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "part-1.csv", index=False)
    pd.DataFrame({"a": [2]}).to_csv(tmp_path / "part-2.csv", index=False)

    result = load_csv_files(str(tmp_path), "part-*.csv")

    assert result["a"].tolist() == [1, 2]

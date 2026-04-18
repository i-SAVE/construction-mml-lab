import pandas as pd

from src.data.quality import clean_training_frame


def test_clean_training_frame_removes_duplicates_and_high_missing_columns():
    df = pd.DataFrame(
        {
            "feature_ok": [1, 1, 2],
            "feature_bad": [None, None, 5],
            "SalePrice": [10, 10, 20],
        }
    )

    cleaned, summary = clean_training_frame(df, target="SalePrice", missing_threshold=0.49)

    assert summary.duplicates_removed == 1
    assert "feature_bad" in summary.dropped_columns
    assert "feature_bad" not in cleaned.columns

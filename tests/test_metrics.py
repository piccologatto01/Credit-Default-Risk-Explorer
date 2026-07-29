import numpy as np
import pandas as pd
import pytest

from src.risk_analytics.io import infer_id, infer_target
from src.risk_analytics.metrics import gini_from_auc, risk_deciles, threshold_at_recall


def test_gini_is_normalized_auc():
    assert gini_from_auc(0.5) == 0.0
    assert gini_from_auc(0.8) == pytest.approx(0.6)


def test_threshold_respects_minimum_recall():
    y_true = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.05, 0.10, 0.60, 0.55, 0.90])
    point = threshold_at_recall(y_true, scores, minimum_recall=1.0)
    assert point["recall"] == 1.0
    assert point["precision"] == pytest.approx(2 / 3)


def test_risk_deciles_are_sorted_from_high_risk():
    y_true = np.array([0] * 90 + [1] * 10)
    scores = np.linspace(0, 1, 100)
    result = risk_deciles(y_true, scores)
    assert result.iloc[0]["default_rate"] == 1.0
    assert result.iloc[0]["risk_decile"] == 1


def test_schema_inference_is_case_insensitive():
    train = pd.DataFrame({"Application_ID": [1, 2, 3], "Feature": [3, 4, 5], "TARGET": [0, 1, 0]})
    assert infer_target(train) == "TARGET"
    assert infer_id(train, None) == "Application_ID"

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def gini_from_auc(roc_auc: float) -> float:
    """Convert ROC-AUC to the normalized Gini coefficient."""
    return 2.0 * float(roc_auc) - 1.0


def threshold_at_recall(y_true: np.ndarray, scores: np.ndarray, minimum_recall: float = 0.70) -> dict[str, float]:
    """Highest-precision operating point that still reaches the requested recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    candidates = np.flatnonzero(recall[:-1] >= minimum_recall)
    if not len(candidates):
        index = int(np.argmax(recall[:-1]))
    else:
        candidate_precision = precision[candidates]
        index = int(candidates[np.argmax(candidate_precision)])
    return {
        "threshold": float(thresholds[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
    }


def curve_tables(y_true: np.ndarray, scores: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
    pr = pd.DataFrame(
        {
            "recall": recall[:-1],
            "precision": precision[:-1],
            "threshold": pr_thresholds,
        }
    )
    return roc, pr


def score_distribution(y_true: np.ndarray, scores: np.ndarray, bins: int = 20) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(scores, edges, right=False) - 1, 0, bins - 1)
    frame = pd.DataFrame({"bucket": bucket, "target": y_true})
    result = frame.groupby(["bucket", "target"], observed=True).size().rename("applications").reset_index()
    full = pd.MultiIndex.from_product([range(bins), [0, 1]], names=["bucket", "target"]).to_frame(index=False)
    result = full.merge(result, on=["bucket", "target"], how="left").fillna({"applications": 0})
    result["applications"] = result["applications"].astype(int)
    result["score_from"] = result["bucket"] / bins
    result["score_to"] = (result["bucket"] + 1) / bins
    return result


def risk_deciles(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"target": y_true, "score": scores}).sort_values("score", ascending=False)
    groups = min(10, len(frame))
    frame["risk_decile"] = pd.qcut(np.arange(len(frame)), q=groups, labels=range(1, groups + 1))
    result = frame.groupby("risk_decile", observed=True).agg(
        applications=("target", "size"),
        defaults=("target", "sum"),
        default_rate=("target", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    ).reset_index()
    total_defaults = max(int(frame["target"].sum()), 1)
    result["captured_defaults"] = result["defaults"].cumsum() / total_defaults
    result["risk_decile"] = result["risk_decile"].astype(int)
    return result


def missingness(frame: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    columns = [column for column in frame.columns if column not in excluded]
    result = pd.DataFrame(
        {
            "feature": columns,
            "missing": [int(frame[column].isna().sum()) for column in columns],
            "missing_rate": [float(frame[column].isna().mean()) for column in columns],
            "unique_values": [int(frame[column].nunique(dropna=True)) for column in columns],
            "dtype": [str(frame[column].dtype) for column in columns],
        }
    )
    return result.sort_values(["missing_rate", "feature"], ascending=[False, True])

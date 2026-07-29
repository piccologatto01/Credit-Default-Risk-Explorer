from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .io import DatasetBundle, load_dataset
from .metrics import (
    curve_tables,
    gini_from_auc,
    missingness,
    risk_deciles,
    score_distribution,
    threshold_at_recall,
)


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    return value


def select_features(bundle: DatasetBundle) -> tuple[list[str], list[str], list[str]]:
    excluded = {bundle.target}
    if bundle.id_column:
        excluded.add(bundle.id_column)
    candidates = [column for column in bundle.train.columns if column not in excluded]
    dropped: list[str] = []
    numeric: list[str] = []
    categorical: list[str] = []
    for column in candidates:
        series = bundle.train[column]
        unique = int(series.nunique(dropna=True))
        if unique <= 1:
            dropped.append(column)
        elif pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
            numeric.append(column)
        elif unique > max(100, int(len(series) * 0.20)):
            dropped.append(column)
        else:
            categorical.append(column)
    if not numeric and not categorical:
        raise ValueError("После проверки качества не осталось пригодных признаков")
    return numeric, categorical, dropped


def build_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical,
            )
        )
    preprocessing = ColumnTransformer(transformers, remainder="drop")
    classifier = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=500,
        random_state=42,
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def _clean_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame.reindex(columns=features).copy()
    numeric = result.select_dtypes(include=["number"]).columns
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan)
    return result


def coefficient_importance(model: Pipeline, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    preprocessing = model.named_steps["preprocessing"]
    names = list(preprocessing.get_feature_names_out())
    coefficients = model.named_steps["classifier"].coef_[0]
    original = []
    categorical_by_length = sorted(categorical, key=len, reverse=True)
    for name in names:
        if name.startswith("num__"):
            raw = name.removeprefix("num__")
            raw = raw.removeprefix("missingindicator_")
            original.append(raw)
            continue
        raw = name.removeprefix("cat__")
        feature = next((column for column in categorical_by_length if raw == column or raw.startswith(f"{column}_")), raw)
        original.append(feature)
    detail = pd.DataFrame({"feature": original, "coefficient": coefficients})
    rows = []
    for feature, group in detail.groupby("feature", observed=True):
        dominant = group.loc[group["coefficient"].abs().idxmax(), "coefficient"]
        rows.append(
            {
                "feature": feature,
                "importance": float(group["coefficient"].abs().sum()),
                "coefficient": float(dominant),
                "direction": "risk_up" if dominant > 0 else "risk_down",
                "odds_ratio": float(np.exp(np.clip(dominant, -20, 20))),
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def run(
    data_dir: Path,
    output_dir: Path,
    target_column: str | None = None,
    id_column: str | None = None,
    minimum_recall: float = 0.70,
    validation_size: float = 0.20,
) -> dict:
    bundle = load_dataset(data_dir, target_column, id_column)
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric, categorical, dropped = select_features(bundle)
    features = numeric + categorical
    X = _clean_features(bundle.train, features)
    y = bundle.train[bundle.target].to_numpy()
    if min(np.bincount(y)) < 2:
        raise ValueError("Для стратифицированной валидации нужно минимум два наблюдения каждого класса")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=validation_size,
        stratify=y,
        random_state=42,
    )
    model = build_model(numeric, categorical)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_valid)[:, 1]

    roc_auc = float(roc_auc_score(y_valid, scores))
    average_precision = float(average_precision_score(y_valid, scores))
    operating = threshold_at_recall(y_valid, scores, minimum_recall)
    predictions = (scores >= operating["threshold"]).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_valid, predictions, labels=[0, 1]).ravel()
    roc, pr = curve_tables(y_valid, scores)
    deciles = risk_deciles(y_valid, scores)
    distribution = score_distribution(y_valid, scores)
    importance = coefficient_importance(model, numeric, categorical)
    missing = missingness(bundle.train, {bundle.target, bundle.id_column} - {None})

    roc.to_csv(output_dir / "roc_curve.csv", index=False)
    pr.to_csv(output_dir / "pr_curve.csv", index=False)
    deciles.to_csv(output_dir / "risk_deciles.csv", index=False)
    distribution.to_csv(output_dir / "score_distribution.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    missing.to_csv(output_dir / "missingness.csv", index=False)

    prevalence = float(y.mean())
    summary = {
        "data_mode": "demo" if data_dir.name.casefold() == "demo" else "real",
        "train_file": bundle.train_path.name,
        "test_file": bundle.test_path.name if bundle.test_path else None,
        "target_column": bundle.target,
        "id_column": bundle.id_column,
        "applications": int(len(bundle.train)),
        "defaults": int(y.sum()),
        "default_rate": prevalence,
        "features_used": len(features),
        "numeric_features": len(numeric),
        "categorical_features": len(categorical),
        "features_dropped": len(dropped),
        "validation_applications": int(len(y_valid)),
        "roc_auc": roc_auc,
        "gini": gini_from_auc(roc_auc),
        "average_precision": average_precision,
        "baseline_average_precision": float(y_valid.mean()),
        "minimum_recall_requested": minimum_recall,
        "operating_threshold": operating["threshold"],
        "precision_at_recall": operating["precision"],
        "recall_at_threshold": operating["recall"],
        "approval_rate": float((scores < operating["threshold"]).mean()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "model": "LogisticRegression + balanced class weights",
        "validation": f"stratified holdout {validation_size:.0%}, random_state=42",
        "interpretation_note": "Коэффициенты показывают связь, но не доказывают причинность.",
        "dropped_features": dropped,
    }

    if bundle.test is not None:
        final_model = build_model(numeric, categorical)
        final_model.fit(X, y)
        test_scores = final_model.predict_proba(_clean_features(bundle.test, features))[:, 1]
        submission = pd.DataFrame({bundle.target: test_scores})
        if bundle.id_column and bundle.id_column in bundle.test:
            submission.insert(0, bundle.id_column, bundle.test[bundle.id_column].to_numpy())
        submission.to_csv(output_dir / "submission.csv", index=False)
        joblib.dump(final_model, output_dir / "credit_scoring_model.joblib")
        summary["submission_rows"] = int(len(submission))
    else:
        joblib.dump(model, output_dir / "credit_scoring_model.joblib")
        summary["submission_rows"] = None

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_value), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit default scoring pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--minimum-recall", type=float, default=0.70)
    parser.add_argument("--validation-size", type=float, default=0.20)
    args = parser.parse_args()
    if not 0 < args.minimum_recall <= 1:
        parser.error("--minimum-recall должен быть в диапазоне (0, 1]")
    if not 0 < args.validation_size < 1:
        parser.error("--validation-size должен быть в диапазоне (0, 1)")
    summary = run(
        args.data_dir,
        args.output_dir,
        args.target_column,
        args.id_column,
        args.minimum_recall,
        args.validation_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_value))


if __name__ == "__main__":
    main()

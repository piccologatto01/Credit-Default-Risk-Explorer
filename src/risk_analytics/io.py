from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TRAIN_NAMES = ("train.csv", "credit_train.csv", "application_train.csv", "training.csv")
TEST_NAMES = ("test.csv", "credit_test.csv", "application_test.csv", "testing.csv")
TARGET_NAMES = (
    "target",
    "default",
    "loan_default",
    "is_default",
    "credit_default",
    "seriousdlqin2yrs",
    "label",
)
ID_NAMES = ("id", "application_id", "request_id", "client_id", "sk_id_curr")


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame | None
    target: str
    id_column: str | None
    train_path: Path
    test_path: Path | None


def _find_named_file(data_dir: Path, names: tuple[str, ...]) -> Path | None:
    files = {path.name.casefold(): path for path in data_dir.glob("*.csv")}
    for name in names:
        if name.casefold() in files:
            return files[name.casefold()]
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"Файл {path} пуст")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _match_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {column.casefold(): column for column in columns}
    return next((lookup[name] for name in candidates if name in lookup), None)


def infer_target(train: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in train.columns:
            raise ValueError(f"Целевая колонка {explicit!r} не найдена")
        return explicit
    named = _match_column(list(train.columns), TARGET_NAMES)
    if named:
        return named
    binary = []
    for column in train.columns:
        values = pd.Series(train[column]).dropna().unique()
        if 1 < len(values) <= 2 and set(pd.to_numeric(values, errors="coerce")) == {0, 1}:
            binary.append(column)
    if len(binary) == 1:
        return binary[0]
    raise ValueError(
        "Не удалось однозначно определить target. Передайте --target-column; "
        f"бинарные кандидаты: {binary or 'нет'}"
    )


def infer_id(train: pd.DataFrame, test: pd.DataFrame | None, explicit: str | None = None) -> str | None:
    if explicit:
        if explicit not in train.columns:
            raise ValueError(f"ID-колонка {explicit!r} не найдена")
        return explicit
    named = _match_column(list(train.columns), ID_NAMES)
    if named:
        return named
    for column in train.columns:
        if train[column].is_unique and (test is None or column in test.columns):
            name = column.casefold()
            if name.endswith("id") or name.startswith("id_"):
                return column
    return None


def load_dataset(
    data_dir: Path,
    target_column: str | None = None,
    id_column: str | None = None,
) -> DatasetBundle:
    train_path = _find_named_file(data_dir, TRAIN_NAMES)
    if train_path is None:
        csv_files = sorted(data_dir.glob("*.csv"))
        if len(csv_files) == 1:
            train_path = csv_files[0]
        else:
            raise FileNotFoundError(
                f"Не найден train CSV в {data_dir}. Ожидается один из: {', '.join(TRAIN_NAMES)}"
            )
    test_path = _find_named_file(data_dir, TEST_NAMES)
    if test_path == train_path:
        test_path = None
    train = _read_csv(train_path)
    test = _read_csv(test_path) if test_path else None
    target = infer_target(train, target_column)
    identifier = infer_id(train, test, id_column)

    target_values = pd.to_numeric(train[target], errors="coerce")
    if target_values.isna().any() or set(target_values.unique()) != {0, 1}:
        raise ValueError(f"Target {target!r} должен содержать только 0/1 без пропусков")
    train[target] = target_values.astype("int8")
    return DatasetBundle(train, test, target, identifier, train_path, test_path)

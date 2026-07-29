from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "demo"


def make_applications(size: int, start_id: int, rng: np.random.Generator, with_target: bool) -> pd.DataFrame:
    age = np.clip(rng.normal(41, 12, size), 18, 78).round()
    monthly_income = np.exp(rng.normal(np.log(75_000), 0.65, size)).round(-2)
    credit_history_months = np.clip((age - 18) * 7 + rng.normal(0, 35, size), 0, 500).round()
    active_loans = np.clip(rng.poisson(1.8, size), 0, 9)
    overdue_30d = np.clip(rng.poisson(0.20, size), 0, 5)
    overdue_90d = np.clip(rng.poisson(0.07, size), 0, 4)
    debt_to_income = np.clip(rng.beta(2.2, 4.5, size) + overdue_30d * 0.05, 0, 1.5)
    requested_amount = np.exp(rng.normal(np.log(180_000), 0.75, size)).round(-3)
    employment_months = np.clip(rng.gamma(2.3, 30, size), 0, 420).round()
    inquiries_6m = np.clip(rng.poisson(0.9, size), 0, 8)
    region = rng.choice(["Москва", "Санкт-Петербург", "Центр", "Юг", "Урал", "Сибирь", "Дальний Восток"], size, p=[.17, .08, .22, .16, .14, .16, .07])
    employment_type = rng.choice(["Наём", "Самозанятый", "ИП", "Госсектор", "Без работы"], size, p=[.58, .12, .08, .18, .04])
    education = rng.choice(["Среднее", "Среднее специальное", "Высшее", "Учёная степень"], size, p=[.18, .32, .47, .03])
    marital_status = rng.choice(["В браке", "Не в браке", "Разведён(а)", "Вдовец/вдова"], size, p=[.52, .31, .12, .05])

    frame = pd.DataFrame(
        {
            "application_id": np.arange(start_id, start_id + size),
            "age": age,
            "monthly_income": monthly_income,
            "credit_history_months": credit_history_months,
            "active_loans": active_loans,
            "overdue_30d": overdue_30d,
            "overdue_90d": overdue_90d,
            "debt_to_income": debt_to_income,
            "requested_amount": requested_amount,
            "employment_months": employment_months,
            "inquiries_6m": inquiries_6m,
            "region": region,
            "employment_type": employment_type,
            "education": education,
            "marital_status": marital_status,
        }
    )
    for index in range(1, 21):
        frame[f"bureau_signal_{index:02d}"] = rng.normal(0, 1, size)

    missing_income = rng.random(size) < 0.09
    missing_employment = rng.random(size) < 0.04
    frame.loc[missing_income, "monthly_income"] = np.nan
    frame.loc[missing_employment, "employment_months"] = np.nan

    if with_target:
        logit = (
            -3.15
            + 0.95 * overdue_30d
            + 1.35 * overdue_90d
            + 1.45 * debt_to_income
            + 0.30 * inquiries_6m
            + 0.26 * active_loans
            + 0.38 * (employment_type == "Без работы")
            + 0.24 * (employment_type == "Самозанятый")
            + 0.40 * missing_income
            - 0.018 * np.nan_to_num(employment_months, nan=45)
            - 0.0025 * credit_history_months
            - 0.000004 * np.nan_to_num(monthly_income, nan=60_000)
            + 0.18 * frame["bureau_signal_01"].to_numpy()
            - 0.14 * frame["bureau_signal_02"].to_numpy()
        )
        probability = 1 / (1 + np.exp(-logit))
        frame["target"] = rng.binomial(1, probability)
    return frame


def main() -> None:
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    train = make_applications(12_000, 1, rng, with_target=True)
    test = make_applications(3_000, 100_001, rng, with_target=False)
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    print(f"Synthetic credit data written to {OUT}: {len(train):,} train / {len(test):,} test")


if __name__ == "__main__":
    main()

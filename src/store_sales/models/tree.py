"""Predeclared efficient tree configurations, shared by training and checkpoint export."""

from lightgbm import LGBMRegressor


def make_tree(name):
    params = dict(
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=80,
        reg_lambda=1.0,
        colsample_bytree=0.9,
        n_jobs=2,
        random_state=42,
        verbosity=-1,
    )
    if name == "B3":
        params.update(n_estimators=600, max_bin=63, colsample_bytree=1.0)
    return LGBMRegressor(**params)


def feature_names(exog=False):
    return (
        ["store", "family", "cluster", "store_type"]
        + [f"origin_lag_{i}" for i in [0, 1, 6, 13, 27, 55]]
        + [
            f"rolling_{metric}_{w}"
            for w in [7, 14, 28, 56]
            for metric in ["mean", "std"]
        ]
        + [
            "zero_fraction_28",
            "store_total_log",
            "family_total_log",
            "horizon",
            "future_promotion_log",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "day_scaled",
            "payday",
            "year_ago_log",
            "year_ago_observed",
            "seasonal_anchor",
            "holiday",
            "workday",
            "origin_observed",
        ]
        + (
            ["oil_at_origin", "transactions_log_at_origin", "transactions_observed"]
            if exog
            else []
        )
    )

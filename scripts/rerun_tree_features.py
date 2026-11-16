"""Re-evaluate trees after the historical-missing-value repair, in an isolated directory."""

from pathlib import Path
import sys, time, json, gc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from store_sales.data import Panel
from store_sales.evaluation import write_prediction

p = Panel(ROOT, ROOT / "outputs/advanced/tree_repaired")
rows = []
dates = ["2016-08-16", "2017-04-16", "2017-06-15", "2017-07-15"]
for name, start, exog, folds in [
    ("B1", "2015-01-01", False, dates),
    ("B1_2016", "2016-01-01", False, dates[-2:]),
    ("B2", "2015-01-01", True, dates[-2:]),
]:
    for date in folds:
        t = p.index(date) - 1
        xx, y, mask = p.tree_rows(p.origins(p.dates[t], start), exog)
        m = LGBMRegressor(
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
        now = time.perf_counter()
        m.fit(xx[mask], y[mask], categorical_feature=[0, 1, 2, 3])
        del xx, y, mask
        gc.collect()
        vx, _, _ = p.tree_rows([t], exog)
        z = m.predict(vx).reshape(p.n, 16)
        sc = write_prediction(p, t, z, name)
        rows.append({"model": name, "date": date, **sc})
        pd.DataFrame(rows).to_csv(p.out / "metrics.csv", index=False)
        (p.out / f"training_{name}_{date}.json").write_text(
            json.dumps(
                {
                    "seconds": time.perf_counter() - now,
                    "train_start": start,
                    "exog": exog,
                    "missing_history": "NaN lags, masked mean/std/zero ratio, explicit observation flag",
                }
            )
        )
        print("REPAIRED_TREE", rows[-1], flush=True)
        del m
        gc.collect()
(p.out / "complete.json").write_text(json.dumps({"complete": True, "rows": len(rows)}))

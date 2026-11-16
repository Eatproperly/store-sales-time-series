"""One fixed, histogram-efficient boosting capacity candidate on the recent development windows."""

from pathlib import Path
import sys, json, gc, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from store_sales.data import Panel
from store_sales.models.tree import make_tree
from store_sales.evaluation import write_prediction

p = Panel(ROOT, ROOT / "outputs/advanced/tree_capacity")
rows = []
for date in ["2017-06-15", "2017-07-15"]:
    t = p.index(date) - 1
    x, y, mask = p.tree_rows(p.origins(p.dates[t]))
    m = make_tree("B3")
    timer = time.perf_counter()
    m.fit(x[mask], y[mask], categorical_feature=[0, 1, 2, 3])
    del x, y, mask
    gc.collect()
    v, _, _ = p.tree_rows([t])
    sc = write_prediction(p, t, m.predict(v).reshape(p.n, 16), "B3")
    rows.append({"model": "B3", "date": date, **sc})
    pd.DataFrame(rows).to_csv(p.out / "metrics.csv", index=False)
    (p.out / f"training_B3_{date}.json").write_text(
        json.dumps(
            {
                "seconds": time.perf_counter() - timer,
                "train_start": "2015-01-01",
                "parameters": m.get_params(),
                "missing_history": "masked statistics and native NaN lags",
            }
        )
    )
    print("CAPACITY_TRIAL", rows[-1], flush=True)
    del m
    gc.collect()
(p.out / "complete.json").write_text(json.dumps({"complete": True}))

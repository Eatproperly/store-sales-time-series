"""Repeat the confirmed neural candidate on the recent two folds; no R labels are used."""

from pathlib import Path
import sys, json, gc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from store_sales.data import Panel
from store_sales.training import outer_fit, predict
from store_sales.evaluation import write_prediction

p = Panel(ROOT, ROOT / "outputs/advanced/scale_seeds")
p.train_start = "2015-01-01"
p.base_windows_per_epoch = 4096
p.inner_mode = "budget"
p.batch_size = 256
rows = []
for seed in [2026, 3407]:
    for date in ["2017-06-15", "2017-07-15"]:
        name = f"mixer3_scale_seed{seed}"
        t = p.index(date) - 1
        m, detail = outer_fit(p, t, "mixer3_scale", width=32, seed=seed)
        sc = write_prediction(p, t, predict(m, p, t), name)
        rows.append({"model": name, "date": date, **sc})
        pd.DataFrame(rows).to_csv(p.out / "metrics.csv", index=False)
        (p.out / f"training_{name}_{date}.json").write_text(
            json.dumps(detail, indent=2)
        )
        print("SEED_REPEAT", rows[-1], flush=True)
        del m
        gc.collect()
(p.out / "complete.json").write_text(json.dumps({"complete": True, "rows": len(rows)}))

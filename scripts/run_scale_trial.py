"""Matched-budget context-level/scale decoder experiment, fixed before evaluating R."""

from pathlib import Path
import sys, json, gc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from store_sales.data import Panel
from store_sales.training import outer_fit, predict
from store_sales.evaluation import write_prediction

p = Panel(ROOT, ROOT / "outputs/advanced/scale_trial")
p.train_start = "2015-01-01"
p.base_windows_per_epoch = 4096
p.inner_mode = "budget"
p.batch_size = 256
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--confirm", action="store_true")
args = parser.parse_args()
rows = (
    pd.read_csv(p.out / "metrics.csv").to_dict("records")
    if (p.out / "metrics.csv").exists()
    else []
)
for date in (
    ["2016-08-16", "2017-04-16"] if args.confirm else ["2017-06-15", "2017-07-15"]
):
    if any(r["date"] == date for r in rows):
        continue
    t = p.index(date) - 1
    m, detail = outer_fit(p, t, "mixer3_scale", width=32)
    result = write_prediction(p, t, predict(m, p, t), "mixer3_scale")
    rows.append({"model": "mixer3_scale", "date": date, **result})
    pd.DataFrame(rows).to_csv(p.out / "metrics.csv", index=False)
    (p.out / f"training_mixer3_scale_{date}.json").write_text(
        json.dumps(detail, indent=2)
    )
    print("SCALE_TRIAL", rows[-1], flush=True)
    del m
    gc.collect()
(p.out / "complete.json").write_text(json.dumps({"complete": True}))

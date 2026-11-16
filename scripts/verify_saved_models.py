"""Verify the persisted inference artifacts against all three exported predictions, without retraining."""

from pathlib import Path
import sys, json, argparse, hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
from lightgbm import Booster
from store_sales.data import Panel
from store_sales.models.mixer import ForecastNet
from store_sales.training import predict

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default=str(ROOT / "outputs/advanced"))
args = parser.parse_args()
p = Panel(ROOT, Path(args.output_dir).resolve())
out = p.out
ck = out / "checkpoints"
sel = json.loads((out / "selection.json").read_text())
t = p.index("2017-08-15")
state = torch.load(ck / "network.pt", map_location="cpu", weights_only=True)
m = ForecastNet(state["sizes"], state["kind"], state["width"])
m.load_state_dict(state["state_dict"])
zn = np.maximum(predict(m, p, t), 0)
b = Booster(model_file=str(ck / "tree.txt"))
x, _, _ = p.tree_rows([t], sel["tree_model"] == "B2")
zt = np.maximum(b.predict(x).reshape(p.n, 16), 0)
results = {"tree": zt, "network": zn}
if sel["winner"] == "B0":
    baseline = Booster(model_file=str(ck / "baseline.txt"))
    ev = p.g["base"].loc[p.g["base"].date.ge("2017-08-16")].copy()
    ev["pred_log"] = np.maximum(baseline.predict(ev[p.g["features"]]), 0)
    key = p.keys(t)
    zz = (
        key.merge(
            ev[["store_nbr", "family", "date", "pred_log"]],
            left_on=["store_nbr", "family", "target_date"],
            right_on=["store_nbr", "family", "date"],
            validate="one_to_one",
        )
        .pred_log.to_numpy()
        .reshape(p.n, 16)
    )
else:
    zz = sel["weight_tree"] * zt + (1 - sel["weight_tree"]) * zn
results["final"] = zz
errors = {}
for name, z in results.items():
    key = p.keys(t)
    key["log_prediction"] = z.ravel()
    ordered = (
        p.g["test"][["id", "date", "store_nbr", "family"]]
        .merge(
            key,
            left_on=["date", "store_nbr", "family"],
            right_on=["target_date", "store_nbr", "family"],
            validate="one_to_one",
        )
        .set_index("id")
        .log_prediction
    )
    saved = pd.read_csv(out / f"submission_{name}.csv")
    difference = float(
        np.max(np.abs(ordered.reindex(saved.id).to_numpy() - np.log1p(saved.sales)))
    )
    assert difference < 1e-5, (name, difference)
    errors[name] = difference
meta = json.loads((ck / "metadata.json").read_text())
meta.update(
    weights_reproduce_saved_predictions=True, max_abs_log_prediction_difference=errors
)
(ck / "metadata.json").write_text(json.dumps(meta, indent=2))
(out / "saved_model_verification.json").write_text(
    json.dumps(
        {
            "weights_reproduce_saved_predictions": True,
            "max_abs_log_prediction_difference": errors,
            "model_file_sha256": {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in ck.iterdir()
                if f.suffix in [".pt", ".txt"]
            },
        },
        indent=2,
    )
)
print("SAVED_MODEL_INFERENCE_VERIFIED", errors, flush=True)

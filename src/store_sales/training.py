import time, random, json
import numpy as np
import torch
from .models.mixer import ForecastNet


def predict(model, panel, t, batch=256):
    model.eval()
    outs = []
    with torch.no_grad():
        for a in range(0, panel.n, batch):
            ss = np.arange(a, min(a + batch, panel.n))
            inp = panel.inputs(np.full(len(ss), t), ss)
            device = next(model.parameters()).device
            outs.append(
                model(*[torch.from_numpy(x).to(device) for x in inp]).cpu().numpy()
            )
    return np.concatenate(outs)


def train(
    panel,
    cutoff,
    kind,
    epochs=8,
    seed=42,
    start=None,
    width=32,
    validation_origin=None,
    patience=6,
):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForecastNet(panel.sizes, kind, width).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    origins = panel.origins(
        cutoff, start or getattr(panel, "train_start", "2015-01-01")
    )
    ids = np.arange(len(origins) * panel.n)
    rng = np.random.default_rng(seed)
    curves = []
    timer = time.perf_counter()
    best_value = float("inf")
    stale = 0
    for epoch in range(epochs):
        model.train()
        rng.shuffle(ids)
        loss_sum = 0.0
        n = 0
        limit = getattr(panel, "windows_per_epoch", 4096)
        epoch_ids = ids[:limit] if limit else ids
        batch = getattr(panel, "batch_size", 256)
        for a in range(0, len(epoch_ids), batch):
            ii = epoch_ids[a : a + batch]
            tt = origins[ii // panel.n]
            ss = ii % panel.n
            inp = panel.inputs(tt, ss)
            pred = model(*[torch.from_numpy(x).to(device) for x in inp])
            days = tt[:, None] + np.arange(1, 17)
            y = torch.from_numpy(panel.z[days, ss[:, None]]).to(device)
            mask = torch.from_numpy(panel.mask[days, ss[:, None]]).to(device)
            loss = ((pred - y).square() * mask).sum() / mask.sum().clamp(min=1)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            loss_sum += loss.item() * int(mask.sum())
            n += int(mask.sum())
        curves.append(
            {
                "epoch": epoch + 1,
                "train_log_mse": loss_sum / max(n, 1),
                "seconds": time.perf_counter() - timer,
            }
        )
        if validation_origin is not None:
            from .evaluation import score

            v = predict(model, panel, validation_origin)
            value = score(
                panel.z[validation_origin + 1 : validation_origin + 17].T,
                v,
                panel.mask[validation_origin + 1 : validation_origin + 17].T,
            )["RMSLE"]
            curves[-1]["inner_rmsle"] = value
            if value < best_value - 1e-6:
                best_value = value
                stale = 0
            else:
                stale += 1
        print(
            kind,
            cutoff,
            "epoch",
            epoch + 1,
            "loss",
            round(curves[-1]["train_log_mse"], 5),
            "sec",
            round(curves[-1]["seconds"], 1),
            flush=True,
        )
        if validation_origin is not None and stale >= patience:
            break
    return model, curves


def outer_fit(panel, t, kind, max_epochs=8, seed=42, width=32):
    # A precommitted extra CPU budget probes whether the promising 3-block net is undertrained.
    is_long = (
        kind.endswith(("_long", "_scale"))
        and getattr(panel, "inner_mode", "budget") != "patience"
    )
    if is_long:
        max_epochs = 16
    panel.windows_per_epoch = (
        8192 if is_long else getattr(panel, "base_windows_per_epoch", 4096)
    )
    # Internal 16-day validation selects epoch; outer validation remains unused.
    inner_t = t - 16
    best = float("inf")
    best_epoch = 1
    # CPU compares independent epoch budgets; the GPU protocol supports patience-based stopping.
    # For bounded CPU execution use independently trained epoch budgets 2 and max_epochs.
    trials = sorted(set([2, max_epochs]))
    inner = []
    from .evaluation import score

    if getattr(panel, "inner_mode", "budget") == "patience":
        _, curve = train(
            panel,
            panel.dates[inner_t],
            kind,
            max_epochs,
            seed,
            width=width,
            validation_origin=inner_t,
            patience=6,
        )
        best_epoch = int(np.argmin([r["inner_rmsle"] for r in curve])) + 1
        inner = curve
        trials = []
    for ep in trials:
        m, c = train(panel, panel.dates[inner_t], kind, ep, seed, width=width)
        z = predict(m, panel, inner_t)
        value = score(
            panel.z[inner_t + 1 : inner_t + 17].T,
            z,
            panel.mask[inner_t + 1 : inner_t + 17].T,
        )["RMSLE"]
        inner.append({"epochs": ep, "inner_rmsle": value})
        if value < best:
            best = value
            best_epoch = ep
    model, curve = train(panel, panel.dates[t], kind, best_epoch, seed, width=width)
    return model, {
        "inner_trials": inner,
        "selected_epochs": best_epoch,
        "training_curve": curve,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seed": seed,
        "width": width,
        "kind": kind,
        "windows_per_epoch": panel.windows_per_epoch,
        "max_epochs": max_epochs,
    }

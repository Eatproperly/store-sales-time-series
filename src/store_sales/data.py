"""Shared origin-based arrays; future targets never enter model inputs."""

import os, json, contextlib, io
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat


class Panel:
    def __init__(self, root, out_dir=None):
        self.root = Path(root)
        self.out = Path(out_dir) if out_dir else self.root / "outputs/advanced"
        self.out.mkdir(parents=True, exist_ok=True)
        # Reuse the frozen, actually executed baseline's preprocessing and estimator.
        notebook = nbformat.read(self.root / "Store_Sales.ipynb", as_version=4)
        prev = Path.cwd()
        os.environ["STORE_SALES_DATA_DIR"] = str(self.root / "data")
        os.chdir(self.out)
        g = {}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for c in notebook.cells:
                    if c.cell_type != "code":
                        continue
                    if not c.source.startswith(
                        (
                            "import os, sys",
                            "files=[",
                            "matched=holidays",
                            "panel=train.pivot",
                            "DEV=[",
                        )
                    ):
                        continue
                    exec(compile(c.source, "frozen_baseline", "exec"), g)
        finally:
            os.chdir(prev)
        self.g = g
        self.dates = g["panel"].index
        self.cols = g["panel"].columns
        self.n = len(self.cols)
        self.y = g["panel"].to_numpy().astype("float32")
        self.mask = np.isfinite(self.y)
        self.z = np.log1p(np.nan_to_num(self.y, nan=0))
        full = pd.concat([g["train"].drop(columns="sales"), g["test"]])
        self.promo = (
            full.pivot(
                index="date", columns=["store_nbr", "family"], values="onpromotion"
            )
            .reindex(index=self.dates, columns=self.cols)
            .fillna(0)
            .to_numpy(dtype="float32")
        )
        self.store = np.array(self.cols.get_level_values(0))
        self.family = np.array(self.cols.get_level_values(1).astype(str))
        meta = g["stores"].set_index("store_nbr").reindex(self.store)
        self.static = np.stack(
            [
                pd.Categorical(self.store).codes,
                pd.Categorical(self.family).codes,
                np.arange(self.n),
                pd.Categorical(meta.cluster).codes,
                pd.Categorical(meta.type).codes,
            ],
            axis=1,
        ).astype("int64")
        self.sizes = [int(self.static[:, i].max() + 1) for i in range(5)]
        stotal = np.stack(
            [
                np.log1p(np.nansum(self.y[:, self.store == s], axis=1))
                for s in np.unique(self.store)
            ],
            axis=1,
        )
        ftotal = np.stack(
            [
                np.log1p(np.nansum(self.y[:, self.family == f], axis=1))
                for f in np.unique(self.family)
            ],
            axis=1,
        )
        oil = (
            g["oil"]
            .set_index("date")
            .dcoilwtico.reindex(self.dates)
            .ffill()
            .fillna(0)
            .to_numpy(dtype="float32")
            / 100
        )
        tx = (
            g["transactions"]
            .pivot(index="date", columns="store_nbr", values="transactions")
            .reindex(index=self.dates, columns=np.unique(self.store))
        )
        self.history = np.stack(
            [
                self.z,
                np.log1p(self.promo),
                stotal[:, self.static[:, 0]],
                ftotal[:, self.static[:, 1]],
                np.broadcast_to(oil[:, None], self.z.shape),
                np.log1p(tx.fillna(0).to_numpy(dtype="float32"))[:, self.static[:, 0]],
                self.mask.astype("float32"),
                tx.notna().to_numpy(dtype="float32")[:, self.static[:, 0]],
            ],
            axis=-1,
        ).astype("float32")
        self.holiday = np.zeros((*self.z.shape, 3), dtype="float32")
        hol = g["hol"]
        si = {s: np.flatnonzero(self.store == s) for s in np.unique(self.store)}
        for row in hol.itertuples(index=False):
            d = self.dates.get_indexer([row.date])[0]
            if d >= 0:
                self.holiday[d, si[row.store_nbr], :] = [
                    row.is_holiday,
                    0,
                    row.is_workday,
                ]
        self.y.setflags(write=False)

    def index(self, date):
        return int(self.dates.get_indexer([pd.Timestamp(date)])[0])

    def origins(self, cutoff, start="2015-01-01"):
        last = self.index(cutoff) - 16
        first = max(112, self.index(start))
        days = np.arange(first, last + 1)
        # Rotate origin weekday; recent 56 days are retained daily.
        chosen = days[((days - first) % 8 == 0) | (days >= last - 55)]
        return chosen.astype("int64")

    def keys(self, origin):
        return pd.DataFrame(
            {
                "origin": self.dates[origin],
                "store_nbr": np.repeat(self.store, 16),
                "family": np.repeat(self.family, 16),
                "h": np.tile(np.arange(1, 17), self.n),
                "target_date": np.tile(self.dates[origin + 1 : origin + 17], self.n),
            }
        )

    def inputs(self, origins, sids):
        origins = np.asarray(origins)
        sids = np.asarray(sids)
        days = origins[:, None] + np.arange(1, 17)
        past = self.history[origins[:, None] + np.arange(-111, 1), sids[:, None]].copy()
        obs = past[:, :, 6:7]
        count = np.maximum(obs.sum(axis=1, keepdims=True), 1)
        mu = (past[:, :, :6] * obs).sum(1, keepdims=True) / count
        sd = np.sqrt(
            ((past[:, :, :6] - mu) ** 2 * obs).sum(1, keepdims=True) / count
        ).clip(0.1)
        past[:, :, :6] = (past[:, :, :6] - mu) / sd
        # Last observed same weekday for each future horizon, never after origin.
        ref = origins[:, None] - (
            (
                self.dates[origins].dayofweek.to_numpy()[:, None]
                - self.dates[days.ravel()].dayofweek.to_numpy().reshape(days.shape)
            )
            % 7
        )
        inds = ref[:, :, None] - 7 * np.arange(4)[None, None, :]
        vv = self.z[inds, sids[:, None, None]]
        mm = self.mask[inds, sids[:, None, None]]
        fallback = (
            self.z[origins[:, None] + np.arange(-27, 1), sids[:, None]]
            * self.mask[origins[:, None] + np.arange(-27, 1), sids[:, None]]
        ).sum(1) / np.maximum(
            self.mask[origins[:, None] + np.arange(-27, 1), sids[:, None]].sum(1), 1
        )
        anchor = np.where(
            mm.any(2), (vv * mm).sum(2) / np.maximum(mm.sum(2), 1), fallback[:, None]
        ).astype("float32")
        annual = days - 364
        validannual = annual >= 0
        year = self.z[np.maximum(annual, 0), sids[:, None]] * validannual
        dow = self.dates[days.ravel()].dayofweek.to_numpy().reshape(days.shape)
        month = self.dates[days.ravel()].month.to_numpy().reshape(days.shape)
        dom = self.dates[days.ravel()].day.to_numpy().reshape(days.shape)
        fut = np.stack(
            [
                np.log1p(self.promo[days, sids[:, None]]),
                np.sin(2 * np.pi * dow / 7),
                np.cos(2 * np.pi * dow / 7),
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
                dom / 31,
                (
                    (dom == 15)
                    | self.dates[days.ravel()].is_month_end.reshape(days.shape)
                ).astype(float),
                year,
                self.mask[np.maximum(annual, 0), sids[:, None]] * validannual,
                anchor,
                *[self.holiday[days, sids[:, None], j] for j in [0, 2]],
            ],
            axis=-1,
        ).astype("float32")
        latest = np.where(self.mask[origins, sids], self.z[origins, sids], mu[:, 0, 0])
        stats = np.stack([mu[:, 0, 0], sd[:, 0, 0], latest], axis=-1)
        fut = np.concatenate(
            [fut, np.broadcast_to(stats[:, None, :], (len(sids), 16, 3))], axis=-1
        ).astype("float32")
        return past.astype("float32"), fut, self.static[sids], anchor

    def tree_rows(self, origins, exog=False):
        chunks = []
        ys = []
        masks = []
        for t in origins:
            ss = np.arange(self.n)
            _, future, stat, anchor = self.inputs(np.full(self.n, t), ss)
            future = future[:, :, :12]
            vals = [
                np.broadcast_to(stat[:, i, None], (self.n, 16)) for i in [0, 1, 3, 4]
            ]
            for lag in [0, 1, 6, 13, 27, 55]:
                vals.append(
                    np.broadcast_to(
                        np.where(self.mask[t - lag], self.z[t - lag], np.nan)[:, None],
                        (self.n, 16),
                    )
                )
            for w in [7, 14, 28, 56]:
                a = np.where(
                    self.mask[t - w + 1 : t + 1], self.z[t - w + 1 : t + 1], np.nan
                )
                vals += [
                    np.broadcast_to(np.nanmean(a, axis=0)[:, None], (self.n, 16)),
                    np.broadcast_to(np.nanstd(a, axis=0)[:, None], (self.n, 16)),
                ]
            zero_ratio = (
                (self.z[t - 27 : t + 1] == 0) & self.mask[t - 27 : t + 1]
            ).sum(0) / np.maximum(self.mask[t - 27 : t + 1].sum(0), 1)
            vals += [
                np.broadcast_to(zero_ratio[:, None], (self.n, 16)),
                np.broadcast_to(self.history[t, :, 2, None], (self.n, 16)),
                np.broadcast_to(self.history[t, :, 3, None], (self.n, 16)),
                np.tile(np.arange(1, 17), (self.n, 1)),
                *np.moveaxis(future, -1, 0),
            ]
            vals.append(np.broadcast_to(self.mask[t, :, None], (self.n, 16)))
            if exog:
                vals += [
                    np.broadcast_to(self.history[t, :, i, None], (self.n, 16))
                    for i in [4, 5, 7]
                ]
            chunks.append(
                np.stack(vals, axis=-1).reshape(-1, len(vals)).astype("float32")
            )
            ys.append(self.z[t + 1 : t + 17].T.ravel())
            masks.append(self.mask[t + 1 : t + 17].T.ravel())
        return np.concatenate(chunks), np.concatenate(ys), np.concatenate(masks)

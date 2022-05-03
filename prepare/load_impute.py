import os
import sys
import uuid
from itertools import product, zip_longest
from pathlib import Path

import numpy as np
import omegaconf
import pandas as pd
import pandera as pa
import joblib
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from tsmoothie.smoother import DecomposeSmoother, KalmanSmoother, SpectralSmoother

wd = Path.cwd()
cfg = OmegaConf.load(wd / "conf" / "config.yml")

source = wd / cfg.catalogue.clean
# target = wd / cfg.catalogue.impute
target = wd / cfg.catalogue.model_in
target.parent.mkdir(parents=True, exist_ok=True)


dt_start = cfg.params.dt_start
dt_end = cfg.params.dt_end


df = pd.read_csv(source, parse_dates=["timestamp"], index_col=["timestamp"]).loc[
    dt_start:dt_end
]


# qc
invalid_mask = df.le(-5) | df.gt(35)
df.loc[:, :] = np.where(invalid_mask, np.nan, df.values)


schema = pa.DataFrameSchema(
    {
        col: pa.Column(float, checks=[pa.Check.gt(-5), pa.Check.le(35)], nullable=True)
        for col in df.columns
    }
)

resampler = cfg.params.resampler
n_seasons = 24 // resampler
n_longseasons = 365 * (24 // resampler)

df = schema(df)
df = df.resample(f"{resampler}H").mean()

# impute for fillna
n_neighbors = cfg.params.n_neighbors
imputer = KNNImputer(n_neighbors=n_neighbors)
df.loc[:, :] = imputer.fit_transform(df)

# outlier detection & removal
clf = LocalOutlierFactor(n_neighbors=n_neighbors)
outlier_mask = clf.fit_predict(df)
df[outlier_mask] = np.nan


# imputation using kalman filter
mmap_path = Path(cfg.catalogue.scratch)
mmap_path.parent.mkdir(parents=True, exist_ok=True)

if mmap_path.exists():
    mmap_path.unlink()

# _ = joblib.dump(df.values, mmap_path)
_ = joblib.dump(df, mmap_path)
output = joblib.load(mmap_path, mmap_mode="w+")


def smoothe_col(indice, sid, component_noise=None):
    component_noise = {
        "level": 0.5,
        "trend": 0.1,
        "season": 0.1,
        "longseason": 0.1,
    }

    data = df.loc[:, [sid]].values.T.astype(np.float64)
    smoother = KalmanSmoother(
        component="level_trend_season",
        component_noise=component_noise,
        n_seasons=n_seasons,
        n_longseasons=n_longseasons,
    )
    # output[:, indice] = smoother.smooth(data)
    # output[:, indice] = smoother.smooth(data).smooth_data
    output.loc[:, sid] = smoother.smooth(data).smooth_data.flatten()


with Parallel(n_jobs=10) as parallel:
    _ = parallel(delayed(smoothe_col)(indice, sid) for indice, sid in enumerate(df.columns))
# sanity test
assert df.mean().mean() != output.mean().mean()
output.to_csv(target)


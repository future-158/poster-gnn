from pathlib import Path
from warnings import simplefilter

import numpy as np
import pandas as pd
import tqdm
from joblib import Parallel, delayed
import joblib
from omegaconf import OmegaConf

# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.linear_model import Lasso, LassoCV, RidgeClassifierCV
# from sklearn.pipeline import make_pipeline
# from sktime.datasets import load_arrow_head  # univariate dataset
# from sktime.datasets import load_airline
# from sktime.forecasting.arima import ARIMA, AutoARIMA
# from sktime.forecasting.base import ForecastingHorizon
# from sktime.forecasting.compose import (EnsembleForecaster,
#                                         MultiplexForecaster,
#                                         TransformedTargetForecaster,
#                                         make_reduction)
# from sktime.forecasting.exp_smoothing import ExponentialSmoothing
# from sktime.forecasting.model_evaluation import evaluate
# from sktime.forecasting.model_selection import (ExpandingWindowSplitter,
#                                                 ForecastingGridSearchCV,
#                                                 SlidingWindowSplitter,
#                                                 temporal_train_test_split)
# from sktime.forecasting.naive import NaiveForecaster
# from sktime.forecasting.theta import ThetaForecaster
# from sktime.forecasting.trend import PolynomialTrendForecaster
# from sktime.transformations.panel.rocket import Rocket
# from sktime.transformations.series.detrend import Deseasonalizer, Detrender
# from sktime.utils.plotting import plot_series

simplefilter("ignore", FutureWarning)
# %matplotlib inline


cfg = OmegaConf.load("config.yaml")
df = pd.read_parquet(Path(cfg.dataset_path))


assert df.index.to_series().diff().value_counts().nunique() == 1
df = df.loc["2018":"2021"]
df = df.interpolate(method="linear", limit=12, limit_direction="both")
df.index.freq = "1H"
pred_table = df.copy()


mmap_path = Path("scratch/arr.mmap")
if mmap_path.exists():
    mmap_path.unlink()

_ = joblib.dump(pred_table.values, mmap_path)
large_memmap = joblib.load(mmap_path, mmap_mode="w+")


def impute_column(col_id: int):
    nan_mask = np.isnan(large_memmap[:, col_id])
    if nan_mask.sum() == 0:
        return 1

    nan_iloc = np.flatnonzero(nan_mask)
    valid_iloc = np.flatnonzero(~nan_mask)

    nf = NaiveForecaster("last", sp=24)
    nf.fit(large_memmap[valid_iloc, col_id])
    large_memmap[nan_iloc, col_id] = nf.predict(nan_iloc).flatten()

    inversed = large_memmap[::-1, col_id]

    nan_mask = np.isnan(inversed)
    if nan_mask.sum() == 0:
        return 2

    nf = NaiveForecaster("last", sp=24)
    nan_iloc = np.flatnonzero(nan_mask)
    valid_iloc = np.flatnonzero(~nan_mask)

    nf.fit(inversed[valid_iloc])

    large_memmap[nan_iloc, col_id] = nf.predict(inversed[nan_iloc]).flatten()[::-1]
    assert large_memmap[:, col_id].isna().sum() == 0
    return 3


status_codes = Parallel(n_jobs=10, max_nbytes=None)(
    delayed(impute_column)(i) for i, col in enumerate(pred_table.columns)
)

imputed = pd.DataFrame(large_memmap, index=pred_table.index, columns=pred_table.columns)

# from sktime.forecasting.bats import BATS
# forecaster = BATS(sp=24, use_trend=True, use_box_cox=False)
# series = large_memmap[:, 1]
# forecaster.fit(series)
# y_pred = forecaster.predict(fh=100)

# plot_series(
#     df.iloc[:,1],
#     pd.Series(large_memmap[:, 1], index=df.index),
#     labels=["y_test", "y_pred"])
joblib.dump(large_memmap, cfg.imputed_path)
# (35064, 111)

_ = pd.DataFrame(large_memmap).to_csv(cfg.imputed_path, index=None, header=None)

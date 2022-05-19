import os
import torch
import pandas as pd
import hydra
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd
# import pandera as pa


wd = Path.cwd()
cfg = OmegaConf.load("conf/config.yml")
pd.read_csv(cfg.catalogue.model_in).isna().sum().sum()

args = cfg.args
data_file = cfg.prep.src
data = pd.read_csv(data_file, parse_dates=["timestamp"], index_col=["timestamp"])


drop_mask = data.describe().loc["min"].le(-2) | data.describe().loc["max"].gt(30)
data = data.loc[:, ~drop_mask]

schema = pa.DataFrameSchema(
    {
        ".+": pa.Column(
            float,
            nullable=True,
            checks=[
                pa.Check.greater_than_or_equal_to(-2),
                pa.Check.less_than_or_equal_to(30),
            ],
            regex=True,
        ),
        # "cat_var_.+": pa.Column(
        #     pa.Category,
        #     checks=pa.Check.isin(categories),
        #     coerce=True,
        #     regex=True,
        # ),
    }
)
print(schema.validate(data).head())


# index = pd.date_range(start='2020-01-01', periods=data.shape[0], freq='H')
data = (
    data
    # .set_index(index)
    .resample(cfg.prep.resample).agg(cfg.prep.agg)
    # .interpolate('linear')
    # .dropna()
)


X = data.values
from sklearn.impute import KNNImputer

# X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2)
data.loc[:, :] = imputer.fit_transform(X)

assert data.isna().mean().sum() == 0
data.to_csv(cfg.prep.dest, index=False, header=False)

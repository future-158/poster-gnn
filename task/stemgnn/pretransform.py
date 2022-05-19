import os
import torch
import pandas as pd
import hydra
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd
import pandera as pa


offset = 273.15


wd = Path.cwd()
cfg = OmegaConf.load("conf/config.yml")
df = pd.read_csv(cfg.catalog.model_in, parse_dates=[0], index_col=[0])
ostia = pd.read_csv(cfg.catalog.ostia, parse_dates=[0], index_col=[0])
ostia -= offset

rs = np.random.RandomState(889)
selected = rs.choice(ostia.columns, df.shape[1])
df = df.join(ostia[selected])
df.to_csv(cfg.catalog.merge, index=True)










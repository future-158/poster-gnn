import argparse
import itertools
import json
import logging
import operator
import os
import pickle
import re
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from functools import reduce
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional
from sklearn.cluster import KMeans
import numpy as np

import matplotlib.pyplot as plt
from libpysal.weights.contiguity import Queen
from libpysal import examples
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import splot


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from omegaconf import OmegaConf
from sklearn.metrics import *

cfg = OmegaConf.load('conf/config.yml')

ckpt_dir =  Path(cfg.catalog.checkpoint)
src_dir = ckpt_dir / 'output' / 'poster' / 'test'
best_config_file = list(ckpt_dir.glob('**/config.yaml'))[0]
best_cfg = OmegaConf.load(best_config_file)
# ['predict_abs_error.csv', 'target.csv', 'predict_ape.csv', 'predict.csv']
pred = pd.read_csv(src_dir / 'predict.csv', header=None)
target = pd.read_csv(src_dir / 'target.csv', header=None)

data_file = cfg.catalogue.model_in
data = pd.read_csv(data_file, index_col=[0], parse_dates=[0])
test_data = data.loc["2021"]
np.testing.assert_allclose(
    target.values,
    test_data[best_cfg.args.window_size:],
    atol=1e-5
)

renamer = {
    'koofs': 'koofs', 'cwbuoy': 'kma', 'buoy': 'kma', 
}

new_columns = data.columns.astype(str).str.split('_').str.get(0).map(renamer) + '_' + data.columns.astype(str).str.split('_', n=1).str.get(1)
pred.columns  = new_columns
target.columns  = new_columns


gdf = (
    pd.read_csv(cfg.catalog.geo_file)
    .assign(ssid = lambda df: df.stype + '_' + df.sid)
)

kmeans = KMeans(n_clusters=3, random_state=0).fit(
    gdf[['Latitude', 'Longitude']]
)
gdf['label'] = kmeans.labels_
gdf['label'] = gdf['label'].map({
    0: 'west',
    1: 'south',
    2: 'east'
})

ssid_rmse = (
    np.sqrt(np.power(target - pred, 2).mean())
    .to_frame(name='rmse')
    .reset_index()
    .rename(columns=dict(index='ssid')))


gdf = gpd.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))


y = gdf['Donatns'].values
w = Queen.from_dataframe(gdf)
w.transform = 'r'

from esda.moran import Moran
w = Queen.from_dataframe(gdf)
moran = Moran(y, w)
moran.I


from splot.esda import moran_scatterplot
fig, ax = moran_scatterplot(moran, aspect_equal=True)
plt.show()


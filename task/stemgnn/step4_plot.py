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


ssid_rmse = ssid_rmse.merge(gdf, how='left', on ='ssid').dropna()

# fig = px.violin(pdf, y="vis_khoa", x="vis_sijung",  box=True, points="outliers")
fig = px.box(
    ssid_rmse,
    x="label", 
    y="rmse",
    width=1280, height=720,
    points="all",
    )

fig.update_layout(
    title='해역별 비교',
    xaxis_title="해역",
    yaxis_title="RMSE",
    font=dict(
        # family="Courier New, monospace",
        size=18,
        # color="RebeccaPurple"
    )
)

# fig.update_xaxes(
#         tickvals = list(t2i.values()),
#         ticktext = list(t2i.keys())
#         )
# fig.update_yaxes(tickvals=[500, 1000,3000,10000,20000])

# fig.add_hline(y=1000, line_width=3, line_dash="dash", line_color="green")
# fig.add_hline(y=3000, line_width=3, line_dash="dash", line_color="green")
# fig.show()

report_dir = Path.cwd() / 'data' / 'report'
report_dir.mkdir(parents=True, exist_ok=True)
fig.write_image(
    # report_dir / f'box_plot.svg'    
    report_dir / f'box_plot.png'    
    )

#  시계열 그림
# morans plot

ssid_rmse.groupby('label')['ssid'].sample(1).tolist()

value_vars = [renamer[x] for x in value_vars]
pdf = pdf.rename(columns=renamer)
melt_align = pd.melt(pdf, id_vars=['timestamp'], value_vars=value_vars)
fig = px.scatter(melt_align, x="timestamp", y="value", color='variable', width=1280, height=720)
fig.update_layout(
    title=f'{station} {month.strftime("%Y-%m")}',
    xaxis_title="시간",
    yaxis_title="시정(m)",
    legend_title="구분",
    font=dict(
        # family="Courier New, monospace",
        size=18,
        # color="RebeccaPurple"
    )
)

tickvals = [500,  1000,   3000, 10000,20000]
fig.update_yaxes(tickvals=tickvals)
xmin = melt_align.timestamp.min().normalize() + pd.offsets.MonthBegin() - pd.offsets.MonthBegin()
xmax = (melt_align.timestamp.min() + pd.offsets.MonthEnd()).normalize().replace(hour=23, minute=59)
fig.update_xaxes(range=[xmin, xmax])    

fig.add_hline(y=1000, line_width=3, line_dash="dash", line_color="green")
fig.add_hline(y=3000, line_width=3, line_dash="dash", line_color="green")

# rect_li = (
#     melt_align
#     # .groupby(pd.Grouper(key='timestamp', freq='H'))
#     .groupby(pd.Grouper(key='timestamp', freq='D'))
#     ['value']
#     .min()
#     .pipe(lambda ser: ser[ser<1000])
#     .index
# )
rect_li = pdf[pdf.Khoa.le(1000)].timestamp.dt.to_period('d').dt.to_timestamp().drop_duplicates()

for rect in rect_li:
    # x0, x1 = rect, rect + pd.Timedelta(hours=1)
    x0, x1 = rect, rect + pd.Timedelta(days=1)
    # fig.add_hrect(x0=x0, x1=x1, y0=0, y1=1000, line_width=0, fillcolor="red", )
    # fig.add_vrect(x0=x0, x1=x1, fillcolor="red", opacity=0.2, line_width=0.5)
    fig.add_vrect(x0=x0, x1=x1, fillcolor="red", opacity=0.2, line_width=0.5,
                annotation_text=rect.strftime('%d일'), annotation_position="top left",
                )
        
    # fig.add_shape(
    #     type="rect",
    #     xref="x", yref="y",
    #     x0=x0, y0=0,
    #     x1=x1, y1=1000,
    #     line=dict(
    #         color="RoyalBlue",
    #         width=0,
    #     ),
    #     # fillcolor="LightSkyBlue",
    #     fillcolor="LightGreen",
    #     opacity=0.5,
    # )

dest = out_dir / f'monthly_{station}_{month.strftime("%Y%m")}.html'
# fig.write_html(dest)
fig.write_image(dest.with_suffix('.png'))


    # px.scatter category_orders, range_x, range_y


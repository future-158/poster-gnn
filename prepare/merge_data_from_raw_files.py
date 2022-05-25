import subprocess
import time
import pandas as pd
import io
import sys
from zipfile import ZipFile
import os
import missingno as msno

import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import dask.dataframe as dd
from dask import delayed


"""
아래 돌릴려면 raw datafile을 다운받아야함.
steps:

- aws configure
    AWS Access Key ID [None]: seafog
    AWS Secret Access Key [None]: defaultpassword
    Default region name [None]: ENTER
    Default output format [None]: ENTER

- aws configure set default.s3.signature_version s3v4
- aws --endpoint-url http://oldgpu:9000 s3 cp s3://poster/data  data --recursive
"""

cfg = OmegaConf.load('conf/config.yml')
buoy_dir = Path(cfg.catalogue.buoy_dir)
fargo_dir = Path(cfg.catalogue.fargo_dir)
koofs_path = Path(cfg.catalogue.koofs_path)

files = [x for x in buoy_dir.glob('**/*.zip') if x.name.startswith('MARINE_BUOY')]

df_li = []
for file in files:
    with ZipFile(file) as z:
        for inner_file in z.filelist:
            df  = pd.read_csv(
                io.BytesIO(z.open(inner_file).read()),
                encoding='cp949'
            )
            df_li.append(df)
buoy = pd.concat(df_li)

        
files = [x for x in fargo_dir.glob('**/*.zip')
 if x.name.startswith('MARINE_CWBUOY')]

df_li = []
for file in files:
    with ZipFile(file) as z:
        for inner_file in z.filelist:
            df  = pd.read_csv(
                io.BytesIO(z.open(inner_file).read()),
                encoding='cp949'
            )
            df_li.append(df)
cwbuoy = pd.concat(df_li)
koofs = pd.read_csv(koofs_path, parse_dates=['관측시간'], infer_datetime_format=True)

header = ['sid', 'timestamp', 'sst']
buoy =  buoy[['지점','일시',  '수온(°C)']]
cwbuoy = cwbuoy[['지점', '일시', '수온(°C)']]
koofs = koofs[['관측소명', '관측시간', '수온(°C)']]
buoy.columns = header
cwbuoy.columns = header
koofs.columns = header

astype = {
    'sid':pd.StringDtype(),
    'timestamp': pd.StringDtype(),
    'sst': np.float32
}

buoy = buoy.astype(astype)
cwbuoy = cwbuoy.astype(astype)
koofs = koofs.astype(astype)

cat = pd.concat([
    buoy.assign(sid=lambda x: 'buoy_' + x.sid), 
    cwbuoy.assign(sid=lambda x: 'cwbuoy_' + x.sid), 
    koofs.assign(sid=lambda x: 'koofs_' + x.sid)
    ])

cat.timestamp = pd.to_datetime(cat.timestamp, format='%Y-%m-%d %H:%M')

start_dt = pd.to_datetime(cfg.start_dt)
end_dt = pd.to_datetime(cfg.end_dt)

mask = np.logical_and(
    cat.timestamp >= start_dt,
    cat.timestamp < end_dt

)
cat = cat[mask]
pivot = (
    pd.pivot_table(cat, index='timestamp', columns='sid')
    ['sst']
)

invalid_sid_mask = pivot.isna().mean().gt(0.3)
pivot = pivot.loc[:, ~invalid_sid_mask]
msno.matrix(pivot, freq='MS')
# pivot.isna().mean().plot.hist(bins=100)
pivot.to_csv(cfg.catalogue.clean)

# how to imputation 
# feature propagation
# 1d imputation



    

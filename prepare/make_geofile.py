from ast import parse
import itertools
import subprocess
import time
import pandas as pd
from itertools import product
import io
import sys
from zipfile import ZipFile
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import geopandas
import numpy as np
from sklearn.neighbors import KDTree

cfg = OmegaConf.load('config.yml')
geo_dir = Path(cfg.catalogue.geo_dir)

filenames = [
    geo_dir / "META_관측지점정보_기상청_파고부이.csv",
    geo_dir / "META_관측지점정보_기상청_해양부이.csv"
]

usecols = ['지점', '종료일', '지점명','위도', '경도']
renamer = {
    '지점': 'sid',
    '지점명': 'sname',
    '위도':'Latitude',
    '경도':'Longitude'}

proj_cols = ['stype', 'sid', 'sname', 'Latitude', 'Longitude']


df_li = []
for filename in filenames:
    df = pd.read_csv(
        filename,
        encoding='cp949',
        parse_dates=['종료일']
        # usecols=usecols
        )
    df['stype'] = 'kma'
    df = df[df.종료일.isna()]
    df_li.append(
        df.rename(columns=renamer)[proj_cols]
    )


sheet_names = ['해양관측부이', '조위관측소']
renamer = {
    '관측소 번호': 'sid',
    '관측소 명': 'sname',
    '위도':'Latitude',
    '경도':'Longitude'}

for sheet_name in sheet_names:
    usecols = ['관측소 번호', '관측소 명', '위도', '경도']
    df = pd.read_excel(
        geo_dir / 'META_관측지점정보_조사원.xlsx'
        ,sheet_name=sheet_name)

    df['stype'] = 'koofs'
    df_li.append(
        df.rename(columns=renamer)[proj_cols]
    )

table = pd.concat(df_li)
assert table.sid.duplicated().sum() == 0
table.to_csv(cfg.catalogue.geo_path, index=False)


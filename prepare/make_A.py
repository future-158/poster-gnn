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
geo_path = Path(cfg.catalogue.geo_path)
dataset_path = Path(cfg.catalogue.dataset_path)


table = pd.read_csv(geo_path)
table.sid = table.sid.astype(str)
sids = pd.read_csv(dataset_path, index_col=['timestamp']).columns

sid_encoder = {sid: sid.split('_', maxsplit=1)[1] for sid in sids}
sid_decoder = {v:k for k,v in sid_encoder.items()}

table  = table[table.sid.isin(sid_encoder.values())]

coords = ['Latitude', 'Longitude']
X = table[coords].values
tree = KDTree(X, leaf_size=2)              
dist, ind = tree.query(X,  k=cfg.n_neighbours)                
n_idx = ind[:,:] # use knn3

neighbour_map = (
    pd.DataFrame(
        table.sid.values[n_idx]).applymap(lambda x: sid_decoder[x])
    .set_index(0)
)

neighbour_map.index.name = 'sid'
neighbour_map.columns = [f'knn_{i}' for i in range(cfg.n_neighbours) if i>0]
neighbour_map.to_csv(cfg.catalogue.A_path)



    


    













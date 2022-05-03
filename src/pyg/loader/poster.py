import io
import json
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from omegaconf import OmegaConf
from six.moves import urllib
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from torch_geometric_temporal.dataset import *
from torch_geometric_temporal.nn import *
from torch_geometric_temporal.signal import *

class PosterSSTDataLoader(object):
    def __init__(self, cfg):
        self.source = Path(cfg.wd) / cfg.catalogue.model_in
        self.meta_path = Path(cfg.wd) / cfg.catalogue.meta
        self.scaler = Path(cfg.wd) / cfg.catalogue.scaler
        self.scaler.parent.mkdir(parents=True, exist_ok=True)
        self.meta =  Path(cfg.wd) / cfg.catalogue.meta
        self._read_web_data()


    def _read_web_data(self):
        data = pd.read_csv(self.source, parse_dates=['timestamp'], index_col=['timestamp'])
        data = data.interpolate(method='linear').ffill().bfill()
        
        assert data.index.is_monotonic
        ss = StandardScaler()
        data.loc[:,:] = ss.fit_transform(data)
        joblib.dump(ss, self.scaler)
        self.data = data

    def _get_edges(self):
        # meta = pd.read_csv(META_PATH)
        # columns = self.data.columns
        # meta = meta[meta.sid.isin(columns)]
        # coords = ['Latitude', 'Longitude']
        # cls2idx = {k: i for i, k in enumerate(columns)}
        
        # X = meta[coords].values
        # tree = KDTree(X, leaf_size=2)              
        # dist, ind = tree.query(X,  k=4)                
        # n_idx = ind[:,1:]
        # n_sids = meta[['sid']].values[n_idx][...,0]

        # n_li = []
        # for src, trt in zip(meta.sid, n_sids):
        #     n_li.extend(list(product([src], trt)))

        # self._edges = np.array(
        #     [[cls2idx[a],cls2idx[b]] for a,b in n_li ]
        # ).T

        # self._edges = list(self._edges)

        meta = pd.read_csv(self.meta)
        cls2idx = {k: i for i, k in enumerate(meta.sid)}
        edges = (
            pd.melt(meta, id_vars=['sid'], value_vars = ['knn_1', 'knn_2', 'knn_3'])
            .rename(columns={'sid':'source', 'value':'target'})
            .assign(
                source= lambda df: df.source.map(cls2idx),
                target= lambda df: df.target.map(cls2idx),
                )
            [['source','target']]
        ).values.tolist()

        edges = [*edges, *[ [v,v] for v in cls2idx.values()]]
        self._edges = list(zip(*edges))
        
    def _get_edge_weights(self):
        self._edge_weights = np.ones_like(self._edges)
        self._edge_weights = list(self._edge_weights)[0]

    def _get_targets_and_features(self, lag_step: int, pred_step: int):
        data = self.data
        window  = lag_step+pred_step
        n_samples = data.shape[0] - window + 1
    
        idx = np.arange(n_samples)[:, None] + np.arange(window)

        self.features = np.transpose(
            data.values[idx],
            (0,2,1)
        )[:,:,:lag_step]

        self.targets = np.transpose(
            data.values[idx],
            (0,2,1)
        )[:,:,-1]

        self.features = list(self.features)
        self.targets = list(self.targets)


    def get_dataset(self, lag_step: int = 24, pred_step: int = 12) -> StaticGraphTemporalSignal:
        """Returning the England COVID19 data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The England Covid dataset.
        """

        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features(lag_step, pred_step)
        # dataset = DynamicGraphTemporalSignal(
        #     self._edges, self._edge_weights, self.features, self.targets
        # )
        static_dl = StaticGraphTemporalSignal(
            self._edges,
            self._edge_weights,
            self.features,
            self.targets)
        return static_dl


if __name__ == '__main__':
    cfg_path = Path.cwd() / 'conf' / 'config.yml'
    cfg = OmegaConf.load(cfg_path)
    loader = PosterSSTDataLoader(cfg)

    dataset = loader.get_dataset(
        lag_step=cfg.params.lag_step,
        pred_step=cfg.params.pred_step
        )

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)
    next(iter(train_dataset)).x.shape
    next(iter(train_dataset)).edge_index.dtype
    next(iter(train_dataset)).edge_attr.dtype
    next(iter(train_dataset)).y.dtype


    
    


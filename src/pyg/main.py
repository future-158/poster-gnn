import os
import sys
import uuid
import socket
from pathlib import Path

import joblib
import hydra
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from hydra import utils
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.nn.recurrent import MPNNLSTM
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm

from loader.poster import PosterSSTDataLoader

# os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE']  = '1'
# os.environ['AWS_ACCESS_KEY_ID']="minio"
# os.environ['AWS_SECRET_ACCESS_KEY']="minio123"
# os.environ['MLFLOW_S3_ENDPOINT_URL']="http://mlflow_s3:9000"

class RecurrentGCN(torch.nn.Module):
    def __init__(
        self, node_features: int, num_nodes: int, hidden_size: int, out_channels: int
    ):
        super(RecurrentGCN, self).__init__()
        self.recurrent = MPNNLSTM(
            node_features, hidden_size, out_channels, num_nodes, 1, 0.5
        )
        self.linear = torch.nn.Linear(2 * hidden_size + node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

@hydra.main(config_path="../../conf", config_name="config.yml")
def main(cfg: DictConfig) -> float:
    hidden_size = cfg.model.hidden_size
    out_channels = cfg.model.out_channels
    lr = cfg.optimizer.lr
    lag_step = cfg.params.lag_step

    with open_dict(cfg):
        cfg.wd = get_original_cwd()
    
    loader = PosterSSTDataLoader(cfg)
    dataset = loader.get_dataset(lag_step, cfg.params.pred_step)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

    # loader = ChickenpoxDatasetLoader()
    # dataset = loader.get_dataset()
    # node_features = 4
    # num_nodes = 20

    num_epochs = cfg.params.num_epochs
    # node_features = lag_step
    num_nodes,  node_features = next(iter(dataset.features)).shape
    val_freq = 1
    invert_transform = True

    model = RecurrentGCN(
        node_features=node_features,
        num_nodes=num_nodes,
        hidden_size=hidden_size,
        out_channels=out_channels,
    )

    device = "cuda"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_history = []
    # mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
    # mlflow.set_tracking_uri('http://mlflow_tracking_server:5000')

    tracking_port= cfg.mlflow.tracking_port
    TRACKING_SERVER = f'http://localhost:{tracking_port}'
    a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    location = ("127.0.0.1", tracking_port)
    result_of_check = a_socket.connect_ex(location)

    if result_of_check == 0:
        print("Port is open")
        a_socket.close()
    else:
        print("Port is not open")
        a_socket.close()
        sys.exit(0)

    mlflow.set_tracking_uri(TRACKING_SERVER)
    mlflow.set_experiment(cfg.mlflow.runname)

    ss = joblib.load(
        os.path.join(get_original_cwd(),cfg.catalogue.scaler)
    )
    
    # with mlflow.start_run():
    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                'lag_step': lag_step,
                'hidden_size': hidden_size,
                'out_channels': out_channels,
                'lr': lr
            }
        )        
        for epoch in tqdm(range(num_epochs)):
            model.train()
            cost = 0
            for time, snapshot in enumerate(train_dataset):
                y_hat = model(
                    snapshot.x.to(device),
                    snapshot.edge_index.to(device),
                    snapshot.edge_attr.to(device),
                    # snapshot.x,
                    # snapshot.edge_index,
                    # snapshot.edge_attr
                )
                cost = cost + torch.mean((y_hat - snapshot.y.to(device)) ** 2)
            cost = cost / (time + 1)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()

            model.eval()
            cost = 0
            with torch.no_grad():
                for time, snapshot in enumerate(test_dataset):
                    y_hat = model(
                        snapshot.x.to(device),
                        snapshot.edge_index.to(device),
                        snapshot.edge_attr.to(device),
                    )

                    if not invert_transform:
                        cost = cost + torch.mean((y_hat.cpu() - snapshot.y) ** 2)
                    else:    
                        y_hat_revert = ss.inverse_transform(
                            y_hat.reshape(1,-1).cpu()
                        )

                        y_revert = ss.inverse_transform(
                            snapshot.y.reshape(1,-1)
                        )
                                                
                        cost = cost + np.mean((y_hat_revert - y_revert) ** 2)

            cost = cost / (time + 1)
            cost = cost.item()
            # print("MSE: {:.4f}".format(cost))
            val_rmse = np.sqrt(cost)
            print("EPOCH: {}, RMSE: {:.4f}".format(epoch, val_rmse))
            val_history.append(val_rmse)
            # mlflow.log_artifacts(utils.to_absolute_path("configs"))
            mlflow.log_metric('rmse', val_rmse, step=epoch)
        best_rmse = np.min(val_history)
        mlflow.log_metric('best_rmse', best_rmse, step=epoch+1)
        return best_rmse

if __name__ == '__main__':
    main()
    


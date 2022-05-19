import os
import json
import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from datetime import datetime
from models.handler import train, test
import argparse
import pandas as pd
import hydra
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import get_original_cwd, to_absolute_path

wd = Path.cwd()

@hydra.main(config_path="../../conf", config_name="config.yml")
def main(cfg: DictConfig) -> float:
    args = cfg.args
    print(f'Training configs: {args}')
    # data_file = os.path.join('dataset', args.dataset + '.csv')
    # data_file = (
    #     wd / 'dataset' /  args.dataset
    #  ).with_suffix('.csv')
    data_file = Path(get_original_cwd()) / cfg.catalogue.model_in

    result_train_file = os.path.join('output', args.dataset, 'train')
    result_test_file = os.path.join('output', args.dataset, 'test')
    if not os.path.exists(result_train_file):
        os.makedirs(result_train_file)
    if not os.path.exists(result_test_file):
        os.makedirs(result_test_file)
    data = pd.read_csv(data_file, index_col=[0], parse_dates=[0])

    # split data
    # train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
    # valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
    # test_ratio = 1 - train_ratio - valid_ratio
    # train_data = data[:int(train_ratio * len(data))]
    # valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
    # test_data = data[int((train_ratio + valid_ratio) * len(data)):]

    train_data = data.loc[:"2019"].values
    valid_data = data.loc["2020"].values
    test_data = data.loc["2021"].values



    torch.manual_seed(0)
    if __name__ == '__main__':
        if args.train:
            try:
                before_train = datetime.now().timestamp()
                _, normalize_statistic = train(train_data, valid_data, args, result_train_file)
                after_train = datetime.now().timestamp()
                print(f'Training took {(after_train - before_train) / 60} minutes')
            except KeyboardInterrupt:
                print('-' * 99)
                print('Exiting from training early')
        if args.evaluate:
            before_evaluation = datetime.now().timestamp()
            mae, mape, rmse = test(test_data, args, result_train_file, result_test_file)
            after_evaluation = datetime.now().timestamp()
            print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
        print('done')

        summary = dict(mae=mae, mape=mape, rmse=rmse)
        with open(cfg.catalog.output.summary, 'w') as fp:
            json.dump(summary, fp)

        return rmse

if __name__ == "__main__":
    main()    
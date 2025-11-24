import subprocess as sp
import numpy as np
import argparse
from hp_search import call_command, hp_search
from src.train import get_args as get_train_args
import os
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='tensat')
    parser.add_argument('--time_limit', type=int, default=120)
    parser.add_argument('--method',
                        type=str,
                        default='smoothe',
                        choices=['cbc', 'scip', 'cplex', 'smoothe'])
    return parser.parse_args()


def get_hp(path):
    file_sizes = {}
    for file in os.listdir(path):
        if file.endswith('.json'):
            full_path = os.path.join(path, file)
            file_sizes[full_path] = os.path.getsize(full_path)

    # select the median size file as representative
    median_file = sorted(file_sizes, key=file_sizes.get)[len(file_sizes) // 2]
    print(f'Selected {median_file} for hyper-parameter search')
    hp = hp_search(median_file)
    return hp


def launch(path, method, hp=None):
    all_logs = {}
    if method == 'smoothe':
        assert hp is not None
        train_args = get_train_args(default=True)
        train_args.acyclic = True
        for key, value in hp.items():
            setattr(train_args, key, value)

    for file in os.listdir(path):
        if not file.endswith('.json'):
            continue
        print(f'running on {file}')
        if method == 'smoothe':
            train_args.input_file = os.path.join(path, file)
            log = call_command(train_args)

            if log is None:
                min_loss = None
                time = None
            else:
                min_loss = min(log['inference_loss'])
                min_iter = np.argmin(log['inference_loss'])
                time = log['time'][min_iter]
            print(f'File: {file}, Min Loss: {min_loss}, Time: {time}')
            all_logs[file] = log

        elif method in ['cplex', 'cbc', 'scip']:
            command = 'python src/ilp.py' + f' --time_limit {time_limit}'
            command += f' --input_file {os.path.join(path, file)} --solver {method}'
            command += ' --acyclic '

            with open(f'logs/{dataset}_{file}_{method}.log', 'w') as f:
                sp.run(command, shell=True, stdout=f, stdin=f)

    if method == 'smoothe':
        os.makedirs('logs', exist_ok=True)
        dataset = os.path.join(path, 'result').replace('/', '_')
        file_path = os.path.join('logs', f'{dataset}.json')
        json.dump(all_logs, open(file_path, 'w'))
        print(f'All logs saved to {file_path}')


if __name__ == "__main__":
    args = get_args()
    hp = get_hp(args.path)
    hp = {'assumption': 'hybrid', 'sample_freq': 1, 'patience': 40}
    launch(args.path, args.method, hp)

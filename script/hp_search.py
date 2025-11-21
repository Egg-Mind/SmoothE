import subprocess as sp
import argparse
import os
import torch
import numpy as np
from src.train import run
from src.train import get_args as get_train_args


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        type=str,
        default='examples/gym_data/tensat/cyclic/resnet50.json')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--acyclic', action='store_true', default=False)
    return parser.parse_args()


def call_command(args):
    try:
        log = run(args)
    except RuntimeError as e:
        # print("Caught CUDA Error: Insufficient resources")
        if args.batch_size is None:
            args.batch_size = 64
        elif args.batch_size > 1:
            args.batch_size = args.batch_size // 2
            # print(f"Halfing batch size to {args.batch_size}")
        else:
            return None
        log = call_command(args)
        args.batch_size = None
    except ValueError as ve:
        # print(f"Caught ValueError: {str(ve)}")
        return None
    except Exception as ex:
        # print(f"Caught an unexpected exception: {str(ex)}")
        return None
    return log


def hp_search(file):
    best_hp = None
    best_loss = float('inf')
    best_time = float('inf')

    assumptions = ['independent', 'hybrid']
    for assumption in assumptions:
        for sample_freq in [1, 10]:
            train_args = get_train_args(default=True)
            train_args.num_steps = 1000
            train_args.input_file = file

            train_args.assumption = assumption
            train_args.time_limit = 30
            train_args.optimizer = 'rmsprop'
            train_args.base_lr = 1e-2
            train_args.regularizer = 1e-2
            train_args.acyclic = True
            if sample_freq == 1:
                train_args.sample_freq = 1
                train_args.patience = 40
            else:
                train_args.sample_freq = 10
                train_args.patience = 20

            print(
                f'Trying assumption: {assumption}, sample_freq: {sample_freq}')
            log = call_command(train_args)
            if log is None:
                continue
            min_loss = min(log['inference_loss'])
            min_iter = np.argmin(log['inference_loss'])
            time = log['time'][min_iter]
            print(f'Min loss: {min_loss:.2f}, time: {time:.1f}')

            if (min_loss < best_loss) or (min_loss == best_loss
                                          and time < best_time):
                best_hp = {
                    'assumption': assumption,
                    'sample_freq': sample_freq,
                    'patience': 40 if sample_freq == 1 else 20
                }
                best_loss = min_loss
                best_time = time
    print(f'Best hyperparameters: {best_hp}, loss: {best_loss}')


if __name__ == '__main__':
    args = get_args()
    hp = hp_search(args.input_file)

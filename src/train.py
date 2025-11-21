import logging
import random
import argparse
import torch
from collections import defaultdict
from tqdm import tqdm
import os
import time
import json
import numpy as np

from src.sparse_egraph import SparseEGraph


def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def get_args(default=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        type=str,
                        default='examples/cunxi_test_egraph2.dot')
    parser.add_argument('--num_steps', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--time_limit', type=int, default=120)
    parser.add_argument('--random_seed', type=int, default=44)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--sample_freq', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--gumbel_tau', type=float, default=1)
    parser.add_argument('--base_lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--eps", type=float, default=3.0)
    parser.add_argument("--regularizer", type=float, default=1e-2)
    parser.add_argument('--optimizer', type=str, default='rmsprop')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--assumption',
                        type=str,
                        default='independent',
                        choices=['independent', 'correlated', 'hybrid'])
    parser.add_argument('--acyclic', action='store_true', default=False)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument("--greedy_ini", action="store_true", default=False)
    parser.add_argument("--ilp", action="store_true", default=False)
    parser.add_argument('--quad_cost',
                        type=str,
                        default=None,
                        help='path to the quadratic cost file')
    parser.add_argument('--mlp_cost',
                        type=str,
                        default=None,
                        help='path to the mlp cost file')
    if default:
        return parser.parse_args([])
    else:
        return parser.parse_args()


def sample(egraph, verbose=False):
    egraph.eval()
    start_time = time.time()

    with torch.no_grad():
        enodes = egraph(egraph.embedding)
        loss = egraph.compute_loss(enodes, verbose=verbose)
        logging.info(f'Sample loss: {loss:.4f}')
        logging.info(f'sampling time: {time.time() - start_time:.4f}')
    egraph.step()
    egraph.train()
    return loss, enodes.bool()


class EarlyStopper:

    def __init__(self, patience=3, min_delta=0.1):
        self.patience = patience
        self.best_loss = float('inf')
        self.count = 0
        self.min_delta = min_delta

    def __call__(self, loss):
        if self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience


def _build_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    args = {'params': params, 'lr': lr, 'weight_decay': weight_decay}

    optimizers = {
        'momentum':
        lambda: torch.optim.SGD(**args, nesterov=True, momentum=0.9),
        'sgd': lambda: torch.optim.SGD(**args),
        'adam': lambda: torch.optim.Adam(**args),
        'adamw': lambda: torch.optim.AdamW(**args),
        'sparse_adam': lambda: torch.optim.SparseAdam(**args),
        'rmsprop': lambda: torch.optim.RMSprop(**args),
    }
    return optimizers[name]()


def run(args):
    start_time = time.time()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    set_random_seed(seed_value=args.random_seed)

    if args.gpus > 0:
        assert args.gpus == 1
        device = 'cuda'
    else:
        device = 'cpu'

    egraph = SparseEGraph(args.input_file,
                          hidden_dim=args.hidden_dim,
                          batch_size=args.batch_size,
                          gumbel_tau=args.gumbel_tau,
                          soft=True,
                          device=device,
                          gpus=args.gpus,
                          filter_cycles=args.acyclic,
                          eps=args.eps,
                          greedy_ini=args.greedy_ini,
                          assumtion=args.assumption)
    if args.gpus > 0:
        egraph = egraph.cuda()
    egraph.set_temperature_schedule(args.num_steps)
    egraph.reg = args.regularizer
    if args.quad_cost:
        egraph.init_quad_cost(args.quad_cost)
    if args.mlp_cost:
        egraph.init_mlp_cost(args.mlp_cost)

    lr = args.base_lr
    params_to_optimize = [p for p in egraph.parameters() if p.requires_grad]

    optimizer = _build_optimizer(args.optimizer, params_to_optimize, lr,
                                 args.weight_decay)
    training_log = defaultdict(list)
    logging.info(f'cost per node {egraph.cost_per_node}')
    early_stop = EarlyStopper(patience=args.patience)

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    if args.verbose:
        for_loop = tqdm(range(args.num_steps))
    else:
        for_loop = range(args.num_steps)

    for step in for_loop:
        # inference sampling at every sample_freq steps
        if step % args.sample_freq == 0:
            inf_loss, _ = sample(egraph)
        training_log['sample_time'].append(time.time() - start_time)

        # optimization step
        enodes, cyclic_loss = egraph(egraph.embedding)
        training_log['forward_time'].append(time.time() - start_time)
        loss = egraph.compute_loss(enodes, cyclic_loss)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if step % args.sample_freq == 0:
            training_log['inference_loss'].append(inf_loss.item())
            training_log['loss'].append(loss.item())
            training_log['time'].append(time.time() - start_time)
        if time.time() - start_time > args.time_limit:
            logging.info('time limit reached')
            break

        if step > 1 and inf_loss < 1e8 and step % args.sample_freq == 0:
            if early_stop(inf_loss):
                break

    # final sampling
    logging.info(f'finished optimization, now sampling')
    loss, enodes = sample(egraph, verbose=args.verbose)

    # save logs
    training_log['time'].append(time.time() - start_time)
    training_log['loss'].append(loss.item())
    training_log['solution'] = egraph.node_to_id(enodes)
    logging.info(f'training log: {training_log}')
    file_name = os.path.splitext(os.path.basename(
        args.input_file))[0] + '_smoothe'
    json.dump(training_log, open(f'logs/smoothe_log/{file_name}.json', 'w'))
    logging.info('logs dumped to ' + f'logs/smoothe_log/{file_name}.json')
    logging.info(
        f'best inference loss = {np.min(training_log["inference_loss"])}')
    logging.info(
        f'best inference time = {training_log["time"][np.argmin(training_log["inference_loss"])]}'
    )
    return training_log


if __name__ == '__main__':
    args = get_args()
    run(args)

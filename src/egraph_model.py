import re
import os
import json
import logging
from collections import defaultdict
from .egraph_data import EGraphData
from .dag_greedy import greedy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, t


class BatchedLinear(nn.Module):

    def __init__(self, batch_size, in_features, out_features):
        super().__init__()
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.rand(batch_size, in_features, out_features))
        self.bias = nn.Parameter(torch.rand(batch_size, out_features))

    def forward(self, x):
        return torch.bmm(x, self.weight) + self.bias


class BaseEGraph(nn.Module, EGraphData):

    def __init__(self,
                 input_file,
                 hidden_dim=32,
                 gumbel_tau=1.0,
                 dropout=0.0,
                 eps=0.5,
                 batch_size=None,
                 soft=False,
                 aggregate_type='mean',
                 device='cuda',
                 greedy_ini=False,
                 compress=False,
                 drop_self_loops=False,
                 share_proj=True):
        nn.Module.__init__(self)
        EGraphData.__init__(
            self,
            input_file,
            hidden_dim,
            compress,
            drop_self_loops,
            device,
        )
        self.hidden_dim = hidden_dim
        self.gumbel_tau = gumbel_tau
        self.dropout = dropout
        self.eps = eps
        self.aggregate_type = aggregate_type
        self.device = device
        self.minimal_cost = None
        self.soft = soft
        self.p_number = np.e
        self.quadratic_cost = None
        self.greedy_ini = greedy_ini
        self.input_file = input_file
        self.share_proj = share_proj

        if batch_size is None:
            num_class = len(self.eclasses)
            if num_class > 5000:
                batch_size = 40000 / num_class
            else:
                batch_size = 80000 / num_class
            self.batch_size = int(np.clip(2**int(np.log2(batch_size)), 1, 512))
            logging.info(f'Auto set batch size to {self.batch_size}')

        self.set_to_matrix()
        self.init_embedding()
        self.init_params()
        torch.cuda.empty_cache()
        self.step_count = 0
        self.enode_attr = torch.arange(len(self.enodes)).to(self.device)

    def set_adj(self):
        # [N, M]
        # Enode adjacency matrix points from enode to eclass
        self.node_adj = torch.zeros((len(self.enodes), len(self.eclasses)),
                                    dtype=torch.bool,
                                    device=self.device)

        # [M, N]
        # eclass adjacency matrix contains the enodes in the eclass
        self.class_adj = torch.zeros((len(self.eclasses), len(self.enodes)),
                                     dtype=torch.bool,
                                     device=self.device)

        for enode_id in self.enodes:
            eclass_id = [i for i in self.enodes[enode_id].eclass_id]
            self.node_adj[enode_id, eclass_id] = 1

        for eclass_id in self.eclasses:
            self.class_adj[eclass_id, self.eclasses[eclass_id].enode_id] = 1

    def set_to_matrix(self):
        # set the class2node and node2class in a sparse COO format
        n2c_row_index = []
        n2c_col_index = []
        for enode_id in self.enodes:
            eclass_id = [i for i in self.enodes[enode_id].eclass_id]
            n2c_row_index += [enode_id] * len(eclass_id)
            n2c_col_index += eclass_id
        n2c_row_index = torch.tensor(n2c_row_index).to(self.device)
        n2c_col_index = torch.tensor(n2c_col_index).to(self.device)
        self.node2class = SparseTensor(row=n2c_row_index,
                                       col=n2c_col_index,
                                       value=torch.ones(len(n2c_row_index),
                                                        device=self.device),
                                       sparse_sizes=(len(self.enodes),
                                                     len(self.eclasses)))

        c2n_row_index = []
        c2n_col_index = []
        for eclass_id in self.eclasses:
            enode_id = self.eclasses[eclass_id].enode_id
            c2n_row_index += [eclass_id] * len(enode_id)
            c2n_col_index += enode_id
        c2n_row_index = torch.tensor(c2n_row_index).to(self.device)
        c2n_col_index = torch.tensor(c2n_col_index).to(self.device)
        self.class2node = SparseTensor(row=c2n_row_index,
                                       col=c2n_col_index,
                                       value=torch.ones(len(c2n_row_index),
                                                        device=self.device),
                                       sparse_sizes=(len(self.eclasses),
                                                     len(self.enodes)))

    def init_embedding(self):
        self.embedding = torch.rand(self.batch_size,
                                    len(self.enodes),
                                    self.hidden_dim,
                                    device=self.device)
        self.embedding = self.embedding / np.sqrt(self.hidden_dim)
        if self.greedy_ini:
            self.bias = torch.zeros(len(self.enodes)).to(self.device)
            greedy_idx_list = greedy(self, method="faster", ini_greedy=True)
            self.bias[greedy_idx_list] += self.eps
        self.embedding = nn.Parameter(self.embedding, requires_grad=True)

    def init_params(self):
        if self.share_proj:
            self.enode_proj = torch.nn.Linear(in_features=self.hidden_dim,
                                              out_features=self.hidden_dim)
            self.output_proj = torch.nn.Linear(in_features=self.hidden_dim,
                                               out_features=1)
        else:
            self.node_proj = BatchedLinear(self.batch_size, self.hidden_dim,
                                           self.hidden_dim)
            self.output_proj = BatchedLinear(self.batch_size, self.hidden_dim,
                                             1)
        self.activation = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.ReLU(),
        )
        self.dropout = torch.nn.Dropout(p=self.dropout)

    def set_temperature_schedule(self, steps, schedule='constant'):
        if schedule == 'constant':
            self.temperature_schedule = np.full(steps, 1)
        if schedule == 'linear':
            self.temperature_schedule = np.linspace(1, 1e-3, steps)
        elif schedule == 'log':
            self.temperature_schedule = np.logspace(0, -2, steps)

    def dense(self, enode_embedding, context_embedding):
        # enode_embedding: [n, hidden_dim]
        # context_embedding: [1, hidden_dim]
        # return: [n, hidden_dim]
        return self.enode_proj(enode_embedding) + self.context_proj(
            context_embedding)

    def projection(self, enode_embedding):
        # enode_embedding: [n, hidden_dim]
        # return: [n]
        return self.output_proj(enode_embedding).squeeze(-1)

    def forward_embedding(self, embedding):
        logit = self.activation(self.enode_proj(embedding))
        logit = self.projection(self.dropout(logit))
        if self.greedy_ini and not self.training and self.step_count == 0:
            logit[0] = logit[0] + self.bias
        return logit

import logging
import numpy as np
import networkx as nx
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .egraph_model import BaseEGraph
from networkx.algorithms.approximation.treewidth import (treewidth_min_fill_in,
                                                         treewidth_min_degree)

import torch_sparse
from torch_sparse import SparseTensor, t
from torch_sparse import max as spmax
from torch_sparse import sum as spsum
from torch_sparse import min as spmin
from torch_geometric.utils import softmax
from .exp_m import sparse_expm, dense_expm

spmm = torch_sparse.matmul


def sparse_gumbel_softmax(src,
                          row,
                          col,
                          shape,
                          tau=1,
                          hard=False,
                          dim=-1,
                          return_format='torch_sparse'):
    # Generate Gumbel noise for non-zero elements in logits
    gumbel_noise = -torch.empty_like(src).exponential_().log()
    perturbed_logits = (src + gumbel_noise) / tau
    # Apply sparse softmax to the perturbed logits
    if dim == 1:
        y_soft = softmax(perturbed_logits.flatten(), index=row)
    elif dim == 0:
        y_soft = softmax(perturbed_logits.flatten(), index=col)
    else:
        raise ValueError

    # the shape [BM, BN], implicitly reshape it to [BN, BM] here
    if return_format == 'torch_sparse':
        ret = SparseTensor(row=col, col=row, value=y_soft, sparse_sizes=shape)
    elif return_format == 'torch':
        ret = torch.sparse_coo_tensor(indices=torch.stack([col, row]),
                                      values=y_soft,
                                      size=shape)

    if hard:
        _, row_count = torch.unique_consecutive(row, return_counts=True)
        max_per_row = spmax(ret, dim=0)
        max_per_col = torch.repeat_interleave(max_per_row, row_count)
        max_mask = (y_soft == max_per_col)
        hard_ret = SparseTensor(row=col[max_mask],
                                col=row[max_mask],
                                value=torch.ones(max_mask.sum(),
                                                 device=src.device),
                                sparse_sizes=shape)
        neg_ret = SparseTensor(row=col,
                               col=row,
                               value=-y_soft,
                               sparse_sizes=shape)
        ret = hard_ret + neg_ret.detach() + ret
    return ret


class SparseEGraph(BaseEGraph):

    def __init__(
        self,
        input_file,
        batch_size=None,
        hidden_dim=32,
        gumbel_tau=1,
        dropout=0.0,
        eps=0.5,
        soft=False,
        aggregate_type='mean',
        assumtion='hybrid',
        device='cuda',
        gpus=1,
        filter_cycles=False,
        greedy_ini=False,
        compress=True,
        drop_self_loops=True,
    ):
        """
        Notations used in the comment of this class:
        N: number of nodes
        M: number of classes
        H: hidden dimension
        """
        self.batch_size = batch_size
        self.filter_cycles = filter_cycles
        super().__init__(input_file,
                         hidden_dim,
                         gumbel_tau,
                         dropout,
                         eps,
                         batch_size,
                         soft,
                         aggregate_type,
                         device,
                         greedy_ini,
                         compress=compress,
                         drop_self_loops=drop_self_loops)

        self.gpus = max(gpus, 1)
        assert self.batch_size % self.gpus == 0

        B, M, N = self.batch_size, len(self.eclasses), len(self.enodes)
        self.assumption = assumtion

        # [N, M], the classes pointed by the node
        self.node2classT = t(self.node2class)

        self.cyclic_loss = torch.tensor([0.0], device=self.device)
        self.cyclic_count = 0
        self.cyclic_rho = 0.1
        self.cyclic_lambda = 1e-4
        self.set_index()
        self.known_cycles = []
        self._init_cycle_detection_data()

        # for n2n
        # self.get_edge_masks(
        #     (self.node2class @ self.class2node).to_torch_sparse_coo_tensor())

        # for c2c
        self.get_edge_masks(
            (self.class2node @ self.node2class).to_torch_sparse_coo_tensor())

        self.order_var = nn.Parameter(
            torch.full((B, M), 0.5, device=self.device))

    @torch.no_grad()
    def _init_cycle_detection_data(self):
        """
        Pre-computes and caches graph structures in CPU-friendly formats
        for sequential cycle detection.
        """
        N = len(self.enodes)
        M = len(self.eclasses)

        # 1. Create node-to-class mapping (np.array)
        # We build this from self.class2node, which is [M, N]
        # Ensure self.class2node is on the correct device first
        c2n_row = self.class2node.storage._row
        c2n_col = self.class2node.storage._col

        # Find the device of the tensor, default to cpu if not specified
        tensor_device = c2n_row.device if c2n_row is not None else torch.device(
            'cpu')

        n2c_map_tensor = torch.empty(N, dtype=torch.long, device=tensor_device)
        n2c_map_tensor[c2n_col] = c2n_row

        # Move to CPU as numpy for fast Python access
        self.cycle_node_to_class_map = n2c_map_tensor.cpu().numpy()

        # 2. Create node-to-children mapping (list of lists)
        # This is faster to access in pure Python than self.enodes[...].eclass_id
        self.cycle_node_to_children = [
            list(self.enodes[i].eclass_id) for i in range(N)
        ]

        # 3. Ensure roots are processed
        self.set_root()  # self.root is now guaranteed to exist

        logging.info("Pre-computed data for sequential cycle detection.")

    @torch.no_grad()
    def set_index(self):
        B, M, N = self.batch_size, len(self.eclasses), len(self.enodes)
        # [N], the class consists of the node
        # class_per_node_backup = self.class_adj.nonzero()[:, 0]
        class_per_node = self.class2node.storage._row.clone()
        self.class_per_node = nn.Parameter(class_per_node.repeat(B).to(
            self.device),
                                           requires_grad=False)
        self.batch_per_node = torch.repeat_interleave(
            torch.arange(B, device=self.device), N)
        self.node_per_node = self.class2node.storage._col.clone().repeat(B)

        # the eclass indices for each node in the batches
        self.index0 = nn.Parameter(self.batch_per_node * M +
                                   self.class_per_node,
                                   requires_grad=False)
        # the enode indices for each node in the batches
        self.index1 = nn.Parameter(self.batch_per_node * N +
                                   self.node_per_node,
                                   requires_grad=False)

        # A sparse tensor that maps nodes to their dependent classes, [N, M]
        self.node2class = self.node2class.to(self.device)
        row = self.node2class.storage._row
        col = self.node2class.storage._col
        value = self.node2class.storage._value
        nnz = row.numel()
        batch_index = torch.arange(B,
                                   device=self.device).repeat_interleave(nnz)

        # A sparse tensor that maps batched nodes to their dependent classes
        # for all batches, [BN, BM]. This tensor should have been [B, N, M],
        # but there's no efficient implementation for batched sparse sparse
        # matmul in PyTorch or torch_sparse Thus we use a flattened
        # representation with shape [BN, BM]. It can be seen as a big
        # block-diagonal matrix with B blocks of [N, M] on the diagonal.
        # With the sparse representation, there is no memory overhead.
        self.batch_node2class = SparseTensor(
            row=row.repeat(B) + batch_index * N,
            col=col.repeat(B) + batch_index * M,
            value=value.repeat(B),
            sparse_sizes=(B * N, B * M))
        self.batch_node2classT = t(self.batch_node2class)

    def find_cycles(self, batch_choose_enodes):
        """
        Finds cycles in the batch of chosen e-graphs sequentially.
        Uses a lightweight DFS for speed.
        """
        # for n2n only
        # if len(self.edge_masks) == 0:
        #     if len(self.two_hop_indices_A) == 0:
        #         return torch.zeros(self.batch_size, device=self.device)
        #     else:
        #         # [B, N]
        #         batch_choose_eclasses = batch_choose_enodes @ t(
        #             self.class2node).to_torch_sparse_coo_tensor()
        #         cycles_A = batch_choose_eclasses[:, self.two_hop_indices_A]
        #         cycles_B = batch_choose_eclasses[:, self.two_hop_indices_B]
        #         cycles_per_batch = torch.einsum('bn,bn->b', cycles_A, cycles_B)
        #         # cycles_per_batch = batch_choose_eclasses[:, self.
        #         #                                          two_hop_cycles].prod(
        #         #                                              dim=1).sum()
        #         return cycles_per_batch

        # 1. Ensure cycle detection data is ready
        if not hasattr(self, 'cycle_node_to_class_map'):
            self._init_cycle_detection_data()

        # 2. Move data to CPU for processing
        try:
            batch_choose_enodes_cpu = batch_choose_enodes.cpu().bool().numpy()
        except RuntimeError:
            # Fallback if it's already a numpy array or list
            batch_choose_enodes_cpu = np.array(batch_choose_enodes, dtype=bool)

        batch_cycles = []
        batch_cycle_num = []

        # Pre-fetch reusable class attributes
        node_to_class_map = self.cycle_node_to_class_map
        node_to_children = self.cycle_node_to_children
        root_classes = self.root

        # 3. Iterate sequentially over the batch
        for batch in range(self.batch_size):
            chosen_enodes_mask = batch_choose_enodes_cpu[batch]

            enodes = np.where(chosen_enodes_mask)[0]
            if enodes.shape[0] == 0:
                batch_cycle_num.append(0)
                batch_cycles.append([])
                continue

            chosen_classes = node_to_class_map[enodes]
            cls_node_dict = dict(zip(chosen_classes, enodes))
            cycles = [
            ]  # This just stores a list of *one* node from each cycle

            # --- Lightweight DFS: No stack, just detection ---
            # Status: 0 = Todo, 1 = Doing, 2 = Done
            status = {}

            def cycle_dfs_light(class_id):
                status[class_id] = 1  # 1 = Doing

                node_id = cls_node_dict.get(class_id)
                if node_id is None:
                    status[class_id] = 2  # 2 = Done
                    return

                for child in node_to_children[node_id]:
                    child_status = status.get(child, 0)  # 0 = Todo

                    if child_status == 0:  # Todo
                        cycle_dfs_light(child)
                    elif child_status == 1:  # Doing
                        # Cycle detected!
                        cycles.append(class_id)  # Just log one part of it

                status[class_id] = 2  # 2 = Done

            # Run the light DFS
            for root in root_classes:
                if root not in status:
                    cycle_dfs_light(root)

            # --- End of single batch logic ---
            batch_cycle_num.append(len(cycles))
        batch_cycle_num = torch.tensor(batch_cycle_num, device=self.device)
        return batch_cycle_num

    def set_root(self):
        # set the root e-classes
        if hasattr(self, 'vector_root'):
            return self.vector_root
        if hasattr(self, 'root'):
            if isinstance(self.root, int):
                self.root = [self.root]
            elif isinstance(self.root, list):
                self.root = [self.class_mapping[r] for r in self.root]
            else:
                raise NotImplementedError
            self.vector_root = torch.zeros(len(self.eclasses),
                                           dtype=torch.bool,
                                           device=self.device)
            self.vector_root[self.root] = 1
        else:
            self.vector_root = torch.ones(len(self.eclasses),
                                          dtype=torch.bool,
                                          device=self.device)
            self.vector_root[self.node2class.storage._col] = 0
            self.root = self.vector_root.nonzero().squeeze().tolist()
            self.root = np.atleast_1d(self.root)
        return self.vector_root

    def compute_treewidth(self):
        # auxiliary function to compute treewidth of the e-graph
        c2n = self.class2node
        n2c = self.node2class

        M = c2n.size(dim=0)
        N = c2n.size(dim=1)
        c2n_indices = torch.stack([c2n.storage._row, c2n.storage._col], dim=0)
        n2c_indices = torch.stack([n2c.storage._row, n2c.storage._col], dim=0)
        c2n_indices[1] += M
        n2c_indices[0] += M
        n2n = torch.sparse_coo_tensor(
            indices=torch.cat([c2n_indices, n2c_indices], dim=1),
            values=torch.cat([c2n.storage._value, n2c.storage._value], dim=0),
            size=(M + N, M + N))
        n2n = n2n.coalesce()

        scipy_n2n = scipy.sparse.coo_matrix(
            (n2n.values().cpu().bool().numpy(),
             (n2n.indices()[0].cpu().numpy(), n2n.indices()[1].cpu().numpy())),
            shape=(N + M, N + M))
        G = nx.from_scipy_sparse_array(scipy_n2n)
        tw1, _ = treewidth_min_fill_in(G)
        tw2, _ = treewidth_min_degree(G)
        print(f'treewidth_min_fill_in: {tw1}, treewidth_min_degree: {tw2}')

    @torch.no_grad()
    def step(self):
        if self.training:
            self.gumbel_tau = self.temperature_schedule[self.step_count]
            self.step_count += 1
        self.cyclic_loss.zero_()

    def init_params(self):
        # initialize the trainable parameters
        self.enode_proj = torch.nn.Linear(in_features=self.hidden_dim,
                                          out_features=self.hidden_dim)
        self.output_proj = torch.nn.Linear(in_features=self.hidden_dim,
                                           out_features=1)

        self.activation = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim), torch.nn.ReLU())
        self.dropout = torch.nn.Dropout(p=self.dropout)

    def forward_embedding(self, embedding):
        if embedding.shape[-1] == 1:
            return embedding.squeeze(-1)
        else:
            return super().forward_embedding(embedding)

    def dense(self, enode_embedding):
        # enode_embedding: [n, hidden_dim]
        # return: [n, hidden_dim]
        return self.enode_proj(enode_embedding)

    def optimize_sample(self, embedding, hard=False):
        # A fully differentiable sampling function that computes the
        # node probabilities given their embeddings.
        B, M, N = self.batch_size // self.gpus, len(self.eclasses), len(
            self.enodes)
        eps = 1e-10
        device = embedding.device
        node_prob = torch.zeros((B, N), device=device, requires_grad=False)
        node_logits = self.forward_embedding(embedding)  # [B, N]

        # compute the probs from class to node, [BN, BM]
        probs_class2node = sparse_gumbel_softmax(node_logits,
                                                 row=self.index0[:B * N],
                                                 col=self.index1[:B * N],
                                                 shape=(B * N, B * M),
                                                 dim=1,
                                                 tau=self.gumbel_tau,
                                                 hard=hard,
                                                 return_format='torch_sparse')

        if self.filter_cycles and self.cyclic_count > 0:
            # reduce the batch dimension on the class dimension
            reshaped_probs_class2node = probs_class2node.clone()
            reshaped_probs_class2node.storage._col %= M
            reshaped_probs_class2node.storage._sparse_sizes = (B * N, M)
            # reshape to [BN, M], then compute cyclic loss
            self.compute_cyclic_loss(reshaped_probs_class2node)

        # compute the trainsition matrix from class to class, [BM, BM]
        c2c = t(probs_class2node).to_torch_sparse_coo_tensor(
        ) @ self.batch_node2class.to_torch_sparse_coo_tensor()
        indices = c2c.indices()
        c2c_values = c2c.values()

        root_prob = self.set_root().float().to(device)
        root_prob = root_prob.unsqueeze(0).expand(B, M).flatten()

        # use the cached prob if exists to accelerate convergence
        if hasattr(self, 'cache_prob'):
            initial_prob = self.cache_prob
        else:
            initial_prob = root_prob
        class_prob = initial_prob.clone()

        # initialize the class probability
        if self.assumption in ['correlated', 'hybrid']:
            correlated_prob = SparseTensor(row=indices[0],
                                           col=indices[1],
                                           value=torch.empty_like(c2c_values),
                                           sparse_sizes=(B * M, B * M))
        if self.assumption in ['independent', 'hybrid']:
            independent_prob = SparseTensor(row=indices[0],
                                            col=indices[1],
                                            value=torch.empty_like(c2c_values),
                                            sparse_sizes=(B * M, B * M))

        prev_prob = None
        i = 0
        while True:
            i += 1
            if i == 0:
                cur_class_prob = initial_prob
            else:
                # value contains the unconditional eclass probability. it
                # converges when class_prob = c2c_values * class_prob[indices[0]]
                # , where indices[0] is the source eclass index
                value = c2c_values * class_prob[indices[0]]

                cur_class_prob = 0
                if self.assumption in ['correlated', 'hybrid']:
                    # assume all eclasses are correlated
                    correlated_prob.storage._value = value
                    cur_class_prob += correlated_prob.max(dim=0)
                if self.assumption in ['independent', 'hybrid']:
                    # assume all eclasses are independent
                    value = torch.log((1 - value).clamp(min=eps))
                    independent_prob.storage._value = value
                    cur_class_prob += 1 - torch.exp(
                        independent_prob.sum(dim=0))
                if self.assumption == 'hybrid':
                    cur_class_prob /= 2

            class_prob = cur_class_prob
            # Force root e-classes to have prob. 1
            class_prob = torch.maximum(class_prob, root_prob)

            # check convergence
            if i > 1:
                if torch.allclose(class_prob, prev_prob, atol=1e-4):
                    logging.info(f'converged at {i} iter')
                    break
                elif i > 1000:
                    logging.info(f'not converged at {i} iter')
                    break
            prev_prob = class_prob.detach().clone()

        # [BN, BM] @ [BM, 1] -> [BN, 1] -> [B, N]
        node_prob = probs_class2node @ class_prob.view(-1, 1)
        self.cache_prob = class_prob.detach().clone()
        return node_prob.view(B, N), self.cyclic_loss.unsqueeze(0)

    @torch.no_grad()
    def inference_sample(self, embedding):

        def update_inference_class2node(node_logits):
            # Switch to deterministic sample during inference using max
            logits_class2node = SparseTensor(row=self.index0,
                                             col=self.index1,
                                             value=node_logits,
                                             sparse_sizes=(B * M, B * N))
            _, row_count = torch.unique_consecutive(self.index0,
                                                    return_counts=True)
            max_per_col = spmax(logits_class2node, dim=1)
            max_per_row = torch.repeat_interleave(max_per_col, row_count)
            max_mask = (node_logits == max_per_row)
            # implicitly reshape [BM, BN] -> [BN, BM]
            # it maps eclass to its most probable enode
            class2node = SparseTensor(row=self.index1[max_mask],
                                      col=self.index0[max_mask],
                                      value=torch.ones(
                                          max_mask.sum(),
                                          device=embedding.device),
                                      sparse_sizes=(B * N, B * M))
            return class2node, row_count

        B, M, N = self.batch_size, len(self.eclasses), len(self.enodes)
        visited_classes = self.set_root().float()
        # [M] -> [B, M]
        visited_classes = visited_classes.repeat(B).unsqueeze(1)
        visited_nodes = torch.zeros(B * N,
                                    1,
                                    dtype=torch.float,
                                    device=embedding.device)

        node_logits = self.forward_embedding(embedding)  # [B, N]

        # [BM, BN]
        node_logits = node_logits.flatten()
        class2node, row_count = update_inference_class2node(node_logits)

        node_count = 0
        i = 0
        while True:
            i += 1
            # propagate from class to node
            visited_nodes += class2node @ visited_classes
            visited_nodes.clamp_(0, 1)

            # propagate from node to class
            visited_classes += self.batch_node2classT @ visited_nodes
            visited_classes.clamp_(0, 1)

            # check convergence
            current_node_count = visited_nodes.sum()
            if node_count == current_node_count:
                break
            node_count = current_node_count
        return visited_nodes.view(B, N)

    def find_scc(self, n2n):
        scc_labels = scipy.sparse.csgraph.connected_components(
            n2n, connection='strong')
        return scc_labels

    def get_edge_masks(self, n2n):

        # --- 1. Setup & Coalesce ---
        N = n2n.shape[0]
        self.nnz = n2n._nnz()

        # Call coalesce() only ONCE
        coalesced_n2n = n2n.coalesce()
        self.row = coalesced_n2n.indices()[0]
        self.col = coalesced_n2n.indices()[1]
        values = coalesced_n2n.values()
        tensor_device = self.row.device  # Get the GPU device

        # Initialize attributes
        self.edge_masks = []
        self.two_hop_indices_A = []
        self.two_hop_indices_B = []
        self.two_hop_cycles = []
        components = []

        # Move tensor data to CPU for SciPy/NumPy
        row_np = self.row.cpu().numpy()
        col_np = self.col.cpu().numpy()
        values_np = values.cpu().bool().numpy()

        scipy_n2n = scipy.sparse.coo_matrix((values_np, (row_np, col_np)),
                                            shape=(N, N))

        # --- 2. Find SCC ---
        n_scc, scc_labels = self.find_scc(
            scipy_n2n)  # scc_labels is a NumPy array

        # --- 3. Vectorized Computation ---

        # A) Vectorize degree calculation
        #    This gets the size of *all* components at once.
        component_degrees = np.bincount(scc_labels, minlength=n_scc)

        # Check if there are non-trivial components
        if max(component_degrees) == 1:
            self.two_hop_cycles = torch.empty((0, 2),
                                              dtype=torch.long,
                                              device=tensor_device)
            return

        # B) Vectorize component node grouping
        all_nodes = np.arange(N)

        # Sort nodes by the component they belong to
        node_sort_idx = np.argsort(scc_labels)
        sorted_nodes = all_nodes[node_sort_idx]
        sorted_node_labels = scc_labels[node_sort_idx]

        # Find split points between component groups
        node_split_points = np.where(np.diff(sorted_node_labels) != 0)[0] + 1

        # Get arrays of nodes for each component
        # We use np.split to get a list of arrays
        nodes_per_component_list = np.split(sorted_nodes, node_split_points)

        # Map the component label (which might be non-contiguous) to its node list
        component_label_for_nodes = sorted_node_labels[np.concatenate(
            [[0], node_split_points])]
        component_dict = dict(
            zip(component_label_for_nodes, nodes_per_component_list))

        # C) Vectorize edge mask generation
        #    This finds all edges that stay *within* their own component.

        # Find the component label for the source and destination of *every* edge
        edge_src_labels = scc_labels[row_np]
        edge_dst_labels = scc_labels[col_np]

        # A boolean mask of all edges that are *intra-component*
        intra_scc_edge_mask = (edge_src_labels == edge_dst_labels)

        # Get the global indices (0 to nnz-1) of these edges
        all_edge_indices = np.arange(self.nnz)
        intra_scc_edge_indices = all_edge_indices[intra_scc_edge_mask]

        # Get the component label for each of these edges
        intra_scc_edge_labels = edge_src_labels[intra_scc_edge_mask]

        # Sort edge indices by the component they belong to
        edge_sort_idx = np.argsort(intra_scc_edge_labels)
        sorted_edge_indices = intra_scc_edge_indices[edge_sort_idx]
        sorted_edge_labels = intra_scc_edge_labels[edge_sort_idx]

        # Find split points
        edge_split_points = np.where(np.diff(sorted_edge_labels) != 0)[0] + 1

        # Get arrays of edge indices for each component
        edges_per_component_list = np.split(sorted_edge_indices,
                                            edge_split_points)

        # Map the component label to its edge-index list
        component_label_for_edges = sorted_edge_labels[np.concatenate(
            [[0], edge_split_points])]
        edge_group_dict = dict(
            zip(component_label_for_edges, edges_per_component_list))

        # --- 4. Cheap Assembly Loop ---

        for label in range(n_scc):
            # Get pre-computed degree
            degree = component_degrees[label]

            # Get pre-computed edge indices
            edge_indices_np = edge_group_dict.get(label)
            if edge_indices_np is None:
                # This component has no internal edges
                continue

            # Move indices to GPU
            edge_indices_torch = torch.from_numpy(edge_indices_np).to(
                tensor_device)

            if degree == 2:
                self.two_hop_cycles.append(edge_indices_torch)

                # Get pre-computed node list (and convert to list as in original)
                # Only for degree == 2 components
                nodes_np = component_dict.get(label)
                if nodes_np is not None:
                    components.append(nodes_np.tolist())
            elif degree > 2:
                # We create the mask *only* for the few 'degree > 2' cases.
                edge_mask = torch.zeros(self.nnz,
                                        dtype=torch.bool,
                                        device=tensor_device)
                edge_mask.scatter_(0, edge_indices_torch, True)
                self.edge_masks.append((degree, edge_mask))

        # --- 5. Final Stack ---
        if len(self.two_hop_cycles) > 0:
            self.two_hop_cycles = torch.stack(self.two_hop_cycles)
        else:
            # Ensure the empty tensor is on the correct device
            self.two_hop_cycles = torch.empty((0, 2),
                                              dtype=torch.long,
                                              device=tensor_device)

        if len(components) != 0:
            components = torch.tensor(components,
                                      dtype=torch.long,
                                      device=tensor_device)
            self.two_hop_indices_A = components[:, 0]
            self.two_hop_indices_B = components[:, 1]
        else:
            self.two_hop_indices_A = torch.empty((0, ),
                                                 dtype=torch.long,
                                                 device=tensor_device)
            self.two_hop_indices_B = torch.empty((0, ),
                                                 dtype=torch.long,
                                                 device=tensor_device)

    def compute_cyclic_loss(self, probs_class2node):

        B, M, N = self.batch_size // self.gpus, len(self.eclasses), len(
            self.enodes)

        values = probs_class2node.storage.value()
        values = values.view(B, -1).mean(dim=0)
        row = probs_class2node.storage.row()
        col = probs_class2node.storage.col()
        nnz = row.numel() // B
        device = values.device

        probs_class2node = torch.sparse_coo_tensor(
            torch.stack([row[:nnz], col[:nnz]], dim=0), values, (N, M))

        # for n2n
        # n2n = self.node2class.to_torch_sparse_coo_tensor() @ probs_class2node.T

        # for c2c
        n2n = probs_class2node.T @ self.node2class.to_torch_sparse_coo_tensor()

        cyclic_loss = 0
        values = n2n.coalesce().values()
        indices = n2n.coalesce().indices()

        # handle cycles with more than 2 hops
        for degree, edge_mask in self.edge_masks:
            # multiply a constant to increase the numerical stability
            value = values[edge_mask]
            if degree > 5000:
                value *= 3
            # create a new sparse tensor for each SCC
            _, reverse_index = torch.unique(indices[:, edge_mask],
                                            return_inverse=True)
            scc = torch.sparse_coo_tensor(reverse_index, value,
                                          (degree, degree))
            expm = sparse_expm(scc)

            if degree > 5000:
                expm *= 1e-2
            cyclic_loss += expm

        # handle 2-hop cycles separately
        cyclic_loss += values[self.two_hop_cycles].prod(dim=1).sum()

        if cyclic_loss != 0:
            self.cyclic_loss = cyclic_loss
        else:
            logging.info('no cycles detected in training')

    def compute_loss(self,
                     enodes,
                     cyclic_loss=0,
                     backward=True,
                     optim_goal='sum',
                     debug=False,
                     verbose=False):
        assert optim_goal == 'sum'

        # A very large penalty for cycles during inference
        penalty = 1e8
        # map the enode selection back to raw enodes to remove the preprocessing
        raw_enodes = (self.raw2nodes @ enodes.T).T

        # use quadratic/mlp cost if exists
        if hasattr(self, 'quad_cost_mat'):
            loss = self.quad_cost(raw_enodes)
        elif hasattr(self, 'mlp'):
            loss = self.mlp_cost(raw_enodes)
        else:
            loss = self.linear_cost(raw_enodes)

        if self.training:
            loss = loss.mean()

            logging.info(f'Training loss mean: {loss.mean().item():.4f}')
            # impose cyclic loss during training when cycles are detected
            if self.filter_cycles and self.cyclic_count > 0 and (cyclic_loss
                                                                 > 0).any():
                # smart schedule for cyclic loss coefficient
                cyclic_loss = cyclic_loss.mean()
                if self.cyclic_count < 5:
                    cyclic_coef = self.cyclic_count**2
                else:
                    cyclic_coef = 2**self.cyclic_count
                cyclic_coef *= loss.abs().item() * self.reg
                loss += cyclic_loss * cyclic_coef

                logging.info(f'Cyclic loss = {cyclic_loss}')

        else:
            if self.filter_cycles:
                cycle_num = self.find_cycles(enodes)
                loss += cycle_num * penalty
                # smart schedule for cyclic loss coefficient
                if cycle_num.min() > 0:
                    self.cyclic_count = min(self.cyclic_count + 1, 30)
                elif cycle_num.median() == 0:
                    self.cyclic_count = max(self.cyclic_count - 1, 0)

                logging.info(f'cycles: {cycle_num.min()}')

            best_batch = loss.argmin()
            self.best_batch = best_batch
            loss = loss.min()
            if False:
                logging.info(
                    f'selected {self.node_to_id(raw_enodes[best_batch].bool())}'
                )
        return loss

    def forward(self, embedding):
        if self.training:
            return self.optimize_sample(embedding)
        else:
            return self.inference_sample(embedding)

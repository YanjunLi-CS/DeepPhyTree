# -*- coding: utf-8 -*-
"""
@author: Suncy

dataloader for Graph NN Models
"""

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from torchvision import transforms, utils
import os.path as osp
import dgl
import torch
import json
from dl import feat_dict, logger
from dgl.data import DGLDataset
from collections import Counter


class Dataset(DGLDataset):
    def __init__(self, args, phase, device="cpu"):
        self.device = device
        ds_folder = osp.join(args.ds_dir, args.ds_name, args.ds_split)

        if phase in ["train", "valid", "test"]:
            self.node_df = pd.read_csv(f"{ds_folder}/{phase}.csv", low_memory=False)
            self.edge_df = pd.read_csv(f"{ds_folder}/{phase}_edge.csv", low_memory=False)
        else:
            raise NotImplementedError

        self.tree_ids = self.node_df["sim"].unique()  # num of trees

        self.node_feat_cols = feat_dict[args.node_feat_cols]
        self.node_label_cols = args.node_label_cols
        self.edge_feat_cols = feat_dict[args.edge_feat_cols]
        self.n_label = len(feat_dict[args.node_label_cols.split("_cat")[0]])

        # Pre-process the bg nodes
        if args.pro_bg_nodes == "all_zero":
            self.node_df.loc[self.node_df["cluster_id"] == 'Background', self.node_feat_cols] = 0
        else:
            raise NotImplementedError

        self.add_self_loop = args.add_self_loop
        self.bidirection = args.bidirection

        # if phase == "train":
        #     self.transform = Transform(aug=True)
        # else:
        #     self.transform = Transform(aug=False)

    def process(self):        
        pass

    def __getitem__(self, index):
        tree_id = self.tree_ids[index]  # tree of index

        # dgl tree of index
        onetree_node_df = self.node_df[self.node_df['sim'] == tree_id]
        onetree_edge_df = self.edge_df[self.edge_df['sim'] == tree_id]
        src_ids = torch.tensor(onetree_edge_df['from'].values)
        dst_ids = torch.tensor(onetree_edge_df['to'].values)
        src_ids -= 1
        dst_ids -= 1
        g = dgl.graph((src_ids, dst_ids))  # create dgl
        sorted_onetree_node_df = onetree_node_df.sort_values(by='node')

        # assign features and labels for background nodes
        node_feat = sorted_onetree_node_df[self.node_feat_cols].values
        node_label = sorted_onetree_node_df[self.node_label_cols].values
        num_nodes = node_feat.shape[0]
        num_feat = node_feat.shape[1]

        # assign features for nodes and edges, assign labels
        g.ndata["feat"] = torch.tensor(node_feat, dtype=torch.float32)
        g.ndata["label"] = torch.tensor(node_label, dtype=torch.int64)
        g.edata["feat"] = torch.tensor(onetree_edge_df[self.edge_feat_cols].values, dtype=torch.float32)
        # wait for reading weight norm-asinh

        if self.add_self_loop:
            g = dgl.add_self_loop(g)  # TODO: Add self-loop with self-edge weight filled with zero
        if self.bidirection:
            g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)

        g = g.to(self.device)

        return g

    def __len__(self):
        return len(self.tree_ids)   # number of trees


# create batch of trees(aggregate multiples trees to a single tree)
def collate_fn(batch_graphs):
    g = dgl.batch(batch_graphs)
    return g


def gen_label_weight(args):
    # Get the weights for the unbalanced sample based on the positive sample
    # weights inversely proportional to class frequencies in the training data
    ds_folder = osp.join(args.ds_dir, args.ds_name, args.ds_split)
    node_df = pd.read_csv(f'{ds_folder}/train.csv')

    node_label = node_df[args.node_label_cols].values
    label_counter = Counter(node_label)
    n_samples = len(node_label)
    n_classes = len(label_counter)

    label_weights = [n_samples / (n_classes * label_counter[i]) for i in range(n_classes)]
    if args.loss_ignore_bg:
        label_weights[-1] = 0
    return label_weights


# class Transform(object):
#     def __init__(self, aug):
#         self.aug = aug
#
#     def __call__(self, graph, *args, **kwargs):
#         pass



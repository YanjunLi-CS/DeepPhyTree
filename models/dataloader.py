# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:49:45 2021

@author: Suncy

dataloader for Graph NN Models
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from torchvision import transforms, utils
import os.path as osp
import dgl
import torch
import json
from dgl.data import DGLDataset
from dgl.dataloading.pytorch import GraphDataLoader


class Dataset(DGLDataset):
    def __init__(self, args, phase):
        ds_folder = osp.join(args.ds_dir, args.ds_name, args.ds_split)

        if phase in ["train", "valid", "test"]:
            self.node_df = pd.read_csv(f'{ds_folder}/{phase}.csv')
            self.edge_df = pd.read_csv(f'{ds_folder}/{phase}_edge.csv')
        else:
            raise NotImplementedError

        self.tree_ids = self.node_df['sim'].unique()    # num of trees

        # load the feature dictionary
        js_f = osp.join('../aly/feat_dict.json')
        with open(js_f, 'r') as infile:
            feat_dict = json.load(infile)

        self.node_feat_cols = feat_dict[args.node_feat_cols]
        self.node_label_cols = args.node_label_cols
        self.edge_feat_cols = feat_dict[args.edge_feat_cols]
        self.n_label = len(self.node_df[self.node_label_cols].unique())

        # Pre-process the bg nodes
        if args.pro_bg_nodes == "all_zero":
            self.node_df.loc[self.node_df["cluster_id"] == 'Background', self.node_feat_cols] = 0
        else:
            raise NotImplementedError
        self.node_df.loc[self.node_df["cluster_id"] == 'Background', self.node_label_cols] = self.n_label

    def process(self):        
        pass

    def __getitem__(self, index):
        self.tree_id = self.tree_ids[index] # tree of index
        
        # dgl tree of index
        onetree_node_df = self.node_df[self.node_df['sim'] == self.tree_id] 
        onetree_edge_df = self.edge_df[self.edge_df['sim'] == self.tree_id]
        src_ids = torch.tensor(onetree_edge_df['from'].values)
        dst_ids = torch.tensor(onetree_edge_df['to'].values)
        src_ids -= 1
        dst_ids -= 1
        g = dgl.graph((src_ids, dst_ids)) # create dgl 
        sorted_onetree_node_df = onetree_node_df.sort_values(by='node')
        
        # assign features and labels for background nodes
        node_feat = sorted_onetree_node_df[self.node_feat_cols].values
        node_label = sorted_onetree_node_df[self.node_label_cols].values
        num_nodes = node_feat.shape[0]
        num_feat = node_feat.shape[1]

        # assign features for nodes and edges, assign labels
        g.ndata['feat'] = torch.tensor(node_feat)
        g.ndata['label'] = torch.tensor(node_label)
        g.edata['feat'] = torch.tensor(onetree_edge_df[self.edge_feat_cols].values) # wait for reading weight norm-asinh
        
        self._num_labels = 4
        self._labels = node_label
        self._g = g
        
        return self._g

    def __len__(self):
        return len(self.tree_ids) # number of trees


# create batch of trees(aggregate multiples trees to a single tree)
def collate_fn(batch):
    graphs = batch
    g = dgl.batch(graphs)
    return g




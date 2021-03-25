# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:49:45 2021

@author: Suncy

dataloader for Graph NN Models
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os.path as osp
import dgl
import torch
import json
from dgl.data import DGLDataset
from dgl.dataloading.pytorch import GraphDataLoader

# Input data files are available in the read-only "../data/" directory

# load train set and test set 
ds_name = '02172021'
ds_folder = f'../data/{ds_name}'

train_node_df = pd.read_csv(f'{ds_folder}/split_rs123/train.csv')
train_edge_df = pd.read_csv(f'{ds_folder}/split_rs123/train_edge.csv')
test_node_df = pd.read_csv(f'{ds_folder}/split_rs123/test.csv')
test_edge_df = pd.read_csv(f'{ds_folder}/split_rs123/test_edge.csv')


# load the feature dictionary
js_f = osp.join('../aly/feat_dict.json')
with open(js_f, 'r') as infile:
    feat_dict = json.load(infile)
aly_edge_feat_cols = ['weight1', 'weight2'] # todo: change to norm-asinh
label_feat_col = 'dynamic_cat'


class MyDataset(DGLDataset):
    def __init__(self, node_df, edge_df): # read node & edge data file
        self.node_df = node_df
        self.edge_df = edge_df
        
    def process(self):        
        pass

    def __getitem__(self, index):
        self.node_ids = self.node_df['sim'].unique() # num of trees
        self.edge_ids = self.edge_df['sim'].unique()
        self.tree_id = self.node_ids[index] # tree of index
        
        # dgl tree of index
        onetree_node_df = self.node_df[self.node_df['sim'] == self.tree_id] 
        onetree_edge_df = self.edge_df[self.edge_df['sim'] == self.tree_id]
        src_ids = torch.tensor(onetree_edge_df['from'].values)
        dst_ids = torch.tensor(onetree_edge_df['to'].values)
        src_ids -= 1
        dst_ids -= 1
        g = dgl.graph((src_ids, dst_ids)) # create dgl 
        sorted_onetree_node_df = onetree_node_df.sort_values(by='node')
        node_feat_cols = feat_dict['norm_onehot_feats']
        
        # assign features and labels for background nodes
        node_feat = sorted_onetree_node_df[node_feat_cols].values
        node_label = sorted_onetree_node_df[label_feat_col].values
        num_nodes = node_feat.shape[0]
        num_feat = node_feat.shape[1]
        bg_nodes = np.where(sorted_onetree_node_df['cluster_id'].values=='Background')
        for i in bg_nodes[0]:
            node_feat[i] = np.zeros([1, num_feat]) # assign features with 0
            node_label[i] = np.array([3]) # set label to 3(unknown)  
        
        # assign features for nodes and edges, assign labels
        g.ndata['feat'] = torch.tensor(node_feat)
        g.ndata['label'] = torch.tensor(node_label)
        g.edata['feat'] = torch.tensor(onetree_edge_df[aly_edge_feat_cols].values) # wait for reading weight norm-asinh
        
        self._num_labels = 4
        self._labels = node_label
        self._g = g
        
        return self._g
    def __len__(self):
        return len(self.node_df['sim'].unique()) # number of trees
   
# create batch of trees(aggregate multiples trees to a single tree)
def _collate_fn(batch):
    graphs = batch
    g = dgl.batch(graphs)
    return g

batch_size = 10

train_set = MyDataset(train_node_df,train_edge_df)

test_set = MyDataset(test_node_df,test_edge_df)


train_loader = DataLoader(train_set, batch_size, shuffle = True, collate_fn = _collate_fn)
i = 0
for batched_graph in enumerate(train_loader):
    i += 1
    
print(f"Number of trees in train set: {len(train_node_df['sim'].unique())}")
print(f'Number of batches: {i}')  
print(f'Batch size: {batch_size}')
print(batched_graph[1]) # print the last batch
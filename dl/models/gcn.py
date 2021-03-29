#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
gcn.py: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from dl import feat_dict


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        in_feats = len(feat_dict[args.node_feat_cols])
        num_classes = len(feat_dict[args.node_label_cols.split("_cat")[0]])

        h_feat = 256
        self.conv1 = GraphConv(in_feats, h_feat)
        self.conv2 = GraphConv(h_feat, num_classes)

        # h_feats = [256, 256]
        # self.conv1 = GraphConv(in_feats, h_feats[0])
        # self.conv2 = GraphConv(h_feats[0], h_feats[1])
        # self.conv3 = GraphConv(h_feats[1], num_classes)

    def forward(self, g):
        info = dict()
        node_feat = g.ndata["feat"]
        edge_feat = g.edata["feat"]

        h = self.conv1(g, node_feat)
        h = F.relu(h)
        h = self.conv2(g, h)

        # h = self.conv1(g, node_feat)  # , edge_weight=edge_feat
        # h = F.relu(h)
        # h = self.conv2(g, h)
        # h = F.relu(h)
        # h = self.conv3(g, h)    # , edge_weight=edge_feat
        return h, info

    def ce_loss(self, y_pred, y_true, weight=None):
        # print(y_pred.shape, y_true.shape, weight.shape)
        ce = F.cross_entropy(y_pred, y_true, weight=weight, size_average=None, reduce=None, reduction='mean')
        return {"loss": ce}
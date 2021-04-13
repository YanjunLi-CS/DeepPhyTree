#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
gsage.py: 
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dl import feat_dict


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        in_feats = len(feat_dict[args.node_feat_cols])
        num_classes = len(feat_dict[args.node_label_cols.split("_cat")[0]])
        h_feats = 256

        self.conv1 = SAGEConv(in_feats, h_feats, 2, in_feats)
        self.conv2 = SAGEConv(h_feats, num_classes, in_feats, h_feats)
        # self.conv3 = SAGEConv(h_feats, num_classes, h_feats, h_feats)

    def forward(self, g):
        info = dict()
        node_feat = g.ndata["feat"]
        edge_feat = g.edata["feat"]

        h, e_h = self.conv1(g, node_feat, edge_feat)
        h, e_h = F.relu(h), F.relu(e_h)
        h, e_h = self.conv2(g, h, e_h)
        # h, e_h = F.relu(h), F.relu(e_h)
        # h, e_h = self.conv3(g, h, e_h)
        return h, info

    def ce_loss(self, y_pred, y_true, weight=None):
        # print(y_pred.shape, y_true.shape, weight.shape)
        ce = F.cross_entropy(y_pred, y_true, weight=weight, size_average=None, reduce=None, reduction='mean')
        return {"loss": ce}


class SAGEConv(nn.Module):
    """Graph convolution module used by the GraphSAGE model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat, e_in_feat, e_out_feat):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear1 = nn.Linear(in_feat * 2, out_feat)
        self.e_linear = nn.Linear(e_in_feat, e_out_feat)
        self.reduce_linear1 = nn.Linear(in_feat, in_feat)
        self.reduce_linear2 = nn.Linear(in_feat, in_feat)

    def msg_func(self, edges):
        cos_sim = self.cos_sim(torch.unsqueeze(edges.src['h'], dim=1),
                               torch.unsqueeze(edges.dst['h'], dim=1))
        cos_sim = torch.squeeze(cos_sim, 2)
        # print(cos_sim.size(), edges.data['w'].size(), edges.src["h"].size())
        return {"m": cos_sim * edges.data['w'] * edges.src["h"]}

    def cos_sim(self, a, b):
        a_norm = F.normalize(a, p=2, dim=2)
        b_norm = F.normalize(b, p=2, dim=2)
        return torch.matmul(a_norm, b_norm.transpose(1, 2))

    def reduce_func(self, nodes):
        r = self.reduce_linear1(nodes.mailbox["m"])
        r = F.relu(r)
        r = self.reduce_linear2(r)
        r = F.relu(r)
        r = torch.mean(r, dim=1)
        return {"h_N": r}

    def forward(self, g, h, e):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        e : Tensor
            The input edge feature.
        """
        with g.local_scope():
            g.ndata['h'] = h
            edge_feat = self.e_linear(e)
            g.edata["w"] = edge_feat

            # update_all is a message passing API.
            # g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
            # g.update_all(message_func=fn.u_mul_e('h', 'w', 'm'), reduce_func=fn.mean('m', 'h_N'))
            # g.update_all(message_func=self.msg_func, reduce_func=fn.mean('m', 'h_N'))
            g.update_all(message_func=self.msg_func, reduce_func=self.reduce_func)

            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            h = self.linear1(h_total)
            return h, edge_feat
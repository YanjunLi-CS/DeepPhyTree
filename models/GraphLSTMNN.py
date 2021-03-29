# -*- coding: utf-8 -*-
"""

@author: Suncy

Graph LSTM model
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
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


######################################## Network Model ############################################

# message passing user-defined functions
# could define different message passing strategy 

def edge_udf(edges):
    # cat states of edge and source node
    cat_feat = torch.cat((edges.src['hidden_state'],edges.data['hidden_state']),1)
    return {'cat': cat_feat}
  

def node_udf(edges):
    # send edge state to dst node
    return {'hidden_state': edges.data['hidden_state']}


def reducer(nodes):
    # cat states of node and in-bound edge
    cat_feat = torch.cat((nodes.mailbox['hidden_state'][:,0,:],nodes.data['hidden_state']),1)
    return {'cat': cat_feat} 


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        self.hidden_size = 128
        self.num_iter = 2
        # node LSTM & edge LSTM
        # 21 dim node feature & 2 dim edge feature
        self.Node_LSTM = nn.LSTM(input_size=21, hidden_size=self.hidden_size, num_layers=1, batch_first = True)
        self.Edge_LSTM = nn.LSTM(input_size=2, hidden_size=self.hidden_size, num_layers=1, batch_first = True)
        self.m = nn.Sigmoid()
        # message passing network
        self.node_mpn = nn.Linear(2*self.hidden_size, 21)
        self.edge_mpn = nn.Linear(2*self.hidden_size, 2)
        # linear classifier
        self.fc = nn.Linear(self.hidden_size, 4)
    
    def forward(self, graph):
        # num of nodes & edges in batched graph
        num_nodes = graph.ndata['feat'].shape[0]
        num_edges = graph.edata['feat'].shape[0]
        
        # initialization of hidden state and cell state
        graph.ndata['hidden_state'] = Variable(torch.zeros(num_nodes, self.hidden_size))
        graph.ndata['cell_state'] = Variable(torch.zeros(num_nodes, self.hidden_size))
        graph.edata['hidden_state'] = Variable(torch.zeros(num_edges, self.hidden_size))
        graph.edata['cell_state'] = Variable(torch.zeros(num_edges, self.hidden_size))
        
        # convert the data shape for LSTM 
        h_n = graph.ndata['hidden_state'][np.newaxis,:,:]
        c_n = graph.ndata['cell_state'][np.newaxis,:,:]
        h_e = graph.edata['hidden_state'][np.newaxis,:,:]
        c_e = graph.edata['cell_state'][np.newaxis,:,:]

        for i in range(self.num_iter):
            if i == 0: # first iteration, input is feature vec
                node_o, (node_h,node_c) = self.Node_LSTM(graph.ndata['feat'], (h_n, c_n))
                edge_o, (edge_h,edge_c) = self.Edge_LSTM(graph.edata['feat'], (h_e, c_e))
            else: # later iteration, input is message
                node_o, (node_h,node_c) = self.Node_LSTM(graph.ndata['msg'], (h_n, c_n))
                edge_o, (edge_h,edge_c) = self.Edge_LSTM(graph.edata['msg'], (h_e, c_e))
            
            # convert hidden state shape for dgl graph
            graph.ndata['hidden_state'] = node_h[0,:,:]
            graph.ndata['cell_state'] = node_c[0,:,:]
            graph.edata['hidden_state'] = edge_h[0,:,:]
            graph.edata['cell_state'] = edge_c[0,:,:]
            
            # message passing
            graph.apply_edges(edge_udf) # update the feature vector of edges
            graph.edata['msg'] = self.m(self.edge_mpn(graph.edata['cat'])) # generate edge message
            graph.edata['msg'] = graph.edata['msg'][:,np.newaxis,:] # convert message shape for LSTM
            graph.update_all(node_udf,reducer) # send edge state to dst nodes
            graph.ndata['msg'] = self.m(self.node_mpn(graph.ndata['cat'])) # generate node message
            graph.ndata['msg'] = graph.ndata['msg'][:,np.newaxis,:] # convert message shape for LSTM
           
        # linear classifier
        output = self.fc(node_o[:,0,:])
        return F.log_softmax(output)


def train():
    batch_size = 5
    
    # dataset
    train_set = MyDataset(train_node_df,train_edge_df)
    test_set = MyDataset(test_node_df,test_edge_df)
    
    # dataloader
    train_loader = DataLoader(train_set, batch_size, shuffle = True, collate_fn = _collate_fn)
    test_loader = DataLoader(test_set, batch_size, shuffle = True, collate_fn = _collate_fn)
    
    # network setting
    Net = GraphLSTM()
    optimizer = optim.SGD(Net.parameters(), lr=0.01, momentum=0.1, weight_decay = 0.001)

    #train
    for epoch in range(1, 20):
        correct = 0
        for batch_idx, batched_graph in enumerate(train_loader):
            graph = batched_graph
            label = graph.ndata['label']
            
            # convert data shape for LSTM
            graph.ndata['feat'] = Variable(graph.ndata['feat'][:,np.newaxis,:].float())
            graph.edata['feat'] = Variable(graph.edata['feat'][:,np.newaxis,:].float())
            label = Variable(label.long())
            optimizer.zero_grad()
            output = Net(graph)
            loss = F.cross_entropy(output,label)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                pred = output.data.max(1, keepdim=True)[1]
                correct = pred.eq(label.data.view_as(pred)).cpu().sum()
                print('Train Epoch: {} Loss: {:.6f}\tAccuracy: ({:.0f}%)'.format(
                    epoch, loss, 100. * correct / graph.number_of_nodes()))
                correct = 0   
            
    #test
    test_loss = 0
    correct = 0
    num_nodes = 0
    num_bg = 0
    for batched_graph in test_loader:
        graph = batched_graph
        label = graph.ndata['label']
        num_bg += len(np.where(graph.ndata['label']==3)[0])
        graph.ndata['feat'] = Variable(graph.ndata['feat'][:,np.newaxis,:].float())
        graph.edata['feat'] = Variable(graph.edata['feat'][:,np.newaxis,:].float())
        label = Variable(label.long())
        output = Net(graph)
        #get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        num_nodes += graph.number_of_nodes()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, num_nodes,
        100. * correct / num_nodes))
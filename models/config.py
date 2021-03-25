#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
config.py: 
"""
import argparse
import logging


def get_arguments():
    description = "Implementation of DeepPhyTree"
    parser = argparse.ArgumentParser(description=description)

    # General options
    general = parser.add_argument_group("General options")
    general = parser.add_argument("-s", "--seed", type=float, default=123)

    # Data Options
    data = parser.add_argument_group("Data specific options")
    data.add_argument("--ds_name", type=str, default="02172021", help="The name of dataset")
    data.add_argument("--ds_dir", type=str, default="../data/", help="The base folder for data")
    data.add_argument("--ds_split", type=str, default="split_rs123")
    data.add_argument("--num_workers", type=int, default=4, help='The number of workers used for loading data.')
    data.add_argument("--batch_size", type=int, default=128, help="batch size for the graph training")
    data.add_argument("--node_feat_cols", type=str, default="norm_onehot_feats")
    data.add_argument("--node_label_cols", type=str, default="dynamic_cat")
    data.add_argument("--edge_feat_cols", type=str, default="norm_edge_feats_arsinh")
    data.add_argument("--pro_bg_nodes", type=str, default='all_zero')

    args = parser.parse_args()
    return args


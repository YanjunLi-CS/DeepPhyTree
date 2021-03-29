#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
info.py: 
"""

SAVE_PATH = "/data/yanjun/DeepPhyTree/checkpoints/02172021/split_rs123"
BASE_PATH = "/mnt/data2/yanjun/deepclustering"

METRICS = ["loss"]
MIN_METRICS = ["loss"]
MAX_METRICS = []
SUMMARY_ITEMS = ["lr", "acc", "balance_acc", "f1_weighted", "macro_auc_ovo", "weighted_auc_ovo",
                 "macro_auc_ovr", "weighted_auc_ovr"]

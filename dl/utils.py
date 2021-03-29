#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
utils.py: 
"""

import os
import os.path as osp
import numpy as np
import random
from shutil import rmtree
import pandas as pd
import torch.optim as optim
import re
import json
from contextlib import contextmanager
from dl import logger
import time
import torch
from collections import OrderedDict
from warmup_scheduler import GradualWarmupScheduler
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from numpy import interp
from itertools import cycle
from sklearn.metrics import auc, roc_auc_score, roc_curve, accuracy_score, f1_score, balanced_accuracy_score
from dl import feat_dict, args


def set_seed(seed):
    """Set all random seeds."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)


def rm_dump_folder(dump_folder):
    if osp.exists(dump_folder):
        rmtree(dump_folder)


def write_config(dest_dir, f_name, *args):
    if not os.path.exists(dest_dir):
        os.makedirs(os.path.dirname(dest_dir))

    item_list = os.listdir(dest_dir)
    num_config_files = 0
    for x in item_list:
        if re.search(r'{}_config_\d\.json'.format(f_name), x):
            num_config_files += 1

    new_config_name = os.path.join(dest_dir, '{}_config_{}.json'.format(f_name, num_config_files))
    print(num_config_files, new_config_name)
    data_dict_list = []
    for data_dict in args:
        print_dict_byline(data_dict)
        order_dict = OrderedDict(sorted(data_dict.items(), key=lambda t: t[0]))
        data_dict_list.append(order_dict)
    with open(new_config_name, 'w') as fp:
        json.dump(data_dict_list, fp, indent=2)


def print_dict_byline(target_dict):
    for k, v in target_dict.items():
        print(str(k + ':').ljust(15) + str(v))
    print('===============================')


def load_config(dest_dir, f_name, cur_arg, *args):
    if not os.path.exists(dest_dir):
        raise Exception('The config folder does not exist. {}'.format(dest_dir))

    config_f = osp.join(dest_dir, '{}.json'.format(f_name))
    with open(config_f, 'r') as f:
        prev_config_dict = json.load(f)[0]

    changeable_parameters = args if len(args) > 0 else ['mode', 'restore', 'restore_metric', 'log_level']
    cur_config_dict = vars(cur_arg)
    changeable_config_dict = {item: cur_config_dict[item] for item in changeable_parameters}
    cur_arg.__dict__.update(prev_config_dict)
    cur_arg.__dict__.update(changeable_config_dict)
    return cur_arg


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dir(root_dir, x):
    target_dir = os.path.join(root_dir, x)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir


def create_weight_dir(root_weight_dir, metrics_lst):
    weight_dir_dict = {}
    for metric in metrics_lst:
        weight_dir_dict[metric] = create_dir(root_weight_dir, metric + '/')
    return weight_dir_dict


def pytorch_optimizer(model, optimizer, init_lr, weight_decay=0):
    if optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif optimizer == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay, amsgrad=True)


def pytorch_lr_scheduler(optimizer, lr_decay_mode, lr_decay_step_or_patience, lr_decay_rate):
    if lr_decay_mode == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_or_patience, gamma=lr_decay_rate)
    elif lr_decay_mode == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                    factor=lr_decay_rate,
                                                    patience=lr_decay_step_or_patience, verbose=True)
    elif lr_decay_mode == 'warmup':
        after_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_or_patience, gamma=lr_decay_rate)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10,
                                                  after_scheduler=after_scheduler)
        return scheduler_warmup
    else:
        assert ('No defined scheduler for {}.'.format(lr_decay_mode))


def create_flag_dict(metrics, min_metrics, max_metrics):
    flag_dict = OrderedDict()
    for metric in metrics:
        if metric in max_metrics:
            flag_dict[metric] = 0
        elif metric in min_metrics:
            flag_dict[metric] = float('inf')
        flag_dict[metric + '_epoch'] = 0
    return flag_dict


class Summary(object):
    def __init__(self, summary_writer, metric_items, summary_items):
        self.summary_writer = summary_writer
        self.metric_items = metric_items
        self.summary_items = summary_items

    def add_summary(self, step, **kwargs):
        scope = kwargs['scope']
        for k, v in kwargs.items():
            if k in self.metric_items or k in self.summary_items:
                self.summary_writer.add_scalar(tag=scope + '_' + k, scalar_value=v, global_step=step)


@contextmanager
def timer(message):
    """Context manager for timing snippets of code."""
    tick = time.time()
    yield
    tock = time.time()

    diff = tock - tick
    if diff >= 3600:
        duration = "{:.2f}h".format(diff / 3600)
    elif diff >= 60:
        duration = "{:.2f}m".format(round(diff / 60))
    else:
        duration = "{:.2f}s".format(diff)
    logger.info("{}: {}".format(message, duration))


def format_metric_dict(metric_dict, decimals=4):
    for k, v in metric_dict.items():
        if isinstance(v, float) and k != 'lr':
            metric_dict[k] = np.round(v, decimals=decimals)
    return metric_dict


def reduce_lr_on_plateau(lr_scheduler, metric_dict, monitor):
    val_loss = metric_dict[monitor]

    # Note that step should be called after validate()
    lr_scheduler.step(val_loss)


def save_model_update_flag(model, optimizer, weight_dict, flag_dict, metric_dict, min_metrics, max_metrics, epoch):
    for k, v in metric_dict.items():
        if k in min_metrics and v <= flag_dict[k]:
            flag_dict[k] = metric_dict[k]
            flag_dict[k + '_epoch'] = epoch
            save_checkpoint(epoch, model, optimizer, os.path.join(weight_dict[k], 'model.pth.tar'))
            logger.info('Saving for {}'.format(k))
        elif k in max_metrics and v >= flag_dict[k]:
            flag_dict[k] = metric_dict[k]
            flag_dict[k + '_epoch'] = epoch
            save_checkpoint(epoch, model, optimizer, os.path.join(weight_dict[k], 'model.pth.tar'))
            # torch.save(model.state_dict(), os.path.join(weight_dict[k], 'weights.pk'))
            logger.info('Saving for {}'.format(k))


def save_checkpoint(epoch, model, optimizer, filename):
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, filename)


def early_stop(flag_dict, epoch, patience, scope):
    flag_epoches = []
    for k, v in flag_dict.items():
        if k.endswith("epoch"):
            flag_epoches.append(v)
    latest_epoch = max(flag_epoches)
    if epoch - latest_epoch + 1 > patience:
        logger.info('==={} reaches early stop with best model==='.format(scope))
        logger.info('{}'.format(flag_dict))
        return True
    else:
        return False


def get_fpr_tpr(n_classes, y_test_onehot, y_test_predscore):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], y_test_predscore[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # Micro-average: Calculate metrics globally by considering each element of the label indicator matrix as a label.
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_onehot.ravel(), y_test_predscore.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


def plot_roc(n_classes, fpr, tpr, roc_auc):
    from dl import feat_dict
    label_encode_dict = {str(k): v for k, v in feat_dict[args.node_label_cols.split("_cat")[0]].items()}
    fig, axs = plt.subplots(2, 3, figsize=(27, 20))
    lw = 2

    for i, class_id in enumerate(range(n_classes)):
        row, col = np.unravel_index(i, (2, 3))
        ax = axs[row][col]
        ax.plot(fpr[class_id], tpr[class_id], color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[class_id])
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=25)
        ax.set_ylabel('True Positive Rate', fontsize=25)
        if i != 0:
            ax.set_ylabel('')

        ax.set_title(f'ROC for {label_encode_dict[str(class_id)]}-{class_id}')
        ax.legend(loc="lower right")
    plt.show()


def cal_auc(y_test_onehot, y_test_predscore, metric_dict):
    try:
        macro_roc_auc_ovo = roc_auc_score(y_test_onehot, y_test_predscore, multi_class="ovo",
                                          average="macro")
    except ValueError:
        macro_roc_auc_ovo = 0
    # 'weighted': Calculate metrics for each label, and find their average,
    # weighted by support (the number of true instances for each label).
    try:
        weighted_roc_auc_ovo = roc_auc_score(y_test_onehot, y_test_predscore, multi_class="ovo",
                                             average="weighted")
    except ValueError:
        weighted_roc_auc_ovo = 0
    try:
        macro_roc_auc_ovr = roc_auc_score(y_test_onehot, y_test_predscore, multi_class="ovr",
                                          average="macro")
    except ValueError:
        macro_roc_auc_ovr = 0
    try:
        weighted_roc_auc_ovr = roc_auc_score(y_test_onehot, y_test_predscore, multi_class="ovr",
                                             average="weighted")
    except ValueError:
        weighted_roc_auc_ovr = 0
    logger.info("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
                "(weighted by prevalence)"
                .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
    logger.info("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
                "(weighted by prevalence)"
                .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

    metric_dict['macro_auc_ovo'] = macro_roc_auc_ovo
    metric_dict['weighted_auc_ovo'] = weighted_roc_auc_ovo
    metric_dict['macro_auc_ovr'] = macro_roc_auc_ovr
    metric_dict['weighted_auc_ovr'] = weighted_roc_auc_ovr


def plot_merged_roc(n_classes, fpr, tpr, roc_auc):
    # First aggregate all false positive rates
    lw = 2
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    # Calculate metrics for each label, and find their unweighted mean (by linear interpolation).
    # This does not take label imbalance into account.
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 10))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def cal_basic_metric(y_test, y_test_pred, metric_dict):
    acc = accuracy_score(y_test, y_test_pred)
    logger.debug(f'Test Acc: {acc}')

    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    logger.debug(f'Test Ballance Acc: {bal_acc}')

    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    logger.debug(f'Test F1: {f1_weighted}')

    metric_dict['acc'] = acc
    metric_dict['balance_acc'] = bal_acc
    metric_dict['f1_weighted'] = f1_weighted


def cal_pred_metrics(pred_dict):
    y_true, y_predscore = pred_dict["y_true"], pred_dict["y_predscore"]
    classes = [int(i) for i in feat_dict[args.node_label_cols.split("_cat")[0]].keys()]

    # Note: Remove the background
    non_bg_idx = np.where(y_true != classes[-1])
    y_true = y_true[non_bg_idx]
    y_predscore = y_predscore[non_bg_idx][:, :-1]
    classes.pop()

    y_pred = np.argmax(y_predscore, axis=1)
    n_classes = len(classes)
    y_test_onehot = label_binarize(y_true, classes=classes)

    metric_dict = dict()
    cal_basic_metric(y_true, y_pred, metric_dict)
    cal_auc(y_test_onehot, y_predscore, metric_dict)

    # fpr, tpr, roc_auc = get_fpr_tpr(n_classes, y_test_onehot, y_predscore)
    # plot_roc(n_classes, fpr, tpr, roc_auc)
    return metric_dict

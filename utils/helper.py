#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from utils.metrics import ConfusionMatrix
from PIL import Image
import os

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def collate(batch):
    image = [b['image'] for b in batch]  # w, h
    label = [b['label'] for b in batch]
    id = [b['id'] for b in batch]
    adj_s = [b['adj_s'] for b in batch]
    return {'image': image, 'label': label, 'id': id, 'adj_s': adj_s}


def preparefeatureLabel(batch_graph, batch_label, batch_adjs, num_feat):
    batch_size = len(batch_graph)
    labels = torch.LongTensor(batch_size)
    max_node_num = 0

    for i in range(batch_size):
        labels[i] = batch_label[i]
        max_node_num = max(max_node_num, batch_graph[i].shape[0])

    masks = torch.zeros(batch_size, max_node_num)
    adjs = torch.zeros(batch_size, max_node_num, max_node_num)
    batch_node_feat = torch.zeros(batch_size, max_node_num, num_feat)

    for i in range(batch_size):
        cur_node_num = batch_graph[i].shape[0]
        # node attribute feature
        tmp_node_fea = batch_graph[i]
        batch_node_feat[i, 0:cur_node_num] = tmp_node_fea

        # adjs
        adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs[i]

        # masks
        masks[i, 0:cur_node_num] = 1

    node_feat = batch_node_feat.cuda()
    labels = labels.cuda()
    adjs = adjs.cuda()
    masks = masks.cuda()

    return node_feat, labels, adjs, masks


class Trainer(object):
    def __init__(self, n_class, num_feats):
        self.metrics = ConfusionMatrix(n_class)
        self.num_feats = num_feats

    def get_scores(self):
        acc = self.metrics.get_scores()

        return acc

    def reset_metrics(self):
        self.metrics.reset()

    def plot_cm(self):
        self.metrics.plotcm()

    def train(self, sample, model):
        node_feat, labels, adjs, masks = preparefeatureLabel(sample['image'], sample['label'], sample['adj_s'],
                                                             self.num_feats)
        pidList = sample['id']
        pred, labels, loss, probs = model.forward(node_feat, labels, adjs, masks, pidList)

        return pred, labels, loss, probs


class Evaluator(object):
    def __init__(self, n_class, num_feats):
        self.metrics = ConfusionMatrix(n_class)
        self.num_feats = num_feats

    def get_scores(self):
        acc = self.metrics.get_scores()

        return acc

    def reset_metrics(self):
        self.metrics.reset()

    def plot_cm(self):
        return self.metrics.plotcm()

    def eval_test(self, sample, model, graphcam_flag=False):
        node_feat, labels, adjs, masks = preparefeatureLabel(sample['image'], sample['label'], sample['adj_s'],
                                                             self.num_feats)
        pidList = sample['id']
        if not graphcam_flag:
            with torch.no_grad():
                pred, labels, loss, probs = model.forward(node_feat, labels, adjs, masks, pidList)
        else:
            torch.set_grad_enabled(True)
            pred, labels, loss, probs = model.forward(node_feat, labels, adjs, masks, pidList, graphcam_flag=graphcam_flag)
        return pred, labels, loss, probs


# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:30:07 2021

@author: Narmin Ghaffari Laleh
"""

##############################################################################
import pandas as pd
import utils.utils as utils

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from tqdm import tqdm
from sklearn import metrics
import torch.nn as nn
import numpy as np
import time
import torch
import os
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import shutil


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##############################################################################
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler and (not self.finished):
                self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                self.finished = True

            return [base_lr for base_lr in self.after_scheduler.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        print('warmuping...')
        if self.last_epoch <= self.total_epoch:
            warmup_lr = None
            if self.multiplier == 1.0:
                warmup_lr = [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
            else:
                warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                             self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=50, verbose=False):

        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

##############################################################################

class Accuracy_Logger(object):
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

    ##############################################################################
def Summary(model, loader, n_classes, modelName=None):
    model.eval()
    test_error = 0.
    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    slide_ids = loader.dataset.slide_data['FILENAME']
    case_ids = loader.dataset.slide_data['PATIENT']
    patient_results = {}

    for batch_idx, (data, label, _) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        fileName = slide_ids.iloc[batch_idx]
        patient = case_ids.iloc[batch_idx]

        with torch.no_grad():
            _, Y_prob, Y_hat, _, _ = model(data)

        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update(
            {fileName: {'PATIENT': patient, 'FILENAME': fileName, 'prob': probs, 'label': label.item()}})
        error = utils.calculate_error(Y_hat, label)
        test_error += error
    test_error /= len(loader)
    if n_classes == 2:
        for i in range(n_classes):
            fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_probs[:, i], pos_label=i)
            auc = metrics.auc(fpr, tpr)
            aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        auc = np.nanmean(np.array(aucs))
    return patient_results, test_error, auc
##############################################################################    

def Train_model_AttMIL(model, trainLoaders, args, valLoaders=[], criterion=None, optimizer=None, fold=False):
    since = time.time()
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []
    early_stopping = EarlyStopping(patience=args.patience, stop_epoch=args.minEpochToTrain, verbose=True)

    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    my_scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=scheduler)
    for epoch in range(args.max_epochs):
        phase = 'train'
        model.train()
        running_loss = 0.0
        running_corrects = 0
        current_lr = optimizer.param_groups[0]['lr']
        print('Epoch : {}/{}, lr: {}\n'.format(epoch, args.max_epochs - 1, current_lr))
        print('\nTRAINING...\n')
        for inputs, labels in tqdm(trainLoaders):
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                _, y_hat = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs[0].size(0)
                running_corrects += torch.sum(y_hat == labels.data)

        my_scheduler.step()
        epoch_loss = running_loss / len(trainLoaders.dataset)
        epoch_acc = running_corrects.double() / len(trainLoaders.dataset)
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)
        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print()
        if valLoaders:
            print('VALIDATION...\n')
            phase = 'val'
            model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(valLoaders):
                labels = labels.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, labels)
                    _, y_hat = torch.max(outputs, 1)
                running_loss += loss.item() * inputs[0].size(0)
                running_corrects += torch.sum(y_hat == labels.data)
            val_loss = running_loss / len(valLoaders.dataset)
            val_acc = running_corrects.double() / len(valLoaders.dataset)
            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)
            print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss, val_acc))
            if fold == 'FULL':
                ckpt_name = os.path.join(args.result_dir, "bestModel")
            else:
                ckpt_name = os.path.join(args.result_dir, "bestModelFold" + fold)
            early_stopping(epoch, val_loss, model, ckpt_name=ckpt_name)
            if early_stopping.early_stop:
                print('-' * 30)
                print("The Validation Loss Didn't Decrease, Early Stopping!!")
                print('-' * 30)
                break
            print('-' * 30)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, train_loss_history, train_acc_history, val_acc_history, val_loss_history

def Validate_model_AttMIL(model, dataloaders):
    phase = 'test'
    model.eval()
    probsList = []
    index = 1
    for inputs in tqdm(dataloaders):
        print(index, " ===", len(dataloaders))
        with torch.set_grad_enabled(phase == 'train'):
            probs, attentions = model(inputs[0])

            probs = nn.Softmax(dim=1)(probs)
            probs = probs[:, 0]

            probsList = probsList + probs.tolist()
        index = index + 1
    return probsList



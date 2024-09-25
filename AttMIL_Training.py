# -*- coding: utf-8 -*-

##############################################################################

import utils.utils as utils
from extractFeatures import ExtractFeatures
from utils.data_utils import ConcatCohorts_Classic
import numpy as np
import os
import pandas as pd
import random
from sklearn import preprocessing
import torch
from pathlib import Path
from fastai.vision.all import *
from models.model_Attmil import MILModel, MILBagTransform
from utils.core_utils import Train_model_AttMIL, Validate_model_AttMIL

from sklearn.metrics import accuracy_score, roc_curve, f1_score, roc_auc_score, auc
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################

def AttMIL_Training(args):
    targetLabels = args.target_labels
    args.feat_dir = args.feat_dir[0]

    for targetLabel in targetLabels:
        for repeat in range(args.repeatExperiment):

            args.target_label = targetLabel
            random.seed(args.seed)
            args.projectFolder = utils.CreateProjectFolder(args.project_name, args.adressExp, targetLabel, args.model_name, "dPathHRD")
            print(args.projectFolder)
            os.makedirs(args.projectFolder, exist_ok = True)

            args.result_dir = os.path.join(args.projectFolder, 'RESULTS')
            os.makedirs(args.result_dir, exist_ok = True)
            args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
            os.makedirs(args.split_dir, exist_ok = True)

            reportFile  = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
            reportFile.write('-' * 30 + '\n')
            reportFile.write(str(args))
            reportFile.write('-' * 30 + '\n')
            if args.extractFeature:
                imgs = os.listdir(args.datadir_train[0])
                imgs = [os.path.join(args.datadir_train[0], i) for i in imgs]
                ExtractFeatures(data_dir = imgs, feat_dir = args.feat_dir, batch_size = args.batch_size, target_patch_size = -1, filterData = True)

            args.csvFile = ConcatCohorts_Classic(imagesPath = args.datadir_train,
                                                                          cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
                                                                          label = targetLabel, outputPath = args.projectFolder, reportFile = reportFile,
                                                                        csvName = args.csv_name, patientNumber = args.numPatientToUse)

            dataset = pd.read_csv(args.csvFile)
            patientsList = list(dataset["PATIENT"])
            labelsList = list(dataset[args.target_label])

            yTrueLabel = utils.CheckForTargetType(labelsList)
            le = preprocessing.LabelEncoder()
            yTrue = le.fit_transform(yTrueLabel)
            args.num_classes = len(set(yTrue))
            args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))
            utils.Summarize(args, list(yTrue), reportFile)
            if len(patientsList) < 5:
                continue
            print('-' * 30)

            train_data = pd.read_csv(args.csvFile)
            val_data = train_data.groupby(args.target_label, group_keys = False).apply(lambda x: x.sample(frac=0.3, random_state=args.seed))
            train_data['is_valid'] = train_data.PATIENT.isin(val_data['PATIENT'])
            train_data['FILEPATH'] = [i.replace('WSI_Patches', 'FEATURES') for i in train_data['FILEPATH']]
            train_data['FILEPATH'] = [Path(i + '.pt') for i in train_data['FILEPATH']]
            train_data.to_csv(os.path.join(args.split_dir, 'TrainValSplit.csv'), index = False)

            dblock = DataBlock(blocks = (TransformBlock, CategoryBlock),
                               get_x = ColReader('FILEPATH'),
                               get_y = ColReader(args.target_label),
                               splitter = ColSplitter('is_valid'),
                               item_tfms = MILBagTransform(train_data[train_data.is_valid].FILEPATH, 4096))    
            dls = dblock.dataloaders(train_data, bs = args.batch_size, drop_last=False)
            weight = train_data[args.target_label].value_counts().sum() / train_data[args.target_label].value_counts()
            weight /= weight.sum()
            weight = torch.tensor(list(map(weight.get, dls.vocab)))
            criterion = CrossEntropyLossFlat(weight = weight.to(torch.float32))
            model = MILModel(768, args.num_classes, with_attention_scores=True)
            model = model.to(device)
            criterion.to(device)
            optimizer = utils.get_optim(model, args, params = False)

            model, train_loss_history, train_acc_history, val_acc_history, val_loss_history = Train_model_AttMIL(model = model, trainLoaders = dls.train,
                                             valLoaders = dls.valid, criterion = criterion, optimizer = optimizer, args = args, fold = 'FULL')
            torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'finalModel'))
            history = pd.DataFrame(list(zip(train_loss_history, train_acc_history, val_acc_history, val_loss_history)),
                              columns =['train_loss', 'train_acc', 'val_loss', 'val_acc'])
            history.to_csv(os.path.join(args.result_dir, 'TRAIN_HISTORY_FULL' + '.csv'), index = False)
            print('-' * 30)



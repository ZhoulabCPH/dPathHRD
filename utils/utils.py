# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:23:50 2021

@author: Narmin Ghaffari Laleh
"""

##############################################################################

import os 
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torchvision import models
import json
import warnings
import pathlib
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################

def CreateProjectFolder(ExName, ExAdr, targetLabel, model_name, repeat = None):
    if repeat:
        outputPath = Path(ExAdr.parent, ExName + '_' + targetLabel + '_' + str(repeat))
    else:
        outputPath = Path(ExAdr.parent, ExName + '_' + targetLabel)
    return outputPath
   
        
##############################################################################
       
def Print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)

##############################################################################
        
def get_optim(model, args, params = False):
   
    if params:
        temp = model
    else:
        temp = filter(lambda p: p.requires_grad, model.parameters())
        
    if args.opt == "adam":
        optimizer = optim.Adam(temp, lr = args.lr, weight_decay = args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(temp, lr = args.lr, momentum = 0.9, weight_decay = args.reg)
    else:
        raise NotImplementedError
        
    return optimizer


##############################################################################
            
def Collate_features(batch):
    
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return  [img, coords]

##############################################################################
            
def calculate_error(Y_hat, Y):
    
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

##############################################################################

def save_pkl(filename, save_object):
    
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

##############################################################################

def load_pkl(filename):
    
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


###############################################################################

def Set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

###############################################################################
            
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

###############################################################################    

def Summarize(args, labels, reportFile):
    
    print("label column: {}\n".format(args.target_label))
    reportFile.write("label column: {}".format(args.target_label) + '\n')    
    print("label dictionary: {}\n".format(args.target_labelDict))
    reportFile.write("label dictionary: {}".format(args.target_labelDict) + '\n')    
    print("number of classes: {}\n".format(args.num_classes))
    reportFile.write("number of classes: {}".format(args.num_classes) + '\n')    
    for i in range(args.num_classes):
        print('Patient-LVL; Number of samples registered in class %d: %d\n' % (i, labels.count(i)))
        reportFile.write('Patient-LVL; Number of samples registered in class %d: %d' % (i, labels.count(i)) + '\n')           
    print('-' * 30 + '\n')
    reportFile.write('-' * 30 + '\n')

###############################################################################

def ReadExperimentFile(args, deploy = False):
    with open(args.adressExp) as json_file:        
        data = json.load(json_file)

    args.csv_name = 'ProcessedData'
    filename, file_extension = os.path.splitext(args.adressExp)
    args.project_name = args.adressExp.stem

    args.clini_dir = []
    args.slide_dir = []
    args.datadir_train = []
    args.feat_dir = []
    
    if not deploy :
        try:
            datadir_train = data['dataDir_train']
        except:
            datadir_train = ''

        for index, item in enumerate(datadir_train):
            item = Path(item)
            if Path(item , 'WSI_Patches').exists():
                args.datadir_train.append(Path(item , 'WSI_Patches'))
            else:
                raise NameError('NO BLOCK FOLDER FOR ' + item + ' TRAINNG IS FOUND!')
            
            if Path(item, item.stem + '_CLINI.xlsx'):
                    args.clini_dir.append(Path(item, item.stem + '_CLINI.xlsx'))
            else:
                    raise NameError('NO CLINI DATA FOR ' + item + ' IS FOUND!')

            if Path(item, item.stem + '_SLIDE.csv'):
                    args.slide_dir.append(Path(item, item.stem + '_SLIDE.csv'))
            else:
                    raise NameError('NO SLIDE DATA FOR ' + item + ' IS FOUND!')                      
            args.feat_dir.append(Path(item , 'FEATURES'))
              
    try:
        args.target_labels = data['targetLabels']
    except:
        raise NameError('TARGET LABELS ARE NOT DEFINED!')
    
    try:
        args.max_epochs = data['epochs']
    except:
        print('EPOCH NUMBER IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 5\n') 
        print('-' * 30)
        args.max_epochs = 8        

    try:
        args.numPatientToUse = data['numPatientToUse']
    except:
        print('NUMBER OF PATIENTS TO USE IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : ALL\n') 
        print('-' * 30)
        args.numPatientToUse = 'ALL'
        
    try:
        args.seed = int(data['seed']) 
    except:
        print('SEED IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 1\n')   
        print('-' * 30)
        args.seed = 1
    try:
        args.num_classes = int(data['num_classes'])
    except:
        print('-' * 30)
        args.num_classes = 1

    try:
        args.dataDir_test = data['dataDir_test']
    except:
        args.datadir_test = ''
    try:
        args.result_dir = data['result_dir']
    except:
        args.result_dir = ''
        
    try:
        args.model_name = data['modelName']
    except:
        print('-' * 30)
        args.model_name = 'attmil'

    try:
        args.opt = data['opt']
    except:
        print('OPTIMIZER IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : adam\n') 
        print('-' * 30)
        args.opt = 'adam'
        
    try:
        args.lr = data['lr']
    except:
        print('LEARNING RATE IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 0.0001\n')
        print('-' * 30)
        args.lr = 0.0001

    try:
        args.reg = data['reg']
    except:
        print('DECREASE RATE OF LR IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 0.00001\n')   
        print('-' * 30)
        args.reg = 0.00001             
    try:
        args.batch_size = data['batchSize']
          
    except:
        print('BATCH SIZE IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 16\n')
        print('-' * 30)
        args.batch_size = 16


    try:
         args.repeatExperiment = int(data['repeatExperiment'])  
    except:
        print('REPEAT EXPERIEMNT NUMBER IS NOT DEFINED!\nDEFAULT VALUE WILL BE USED : 1\n')  
        print('-' * 30)
        args.repeatExperiment = 1
        
    try:
        args.early_stopping = MakeBool(data['earlyStop'])
    except:
        print('EARLY STOPIING VALUE IS NOT DEFINED!\nDEFAULT VALUE WILL BE USED : TRUE\n')  
        print('-' * 30)
        args.early_stopping = True  
        
    if args.early_stopping:
        try:
            args.minEpochToTrain = data['minEpochToTrain']
        except:
            print('MIN NUMBER OF EPOCHS TO TRAIN IS NOT DEFINED!\nDEFAULT VALUE WILL BE USED : 10\n')   
            print('-' * 30)
            args.minEpochToTrain = 10 
         
        try:
            args.patience = data['patience']
        except:
            print('PATIENCE VALUE FOR EARLY STOPPING IS NOT DEFINED!\n DEFAULT VALUE WILL BE USED : 20\n')
            print('-' * 30)
            args.patience = 20


    try:
         args.gpuNo = int(data['gpuNo'])
    except:
        print('GPU ID VALUE IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 0\n')
        print('-' * 30)
        args.gpuNo = 0


            
    if args.model_name in ['attmil']:
        try:
            args.extractFeature = MakeBool(data['extractFeature'])
        except:
            print('EXTRACT FEATURE VALUE IS NOT DEFINED!\nDEFAULT VALUE WILL BE USED : FALSE\n')  
            print('-' * 30)
            args.extractFeature = False

    return args


###############################################################################

def MakeBool(value):
    if value == 'True':
       return True
    else:
        return False
    
###############################################################################

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

###############################################################################

def isint(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

###############################################################################

def CheckForTargetType(labelsList):
    
    if len(set(labelsList)) >= 5:     
        labelList_temp = [str(i) for i in labelsList]
        checkList1 = [s for s in labelList_temp if isfloat(s)]
        checkList2 = [s for s in labelList_temp if isint(s)]
        if not len(checkList1) == 0 or not len (checkList2):
            med = np.median(labelsList)
            labelsList = [1 if i>med else 0 for i in labelsList]
        else:
            raise NameError('IT IS NOT POSSIBLE TO BINARIZE THE NOT NUMERIC TARGET LIST!')
    return labelsList
                    
###############################################################################            
    
def get_key_from_value(d, val):
    
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None   
    
 ###############################################################################   
    
def get_value_from_key(d, key):
    
    values = [v for k, v in d.items() if k == key]
    if values:
        return values[0]
    return None    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

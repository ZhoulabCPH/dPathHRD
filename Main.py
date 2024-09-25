# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:08:57 2021

@author: Narmin Ghaffari Laleh
"""

###############################################################################


from AttMIL_Training import AttMIL_Training
from AttMIL_Validation import AttMIL_Validation

import utils.utils as utils

from pathlib import Path
import warnings
import argparse
import torch

# %%
parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
pathToExperimentFile = Path("configuration.txt")
parser.add_argument('--adressExp', type = str, default = pathToExperimentFile, help = 'Adress to the experiment File')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
#%%

if __name__ == '__main__':

    args = utils.ReadExperimentFile(args)
    AttMIL_Training(args)
    # AttMIL_Validation(args)

        
        

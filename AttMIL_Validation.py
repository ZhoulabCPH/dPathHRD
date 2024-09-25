# -*- coding: utf-8 -*-

##############################################################################

from fastai.vision.all import *
from models.model_Attmil import MILModel, MILBagTransform
from utils.core_utils import Validate_model_AttMIL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################

def AttMIL_Validation(args):
    dataTemplate = pd.read_csv("data_template.csv", sep=",")
    dataTemplate['FILEPATH'] = [Path(i) for i in dataTemplate['FILEPATH']]

    test_data = pd.read_csv(args.dataDir_test, sep=",")
    test_data['FILEPATH'] = [Path(i) for i in test_data['FILEPATH']]  # Ensure FILEPATH is Path object

    dblock = DataBlock(
        blocks=(TransformBlock,),
        get_x=ColReader('FILEPATH'),
        splitter=FuncSplitter(lambda x: True),
        item_tfms=MILBagTransform(dataTemplate.FILEPATH, 4096)
    )
    dls = dblock.dataloaders(test_data, bs=args.batch_size, drop_last=False)
    test_dl = dls.test_dl(test_data)

    model = MILModel(768, args.num_classes, with_attention_scores=True)
    model = model.to(device)

    print("========================Starting val=================================")
    bestModelPath = os.path.join('./', 'finalModel')
    model.load_state_dict(torch.load(bestModelPath)) 
    model = model.to(device)

    val_probsList = Validate_model_AttMIL(model=model, dataloaders=test_dl)

    val_probs = pd.DataFrame({"predProb": val_probsList})
    testResults = pd.concat([test_data, val_probs], axis=1)
    testResultsPath = os.path.join(args.result_dir, 'result.csv')
    testResults.to_csv(testResultsPath, index=False)

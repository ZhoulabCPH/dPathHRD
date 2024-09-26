# dPathHRD
Code for 'Deep learning-based digital pathological prediction of homologous recombination deficiency and responsive outcomes to platinum chemotherapy in high-grade serous ovarian cancer'

![flowchart](https://github.com/ZhoulabCPH/dPathHRD/blob/master/Graphicalabstract.png)

****
## Dataset 
- [TCGA](https://portal.gdc.cancer.gov/projects/TCGA-OV), we incorporate TCGA-OV cohort into our study, and its open access to all.
- The data from the HMUCH and CHCAMS cohorts are available from the corresponding author upon reasonable request.

## image_preprocessing
- <code>extractTiles-ws.py</code>: Used to segment and filter patches from WSIs. Implemented based on the <code>"Aachen protocol for Deep Learning in Histopathology"</code>.

## get_patches_feature
- <code>models/extractors/ctran.py</code>: Implementation of CTransPath.
- <code>extractFeatures.py</code>: Using pre-trained CTransPath to obtain histopathological features of patches.
  
  Part of the implementation here is based on [CTransPath](https://github.com/Xiyue-Wang/TransPath).

## construction_dPathHRD
- <code>utils/data_utils.py</code>: Generate datasets.
- <code>utils/core_utils.py</code>: Tools used in training.
- <code>modelS/model_Attmil</code>: Implementation of the attMIL model.
- <code>AttMIL_Training</code>: Training the dPathHRD.
- <code>AttMIL_Validation</code>: Evaluation of the dPathHRD  in multi-center external cohorts.

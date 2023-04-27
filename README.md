
# Aux-Drop: Handling Haphazard Inputs in Online Learning Using Auxiliary Dropouts

## Overview
This repository contains datasets and implementation code for the paper, titled "Aux-Drop: Handling Haphazard Inputs in Online Learning Using Auxiliary Dropouts".

## Datasets
### german
https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

### svmguide3
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

### magic04 
https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope

### a8a 
https://archive.ics.uci.edu/ml/datasets/adult

### Italy Power Demand Dataset 
https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

### HIGGS 
Original Source: https://archive.ics.uci.edu/ml/datasets/HIGGS . It contains 11M datapoints and 28 features.
We use only the first 1M data points and first 21 features.
The preprocessed datasets can be found in - https://figshare.com/s/0cd0d6ad4d30a9e91e9a. Save this dataset in the Code/Datasets/HIGGS/data/ folder.
The masking for all the experiments with the HIGGS data can be found here - https://figshare.com/s/644fe204eb591e104184. Save all the file from the link to the Code/Datasets/HIGGS/mask/ folder.

### SUSY 
Original Source: https://archive.ics.uci.edu/ml/datasets/SUSY . It contains 5M datapoints and 18 features.
We use only the first 1M data points and first 8 features.
The preprocessed datasets can be found in - https://figshare.com/s/f4098ce6635f702c89b2. Save this dataset in the Code/Datasets/SUSY/data/ folder.
The masking for all the experiments with the HIGGS data can be found here - https://figshare.com/s/87330bbbbc31b15d44e5. Save all the file from the link to the Code/Datasets/SUSY/mask/ folder.

## Dataset Preparation
### Variable P
### Trapezoidal
### Obsolete Sudden

## Variants
1. Aux_Drop_ODL:
2. Aux-Drop_OGD:
3. AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer -  On ODL framework, Aux-Dropout in AuxLayer and Random dropout in all the other layers
4. AuxDrop_ODL_RandomAllLayer - On ODL framework, Random Dropout applied in all the layers
5. AuxDrop_ODL_RandomInAuxLayer - On ODL framework, Random Dropout applied in the AuxLayer
6. AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst - On ODL framework, Random Dropout applied in the first layer and all the features (base + auxiliary) are passed to the first layer

## Baseline
1. Aux-Net : All the metrics are directly taken from the paper (https://link.springer.com/chapter/10.1007/978-3-031-30105-6_46) hence we do not implement here.

## Control Parameters

## Running the code

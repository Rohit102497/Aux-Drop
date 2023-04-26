
# Aux-Drop: Handling Haphazard Inputs in Online Learning Using Auxiliary Dropouts

## Overview
This repository contains datasets and implementation code for the paper, titled "Aux-Drop: Handling Haphazard Inputs in Online Learning Using Auxiliary Dropouts".

## Datasets
1. german - https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
2. svmguide3 - https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
3. magic04 - https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope
4. a8a - https://archive.ics.uci.edu/ml/datasets/adult
5. Italy Power Demand Dataset - https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
6. HIGGS - https://archive.ics.uci.edu/ml/datasets/HIGGS
7. SUSY - https://archive.ics.uci.edu/ml/datasets/SUSY

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

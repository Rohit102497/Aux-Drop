
# Aux-Drop: Handling Haphazard Inputs in Online Learning Using Auxiliary Dropouts
This paper is accepted at Transactions on Machine Learning Research. The link to the paper is: https://openreview.net/pdf?id=R9CgBkeZ6Z. 

Please cite this paper, in case you are using the code or the paper:\
`@article{\
agarwal2023auxdrop,\
title={Aux-Drop: Handling Haphazard Inputs in Online Learning Using Auxiliary Dropouts},\
author={Rohit Agarwal and Deepak Gupta and Alexander Horsch and Dilip K. Prasad},\
journal={Transactions on Machine Learning Research},\
issn={2835-8856},\
year={2023},\
url={https://openreview.net/forum?id=R9CgBkeZ6Z},\
note={Reproducibility Certification}\
}`\


## Overview
This repository contains datasets and implementation code for the paper, titled "Aux-Drop: Handling Haphazard Inputs in Online Learning Using Auxiliary Dropouts".

## Datasets
We use 7 different datasets for this project. The link of all the datasets can be found below. Moreover, the datasets are also given in their respective folders inside `Code/Datasets` directory. HIGGS and SUSY are big data, hence they are not provided inside the directory. But to run them, please download HIGGS data and mask from the link given below and save them in the `Code/Datasets/HIGGS/data/` folder and `Code/Datasets/HIGGS/mask/` folder respectively. Same goes for the SUSY dataset.

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
 - Original Source: https://archive.ics.uci.edu/ml/datasets/HIGGS . It contains 11M datapoints and 28 features.
 - We use only the first 1M data points and first 21 features.
 - Data: The preprocessed datasets can be found in - https://figshare.com/s/0cd0d6ad4d30a9e91e9a. Save this dataset in the `Code/Datasets/HIGGS/data/` folder.
 - Mask: The masking for all the experiments with the HIGGS data can be found here - https://figshare.com/s/644fe204eb591e104184. Save all the file from the link to the `Code/Datasets/HIGGS/mask/` folder.

### SUSY 
 - Original Source: https://archive.ics.uci.edu/ml/datasets/SUSY . It contains 5M datapoints and 18 features.
 - We use only the first 1M data points and first 8 features.
 - Data: The preprocessed datasets can be found in - https://figshare.com/s/f4098ce6635f702c89b2. Save this dataset in the `Code/Datasets/SUSY/data/` folder.
 - Mask: The masking for all the experiments with the HIGGS data can be found here - https://figshare.com/s/87330bbbbc31b15d44e5. Save all the file from the link to the `Code/Datasets/SUSY/mask/` folder.


## Dataset Preparation
### Variable P
We varied the availability of each auxiliary input feature independently by a uniform distribution of probability $p$, i.e., each auxilairy feature is available for $100p\%$. For more information about this, follow paper - Aux-Net (https://link.springer.com/chapter/10.1007/978-3-031-30105-6_46)

### Trapezoidal
The trapezoidal streams are simulated by splitting the data into 10 chunks. The number of features in each successive chunk increases with the data stream. The first chunk has the first 10\%  of the total features, the second chunk has the first 20\%  features, and so on. For more infomation about this, see paper - OLSF (https://ieeexplore.ieee.org/document/7465766).

### Obsolete Sudden
We demonstrate the effectiveness of Aux-Drop(ODL) in processing the extra information received from auxiliary features in both the SUSY and HIGGS datasets. Here, we design the data in a such way that all of them are sudden features, i.e., there is no information about the existence of these features when the model is defined. The model knows about this feature suddenly at time $t$ after the model deployment. For the SUSY dataset, the first auxiliary feature starts arriving from 100k till 500k, the next auxiliary feature ranges from 200k till 600k, and so on to the 6th auxiliary feature coming from 600k to 1000k instances. Each feature becomes obsolete after arriving for 400k instances. Similarly for the HIGGS dataset, the first auxiliary feature arrives from 50k to 250k instances, the second arrives from 100k to 300k, and so on where every successive auxiliary feature arrives at 50k instances after the previous auxiliary features start arriving and arrive till the next 200k instances.

## Comparison Models
We apply the Aux-Drop on two base architectures, ODL and OGD (https://arxiv.org/abs/1711.03705). Moreover, we also see the preformance of the Aux-Drop with few of its variants (changes in the design).

### ODL and OGD
1. Aux_Drop_ODL: Aux-Drop applied on the ODL architecture is called Aux_Drop_ODL in the code.
2. Aux-Drop_OGD: Aux-Drop applied on the OGD architecture is called Aux_Drop_OGD in the code.

### Variants
1. AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer -  On ODL framework, Aux-Dropout is applied in AuxLayer and Random dropout in all the other layers
2. AuxDrop_ODL_RandomAllLayer - On ODL framework, Random Dropout is applied in all the layers
3. AuxDrop_ODL_RandomInAuxLayer - On ODL framework, Random Dropout is applied in the AuxLayer
4. AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst - On ODL framework, Random Dropout is applied in the first layer and all the features (base + auxiliary) are passed to the first layer.


## Baseline
### Aux-Net
All the metrics are directly taken from the paper (https://link.springer.com/chapter/10.1007/978-3-031-30105-6_46) hence we do not implement here.
### OLVF 
All the metrics are directly taken from the paper (https://ojs.aaai.org/index.php/AAAI/article/view/4192) hence we do not implement here. 
### OLSF
All the metrics are directly taken from the paper (https://ieeexplore.ieee.org/document/7465766) hence we do not implement here. 
### ODL
We implement the ODL code and run it on for two scenarios. 
1. Only Base Feature - First, we run it using all the base features. This gives us a lower limit of the performance.
2. All Features - Then we run it using all the features (considering all the featues are avaialable). This gives us an upper limit of the performance.

## Files
To run the models, see
1. main.py: All the comparison models can be run from this.
2. baseline.py: To run the Baseline model (ODL)

The class definition for each comparison model is given in
 - AuxDrop.py

The class definition for ODL baseline is given in
 - ODL.py

The dataloader for each dataset is given in
 - dataset.py

## Control Parameters

For **main.py** file, 
1. `data_name`: "german", "svmguide3", "magic04", "a8a", "ItalyPowerDemand", "SUSY", "HIGGS"
2. `type`: "variable_p", "trapezoidal", "obsolete_sudden"
3. `model_to_run`: "AuxDrop_ODL", "AuxDrop_OGD", "AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer", "AuxDrop_ODL_RandomAllLayer", "AuxDrop_ODL_RandomInAuxLayer", "AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst"
4. `n`: Learning rate
5. `aux_feat_prob`: If `type = "variable_p"`, then `aux_feat_prob` needs to be defined. It is the availability of each auxiliary input feature independently by a uniform distribution of probability `aux_feat_prob`
6. `dropout_p`: The dropout value of AuxLayer
7. `max_num_hidden_layers`: Number of hidden layers
8. `qtd_neuron_per_hidden_layer`: Number of neurons in each hidden layers except the AuxLayer
9. `n_classes`: The number of output classes
10. `aux_layer`: The position of the AuxLayer in the architecture
11. `n_neuron_aux_layer`: Number of neurons in the AuxLayer
12. `b`: This is a parameter of ODL framework. It represents the discount rate
13. `s`: This is a parameter of ODL framework. It represents the smoothing rate

For **baseline.py** file,
1. `data_name`: "SUSY", "HIGGS"
2. `model_to_run`: "ODL"
3. `data_type`: "only_base", "all_feat"
4. `n`: Learning rate
5. `max_num_hidden_layers`: Number of hidden layers
6. `qtd_neuron_per_hidden_layer`: Number of neurons in each hidden layers
7. `n_classes`: The number of output classes
8. `b`: It represents the discount rate
9. `s`: It represents the smoothing rate

## Dependencies
1. numpy
2. torch
3. pandas
4. random
5. tqdm
6. os
7. pickle

## Running the code

To run the Aux-Drop model, change the control parameters accordingly in the **main.py** file and run
 - `python Code/main.py`

To run the baseline ODL model, change the control parameters accordingly in the **baseline.py** file and run
 - `python Code/baseline.py`



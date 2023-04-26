# Code for German Data

# All libraries
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pandas as pd
from tqdm import tqdm
from ODL import ODL
from dataset import dataset

# Data description
# "SUSY", "HIGGS"
data_name = "SUSY"

# Choose a model to run
# "ODL"
model_to_run = "ODL"

# Choose data type
# "only_base", "all_feat"
data_type = "all_feat"

# Values to change
n = 0.05
max_num_hidden_layers = 11
qtd_neuron_per_hidden_layer = 50
n_classes = 2
batch_size = 1
b = 0.99
s = 0.2
use_cuda = False
number_of_experiments = 1

error_list = []
for ex in range(number_of_experiments):
    print("Experiment number ", ex+1)
    seed = random.randint(0, 10000)

    # Please change the value of hyperparameters in the dataset.py file corresponding to the chose data name
    n_base_feat, _ , X_base, X_aux, _ , _ , Y, label = dataset(data_name, type = "variable_p", aux_feat_prob = 0.99, use_cuda = use_cuda, seed = seed)

    if data_type == "only_base":
          X = X_base
    elif data_type == "all_feat":
          X = np.concatenate((X_base, X_aux), axis = 1)
    else:
          print("Choose correct data type")
          exit()
    model = ODL(features_size = X.shape[1], max_num_hidden_layers = max_num_hidden_layers, qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer,
                 n_classes = n_classes, batch_size=batch_size, b=b, n=n, s=s, use_cuda=use_cuda)
    
    # Run the model
    N = X_base.shape[0]
    for i in tqdm(range(N)):
            model.partial_fit(X[i].reshape(1,X.shape[1]), Y[i].reshape(1))

    prediction = model.prediction
    error = len(prediction) - sum(prediction == label)
    # print("The error in the ", data_name, " dataset is ", error)
    # print(np.sum(aux_mask))
    error_list.append(error)

print("The mean error in the ", data_name, " dataset for ", number_of_experiments, " number of experiments is ", np.mean(error_list), 
        " and the standard deviation is ", np.std(error_list))

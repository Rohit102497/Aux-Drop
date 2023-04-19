import pandas as pd
import random
import numpy as np
import torch
import os


# features_size - Number of base features
# max_num_hidden_layers - Number of hidden layers
# qtd_neuron_per_hidden_layer - Number of nodes in each hidden layer except the AuxLayer
# n_classes - The total number of classes (output labels)
# aux_layer - The position of auxiliary layer. This code does not work if the AuxLayer position is 1. 
# n_neuron_aux_layer - The total numebr of neurons in the AuxLayer
# batch_size - The batch size is always 1 since it is based on stochastic gradient descent
# b - discount rate
# n - learning rate
# s - smoothing rate
# dropout_p - The dropout rate in the AuxLayer
# n_aux_feat - Number of auxiliary features
# aux_feat_prob - The probability of each auxiliary feature being available at each point in time



def dataset(name = "german"):
    # Values to change
    n = 0.1
    aux_feat_prob = 0.27
    dropout_p = 0.3
    max_num_hidden_layers = 6
    qtd_neuron_per_hidden_layer = 50
    n_classes = 2
    aux_layer = 3
    n_aux_feat = 22
    n_neuron_aux_layer = 100
    batch_size = 1
    b = 0.99
    s = 0.2
    use_cuda = False
    seed = 2022 # Change this value for each experiment

    # Initializing seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if name == "german":
        # Data description
        # Path to data
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'german.data-numeric')
        n_feat = 24
        number_of_instances = 1000

        # reading csv files
        data_initial =  pd.read_csv(data_path, sep = "  " , header = None, engine = 'python')
        data_initial.iloc[np.array(data_initial[24].isnull()), 24] = 2.0
        data_shuffled = data_initial.sample(frac = 1) # Randomly shuffling the dataset
        label = np.array(data_shuffled[24] == 1)*1
        data = data_shuffled.iloc[: , :24]
        data.insert(0, column='class', value=label)
        for i in range(data.shape[0]):
                data.iloc[i,3] = int(data.iloc[i,3].split(" ")[1])

        # Masking
        aux_mask = (np.random.random((number_of_instances, n_aux_feat)) < aux_feat_prob).astype(float)

        # Data division
        n_base_feat = data.shape[1] - 1 - n_aux_feat
        Y = np.array(data.iloc[:,:1])
        X_base = np.array(data.iloc[:,1:n_base_feat+1])
        X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        X_aux_new = np.where(aux_mask, X_aux, 0)

        return n_base_feat, max_num_hidden_layers, qtd_neuron_per_hidden_layer, \
                n_classes, aux_layer, n_neuron_aux_layer, batch_size, b,  n, s, \
                dropout_p, n_aux_feat,  use_cuda, X_base, X_aux_new, aux_mask, Y, label
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
from AuxDrop import AuxDrop_ODL, AuxDrop_OGD, AuxDrop_ODL_AuxLayer1stlayer
from AuxDrop import AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer, AuxDrop_ODL_RandomAllLayer 
from AuxDrop import AuxDrop_ODL_RandomInAuxLayer, AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst
from dataset import dataset

# Data description
# "german", "svmguide3", "magic04", "a8a", "ItalyPowerDemand", "SUSY", "HIGGS"
data_name = "magic04"

# Choose the type of data unavailability
# type can be - "variable_p", "trapezoidal", "obsolete_sudden"
type = "variable_p"

# Choose a model to run
# "AuxDrop_ODL" - Aux-Drop applied on ODL framework
#  "AuxDrop_OGD" - Aux-Drop applied on OGD framework
# "AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer" -  On ODL framework, Aux-Dropout in AuxLayer and Random dropout in all the other layers
# "AuxDrop_ODL_RandomAllLayer" - On ODL framework, Random Dropout applied in all the layers
#  "AuxDrop_ODL_RandomInAuxLayer" - On ODL framework, Random Dropout applied in the AuxLayer
# "AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst" - On ODL framework, Random Dropout applied in the first layer and all the features (base + auxiliary) are passed to the first layer
model_to_run = "AuxDrop_ODL"

# Values to change
n = 0.1
aux_feat_prob = 0.27
dropout_p = 0.3
max_num_hidden_layers = 6
qtd_neuron_per_hidden_layer = 50
n_classes = 2
aux_layer = 3
n_neuron_aux_layer = 100
batch_size = 1
b = 0.99
s = 0.2
use_cuda = False
number_of_experiments = 1

print("The model is run in ", model_to_run, " with aux layer as ", aux_layer, " and type of dataset as ", type)

error_list = []
loss_list = []
for ex in range(number_of_experiments):
        print("Experiment number ", ex+1)
        seed = random.randint(0, 10000)

        if data_name == "a8a":
                if type == "trapezoidal" and n_neuron_aux_layer < 600:
                        print("For a8a dataset please set the number of neurons in aux layer to be more than or equal to 600")
                        exit()
                elif type == "variable_p" and n_neuron_aux_layer < 400:
                        print("For a8a dataset please set the number of neurons in aux layer to be more than or equal to 400")
                        exit()

        # Please change the value of hyperparameters in the dataset.py file corresponding to the chose data name
        n_base_feat, n_aux_feat, X_base, X_aux, X_aux_new, aux_mask, Y, label = dataset(data_name, type = type, aux_feat_prob = aux_feat_prob, use_cuda = use_cuda, seed = seed)
        # Note: X_aux_new contains the auxiliary data with some data unavailable. 
        # X_aux contains the auxiliary features with all the data (even the unavailable ones)

        model = None
        if model_to_run == "AuxDrop_ODL":
                if aux_layer == 1:
                        model = AuxDrop_ODL_AuxLayer1stlayer(features_size = n_base_feat, max_num_hidden_layers = max_num_hidden_layers, qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer, 
                                n_classes = n_classes, aux_layer = aux_layer, n_neuron_aux_layer = n_neuron_aux_layer, batch_size = batch_size, b = b, n = n, s = s,
                                dropout_p = dropout_p, n_aux_feat = n_aux_feat,  use_cuda = use_cuda)
                else:
                        # Creating the Aux-Drop(ODL) Model
                        model = AuxDrop_ODL(features_size = n_base_feat, max_num_hidden_layers = max_num_hidden_layers, qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer, 
                                n_classes = n_classes, aux_layer = aux_layer, n_neuron_aux_layer = n_neuron_aux_layer, batch_size = batch_size, b = b, n = n, s = s,
                                dropout_p = dropout_p, n_aux_feat = n_aux_feat,  use_cuda = use_cuda)

        if model_to_run == "AuxDrop_OGD":
                if data_name in ["ItalyPowerDemand", "SUSY", "HIGGS"]:
                        print("You need to make some changes in the code to support AuxDrop_OGD with ", data_name, " dataset")
                        exit()
                # Creating the Aux-Drop(OGD) use this - The position of AuxLayer cannot be 1 here
                if aux_layer == 1:
                        print("Error: Please choose the aux layer position greater than 1")
                        exit()
                else:
                        model = AuxDrop_OGD(features_size = n_base_feat, max_num_hidden_layers = max_num_hidden_layers, 
                                qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer,
                                n_classes = n_classes, aux_layer = aux_layer, n_neuron_aux_layer = n_neuron_aux_layer, 
                                batch_size = batch_size, n_aux_feat = n_aux_feat, n= n, dropout_p = dropout_p)

        if model_to_run == "AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer":
                if data_name in ["german", "svmguide3", "magic04", "a8a"]:
                        model = AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer(features_size = n_base_feat, max_num_hidden_layers = max_num_hidden_layers, 
                                qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer,
                                n_classes = n_classes, aux_layer = aux_layer, n_neuron_aux_layer = n_neuron_aux_layer, 
                                batch_size = batch_size, n_aux_feat = n_aux_feat, n= n, dropout_p = dropout_p)
                else:
                        print("Choose dataset in ", ["german", "svmguide3", "magic04", "a8a"])
                        exit()

        if model_to_run == "AuxDrop_ODL_RandomAllLayer":
                if data_name in ["german", "svmguide3", "magic04", "a8a"]:
                        model = AuxDrop_ODL_RandomAllLayer(features_size = n_base_feat, max_num_hidden_layers = max_num_hidden_layers, 
                                qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer,
                                n_classes = n_classes, aux_layer = aux_layer, n_neuron_aux_layer = n_neuron_aux_layer, 
                                batch_size = batch_size, n_aux_feat = n_aux_feat, n= n, dropout_p = dropout_p)
                else:
                        print("Choose dataset in ", ["german", "svmguide3", "magic04", "a8a"])
                        exit()
        
        if model_to_run == "AuxDrop_ODL_RandomInAuxLayer":
                if data_name in ["german", "svmguide3", "magic04", "a8a"]:
                        model = AuxDrop_ODL_RandomInAuxLayer(features_size = n_base_feat, max_num_hidden_layers = max_num_hidden_layers, 
                                qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer,
                                n_classes = n_classes, aux_layer = aux_layer, n_neuron_aux_layer = n_neuron_aux_layer, 
                                batch_size = batch_size, n_aux_feat = n_aux_feat, n= n, dropout_p = dropout_p)
                else:
                        print("Choose dataset in ", ["german", "svmguide3", "magic04", "a8a"])
                        exit()

        if model_to_run == "AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst":
                if data_name in ["german", "svmguide3", "magic04", "a8a"]:
                        model = AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst(features_size = n_base_feat, max_num_hidden_layers = max_num_hidden_layers, 
                                qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer,
                                n_classes = n_classes, aux_layer = aux_layer, n_neuron_aux_layer = n_neuron_aux_layer, 
                                batch_size = batch_size, n_aux_feat = n_aux_feat, n= n, dropout_p = dropout_p)
                else:
                        print("Choose dataset in ", ["german", "svmguide3", "magic04", "a8a"])
                        exit()

        # Run the model
        N = X_base.shape[0]
        for i in tqdm(range(N)):
                model.partial_fit(X_base[i].reshape(1,n_base_feat), X_aux_new[i].reshape(1, n_aux_feat), aux_mask[i].reshape(1,n_aux_feat), Y[i].reshape(1))

        # Calculate error or loss
        if data_name == "ItalyPowerDemand":
                loss = np.mean(model.loss_array)
                # print("The loss in the ", data_name, " dataset is ", loss)
                loss_list.append(loss)
        else:
                prediction = []
                for i in model.prediction:
                        prediction.append(torch.argmax(i).item())
                error = len(prediction) - sum(prediction == label)
                # print("The error in the ", data_name, " dataset is ", error)
                # print(np.sum(aux_mask))
                error_list.append(error)


if data_name == "ItalyPowerDemand":
        print("The mean loss in the ", data_name, " dataset for ", number_of_experiments, " number of experiments is ", np.mean(loss_list),
              " and the standard deviation is ", np.std(loss_list))
else:
        print("The mean error in the ", data_name, " dataset for ", number_of_experiments, " number of experiments is ", np.mean(error_list), 
                " and the standard deviation is ", np.std(error_list))
        
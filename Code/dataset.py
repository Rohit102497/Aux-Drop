import pandas as pd
import random
import numpy as np
import torch
import os
import pickle

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



def dataset(name = "german", type = "variable_p", aux_feat_prob = 0.5, use_cuda = False, seed = 2022):
    
    # Initializing seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if name == "german":
        # Data description
        # Path to data
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'german.data-numeric')
        n_feat = 24
        n_aux_feat = 22
        n_base_feat = n_feat - n_aux_feat
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
        if type == "trapezoidal":
                num_chunks = 10
                chunk_size = int(number_of_instances/10)
                aux_mask = np.zeros((number_of_instances, n_aux_feat))
                aux_feat_chunk_list = [round((n_feat/num_chunks)*i) - n_base_feat for i in range(1, num_chunks+1)]
                if aux_feat_chunk_list[0] == -1:
                        aux_feat_chunk_list[0] = 0
                aux_feat_chunk_list
                for i in range(num_chunks):
                        aux_mask[chunk_size*i:chunk_size*(i+1), :aux_feat_chunk_list[i]] = 1
        elif type == "variable_p":
                aux_mask = (np.random.random((number_of_instances, n_aux_feat)) < aux_feat_prob).astype(float)
        else:
                print("Please choose the type as \"variable_p\" for ", name, " dataset")
                exit()

        # Data division
        n_base_feat = data.shape[1] - 1 - n_aux_feat
        Y = np.array(data.iloc[:,:1])
        X_base = np.array(data.iloc[:,1:n_base_feat+1])
        X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        X_aux_new = np.where(aux_mask, X_aux, 0)

        return n_base_feat, n_aux_feat,  X_base, X_aux, X_aux_new, aux_mask, Y, label
    
    if name == "svmguide3":
        # Data description
        # Path to data
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'svmguide3.txt')
        n_feat = 21
        n_aux_feat = 19
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 1243

        # reading csv files
        # data_initial =  arff.loadarff(data_path)
        data_initial =  pd.read_csv(data_path, sep=" ", header=None)
        data_initial = data_initial.iloc[:, :22]
        for j in range(1, data_initial.shape[1]):
                for i in range(data_initial.shape[0]):
                        data_initial.iloc[i, j] = data_initial.iloc[i, j].split(":")[1]
        for i in range(data_initial.shape[0]):
                data_initial.iloc[i, 0] = (data_initial.iloc[i, 0] == -1)*1
        data = data_initial.sample(frac = 1)
        label = np.asarray(data[0])

        # Masking
        if type == "trapezoidal":
                num_chunks = 10
                chunk_size = int(number_of_instances/10)
                aux_mask = np.zeros((number_of_instances, n_aux_feat))
                aux_feat_chunk_list = [round((n_feat/num_chunks)*i) - n_base_feat for i in range(1, num_chunks+1)]
                if aux_feat_chunk_list[0] == -1:
                        aux_feat_chunk_list[0] = 0
                aux_feat_chunk_list
                for i in range(num_chunks):
                        aux_mask[chunk_size*i:chunk_size*(i+1), :aux_feat_chunk_list[i]] = 1
        elif type == "variable_p":
                aux_mask = (np.random.random((number_of_instances, n_aux_feat)) < aux_feat_prob).astype(float)
        else:
                print("Please choose the type as \"variable_p\" for ", name, " dataset")
                exit()

        # Data division
        n_base_feat = data.shape[1] - 1 - n_aux_feat
        Y = np.array(data.iloc[:,:1])
        X_base = np.array(data.iloc[:,1:n_base_feat+1], dtype = float)
        X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        X_aux_new = np.where(aux_mask, X_aux, 0)
        
        return n_base_feat, n_aux_feat,  X_base, X_aux, X_aux_new, aux_mask, Y, label
    
    if name == "magic04":
        # Data description
        # Path to data
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'magic04.data')
        n_feat = 10
        n_aux_feat = 8
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 19020

        # reading csv files
        data_initial =  pd.read_csv(data_path, sep=",", header=None)
        data_shuffled = data_initial.sample(frac = 1)
        label = np.array(data_shuffled[n_feat] == "g")*1
        data = data_shuffled.iloc[: , :n_feat]
        data.insert(0, column='class', value=label)


        # Masking
        if type == "trapezoidal":
                num_chunks = 10
                chunk_size = int(number_of_instances/10)
                aux_mask = np.zeros((number_of_instances, n_aux_feat))
                aux_feat_chunk_list = [round((n_feat/num_chunks)*i) - n_base_feat for i in range(1, num_chunks+1)]
                if aux_feat_chunk_list[0] == -1:
                        aux_feat_chunk_list[0] = 0
                aux_feat_chunk_list
                for i in range(num_chunks):
                        aux_mask[chunk_size*i:chunk_size*(i+1), :aux_feat_chunk_list[i]] = 1
        elif type == "variable_p":
                aux_mask = (np.random.random((number_of_instances, n_aux_feat)) < aux_feat_prob).astype(float)
        else:
                print("Please choose the type as \"variable_p\" for ", name, " dataset")
                exit()

        # Data division
        n_base_feat = data.shape[1] - 1 - n_aux_feat
        Y = np.array(data.iloc[:,:1])
        X_base = np.array(data.iloc[:,1:n_base_feat+1])
        X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        X_aux_new = np.where(aux_mask, X_aux, 0)

        return n_base_feat, n_aux_feat,  X_base, X_aux, X_aux_new, aux_mask, Y, label

    if name == "a8a":
        # Data description
        # Path to data
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'a8a.txt')
        n_feat = 123
        n_aux_feat = 121
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 32561

        data = pd.DataFrame(0, index=range(number_of_instances), columns = list(range(1, n_feat+1)))
        # reading csv files
        data_initial =  pd.read_csv(data_path, sep=" ", header=None)
        data_initial = data_initial.iloc[:, :15]
        for j in range(data_initial.shape[0]):
                l = [int(i.split(":")[0])-1 for i in list(data_initial.iloc[j, 1:]) if not pd.isnull(i)]
                data.iloc[j, l] = 1
        label = np.array(data_initial[0] == -1)*1
        data.insert(0, column='class', value=label)
        data = data.sample(frac = 1)
        label = np.array(data["class"])

        # Masking
        if type == "trapezoidal":
                num_chunks = 10
                chunk_size = int(number_of_instances/10)
                aux_mask = np.zeros((number_of_instances, n_aux_feat))
                aux_feat_chunk_list = [round((n_feat/num_chunks)*i) - n_base_feat for i in range(1, num_chunks+1)]
                if aux_feat_chunk_list[0] == -1:
                        aux_feat_chunk_list[0] = 0
                aux_feat_chunk_list
                for i in range(num_chunks):
                        aux_mask[chunk_size*i:chunk_size*(i+1), :aux_feat_chunk_list[i]] = 1
        elif type == "variable_p":
               aux_mask = (np.random.random((number_of_instances, n_aux_feat)) < aux_feat_prob).astype(float)
        else:
                print("Please choose the type as \"variable_p\" for ", name, " dataset")
                exit()

        # Data division
        n_base_feat = data.shape[1] - 1 - n_aux_feat
        Y = np.array(data.iloc[:,:1])
        X_base = np.array(data.iloc[:,1:n_base_feat+1], dtype = float)
        X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        X_aux_new = np.where(aux_mask, X_aux, 0)

        return n_base_feat, n_aux_feat,  X_base, X_aux, X_aux_new, aux_mask, Y, label

    if name == "ItalyPowerDemand":
        # Data description
        # Path to data
        data_path_train = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'ItalyPowerDemand_TRAIN.txt')
        data_path_test = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'ItalyPowerDemand_TEST.txt')
        n_feat = 24
        n_aux_feat = 12
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 1096

        # Load Data
        data_train = pd.read_csv(data_path_train, sep = "  ", header = None, engine = 'python')
        data_test = pd.read_csv(data_path_test, sep = "  ", header = None, engine = 'python')
        data = pd.concat([data_train, data_test])
        label = np.array(data[0] == 1.0)*1
        
        # Masking
        if type == "trapezoidal":
                print("Please choose the type as \"variable_p\" for ", name, " dataset")
                exit()
        elif type == "variable_p":
               aux_mask = (np.random.random((number_of_instances, n_aux_feat)) < aux_feat_prob).astype(float)
        else:
                print("Please choose the type as \"variable_p\" for ", name, " dataset")
                exit()

        # Data division
        n_base_feat = data.shape[1] - 1 - n_aux_feat
        Y = np.array(data.iloc[:,:1]) - 1
        X_base = np.array(data.iloc[:,1:n_base_feat+1], dtype = float)
        X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        X_aux_new = np.where(aux_mask, X_aux, 0)

        return n_base_feat, n_aux_feat,  X_base, X_aux, X_aux_new, aux_mask, Y, label
    
    if name == "SUSY":
        # Data description
        # Path to data
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'data', 'SUSY_1M.csv.gz')
        n_feat = 8
        n_aux_feat = 6
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 1000000 # 1M
        number_of_instances_name = "1M"
        Start = "100k"
        Gap = "100k"
        Stream = "400k"

        # Load Data
        data = pd.read_csv(data_path, compression='gzip', nrows=number_of_instances)
        label = np.array(data["0"] == 1.0)*1

        # Masking
        if type == "variable_p":
                mask_file_name = name + "_" + number_of_instances_name +"_P_" + str(int(aux_feat_prob*100)) + "_AuxFeat_" + str(n_aux_feat) + ".data"
                mask_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'mask', mask_file_name)
                with open(mask_path, 'rb') as file:
                        aux_mask = pickle.load(file)
        elif type == "obsolete_sudden":
                mask_file_name = name + "_" + number_of_instances_name + "_Start" + Start + "_Gap" + Gap + "_Stream" + Stream + "_AuxFeat_" + str(n_aux_feat) + ".data"
                mask_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'mask', mask_file_name)
                with open(mask_path, 'rb') as file:
                        aux_mask = pickle.load(file)
        else:
                print("Please choose the type as \"variable_p\" or \"obsolete_sudden\" for ", name, " dataset")
                exit()

        # Data division
        Y = np.array(data.iloc[:,:1])
        X_base = np.array(data.iloc[:,1:n_base_feat+1], dtype = float)
        X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        X_aux_new = np.where(aux_mask[:number_of_instances], X_aux, 0)

        return n_base_feat, n_aux_feat, X_base, X_aux, X_aux_new, aux_mask, Y, label
    
    if name == "HIGGS":
        # Data description
        # Path to data
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'data', 'HIGGS_1M.csv.gz')
        n_feat = 21
        n_aux_feat = 16
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 1000000 # 1M
        number_of_instances_name = "1M"
        Start = "50k"
        Gap = "50k"
        Stream = "200k"

        # Load Data
        data = pd.read_csv(data_path, compression='gzip', nrows=number_of_instances)
        label = np.array(data["0"] == 1.0)*1

        # Masking
        if type == "variable_p":
                mask_file_name = name + "_" + number_of_instances_name +"_P_" + str(int(aux_feat_prob*100)) + "_AuxFeat_" + str(n_aux_feat) + ".data"
                mask_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'mask', mask_file_name)
                with open(mask_path, 'rb') as file:
                        aux_mask = pickle.load(file)
        elif type == "obsolete_sudden":
                mask_file_name = name + "_" + number_of_instances_name + "_Start" + Start + "_Gap" + Gap + "_Stream" + Stream + "_AuxFeat_" + str(n_aux_feat) + ".data"
                mask_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'mask', mask_file_name)
                with open(mask_path, 'rb') as file:
                        aux_mask = pickle.load(file)
        else:
                print("Please choose the type as \"variable_p\" or \"obsolete_sudden\" for ", name, " dataset")
                exit()

        # Data division
        Y = np.array(data.iloc[:,:1])
        X_base = np.array(data.iloc[:,1:n_base_feat+1], dtype = float)
        X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        X_aux_new = np.where(aux_mask[:number_of_instances], X_aux, 0)

        return n_base_feat, n_aux_feat, X_base, X_aux, X_aux_new, aux_mask, Y, label
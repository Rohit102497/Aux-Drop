# Libraries requied
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim

# code for Aux-Drop(ODL)
class AuxDrop_ODL(nn.Module):
    def __init__(self, features_size, max_num_hidden_layers, qtd_neuron_per_hidden_layer, n_classes, aux_layer, 
                 n_neuron_aux_layer, batch_size=1, b=0.99, n=0.01, s=0.2, dropout_p=0.5, n_aux_feat = 3, use_cuda=False):
        super(AuxDrop_ODL, self).__init__()

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

        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.batch_size = batch_size
        self.n_aux_feat = n_aux_feat

        self.b = Parameter(torch.tensor(
            b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(
            n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(
            s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(
            dropout_p), requires_grad=False).to(self.device)

        # Stores hidden and output layers
        self.hidden_layers = []
        self.output_layers = []

        # The number of hidden layers would be "max_num_hidden_layers". The input to the first layer is the base features i.e. "features_size".
        self.hidden_layers.append(
            nn.Linear(features_size, qtd_neuron_per_hidden_layer))

        for i in range(max_num_hidden_layers - 1):
            # The input to the aux_layer is the outpout coming from its previous layer, i.e., "qtd_neuron_per_hidden_layer" and the 
            # number of auxiliary features, "n_aux_feat".
            if i+2 == aux_layer:
                self.hidden_layers.append(
                    nn.Linear(n_aux_feat + qtd_neuron_per_hidden_layer, n_neuron_aux_layer))
            elif i + 1 == aux_layer:
                self.hidden_layers.append(
                    nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer))
            else:
                self.hidden_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer))
            
        # The number of output layer would be total number of layer ("max_num_hidden_layers") - 2 (no output from the aux layer and the first layer)
        # No output is taken from the first layer following the ODL paper (see ODL code and the baseline paragraph in the experiemnts section in ODL paper)
        for i in range(max_num_hidden_layers - 2):
                self.output_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        self.hidden_layers = nn.ModuleList(self.hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(self.output_layers).to(self.device)
        
        # Alphas are related to the output layers. So, the nummber of alphas would be equal to the number of hidden layers (max_num_hidden_layers) - 2.
        # The alpha value sums to 1 and is equal at the beginning of the training. 
        self.alpha = Parameter(torch.Tensor(self.max_num_hidden_layers - 2).fill_(1 / (self.max_num_hidden_layers - 2)),
                               requires_grad=False).to(self.device)


        # We store loss, alpha and prediction parameter.
        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

    # Initialize the gradients of all the parameters with 0.
    def zero_grad(self):
        for i in range(self.max_num_hidden_layers - 2):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    # This function predicts the output from each layer,calculates the loss and do gradient descent and alpha update.
    def update_weights(self, X, aux_feat, aux_mask, Y, show_loss):
        Y = torch.from_numpy(Y).to(self.device)

        # Predictions from each layer using the implemented forward function.
        predictions_per_layer = self.forward(X, aux_feat, aux_mask)

        real_output = torch.sum(torch.mul(
                self.alpha.view(self.max_num_hidden_layers - 2, 1).repeat(1, self.batch_size).view(
                    self.max_num_hidden_layers - 2, self.batch_size, 1), predictions_per_layer), 0)
        self.prediction.append(real_output)
        
        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(real_output.view(self.batch_size, self.n_classes), Y.view(self.batch_size).long())
        self.loss_array.append(loss.detach().numpy())

        if show_loss:
            if (len(self.loss_array) % 1000) == 0:
                print("WARNING: Set 'show_loss' to 'False' when not debugging. "
                      "It will deteriorate the fitting performance.")
                loss = np.mean(self.loss_array[-1000:])
                print("Alpha:" + str(self.alpha.data.cpu().numpy()))
                print("Training Loss: " + str(loss))
        
        losses_per_layer = []

        for out in predictions_per_layer:
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(out.view(self.batch_size, self.n_classes), Y.view(
                self.batch_size).long())
            losses_per_layer.append(loss)

        w = [None] * (len(losses_per_layer) + 2) 
        b = [None] * (len(losses_per_layer) + 2)

        with torch.no_grad():

            for i in range(len(losses_per_layer)):
                losses_per_layer[i].backward(retain_graph = True)
                
                self.output_layers[i].weight.data -= self.n* \
                                                     self.alpha[i]*self.output_layers[i].weight.grad.data
                self.output_layers[i].bias.data -= self.n * \
                                                   self.alpha[i] * self.output_layers[i].bias.grad.data

                if i < self.aux_layer - 2:
                    for j in range(i+2):
                        if w[j] is None:
                            w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                        else:
                            w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data
                if i > self.aux_layer - 3:
                    for j in range(i+3):
                        if w[j] is None:
                            w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                        else:
                            w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad()

            for i in range(self.max_num_hidden_layers):
              self.hidden_layers[i].weight.data -= self.n * w[i]
              self.hidden_layers[i].bias.data -= self.n * b[i]

            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(
                    self.alpha[i], self.s / (len(losses_per_layer)))
            
        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(
            self.alpha / z_t, requires_grad=False).to(self.device)
        
        # To save the loss
        detached_loss = []
        for i in range(len(losses_per_layer)):
            detached_loss.append(losses_per_layer[i].detach().numpy())
        self.layerwise_loss_array.append(np.asarray(detached_loss))
        self.alpha_array.append(self.alpha.detach().numpy())

    # Forward pass. Get the output from each layer.     
    def forward(self, X, aux_feat, aux_mask):
        hidden_connections = []
        linear_x = []
        relu_x = []

        X = torch.from_numpy(X).float().to(self.device)
        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Forward pass of the first hidden layer. Apply the linear transformation and then relu. The output from the relu is the input
        # passed to the next layer.
        linear_x.append(self.hidden_layers[0](X))
        relu_x.append(F.relu(linear_x[0]))
        hidden_connections.append(relu_x[0])


        for i in range(1, self.max_num_hidden_layers):
            # Forward pass to the Aux layer.
            if i==self.aux_layer-1:
                # Input to the aux layer will be the output from its previous layer and the incoming auxiliary inputs.
                linear_x.append(self.hidden_layers[i](torch.cat((aux_feat, hidden_connections[i - 1]), dim=1)))
                relu_x.append(F.relu(linear_x[i]))
                # We apply dropout in the aux layer.
                # Based on the incoming aux data, the aux inputs which do not come gets included in the dropout probability.
                # Based on that we calculate the probability to drop the left over neurons in auxiliary layer.
                aux_p = (self.p * self.n_neuron_aux_layer - (aux_mask.size()[1] - torch.sum(aux_mask)))/(self.n_neuron_aux_layer 
                            - aux_mask.size()[1])
                binomial = torch.distributions.binomial.Binomial(probs=1-aux_p)
                non_aux_mask = binomial.sample([1, self.n_neuron_aux_layer - aux_mask.size()[1]])
                mask = torch.cat((aux_mask, non_aux_mask), dim = 1)
                hidden_connections.append(relu_x[i]*mask*(1.0/(1-self.p)))
            else:
                linear_x.append(self.hidden_layers[i](hidden_connections[i - 1]))
                relu_x.append(F.relu(linear_x[i]))
                hidden_connections.append(relu_x[i])
        
        output_class = []

        for i in range(self.max_num_hidden_layers-1):
            if i < self.aux_layer-2:
                output_class.append(
                    F.softmax(self.output_layers[i](hidden_connections[i+1]), dim=1))
            if i > self.aux_layer-2:
                output_class.append(
                    F.softmax(self.output_layers[i-1](hidden_connections[i+1]), dim=1))

        pred_per_layer = torch.stack(output_class)

        return pred_per_layer

    def validate_input_X(self, data):
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions.")
    
    def validate_input_Y(self, data):
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions.")
    
    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, aux_data, aux_mask, Y_data, show_loss)


# ------------------------------------------------------------------------------------------------- #

# code for Aux-Drop(ODL) when the position of AuxLayer is 1
class AuxDrop_ODL_AuxLayer1stlayer(nn.Module):
    def __init__(self, features_size, 
                 max_num_hidden_layers, qtd_neuron_per_hidden_layer, 
                 n_classes, aux_layer, n_neuron_aux_layer, 
                 batch_size=1, b=0.99, n=0.01, s=0.2, 
                 dropout_p=0.5, n_aux_feat = 3, use_cuda=False):
        super(AuxDrop_ODL_AuxLayer1stlayer, self).__init__()

        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.batch_size = batch_size

        self.b = Parameter(torch.tensor(
            b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(
            n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(
            s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(
            dropout_p), requires_grad=False).to(self.device)
        self.n_aux_feat = n_aux_feat

        # Stores hidden and output layers
        self.hidden_layers = []
        self.output_layers = []

        # The number of hidden layers would be "max_num_hidden_layers". The input to the first layer is the base features i.e. "features_size" and the auxiliary features.
        self.hidden_layers.append(
            nn.Linear(features_size + n_aux_feat, n_neuron_aux_layer))
        self.hidden_layers.append(
                    nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer))
        for i in range(max_num_hidden_layers - 2):
            self.hidden_layers.append(
                nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer))
            
        # The number of output layer would be total number of layer ("max_num_hidden_layers") - 1 (no output from the aux layer/first layer)
        for i in range(max_num_hidden_layers - 1):
                self.output_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        self.hidden_layers = nn.ModuleList(self.hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(self.output_layers).to(self.device)
        
        # Alphs is related to the output layers. So, the nummber of alphas would be equal to the number of hidden layers (max_num_hidden_layers) - 1.
        # The alpha value sums to 1 and is equal at the beginning of the training. 
        self.alpha = Parameter(torch.Tensor(self.max_num_hidden_layers - 1).fill_(1 / (self.max_num_hidden_layers - 1)),
                               requires_grad=False).to(self.device)


        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

    # Initialize the gradients of all the parameters with 0.
    def zero_grad(self):
        for i in range(self.max_num_hidden_layers - 1):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    # This function predicts the output from each layer,calculates the loss and do gradient descent and alpha update.
    def update_weights(self, X, aux_feat, aux_mask, Y, show_loss):
        Y = torch.from_numpy(Y).to(self.device)

        # Predictions from each layer using the implemented forward function.
        predictions_per_layer = self.forward(X, aux_feat, aux_mask)

        real_output = torch.sum(torch.mul(
                self.alpha.view(self.max_num_hidden_layers - 1, 1).repeat(1, self.batch_size).view(
                    self.max_num_hidden_layers - 1, self.batch_size, 1), predictions_per_layer), 0)
        self.prediction.append(real_output)

        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(real_output.view(self.batch_size, self.n_classes), Y.view(self.batch_size).long())
        self.loss_array.append(loss.detach().numpy())
        if show_loss:
            if (len(self.loss_array) % 1000) == 0:
                print("WARNING: Set 'show_loss' to 'False' when not debugging. "
                      "It will deteriorate the fitting performance.")
                loss = np.mean(self.loss_array[-1000:])
                print("Alpha:" + str(self.alpha.data.cpu().numpy()))
                print("Training Loss: " + str(loss))
        
        losses_per_layer = []

        for out in predictions_per_layer:
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(out.view(self.batch_size, self.n_classes), Y.view(
                self.batch_size).long())
            losses_per_layer.append(loss)
        
        w = [None] * (len(losses_per_layer) + 1) 
        b = [None] * (len(losses_per_layer) + 1)

        with torch.no_grad():

            for i in range(len(losses_per_layer)):
                losses_per_layer[i].backward(retain_graph = True)                
                
                self.output_layers[i].weight.data -= self.n* \
                                                     self.alpha[i]*self.output_layers[i].weight.grad.data
                self.output_layers[i].bias.data -= self.n * \
                                                   self.alpha[i] * self.output_layers[i].bias.grad.data

                for j in range(i+2):
                    if w[j] is None:
                        w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                    else:
                        w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad()

            for i in range(self.max_num_hidden_layers):
              self.hidden_layers[i].weight.data -= self.n * w[i]
              self.hidden_layers[i].bias.data -= self.n * b[i]

            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(
                    self.alpha[i], self.s / (len(losses_per_layer)))
            
        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(
            self.alpha / z_t, requires_grad=False).to(self.device)
        
        # To save the loss
        detached_loss = []
        for i in range(len(losses_per_layer)):
            detached_loss.append(losses_per_layer[i].detach().numpy())
        self.layerwise_loss_array.append(np.asarray(detached_loss))
        self.alpha_array.append(self.alpha.detach().numpy())

    # Forward pass. Get the output from each layer.     
    def forward(self, X, aux_feat, aux_mask):
        hidden_connections = []
        linear_x = []
        relu_x = []

        X = torch.from_numpy(X).float().to(self.device)
        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Forward pass to the Aux layer.
        # Input to the aux layer will be the output from its previous layer and the incoming auxiliary inputs.
        linear_x.append(self.hidden_layers[0](torch.cat((aux_feat, X), dim=1)))
        relu_x.append(F.relu(linear_x[0]))
        # Based on the incoming aux data, the aux inputs which do not come gets included in the dropout probability.
        # Based on that we calculate the probability to drop the left over neurons in auxiliary layer.
        aux_p = (self.p * self.n_neuron_aux_layer - 
                 (aux_mask.size()[1] - torch.sum(aux_mask)))/(self.n_neuron_aux_layer 
                    - aux_mask.size()[1])
        binomial = torch.distributions.binomial.Binomial(probs=1-aux_p)
        non_aux_mask = binomial.sample([1, self.n_neuron_aux_layer - aux_mask.size()[1]])
        mask = torch.cat((aux_mask, non_aux_mask), dim = 1)
        hidden_connections.append(relu_x[0]*mask*(1.0/(1-self.p)))


        for i in range(1, self.max_num_hidden_layers):
            linear_x.append(self.hidden_layers[i](hidden_connections[i - 1]))
            relu_x.append(F.relu(linear_x[i]))
            hidden_connections.append(relu_x[i])
        
        output_class = []

        for i in range(self.max_num_hidden_layers-1):
            output_class.append(
                F.softmax(self.output_layers[i](hidden_connections[i+1]), dim=1))

        pred_per_layer = torch.stack(output_class)

        return pred_per_layer

    def validate_input_X(self, data):
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions.")
    
    def validate_input_Y(self, data):
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions.")
    
    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, aux_data, aux_mask, Y_data, show_loss)

# ------------------------------------------------------------------------------------------------- #

# Aux-Drop (OGD) code
class AuxDrop_OGD(nn.Module):
    def __init__(self, features_size, max_num_hidden_layers = 5, qtd_neuron_per_hidden_layer = 100, n_classes = 2, aux_layer = 3, 
                 n_neuron_aux_layer = 100, batch_size=1, n_aux_feat = 3, n=0.01, dropout_p=0.5):
        super(AuxDrop_OGD, self).__init__()

        self.features_size = features_size 
        self.max_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.aux_layer = aux_layer 
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.n_aux_feat = n_aux_feat
        self.n_classes = n_classes
        self.p = dropout_p
        self.batch_size = batch_size
        self.n = n

        # Stores hidden layers
        self.hidden_layers = []

        self.hidden_layers.append(
            nn.Linear(self.features_size, self.qtd_neuron_per_hidden_layer, bias=True))
        for i in range(self.max_layers - 1):
            # The input to the aux_layer is the outpout coming from its previous layer, i.e., "qtd_neuron_per_hidden_layer" and the 
            # number of auxiliary features, "n_aux_feat".
            if i+2 == self.aux_layer:
                self.hidden_layers.append(
                    nn.Linear(self.n_aux_feat + self.qtd_neuron_per_hidden_layer, self.n_neuron_aux_layer, bias=True))
            elif i + 1 == self.aux_layer:
                self.hidden_layers.append(
                    nn.Linear(self.n_neuron_aux_layer, self.qtd_neuron_per_hidden_layer, bias=True))
            else:
                self.hidden_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer, bias=True))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.output_layer = nn.Linear(self.qtd_neuron_per_hidden_layer, 
            self.n_classes, bias=True)

        self.loss_fn = nn.CrossEntropyLoss()
        
        self.prediction = []
        self.loss_array = []
    
    def forward(self, X, aux_feat, aux_mask):

        X = torch.from_numpy(X).float()
        aux_feat = torch.from_numpy(aux_feat).float()
        aux_mask = torch.from_numpy(aux_mask).float()

        # Forward pass of the first hidden layer. Apply the linear transformation and then relu. The output from the relu is the input
        # passed to the next layer.
        inp = F.relu(self.hidden_layers[0](X))

        for i in range(1, self.max_layers):
            # Forward pass to the Aux layer.
            if i==self.aux_layer-1:
                # Input to the aux layer will be the output from its previous layer and the incoming auxiliary inputs.
                inp = F.relu(self.hidden_layers[i](torch.cat((aux_feat, inp), dim=1)))
                # Based on the incoming aux data, the aux inputs which do not come gets included in the dropout probability.
                # Based on that we calculate the probability to drop the left over neurons in auxiliary layer.
                
                aux_p = (self.p * self.n_neuron_aux_layer - (aux_mask.size()[1] - torch.sum(aux_mask)))/(self.n_neuron_aux_layer 
                            - aux_mask.size()[1])
                binomial = torch.distributions.binomial.Binomial(probs=1-aux_p)
                non_aux_mask = binomial.sample([1, self.n_neuron_aux_layer - aux_mask.size()[1]])
                mask = torch.cat((aux_mask, non_aux_mask), dim = 1)
                inp = inp*mask*(1.0/(1-self.p))
            else:
                inp = F.relu(self.hidden_layers[i](inp))
        
        out = F.softmax(self.output_layer(inp), dim=1)

        return out

    def validate_input_X(self, data):
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions.")
    
    def validate_input_Y(self, data):
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions.")


    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)

        optimizer = optim.SGD(self.parameters(), lr=self.n)
        optimizer.zero_grad()
        y_pred = self.forward(X_data, aux_data, aux_mask)
        self.prediction.append(y_pred)
        loss = self.loss_fn(y_pred, torch.tensor(Y_data))
        self.loss_array.append(loss.item())
        loss.backward()
        optimizer.step()

        if show_loss:
            print("Loss is: ", loss)

# ------------------------------------------------------------------------------------------------- #
# Aux-Drop (OGD) code with Directed Dropout in AuxLayer and Random in other layer
class AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer(nn.Module):
    def __init__(self, features_size, max_num_hidden_layers, qtd_neuron_per_hidden_layer, n_classes, aux_layer, 
                 n_neuron_aux_layer, batch_size=1, b=0.99, n=0.01, s=0.2, dropout_p=0.5, n_aux_feat = 3, use_cuda=False):
        super(AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer, self).__init__()

        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.batch_size = batch_size

        self.b = Parameter(torch.tensor(
            b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(
            n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(
            s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(
            dropout_p), requires_grad=False).to(self.device)
        self.n_aux_feat = n_aux_feat

        # Stores hidden and output layers
        self.hidden_layers = []
        self.output_layers = []

        # The number of hidden layers would be "max_num_hidden_layers". The input to the first layer is the base features i.e. "features_size".
        self.hidden_layers.append(
            nn.Linear(features_size, qtd_neuron_per_hidden_layer))

        for i in range(max_num_hidden_layers - 1):
            # The input to the aux_layer is the outpout coming from its previous layer, i.e., "qtd_neuron_per_hidden_layer" and the 
            # number of auxiliary features, "n_aux_feat".
            if i+2 == aux_layer:
                self.hidden_layers.append(
                    nn.Linear(n_aux_feat + qtd_neuron_per_hidden_layer, n_neuron_aux_layer))
            elif i + 1 == aux_layer:
                self.hidden_layers.append(
                    nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer))
            else:
                self.hidden_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer))
            
        # The number of output layer would be total number of layer ("max_num_hidden_layers") - 2 (no output from the aux layer and the first layer)
        for i in range(max_num_hidden_layers - 2):
                self.output_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        self.hidden_layers = nn.ModuleList(self.hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(self.output_layers).to(self.device)
        
        # Alphs is related to the output layers. So, the nummber of alphas would be equal to the number of hidden layers (max_num_hidden_layers) - 2.
        # The alpha value sums to 1 and is equal at the beginning of the training. 
        self.alpha = Parameter(torch.Tensor(self.max_num_hidden_layers - 2).fill_(1 / (self.max_num_hidden_layers - 2)),
                               requires_grad=False).to(self.device)


        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

    # Initialize the graadients of all the parameters with 0.
    def zero_grad(self):
        for i in range(self.max_num_hidden_layers - 2):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    # This function predicts the output from each layer,calculates the loss and do gradient descent and alpha update.
    def update_weights(self, X, aux_feat, aux_mask, Y, show_loss):
        Y = torch.from_numpy(Y).to(self.device)
        predictions_per_layer = self.forward(X, aux_feat, aux_mask, training=True)

        real_output = torch.sum(torch.mul(
                self.alpha.view(self.max_num_hidden_layers - 2, 1).repeat(1, self.batch_size).view(
                    self.max_num_hidden_layers - 2, self.batch_size, 1), predictions_per_layer), 0)
        self.prediction.append(real_output)
        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(real_output.view(self.batch_size, self.n_classes), Y.view(self.batch_size).long())
        self.loss_array.append(loss.detach().numpy())
        if show_loss:
            if (len(self.loss_array) % 1000) == 0:
                print("WARNING: Set 'show_loss' to 'False' when not debugging. "
                      "It will deteriorate the fitting performance.")
                loss = np.mean(self.loss_array[-1000:])
                print("Alpha:" + str(self.alpha.data.cpu().numpy()))
                print("Training Loss: " + str(loss))
        
        losses_per_layer = []

        for out in predictions_per_layer:
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(out.view(self.batch_size, self.n_classes), Y.view(
                self.batch_size).long())
            losses_per_layer.append(loss)

        w = [None] * (len(losses_per_layer) + 2) 
        b = [None] * (len(losses_per_layer) + 2)

        with torch.no_grad():

            for i in range(len(losses_per_layer)):
                losses_per_layer[i].backward(retain_graph = True)
                self.output_layers[i].weight.data -= self.n* \
                                                     self.alpha[i]*self.output_layers[i].weight.grad.data
                self.output_layers[i].bias.data -= self.n * \
                                                   self.alpha[i] * self.output_layers[i].bias.grad.data

                if i < self.aux_layer - 2:
                    for j in range(i+2):
                        if w[j] is None:
                            w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                        else:
                            w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data
                if i > self.aux_layer - 3:
                    for j in range(i+3):
                        if w[j] is None:
                            w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                        else:
                            w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad()

            for i in range(self.max_num_hidden_layers):
              self.hidden_layers[i].weight.data -= self.n * w[i]
              self.hidden_layers[i].bias.data -= self.n * b[i]

            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(
                    self.alpha[i], self.s / (len(losses_per_layer)))
            
        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(
            self.alpha / z_t, requires_grad=False).to(self.device)
        
        # To save the loss
        detached_loss = []
        for i in range(len(losses_per_layer)):
            detached_loss.append(losses_per_layer[i].detach().numpy())
        self.layerwise_loss_array.append(np.asarray(detached_loss))
        self.alpha_array.append(self.alpha.detach().numpy())

    # Forward pass. Get the output from each layer.     
    def forward(self, X, aux_feat, aux_mask, training):
        hidden_connections = []
        linear_x = []
        relu_x = []

        X = torch.from_numpy(X).float().to(self.device)
        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Forward pass of the first hidden layer. Apply the linear transformation and then relu. The output from the relu is the input
        # passed to the next layer.
        linear_x.append(self.hidden_layers[0](X))
        relu_x.append(F.relu(linear_x[0]))
        hidden_connections.append(relu_x[0])


        for i in range(1, self.max_num_hidden_layers):
            # Forward pass to the Aux layer.
            if i==self.aux_layer-1:
                # Input to the aux layer will be the output from its previous layer and the incoming auxiliary inputs.
                linear_x.append(self.hidden_layers[i](torch.cat((aux_feat, hidden_connections[i - 1]), dim=1)))
                relu_x.append(F.relu(linear_x[i]))
                # In the training process, we apply dropout in the aux layer.
                aux_p = (self.p * self.n_neuron_aux_layer - (aux_mask.size()[1] - torch.sum(aux_mask)))/(self.n_neuron_aux_layer 
                            - aux_mask.size()[1])
                binomial = torch.distributions.binomial.Binomial(probs=1-aux_p)
                non_aux_mask = binomial.sample([1, self.n_neuron_aux_layer - aux_mask.size()[1]])
                mask = torch.cat((aux_mask, non_aux_mask), dim = 1)
                hidden_connections.append(relu_x[i]*mask*(1.0/(1-self.p)))
            else:
                linear_x.append(self.hidden_layers[i](hidden_connections[i - 1]))
                relu_x.append(F.relu(linear_x[i]))
                binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
                mask = binomial.sample([1, self.qtd_neuron_per_hidden_layer])
                hidden_connections.append(relu_x[i]*mask*(1.0/(1-self.p)))
        
        output_class = []

        for i in range(self.max_num_hidden_layers-1):
            if i < self.aux_layer-2:
                output_class.append(
                    F.softmax(self.output_layers[i](hidden_connections[i+1]), dim=1))
            if i > self.aux_layer-2:
                output_class.append(
                    F.softmax(self.output_layers[i-1](hidden_connections[i+1]), dim=1))

        pred_per_layer = torch.stack(output_class)
        return pred_per_layer

    def validate_input_X(self, data):
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions.")
    
    def validate_input_Y(self, data):
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions.")
    
    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=True):
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, aux_data, aux_mask, Y_data, show_loss)

# ------------------------------------------------------------------------------------------------- #
# Aux-Drop (OGD) code with Random Dropout in all layers
class AuxDrop_ODL_RandomAllLayer(nn.Module):
    def __init__(self, features_size, max_num_hidden_layers, qtd_neuron_per_hidden_layer, n_classes, aux_layer, 
                 n_neuron_aux_layer, batch_size=1, b=0.99, n=0.01, s=0.2, dropout_p=0.5, n_aux_feat = 3, use_cuda=False):
        super(AuxDrop_ODL_RandomAllLayer, self).__init__()

        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.batch_size = batch_size

        self.b = Parameter(torch.tensor(
            b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(
            n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(
            s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(
            dropout_p), requires_grad=False).to(self.device)
        self.n_aux_feat = n_aux_feat

        # Stores hidden and output layers
        self.hidden_layers = []
        self.output_layers = []

        # The number of hidden layers would be "max_num_hidden_layers". The input to the first layer is the base features i.e. "features_size".
        self.hidden_layers.append(
            nn.Linear(features_size, qtd_neuron_per_hidden_layer))

        for i in range(max_num_hidden_layers - 1):
            # The input to the aux_layer is the outpout coming from its previous layer, i.e., "qtd_neuron_per_hidden_layer" and the 
            # number of auxiliary features, "n_aux_feat".
            if i+2 == aux_layer:
                self.hidden_layers.append(
                    nn.Linear(n_aux_feat + qtd_neuron_per_hidden_layer, n_neuron_aux_layer))
            elif i + 1 == aux_layer:
                self.hidden_layers.append(
                    nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer))
            else:
                self.hidden_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer))
            
        # The number of output layer would be total number of layer ("max_num_hidden_layers") - 2 (no output from the aux layer and the first layer)
        for i in range(max_num_hidden_layers - 2):
                self.output_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        self.hidden_layers = nn.ModuleList(self.hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(self.output_layers).to(self.device)
        
        # Alphas is related to the output layers. So, the nummber of alphas would be equal to the number of hidden layers (max_num_hidden_layers) - 2.
        # The alpha value sums to 1 and is equal at the beginning of the training. 
        self.alpha = Parameter(torch.Tensor(self.max_num_hidden_layers - 2).fill_(1 / (self.max_num_hidden_layers - 2)),
                               requires_grad=False).to(self.device)


        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

    # Initialize the graadients of all the parameters with 0.
    def zero_grad(self):
        for i in range(self.max_num_hidden_layers - 2):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    # This function predicts the output from each layer,calculates the loss and do gradient descent and alpha update.
    def update_weights(self, X, aux_feat, aux_mask, Y, show_loss):
        Y = torch.from_numpy(Y).to(self.device)

        # Predictions from each layer using the implemented forward function.
        # Training = True means that apply dropout since it is for training part
        predictions_per_layer = self.forward(X, aux_feat, aux_mask, training=True)

        real_output = torch.sum(torch.mul(
                self.alpha.view(self.max_num_hidden_layers - 2, 1).repeat(1, self.batch_size).view(
                    self.max_num_hidden_layers - 2, self.batch_size, 1), predictions_per_layer), 0)
        self.prediction.append(real_output)
        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(real_output.view(self.batch_size, self.n_classes), Y.view(self.batch_size).long())
        self.loss_array.append(loss.detach().numpy())

        if show_loss:
            if (len(self.loss_array) % 1000) == 0:
                print("WARNING: Set 'show_loss' to 'False' when not debugging. "
                      "It will deteriorate the fitting performance.")
                loss = np.mean(self.loss_array[-1000:])
                print("Alpha:" + str(self.alpha.data.cpu().numpy()))
                print("Training Loss: " + str(loss))
        
        losses_per_layer = []

        for out in predictions_per_layer:
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(out.view(self.batch_size, self.n_classes), Y.view(
                self.batch_size).long())
            losses_per_layer.append(loss)
        
        w = [None] * (len(losses_per_layer) + 2) 
        b = [None] * (len(losses_per_layer) + 2)

        with torch.no_grad():

            for i in range(len(losses_per_layer)):
                losses_per_layer[i].backward(retain_graph = True)
                 
                self.output_layers[i].weight.data -= self.n* \
                                                     self.alpha[i]*self.output_layers[i].weight.grad.data
                self.output_layers[i].bias.data -= self.n * \
                                                   self.alpha[i] * self.output_layers[i].bias.grad.data

                if i < self.aux_layer - 2:
                    for j in range(i+2):
                        if w[j] is None:
                            w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                        else:
                            w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data
                if i > self.aux_layer - 3:
                    for j in range(i+3):
                        if w[j] is None:
                            w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                        else:
                            w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad()

            for i in range(self.max_num_hidden_layers):
              self.hidden_layers[i].weight.data -= self.n * w[i]
              self.hidden_layers[i].bias.data -= self.n * b[i]

            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(
                    self.alpha[i], self.s / (len(losses_per_layer)))
            
        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(
            self.alpha / z_t, requires_grad=False).to(self.device)
        
        # To save the loss
        detached_loss = []
        for i in range(len(losses_per_layer)):
            detached_loss.append(losses_per_layer[i].detach().numpy())
        self.layerwise_loss_array.append(np.asarray(detached_loss))
        self.alpha_array.append(self.alpha.detach().numpy())

    # Forward pass. Get the output from each layer.     
    def forward(self, X, aux_feat, aux_mask, training):
        hidden_connections = []
        linear_x = []
        relu_x = []

        X = torch.from_numpy(X).float().to(self.device)
        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Forward pass of the first hidden layer. Apply the linear transformation and then relu. The output from the relu is the input
        # passed to the next layer.
        linear_x.append(self.hidden_layers[0](X))
        relu_x.append(F.relu(linear_x[0]))
        hidden_connections.append(relu_x[0])


        for i in range(1, self.max_num_hidden_layers):
            # Forward pass to the Aux layer.
            if i==self.aux_layer-1:
                # Input to the aux layer will be the output from its previous layer and the incoming auxiliary inputs.
                linear_x.append(self.hidden_layers[i](torch.cat((aux_feat, hidden_connections[i - 1]), dim=1)))
                relu_x.append(F.relu(linear_x[i]))
                binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
                mask = binomial.sample([1, self.n_neuron_aux_layer])
                hidden_connections.append(relu_x[i]*mask*(1.0/(1-self.p)))
            else:
                linear_x.append(self.hidden_layers[i](hidden_connections[i - 1]))
                relu_x.append(F.relu(linear_x[i]))
                binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
                mask = binomial.sample([1, self.qtd_neuron_per_hidden_layer])
                hidden_connections.append(relu_x[i]*mask*(1.0/(1-self.p)))

        output_class = []

        for i in range(self.max_num_hidden_layers-1):
            if i < self.aux_layer-2:
                output_class.append(
                    F.softmax(self.output_layers[i](hidden_connections[i+1]), dim=1))
            if i > self.aux_layer-2:
                output_class.append(
                    F.softmax(self.output_layers[i-1](hidden_connections[i+1]), dim=1))

        pred_per_layer = torch.stack(output_class)

        return pred_per_layer

    def validate_input_X(self, data):
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions.")
    
    def validate_input_Y(self, data):
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions.")
    
    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, aux_data, aux_mask, Y_data, show_loss)

# ------------------------------------------------------------------------------------------------- #
# Aux-Drop (OGD) code with Random in AuxLayer layers
class AuxDrop_ODL_RandomInAuxLayer(nn.Module):
    def __init__(self, features_size, max_num_hidden_layers, qtd_neuron_per_hidden_layer, n_classes, aux_layer, 
                 n_neuron_aux_layer, batch_size=1, b=0.99, n=0.01, s=0.2, dropout_p=0.5, n_aux_feat = 3, use_cuda=False):
        super(AuxDrop_ODL_RandomInAuxLayer, self).__init__()

        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.batch_size = batch_size

        self.b = Parameter(torch.tensor(
            b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(
            n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(
            s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(
            dropout_p), requires_grad=False).to(self.device)
        self.n_aux_feat = n_aux_feat

        # Stores hidden and output layers
        self.hidden_layers = []
        self.output_layers = []

        # The number of hidden layers would be "max_num_hidden_layers". The input to the first layer is the base features i.e. "features_size".
        self.hidden_layers.append(
            nn.Linear(features_size, qtd_neuron_per_hidden_layer))

        for i in range(max_num_hidden_layers - 1):
            # The input to the aux_layer is the outpout coming from its previous layer, i.e., "qtd_neuron_per_hidden_layer" and the 
            # number of auxiliary features, "n_aux_feat".
            if i+2 == aux_layer:
                self.hidden_layers.append(
                    nn.Linear(n_aux_feat + qtd_neuron_per_hidden_layer, n_neuron_aux_layer))
            elif i + 1 == aux_layer:
                self.hidden_layers.append(
                    nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer))
            else:
                self.hidden_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer))
            
        # The number of output layer would be total number of layer ("max_num_hidden_layers") - 2 (no output from the aux layer and the first layer)
        for i in range(max_num_hidden_layers - 2):
                self.output_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        self.hidden_layers = nn.ModuleList(self.hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(self.output_layers).to(self.device)
        
        # Alphs is related to the output layers. So, the nummber of alphas would be equal to the number of hidden layers (max_num_hidden_layers) - 2.
        # The alpha value sums to 1 and is equal at the beginning of the training. 
        self.alpha = Parameter(torch.Tensor(self.max_num_hidden_layers - 2).fill_(1 / (self.max_num_hidden_layers - 2)),
                               requires_grad=False).to(self.device)


        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

    # Initialize the graadients of all the parameters with 0.
    def zero_grad(self):
        for i in range(self.max_num_hidden_layers - 2):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    # This function predicts the output from each layer,calculates the loss and do gradient descent and alpha update.
    def update_weights(self, X, aux_feat, aux_mask, Y, show_loss):
        Y = torch.from_numpy(Y).to(self.device)

        # Predictions from each layer using the implemented forward function.
        # Training = True means that apply dropout since it is for training part
        predictions_per_layer = self.forward(X, aux_feat, aux_mask, training=True)

        real_output = torch.sum(torch.mul(
                self.alpha.view(self.max_num_hidden_layers - 2, 1).repeat(1, self.batch_size).view(
                    self.max_num_hidden_layers - 2, self.batch_size, 1), predictions_per_layer), 0)
        self.prediction.append(real_output)
        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(real_output.view(self.batch_size, self.n_classes), Y.view(self.batch_size).long())
        self.loss_array.append(loss.detach().numpy())
        if show_loss:
            if (len(self.loss_array) % 1000) == 0:
                print("WARNING: Set 'show_loss' to 'False' when not debugging. "
                      "It will deteriorate the fitting performance.")
                loss = np.mean(self.loss_array[-1000:])
                print("Alpha:" + str(self.alpha.data.cpu().numpy()))
                print("Training Loss: " + str(loss))
                # self.loss_array.clear()
        
        losses_per_layer = []

        for out in predictions_per_layer:
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(out.view(self.batch_size, self.n_classes), Y.view(
                self.batch_size).long())
            losses_per_layer.append(loss)

        w = [None] * (len(losses_per_layer) + 2) 
        b = [None] * (len(losses_per_layer) + 2)

        with torch.no_grad():

            for i in range(len(losses_per_layer)):
                losses_per_layer[i].backward(retain_graph = True)
                
                self.output_layers[i].weight.data -= self.n* \
                                                     self.alpha[i]*self.output_layers[i].weight.grad.data
                self.output_layers[i].bias.data -= self.n * \
                                                   self.alpha[i] * self.output_layers[i].bias.grad.data

                if i < self.aux_layer - 2:
                    for j in range(i+2):
                        if w[j] is None:
                            w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                        else:
                            w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data
                if i > self.aux_layer - 3:
                    for j in range(i+3):
                        if w[j] is None:
                            w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                        else:
                            w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                            b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad()

            for i in range(self.max_num_hidden_layers):
              self.hidden_layers[i].weight.data -= self.n * w[i]
              self.hidden_layers[i].bias.data -= self.n * b[i]

            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(
                    self.alpha[i], self.s / (len(losses_per_layer)))
            
        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(
            self.alpha / z_t, requires_grad=False).to(self.device)
        
        # To save the loss
        detached_loss = []
        for i in range(len(losses_per_layer)):
            detached_loss.append(losses_per_layer[i].detach().numpy())
        self.layerwise_loss_array.append(np.asarray(detached_loss))
        self.alpha_array.append(self.alpha.detach().numpy())

    # Forward pass. Get the output from each layer.     
    def forward(self, X, aux_feat, aux_mask, training):
        hidden_connections = []
        linear_x = []
        relu_x = []

        X = torch.from_numpy(X).float().to(self.device)
        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Forward pass of the first hidden layer. Apply the linear transformation and then relu. The output from the relu is the input
        # passed to the next layer.
        linear_x.append(self.hidden_layers[0](X))
        relu_x.append(F.relu(linear_x[0]))
        hidden_connections.append(relu_x[0])


        for i in range(1, self.max_num_hidden_layers):
            # Forward pass to the Aux layer.
            if i==self.aux_layer-1:
                # Input to the aux layer will be the output from its previous layer and the incoming auxiliary inputs.
                linear_x.append(self.hidden_layers[i](torch.cat((aux_feat, hidden_connections[i - 1]), dim=1)))
                relu_x.append(F.relu(linear_x[i]))
                binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
                mask = binomial.sample([1, self.n_neuron_aux_layer])
                hidden_connections.append(relu_x[i]*mask*(1.0/(1-self.p)))
            else:
                linear_x.append(self.hidden_layers[i](hidden_connections[i - 1]))
                relu_x.append(F.relu(linear_x[i]))
                hidden_connections.append(relu_x[i])
        
        output_class = []

        for i in range(self.max_num_hidden_layers-1):
            if i < self.aux_layer-2:
                output_class.append(
                    F.softmax(self.output_layers[i](hidden_connections[i+1]), dim=1))
            if i > self.aux_layer-2:
                output_class.append(
                    F.softmax(self.output_layers[i-1](hidden_connections[i+1]), dim=1))

        pred_per_layer = torch.stack(output_class)

        return pred_per_layer

    def validate_input_X(self, data):
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions.")
    
    def validate_input_Y(self, data):
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions.")
    
    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, aux_data, aux_mask, Y_data, show_loss)


# ------------------------------------------------------------------------------------------------- #
# Aux-Drop (OGD) code with Random in First layers and all features passed directly to it
class AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst(nn.Module):
    def __init__(self, features_size, max_num_hidden_layers, qtd_neuron_per_hidden_layer, n_classes, aux_layer, 
                 n_neuron_aux_layer, batch_size=1, b=0.99, n=0.01, s=0.2, dropout_p=0.5, n_aux_feat = 3, use_cuda=False):
        super(AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst, self).__init__()

        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.batch_size = batch_size

        self.b = Parameter(torch.tensor(
            b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(
            n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(
            s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(
            dropout_p), requires_grad=False).to(self.device)
        self.n_aux_feat = n_aux_feat

        # Stores hidden and output layers
        self.hidden_layers = []
        self.output_layers = []

        # The number of hidden layers would be "max_num_hidden_layers". The input to the first layer is the base features i.e. "features_size" and the auxiliary features.
        self.hidden_layers.append(
            nn.Linear(features_size + n_aux_feat, n_neuron_aux_layer))
        self.hidden_layers.append(
                    nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer))
        for i in range(max_num_hidden_layers - 2):
            self.hidden_layers.append(
                nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer))
            
        # The number of output layer would be total number of layer ("max_num_hidden_layers") - 1 (no output from the aux layer/first layer)
        for i in range(max_num_hidden_layers - 1):
                self.output_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        self.hidden_layers = nn.ModuleList(self.hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(self.output_layers).to(self.device)
        
        # Alphs is related to the output layers. So, the nummber of alphas would be equal to the number of hidden layers (max_num_hidden_layers) - 1.
        # The alpha value sums to 1 and is equal at the beginning of the training. 
        self.alpha = Parameter(torch.Tensor(self.max_num_hidden_layers - 1).fill_(1 / (self.max_num_hidden_layers - 1)),
                               requires_grad=False).to(self.device)

        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

    # Initialize the gradients of all the parameters with 0.
    def zero_grad(self):
        for i in range(self.max_num_hidden_layers - 1):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    # This function predicts the output from each layer,calculates the loss and do gradient descent and alpha update.
    def update_weights(self, X, aux_feat, aux_mask, Y, show_loss):
        Y = torch.from_numpy(Y).to(self.device)

        # Predictions from each layer using the implemented forward function.
        predictions_per_layer = self.forward(X, aux_feat, aux_mask)

        real_output = torch.sum(torch.mul(
                self.alpha.view(self.max_num_hidden_layers - 1, 1).repeat(1, self.batch_size).view(
                    self.max_num_hidden_layers - 1, self.batch_size, 1), predictions_per_layer), 0)
        self.prediction.append(real_output)

        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(real_output.view(self.batch_size, self.n_classes), Y.view(self.batch_size).long())
        self.loss_array.append(loss.detach().numpy())
        if show_loss:
            if (len(self.loss_array) % 1000) == 0:
                print("WARNING: Set 'show_loss' to 'False' when not debugging. "
                      "It will deteriorate the fitting performance.")
                loss = np.mean(self.loss_array[-1000:])
                print("Alpha:" + str(self.alpha.data.cpu().numpy()))
                print("Training Loss: " + str(loss))
        
        losses_per_layer = []

        for out in predictions_per_layer:
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(out.view(self.batch_size, self.n_classes), Y.view(
                self.batch_size).long())
            losses_per_layer.append(loss)
        
        w = [None] * (len(losses_per_layer) + 1) 
        b = [None] * (len(losses_per_layer) + 1)

        with torch.no_grad():

            for i in range(len(losses_per_layer)):
                losses_per_layer[i].backward(retain_graph = True)                
                
                self.output_layers[i].weight.data -= self.n* \
                                                     self.alpha[i]*self.output_layers[i].weight.grad.data
                self.output_layers[i].bias.data -= self.n * \
                                                   self.alpha[i] * self.output_layers[i].bias.grad.data

                for j in range(i+2):
                    if w[j] is None:
                        w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                    else:
                        w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad()

            for i in range(self.max_num_hidden_layers):
              self.hidden_layers[i].weight.data -= self.n * w[i]
              self.hidden_layers[i].bias.data -= self.n * b[i]

            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(
                    self.alpha[i], self.s / (len(losses_per_layer)))
            
        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(
            self.alpha / z_t, requires_grad=False).to(self.device)
        
        # To save the loss
        detached_loss = []
        for i in range(len(losses_per_layer)):
            detached_loss.append(losses_per_layer[i].detach().numpy())
        self.layerwise_loss_array.append(np.asarray(detached_loss))
        self.alpha_array.append(self.alpha.detach().numpy())

    # Forward pass. Get the output from each layer.     
    def forward(self, X, aux_feat, aux_mask):
        hidden_connections = []
        linear_x = []
        relu_x = []

        X = torch.from_numpy(X).float().to(self.device)
        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Forward pass to the Aux layer.
        # Input to the aux layer will be the output from its previous layer and the incoming auxiliary inputs.
        linear_x.append(self.hidden_layers[0](torch.cat((aux_feat, X), dim=1)))
        relu_x.append(F.relu(linear_x[0]))
        # Based on the incoming aux data, the aux inputs which do not come gets included in the dropout probability.
        # Based on that we calculate the probability to drop the left over neurons in auxiliary layer.
        binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
        mask = binomial.sample([1, self.n_neuron_aux_layer])
        hidden_connections.append(relu_x[0]*mask*(1.0/(1-self.p)))

        for i in range(1, self.max_num_hidden_layers):
            linear_x.append(self.hidden_layers[i](hidden_connections[i - 1]))
            relu_x.append(F.relu(linear_x[i]))
            hidden_connections.append(relu_x[i])
        
        output_class = []

        for i in range(self.max_num_hidden_layers-1):
            output_class.append(
                F.softmax(self.output_layers[i](hidden_connections[i+1]), dim=1))

        pred_per_layer = torch.stack(output_class)

        return pred_per_layer

    def validate_input_X(self, data):
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions.")
    
    def validate_input_Y(self, data):
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions.")
    
    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, aux_data, aux_mask, Y_data, show_loss)

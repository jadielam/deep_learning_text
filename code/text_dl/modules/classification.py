from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from text_dl.modules import Module

class Classifier(Module):
    '''
    Standard multilayer perceptron classifier.
    '''
    def __init__(self, nb_classes, input_size, nb_layers = 3,
                classifier_function = partial(F.softmax, 1),
                activation_function = F.relu, 
                hidden_dimension = 1024,
                dropout = 0.2):
        '''
        Arguments:
        - nb_classes (int): number of classes for classification
        - input_size (int): length of the input vector
        - nb_layers (int): number of layers for the classifier
        - classifier_function: (:obj:`torch.nn.functional`) A function to apply to the last layer.
                             Usually will be softmax or sigmoid
        '''
        super(Classifier, self).__init__()
        self.classifier_function = classifier_function
        self.activation_function = activation_function

        dims = []
        for i in range(max(nb_layers, 1)):
            dim_entry = [hidden_dimension, hidden_dimension]
            if i == 0: 
                dim_entry[0] = input_size
            if i == max(nb_layers, 1) - 1:
                dim_entry[1] = nb_classes
            dims.append(dim_entry)

        self.dropout_layer = nn.Dropout(p = dropout)
        self.layers = []
        for i in range(len(dims)):
            linear_layer = nn.Linear(dims[i][0], dims[i][1])
            self.layers.append(linear_layer)
            super(Classifier, self).add_module("linear_{}".format(i), linear_layer)

    def forward(self, input_t):
        '''
        Arguments:
        - input_t_l (:obj:`torch.Tensor`): input tensor to use for classification of size (batch, input_size)

        Returns:
        - output (:obj:`torch.Tensor`) of size (batch, nb_classes)
        '''
        next_t = input_t
        next_t = self.dropout_layer(next_t)
        for i in range(len(self.layers) - 1):
            next_t = self.activation_function(self.layers[i](next_t))
        next_t = self.layers[len(self.layers) - 1](next_t)
        output = self.classifier_function(next_t)
        return output

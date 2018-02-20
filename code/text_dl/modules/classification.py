import torch
import torch.nn as nn
import torcn.nn.functional as F

class Classifier(nn.Module):
    '''
    Standard multilayer perceptron classifier.
    '''
    def __init__(self, nb_classes, input_size, nb_layers = 2,
                classifier_function = F.softmax,
                activation_function = F.relu):
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
            dim_entry = [1024, 1024]
            if i == 0: 
                dim_entry[0] = input_size
            if i == max(nb_layers, 1) - 1:
                dim_entry[1] = nb_classes
            dims.append(dim_entry)

        self.layers = [nn.Linear(dims[i][0], dims[i][1]) for i in range(len(dims))]

    def forward(self, input_t):
        '''
        Arguments:
        - input_t (:obj:`torch.Tensor`): input tensor to use for classification

        Returns:
        - output (:obj:`torch.Tensor`)
        '''
        next_t = input_t
        for i in range(len(self.layers)):
            next_t = self.activation_function(self.layers[i](next_t))
        output = self.classifier_function(next_t)
        return output

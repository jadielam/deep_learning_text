'''
Contains model definitions for the different kinds of problems to solve.
'''

import itertools.ifilter as ifilter
import torch.nn as nn
from text_dl.common.devices import use_cuda

class Model(nn.Module):
    '''
    The model class is the parent class for all the 
    model classes.
    '''
    def __init__(self):
        pass
        
    def loss(self):
        '''
        Forward pass of the model including loss
        function
        '''
        raise NotImplementedError("Class %s doesn't implement loss()" % (self.__class__.__name__))
    
    def trainable_parameters(self):
        '''
        Returns the parameters that are optimizable.
        In this way you can instantiate an optimizer to use them
        '''
        return ifilter(lambda p: p.requires_grad, 
                super(Model, self).parameters())
    
    def use_cuda(self):
        '''
        Returns true if GPU is model is deployed in gpu, otherwise
        returns false

        #TODO: I have to change this to something more seamless,
        #so that I don't have to start constantly calling this function
        '''
        return use_cuda

def model_factory(conf, vocab):
    '''
    Returns a model built according to the conf parameters.
    
    Arguments:
    - conf (dict): model configuration
    - vocab (:obj:`torchtext.vocab.Vocab`): Vocabulary to be used for embeddings.
    '''
    pass
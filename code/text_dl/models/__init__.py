'''
Contains model definitions for the different kinds of problems to solve.
'''

import itertools.ifilter as ifilter
import torch.nn as nn
from text_dl.common.devices import use_cuda
from text_dl.modules.embeddings import embedding_factory
from text_dl.models.multiclassification.simple_multiclassification import SimpleMulticlassificationModel

class Model(nn.Module):
    '''
    The model class is the parent class for all the 
    model classes.
    '''
    def __init__(self):
        pass
        
    def loss(self, input_t, ground_truth):
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
    if conf['type'] == 'multiclassification':
        nb_classes = conf['params']['nb_classes']
        train_embedding = conf['params']['train_embedding']
        batch_size = conf['batch_size']
        max_sequence_length = conf['max_sequence_length']
        embeddings = embedding_factory(vocab, train_embedding)
        model = SimpleMulticlassificationModel(embeddings, batch_size, 
                                            nb_classes, max_sequence_length)
        return model
    else:
        raise ValueError("Incorrect model type: {}".format(conf['type']))
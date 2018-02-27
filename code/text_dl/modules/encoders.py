import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from text_dl.modules import Module

class EncoderRNN(Module):
    '''
    Encodes input using an RNN
    '''
    def __init__(self, embedding,
                    batch_size,
                    bidirectional = False):
        super(EncoderRNN, self).__init__()
        self.embedding = embedding
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.hidden_size = self.embedding.embedding_dim
        self.gru = nn.GRU(self.hidden_size, self.hidden_size,
                         bidirectional = bidirectional, 
                         batch_first = False)
        
    def forward(self, input_t, hidden):
        '''
        Arguments:

        - input_t (:obj:`torch.Tensor`): Tensor of shape (N, W) N is mini-batch size, W is number of indices to extract per minibatch.
        - hidden (:obj:`torch.Tensor`): the hidden state of the previous call to forward. The initial hidden state is given by self.init_hidden()
                                        The hidden dimension is (num_layers * num_directions, batch_size, hidden_size)
        '''
        embedded = self.embedding(input_t)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
        
    def init_hidden(self, use_cuda):
        '''
        Returns the initialized first hidden state.
        
        Arguments:
        - batch_size (int): The size of the batch
        '''
        num_directions = 2 if self.bidirectional else 1
        num_layers = self.gru.num_layers
        
        result = Variable(torch.zeros(num_layers * num_directions, self.batch_size, self.hidden_size))
        
        if use_cuda:
            return result.cuda()
        else:
            return result
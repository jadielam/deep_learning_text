import itertools.ifilter as ifilter
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
    def __init__(self, vocab, train_embedding = False, 
                    bidirectional = False):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab.vectors.shape[0]
        self.hidden_size = vocab.vectors.shape[1]
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embedding.weight = nn.Parameter(vocab.vectors)
        self.embedding.weight.requires_grad = train_embedding
        self.gru = nn.GRU(self.hidden_size, self.hidden_size,
                         bidirectional = bidirectional)
        
    def forward(self, vocab_index, hidden):
        '''
        vocab_index: (int) the vocabulary index of the original token in the vocabulary. The initial input token is the start-of-string <SOS> token
        hidden: the hidden state of the previous call to forward. The initial hidden state is given by self.init_hidden()
        '''
        embedded = self.embedding(vocab_index).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
        
    def init_hidden(self, use_cuda):
        '''
        Returns the initialized first hidden state.
        '''
        num_directions = 2 if self.bidirectional else 1
        num_layers = self.gru.num_layers

        result = Variable(torch.zeros(num_layers * num_directions, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
import itertools.ifilter as ifilter
import torch
import torch.nn as nn
import torch.nn.functional as F

from text_dl.modules import Module

class AttentionRNNDecoder(Module):
    def __init__(self, vocab, max_length, 
                dropout_p = 0.1, train_embedding = False,
                bidirectional = False):
        '''
        hidden_size: (int) the size of the word vectors
        vocab_size: (int) the number of words in the vocabulary
        max_length: (int) the maximum possible length of a text sequence.
        dropout_p: (float) dropout rate
        '''
        super(AttentionRNNDecoder, self).__init__()
        self.hidden_size = vocab.vectors.shape[1]
        self.vocab_size = vocab.vectors.shape[0]
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embedding.weight = nn.Parameters(vocab.vectors)
        self.embedding.weight.requires_grad = train_embedding
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_state * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, bidirectional = bidirectional)
        
    def forward(self, vocab_index, hidden, encoder_outputs):
        '''
        vocab_index: (int) the vocabulary index of the last output of the decoder. The initial input token is the start-of-string <SOS> token
        hidden: (torch.Tensor) the first hidden state is the context vector (the encoder's last output)
        encoder_outputs: (torch.Tensor) they are the sequence of encoder outputs.
        '''
        embedded = self.embedding(vocab_index).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1), 1), dim = 1
        )
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        
        return output, hidden, attn_weights

class AttentionClassificationDecoder(Module):
    def __init__(self, max_length, hidden_size):
        '''
        max_length: (int) the maximum possible length of a text sequence.
        dropout_p: (float) dropout rate
        '''
        super(AttentionRNNDecoder, self).__init__()
        self.max_length = max_length
        self.hidden_size = hidden_size
        
        self.attn = nn.Linear(self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_state, self.hidden_size)
        
    def forward(self, hidden, encoder_outputs):
        '''
        hidden: (torch.Tensor) the first hidden state is the context vector (the encoder's last output)
        encoder_outputs: (torch.Tensor) they are the sequence of encoder outputs.
        '''
        
        attn_weights = F.softmax(
            self.attn(hidden[0], 1), dim = 1
        )
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = self.attn_combine(attn_applied[0]).unsqueeze(0)
        output = F.relu(output)

        return output


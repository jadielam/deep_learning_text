import torch
import torch.nn as nn
import torch.nn.functional as F

from text_dl.modules import Module

class AttentionDecoder(Module):
    def __init__(self, max_length, hidden_size, nb_outputs = 1):
        '''
        max_length: (int) the maximum possible length of a text sequence.
        dropout_p: (float) dropout rate
        '''
        super(AttentionDecoder, self).__init__()
        self.max_length = max_length
        self.nb_outputs = nb_outputs
        self.hidden_size = hidden_size
        
        self.attn_layers = []
        for i in range(self.nb_outputs):
            attn1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
            attn2 = nn.Linear(self.hidden_size, 1)
            self.attn_layers.append((attn1, attn2))
            super(AttentionDecoder, self).add_module("attn1_{}".format(i), attn1)
            super(AttentionDecoder, self).add_module("attn2_{}".format(i), attn2)

    def forward(self, hidden, encoder_outputs):
        '''
        Combines the sequence of encoder outputs using an attention mechanism.

        Arguments:

        - hidden (:obj:`torch.Tensor`): of size (num_layers * num_directions, batch_size, hidden_size) (the first hidden state is the context vector (the encoder's last output)
        - encoder_outputs (:obj:`torch.Tensor`): of size (seq_len, batch, hidden_size) othey are the sequence of encoder outputs

        Returns:
        - output (:obj:`torch.Tensor`): of size (batch, hidden_size)
        '''
        batch_size = encoder_outputs.size()[1]
        seq_len = encoder_outputs.size()[0]
        hid_last_dim = hidden.size()[2]
        new_hidden = hidden.transpose(0, 1).contiguous().view(1, batch_size, -1).expand(seq_len, batch_size, 2 * hid_last_dim)
        final_input = torch.cat([new_hidden, encoder_outputs], dim = 2)

        attn_applied_outputs = []
        for i in range(len(self.attn_layers)):
            attn1 = self.attn_layers[i][0]
            attn2 = self.attn_layers[i][1]

            attn_weights = F.relu(attn1(final_input))
            attn_weights = F.softmax(attn2(attn_weights), 1)
            # attn_weights has shape (seq_len, batch, 1)
        
            attn_weights = attn_weights.transpose(0, 1).transpose(1, 2)
            attn_applied = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1)).squeeze()
            #attn_applied has shape (batch, hidden_size)
            attn_applied_outputs.append(attn_applied)

        return attn_applied_outputs


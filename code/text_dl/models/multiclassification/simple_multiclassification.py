import torch
import torch.nn as nn
import torch.nn.functional as F
from text_dl.models.model import Model
from text_dl.common.devices import use_cuda
from text_dl.modules.encoders import EncoderRNN
from text_dl.modules.decoders import AttentionDecoder
from text_dl.modules.classification import Classifier

class SimpleMulticlassificationModel(Model):
    def __init__(self, embedding, batch_size, nb_classes, max_sequence_length):
        super(SimpleMulticlassificationModel, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.hidden_size = embedding.embedding_dim * 2 # Since we are doing bidirectional
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        
        # Modules
        self.encoder = EncoderRNN(embedding, batch_size, bidirectional = True)
        self.decoder = AttentionDecoder(max_sequence_length, self.hidden_size, 1)
        self.classifier = Classifier(nb_classes, self.hidden_size * 1, classifier_function = F.sigmoid)

        # Loss
        self.criterion = nn.BCELoss()
        
    def forward(self, input_t):
        '''
        Arguments:

        - input_t (:obj:`torch.Tensor`) of size (seq_len, batch)
        '''
        initial_hidden = self.encoder.init_hidden(use_cuda)
        encoder_output, hidden = self.encoder(input_t, initial_hidden)
        attn_applied = self.decoder(hidden, encoder_output)
        attn_applied = torch.cat(attn_applied, 1)
        classification = self.classifier(attn_applied)
        return classification

    def loss(self, input_t, ground_truth):
        '''
        Arguments:

        - input_t (:obj:`torch.Tensor`) of size (seq_len, batch)
        '''
        classification = self.forward(input_t)
        loss = self.criterion(classification, ground_truth)
        return loss

        
    
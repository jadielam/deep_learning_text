import torch
import torch.nn as nn
import torch.nn.functional as F
from text_dl.models.model import Model
from text_dl.common.devices import use_cuda
from text_dl.modules.encoders import EncoderRNN
from text_dl.modules.decoders import AttentionDecoder
from text_dl.modules.classification import Classifier

class SimpleMulticlassificationModel(Model):
    def __init__(self, embedding, nb_classes, max_sequence_length = 300,
                hidden_size = None, classifier_layers = 3, classifier_hidden = 1024,
                gru_dropout = 0, classification_dropout = 0.2):
        
        super(SimpleMulticlassificationModel, self).__init__(**kwargs)
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        if self.hidden_size is None:
            self.hidden_size = embedding.embedding_dim
        self.nb_classes = nb_classes
        
        # Modules
        self.encoder = EncoderRNN(embedding, bidirectional = True, hidden_size = hidden_size, gru_dropout = gru_dropout)
        self.decoder = AttentionDecoder(max_sequence_length, self.hidden_size * 2, 1)
        self.classifier = Classifier(nb_classes, self.hidden_size * 2, classifier_function = F.sigmoid,
                                    nb_layers = classifier_layers, hidden_dimension = classifier_hidden,
                                    dropout = classification_dropout, **kwargs)

        # Loss
        self.criterion = nn.BCELoss()
        
    def forward(self, input_t):
        '''
        Arguments:

        - input_t (:obj:`torch.Tensor`) of size (seq_len, batch)
        '''
        batch_size = input_t.size()[1]
        initial_hidden = self.encoder.init_hidden(batch_size, use_cuda)
        encoder_output, hidden = self.encoder(input_t, initial_hidden)
        attn_applied = self.decoder(hidden, encoder_output)
        attn_applied = torch.cat(attn_applied, 1)
        classification = self.classifier(attn_applied)
        return classification

    def loss(self, input_t, ground_truth):
        '''
        Arguments:

        - input_t (:obj:`torch.Tensor`) of size (seq_len, batch)
        - ground_truth (:obj:`torch.Tensor`) of size (batch, nb_classes)

        Returns:
        - loss (:obj:`torch.Tensor`) of size (1)
        '''
        classification = self.forward(input_t)
        loss = self.criterion(classification, ground_truth)
        return loss

        
    
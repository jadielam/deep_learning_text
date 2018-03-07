import torch
import torch.nn as nn
import torch.nn.functional as F
from text_dl.models.model import Model
from text_dl.common.devices import use_cuda
from text_dl.modules.encoders import EncoderRNN
from text_dl.modules.decoders import AttentionDecoder
from text_dl.modules.classification import Classifier

class HierarchicalMulticlassificationModel(Model):
    def __init__(self, embedding, nb_classes, max_sentence_length = 300, max_doc_length = 20,
                gru_dropout = 0):
        super(HierarchicalMulticlassificationModel, self).__init__()
        
        #1. Initiliazing modules
        self.hidden_size = embedding.embedding_dim
        self.sentence_encoder = EncoderRNN(embedding, bidirectional = True, 
                                            hidden_size = self.hidden_size, 
                                            gru_dropout = gru_dropout)
        self.sentence_attn = AttentionDecoder(max_sentence_length, self.hidden_size * 2, 1)
        self.doc_encoder = nn.GRU(self.hidden_size * 2, self.hidden_size * 2, 
                                    bidirectional = True, 
                                    dropout = gru_dropout)
        self.doc_attn = AttentionDecoder(max_doc_length, self.hidden_size * 4, 1)
        self.classifier = Classifier(nb_classes, self.hidden_size * 4, 
                            classifier_function = F.sigmoid,
                            dropout = 0.3)
        
        #2. Initializing loss
        self.criterion = nn.BCELoss()

        #3. Initializing other variables

    
    def forward(self, input_t):
        '''
        Arguments:
            - input_t (:obj:`torch.Tensor`) of size (docs, sen, word, dim)
        
        Returns:
            - classification (:obj:`torch.Tensor`) of size (docs, nb_classes)
        '''
        
        #1. Get tensors working
        
    
    def loss(self, input_t, ground_truth):
        '''
        Arguments:
            - input (:obj:`torch.Tensor`) of size (docs, sen, word, dim)
            - ground_truth (:obj:`torch.Tensor) of size (docs, nb_classes)

        Returns:
            - loss (:obj:`torch.Tensor`) of size (1)
        '''
        classification = self.forward(input_t)
        loss = self.criterion(classification, ground_truth)
        return loss
        
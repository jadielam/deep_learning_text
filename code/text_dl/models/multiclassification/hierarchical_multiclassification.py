import torch
import torch.nn as nn
import torch.nn.functional as F
from text_dl.models.model import Model
from text_dl.common.devices import use_cuda
from text_dl.modules.encoders import EncoderRNN

class HierarchicalMulticlassificationModel(Model):
    def __init__(self, embedding, batch_size, nb_classes, max_sequence_length):
        super(HierarchicalMulticlassificationModel, self).__init__()
    
    def forward(self, input_t):
        pass
    
    def loss(self, input_t, ground_truth):
        pass
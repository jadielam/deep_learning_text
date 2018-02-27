import torch.nn as nn

class Module(nn.Module):
    def __init__(self):
        pass
        
    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad, super(Module, self).parameters())


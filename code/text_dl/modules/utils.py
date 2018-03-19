import torch
import torch.nn as nn

class Elementwise(nn.ModuleList):
    '''
    A simple network container.
    Parameters are a list of modules.
    Inputs are a N-D variable whose last dimension is
    the same length as the list.

    Outputs are the results of applying modules to inputs
    elementwise.

    An optional merge parameter allows the outputs to be
    reduced to a single variable
    '''
    def __init__(self, merge = None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)
    
    def forward(self, input_t):
        '''
        Arguments:
            - input (::obj::`torch.Tensor`): of dim > 1. 
              input.size()[input.dim() - 1] must be equal to
              the number of modules in the list of modules
        '''
        dim = input_t.dim()
        inputs = [feat.squeeze(dim) for feat in input_t.split(1, dim)]
        assert len(self) == len(inputs)
        outputs = [f(x) for f, x in zip(self, inputs)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, dim)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs

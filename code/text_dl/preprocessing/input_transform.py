import torch
from torch.autograd import Variable

def text_field_t(batch):
    return batch.text

def pad_batches(batches):

    # assuming trailing dimensions and type of all the Variables
    # in sequences are same and fetching those from sequences[0]
    max_seq_len = 0
    for _, batch in enumerate(batches):
        if max_seq_len < batch.size()[0]:
            max_seq_len = batch.size()[0]
    
    out_batches = []
    
    for _, batch in enumerate(batches):
        out_dims = list(batch.size())
        length = out_dims[0]
        out_dims[0] = max_seq_len
        out_batch = Variable(batch.data.new(*out_dims).fill_(1))
        out_batch[:length, :] = batch
        out_batches.append(out_batch)

    return out_batches

def text_ner_field_t(batch):
    [batch.text, batch.ner] = pad_batches([batch.text, batch.ner])
    stacked = torch.stack([batch.text, batch.ner], 2)
    return stacked

def input_transform_factory(transform_type):
    if transform_type == 'text_field':
        return text_field_t
    elif transform_type == 'text_ner_field':
        return text_ner_field_t
import torch
import torchtext.data as data
from text_dl.common.devices import use_cuda
from text_dl.preprocessing.vectors import OneHotEncoderVectors

def csv_generator(training_path, batch_size, validation_path = None, vocab_type = "glove.twitter.27B.200d"):
    '''
    Arguments:

    - training_path (str): path to the csv training file
    - batch_size (int): size of the batch
    - validation_path (str): path to the csv of the validation file
    - vocab_type (str): Indicates the pretrained word vectors type that will be used.
    '''
    text_field = data.Field(sequential = True, eos_token = '<eos>',
                            init_token = '<sos>', pad_token = '<pad>',
                            unk_token = '<unk>')
    target_field = data.Field(sequential = False, use_vocab = False, 
                                tensor_type = torch.LongTensor)
    fields = [("text", text_field), ("target", target_field)]
    
    datasets = [data.TabularDataset(path = a, format = "csv", 
                                    fields = fields, skip_header = True) if a else None for a in 
                                    [training_path, validation_path]]
    text_field.build_vocab(*[a for a in datasets if a], vectors = vocab_type)
    vocabulary = text_field.vocab

    device_type = None if use_cuda else -1
    [training_itr, validation_itr] = [data.Iterator(dts, batch_size = batch_size, train = train,
                                        device = device_type, sort = False) if dts else None for (dts, train) in zip(datasets, [True, False])]
    return [vocabulary], training_itr, validation_itr

def csv_extra_vocabs_generator(training_path, batch_size, validation_path = None, 
                                vocab_type = "glove.twitter.27B.200d", 
                                entities_annotations = ['B-PARK', 'I-PARK', 'B-ATTR', 'I-ATTR', 'B-REST', 'I-REST', 'B-RESO', 'I-RESO', 'B-PLAC', 'I-PLAC', 'B-CHAR', 'I-CHAR', 'B-ENTE', 'I-ENTE']):
    text_field = data.Field(sequential = True, eos_token = '<eos>',
                            init_token = '<sos>', pad_token = '<pad>',
                            unk_token = '<unk>')
    
    annotations_field = data.Field(sequential = True, eos_token = '<eos>',
                                    init_token = '<sos>', pad_token = '<pad>',
                                    unk_token = '<unk>')
    
    target_field = data.Field(sequential = False, use_vocab = False, 
                                tensor_type = torch.LongTensor)
    
    fields = [("text", text_field), ("ner", annotations_field), ("target", target_field)]
    
    datasets = [data.TabularDataset(path = a, format = "csv", 
                                    fields = fields, skip_header = True) if a else None for a in 
                                    [training_path, validation_path]]
    
    text_field.build_vocab(*[a for a in datasets if a], vectors = vocab_type)
    annotations_field.build_vocab(*[a for a in datasets if a], 
                                    vectors = OneHotEncoderVectors(entities_annotations))
    text_vocabulary = text_field.vocab
    annotations_vocabulary = annotations_field.vocab

    device_type = None if use_cuda else -1
    [training_itr, validation_itr] = [data.Iterator(dts, batch_size = batch_size, train = train,
                                        device = device_type, sort = False) if dts else None for (dts, train) in zip(datasets, [True, False])]
    return [text_vocabulary, annotations_vocabulary], training_itr, validation_itr
